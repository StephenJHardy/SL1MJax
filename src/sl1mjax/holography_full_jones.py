"""Recover full-Jones holography samples from moving--reference visibilities.

Off-diagonal terms are estimated only after the compact-calibrator RIME is
inverted with the supplied source coherency. The artifact is never frozen.
Do not infer leakage outside the sampled raster.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sl1mjax.beam_conventions import BeamCalibrationState, require_beam_calibration_state
from sl1mjax.beam_operator import unique_visibility_times
from sl1mjax.holography import (
    HolographyObservation,
    Memo195LowerCRaster,
    moving_reference_row_mask,
)
from sl1mjax.holography_diagonal import (
    HolographyDiagonalArtifact,
    HolographyDiagonalSample,
    _interp_complex_delay,
    _local_neighbors,
    _median_spacing,
    _moving_and_reference,
    _raster_label,
    _source_coherency,
    _unique_offsets,
    four_hand_active_rows,
)
from sl1mjax.holography_holoraster_coordinates import (
    source_lm_feed_from_commanded_azelgeo,
)
from sl1mjax.holography_pointing_maps import visit_id_per_time
from sl1mjax.holography_reference_jones import (
    CALIBRATION_ADEQUACY_NOT_OPERATOR_NOTE,
    ON_AXIS_BEAM_IDENTITY_NOTE,
    PREDICTION_EQUIVALENT_BEAM_NOTE,
    ZERO_QU_IS_ABLATION_NOTE,
    coherent_residual_report,
)
from sl1mjax.polarization import Correlation, Receptor, invert_jones, pack_coherency
from sl1mjax.voltage_beam import BeamCoordinates, BeamEvaluation, _pointing_relative_lm

ON_AXIS_REFACTOR_NOTE = (
    "The on-axis gauge is a complete refactorization "
    "A_m=E_m(0), R'_m=R_m A_m, E'_m=A_m^{-1} E_m(s). "
    "Every predicted visibility is preserved and E'_m(0)=I. "
    "Estimate A_m from training-only origin samples and propagate "
    "its uncertainty. Unpinned maps are a diagnostic only."
)
UNPINNED_MAPS_ARE_DIAGNOSTIC_NOTE = (
    "Unpinned RL/LR maps diagnose Jones-factor freedom. They are not scientific competitors."
)
HOLOGRAPHY_HOLDOUT_BEFORE_MAPS_NOTE = (
    "Split holography samples into recovery and untouched visibility "
    "holdouts before estimating normalization, support, or maps."
)
ONE_AXIS_HOLDOUT_NOTE = (
    "Score each visibility holdout axis separately. Their training sets "
    "and scientific meanings differ. Do not combine them into one number."
)
LORO_HOLDOUT_NOTE = (
    "Leave-one-reference-out recovers each mover’s map from the other "
    "references and predicts the held-out reference. Spatial and antenna "
    "support stay intact. This is the strongest immediate full-Jones test."
)
LOVO_HOLDOUT_NOTE = (
    "Leave-one-visit-out holds out one physical raster pass, including "
    "that pass's origin. Visit labels come from HOLORASTER scan groups, "
    "not MAIN row order. Earlier first-encountered-row 'later visit' "
    "scores are not scientific visit-transfer evidence. Score exact "
    "repeated cells separately from spatially interpolated cells."
)
SPATIAL_CHECKERBOARD_NOTE = (
    "Spatial checkerboard tests interpolation within a visit. Every test "
    "point must lie inside declared training support. Report distance to "
    "nearby training cells. Boundary extrapolation is a separate category."
)
LOMO_HOLDOUT_NOTE = (
    "Leave-one-mover-out cannot test a per-antenna empirical map. It "
    "tests an array-average, CASSBEAM, or hierarchical antenna-population "
    "model. Zero predictions for a held-out mover are expected under a "
    "per-antenna representation and do not count against it."
)
DEVELOPMENT_SET_NOTE = (
    "If one-axis results are used to change interpolation settings, they "
    "become development sets. Reserve another visit, spatial partition, "
    "or deterministic outer fold for the eventual final claim."
)
INTERPOLATOR_FROZEN_ON_TRAIN_NOTE = (
    "Freeze the interpolator and support rule using training data only. "
    "Report the fraction of holdout rows with genuine interpolation "
    "support. Score supported rows and the complete holdout separately."
)
ONE_AXIS_HOLDOUT_AXES = (
    "leave_one_reference_out",
    "leave_one_visit_out",
    "spatial_checkerboard",
    "leave_one_mover_out",
)
INTERPOLATOR_NEIGHBOR_K = 4
INTERPOLATOR_RADIUS_SCALE = 1.5

HOLOGRAPHY_FULL_JONES_SCHEMA_VERSION = 1
ARRAY_AVERAGE_ANTENNA_ID = -1
_RECEPTORS = (Receptor.R, Receptor.L)


@dataclass(frozen=True)
class HolographyFullJonesSample:
    """One recovered native Jones sample at a raster cell."""

    moving_antenna_id: int
    unique_time_s: float
    offset_lm_rad: NDArray[np.float64]
    frequency_hz: float
    jones: NDArray[np.complex128]
    sigma: NDArray[np.float64]
    n_reference: int
    weight: float
    copolar_valid: bool
    off_diagonal_valid: bool
    raster: str


@dataclass(frozen=True)
class HolographyFullJonesArtifact:
    """Unfrozen empirical full-Jones holography product."""

    samples: tuple[HolographyFullJonesSample, ...]
    jones: NDArray[np.complex128]
    valid: NDArray[np.bool_]
    off_diagonal_valid: NDArray[np.bool_]
    calibration_state: str
    source_name: str
    source_model_version: str
    reference_combination: str
    receptor_convention: str
    offset_sign: str
    schema_version: int = HOLOGRAPHY_FULL_JONES_SCHEMA_VERSION
    frozen: bool = False
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if int(self.schema_version) != HOLOGRAPHY_FULL_JONES_SCHEMA_VERSION:
            raise ValueError("unsupported holography full-Jones schema")
        if self.frozen:
            raise ValueError("empirical holography full Jones is not frozen")
        jones = np.asarray(self.jones, dtype=np.complex128)
        valid = np.asarray(self.valid, dtype=bool)
        leakage = np.asarray(self.off_diagonal_valid, dtype=bool)
        n = len(self.samples)
        if jones.shape != (n, 2, 2):
            raise ValueError("full Jones must have shape (sample, 2, 2)")
        if valid.shape != (n,) or leakage.shape != (n,):
            raise ValueError("validity masks must have one value per sample")
        object.__setattr__(self, "jones", jones)
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "off_diagonal_valid", leakage)
        object.__setattr__(self, "samples", tuple(self.samples))
        object.__setattr__(self, "notes", tuple(self.notes))

    def to_diagonal(self) -> HolographyDiagonalArtifact:
        """Clear off-diagonals. The diagonal voltages are unchanged."""

        samples = []
        planes = []
        for sample in self.samples:
            samples.append(
                HolographyDiagonalSample(
                    moving_antenna_id=sample.moving_antenna_id,
                    unique_time_s=sample.unique_time_s,
                    offset_lm_rad=sample.offset_lm_rad,
                    frequency_hz=sample.frequency_hz,
                    e_r=complex(sample.jones[0, 0]),
                    e_l=complex(sample.jones[1, 1]),
                    sigma_r=float(sample.sigma[0, 0]),
                    sigma_l=float(sample.sigma[1, 1]),
                    n_reference=sample.n_reference,
                    weight=sample.weight,
                    valid=sample.copolar_valid,
                    raster=sample.raster,
                )
            )
            plane = np.zeros((2, 2), dtype=np.complex128)
            plane[0, 0] = sample.jones[0, 0]
            plane[1, 1] = sample.jones[1, 1]
            planes.append(plane)
        return HolographyDiagonalArtifact(
            samples=tuple(samples),
            jones=np.stack(planes, axis=0),
            valid=np.array([sample.copolar_valid for sample in self.samples], dtype=bool),
            off_diagonal_valid=np.zeros(len(self.samples), dtype=bool),
            calibration_state=self.calibration_state,
            source_name=self.source_name,
            reference_combination=self.reference_combination,
            frozen=False,
            notes=("diagonal projection of unfrozen full Jones",),
        )


def _invert_jones_jax(matrices):
    import jax.numpy as jnp

    a = matrices[..., 0, 0]
    b = matrices[..., 0, 1]
    c = matrices[..., 1, 0]
    d = matrices[..., 1, 1]
    det = a * d - b * c
    inverse = jnp.zeros_like(matrices)
    finite = jnp.isfinite(det) & (det != 0)
    inv_det = jnp.where(finite, 1.0 / det, jnp.nan)
    inverse = inverse.at[..., 0, 0].set(d * inv_det)
    inverse = inverse.at[..., 0, 1].set(-b * inv_det)
    inverse = inverse.at[..., 1, 0].set(-c * inv_det)
    inverse = inverse.at[..., 1, 1].set(a * inv_det)
    return inverse


def _sky_frame_residual_jax(feed, chi):
    import jax.numpy as jnp

    rotation = jnp.exp(-1j * jnp.asarray(chi, dtype=jnp.float64))
    para = jnp.zeros(feed.shape, dtype=feed.dtype)
    para = para.at[..., 0, 0].set(rotation)
    para = para.at[..., 1, 1].set(jnp.conjugate(rotation))
    return _invert_jones_jax(para) @ feed @ para


def _recover_row_jones_jax(packed, source, r_moving, r_ref, moving_is_p):
    import jax
    import jax.numpy as jnp

    def kernel(packed, source, r_moving, r_ref, moving_is_p):
        r_m = r_moving[:, None, :, :]
        r_r = r_ref[:, None, :, :]
        inverse_s_p = _invert_jones_jax(
            jnp.matmul(source, jnp.conjugate(jnp.swapaxes(r_r, -1, -2)))
        )
        inverse_s_q = _invert_jones_jax(jnp.matmul(r_r, source))
        moving_p = moving_is_p[:, None, None, None]
        inverse_s = jnp.where(moving_p, inverse_s_p, inverse_s_q)
        inv_r_m = _invert_jones_jax(r_m)
        jones_p = jnp.matmul(jnp.matmul(inv_r_m, packed), inverse_s)
        jones_q = jnp.matmul(
            inv_r_m,
            jnp.conjugate(jnp.swapaxes(jnp.matmul(inverse_s, packed), -1, -2)),
        )
        jones = jnp.where(moving_p, jones_p, jones_q)
        finite = jnp.all(jnp.isfinite(inverse_s), axis=(-2, -1)) & jnp.all(
            jnp.isfinite(jones), axis=(-2, -1)
        )
        return jones, finite

    return jax.jit(kernel)(packed, source, r_moving, r_ref, moving_is_p)


def _predict_vis_jax(r_m, beam, source, r_r, moving_is_p):
    import jax
    import jax.numpy as jnp

    def kernel(r_m, beam, source, r_r, moving_is_p):
        left = jnp.matmul(r_m[:, None, :, :], beam)
        right = jnp.conjugate(jnp.swapaxes(r_r[:, None, :, :], -1, -2))
        vis_p = jnp.matmul(left, jnp.matmul(source, right))
        vis_q = jnp.matmul(
            r_r[:, None, :, :], jnp.matmul(source, jnp.conjugate(jnp.swapaxes(left, -1, -2)))
        )
        return jnp.where(moving_is_p[:, None, None, None], vis_p, vis_q)

    return jax.jit(kernel)(r_m, beam, source, r_r, moving_is_p)


def _antenna_jones_planes(
    jones: ArrayLike | Mapping[int, ArrayLike] | None,
    antenna_id: ArrayLike,
) -> NDArray[np.complex128]:
    """Gather per-antenna 2×2 Jones without a Python row loop."""

    ants = np.asarray(antenna_id, dtype=np.int32).reshape(-1)
    identity = np.eye(2, dtype=np.complex128)
    if jones is None:
        return np.broadcast_to(identity, (ants.size, 2, 2)).copy()
    if isinstance(jones, Mapping):
        used = np.unique(ants)
        if used.size == 0:
            return np.zeros((0, 2, 2), dtype=np.complex128)
        for ant in used:
            if int(ant) not in jones:
                raise ValueError(
                    f"no residual reference Jones for antenna {int(ant)}; "
                    "do not silently substitute identity"
                )
        n_slot = int(max(int(np.max(used)), max(int(key) for key in jones))) + 1
        table = np.zeros((n_slot, 2, 2), dtype=np.complex128)
        for ant, plane in jones.items():
            table[int(ant)] = np.asarray(plane, dtype=np.complex128)
        return table[ants]
    stacked = np.asarray(jones, dtype=np.complex128)
    if stacked.shape == (2, 2):
        return np.broadcast_to(stacked, (ants.size, 2, 2)).copy()
    return stacked[ants]


def _recover_full_jones_row_arrays(
    observation: HolographyObservation,
    visibility: ArrayLike | None = None,
    *,
    mask_unsettled: bool = True,
    reference_antenna_id: int | None = None,
    reference_jones: ArrayLike | Mapping[int, ArrayLike] | None = None,
    residual_jones: ArrayLike | Mapping[int, ArrayLike] | None = None,
    parallactic_angle_rad: ArrayLike | None = None,
    row_mask: ArrayLike | None = None,
) -> dict[str, object]:
    """Invert the compact-calibrator RIME on selected moving–reference samples.

    Returns per-(row, channel) Jones before any dwell or reference average.
    Flagged correlations receive zero weight and do not invent leakage.
    """

    measured = (
        np.asarray(observation.block.visibility, dtype=np.complex128)
        if visibility is None
        else np.asarray(visibility, dtype=np.complex128)
    )
    if measured.shape != observation.block.visibility.shape:
        raise ValueError("visibility must match the observation visibility shape")
    names = tuple(
        item.value if isinstance(item, Correlation) else str(item)
        for item in observation.block.correlations
    )
    if any(name not in names for name in ("RR", "RL", "LR", "LL")):
        raise ValueError("full-Jones recovery needs RR, RL, LR, and LL")
    source = _source_coherency(observation)
    packed = pack_coherency(measured, observation.block.correlations, _RECEPTORS)
    moving_ref = moving_reference_row_mask(observation.block, observation.pointing)
    active = observation.active_row_mask(
        mask_unsettled=mask_unsettled,
        moving_reference_only=True,
    )
    if row_mask is not None:
        active = active & np.asarray(row_mask, dtype=bool).reshape(-1)
    rows = np.flatnonzero(moving_ref & active)
    if rows.size == 0:
        raise ValueError("full-Jones recovery needs usable moving--reference rows")
    offsets, inverse, pointing_valid, _settled, _moving = observation.pointing_state()
    roles = np.full(offsets.shape[:2], "unknown", dtype="U16")
    for index, antenna in enumerate(observation.pointing.antenna_id):
        roles[:, int(antenna)] = observation.pointing.role[:, index]
    unique_times, _ = unique_visibility_times(observation.block.time_s)
    weight_map = np.asarray(observation.block.weight, dtype=np.float64)
    corr_index = {name: i for i, name in enumerate(names)}
    all_antenna_residual = residual_jones if residual_jones is not None else None
    chi = None
    if all_antenna_residual is not None:
        if parallactic_angle_rad is not None:
            chi = np.asarray(parallactic_angle_rad, dtype=np.float64)
        elif observation.antenna_position_m and observation.phase_centre_rad is not None:
            from sl1mjax.calibration_terms import parallactic_angle_rad as _chi

            chi = _chi(
                unique_times,
                observation.phase_centre_rad,
                np.asarray(observation.antenna_position_m, dtype=np.float64),
            )
    time_index = np.asarray(inverse, dtype=np.int32)[rows]
    antenna_p = np.asarray(observation.block.antenna1, dtype=np.int32)[rows]
    antenna_q = np.asarray(observation.block.antenna2, dtype=np.int32)[rows]
    keep = pointing_valid[time_index, antenna_p] & pointing_valid[time_index, antenna_q]
    role_p = roles[time_index, antenna_p]
    role_q = roles[time_index, antenna_q]
    moving_is_p = role_p == "moving"
    moving_is_q = role_q == "moving"
    keep &= moving_is_p ^ moving_is_q
    moving = np.where(moving_is_p, antenna_p, antenna_q)
    reference = np.where(moving_is_p, antenna_q, antenna_p)
    if reference_antenna_id is not None:
        keep &= reference == int(reference_antenna_id)
    if not bool(np.any(keep)):
        raise ValueError("full-Jones recovery found no usable samples")
    time_index = time_index[keep]
    antenna_p = antenna_p[keep]
    moving = moving[keep]
    reference = reference[keep]
    moving_is_p = moving_is_p[keep]
    n_chan = int(observation.block.frequency_hz.size)
    source_planes = np.asarray(source, dtype=np.complex128)[rows[keep]]
    packed_sel = np.asarray(packed, dtype=np.complex128)[rows[keep]]
    flag_sel = np.asarray(observation.block.flag, dtype=bool)[rows[keep]]
    weight_sel = weight_map[rows[keep]]
    element_ok = np.empty(packed_sel.shape, dtype=bool)
    element_weight = np.empty(packed_sel.shape, dtype=np.float64)
    element_ok[..., 0, 0] = ~flag_sel[..., corr_index["RR"]]
    element_ok[..., 0, 1] = ~flag_sel[..., corr_index["RL"]]
    element_ok[..., 1, 0] = ~flag_sel[..., corr_index["LR"]]
    element_ok[..., 1, 1] = ~flag_sel[..., corr_index["LL"]]
    element_weight[..., 0, 0] = weight_sel[..., corr_index["RR"]]
    element_weight[..., 0, 1] = weight_sel[..., corr_index["RL"]]
    element_weight[..., 1, 0] = weight_sel[..., corr_index["LR"]]
    element_weight[..., 1, 1] = weight_sel[..., corr_index["LL"]]
    element_weight = np.where(element_ok, element_weight, 0.0)
    any_ok = np.any(element_ok, axis=(-2, -1))
    all_ok = np.all(element_ok, axis=(-2, -1))
    unpolarized = (
        (np.abs(source_planes[..., 0, 1]) <= 1.0e-15)
        & (np.abs(source_planes[..., 1, 0]) <= 1.0e-15)
        & (
            np.abs(source_planes[..., 0, 0] - source_planes[..., 1, 1])
            <= 1.0e-12 * np.maximum(np.abs(source_planes[..., 0, 0]), 1.0)
        )
    )
    usable = any_ok & (unpolarized | all_ok)
    import jax.numpy as jnp

    if all_antenna_residual is not None:
        r_moving = _antenna_jones_planes(all_antenna_residual, moving)
        r_ref = _antenna_jones_planes(all_antenna_residual, reference)
        if chi is not None:
            r_moving = np.asarray(
                _sky_frame_residual_jax(jnp.asarray(r_moving), jnp.asarray(chi[time_index, moving]))
            )
            r_ref = np.asarray(
                _sky_frame_residual_jax(jnp.asarray(r_ref), jnp.asarray(chi[time_index, reference]))
            )
    else:
        r_moving = np.broadcast_to(np.eye(2, dtype=np.complex128), (moving.size, 2, 2)).copy()
        r_ref = _antenna_jones_planes(reference_jones, reference)
    jones, jones_ok = _recover_row_jones_jax(
        jnp.asarray(packed_sel),
        jnp.asarray(source_planes),
        jnp.asarray(r_moving),
        jnp.asarray(r_ref),
        jnp.asarray(moving_is_p),
    )
    jones = np.asarray(jones)
    usable &= np.asarray(jones_ok)
    if not bool(np.any(usable)):
        raise ValueError("full-Jones recovery found no usable samples")
    visits = visit_id_per_time(observation)
    return {
        "usable": usable,
        "jones": jones,
        "element_weight": element_weight,
        "moving": np.asarray(moving, dtype=np.int32),
        "reference": np.asarray(reference, dtype=np.int32),
        "time_index": np.asarray(time_index, dtype=np.int32),
        "antenna_p": np.asarray(antenna_p, dtype=np.int32),
        "rows": np.asarray(rows[keep], dtype=np.int64),
        "n_chan": n_chan,
        "unique_times": unique_times,
        "offsets": offsets,
        "visit_id": np.asarray(visits, dtype=np.int32)[time_index],
        "all_antenna_residual": all_antenna_residual,
        "reference_jones": reference_jones,
    }


def recover_holography_full_jones_rows(
    observation: HolographyObservation,
    visibility: ArrayLike | None = None,
    *,
    mask_unsettled: bool = True,
    reference_antenna_id: int | None = None,
    reference_jones: ArrayLike | Mapping[int, ArrayLike] | None = None,
    residual_jones: ArrayLike | Mapping[int, ArrayLike] | None = None,
    parallactic_angle_rad: ArrayLike | None = None,
    row_mask: ArrayLike | None = None,
) -> dict[str, NDArray]:
    """Per-row, per-channel Jones before dwell averaging or the SNR mask."""

    arrays = _recover_full_jones_row_arrays(
        observation,
        visibility,
        mask_unsettled=mask_unsettled,
        reference_antenna_id=reference_antenna_id,
        reference_jones=reference_jones,
        residual_jones=residual_jones,
        parallactic_angle_rad=parallactic_angle_rad,
        row_mask=row_mask,
    )
    usable = np.asarray(arrays["usable"], dtype=bool)
    n_chan = int(arrays["n_chan"])
    channel = np.broadcast_to(np.arange(n_chan, dtype=np.int32), usable.shape)
    moving = np.broadcast_to(np.asarray(arrays["moving"])[:, None], usable.shape)
    reference = np.broadcast_to(np.asarray(arrays["reference"])[:, None], usable.shape)
    time_index = np.broadcast_to(np.asarray(arrays["time_index"])[:, None], usable.shape)
    visit = np.broadcast_to(np.asarray(arrays["visit_id"])[:, None], usable.shape)
    rows = np.broadcast_to(np.asarray(arrays["rows"])[:, None], usable.shape)
    offsets = np.asarray(arrays["offsets"], dtype=np.float64)
    unique_times = np.asarray(arrays["unique_times"], dtype=np.float64)
    offset = offsets[np.asarray(arrays["time_index"]), np.asarray(arrays["moving"])]
    offset = np.broadcast_to(offset[:, None, :], usable.shape + (2,))
    frequency = np.broadcast_to(observation.block.frequency_hz[None, :], usable.shape)
    return {
        "row": rows[usable],
        "channel": channel[usable],
        "moving_id": moving[usable],
        "reference_id": reference[usable],
        "time_index": time_index[usable],
        "visit_id": visit[usable],
        "unique_time_s": unique_times[time_index[usable]],
        "offset_lm_rad": offset[usable],
        "frequency_hz": frequency[usable],
        "jones": np.asarray(arrays["jones"], dtype=np.complex128)[usable],
        "weight": np.asarray(arrays["element_weight"], dtype=np.float64)[usable],
    }


def recover_holography_full_jones(
    observation: HolographyObservation,
    visibility: ArrayLike | None = None,
    *,
    mask_unsettled: bool = True,
    reference_antenna_id: int | None = None,
    reference_jones: ArrayLike | Mapping[int, ArrayLike] | None = None,
    residual_jones: ArrayLike | Mapping[int, ArrayLike] | None = None,
    parallactic_angle_rad: ArrayLike | None = None,
    row_mask: ArrayLike | None = None,
) -> HolographyFullJonesArtifact:
    """Recover native 2×2 Jones from :math:`V=R_m E_m(s) S R_r^H`.

    ``residual_jones`` is the feed-frame :math:`I+\\varepsilon` for every
    antenna, including movers. When supplied it is rotated by P(χ) into
    the sky frame. ``reference_jones`` remains the identity-on-moving
    fallback used by the blocked identity-gauge product.
    Flagged correlations receive zero weight and do not invent leakage.
    The artifact is never frozen.
    """

    arrays = _recover_full_jones_row_arrays(
        observation,
        visibility,
        mask_unsettled=mask_unsettled,
        reference_antenna_id=reference_antenna_id,
        reference_jones=reference_jones,
        residual_jones=residual_jones,
        parallactic_angle_rad=parallactic_angle_rad,
        row_mask=row_mask,
    )
    usable = np.asarray(arrays["usable"], dtype=bool)
    n_chan = int(arrays["n_chan"])
    jones = np.asarray(arrays["jones"])
    element_weight = np.asarray(arrays["element_weight"])
    moving = np.asarray(arrays["moving"])
    time_index = np.asarray(arrays["time_index"])
    unique_times = np.asarray(arrays["unique_times"])
    offsets = np.asarray(arrays["offsets"])
    all_antenna_residual = arrays["all_antenna_residual"]
    reference_jones = arrays["reference_jones"]
    memo = Memo195LowerCRaster()
    channel = np.broadcast_to(np.arange(n_chan, dtype=np.int32), usable.shape)
    moving_f = np.broadcast_to(moving[:, None], usable.shape)
    time_f = np.broadcast_to(time_index[:, None], usable.shape)
    keys = np.stack(
        [moving_f[usable], time_f[usable], channel[usable]],
        axis=1,
    )
    uniq, inverse_g = np.unique(keys, axis=0, return_inverse=True)
    n_group = int(uniq.shape[0])
    planes_f = jones[usable]
    weights_f = element_weight[usable]
    total = np.zeros((n_group, 2, 2), dtype=np.float64)
    weighted = np.zeros((n_group, 2, 2), dtype=np.complex128)
    np.add.at(total, inverse_g, weights_f)
    np.add.at(weighted, inverse_g, weights_f * planes_f)
    mean = np.zeros((n_group, 2, 2), dtype=np.complex128)
    supported = total > 0.0
    mean = np.where(supported, weighted / np.maximum(total, 1.0e-30), 0.0)
    scatter_acc = np.zeros((n_group, 2, 2), dtype=np.float64)
    np.add.at(scatter_acc, inverse_g, weights_f * np.abs(planes_f - mean[inverse_g]) ** 2)
    scatter = np.sqrt(scatter_acc / np.maximum(total, 1.0e-30))
    sigma = np.full((n_group, 2, 2), np.nan, dtype=np.float64)
    sigma = np.where(
        supported,
        np.where(scatter > 0.0, scatter, 1.0 / np.sqrt(np.maximum(total, 1.0e-30))),
        np.nan,
    )
    counts = np.zeros(n_group, dtype=np.int32)
    np.add.at(counts, inverse_g, 1)
    order = np.lexsort((uniq[:, 2], uniq[:, 1], uniq[:, 0]))
    samples: list[HolographyFullJonesSample] = []
    planes = []
    copolar = []
    leakage = []
    for index in order:
        moving_id = int(uniq[index, 0])
        t_index = int(uniq[index, 1])
        chan = int(uniq[index, 2])
        offset = np.asarray(offsets[t_index, moving_id], dtype=np.float64)
        plane = np.asarray(mean[index], dtype=np.complex128)
        sample = HolographyFullJonesSample(
            moving_antenna_id=moving_id,
            unique_time_s=float(unique_times[t_index]),
            offset_lm_rad=offset,
            frequency_hz=float(observation.block.frequency_hz[chan]),
            jones=plane,
            sigma=np.asarray(sigma[index], dtype=np.float64),
            n_reference=int(counts[index]),
            weight=float(np.mean(total[index])),
            copolar_valid=bool(total[index, 0, 0] > 0.0 and total[index, 1, 1] > 0.0),
            off_diagonal_valid=bool(total[index, 0, 1] > 0.0 and total[index, 1, 0] > 0.0),
            raster=_raster_label(offset, memo),
        )
        samples.append(sample)
        planes.append(plane)
        copolar.append(sample.copolar_valid)
        leakage.append(sample.off_diagonal_valid)
    source_version = (
        observation.source_model.standard
        if observation.source_model is not None
        else "stokes_i_point"
    )
    return HolographyFullJonesArtifact(
        samples=tuple(samples),
        jones=np.stack(planes, axis=0),
        valid=np.asarray(copolar, dtype=bool),
        off_diagonal_valid=np.asarray(leakage, dtype=bool),
        calibration_state=observation.calibration_state,
        source_name=observation.source_name,
        source_model_version=source_version,
        reference_combination="inverse_variance_moving_reference",
        receptor_convention="circular_R_L",
        offset_sign=observation.pointing.offset_sign,
        frozen=False,
        notes=(
            "empirical full Jones; not frozen",
            "RL and LR are independent; conjugacy is not imposed",
            "off-diagonal support is limited to sampled raster cells",
            (
                "all-antenna residual Jones R_m and R_r applied"
                if all_antenna_residual is not None
                else "per-antenna residual reference Jones supplied"
                if isinstance(reference_jones, Mapping)
                else "reference Jones supplied"
                if reference_jones is not None
                else "reference antennas treated as on-axis identity"
            ),
        ),
    )


def pin_on_axis_jones_to_identity(
    artifact: HolographyFullJonesArtifact,
    *,
    on_axis_atol_rad: float = 1.0e-6,
) -> HolographyFullJonesArtifact:
    """Left-multiply each antenna/channel so :math:`E_p(0)=I`.

    The nearest-to-axis sample is treated as :math:`E_p(0)`. After this
    pin, :math:`R_p(t)` holds the complete on-axis residual and
    :math:`E_p(s)` holds only direction-dependent variation.
    """

    if artifact.frozen:
        raise ValueError("empirical holography full Jones is not frozen")
    groups: dict[tuple[int, int], list[int]] = {}
    for index, sample in enumerate(artifact.samples):
        groups.setdefault(
            (int(sample.moving_antenna_id), int(round(sample.frequency_hz))),
            [],
        ).append(index)
    planes = np.array(artifact.jones, copy=True)
    samples = list(artifact.samples)
    n_pin = 0
    used_nearest = False
    for indices in groups.values():
        radii = [
            float(
                np.hypot(
                    float(samples[index].offset_lm_rad[0]), float(samples[index].offset_lm_rad[1])
                )
            )
            for index in indices
        ]
        nearest = indices[int(np.argmin(radii))]
        if radii[int(np.argmin(radii))] > float(on_axis_atol_rad):
            used_nearest = True
        on_axis = planes[nearest]
        if not bool(np.all(np.isfinite(on_axis))) or abs(complex(np.linalg.det(on_axis))) < 1.0e-12:
            continue
        inverse = invert_jones(on_axis)
        for index in indices:
            pinned = inverse @ planes[index]
            planes[index] = pinned
            samples[index] = replace(samples[index], jones=pinned)
        n_pin += 1
    if n_pin == 0:
        return artifact
    return replace(
        artifact,
        samples=tuple(samples),
        jones=planes,
        notes=artifact.notes
        + (
            ON_AXIS_BEAM_IDENTITY_NOTE,
            f"pinned {n_pin} antenna/channel groups to E_p(0)=I",
            "used nearest sample as E_p(0)" if used_nearest else "used on-axis sample as E_p(0)",
        ),
    )


def average_holography_full_jones(
    artifact: HolographyFullJonesArtifact,
    *,
    offset_atol_rad: float = 2.908882086657216e-05,
) -> HolographyFullJonesArtifact:
    """Inverse-variance average of antenna-specific cells at matching offsets.

    Antenna-specific samples stay in the input artifact. The returned product
    is an unfrozen array-average. Residual antenna scatter is the uncertainty.
    """

    if artifact.frozen:
        raise ValueError("empirical holography full Jones is not frozen")
    groups: dict[tuple[int, int, int], list[HolographyFullJonesSample]] = {}
    scale = 1.0 / float(offset_atol_rad)
    for sample in artifact.samples:
        if not sample.copolar_valid or sample.moving_antenna_id < 0:
            continue
        key = (
            int(round(float(sample.offset_lm_rad[0]) * scale)),
            int(round(float(sample.offset_lm_rad[1]) * scale)),
            int(round(sample.frequency_hz)),
        )
        groups.setdefault(key, []).append(sample)
    if not groups:
        raise ValueError("array-average needs antenna-specific holography samples")
    samples: list[HolographyFullJonesSample] = []
    planes = []
    copolar = []
    leakage = []
    included: set[int] = set()
    for key in sorted(groups):
        members = groups[key]
        if len({sample.moving_antenna_id for sample in members}) < 2:
            continue
        included.update(sample.moving_antenna_id for sample in members)
        mean, sigma, n_ref, weight, ok, leak_ok = _combine_jones(
            [(sample.jones, 1.0 / np.maximum(sample.sigma, 1.0e-15) ** 2) for sample in members]
        )
        offset = np.mean([sample.offset_lm_rad for sample in members], axis=0)
        sample = HolographyFullJonesSample(
            moving_antenna_id=ARRAY_AVERAGE_ANTENNA_ID,
            unique_time_s=float(np.mean([member.unique_time_s for member in members])),
            offset_lm_rad=np.asarray(offset, dtype=np.float64),
            frequency_hz=float(members[0].frequency_hz),
            jones=mean,
            sigma=sigma,
            n_reference=n_ref,
            weight=weight,
            copolar_valid=ok,
            off_diagonal_valid=leak_ok and all(member.off_diagonal_valid for member in members),
            raster=members[0].raster,
        )
        samples.append(sample)
        planes.append(mean)
        copolar.append(ok)
        leakage.append(sample.off_diagonal_valid)
    if not samples:
        raise ValueError("array-average needs at least two moving antennas at one cell")
    antennas = ",".join(str(antenna) for antenna in sorted(included))
    return HolographyFullJonesArtifact(
        samples=tuple(samples),
        jones=np.stack(planes, axis=0),
        valid=np.asarray(copolar, dtype=bool),
        off_diagonal_valid=np.asarray(leakage, dtype=bool),
        calibration_state=artifact.calibration_state,
        source_name=artifact.source_name,
        source_model_version=artifact.source_model_version,
        reference_combination="inverse_variance_array_average",
        receptor_convention=artifact.receptor_convention,
        offset_sign=artifact.offset_sign,
        frozen=False,
        notes=artifact.notes
        + (
            "array-average of antenna-specific samples; not frozen",
            f"included_moving_antennas={antennas}",
        ),
    )


def interpolate_holography_full_jones(
    artifact: HolographyFullJonesArtifact,
    offset_lm_rad: ArrayLike,
    *,
    moving_antenna_id: int,
    frequency_hz: float,
) -> tuple[NDArray[np.complex128], bool, bool]:
    """Inverse-distance in commanded ``(l, m)``, then linear in frequency.

    Does not invent leakage outside the sampled spatial box or frequency span.
    """

    query = np.asarray(offset_lm_rad, dtype=np.float64).reshape(2)
    exact = _interpolation_candidates(artifact, moving_antenna_id, frequency_hz)
    if exact:
        return _spatial_full_jones(exact, query)
    by_frequency: dict[int, list[HolographyFullJonesSample]] = {}
    for sample in _interpolation_candidates(artifact, moving_antenna_id, None):
        by_frequency.setdefault(int(round(sample.frequency_hz)), []).append(sample)
    frequencies: list[float] = []
    planes: list[np.ndarray] = []
    leaks: list[bool] = []
    for members in by_frequency.values():
        plane, ok, leak = _spatial_full_jones(members, query)
        if ok:
            frequencies.append(float(members[0].frequency_hz))
            planes.append(plane)
            leaks.append(leak)
    zero = np.zeros((2, 2), dtype=np.complex128)
    if len(frequencies) < 2:
        return zero, False, False
    order = np.argsort(frequencies)
    freqs = np.asarray(frequencies, dtype=np.float64)[order]
    if float(frequency_hz) < float(freqs[0]) - 1.0 or float(frequency_hz) > float(freqs[-1]) + 1.0:
        return zero, False, False
    stacked = np.stack([planes[int(index)] for index in order], axis=0)
    jones = np.zeros((2, 2), dtype=np.complex128)
    for row in range(2):
        for col in range(2):
            jones[row, col] = _interp_complex_delay(frequency_hz, freqs, stacked[:, row, col])
    return jones, True, all(leaks)


def interpolate_holography_full_jones_batch(
    artifact: HolographyFullJonesArtifact,
    offset_lm_rad: ArrayLike,
    *,
    moving_antenna_id: ArrayLike,
    frequency_hz: ArrayLike,
) -> tuple[NDArray[np.complex128], NDArray[np.bool_], NDArray[np.bool_]]:
    """Training-only spatial interpolation for many query cells."""

    queries = np.asarray(offset_lm_rad, dtype=np.float64).reshape(-1, 2)
    movers = np.asarray(moving_antenna_id, dtype=np.int32).reshape(-1)
    frequencies = np.asarray(frequency_hz, dtype=np.float64).reshape(-1)
    if queries.shape[0] != movers.size or queries.shape[0] != frequencies.size:
        raise ValueError("offset, moving_antenna_id, and frequency_hz must match")
    jones = np.zeros((queries.shape[0], 2, 2), dtype=np.complex128)
    ok = np.zeros(queries.shape[0], dtype=bool)
    leak = np.zeros(queries.shape[0], dtype=bool)
    groups: dict[tuple[int, int], list[int]] = {}
    for index, (mover, frequency) in enumerate(zip(movers, frequencies, strict=True)):
        groups.setdefault((int(mover), int(round(float(frequency)))), []).append(index)
    for (mover, _freq_key), rows in groups.items():
        frequency = float(frequencies[rows[0]])
        candidates = _interpolation_candidates(artifact, mover, frequency)
        if candidates:
            plane, valid, leakage = _spatial_full_jones_batch(candidates, queries[np.asarray(rows)])
            jones[rows] = plane
            ok[rows] = valid
            leak[rows] = leakage
            continue
        fallback = _interpolation_candidates(artifact, mover, None)
        if not fallback:
            continue
        by_frequency: dict[int, list[HolographyFullJonesSample]] = {}
        for sample in fallback:
            by_frequency.setdefault(int(round(sample.frequency_hz)), []).append(sample)
        if len(by_frequency) < 2:
            continue
        freqs = []
        planes = []
        leaks = []
        valids = None
        for members in by_frequency.values():
            plane, valid, leakage = _spatial_full_jones_batch(members, queries[np.asarray(rows)])
            freqs.append(np.full(len(rows), float(members[0].frequency_hz)))
            planes.append(plane)
            leaks.append(leakage)
            valids = valid if valids is None else valids & valid
        order = np.argsort([float(item[0]) for item in freqs])
        freq_axis = np.stack([freqs[int(index)] for index in order], axis=1)
        stacked = np.stack([planes[int(index)] for index in order], axis=1)
        if (
            float(frequency) < float(np.min(freq_axis)) - 1.0
            or float(frequency) > float(np.max(freq_axis)) + 1.0
        ):
            continue
        interp = np.zeros((len(rows), 2, 2), dtype=np.complex128)
        for row in range(2):
            for col in range(2):
                for item, freq_row, values in zip(
                    range(len(rows)), freq_axis, stacked[..., row, col], strict=True
                ):
                    interp[item, row, col] = _interp_complex_delay(frequency, freq_row, values)
        jones[rows] = interp
        ok[rows] = False if valids is None else valids
        leak[rows] = np.all(np.stack(leaks, axis=0), axis=0) if leaks else False
    return jones, ok, leak


def _spatial_full_jones_batch(
    candidates: list[HolographyFullJonesSample],
    queries: np.ndarray,
) -> tuple[NDArray[np.complex128], NDArray[np.bool_], NDArray[np.bool_]]:
    n_query = int(queries.shape[0])
    jones = np.zeros((n_query, 2, 2), dtype=np.complex128)
    ok = np.zeros(n_query, dtype=bool)
    leak = np.zeros(n_query, dtype=bool)
    by_raster: dict[str, list[HolographyFullJonesSample]] = {}
    for sample in candidates:
        by_raster.setdefault(str(getattr(sample, "raster", "dense")), []).append(sample)
    picked_dist = []
    picked_jones = []
    picked_leak = []
    exact_plane = np.zeros((n_query, 2, 2), dtype=np.complex128)
    exact_ok = np.zeros(n_query, dtype=bool)
    exact_leak = np.zeros(n_query, dtype=bool)
    for members in by_raster.values():
        offsets = np.asarray([sample.offset_lm_rad for sample in members], dtype=np.float64)
        planes = np.stack(
            [np.asarray(sample.jones, dtype=np.complex128) for sample in members], axis=0
        )
        leak_ok = np.asarray([sample.off_diagonal_valid for sample in members], dtype=bool)
        spacing = _median_spacing(offsets)
        radius = 1.5 * spacing if spacing > 0.0 else 1.0e-15
        dist = np.hypot(
            queries[:, None, 0] - offsets[None, :, 0],
            queries[:, None, 1] - offsets[None, :, 1],
        )
        exact = dist <= 1.0e-15
        has_exact = np.any(exact, axis=1)
        nearest = np.argmin(np.where(exact, dist, np.inf), axis=1)
        exact_plane[has_exact] = planes[nearest[has_exact]]
        exact_ok[has_exact] = True
        exact_leak[has_exact] = leak_ok[nearest[has_exact]]
        usable = (~has_exact)[:, None] & (dist <= radius + 1.0e-15)
        if not bool(np.any(usable)):
            continue
        ranked = np.where(usable, dist, np.inf)
        k = min(4, ranked.shape[1])
        order = np.argpartition(ranked, k - 1, axis=1)[:, :k]
        gather = np.take_along_axis(ranked, order, axis=1)
        picked_dist.append(gather)
        picked_jones.append(planes[order])
        picked_leak.append(leak_ok[order])
    if picked_dist:
        distances = np.concatenate(picked_dist, axis=1)
        planes = np.concatenate(picked_jones, axis=1)
        leak_ok = np.concatenate(picked_leak, axis=1)
        finite = np.isfinite(distances)
        weights = np.zeros_like(distances)
        np.divide(1.0, np.maximum(distances, 1.0e-15), out=weights, where=finite)
        weight_sum = np.sum(weights, axis=1, keepdims=True)
        finite_row = (weight_sum[:, 0] > 0.0) & ~exact_ok
        if bool(np.any(finite_row)):
            np.divide(weights, weight_sum, out=weights, where=weight_sum > 0.0)
            copolar = np.sum(weights[finite_row, :, None, None] * planes[finite_row], axis=1)
            jones[finite_row, 0, 0] = copolar[:, 0, 0]
            jones[finite_row, 1, 1] = copolar[:, 1, 1]
            leak_w = np.where(leak_ok[finite_row], weights[finite_row], 0.0)
            leak_den = np.sum(leak_w, axis=1)
            leak_num = np.sum(leak_w[:, :, None, None] * planes[finite_row], axis=1)
            has_leak = leak_den > 0.0
            jones[finite_row, 0, 1] = np.where(
                has_leak, leak_num[:, 0, 1] / np.maximum(leak_den, 1.0e-30), np.nan
            )
            jones[finite_row, 1, 0] = np.where(
                has_leak, leak_num[:, 1, 0] / np.maximum(leak_den, 1.0e-30), np.nan
            )
            ok[finite_row] = True
            leak[finite_row] = has_leak
    jones[exact_ok] = exact_plane[exact_ok]
    ok[exact_ok] = True
    leak[exact_ok] = exact_leak[exact_ok]
    return jones, ok, leak


def _spatial_full_jones(
    candidates: list[HolographyFullJonesSample],
    query: np.ndarray,
) -> tuple[NDArray[np.complex128], bool, bool]:
    zero = np.zeros((2, 2), dtype=np.complex128)
    selected = _local_neighbors(candidates, query)
    if not selected:
        return zero, False, False
    offsets = np.array([sample.offset_lm_rad for sample in selected], dtype=np.float64)
    distances = np.hypot(offsets[:, 0] - query[0], offsets[:, 1] - query[1])
    if float(np.min(distances)) <= 1.0e-15:
        sample = selected[int(np.argmin(distances))]
        return np.array(sample.jones, copy=True), True, bool(sample.off_diagonal_valid)
    weights = 1.0 / np.maximum(distances, 1.0e-15)
    weights = weights / np.sum(weights)
    jones = np.zeros((2, 2), dtype=np.complex128)
    leak_num = np.zeros((2, 2), dtype=np.complex128)
    leak_den = 0.0
    for weight, sample in zip(weights, selected, strict=True):
        plane = np.asarray(sample.jones, dtype=np.complex128)
        jones[0, 0] = jones[0, 0] + weight * plane[0, 0]
        jones[1, 1] = jones[1, 1] + weight * plane[1, 1]
        if sample.off_diagonal_valid and np.isfinite(plane[0, 1]) and np.isfinite(plane[1, 0]):
            leak_num = leak_num + weight * plane
            leak_den += float(weight)
    if leak_den > 0.0:
        jones[0, 1] = leak_num[0, 1] / leak_den
        jones[1, 0] = leak_num[1, 0] / leak_den
        leakage = True
    else:
        jones[0, 1] = np.nan + 1j * np.nan
        jones[1, 0] = np.nan + 1j * np.nan
        leakage = False
    return jones, True, leakage


def mask_unsupported_off_diagonals(
    artifact: HolographyFullJonesArtifact,
    *,
    min_snr: float = 3.0,
) -> HolographyFullJonesArtifact:
    """Clear unsupported off-diagonals. Invalid samples stay invalid, not zero."""

    samples = []
    planes = []
    leakage = []
    for sample in artifact.samples:
        sigma = np.asarray(sample.sigma, dtype=np.float64)
        plane = np.array(sample.jones, dtype=np.complex128, copy=True)
        snr_rl = np.abs(plane[0, 1]) / np.maximum(sigma[0, 1], 1.0e-30)
        snr_lr = np.abs(plane[1, 0]) / np.maximum(sigma[1, 0], 1.0e-30)
        supported = bool(
            sample.off_diagonal_valid
            and np.isfinite(snr_rl)
            and np.isfinite(snr_lr)
            and (snr_rl >= float(min_snr) or snr_lr >= float(min_snr))
        )
        if not supported:
            plane[0, 1] = np.nan + 1j * np.nan
            plane[1, 0] = np.nan + 1j * np.nan
        samples.append(replace(sample, jones=plane, off_diagonal_valid=supported))
        planes.append(plane)
        leakage.append(supported)
    return replace(
        artifact,
        samples=tuple(samples),
        jones=np.stack(planes, axis=0),
        off_diagonal_valid=np.asarray(leakage, dtype=bool),
        notes=artifact.notes
        + (f"off-diagonals with SNR < {float(min_snr):.1f} are invalid, not zero",),
    )


def zero_unsupported_off_diagonals(
    artifact: HolographyFullJonesArtifact,
    *,
    min_snr: float = 3.0,
) -> HolographyFullJonesArtifact:
    """Training-only SNR mask. Unsupported leaks become zero, not dropped rows."""

    masked = mask_unsupported_off_diagonals(artifact, min_snr=min_snr)
    planes = np.array(masked.jones, dtype=np.complex128, copy=True)
    leak = np.asarray(masked.off_diagonal_valid, dtype=bool)
    planes[~leak, 0, 1] = 0.0
    planes[~leak, 1, 0] = 0.0
    samples = tuple(
        replace(sample, jones=plane) for sample, plane in zip(masked.samples, planes, strict=True)
    )
    return replace(
        masked,
        samples=samples,
        jones=planes,
        notes=masked.notes
        + ("unsupported off-diagonals are zeroed; those rows stay in the total score",),
    )


class EmpiricalHolographyVoltageBeam:
    """Unfrozen interpolating backend for recovered holography Jones.

    Production imaging must not use this until a freeze pin exists.
    """

    model_id: str = "holography_empirical_unfrozen"
    antenna_planes_from_parallactic: bool = False

    def __init__(
        self,
        artifact: HolographyFullJonesArtifact,
        *,
        allow_unfrozen: bool = False,
    ) -> None:
        if artifact.frozen:
            raise ValueError("empirical holography full Jones is not frozen")
        if not allow_unfrozen:
            raise ValueError(
                "empirical holography beam is unfrozen; pass allow_unfrozen=True "
                "for isolated diagnostics only"
            )
        self.artifact = artifact
        self.allow_unfrozen = True

    def evaluate(
        self,
        coordinates: BeamCoordinates,
        *,
        calibration_state: BeamCalibrationState | str,
    ) -> BeamEvaluation:
        state = require_beam_calibration_state(calibration_state)
        if state.value != self.artifact.calibration_state:
            raise ValueError(
                "empirical holography beam calibration_state does not match the artifact"
            )
        l_off, m_off = _pointing_relative_lm(coordinates)
        antennas = (
            np.array([0], dtype=np.int32)
            if coordinates.antenna_id is None
            else np.asarray(coordinates.antenna_id)
        )
        n_dir = int(l_off.size)
        n_chan = int(coordinates.frequency_hz.size)
        jones = np.zeros((antennas.size, n_dir, n_chan, 2, 2), dtype=np.complex128)
        valid = np.zeros((antennas.size, n_dir, n_chan), dtype=bool)
        leakage = np.zeros(valid.shape, dtype=bool)
        used_anchor = False
        for antenna_index, antenna in enumerate(antennas):
            for direction in range(n_dir):
                commanded = np.array([-l_off[direction], -m_off[direction]], dtype=np.float64)
                on_axis = float(np.hypot(l_off[direction], m_off[direction])) <= 1.0e-15
                for channel, frequency in enumerate(coordinates.frequency_hz):
                    plane, ok, leak = interpolate_holography_full_jones(
                        self.artifact,
                        commanded,
                        moving_antenna_id=int(antenna),
                        frequency_hz=float(frequency),
                    )
                    if (not ok) and on_axis and state is BeamCalibrationState.CASA_PARANG_TRUE:
                        plane = np.eye(2, dtype=np.complex128)
                        ok = True
                        leak = True
                        used_anchor = True
                    jones[antenna_index, direction, channel] = plane
                    valid[antenna_index, direction, channel] = ok
                    leakage[antenna_index, direction, channel] = leak
        return BeamEvaluation(
            jones=jones,
            valid=valid,
            off_diagonal_valid=leakage,
            provenance={
                "model_id": self.model_id,
                "artifact_id": "holography_empirical_unfrozen",
                "kind": "empirical",
                "frozen": False,
                "calibration_state": state.value,
                "experimental": True,
                "on_axis_identity_anchor": used_anchor,
            },
        )


def _interpolation_candidates(
    artifact: HolographyFullJonesArtifact,
    moving_antenna_id: int,
    frequency_hz: float | None,
) -> list[HolographyFullJonesSample]:
    matching = [
        sample
        for sample in artifact.samples
        if sample.copolar_valid
        and (
            frequency_hz is None
            or np.isclose(sample.frequency_hz, frequency_hz, rtol=0.0, atol=1.0)
        )
    ]
    specific = [sample for sample in matching if sample.moving_antenna_id == moving_antenna_id]
    if specific:
        return specific
    return [sample for sample in matching if sample.moving_antenna_id < 0]


def _combine_jones(
    estimates: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[NDArray[np.complex128], NDArray[np.float64], int, float, bool, bool]:
    planes = np.stack([item[0] for item in estimates], axis=0)
    weights = np.stack([item[1] for item in estimates], axis=0)
    total = np.sum(weights, axis=0)
    mean = np.zeros((2, 2), dtype=np.complex128)
    sigma = np.full((2, 2), np.nan)
    usable = total > 0.0
    if np.any(usable):
        mean = np.where(usable, np.sum(weights * planes, axis=0) / np.maximum(total, 1.0e-30), 0.0)
        scatter = np.sqrt(
            np.sum(weights * np.abs(planes - mean) ** 2, axis=0) / np.maximum(total, 1.0e-30)
        )
        sigma = np.where(
            usable,
            np.where(scatter > 0.0, scatter, 1.0 / np.sqrt(np.maximum(total, 1.0e-30))),
            np.nan,
        )
    copolar_ok = bool(total[0, 0] > 0.0 and total[1, 1] > 0.0)
    leak_ok = bool(total[0, 1] > 0.0 and total[1, 0] > 0.0)
    return (
        np.asarray(mean, dtype=np.complex128),
        np.asarray(sigma, dtype=np.float64),
        len(estimates),
        float(np.mean(total)),
        copolar_ok,
        leak_ok,
    )


def offdiag_beam_map_difference(
    first: HolographyFullJonesArtifact,
    second: HolographyFullJonesArtifact,
    *,
    offset_atol_rad: float = 2.908882086657216e-05,
) -> dict[str, object]:
    """Compare recovered RL/LR maps, not copolar amplitudes."""

    scale = 1.0 / float(offset_atol_rad)

    def _key(sample: HolographyFullJonesSample) -> tuple[int, int, int, int]:
        return (
            int(sample.moving_antenna_id),
            int(round(float(sample.offset_lm_rad[0]) * scale)),
            int(round(float(sample.offset_lm_rad[1]) * scale)),
            int(round(sample.frequency_hz)),
        )

    by_key = {_key(sample): sample.jones for sample in second.samples if sample.off_diagonal_valid}
    rl = []
    lr = []
    for sample in first.samples:
        if not sample.off_diagonal_valid:
            continue
        other = by_key.get(_key(sample))
        if other is None:
            continue
        left = np.asarray(sample.jones, dtype=np.complex128)
        right = np.asarray(other, dtype=np.complex128)
        rl.append(float(np.abs(left[0, 1] - right[0, 1])))
        lr.append(float(np.abs(left[1, 0] - right[1, 0])))
    return {
        "n": len(rl),
        "median_abs_rl": float(np.median(rl)) if rl else float("nan"),
        "median_abs_lr": float(np.median(lr)) if lr else float("nan"),
        "notes": (PREDICTION_EQUIVALENT_BEAM_NOTE,),
    }


def classify_prediction_equivalent_beam_maps(
    *,
    visibility_equivalent: bool,
    median_abs_rl_map_delta: float,
    median_abs_rl_map_delta_after_pin: float,
    map_threshold: float = 0.01,
    full_jones_rl_residual: float | None = None,
    diagonal_rl_residual: float | None = None,
    full_jones_rr_residual: float | None = None,
    predictions_agree_after_pin: bool | None = None,
    n_failing_antennas: int = 0,
    n_antennas: int | None = None,
) -> dict[str, object]:
    """Decide whether factorization ambiguity survives the beam gauge."""

    maps_differ = np.isfinite(median_abs_rl_map_delta) and float(median_abs_rl_map_delta) >= float(
        map_threshold
    )
    maps_resolved = np.isfinite(median_abs_rl_map_delta_after_pin) and float(
        median_abs_rl_map_delta_after_pin
    ) < float(map_threshold)
    beats_diagonal = None
    if (
        full_jones_rl_residual is not None
        and diagonal_rl_residual is not None
        and np.isfinite(full_jones_rl_residual)
        and np.isfinite(diagonal_rl_residual)
    ):
        beats_diagonal = float(full_jones_rl_residual) < 0.8 * max(
            float(diagonal_rl_residual), 1.0e-4
        )
    predictions_ok = (
        bool(visibility_equivalent)
        if predictions_agree_after_pin is None
        else bool(predictions_agree_after_pin)
    )
    few_fail = (
        int(n_failing_antennas) > 0
        and n_antennas is not None
        and int(n_failing_antennas) <= max(2, int(np.ceil(0.2 * int(n_antennas))))
    )
    copolar_broken = (
        full_jones_rr_residual is not None
        and np.isfinite(full_jones_rr_residual)
        and float(full_jones_rr_residual) >= 0.15
    )
    if few_fail and maps_resolved:
        outcome = "antenna_support_problem"
        status = "warn"
        blocking = False
    elif maps_differ and not maps_resolved:
        outcome = "maps_still_differ_after_pin"
        status = "fail"
        blocking = True
    elif maps_resolved and beats_diagonal is False:
        if copolar_broken:
            outcome = "inconclusive"
            status = "warn"
            blocking = False
        else:
            outcome = "maps_agree_offdiag_lack_predictive_evidence"
            status = "warn"
            blocking = False
    elif maps_resolved and predictions_ok:
        outcome = "maps_and_predictions_agree_after_pin"
        status = "pass"
        blocking = False
    else:
        outcome = "inconclusive"
        status = "warn"
        blocking = False
    unique_physical = maps_resolved
    return {
        "status": status,
        "blocking": blocking,
        "outcome": outcome,
        "visibility_equivalent": bool(visibility_equivalent),
        "maps_differ_before_pin": maps_differ,
        "maps_agree_after_pin": maps_resolved,
        "predictions_agree_after_pin": predictions_ok,
        "full_jones_beats_diagonal_rl": beats_diagonal,
        "copolar_prediction_broken": copolar_broken,
        "n_failing_antennas": int(n_failing_antennas),
        "jones_factors_not_unique": True,
        "unpinned_maps_are_diagnostic": True,
        "beam_maps_uniquely_physical": unique_physical if visibility_equivalent else None,
        "notes": (
            PREDICTION_EQUIVALENT_BEAM_NOTE,
            ON_AXIS_BEAM_IDENTITY_NOTE,
            ON_AXIS_REFACTOR_NOTE,
            UNPINNED_MAPS_ARE_DIAGNOSTIC_NOTE,
            HOLOGRAPHY_HOLDOUT_BEFORE_MAPS_NOTE,
            CALIBRATION_ADEQUACY_NOT_OPERATOR_NOTE,
            ZERO_QU_IS_ABLATION_NOTE,
        ),
    }


def holography_visibility_holdout_masks(
    observation: HolographyObservation,
    *,
    held_reference_id: int | None = None,
    held_moving_id: int | None = None,
    on_axis_atol_rad: float = 1.0e-6,
    offset_scale: float = 1.0e6,
) -> dict[str, NDArray[np.bool_]]:
    """Sealed mover–reference holdouts, fixed before any map estimate.

    ``later_visits`` still follows MAIN encounter order and is not a
    physical raster-pass split. Scientific visit holdouts use
    ``holography_one_axis_holdout_masks(..., axis="leave_one_visit_out")``.
    """

    moving_ref = moving_reference_row_mask(observation.block, observation.pointing)
    usable = (
        observation.active_row_mask(mask_unsettled=True, moving_reference_only=True)
        & moving_ref
        & four_hand_active_rows(observation)
    )
    offsets, inverse, pointing_valid, _settled, _moving = observation.pointing_state()
    roles = np.full(offsets.shape[:2], "unknown", dtype="U16")
    for index, antenna in enumerate(observation.pointing.antenna_id):
        roles[:, int(antenna)] = observation.pointing.role[:, index]
    n_row = int(observation.block.time_s.size)
    later = np.zeros(n_row, dtype=bool)
    spatial = np.zeros(n_row, dtype=bool)
    held_ref = np.zeros(n_row, dtype=bool)
    held_move = np.zeros(n_row, dtype=bool)
    origin = np.zeros(n_row, dtype=bool)
    first_visit: dict[tuple[int, int, int, int], int] = {}
    cells = []
    for row in np.flatnonzero(usable):
        time_index = int(inverse[row])
        antenna_p = int(observation.block.antenna1[row])
        antenna_q = int(observation.block.antenna2[row])
        if not (pointing_valid[time_index, antenna_p] and pointing_valid[time_index, antenna_q]):
            continue
        moving, reference = _moving_and_reference(antenna_p, antenna_q, roles[time_index])
        offset = np.asarray(offsets[time_index, moving], dtype=np.float64)
        cell = (
            int(moving),
            int(round(float(offset[0]) * float(offset_scale))),
            int(round(float(offset[1]) * float(offset_scale))),
        )
        visit = (int(moving), int(reference), cell[1], cell[2])
        if visit not in first_visit:
            first_visit[visit] = int(row)
        else:
            later[row] = True
        cells.append(
            (int(row), cell, float(np.hypot(offset[0], offset[1])), int(moving), int(reference))
        )
        if float(np.hypot(offset[0], offset[1])) <= float(on_axis_atol_rad):
            origin[row] = True
        if held_reference_id is not None and int(reference) == int(held_reference_id):
            held_ref[row] = True
        if held_moving_id is not None and int(moving) == int(held_moving_id):
            held_move[row] = True
    unique_cells = sorted({item[1] for item in cells})
    hold_cells = set(unique_cells[1::2])
    for row, cell, radius, _moving, _reference in cells:
        if cell in hold_cells and radius > float(on_axis_atol_rad):
            spatial[row] = True
    train = usable & ~later & ~spatial & ~held_ref & ~held_move
    return {
        "usable": usable,
        "train": train,
        "later_visits": usable & later,
        "spatial": usable & spatial,
        "held_reference": usable & held_ref,
        "held_moving": usable & held_move,
        "origin_train": train & origin,
        "holdout": usable & ~train,
    }


@dataclass(frozen=True)
class FrozenInterpolationSupport:
    """Neighbor rule and training support frozen from training offsets only."""

    offsets_by_antenna: Mapping[int, NDArray[np.float64]]
    spacing_by_antenna: Mapping[int, float]
    radius_by_antenna: Mapping[int, float]
    neighbor_k: int = INTERPOLATOR_NEIGHBOR_K
    radius_scale: float = INTERPOLATOR_RADIUS_SCALE
    exact_atol_rad: float = 1.0e-15

    def classify_offset(self, antenna_id: int, offset_lm_rad: ArrayLike) -> dict[str, object]:
        """Label one query as exact, interpolation, extrapolation, or unsupported."""

        query = np.asarray(offset_lm_rad, dtype=np.float64).reshape(2)
        offsets = self.offsets_by_antenna.get(int(antenna_id))
        if offsets is None:
            offsets = np.zeros((0, 2), dtype=np.float64)
        else:
            offsets = np.asarray(offsets, dtype=np.float64).reshape(-1, 2)
        if offsets.shape[0] == 0:
            return {
                "category": "unsupported",
                "n_neighbors": 0,
                "distance_rad": float("nan"),
                "inside_training_bbox": False,
                "n_training": 0,
            }
        distances = np.hypot(offsets[:, 0] - query[0], offsets[:, 1] - query[1])
        nearest = float(np.min(distances))
        radius = float(self.radius_by_antenna.get(int(antenna_id), 0.0))
        n_neighbors = int(np.sum(distances <= radius + 1.0e-15))
        low = np.min(offsets, axis=0)
        high = np.max(offsets, axis=0)
        inside = bool(np.all(query >= low - 1.0e-15) and np.all(query <= high + 1.0e-15))
        if nearest <= float(self.exact_atol_rad):
            category = "exact"
        elif inside and n_neighbors > 0:
            category = "interpolation"
        elif not inside:
            category = "extrapolation"
        else:
            category = "unsupported"
        return {
            "category": category,
            "n_neighbors": n_neighbors,
            "distance_rad": nearest,
            "inside_training_bbox": inside,
            "n_training": int(offsets.shape[0]),
        }

    def classify_offsets(
        self, antenna_id: ArrayLike, offset_lm_rad: ArrayLike
    ) -> dict[str, np.ndarray]:
        """Label many query offsets with the frozen training support."""

        antennas = np.asarray(antenna_id, dtype=np.int32).reshape(-1)
        queries = np.asarray(offset_lm_rad, dtype=np.float64).reshape(-1, 2)
        if antennas.size != queries.shape[0]:
            raise ValueError("antenna_id and offset_lm_rad must have the same length")
        category = np.full(antennas.size, "unsupported", dtype="U16")
        distance = np.full(antennas.size, np.nan, dtype=np.float64)
        n_neighbors = np.zeros(antennas.size, dtype=np.int32)
        inside = np.zeros(antennas.size, dtype=bool)
        for antenna in np.unique(antennas):
            rows = np.flatnonzero(antennas == int(antenna))
            train = self.offsets_by_antenna.get(int(antenna))
            if train is None or np.asarray(train).size == 0:
                continue
            points = np.asarray(train, dtype=np.float64).reshape(-1, 2)
            delta = queries[rows, None, :] - points[None, :, :]
            dist = np.hypot(delta[..., 0], delta[..., 1])
            nearest = np.min(dist, axis=1)
            radius = float(self.radius_by_antenna.get(int(antenna), 0.0))
            neighbors = np.sum(dist <= radius + 1.0e-15, axis=1)
            low = np.min(points, axis=0)
            high = np.max(points, axis=0)
            in_box = np.all(queries[rows] >= low - 1.0e-15, axis=1) & np.all(
                queries[rows] <= high + 1.0e-15, axis=1
            )
            exact = nearest <= float(self.exact_atol_rad)
            interp = (~exact) & in_box & (neighbors > 0)
            extra = (~exact) & (~in_box)
            labels = np.full(rows.size, "unsupported", dtype="U16")
            labels[exact] = "exact"
            labels[interp] = "interpolation"
            labels[extra] = "extrapolation"
            category[rows] = labels
            distance[rows] = nearest
            n_neighbors[rows] = neighbors
            inside[rows] = in_box
        return {
            "category": category,
            "distance_rad": distance,
            "n_neighbors": n_neighbors,
            "inside_training_bbox": inside,
        }


def freeze_interpolation_support(
    offsets_by_antenna: Mapping[int, ArrayLike],
    *,
    neighbor_k: int = INTERPOLATOR_NEIGHBOR_K,
    radius_scale: float = INTERPOLATOR_RADIUS_SCALE,
    exact_atol_rad: float = 1.0e-15,
) -> FrozenInterpolationSupport:
    """Freeze spacing and neighbor radius from training cells only."""

    stored: dict[int, NDArray[np.float64]] = {}
    spacing: dict[int, float] = {}
    radius: dict[int, float] = {}
    for antenna, values in offsets_by_antenna.items():
        points = _unique_offsets(np.asarray(values, dtype=np.float64).reshape(-1, 2))
        stored[int(antenna)] = points
        gap = _median_spacing(points)
        spacing[int(antenna)] = float(gap)
        radius[int(antenna)] = float(radius_scale) * float(gap) if gap > 0.0 else 1.0e-15
    return FrozenInterpolationSupport(
        offsets_by_antenna=stored,
        spacing_by_antenna=spacing,
        radius_by_antenna=radius,
        neighbor_k=int(neighbor_k),
        radius_scale=float(radius_scale),
        exact_atol_rad=float(exact_atol_rad),
    )


def moving_reference_row_geometry(
    observation: HolographyObservation,
    *,
    mask_unsettled: bool = True,
    offset_scale: float = 1.0e6,
    require_all_channels: bool = True,
) -> dict[str, NDArray]:
    """Mover, reference, offset, cell and time for usable moving–reference rows."""

    moving_ref = moving_reference_row_mask(observation.block, observation.pointing)
    active = np.asarray(observation.block.active, dtype=bool)
    four_hand = (
        four_hand_active_rows(observation)
        if require_all_channels
        else np.all(np.any(active, axis=1), axis=1)
    )
    usable = (
        observation.active_row_mask(mask_unsettled=mask_unsettled, moving_reference_only=True)
        & moving_ref
        & four_hand
    )
    offsets, inverse, pointing_valid, _settled, _moving = observation.pointing_state()
    roles = np.full(offsets.shape[:2], "unknown", dtype="U16")
    for index, antenna in enumerate(observation.pointing.antenna_id):
        roles[:, int(antenna)] = observation.pointing.role[:, index]
    n_row = int(observation.block.time_s.size)
    time_all = np.asarray(inverse, dtype=np.int32)
    antenna_p = np.asarray(observation.block.antenna1, dtype=np.int32)
    antenna_q = np.asarray(observation.block.antenna2, dtype=np.int32)
    pointing_ok = pointing_valid[time_all, antenna_p] & pointing_valid[time_all, antenna_q]
    usable = np.asarray(usable, dtype=bool) & pointing_ok
    role_p = roles[time_all, antenna_p]
    moving_is_p = role_p == "moving"
    moving = np.where(moving_is_p, antenna_p, antenna_q).astype(np.int32)
    reference = np.where(moving_is_p, antenna_q, antenna_p).astype(np.int32)
    moving_id = np.full(n_row, -1, dtype=np.int32)
    reference_id = np.full(n_row, -1, dtype=np.int32)
    offset_lm = np.full((n_row, 2), np.nan, dtype=np.float64)
    source_lm = np.full((n_row, 2), np.nan, dtype=np.float64)
    cell_l = np.full(n_row, np.iinfo(np.int64).min, dtype=np.int64)
    cell_m = np.full(n_row, np.iinfo(np.int64).min, dtype=np.int64)
    time_index = np.full(n_row, -1, dtype=np.int32)
    radius = np.full(n_row, np.nan, dtype=np.float64)
    if bool(np.any(usable)):
        moving_id[usable] = moving[usable]
        reference_id[usable] = reference[usable]
        time_index[usable] = time_all[usable]
        offset_lm[usable] = offsets[time_all[usable], moving[usable]]
        source_lm[usable] = source_lm_feed_from_commanded_azelgeo(offset_lm[usable])
        cell_l[usable] = np.rint(offset_lm[usable, 0] * float(offset_scale)).astype(np.int64)
        cell_m[usable] = np.rint(offset_lm[usable, 1] * float(offset_scale)).astype(np.int64)
        radius[usable] = np.hypot(offset_lm[usable, 0], offset_lm[usable, 1])
    return {
        "usable": usable,
        "moving_id": moving_id,
        "reference_id": reference_id,
        "commanded_offset_azelgeo": offset_lm,
        "source_lm_feed": source_lm,
        "offset_lm_rad": offset_lm,
        "offset_lm_meaning": "commanded_azelgeo_legacy",
        "cell_l": cell_l,
        "cell_m": cell_m,
        "time_index": time_index,
        "radius_rad": radius,
        "unique_time_s": observation.block.time_s,
    }


def holography_one_axis_holdout_masks(
    observation: HolographyObservation,
    *,
    axis: str,
    held_reference_id: int | None = None,
    held_moving_id: int | None = None,
    on_axis_atol_rad: float = 1.0e-6,
    offset_scale: float = 1.0e6,
    reserve_outer_fold: bool = True,
    require_all_channels: bool = True,
) -> dict[str, object]:
    """One visibility holdout axis. Axes are never unioned."""

    if axis not in ONE_AXIS_HOLDOUT_AXES:
        raise ValueError(f"unknown one-axis holdout {axis!r}")
    geometry = moving_reference_row_geometry(
        observation,
        offset_scale=offset_scale,
        require_all_channels=require_all_channels,
    )
    usable = np.asarray(geometry["usable"], dtype=bool)
    moving_id = np.asarray(geometry["moving_id"], dtype=np.int32)
    reference_id = np.asarray(geometry["reference_id"], dtype=np.int32)
    cell_l = np.asarray(geometry["cell_l"], dtype=np.int64)
    cell_m = np.asarray(geometry["cell_m"], dtype=np.int64)
    time_index = np.asarray(geometry["time_index"], dtype=np.int32)
    radius = np.asarray(geometry["radius_rad"], dtype=np.float64)
    n_row = int(usable.size)
    later = np.zeros(n_row, dtype=bool)
    spatial = np.zeros(n_row, dtype=bool)
    held_ref = np.zeros(n_row, dtype=bool)
    held_move = np.zeros(n_row, dtype=bool)
    origin = np.zeros(n_row, dtype=bool)
    origin[usable] = radius[usable] <= float(on_axis_atol_rad)
    if held_reference_id is not None:
        held_ref[usable] = reference_id[usable] == int(held_reference_id)
    if held_moving_id is not None:
        held_move[usable] = moving_id[usable] == int(held_moving_id)
    visits = visit_id_per_time(observation)
    visit_of_row = np.full(n_row, -1, dtype=np.int32)
    valid_time = usable & (time_index >= 0)
    visit_of_row[valid_time] = visits[time_index[valid_time]]
    if int(np.unique(visits).size) >= 2:
        later[usable] = visit_of_row[usable] == int(np.max(visits))
    rows = np.flatnonzero(usable)
    if rows.size:
        cell_keys = np.stack([moving_id[rows], cell_l[rows], cell_m[rows]], axis=1)
        unique_cells = np.unique(cell_keys, axis=0)
        hold_cells = unique_cells[1::2]
        if hold_cells.size:
            cell_view = (
                np.ascontiguousarray(cell_keys)
                .view(np.dtype((np.void, cell_keys.dtype.itemsize * cell_keys.shape[1])))
                .reshape(-1)
            )
            hold_view = (
                np.ascontiguousarray(hold_cells)
                .view(np.dtype((np.void, hold_cells.dtype.itemsize * hold_cells.shape[1])))
                .reshape(-1)
            )
            spatial[rows] = np.isin(cell_view, hold_view) & (radius[rows] > float(on_axis_atol_rad))
    if axis == "leave_one_reference_out":
        if held_reference_id is None:
            raise ValueError("leave-one-reference-out needs held_reference_id")
        holdout = usable & held_ref
        train = usable & ~held_ref
    elif axis == "leave_one_visit_out":
        holdout = usable & later
        train = usable & ~later
    elif axis == "spatial_checkerboard":
        holdout = usable & spatial
        train = usable & ~spatial
    else:
        if held_moving_id is None:
            raise ValueError("leave-one-mover-out needs held_moving_id")
        holdout = usable & held_move
        train = usable & ~held_move
    reserved = np.zeros(n_row, dtype=bool)
    if reserve_outer_fold and bool(np.any(holdout)):
        reserved = _reserved_outer_fold_mask(
            axis,
            holdout=holdout,
            moving_id=moving_id,
            cell_l=cell_l,
            cell_m=cell_m,
            time_index=time_index,
        )
    scored = holdout & ~reserved
    return {
        "usable": usable,
        "train": train,
        "holdout": holdout,
        "reserved": reserved,
        "scored": scored,
        "later_visits": usable & later,
        "spatial": usable & spatial,
        "held_reference": usable & held_ref,
        "held_moving": usable & held_move,
        "origin_train": train & origin,
    }


def _reserved_outer_fold_mask(
    axis: str,
    *,
    holdout: NDArray[np.bool_],
    moving_id: NDArray[np.int32],
    cell_l: NDArray[np.int64],
    cell_m: NDArray[np.int64],
    time_index: NDArray[np.int32],
) -> NDArray[np.bool_]:
    reserved = np.zeros(holdout.shape, dtype=bool)
    rows = np.flatnonzero(holdout)
    if rows.size == 0:
        return reserved
    if axis == "spatial_checkerboard":
        cells = sorted({(int(moving_id[row]), int(cell_l[row]), int(cell_m[row])) for row in rows})
        held = set(cells[3::4])
        for row in rows:
            if (int(moving_id[row]), int(cell_l[row]), int(cell_m[row])) in held:
                reserved[row] = True
        return reserved
    times = sorted({int(time_index[row]) for row in rows})
    if not times:
        return reserved
    start = max(1, int(np.ceil(0.75 * len(times))))
    late = set(times[start:])
    for row in rows:
        if int(time_index[row]) in late:
            reserved[row] = True
    return reserved


def thin_training_rows(
    observation: HolographyObservation,
    row_mask: ArrayLike,
    *,
    max_per_cell: int = 4,
    offset_scale: float = 1.0e6,
    require_all_channels: bool = True,
) -> NDArray[np.bool_]:
    """Keep the earliest ``max_per_cell`` rows per mover–reference cell."""

    mask = np.asarray(row_mask, dtype=bool).reshape(-1)
    geometry = moving_reference_row_geometry(
        observation,
        offset_scale=offset_scale,
        require_all_channels=require_all_channels,
    )
    kept = np.zeros(mask.shape, dtype=bool)
    order = np.flatnonzero(mask & geometry["usable"])
    if order.size == 0:
        return kept
    times = np.asarray(geometry["unique_time_s"], dtype=np.float64)
    moving_id = np.asarray(geometry["moving_id"], dtype=np.int32)
    reference_id = np.asarray(geometry["reference_id"], dtype=np.int32)
    cell_l = np.asarray(geometry["cell_l"], dtype=np.int64)
    cell_m = np.asarray(geometry["cell_m"], dtype=np.int64)
    order = order[
        np.lexsort(
            (
                times[order],
                cell_m[order],
                cell_l[order],
                reference_id[order],
                moving_id[order],
            )
        )
    ]
    keys = np.stack(
        [moving_id[order], reference_id[order], cell_l[order], cell_m[order]],
        axis=1,
    )
    change = np.ones(order.size, dtype=bool)
    change[1:] = np.any(keys[1:] != keys[:-1], axis=1)
    starts = np.flatnonzero(change)
    rank = np.arange(order.size) - np.repeat(starts, np.diff(np.append(starts, order.size)))
    kept[order[rank < int(max_per_cell)]] = True
    return kept


def freeze_interpolation_support_from_rows(
    observation: HolographyObservation,
    row_mask: ArrayLike,
    *,
    neighbor_k: int = INTERPOLATOR_NEIGHBOR_K,
    radius_scale: float = INTERPOLATOR_RADIUS_SCALE,
    offset_scale: float = 1.0e6,
) -> FrozenInterpolationSupport:
    """Freeze the interpolator from training rows, never from holdout cells."""

    mask = np.asarray(row_mask, dtype=bool).reshape(-1)
    geometry = moving_reference_row_geometry(observation, offset_scale=offset_scale)
    usable = mask & np.asarray(geometry["usable"], dtype=bool)
    rows = np.flatnonzero(usable)
    packed: dict[int, NDArray[np.float64]] = {}
    if rows.size:
        moving_id = np.asarray(geometry["moving_id"], dtype=np.int32)
        cell_l = np.asarray(geometry["cell_l"], dtype=np.int64)
        cell_m = np.asarray(geometry["cell_m"], dtype=np.int64)
        offsets = np.asarray(geometry["offset_lm_rad"], dtype=np.float64)
        keys = np.stack([moving_id[rows], cell_l[rows], cell_m[rows]], axis=1)
        _, first = np.unique(keys, axis=0, return_index=True)
        chosen = rows[first]
        for antenna in np.unique(moving_id[chosen]):
            members = chosen[moving_id[chosen] == int(antenna)]
            packed[int(antenna)] = offsets[members]
    return freeze_interpolation_support(packed, neighbor_k=neighbor_k, radius_scale=radius_scale)


def classify_holdout_interpolation_support(
    support: FrozenInterpolationSupport,
    geometry: Mapping[str, NDArray],
    row_mask: ArrayLike,
    *,
    representation: str = "per_antenna",
    array_average_id: int = ARRAY_AVERAGE_ANTENNA_ID,
) -> dict[str, object]:
    """Label holdout rows with a training-only support rule."""

    mask = np.asarray(row_mask, dtype=bool).reshape(-1)
    n_row = int(mask.size)
    category = np.full(n_row, "ignored", dtype="U16")
    distance = np.full(n_row, np.nan, dtype=np.float64)
    n_neighbors = np.zeros(n_row, dtype=np.int32)
    inside = np.zeros(n_row, dtype=bool)
    moving_id = np.asarray(geometry["moving_id"], dtype=np.int32)
    offsets = np.asarray(geometry["offset_lm_rad"], dtype=np.float64)
    rows = np.flatnonzero(mask)
    lookup = array_average_id if representation == "array_average" else None
    antennas = (
        np.full(rows.shape, int(lookup), dtype=np.int32) if lookup is not None else moving_id[rows]
    )
    labeled = support.classify_offsets(antennas, offsets[rows])
    category[rows] = labeled["category"]
    distance[rows] = labeled["distance_rad"]
    n_neighbors[rows] = labeled["n_neighbors"]
    inside[rows] = labeled["inside_training_bbox"]
    n_hold = int(np.sum(mask))
    exact = mask & (category == "exact")
    interpolation = mask & (category == "interpolation")
    supported = exact | interpolation
    return {
        "category": category,
        "distance_rad": distance,
        "n_neighbors": n_neighbors,
        "inside_training_bbox": inside,
        "exact": exact,
        "interpolation": interpolation,
        "extrapolation": mask & (category == "extrapolation"),
        "unsupported": mask & (category == "unsupported"),
        "supported": supported,
        "n_holdout": n_hold,
        "n_supported": int(np.sum(supported)),
        "support_fraction": float(np.sum(supported) / n_hold) if n_hold else float("nan"),
        "n_exact": int(np.sum(exact)),
        "n_interpolation": int(np.sum(interpolation)),
        "n_extrapolation": int(np.sum(mask & (category == "extrapolation"))),
        "n_unsupported": int(np.sum(mask & (category == "unsupported"))),
        "median_distance_rad": (float(np.nanmedian(distance[mask])) if n_hold else float("nan")),
        "representation": representation,
        "notes": (INTERPOLATOR_FROZEN_ON_TRAIN_NOTE,),
    }


def _hand_residual(
    measured: NDArray[np.complex128],
    predicted: NDArray[np.complex128],
    intensity: NDArray[np.float64],
    row_mask: NDArray[np.bool_],
    row: int,
    col: int,
) -> NDArray[np.complex128]:
    vis = np.asarray(measured, dtype=np.complex128)
    pred = np.asarray(predicted, dtype=np.complex128)
    if vis.ndim == 4:
        vis = vis[row_mask, 0, row, col]
        pred = pred[row_mask, 0, row, col]
    else:
        vis = vis[row_mask, row, col]
        pred = pred[row_mask, row, col]
    scale = np.maximum(
        np.abs(np.asarray(intensity, dtype=np.float64).reshape(-1)[row_mask]), 1.0e-3
    )
    residual = (vis - pred) / scale
    residual = np.where(np.isfinite(residual) & np.isfinite(pred), residual, np.nan + 1j * np.nan)
    return residual


def _clustered_abs_improvement(
    full: NDArray[np.complex128],
    diagonal: NDArray[np.complex128],
    group_ids: NDArray,
) -> dict[str, object]:
    if full.size == 0 or diagonal.size == 0 or full.size != diagonal.size:
        return {"n_clusters": 0, "n_improved": 0, "mean": float("nan"), "std": float("nan")}
    labels = np.asarray(group_ids).reshape(-1)
    if labels.size != full.size:
        return {"n_clusters": 0, "n_improved": 0, "mean": float("nan"), "std": float("nan")}
    improvements = []
    for label in np.unique(labels):
        keep = labels == label
        if not bool(np.any(keep)):
            continue
        improvements.append(
            float(np.median(np.abs(diagonal[keep])) - np.median(np.abs(full[keep])))
        )
    packed = np.asarray(improvements, dtype=np.float64)
    return {
        "n_clusters": int(packed.size),
        "n_improved": int(np.sum(packed > 0.0)),
        "mean": float(np.mean(packed)) if packed.size else float("nan"),
        "std": float(np.std(packed, ddof=1)) if packed.size > 1 else 0.0,
    }


def paired_full_versus_diagonal_scores(
    measured: ArrayLike,
    predicted_full: ArrayLike,
    predicted_diagonal: ArrayLike,
    *,
    stokes_i: ArrayLike,
    row_mask: ArrayLike,
    dwell_ids: ArrayLike | None = None,
    mover_ids: ArrayLike | None = None,
    reference_ids: ArrayLike | None = None,
    cell_ids: ArrayLike | None = None,
) -> dict[str, object]:
    """Paired full-Jones vs diagonal scores on identical rows."""

    mask = np.asarray(row_mask, dtype=bool).reshape(-1)
    intensity = np.asarray(stokes_i, dtype=np.float64).reshape(-1)
    vis = np.asarray(measured, dtype=np.complex128)
    pred_full = np.asarray(predicted_full, dtype=np.complex128)
    pred_diag = np.asarray(predicted_diagonal, dtype=np.complex128)
    finite = mask.copy()
    if pred_full.ndim == 4:
        finite &= np.isfinite(pred_full[:, 0, 0, 0]) & np.isfinite(pred_diag[:, 0, 0, 0])
    else:
        finite &= np.isfinite(pred_full[:, 0, 0]) & np.isfinite(pred_diag[:, 0, 0])
    hands = {
        "rr": (0, 0),
        "ll": (1, 1),
        "rl": (0, 1),
        "lr": (1, 0),
    }
    aligned = {}
    for name, (row, col) in hands.items():
        aligned[name] = (
            _hand_residual(vis, pred_full, intensity, finite, row, col),
            _hand_residual(vis, pred_diag, intensity, finite, row, col),
        )
    keep = np.ones(int(np.sum(finite)), dtype=bool)
    for full, diag in aligned.values():
        keep &= np.isfinite(full) & np.isfinite(diag)
    residuals = {name: (full[keep], diag[keep]) for name, (full, diag) in aligned.items()}
    jy = {}
    for name, (row, col) in hands.items():
        if vis.ndim == 4:
            meas = vis[finite, 0, row, col]
            full_v = pred_full[finite, 0, row, col]
            diag_v = pred_diag[finite, 0, row, col]
        else:
            meas = vis[finite, row, col]
            full_v = pred_full[finite, row, col]
            diag_v = pred_diag[finite, row, col]
        jy[name] = (np.abs(meas - full_v)[keep], np.abs(meas - diag_v)[keep])
    report: dict[str, object] = {
        "n": int(np.sum(mask)),
        "n_meaning": "rows in the supplied mask",
        "n_finite": int(np.sum(keep)),
        "identical_rows": True,
    }
    for name, (full, diag) in residuals.items():
        report[f"full_median_abs_{name}_over_i"] = (
            float(np.median(np.abs(full))) if full.size else float("nan")
        )
        report[f"diagonal_median_abs_{name}_over_i"] = (
            float(np.median(np.abs(diag))) if diag.size else float("nan")
        )
        report[f"full_median_abs_{name}_jy"] = (
            float(np.median(jy[name][0])) if jy[name][0].size else float("nan")
        )
        report[f"diagonal_median_abs_{name}_jy"] = (
            float(np.median(jy[name][1])) if jy[name][1].size else float("nan")
        )
        report[name] = {
            "full": coherent_residual_report(full),
            "diagonal": coherent_residual_report(diag),
        }
    rl_full, rl_diag = residuals["rl"]
    lr_full, lr_diag = residuals["lr"]
    rr_full, rr_diag = residuals["rr"]
    ll_full, ll_diag = residuals["ll"]
    rl_improved = bool(
        rl_full.size
        and rl_diag.size
        and float(np.median(np.abs(rl_full)))
        < 0.95 * max(float(np.median(np.abs(rl_diag))), 1.0e-12)
    )
    lr_improved = bool(
        lr_full.size
        and lr_diag.size
        and float(np.median(np.abs(lr_full)))
        < 0.95 * max(float(np.median(np.abs(lr_diag))), 1.0e-12)
    )
    rr_ok = not (
        rr_full.size
        and rr_diag.size
        and float(np.median(np.abs(rr_full)))
        > 1.15 * max(float(np.median(np.abs(rr_diag))), 1.0e-12)
    )
    ll_ok = not (
        ll_full.size
        and ll_diag.size
        and float(np.median(np.abs(ll_full)))
        > 1.15 * max(float(np.median(np.abs(ll_diag))), 1.0e-12)
    )
    clusters = {
        "dwell": None if dwell_ids is None else np.asarray(dwell_ids).reshape(-1)[finite][keep],
        "mover": None if mover_ids is None else np.asarray(mover_ids).reshape(-1)[finite][keep],
        "reference": None
        if reference_ids is None
        else np.asarray(reference_ids).reshape(-1)[finite][keep],
        "spatial_cell": None
        if cell_ids is None
        else np.asarray(cell_ids).reshape(-1)[finite][keep],
    }
    clustered = {}
    for name, labels in clusters.items():
        if labels is None:
            continue
        clustered[name] = {
            "rl": _clustered_abs_improvement(rl_full, rl_diag, labels),
            "lr": _clustered_abs_improvement(lr_full, lr_diag, labels),
        }
    report.update(
        {
            "rl_improved": rl_improved,
            "lr_improved": lr_improved,
            "rr_ll_regression": bool(not rr_ok or not ll_ok),
            "rr_ll_not_regressed": bool(rr_ok and ll_ok),
            "clustered_improvement": clustered,
            "bootstrap_improvement_over_i": {
                "rl": _bootstrap_median_improvement(rl_full, rl_diag),
                "lr": _bootstrap_median_improvement(lr_full, lr_diag),
                "rr": _bootstrap_median_improvement(rr_full, rr_diag),
                "ll": _bootstrap_median_improvement(ll_full, ll_diag),
            },
        }
    )
    return report


def _bootstrap_median_improvement(
    full: NDArray[np.complex128],
    diagonal: NDArray[np.complex128],
    *,
    n_boot: int = 400,
    seed: int = 0,
) -> dict[str, float]:
    full_abs = np.abs(np.asarray(full))
    diag_abs = np.abs(np.asarray(diagonal))
    n = int(full_abs.size)
    if n == 0:
        return {
            "n_boot": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "fraction_positive": float("nan"),
        }
    rng = np.random.default_rng(int(seed))
    index = rng.integers(0, n, size=(int(n_boot), n))
    samples = np.median(diag_abs[index], axis=1) - np.median(full_abs[index], axis=1)
    return {
        "n_boot": int(n_boot),
        "mean": float(np.mean(samples)),
        "std": float(np.std(samples, ddof=1)) if int(n_boot) > 1 else 0.0,
        "fraction_positive": float(np.mean(samples > 0.0)),
    }


def score_full_versus_diagonal_regions(
    measured: ArrayLike,
    predicted_full: ArrayLike,
    predicted_diagonal: ArrayLike,
    *,
    stokes_i: ArrayLike,
    row_mask: ArrayLike,
    voltage: ArrayLike,
    dwell_ids: ArrayLike | None = None,
    mover_ids: ArrayLike | None = None,
    reference_ids: ArrayLike | None = None,
    cell_ids: ArrayLike | None = None,
) -> dict[str, object]:
    """Paired full-vs-diagonal scores in the fixed voltage-response bins."""

    from sl1mjax.holography_alignment import voltage_response_region_masks

    scored = np.asarray(row_mask, dtype=bool).reshape(-1)
    regions = {"all": scored}
    for name, region in voltage_response_region_masks(voltage).items():
        regions[name] = region & scored
    return {
        name: paired_full_versus_diagonal_scores(
            measured,
            predicted_full,
            predicted_diagonal,
            stokes_i=stokes_i,
            row_mask=mask,
            dwell_ids=dwell_ids,
            mover_ids=mover_ids,
            reference_ids=reference_ids,
            cell_ids=cell_ids,
        )
        for name, mask in regions.items()
    }


def classify_loro_full_versus_diagonal(
    *,
    rl_improved: bool | None,
    lr_improved: bool | None,
    rr_ll_regression: bool | None,
    n_finite: int,
    min_exact_cells: int = 100,
) -> dict[str, object]:
    """Fair RL/LR test on source-normalized LORO exact cells."""

    if int(n_finite) < int(min_exact_cells):
        outcome = "inconclusive"
        status = "inconclusive"
        blocking = False
    elif rr_ll_regression:
        outcome = "rr_ll_regressed"
        status = "fail"
        blocking = True
    elif rl_improved and lr_improved:
        outcome = "full_jones_improves_crosshands"
        status = "pass"
        blocking = False
    else:
        outcome = "no_crosshand_improvement"
        status = "fail"
        blocking = True
    return {
        "gate": "loro_full_versus_diagonal",
        "status": status,
        "blocking": blocking,
        "outcome": outcome,
        "n_finite": int(n_finite),
        "min_exact_cells": int(min_exact_cells),
        "rl_improved": rl_improved,
        "lr_improved": lr_improved,
        "rr_ll_regression": rr_ll_regression,
        "full_jones_blocked": True,
        "notes": (
            LORO_HOLDOUT_NOTE,
            "Unsupported off-diagonals are zeroed and remain in the total score.",
            "Full Jones stays unfrozen until this fair RL/LR test passes.",
        ),
    }


def combine_loro_full_versus_diagonal(
    per_reference: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """Do not pool references. Every well-supported LORO must not regress RR/LL."""

    decided = [
        dict(report)
        for report in per_reference.values()
        if report.get("status") in {"pass", "fail"}
    ]
    passed = [report for report in decided if report.get("status") == "pass"]
    failed = [report for report in decided if report.get("status") == "fail"]
    if failed:
        status = "fail"
        outcome = "reference_loro_failed"
        blocking = True
    elif passed:
        status = "pass"
        outcome = "all_supported_references_improve_crosshands"
        blocking = False
    else:
        status = "inconclusive"
        outcome = "no_supported_reference_loro"
        blocking = False
    return {
        "gate": "loro_full_versus_diagonal",
        "status": status,
        "blocking": blocking,
        "outcome": outcome,
        "n_references": len(per_reference),
        "n_decided": len(decided),
        "n_passed": len(passed),
        "n_failed": len(failed),
        "pooled_score": None,
        "full_jones_blocked": True,
        "full_jones_frozen": False,
        "spw5_closed": status != "pass",
        "most_important_next_artifact": "cassbeam_diagonal_low_order_correction",
        "per_reference": {name: dict(report) for name, report in per_reference.items()},
        "notes": (LORO_HOLDOUT_NOTE, ONE_AXIS_HOLDOUT_NOTE),
    }


def classify_one_axis_visibility_holdout(
    *,
    axis: str,
    representation: str,
    support_fraction: float,
    n_holdout: int,
    n_supported: int,
    n_finite: int,
    rl_improved: bool | None,
    lr_improved: bool | None,
    rr_ll_regression: bool | None,
    supported_rl_improved: bool | None = None,
    supported_lr_improved: bool | None = None,
    supported_rr_ll_regression: bool | None = None,
    supported_n_finite: int | None = None,
) -> dict[str, object]:
    """Classify one holdout axis. Never pool axes."""

    if axis not in ONE_AXIS_HOLDOUT_AXES:
        raise ValueError(f"unknown one-axis holdout {axis!r}")
    notes = {
        "leave_one_reference_out": LORO_HOLDOUT_NOTE,
        "leave_one_visit_out": LOVO_HOLDOUT_NOTE,
        "spatial_checkerboard": SPATIAL_CHECKERBOARD_NOTE,
        "leave_one_mover_out": LOMO_HOLDOUT_NOTE,
    }
    if axis == "leave_one_mover_out" and representation == "per_antenna" and int(n_finite) == 0:
        outcome = "expected_unsupported"
        status = "pass"
        blocking = False
    elif n_holdout <= 0:
        outcome = "inconclusive"
        status = "warn"
        blocking = False
    elif (
        representation == "per_antenna"
        and np.isfinite(support_fraction)
        and float(support_fraction) < 0.2
    ):
        outcome = "inconclusive_support_starved"
        status = "warn"
        blocking = False
    elif supported_rr_ll_regression is True or (
        supported_rr_ll_regression is None and rr_ll_regression is True
    ):
        outcome = "rr_ll_regressed"
        status = "fail"
        blocking = True
    elif (
        (supported_rl_improved if supported_rl_improved is not None else rl_improved)
        and (supported_lr_improved if supported_lr_improved is not None else lr_improved)
        and not (
            supported_rr_ll_regression
            if supported_rr_ll_regression is not None
            else rr_ll_regression
        )
    ):
        outcome = "full_jones_improves_supported_holdout"
        status = "pass"
        blocking = False
    elif n_finite == 0 or (supported_n_finite is not None and int(supported_n_finite) == 0):
        outcome = "inconclusive"
        status = "warn"
        blocking = False
    else:
        outcome = "no_crosshand_improvement"
        status = "warn"
        blocking = False
    return {
        "axis": axis,
        "representation": representation,
        "status": status,
        "blocking": blocking,
        "outcome": outcome,
        "support_fraction": float(support_fraction) if np.isfinite(support_fraction) else None,
        "n_holdout": int(n_holdout),
        "n_supported": int(n_supported),
        "n_finite": int(n_finite),
        "rl_improved": rl_improved,
        "lr_improved": lr_improved,
        "rr_ll_regression": rr_ll_regression,
        "supported_rl_improved": supported_rl_improved,
        "supported_lr_improved": supported_lr_improved,
        "supported_rr_ll_regression": supported_rr_ll_regression,
        "do_not_combine_axes": True,
        "notes": (
            notes[axis],
            ONE_AXIS_HOLDOUT_NOTE,
            INTERPOLATOR_FROZEN_ON_TRAIN_NOTE,
            DEVELOPMENT_SET_NOTE,
        ),
    }


def combine_one_axis_results(axes: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    """Keep per-axis reports separate. There is no pooled score."""

    return {
        "pooled_score": None,
        "do_not_combine": True,
        "axes": {name: dict(report) for name, report in axes.items()},
        "notes": (ONE_AXIS_HOLDOUT_NOTE, DEVELOPMENT_SET_NOTE),
    }


def estimate_training_origin_jones(
    artifact: HolographyFullJonesArtifact,
    *,
    on_axis_atol_rad: float = 1.0e-6,
) -> dict[int, dict[str, object]]:
    """Inverse-variance :math:`A_m=E_m(0)` from training-only origin samples."""

    groups: dict[int, list[HolographyFullJonesSample]] = {}
    for sample in artifact.samples:
        if not sample.copolar_valid:
            continue
        groups.setdefault(int(sample.moving_antenna_id), []).append(sample)
    origins: dict[int, dict[str, object]] = {}
    for antenna, members in groups.items():
        radii = [
            float(np.hypot(float(sample.offset_lm_rad[0]), float(sample.offset_lm_rad[1])))
            for sample in members
        ]
        selected = [
            sample
            for sample, radius in zip(members, radii, strict=True)
            if radius <= float(on_axis_atol_rad)
        ]
        used_nearest = False
        if not selected:
            selected = [members[int(np.argmin(radii))]]
            used_nearest = True
        planes = np.stack(
            [np.asarray(sample.jones, dtype=np.complex128) for sample in selected], axis=0
        )
        weights = np.stack(
            [
                1.0 / np.maximum(np.asarray(sample.sigma, dtype=np.float64) ** 2, 1.0e-12)
                for sample in selected
            ],
            axis=0,
        )
        total = np.sum(weights, axis=0)
        mean = np.sum(weights * planes, axis=0) / np.maximum(total, 1.0e-30)
        scatter = np.sqrt(
            np.sum(weights * np.abs(planes - mean) ** 2, axis=0) / np.maximum(total, 1.0e-30)
        )
        origins[int(antenna)] = {
            "jones": np.asarray(mean, dtype=np.complex128),
            "sigma": np.asarray(scatter, dtype=np.float64),
            "n": len(selected),
            "used_nearest": used_nearest,
        }
    return origins


def refactor_on_axis_gauge(
    artifact: HolographyFullJonesArtifact,
    residual_jones: Mapping[int, ArrayLike],
    *,
    origin: Mapping[int, Mapping[str, object]] | None = None,
    on_axis_atol_rad: float = 1.0e-6,
) -> tuple[
    HolographyFullJonesArtifact, dict[int, NDArray[np.complex128]], dict[int, dict[str, object]]
]:
    """Apply :math:`A_m=E_m(0)`, :math:`R'_m=R_m A_m`, :math:`E'_m=A_m^{-1}E_m`."""

    if artifact.frozen:
        raise ValueError("empirical holography full Jones is not frozen")
    estimated = origin or estimate_training_origin_jones(
        artifact, on_axis_atol_rad=on_axis_atol_rad
    )
    planes = np.array(artifact.jones, copy=True)
    samples = list(artifact.samples)
    updated = {
        int(ant): np.array(np.asarray(plane, dtype=np.complex128), copy=True)
        for ant, plane in residual_jones.items()
    }
    n_pin = 0
    for antenna, report in estimated.items():
        axis = np.asarray(report["jones"], dtype=np.complex128)
        if not bool(np.all(np.isfinite(axis))) or abs(complex(np.linalg.det(axis))) < 1.0e-12:
            continue
        inverse = invert_jones(axis)
        current = updated.get(int(antenna), np.eye(2, dtype=np.complex128))
        updated[int(antenna)] = np.asarray(current, dtype=np.complex128) @ axis
        for index, sample in enumerate(samples):
            if int(sample.moving_antenna_id) != int(antenna):
                continue
            pinned = inverse @ planes[index]
            planes[index] = pinned
            samples[index] = replace(sample, jones=pinned)
        n_pin += 1
    pinned = replace(
        artifact,
        samples=tuple(samples),
        jones=planes,
        notes=artifact.notes
        + (
            ON_AXIS_REFACTOR_NOTE,
            ON_AXIS_BEAM_IDENTITY_NOTE,
            f"refactored {n_pin} moving antennas to E_m(0)=I",
            UNPINNED_MAPS_ARE_DIAGNOSTIC_NOTE,
        ),
    )
    return pinned, updated, dict(estimated)


def predict_moving_reference_from_beam(
    observation: HolographyObservation,
    *,
    residual_jones: Mapping[int, ArrayLike],
    artifact: HolographyFullJonesArtifact,
    row_mask: ArrayLike,
    parallactic_angle_rad: ArrayLike | None = None,
    diagonal_only: bool = False,
) -> NDArray[np.complex128]:
    """Predict :math:`V=R_m E_m(s) S R_r^H` on selected mover–reference rows."""

    import jax.numpy as jnp

    mask = np.asarray(row_mask, dtype=bool).reshape(-1)
    source = np.asarray(_source_coherency(observation), dtype=np.complex128)
    packed = pack_coherency(
        observation.block.visibility, observation.block.correlations, _RECEPTORS
    )
    predicted = np.full(packed.shape, np.nan + 1j * np.nan, dtype=np.complex128)
    offsets, inverse, pointing_valid, _settled, _moving = observation.pointing_state()
    roles = np.full(offsets.shape[:2], "unknown", dtype="U16")
    for index, antenna in enumerate(observation.pointing.antenna_id):
        roles[:, int(antenna)] = observation.pointing.role[:, index]
    rows = np.flatnonzero(mask)
    if rows.size == 0:
        return predicted
    time_index = np.asarray(inverse, dtype=np.int32)[rows]
    antenna_p = np.asarray(observation.block.antenna1, dtype=np.int32)[rows]
    antenna_q = np.asarray(observation.block.antenna2, dtype=np.int32)[rows]
    keep = pointing_valid[time_index, antenna_p] & pointing_valid[time_index, antenna_q]
    if not bool(np.any(keep)):
        return predicted
    rows = rows[keep]
    time_index = time_index[keep]
    antenna_p = antenna_p[keep]
    antenna_q = antenna_q[keep]
    role_p = roles[time_index, antenna_p]
    moving = np.where(role_p == "moving", antenna_p, antenna_q).astype(np.int32)
    reference = np.where(role_p == "moving", antenna_q, antenna_p).astype(np.int32)
    offset = offsets[time_index, moving]
    n_chan = int(observation.block.frequency_hz.size)
    movers = np.repeat(moving, n_chan)
    query = np.repeat(offset, n_chan, axis=0)
    freqs = np.broadcast_to(observation.block.frequency_hz, (moving.size, n_chan)).reshape(-1)
    beams, ok, leak = interpolate_holography_full_jones_batch(
        artifact,
        query,
        moving_antenna_id=movers,
        frequency_hz=freqs,
    )
    beams = beams.reshape(moving.size, n_chan, 2, 2)
    ok = ok.reshape(moving.size, n_chan)
    leak = leak.reshape(moving.size, n_chan)
    if diagonal_only:
        beams[..., 0, 1] = 0.0
        beams[..., 1, 0] = 0.0
    else:
        off = ~leak
        beams[off, 0, 1] = 0.0
        beams[off, 1, 0] = 0.0
    r_m = _antenna_jones_planes(residual_jones, moving)
    r_r = _antenna_jones_planes(residual_jones, reference)
    chi = (
        None
        if parallactic_angle_rad is None
        else np.asarray(parallactic_angle_rad, dtype=np.float64)
    )
    if chi is not None:
        r_m = np.asarray(
            _sky_frame_residual_jax(jnp.asarray(r_m), jnp.asarray(chi[time_index, moving]))
        )
        r_r = np.asarray(
            _sky_frame_residual_jax(jnp.asarray(r_r), jnp.asarray(chi[time_index, reference]))
        )
    source_rows = source[rows]
    if source_rows.ndim == 3:
        source_rows = source_rows[:, None, :, :]
    vis = np.asarray(
        _predict_vis_jax(
            jnp.asarray(r_m),
            jnp.asarray(beams),
            jnp.asarray(source_rows),
            jnp.asarray(r_r),
            jnp.asarray(moving == antenna_p),
        )
    )
    predicted[rows] = np.where(ok[..., None, None], vis, predicted[rows])
    return predicted


def visibility_correlation_holdout_report(
    measured: ArrayLike,
    predicted: ArrayLike,
    *,
    stokes_i: ArrayLike,
    row_mask: ArrayLike,
    cluster_ids: ArrayLike | None = None,
    antenna_groups: ArrayLike | None = None,
    baseline_groups: ArrayLike | None = None,
) -> dict[str, object]:
    """Clustered RR/LL/RL/LR residuals against a sealed visibility holdout."""

    mask = np.asarray(row_mask, dtype=bool).reshape(-1)
    vis = np.asarray(measured, dtype=np.complex128)
    pred = np.asarray(predicted, dtype=np.complex128)
    if vis.ndim == 4:
        vis = vis[mask, 0]
        pred = pred[mask, 0]
    else:
        vis = vis[mask]
        pred = pred[mask]
    intensity = np.asarray(stokes_i, dtype=np.float64).reshape(-1)[mask]
    scale = np.maximum(np.abs(intensity), 1.0e-3)
    residual = (vis - pred) / scale[:, None, None]
    labels = None if cluster_ids is None else np.asarray(cluster_ids).reshape(-1)[mask]
    hands = {
        "rr": residual[:, 0, 0],
        "ll": residual[:, 1, 1],
        "rl": residual[:, 0, 1],
        "lr": residual[:, 1, 0],
    }
    report = {"n": int(np.sum(mask))}
    for name, values in hands.items():
        report[name] = coherent_residual_report(values, group_ids=labels)
        report[f"median_abs_{name}_over_i"] = report[name].get("median_abs")
    if antenna_groups is not None:
        report["rl_by_antenna"] = coherent_residual_report(
            residual[:, 0, 1], group_ids=np.asarray(antenna_groups).reshape(-1)[mask]
        )
    if baseline_groups is not None:
        report["rl_by_baseline"] = coherent_residual_report(
            residual[:, 0, 1], group_ids=np.asarray(baseline_groups).reshape(-1)[mask]
        )
    return report
