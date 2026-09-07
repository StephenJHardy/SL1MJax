"""Direction-independent reference/visit alignment of holography maps.

Recovered cells currently mix the beam with leftover factors

    Ê_{mrv}(s) ≈ A_{mv} E_m(s) B_{rv}^H.

``A`` is estimated from origin moving–reference dwells and ``B`` from
HOLORASTER reference–reference rows. Non-origin moving–reference rows
never enter those fits. Full Jones stays blocked until aligned exact-cell
transfer reaches the established copolar floor.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sl1mjax.holography import HolographyObservation, Memo195LowerCRaster
from sl1mjax.holography_diagonal import (
    _raster_label,
    _source_coherency,
    four_hand_active_rows,
    four_hand_sample_weight,
)
from sl1mjax.holography_full_jones import (
    HOLOGRAPHY_FULL_JONES_SCHEMA_VERSION,
    HolographyFullJonesArtifact,
    HolographyFullJonesSample,
    _antenna_jones_planes,
    _predict_vis_jax,
    _recover_row_jones_jax,
    _sky_frame_residual_jax,
    interpolate_holography_full_jones_batch,
)
from sl1mjax.holography_pointing_maps import visit_id_per_time  # noqa: F401
from sl1mjax.polarization import Receptor, invert_jones, pack_coherency

REFERENCE_VISIT_ALIGNED_COPOLAR_TRANSFER = "reference_visit_aligned_copolar_transfer"
UNCALIBRATED_VISIT_TRANSFER = "uncalibrated_visit_transfer"
BEAM_REPEATABILITY = "beam_repeatability"
COPOLAR_FLOOR = 0.05
VOLTAGE_MAINLOBE = 0.5
VOLTAGE_MID = 0.2
VOLTAGE_RESPONSE_REGIONS = (
    ("main_lobe", VOLTAGE_MAINLOBE, np.inf),
    ("mid", VOLTAGE_MID, VOLTAGE_MAINLOBE),
    ("outer_diagnostic", 0.0, VOLTAGE_MID),
)
ALIGNMENT_NOTE = (
    "Leftover reference/visit factors are direction-independent if origin "
    "and reference-reference alignment reduce source-normalized exact-cell "
    "RR/LL in the voltage>=0.5 main lobe to the established few-percent "
    "floor. Residuals use source-model I, not measured beam-attenuated "
    "RR/LL. Non-origin moving-reference rows are held out of the fit."
)
UNCALIBRATED_VISIT_NOTE = (
    "Uncalibrated visit transfer holds out the entire visit, including "
    "its origin. It measures whether A and B transfer globally."
)
BEAM_REPEATABILITY_NOTE = (
    "Holography beam repeatability allows a visit's on-axis calibration "
    "samples, then holds out its non-origin directions. That is the test "
    "of whether the beam shape repeats."
)
FULL_JONES_BLOCKED_UNTIL_ALIGNMENT_NOTE = (
    "The held-reference copolar test is no longer the Full Jones blocker. "
    "The LORO full-versus-diagonal test found no transferable leak at the "
    "present resolution, and the constant-leak injection route is closed. "
    "Full Jones stays unfrozen pending the high-resolution CASSBEAM "
    "direct visibility-domain template test. Pass-1/pass-2 exact-cell "
    "visit tests are not testable on THOL0001."
)
LORO_LEAKAGE_SENSITIVITY = "loro_leakage_sensitivity"
HIGHRES_CASSBEAM_DIRECT = "highres_cassbeam_direct_visibility_validation"
JONES_MAP_DISAGREEMENT_IS_DIAGNOSTIC_NOTE = (
    "A large relative Jones-map disagreement is diagnostic only. "
    "Factor-level relative error can be large while the RIME visibility "
    "error stays a few percent, especially for small or gauge-sensitive "
    "entries. Convert each Jones element through the RIME before assigning "
    "scientific meaning."
)
LORO_FULL_VERSUS_DIAGONAL = "loro_full_versus_diagonal"


def holoraster_pair_masks(observation: HolographyObservation) -> dict[str, NDArray]:
    """Vectorized moving–reference, reference–reference, and origin masks."""

    offsets, inverse, pointing_valid, settled, _moving = observation.pointing_state()
    roles = np.full(offsets.shape[:2], "unknown", dtype="U16")
    for index, antenna in enumerate(observation.pointing.antenna_id):
        roles[:, int(antenna)] = observation.pointing.role[:, index]
    time_index = np.asarray(inverse, dtype=np.int32)
    antenna_p = np.asarray(observation.block.antenna1, dtype=np.int32)
    antenna_q = np.asarray(observation.block.antenna2, dtype=np.int32)
    settled_ok = (
        pointing_valid[time_index, antenna_p]
        & pointing_valid[time_index, antenna_q]
        & settled[time_index, antenna_p]
        & settled[time_index, antenna_q]
    )
    four_hand = four_hand_active_rows(observation)
    role_p = roles[time_index, antenna_p]
    role_q = roles[time_index, antenna_q]
    moving_is_p = role_p == "moving"
    moving_is_q = role_q == "moving"
    ref_p = role_p == "reference"
    ref_q = role_q == "reference"
    moving_ref = settled_ok & (moving_is_p ^ moving_is_q) & (ref_p | ref_q)
    ref_ref = settled_ok & ref_p & ref_q
    moving = np.where(moving_is_p, antenna_p, antenna_q).astype(np.int32)
    reference = np.where(moving_is_p, antenna_q, antenna_p).astype(np.int32)
    offset = np.full((antenna_p.size, 2), np.nan, dtype=np.float64)
    if bool(np.any(moving_ref)):
        offset[moving_ref] = offsets[time_index[moving_ref], moving[moving_ref]]
    radius = np.hypot(offset[:, 0], offset[:, 1])
    origin = moving_ref & np.isfinite(radius) & (radius <= 1.0e-6)
    return {
        "time_index": time_index,
        "antenna1": antenna_p,
        "antenna2": antenna_q,
        "moving_id": moving,
        "reference_id": reference,
        "moving_is_p": moving_is_p & moving_ref,
        "offset_lm_rad": offset,
        "radius_rad": radius,
        "settled_valid": settled_ok,
        "four_hand": four_hand,
        "moving_reference": moving_ref,
        "four_hand_moving_reference": moving_ref & four_hand,
        "reference_reference": ref_ref & four_hand,
        "origin": origin,
        "alignment_rows": (ref_ref | origin) & four_hand,
        "roles": roles,
        "offsets": offsets,
    }


def alignment_fit_mask(
    observation: HolographyObservation,
    *,
    exclude: ArrayLike | None = None,
) -> NDArray[np.bool_]:
    """Origin moving–reference plus reference–reference. Never off-axis MR."""

    geometry = holoraster_pair_masks(observation)
    allowed = np.asarray(geometry["alignment_rows"], dtype=bool)
    if exclude is not None:
        allowed = allowed & ~np.asarray(exclude, dtype=bool).reshape(-1)
    if bool(np.any(geometry["moving_reference"] & ~geometry["origin"] & allowed)):
        raise ValueError("alignment fits must not use non-origin moving-reference rows")
    return allowed


def _packed_planes(observation: HolographyObservation) -> tuple[NDArray, NDArray]:
    packed = pack_coherency(
        observation.block.visibility,
        observation.block.correlations,
        (Receptor.R, Receptor.L),
    )
    source = _source_coherency(observation)
    if packed.ndim == 4:
        vis = packed[:, 0]
        source_planes = source[:, 0] if source.ndim == 4 else source
    else:
        vis = packed
        source_planes = source
    return vis, source_planes


def _sky_residuals(
    residual_jones: Mapping[int, ArrayLike],
    parallactic_angle_rad: ArrayLike,
    time_index: NDArray[np.int32],
    antenna_id: NDArray[np.int32],
) -> NDArray[np.complex128]:
    import jax.numpy as jnp

    chi = np.asarray(parallactic_angle_rad, dtype=np.float64)
    planes = _antenna_jones_planes(residual_jones, antenna_id)
    return np.asarray(
        _sky_frame_residual_jax(jnp.asarray(planes), jnp.asarray(chi[time_index, antenna_id]))
    )


def _factor_table(
    n_visit: int,
    n_antenna: int,
    factors: Mapping[tuple[int, int], ArrayLike] | None = None,
) -> NDArray[np.complex128]:
    table = np.broadcast_to(np.eye(2, dtype=np.complex128), (n_visit, n_antenna, 2, 2)).copy()
    if factors:
        for (visit, antenna), plane in factors.items():
            table[int(visit), int(antenna)] = np.asarray(plane, dtype=np.complex128)
    return table


def estimate_reference_visit_B(
    observation: HolographyObservation,
    *,
    residual_jones: Mapping[int, ArrayLike],
    parallactic_angle_rad: ArrayLike,
    reference_antenna_id: int,
    visit_id: ArrayLike,
    row_mask: ArrayLike | None = None,
) -> dict[tuple[int, int], NDArray[np.complex128]]:
    """Fit :math:`B_{rv}` from reference–reference rows only."""

    geometry = holoraster_pair_masks(observation)
    visits = np.asarray(visit_id, dtype=np.int32).reshape(-1)
    if visits.size != int(observation.pointing.unique_time_s.size):
        raise ValueError("visit_id must have one label per unique time")
    mask = np.asarray(geometry["reference_reference"], dtype=bool)
    if row_mask is not None:
        mask = mask & np.asarray(row_mask, dtype=bool).reshape(-1)
    rows = np.flatnonzero(mask)
    factors: dict[tuple[int, int], NDArray[np.complex128]] = {}
    n_ant = int(np.max(observation.pointing.antenna_id)) + 1
    n_visit = int(np.max(visits)) + 1 if visits.size else 1
    if rows.size == 0:
        return factors
    vis, source = _packed_planes(observation)
    time_index = geometry["time_index"][rows]
    ant_p = geometry["antenna1"][rows]
    ant_q = geometry["antenna2"][rows]
    visit = visits[time_index]
    r_p = _sky_residuals(residual_jones, parallactic_angle_rad, time_index, ant_p)
    r_q = _sky_residuals(residual_jones, parallactic_angle_rad, time_index, ant_q)
    transformed = (
        invert_jones(r_p) @ vis[rows] @ invert_jones(np.conjugate(np.swapaxes(r_q, -1, -2)))
    )
    source_inv = invert_jones(source[rows])
    from_p = (source_inv @ transformed).swapaxes(-1, -2).conj()
    from_q = transformed @ source_inv
    acc = np.zeros((n_visit, n_ant, 2, 2), dtype=np.complex128)
    count = np.zeros((n_visit, n_ant), dtype=np.float64)
    weight = four_hand_sample_weight(observation)[rows]
    pin_q = ant_p == int(reference_antenna_id)
    pin_p = ant_q == int(reference_antenna_id)
    if bool(np.any(pin_q)):
        np.add.at(acc, (visit[pin_q], ant_q[pin_q]), from_p[pin_q] * weight[pin_q, None, None])
        np.add.at(count, (visit[pin_q], ant_q[pin_q]), weight[pin_q])
    if bool(np.any(pin_p)):
        np.add.at(acc, (visit[pin_p], ant_p[pin_p]), from_q[pin_p] * weight[pin_p, None, None])
        np.add.at(count, (visit[pin_p], ant_p[pin_p]), weight[pin_p])
    for visit_i in range(n_visit):
        factors[(visit_i, int(reference_antenna_id))] = np.eye(2, dtype=np.complex128)
        for antenna in range(n_ant):
            if antenna == int(reference_antenna_id):
                continue
            if count[visit_i, antenna] <= 0.0:
                continue
            plane = acc[visit_i, antenna] / count[visit_i, antenna]
            if bool(np.all(np.isfinite(plane))) and abs(complex(np.linalg.det(plane))) > 1.0e-12:
                factors[(visit_i, antenna)] = plane
    return factors


def estimate_mover_visit_A(
    observation: HolographyObservation,
    *,
    residual_jones: Mapping[int, ArrayLike],
    parallactic_angle_rad: ArrayLike,
    visit_id: ArrayLike,
    reference_factors: Mapping[tuple[int, int], ArrayLike],
    row_mask: ArrayLike | None = None,
) -> dict[tuple[int, int], NDArray[np.complex128]]:
    """Fit :math:`A_{mv}` from origin moving–reference dwells only.

    ``B_r`` enters the RIME inversion as part of the reference Jones.
    Post-hoc removal after an unpolarized recover is not used.
    """

    import jax.numpy as jnp

    geometry = holoraster_pair_masks(observation)
    visits = np.asarray(visit_id, dtype=np.int32).reshape(-1)
    mask = np.asarray(geometry["origin"] & geometry["four_hand"], dtype=bool)
    if row_mask is not None:
        mask = mask & np.asarray(row_mask, dtype=bool).reshape(-1)
    if bool(np.any(geometry["moving_reference"] & ~geometry["origin"] & mask)):
        raise ValueError("A_mv fits must not use non-origin moving-reference rows")
    rows = np.flatnonzero(mask)
    if rows.size == 0:
        return {}
    vis, source = _packed_planes(observation)
    time_index = geometry["time_index"][rows]
    moving = geometry["moving_id"][rows]
    reference = geometry["reference_id"][rows]
    visit = visits[time_index]
    r_m = _sky_residuals(residual_jones, parallactic_angle_rad, time_index, moving)
    r_r = _sky_residuals(residual_jones, parallactic_angle_rad, time_index, reference)
    n_ant = int(np.max(observation.pointing.antenna_id)) + 1
    n_visit = int(np.max(visits)) + 1
    b_table = _factor_table(n_visit, n_ant, reference_factors)
    r_r = apply_visit_factors(r_r, b_table[visit, reference])
    beams, _ok = _recover_row_jones_jax(
        jnp.asarray(vis[rows][:, None, :, :]),
        jnp.asarray(source[rows][:, None, :, :]),
        jnp.asarray(r_m),
        jnp.asarray(r_r),
        jnp.asarray(geometry["moving_is_p"][rows]),
    )
    implied = np.asarray(beams)[:, 0]
    acc = np.zeros((n_visit, n_ant, 2, 2), dtype=np.complex128)
    count = np.zeros((n_visit, n_ant), dtype=np.float64)
    weight = four_hand_sample_weight(observation)[rows]
    np.add.at(acc, (visit, moving), implied * weight[:, None, None])
    np.add.at(count, (visit, moving), weight)
    factors = {}
    for visit_i in range(n_visit):
        for antenna in np.unique(moving):
            if count[visit_i, int(antenna)] <= 0.0:
                continue
            plane = acc[visit_i, int(antenna)] / count[visit_i, int(antenna)]
            if bool(np.all(np.isfinite(plane))) and abs(complex(np.linalg.det(plane))) > 1.0e-12:
                factors[(visit_i, int(antenna))] = plane
    return factors


def align_recovered_jones(
    recovered: ArrayLike,
    a_moving: ArrayLike,
    b_reference: ArrayLike,
) -> NDArray[np.complex128]:
    """Undo :math:`Ê = A E B^H` with :math:`E = A^{-1} Ê B^{-H}`."""

    return (
        invert_jones(a_moving)
        @ np.asarray(recovered, dtype=np.complex128)
        @ invert_jones(
            np.conjugate(np.swapaxes(np.asarray(b_reference, dtype=np.complex128), -1, -2))
        )
    )


def apply_visit_factors(
    residual: ArrayLike,
    factor: ArrayLike,
) -> NDArray[np.complex128]:
    """Right-multiply a residual Jones by a visit factor."""

    return np.asarray(residual, dtype=np.complex128) @ np.asarray(factor, dtype=np.complex128)


def factor_tables(
    n_visit: int,
    n_antenna: int,
    mover_factors: Mapping[tuple[int, int], ArrayLike] | None = None,
    reference_factors: Mapping[tuple[int, int], ArrayLike] | None = None,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Identity-filled :math:`A_{mv}` and :math:`B_{rv}` tables."""

    return (
        _factor_table(n_visit, n_antenna, mover_factors),
        _factor_table(n_visit, n_antenna, reference_factors),
    )


def copy_visit_factors(
    factors: Mapping[tuple[int, int], ArrayLike],
    *,
    source_visit: int,
    dest_visit: int,
) -> dict[tuple[int, int], NDArray[np.complex128]]:
    """Copy antenna factors from one visit onto another missing visit."""

    copied = {
        (int(visit), int(antenna)): np.asarray(plane, dtype=np.complex128)
        for (visit, antenna), plane in factors.items()
    }
    dest = int(dest_visit)
    src = int(source_visit)
    for (visit, antenna), plane in list(copied.items()):
        if visit == src and (dest, antenna) not in copied:
            copied[(dest, antenna)] = np.array(plane, copy=True)
    return copied


def _relative_cell_residual(planes: NDArray[np.complex128], cell: NDArray) -> NDArray[np.float64]:
    if planes.shape[0] == 0:
        return np.zeros((0, 2, 2), dtype=np.float64)
    _uniq, inverse, count = np.unique(cell, axis=0, return_inverse=True, return_counts=True)
    acc = np.zeros((_uniq.shape[0], 2, 2), dtype=np.complex128)
    np.add.at(acc, inverse, planes)
    mean = acc / np.maximum(count, 1.0)[:, None, None]
    residual = planes - mean[inverse]
    scale = np.maximum(np.abs(mean[inverse, 0, 0]), 1.0e-3)
    rel = np.abs(residual) / scale[:, None, None]
    return np.where(count[inverse][:, None, None] >= 2, rel, np.nan)


def identical_cell_disagreement(
    jones: ArrayLike,
    keys: ArrayLike,
    *,
    aligned: ArrayLike | None = None,
) -> dict[str, object]:
    """Compare maps that share a mover/cell but differ in reference or visit."""

    planes = np.asarray(jones, dtype=np.complex128)
    labels = np.asarray(keys)
    aligned_planes = planes if aligned is None else np.asarray(aligned, dtype=np.complex128)
    if planes.shape[0] == 0:
        return {"n_pairs": 0, "median_rel_rr": float("nan"), "median_rel_ll": float("nan")}
    cell = labels[:, :3]
    rel = _relative_cell_residual(planes, cell)
    rel_aligned = _relative_cell_residual(aligned_planes, cell)
    finite = np.isfinite(rel[:, 0, 0])
    if not bool(np.any(finite)):
        return {"n_pairs": 0, "median_rel_rr": float("nan"), "median_rel_ll": float("nan")}

    def _median(stack: NDArray[np.float64], row: int, col: int) -> float:
        values = stack[finite, row, col]
        return float(np.median(values)) if values.size else float("nan")

    report = {
        "n_pairs": int(np.sum(finite)),
        "median_rel_rr": _median(rel, 0, 0),
        "median_rel_ll": _median(rel, 1, 1),
        "median_rel_rl": _median(rel, 0, 1),
        "median_rel_lr": _median(rel, 1, 0),
        "aligned_median_rel_rr": _median(rel_aligned, 0, 0),
        "aligned_median_rel_ll": _median(rel_aligned, 1, 1),
        "aligned_median_rel_rl": _median(rel_aligned, 0, 1),
        "aligned_median_rel_lr": _median(rel_aligned, 1, 0),
        "scientific": False,
        "notes": (JONES_MAP_DISAGREEMENT_IS_DIAGNOSTIC_NOTE,),
    }
    return report


def visibility_error_from_jones_disagreement(
    jones: ArrayLike,
    keys: ArrayLike,
    source_plane: ArrayLike,
) -> dict[str, object]:
    """Convert same-cell Jones scatter into unpolarized visibility error."""

    planes = np.asarray(jones, dtype=np.complex128)
    labels = np.asarray(keys)
    source = np.asarray(source_plane, dtype=np.complex128)
    if source.ndim == 2:
        source = np.broadcast_to(source, (planes.shape[0], 2, 2))
    if planes.shape[0] == 0:
        return {"n_pairs": 0, "scientific": False}
    cell = labels[:, :3]
    _uniq, inverse, count = np.unique(cell, axis=0, return_inverse=True, return_counts=True)
    acc = np.zeros((_uniq.shape[0], 2, 2), dtype=np.complex128)
    np.add.at(acc, inverse, planes)
    mean = acc / np.maximum(count, 1.0)[:, None, None]
    vis = planes @ source @ np.conjugate(np.swapaxes(planes, -1, -2))
    vis_mean = mean @ source[0] @ np.conjugate(np.swapaxes(mean, -1, -2))
    delta = vis - vis_mean[inverse]
    keep = count[inverse] >= 2
    intensity = np.maximum(np.abs(0.5 * (source[:, 0, 0] + source[:, 1, 1])), 1.0e-3)

    def _hand(row: int, col: int) -> dict[str, float]:
        values = delta[keep, row, col]
        finite = values[np.isfinite(values)]
        scale = intensity[keep][np.isfinite(values)]
        if finite.size == 0:
            return {"median_jy": float("nan"), "median_over_i": float("nan")}
        return {
            "median_jy": float(np.median(np.abs(finite))),
            "median_over_i": float(np.median(np.abs(finite) / scale)),
        }

    return {
        "n_pairs": int(np.sum(keep)),
        "scientific": False,
        "rr": _hand(0, 0),
        "rl": _hand(0, 1),
        "lr": _hand(1, 0),
        "ll": _hand(1, 1),
        "notes": (JONES_MAP_DISAGREEMENT_IS_DIAGNOSTIC_NOTE,),
    }


def direction_residual_after_alignment(
    radius_rad: ArrayLike,
    residual_rr: ArrayLike,
    residual_ll: ArrayLike,
) -> dict[str, object]:
    """Ask whether leftover copolar error grows with beam radius."""

    radius = np.asarray(radius_rad, dtype=np.float64).reshape(-1)
    rr = np.asarray(residual_rr, dtype=np.float64).reshape(-1)
    ll = np.asarray(residual_ll, dtype=np.float64).reshape(-1)
    finite = np.isfinite(radius) & np.isfinite(rr) & np.isfinite(ll)
    if int(np.sum(finite)) < 8:
        return {"n": int(np.sum(finite)), "slope_rr": float("nan"), "direction_dependent": None}
    x = radius[finite]
    y = 0.5 * (rr[finite] + ll[finite])
    design = np.stack([np.ones(x.size), x], axis=1)
    slope = float(np.linalg.lstsq(design, y, rcond=None)[0][1])
    inner = y[x <= np.median(x)]
    outer = y[x > np.median(x)]
    return {
        "n": int(x.size),
        "slope_rr_ll": slope,
        "inner_median": float(np.median(inner)) if inner.size else float("nan"),
        "outer_median": float(np.median(outer)) if outer.size else float("nan"),
        "direction_dependent": bool(
            np.isfinite(slope)
            and slope > 0.0
            and float(np.median(outer)) > 2.0 * max(float(np.median(inner)), 0.01)
        ),
    }


def apparent_voltage_response(
    measured: ArrayLike,
    intensity: ArrayLike,
    rr_ok: ArrayLike,
    ll_ok: ArrayLike,
) -> NDArray[np.float64]:
    """Apparent copolar voltage response ``|V| / I_model``. Not a residual."""

    vis = np.asarray(measured)
    if vis.ndim == 4:
        vis = vis[:, 0]
    scale = np.maximum(np.abs(np.asarray(intensity, dtype=np.float64).reshape(-1)), 1.0e-3)
    rr_ok = np.asarray(rr_ok, dtype=bool).reshape(-1)
    ll_ok = np.asarray(ll_ok, dtype=bool).reshape(-1)
    rr = np.abs(vis[:, 0, 0]) / scale
    ll = np.abs(vis[:, 1, 1]) / scale
    count = rr_ok.astype(np.float64) + ll_ok.astype(np.float64)
    total = np.where(rr_ok, rr, 0.0) + np.where(ll_ok, ll, 0.0)
    return np.where(count > 0.0, total / count, np.nan)


def voltage_response_region_masks(voltage: ArrayLike) -> dict[str, NDArray[np.bool_]]:
    """Fixed beam-response bins. Outer cells are diagnostic only."""

    response = np.asarray(voltage, dtype=np.float64).reshape(-1)
    return {
        "main_lobe": np.isfinite(response) & (response >= VOLTAGE_MAINLOBE),
        "mid": np.isfinite(response) & (response >= VOLTAGE_MID) & (response < VOLTAGE_MAINLOBE),
        "outer_diagnostic": np.isfinite(response) & (response < VOLTAGE_MID),
    }


def _median_abs(values: ArrayLike) -> float:
    array = np.asarray(values)
    if np.iscomplexobj(array):
        array = np.abs(array)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return float("nan")
    return float(np.median(finite))


def score_copolar_residuals(
    measured: ArrayLike,
    predicted: ArrayLike,
    intensity: ArrayLike,
    rr_ok: ArrayLike,
    ll_ok: ArrayLike,
    row_mask: ArrayLike,
    voltage: ArrayLike | None = None,
) -> dict[str, object]:
    """Source-normalized and Jy copolar residuals, split by voltage region."""

    from sl1mjax.holography_full_jones import _hand_residual

    vis = np.asarray(measured)
    pred = np.asarray(predicted)
    if pred.ndim == 4:
        pred = pred[:, 0]
    if vis.ndim == 4:
        vis = vis[:, 0]
    intensity = np.asarray(intensity, dtype=np.float64).reshape(-1)
    rr_ok = np.asarray(rr_ok, dtype=bool).reshape(-1) & np.asarray(row_mask, dtype=bool).reshape(-1)
    ll_ok = np.asarray(ll_ok, dtype=bool).reshape(-1) & np.asarray(row_mask, dtype=bool).reshape(-1)
    if voltage is None:
        voltage = apparent_voltage_response(vis, intensity, rr_ok, ll_ok)
    voltage = np.asarray(voltage, dtype=np.float64).reshape(-1)
    scored_rows = np.asarray(row_mask, dtype=bool).reshape(-1)
    regions = {"all": scored_rows}
    for name, region in voltage_response_region_masks(voltage).items():
        regions[name] = region & scored_rows

    def _one(mask: NDArray[np.bool_]) -> dict[str, object]:
        rr_mask = mask & rr_ok
        ll_mask = mask & ll_ok
        rr = _hand_residual(vis, pred, intensity, rr_mask, 0, 0)
        ll = _hand_residual(vis, pred, intensity, ll_mask, 1, 1)
        rr_jy = np.abs(vis[rr_mask, 0, 0] - pred[rr_mask, 0, 0])
        ll_jy = np.abs(vis[ll_mask, 1, 1] - pred[ll_mask, 1, 1])
        return {
            "n": int(np.sum(mask)),
            "n_meaning": "scored rows in this region after the row mask",
            "n_rr": int(np.sum(rr_mask)),
            "n_ll": int(np.sum(ll_mask)),
            "n_finite": int(min(np.sum(np.isfinite(rr)), np.sum(np.isfinite(ll)))),
            "median_abs_rr_over_i": _median_abs(rr),
            "median_abs_ll_over_i": _median_abs(ll),
            "median_abs_rr_jy": _median_abs(rr_jy),
            "median_abs_ll_jy": _median_abs(ll_jy),
        }

    scored = {name: _one(mask) for name, mask in regions.items()}
    scored["voltage_response"] = {
        "definition": "mean(|V_RR|, |V_LL|) / I_model on individually valid hands",
        "regions": {
            "main_lobe": f">= {VOLTAGE_MAINLOBE}",
            "mid": f"{VOLTAGE_MID} to {VOLTAGE_MAINLOBE}",
            "outer_diagnostic": f"< {VOLTAGE_MID}",
        },
    }
    return scored


def visit_holdout_masks(
    observation: HolographyObservation,
    *,
    kind: str,
    visit_id: ArrayLike,
    held_visit: int,
    held_reference_id: int | None = None,
    reserve_outer_fold: bool = True,
) -> dict[str, NDArray]:
    """Split uncalibrated visit transfer from beam-repeatability holdouts."""

    if kind not in {UNCALIBRATED_VISIT_TRANSFER, BEAM_REPEATABILITY, "leave_one_reference_out"}:
        raise ValueError(f"unknown alignment holdout {kind!r}")
    geometry = holoraster_pair_masks(observation)
    visits = np.asarray(visit_id, dtype=np.int32).reshape(-1)
    time_index = geometry["time_index"]
    usable = np.asarray(geometry["moving_reference"], dtype=bool)
    four_hand = np.asarray(geometry["four_hand"], dtype=bool)
    visit_rows = visits[time_index] == int(held_visit)
    origin = np.asarray(geometry["origin"], dtype=bool)
    if kind == "leave_one_reference_out":
        if held_reference_id is None:
            raise ValueError("leave-one-reference-out needs held_reference_id")
        holdout = usable & (geometry["reference_id"] == int(held_reference_id)) & ~origin
        train = usable & four_hand & (geometry["reference_id"] != int(held_reference_id))
        alignment_exclude = usable & (geometry["reference_id"] == int(held_reference_id)) & ~origin
    elif kind == UNCALIBRATED_VISIT_TRANSFER:
        holdout = usable & visit_rows
        train = usable & four_hand & ~visit_rows
        alignment_exclude = visit_rows
    else:
        holdout = usable & visit_rows & ~origin
        train = usable & four_hand & (~visit_rows | origin)
        alignment_exclude = usable & visit_rows & ~origin
    reserved = np.zeros(usable.shape, dtype=bool)
    if reserve_outer_fold and bool(np.any(holdout)):
        times = np.sort(np.unique(time_index[holdout]))
        late = set(times[max(1, int(np.ceil(0.75 * times.size))) :].tolist())
        reserved = holdout & np.isin(time_index, list(late))
    return {
        "usable": usable,
        "train": train,
        "holdout": holdout,
        "reserved": reserved,
        "scored": holdout & ~reserved,
        "alignment_exclude": alignment_exclude,
        "origin": origin,
    }


def recover_rows(
    observation: HolographyObservation,
    *,
    residual_jones: Mapping[int, ArrayLike],
    parallactic_angle_rad: ArrayLike,
    row_mask: ArrayLike,
    mover_factors: ArrayLike | None = None,
    reference_factors: ArrayLike | None = None,
    visit_id: ArrayLike | None = None,
) -> tuple[NDArray[np.complex128], dict[str, NDArray]]:
    """Recover per-row :math:`E_m` on selected moving–reference rows."""

    import jax.numpy as jnp

    geometry = holoraster_pair_masks(observation)
    mask = (
        np.asarray(row_mask, dtype=bool).reshape(-1)
        & geometry["moving_reference"]
        & geometry["four_hand"]
    )
    rows = np.flatnonzero(mask)
    vis, source = _packed_planes(observation)
    time_index = geometry["time_index"][rows]
    moving = geometry["moving_id"][rows]
    reference = geometry["reference_id"][rows]
    r_m = _sky_residuals(residual_jones, parallactic_angle_rad, time_index, moving)
    r_r = _sky_residuals(residual_jones, parallactic_angle_rad, time_index, reference)
    if mover_factors is not None and reference_factors is not None and visit_id is not None:
        visits = np.asarray(visit_id, dtype=np.int32)[time_index]
        r_m = apply_visit_factors(r_m, np.asarray(mover_factors)[visits, moving])
        r_r = apply_visit_factors(r_r, np.asarray(reference_factors)[visits, reference])
    beams, ok = _recover_row_jones_jax(
        jnp.asarray(vis[rows][:, None, :, :]),
        jnp.asarray(source[rows][:, None, :, :]),
        jnp.asarray(r_m),
        jnp.asarray(r_r),
        jnp.asarray(geometry["moving_is_p"][rows]),
    )
    recovered = np.asarray(beams)[:, 0]
    meta = {
        "rows": rows,
        "time_index": time_index,
        "moving_id": moving,
        "reference_id": reference,
        "offset_lm_rad": geometry["offset_lm_rad"][rows],
        "radius_rad": geometry["radius_rad"][rows],
        "ok": np.asarray(ok)[:, 0] if np.asarray(ok).ndim > 1 else np.asarray(ok),
    }
    return recovered, meta


def artifact_from_recovered_rows(
    observation: HolographyObservation,
    recovered: ArrayLike,
    meta: Mapping[str, NDArray],
) -> HolographyFullJonesArtifact:
    """Average aligned per-row Jones onto (mover, time) samples."""

    planes = np.asarray(recovered, dtype=np.complex128)
    ok = np.asarray(meta["ok"], dtype=bool) & np.all(np.isfinite(planes), axis=(-2, -1))
    keys = np.stack([meta["moving_id"], meta["time_index"]], axis=1)
    if not bool(np.any(ok)):
        raise ValueError("aligned recovery found no usable samples")
    uniq, inverse = np.unique(keys[ok], axis=0, return_inverse=True)
    acc = np.zeros((uniq.shape[0], 2, 2), dtype=np.complex128)
    count = np.zeros((uniq.shape[0], 1, 1), dtype=np.float64)
    np.add.at(acc, inverse, planes[ok])
    np.add.at(count, inverse, 1.0)
    mean = acc / np.maximum(count, 1.0)
    unique_times = observation.pointing.unique_time_s
    frequency = float(observation.block.frequency_hz[0])
    memo = Memo195LowerCRaster()
    samples = []
    stacked = []
    for index, key in enumerate(uniq):
        offset = meta["offset_lm_rad"][ok][inverse == index][0]
        plane = mean[index]
        sample = HolographyFullJonesSample(
            moving_antenna_id=int(key[0]),
            unique_time_s=float(unique_times[int(key[1])]),
            offset_lm_rad=np.asarray(offset, dtype=np.float64),
            frequency_hz=frequency,
            jones=plane,
            sigma=np.ones((2, 2), dtype=np.float64),
            n_reference=int(np.sum(inverse == index)),
            weight=1.0,
            copolar_valid=True,
            off_diagonal_valid=True,
            raster=_raster_label(offset, memo),
        )
        samples.append(sample)
        stacked.append(plane)
    return HolographyFullJonesArtifact(
        samples=tuple(samples),
        jones=np.stack(stacked, axis=0),
        valid=np.ones(len(samples), dtype=bool),
        off_diagonal_valid=np.ones(len(samples), dtype=bool),
        calibration_state=observation.calibration_state,
        source_name=observation.source_name,
        source_model_version="aligned_reference_visit",
        reference_combination="aligned_per_row_then_time",
        receptor_convention="circular_R_L",
        offset_sign=observation.pointing.offset_sign,
        frozen=False,
        schema_version=HOLOGRAPHY_FULL_JONES_SCHEMA_VERSION,
        notes=(ALIGNMENT_NOTE,),
    )


def predict_aligned_moving_reference(
    observation: HolographyObservation,
    *,
    residual_jones: Mapping[int, ArrayLike],
    artifact: HolographyFullJonesArtifact,
    row_mask: ArrayLike,
    parallactic_angle_rad: ArrayLike,
    visit_id: ArrayLike,
    mover_factors: ArrayLike,
    reference_factors: ArrayLike,
    diagonal_only: bool = False,
) -> NDArray[np.complex128]:
    """Predict :math:`V=R_m A_{mv} E S B_{rv}^H R_r^H`."""

    import jax.numpy as jnp

    geometry = holoraster_pair_masks(observation)
    mask = np.asarray(row_mask, dtype=bool).reshape(-1) & geometry["moving_reference"]
    vis, source = _packed_planes(observation)
    predicted = np.full((vis.shape[0], 2, 2), np.nan + 1j * np.nan, dtype=np.complex128)
    rows = np.flatnonzero(mask)
    if rows.size == 0:
        return predicted
    time_index = geometry["time_index"][rows]
    moving = geometry["moving_id"][rows]
    reference = geometry["reference_id"][rows]
    offset = geometry["offset_lm_rad"][rows]
    visits = np.asarray(visit_id, dtype=np.int32)[time_index]
    frequency = np.full(rows.size, float(observation.block.frequency_hz[0]))
    beams, ok, leak = interpolate_holography_full_jones_batch(
        artifact,
        offset,
        moving_antenna_id=moving,
        frequency_hz=frequency,
    )
    if diagonal_only:
        beams[..., 0, 1] = 0.0
        beams[..., 1, 0] = 0.0
    else:
        beams[~leak, 0, 1] = 0.0
        beams[~leak, 1, 0] = 0.0
    r_m = apply_visit_factors(
        _sky_residuals(residual_jones, parallactic_angle_rad, time_index, moving),
        np.asarray(mover_factors)[visits, moving],
    )
    r_r = apply_visit_factors(
        _sky_residuals(residual_jones, parallactic_angle_rad, time_index, reference),
        np.asarray(reference_factors)[visits, reference],
    )
    vis_hat = np.asarray(
        _predict_vis_jax(
            jnp.asarray(r_m),
            jnp.asarray(beams[:, None, :, :]),
            jnp.asarray(source[rows][:, None, :, :]),
            jnp.asarray(r_r),
            jnp.asarray(geometry["moving_is_p"][rows]),
        )
    )[:, 0]
    predicted[rows] = np.where(ok[:, None, None], vis_hat, predicted[rows])
    return predicted


def classify_reference_visit_aligned_copolar_transfer(
    *,
    unaligned_rr: float,
    unaligned_ll: float,
    aligned_rr: float,
    aligned_ll: float,
    direction_dependent: bool | None,
    copolar_floor: float = COPOLAR_FLOOR,
    n_finite: int | None = None,
    min_exact_cells: int = 100,
    availability: str = "available",
) -> dict[str, object]:
    """Classify one alignment test. Unavailable data is not a scientific fail."""

    if availability not in {"available", "insufficient_evidence", "not_testable"}:
        raise ValueError(f"unknown availability {availability!r}")
    too_few = n_finite is not None and int(n_finite) < int(min_exact_cells)
    reduced = (
        not too_few
        and np.isfinite(aligned_rr)
        and np.isfinite(aligned_ll)
        and aligned_rr < float(copolar_floor)
        and aligned_ll < float(copolar_floor)
    )
    still_large = (
        np.isfinite(aligned_rr)
        and np.isfinite(aligned_ll)
        and max(aligned_rr, aligned_ll)
        >= 0.5 * max(float(unaligned_rr), float(unaligned_ll), 1.0e-6)
        and max(aligned_rr, aligned_ll) >= float(copolar_floor)
    )
    if too_few and availability == "not_testable":
        outcome = "not_testable_on_this_dataset"
        status = "inconclusive"
        blocking = False
    elif too_few and availability == "insufficient_evidence":
        outcome = "inconclusive"
        status = "inconclusive"
        blocking = False
    elif too_few:
        outcome = "insufficient_exact_cells"
        status = "fail"
        blocking = True
    elif reduced and not direction_dependent:
        outcome = "direction_independent_alignment"
        status = "pass"
        blocking = False
    elif reduced and direction_dependent:
        outcome = "alignment_helps_but_direction_dependent"
        status = "fail"
        blocking = True
    elif still_large:
        outcome = "beam_repeatability_or_pointing"
        status = "fail"
        blocking = True
    else:
        outcome = "partial_alignment"
        status = "fail"
        blocking = True
    return {
        "gate": REFERENCE_VISIT_ALIGNED_COPOLAR_TRANSFER,
        "status": status,
        "blocking": blocking,
        "outcome": outcome,
        "availability": availability,
        "unaligned_rr_over_i": float(unaligned_rr) if np.isfinite(unaligned_rr) else None,
        "unaligned_ll_over_i": float(unaligned_ll) if np.isfinite(unaligned_ll) else None,
        "aligned_rr_over_i": float(aligned_rr) if np.isfinite(aligned_rr) else None,
        "aligned_ll_over_i": float(aligned_ll) if np.isfinite(aligned_ll) else None,
        "n_finite": None if n_finite is None else int(n_finite),
        "min_exact_cells": int(min_exact_cells),
        "copolar_floor": float(copolar_floor),
        "direction_dependent": direction_dependent,
        "full_jones_blocked": True,
        "comparison_fair": bool(reduced and not direction_dependent),
        "notes": (
            ALIGNMENT_NOTE,
            UNCALIBRATED_VISIT_NOTE,
            BEAM_REPEATABILITY_NOTE,
            FULL_JONES_BLOCKED_UNTIL_ALIGNMENT_NOTE,
        ),
    }


def combine_alignment_transfer_gates(
    *,
    leave_one_reference_out: Mapping[str, object],
    beam_repeatability: Mapping[str, object],
    uncalibrated_visit_transfer: Mapping[str, object],
) -> dict[str, object]:
    """The available held-reference copolar test decides this gate.

    Visit tests that this dataset cannot support are not scientific failures
    and do not keep Full Jones blocked.
    """

    loro_ok = leave_one_reference_out.get("status") == "pass"
    return {
        "gate": REFERENCE_VISIT_ALIGNED_COPOLAR_TRANSFER,
        "status": "pass" if loro_ok else "fail",
        "blocking": not loro_ok,
        "outcome": (
            "held_reference_copolar_passed" if loro_ok else "held_reference_copolar_failed"
        ),
        "leave_one_reference_out": dict(leave_one_reference_out),
        "beam_repeatability": dict(beam_repeatability),
        "uncalibrated_visit_transfer": dict(uncalibrated_visit_transfer),
        "uncalibrated_visit_is_global_transfer": True,
        "beam_repeatability_is_shape_test": True,
        "visit_tests_do_not_fail_the_beam": True,
        "full_jones_blocked": True,
        "full_jones_block_reason": (
            "missing_fair_rl_lr_test" if loro_ok else "loro_copolar_floor_not_met"
        ),
        "most_important_next_artifact": "cassbeam_diagonal_low_order_correction",
        "comparison_fair": bool(loro_ok),
        "notes": (
            ALIGNMENT_NOTE,
            UNCALIBRATED_VISIT_NOTE,
            BEAM_REPEATABILITY_NOTE,
            FULL_JONES_BLOCKED_UNTIL_ALIGNMENT_NOTE,
            JONES_MAP_DISAGREEMENT_IS_DIAGNOSTIC_NOTE,
        ),
    }
