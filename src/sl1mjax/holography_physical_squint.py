"""Independent physical R/L squint and common-width diagonal-beam experiment.

This is isolated from the previous coupled ``squint_scale`` warp and from
production beam selection. Native CASSBEAM squint is the identity prior.
SPW 5 stays sealed. Full Jones is untouched.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sl1mjax.holography_beam_prior import (
    compare_holoraster_stages,
    predict_from_moving_beams,
    squeeze_sample_jones,
)
from sl1mjax.holography_cassbeam_correction import (
    ARCMIN_TO_RAD,
    CASSBEAM_LL_PEAK_ARCMIN,
    CASSBEAM_RR_PEAK_ARCMIN,
    FeedFrameLookup,
    apply_parallactic_after_correction,
    renormalize_to_cassbeam_origin,
    require_identity_matches_baseline,
)
from sl1mjax.holography_diagonal_correction import refuse_spw5, require_boresight_unity

PHYSICAL_SQUINT_EXPERIMENT = "spw4_physical_squint_width_v1"
IDENTITY_ATOL = 1.0e-12
DEVELOPMENT_NOTE = (
    "SPW-4 results are development evidence. Spatial and mover holdouts "
    "already influenced model development."
)
NATIVE_SQUINT_NOTE = "Native CASSBEAM squint is the identity prior and is never removed by default."
NO_ENVELOPE_NOTE = "This experiment fits no real envelope, phase, sidelobe, or azimuthal term."


def native_hand_centers_rad() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Frozen CASSBEAM 20%-of-peak centres. Not re-estimated from HOLORASTER."""

    right = np.asarray(CASSBEAM_RR_PEAK_ARCMIN, dtype=np.float64) * ARCMIN_TO_RAD
    left = np.asarray(CASSBEAM_LL_PEAK_ARCMIN, dtype=np.float64) * ARCMIN_TO_RAD
    return right, left


def native_separation_rad() -> NDArray[np.float64]:
    right, left = native_hand_centers_rad()
    return right - left


def native_common_center_rad() -> NDArray[np.float64]:
    right, left = native_hand_centers_rad()
    return 0.5 * (right + left)


def _offset_pairs(offset_lm_rad: ArrayLike) -> NDArray[np.float64]:
    offset = np.asarray(offset_lm_rad, dtype=np.float64)
    if offset.ndim == 1:
        offset = offset.reshape(1, 2)
    if offset.ndim != 2 or offset.shape[1] != 2:
        raise ValueError("offset_lm_rad must have shape (sample, 2)")
    return offset


def _vector2(values: ArrayLike, name: str) -> NDArray[np.float64]:
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if vector.size != 2:
        raise ValueError(f"{name} must be a length-2 (l, m) vector")
    return vector


@dataclass(frozen=True)
class PhysicalBeamState:
    """Independent common width, common pointing, and R/L separation.

    Identity is ``width=1``, ``pointing=0``, ``delta=delta_native``. That
    must reproduce the frozen CASSBEAM prediction. Pointing is carried for
    algebra only; this experiment does not fit it.
    """

    width: float = 1.0
    pointing_lm_rad: tuple[float, float] = (0.0, 0.0)
    delta_lm_rad: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if float(self.width) <= 0.0:
            raise ValueError("common width must be positive")
        pointing = tuple(float(item) for item in self.pointing_lm_rad)
        if len(pointing) != 2:
            raise ValueError("pointing_lm_rad must be (l, m)")
        object.__setattr__(self, "pointing_lm_rad", pointing)
        if self.delta_lm_rad is None:
            native = native_separation_rad()
            object.__setattr__(self, "delta_lm_rad", (float(native[0]), float(native[1])))
        else:
            delta = tuple(float(item) for item in self.delta_lm_rad)
            if len(delta) != 2:
                raise ValueError("delta_lm_rad must be (l, m)")
            object.__setattr__(self, "delta_lm_rad", delta)

    def is_identity(self) -> bool:
        native = native_separation_rad()
        return (
            abs(float(self.width) - 1.0) <= 1.0e-15
            and abs(self.pointing_lm_rad[0]) <= 1.0e-15
            and abs(self.pointing_lm_rad[1]) <= 1.0e-15
            and abs(self.delta_lm_rad[0] - float(native[0])) <= 1.0e-15
            and abs(self.delta_lm_rad[1] - float(native[1])) <= 1.0e-15
        )

    def pointing(self) -> NDArray[np.float64]:
        return np.asarray(self.pointing_lm_rad, dtype=np.float64)

    def delta(self) -> NDArray[np.float64]:
        return np.asarray(self.delta_lm_rad, dtype=np.float64)


IDENTITY_PHYSICAL = PhysicalBeamState()


def refuse_old_squint_scale_coupling() -> None:
    raise RuntimeError("the coupled (width + squint_scale - 1) warp is not a physical squint")


def refuse_envelope_or_phase(term: str) -> None:
    name = str(term)
    if name in {
        "envelope",
        "sidelobe",
        "azimuthal",
        "phase",
        "first_sidelobe_radius_amplitude",
        "low_order_azimuthal",
        "rl_squint_scale",
    }:
        raise RuntimeError(f"{name} is outside the physical squint/width experiment")


def refuse_pointing_fit() -> None:
    raise RuntimeError("pointing remains rejected; this experiment does not fit p")


def target_hand_centers_rad(
    state: PhysicalBeamState = IDENTITY_PHYSICAL,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """t_R = c_bar + p + delta/2 and t_L = c_bar + p - delta/2.

    Implemented as ``c_h + p ± (delta - delta_native)/2`` so identity
    (``p=0``, ``delta=delta_native``) is exactly the native centres.
    """

    center_r, center_l = native_hand_centers_rad()
    extra = 0.5 * (state.delta() - native_separation_rad())
    pointing = state.pointing()
    return center_r + pointing + extra, center_l + pointing - extra


def physical_query_coordinates(
    offset_lm_rad: ArrayLike,
    state: PhysicalBeamState = IDENTITY_PHYSICAL,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """q_h = c_h + (x - t_h) / w. Identity returns the commanded coordinates."""

    offset = _offset_pairs(offset_lm_rad)
    center_r, center_l = native_hand_centers_rad()
    target_r, target_l = target_hand_centers_rad(state)
    width = float(state.width)
    query_r = center_r + (offset - target_r) / width
    query_l = center_l + (offset - target_l) / width
    return query_r, query_l


def apply_physical_feed_frame(
    offset_lm_rad: ArrayLike,
    lookup: FeedFrameLookup,
    state: PhysicalBeamState = IDENTITY_PHYSICAL,
) -> NDArray[np.complex128]:
    """Diagonal feed-frame Jones. No envelope, no parallactic rotation."""

    refuse_spw5(opened=False)
    offset = _offset_pairs(offset_lm_rad)
    query_r, query_l = physical_query_coordinates(offset, state)
    looked_r = np.asarray(lookup(query_r), dtype=np.complex128)
    looked_l = np.asarray(lookup(query_l), dtype=np.complex128)
    if looked_r.shape[0] != offset.shape[0] or looked_l.shape[0] != offset.shape[0]:
        raise ValueError("feed-frame lookup must return one Jones matrix per sample")
    jones = np.zeros((offset.shape[0], 2, 2), dtype=np.complex128)
    jones[:, 0, 0] = looked_r[:, 0, 0]
    jones[:, 1, 1] = looked_l[:, 1, 1]
    origin = np.zeros((1, 2), dtype=np.float64)
    q0_r, q0_l = physical_query_coordinates(origin, state)
    raw0 = np.zeros((2, 2), dtype=np.complex128)
    raw0[0, 0] = np.asarray(lookup(q0_r), dtype=np.complex128)[0, 0, 0]
    raw0[1, 1] = np.asarray(lookup(q0_l), dtype=np.complex128)[0, 1, 1]
    cass0 = np.asarray(lookup(origin), dtype=np.complex128)[0]
    corrected = renormalize_to_cassbeam_origin(jones, raw0, cass0)
    origin_corr = renormalize_to_cassbeam_origin(raw0[None, ...], raw0, cass0)[0]
    require_boresight_unity(
        origin_corr[0, 0] / cass0[0, 0],
        origin_corr[1, 1] / cass0[1, 1],
    )
    return corrected


def physical_path_stages(
    offset_lm_rad: ArrayLike,
    lookup: FeedFrameLookup,
    state: PhysicalBeamState,
    *,
    residual_jones: Mapping[int, ArrayLike],
    moving_id: ArrayLike,
    reference_id: ArrayLike,
    moving_is_p: ArrayLike,
    chi_moving: ArrayLike,
    chi_reference: ArrayLike,
    source: ArrayLike,
) -> dict[str, NDArray[np.complex128]]:
    """Feed, sky, and :math:`R_m E_m S R_r^H` from the physical parameterisation."""

    feed = apply_physical_feed_frame(offset_lm_rad, lookup, state)
    sky = apply_parallactic_after_correction(feed, chi_moving)
    vis = predict_from_moving_beams(
        residual_jones,
        moving_id,
        reference_id,
        moving_is_p,
        chi_moving,
        chi_reference,
        sky,
        source,
    )
    if vis.ndim == 4:
        if vis.shape[1] != 1:
            raise ValueError("physical-squint predictions must stay on one native channel")
        vis = vis[:, 0]
    return {"feed": feed, "sky": sky, "visibility": vis}


def require_physical_identity_matches_comparison(
    physical: Mapping[str, ArrayLike],
    comparison,
    *,
    atol: float = IDENTITY_ATOL,
) -> dict[str, object]:
    """Fail closed unless every staged RR/LL residual is at the identity gate."""

    report = compare_holoraster_stages(comparison, physical, atol=atol)
    if report["first_divergent_stage"] is not None:
        raise ValueError(
            "physical identity must reproduce the frozen comparison path; "
            f"first divergent stage is {report['first_divergent_stage']}"
        )
    if hasattr(comparison, "visibility"):
        compared = comparison.visibility
    else:
        compared = comparison["visibility"]
    require_identity_matches_baseline(
        squeeze_sample_jones(physical["visibility"]),
        squeeze_sample_jones(compared),
        atol=atol,
    )
    return report


def parameterization_record() -> dict[str, object]:
    right, left = native_hand_centers_rad()
    delta = native_separation_rad()
    return {
        "artifact": PHYSICAL_SQUINT_EXPERIMENT,
        "query": "q_h = c_h + (x - t_h) / w",
        "targets": "t_R = c_bar + p + delta/2; t_L = c_bar + p - delta/2",
        "identity": "w=1, p=0, delta=delta_native => q_h = x",
        "native_c_r_arcmin": [float(item) / ARCMIN_TO_RAD for item in right],
        "native_c_l_arcmin": [float(item) / ARCMIN_TO_RAD for item in left],
        "native_delta_arcmin": [float(item) / ARCMIN_TO_RAD for item in delta],
        "native_separation_arcmin": float(np.hypot(*delta) / ARCMIN_TO_RAD),
        "pointing_fitted": False,
        "envelope_fitted": False,
        "phase_fitted": False,
        "old_squint_scale_used": False,
        "notes": [DEVELOPMENT_NOTE, NATIVE_SQUINT_NOTE, NO_ENVELOPE_NOTE],
    }
