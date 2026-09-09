"""CASSBEAM × nested diagonal correction in the antenna/feed frame.

Corrections are coordinate warps and real multiplicative envelopes. They
are applied before parallactic rotation. Every shifted or warped
candidate is renormalized so :math:`C_R(0)=C_L(0)=1`. The first ladder
does not fit phase and does not open SPW 5.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sl1mjax.beam_conventions import BeamCalibrationState
from sl1mjax.holography_alignment import voltage_response_region_masks
from sl1mjax.holography_beam_prior import (
    apply_parallactic_to_beams,
    predict_from_moving_beams,
    squeeze_sample_jones,
    unique_feed_frame_jones,
)
from sl1mjax.holography_diagonal_correction import (
    FIRST_LADDER_TERMS,
    HOLORASTER_FIELD_ID,
    PAIRED_DELTA_BOOTSTRAP,
    SPW4_TRAINING_FREQUENCY_HZ,
    MoverPairedDeltas,
    PairedLossDifference,
    complex_visibility_loss,
    mover_cluster_ids,
    mover_paired_deltas,
    refuse_c147_training,
    refuse_phase_in_first_ladder,
    refuse_spw5,
    require_boresight_unity,
    score_paired_holdout,
    select_nested_correction,
    spatial_cluster_ids,
)
from sl1mjax.polarization import apply_jones_to_coherency

FeedFrameLookup = Callable[[ArrayLike], NDArray[np.complex128]]

CASSBEAM_RR_PEAK_ARCMIN = (0.21336572750765215, 0.14677094162432322)
CASSBEAM_LL_PEAK_ARCMIN = (-0.07827745912401778, -0.14352701647414165)
NOMINAL_FIRST_SIDELOBE_ARCMIN = 22.0
SIDELOBE_BUMP_WIDTH_ARCMIN = 6.0
AZIMUTHAL_CORE_ARCMIN = 8.0
ARCMIN_TO_RAD = np.pi / (180.0 * 60.0)

IDENTITY_STATE_NOTE = (
    "The identity candidate is CASSBEAM with C=1. It must reproduce the "
    "frozen baseline visibilities numerically before any term is added."
)
WARP_NOTE = (
    "Prefer coordinate warps and smooth multiplicative envelopes over "
    "additive maps. Phase is not a first-ladder parameter."
)
_TERM_FIELDS = {
    "identity": (),
    "rl_squint_scale": ("squint_scale",),
    "beam_width": ("width_scale",),
    "pointing_offset": ("pointing_l_rad", "pointing_m_rad"),
    "first_sidelobe_radius_amplitude": ("sidelobe_radius_scale", "sidelobe_amplitude"),
    "low_order_azimuthal": ("azimuthal",),
}
_IDENTITY_VALUES = {
    "squint_scale": 1.0,
    "width_scale": 1.0,
    "pointing_l_rad": 0.0,
    "pointing_m_rad": 0.0,
    "sidelobe_radius_scale": 1.0,
    "sidelobe_amplitude": 1.0,
    "azimuthal": (0.0, 0.0, 0.0, 0.0),
}
ENVELOPE_POSITIVE_FLOOR = 1.0e-6


def _value_is_identity(name: str, value: object) -> bool:
    identity = _IDENTITY_VALUES[name]
    if name == "azimuthal":
        return tuple(float(item) for item in value) == identity  # type: ignore[arg-type]
    return abs(float(value) - float(identity)) <= 1.0e-15  # type: ignore[arg-type]


@dataclass(frozen=True)
class CorrectionState:
    """Nested amplitude correction. Identity values are exact no-ops."""

    squint_scale: float = 1.0
    width_scale: float = 1.0
    pointing_l_rad: float = 0.0
    pointing_m_rad: float = 0.0
    sidelobe_radius_scale: float = 1.0
    sidelobe_amplitude: float = 1.0
    azimuthal: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    accepted_terms: tuple[str, ...] = ("identity",)

    def __post_init__(self) -> None:
        if self.width_scale <= 0.0 or self.sidelobe_radius_scale <= 0.0:
            raise ValueError("width and sidelobe radius scales must be positive")
        object.__setattr__(self, "azimuthal", tuple(float(item) for item in self.azimuthal))
        if len(self.azimuthal) != 4:
            raise ValueError("azimuthal envelope is (cosφ, sinφ, cos2φ, sin2φ)")
        terms = tuple(self.accepted_terms)
        object.__setattr__(self, "accepted_terms", terms)
        if not terms or terms[0] != "identity":
            raise ValueError("accepted_terms must start with identity")
        for term in terms:
            refuse_phase_in_first_ladder(term)
        expected = FIRST_LADDER_TERMS[: len(terms)]
        if terms != expected:
            raise ValueError("accepted_terms must be a prefix of the first ladder")
        allowed: set[str] = set()
        for term in terms:
            allowed.update(_TERM_FIELDS[term])
        for name in _IDENTITY_VALUES:
            if name in allowed:
                continue
            if not _value_is_identity(name, getattr(self, name)):
                raise ValueError(f"{name} can change only after that ladder term is accepted")

    def is_identity(self) -> bool:
        return self.accepted_terms == ("identity",)


IDENTITY_CORRECTION = CorrectionState()


def refuse_additive_map(kind: str) -> None:
    if str(kind) in {"pixel_map", "additive_residual", "binned_additive", "additive"}:
        raise RuntimeError("prefer coordinate warps and multiplicative envelopes")


def cassbeam_hand_peaks_rad() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Frozen CASSBEAM 20%-of-peak centroids. Not re-estimated from HOLORASTER."""

    rr = np.asarray(CASSBEAM_RR_PEAK_ARCMIN, dtype=np.float64) * ARCMIN_TO_RAD
    ll = np.asarray(CASSBEAM_LL_PEAK_ARCMIN, dtype=np.float64) * ARCMIN_TO_RAD
    return rr, ll


def _offset_pairs(offset_lm_rad: ArrayLike) -> NDArray[np.float64]:
    offset = np.asarray(offset_lm_rad, dtype=np.float64)
    if offset.ndim == 1:
        offset = offset.reshape(1, 2)
    if offset.ndim != 2 or offset.shape[1] != 2:
        raise ValueError("offset_lm_rad must have shape (sample, 2)")
    return offset


def _radial_scale(
    offset: NDArray[np.float64],
    scale: float,
    radius_rad: float,
) -> NDArray[np.float64]:
    if abs(float(scale) - 1.0) < 1.0e-15:
        return offset
    radius = np.hypot(offset[:, 0], offset[:, 1])
    window = 1.0 - np.exp(-((radius / float(radius_rad)) ** 2))
    factor = 1.0 + (float(scale) - 1.0) * window
    return offset * factor[:, None]


def warp_feed_frame_coordinates(
    offset_lm_rad: ArrayLike,
    state: CorrectionState,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Apply accepted warps in ladder order, independently for R and L."""

    offset = _offset_pairs(offset_lm_rad)
    peak_r, peak_l = cassbeam_hand_peaks_rad()
    right = offset - (float(state.squint_scale) - 1.0) * peak_r
    left = offset - (float(state.squint_scale) - 1.0) * peak_l
    right = right / float(state.width_scale)
    left = left / float(state.width_scale)
    shift = np.array([state.pointing_l_rad, state.pointing_m_rad], dtype=np.float64)
    right = right - shift
    left = left - shift
    radius0 = NOMINAL_FIRST_SIDELOBE_ARCMIN * ARCMIN_TO_RAD
    return (
        _radial_scale(right, state.sidelobe_radius_scale, radius0),
        _radial_scale(left, state.sidelobe_radius_scale, radius0),
    )


def multiplicative_envelope(
    offset_lm_rad: ArrayLike,
    state: CorrectionState,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Real copolar envelope with :math:`C(0)=1` on the commanded origin."""

    offset = _offset_pairs(offset_lm_rad)
    radius = np.hypot(offset[:, 0], offset[:, 1])
    angle = np.arctan2(offset[:, 0], offset[:, 1])
    r0 = NOMINAL_FIRST_SIDELOBE_ARCMIN * ARCMIN_TO_RAD
    sigma = SIDELOBE_BUMP_WIDTH_ARCMIN * ARCMIN_TO_RAD
    core = AZIMUTHAL_CORE_ARCMIN * ARCMIN_TO_RAD
    bump = np.exp(-0.5 * ((radius - r0) / sigma) ** 2)
    bump0 = float(np.exp(-0.5 * (r0 / sigma) ** 2))
    amplitude = float(state.sidelobe_amplitude)
    sidelobe = (1.0 + (amplitude - 1.0) * bump) / (1.0 + (amplitude - 1.0) * bump0)
    window = 1.0 - np.exp(-((radius / core) ** 2))
    a_c, a_s, b_c, b_s = state.azimuthal
    azimuth = 1.0 + window * (
        a_c * np.cos(angle)
        + a_s * np.sin(angle)
        + b_c * np.cos(2.0 * angle)
        + b_s * np.sin(2.0 * angle)
    )
    envelope = sidelobe * azimuth
    if np.any(envelope <= 0.0):
        raise ValueError("amplitude-only envelope must stay strictly positive")
    return envelope, envelope


def require_positive_envelope(offset_lm_rad: ArrayLike, state: CorrectionState) -> None:
    """Refuse a sign-changing envelope on the supplied raster offsets."""

    env_r, env_l = multiplicative_envelope(offset_lm_rad, state)
    if np.any(env_r <= 0.0) or np.any(env_l <= 0.0) or not np.all(np.isfinite(env_r * env_l)):
        raise ValueError("amplitude-only envelope must stay strictly positive")


def renormalize_to_cassbeam_origin(
    jones: ArrayLike,
    warped_origin: ArrayLike,
    cassbeam_origin: ArrayLike,
) -> NDArray[np.complex128]:
    """Force the commanded origin back onto the CASSBEAM flux gauge."""

    values = np.asarray(jones, dtype=np.complex128)
    raw0 = np.asarray(warped_origin, dtype=np.complex128).reshape(2, 2)
    cass0 = np.asarray(cassbeam_origin, dtype=np.complex128).reshape(2, 2)
    if raw0[0, 0] == 0.0 or raw0[1, 1] == 0.0:
        raise ValueError("warped origin Jones vanished; cannot renormalize")
    scale_r = cass0[0, 0] / raw0[0, 0]
    scale_l = cass0[1, 1] / raw0[1, 1]
    out = np.zeros_like(values)
    out[..., 0, 0] = values[..., 0, 0] * scale_r
    out[..., 1, 1] = values[..., 1, 1] * scale_l
    return out


def apply_feed_frame_correction(
    offset_lm_rad: ArrayLike,
    lookup: FeedFrameLookup,
    state: CorrectionState = IDENTITY_CORRECTION,
) -> NDArray[np.complex128]:
    """Correct CASSBEAM in the antenna/feed frame. No parallactic rotation."""

    offset = _offset_pairs(offset_lm_rad)
    warped_r, warped_l = warp_feed_frame_coordinates(offset, state)
    looked_r = np.asarray(lookup(warped_r), dtype=np.complex128)
    looked_l = np.asarray(lookup(warped_l), dtype=np.complex128)
    if looked_r.shape[0] != offset.shape[0] or looked_l.shape[0] != offset.shape[0]:
        raise ValueError("feed-frame lookup must return one Jones matrix per sample")
    env_r, env_l = multiplicative_envelope(offset, state)
    require_positive_envelope(offset, state)
    jones = np.zeros((offset.shape[0], 2, 2), dtype=np.complex128)
    jones[:, 0, 0] = looked_r[:, 0, 0] * env_r
    jones[:, 1, 1] = looked_l[:, 1, 1] * env_l
    origin = np.zeros((1, 2), dtype=np.float64)
    wr0, wl0 = warp_feed_frame_coordinates(origin, state)
    raw0 = np.zeros((2, 2), dtype=np.complex128)
    raw0[0, 0] = np.asarray(lookup(wr0), dtype=np.complex128)[0, 0, 0]
    raw0[1, 1] = np.asarray(lookup(wl0), dtype=np.complex128)[0, 1, 1]
    cass0 = np.asarray(lookup(origin), dtype=np.complex128)[0]
    corrected = renormalize_to_cassbeam_origin(jones, raw0, cass0)
    origin_corr = renormalize_to_cassbeam_origin(raw0[None, ...], raw0, cass0)[0]
    require_boresight_unity(
        origin_corr[0, 0] / cass0[0, 0],
        origin_corr[1, 1] / cass0[1, 1],
    )
    return corrected


def apply_parallactic_after_correction(
    feed_jones: ArrayLike,
    parallactic_angle_rad: ArrayLike,
    *,
    calibration_state: BeamCalibrationState | str = "casa_parang_true",
) -> NDArray[np.complex128]:
    """Rotate an already-corrected feed-frame Jones into the sky frame."""

    jones = np.asarray(feed_jones, dtype=np.complex128)
    chi = np.asarray(parallactic_angle_rad, dtype=np.float64).reshape(-1)
    if chi.size == 1:
        chi = np.full(jones.shape[0], float(chi[0]), dtype=np.float64)
    if chi.size != jones.shape[0]:
        raise ValueError("parallactic_angle_rad must match the Jones sample axis")
    return apply_parallactic_to_beams(jones, chi, calibration_state)


def predict_moving_reference_vis(
    moving_jones: ArrayLike,
    source: ArrayLike,
    moving_is_p: ArrayLike,
    *,
    reference_jones: ArrayLike | None = None,
) -> NDArray[np.complex128]:
    """Complex :math:`V=E_m S E_r^{\\mathrm{H}}` on moving–reference rows."""

    e_m = np.asarray(moving_jones, dtype=np.complex128)
    sky = np.asarray(source, dtype=np.complex128)
    mover_p = np.asarray(moving_is_p, dtype=bool).reshape(-1)
    if e_m.ndim != 3 or e_m.shape[-2:] != (2, 2):
        raise ValueError("moving_jones must have shape (sample, 2, 2)")
    if sky.ndim == 2:
        sky = np.broadcast_to(sky, e_m.shape)
    if sky.shape != e_m.shape:
        raise ValueError("source coherency must broadcast to the moving Jones")
    if reference_jones is None:
        e_r = np.broadcast_to(np.eye(2, dtype=np.complex128), e_m.shape)
    else:
        e_r = np.asarray(reference_jones, dtype=np.complex128)
        if e_r.ndim == 2:
            e_r = np.broadcast_to(e_r, e_m.shape)
    vis_p = apply_jones_to_coherency(sky, e_m, e_r)
    vis_q = apply_jones_to_coherency(sky, e_r, e_m)
    return np.where(mover_p[:, None, None], vis_p, vis_q)


def require_identity_matches_baseline(
    identity: ArrayLike,
    baseline: ArrayLike,
    *,
    atol: float = 1.0e-12,
) -> None:
    """The identity candidate is the frozen CASSBEAM baseline, not a new fit."""

    left = np.asarray(identity)
    right = np.asarray(baseline)
    if left.shape != right.shape:
        raise ValueError("identity and baseline arrays must have the same shape")
    if not np.allclose(left, right, rtol=0.0, atol=float(atol), equal_nan=True):
        raise ValueError("identity candidate must reproduce the frozen baseline numerically")


def require_aligned_frozen_prediction(
    offset_lm_rad: ArrayLike,
    moving_id: ArrayLike,
    reference_id: ArrayLike,
    predicted: ArrayLike,
    frozen_offset_lm_rad: ArrayLike,
    frozen_moving_id: ArrayLike,
    frozen_reference_id: ArrayLike,
    frozen_predicted: ArrayLike,
    *,
    atol: float = 1.0e-12,
) -> None:
    """Compare to the frozen HOLORASTER arrays in row order. Do not key-match."""

    offset = _offset_pairs(offset_lm_rad)
    frozen_offset = _offset_pairs(frozen_offset_lm_rad)
    moving = np.asarray(moving_id, dtype=np.int32).reshape(-1)
    reference = np.asarray(reference_id, dtype=np.int32).reshape(-1)
    frozen_moving = np.asarray(frozen_moving_id, dtype=np.int32).reshape(-1)
    frozen_reference = np.asarray(frozen_reference_id, dtype=np.int32).reshape(-1)
    if offset.shape != frozen_offset.shape or moving.shape != frozen_moving.shape:
        raise ValueError("frozen HOLORASTER rows are not the same set")
    if not np.array_equal(moving, frozen_moving) or not np.array_equal(
        reference, frozen_reference
    ):
        raise ValueError("frozen HOLORASTER antenna order drifted")
    if not np.allclose(offset, frozen_offset, rtol=0.0, atol=1.0e-15, equal_nan=True):
        raise ValueError("frozen HOLORASTER offsets drifted")
    require_identity_matches_baseline(predicted, frozen_predicted, atol=atol)


def identity_predictions(baseline: ArrayLike) -> NDArray[np.complex128]:
    """Return the frozen baseline. Identity does not re-evaluate CASSBEAM."""

    refuse_spw5(opened=False)
    return np.asarray(baseline, dtype=np.complex128).copy()


def next_term(state: CorrectionState) -> str | None:
    accepted = set(state.accepted_terms)
    for term in FIRST_LADDER_TERMS:
        if term not in accepted:
            return term
    return None


def accepted_prefix(term: str, **updates: float | tuple[float, ...]) -> CorrectionState:
    """Identity plus every ladder term through ``term``."""

    state = IDENTITY_CORRECTION
    for name in FIRST_LADDER_TERMS[1:]:
        kwargs = dict(updates) if name == term else {}
        state = with_accepted_term(state, name, **kwargs)
        if name == term:
            return state
    raise ValueError(f"unknown first-ladder term {term!r}")


def with_accepted_term(
    state: CorrectionState,
    term: str,
    **updates: float | tuple[float, ...],
) -> CorrectionState:
    refuse_phase_in_first_ladder(term)
    if term in state.accepted_terms:
        raise ValueError(f"{term} is already in the accepted prefix")
    expected = next_term(state)
    if term != expected:
        raise ValueError(f"next nested term is {expected}, not {term}")
    return replace(state, accepted_terms=state.accepted_terms + (term,), **updates)


def identity_residual_jones(antenna_ids: ArrayLike) -> dict[int, NDArray[np.complex128]]:
    """Identity residual Jones for every supplied antenna id."""

    eye = np.eye(2, dtype=np.complex128)
    return {
        int(antenna): eye.copy()
        for antenna in np.unique(np.asarray(antenna_ids, dtype=np.int32).reshape(-1))
    }


def _one_channel_vis(values: ArrayLike, name: str, dtype: type) -> NDArray:
    array = np.asarray(values, dtype=dtype)
    if array.ndim == 4:
        if array.shape[1] != 1:
            raise ValueError(f"{name} must contain exactly one channel")
        array = array[:, 0]
    if array.ndim != 3 or array.shape[-2:] != (2, 2):
        raise ValueError(f"{name} must have shape (sample, 2, 2) or (sample, 1, 2, 2)")
    return array


def _boolean_mask(values: ArrayLike, n: int, name: str) -> NDArray[np.bool_]:
    array = np.asarray(values)
    if array.dtype != np.bool_ and array.dtype != bool:
        raise ValueError(f"{name} must be a boolean mask")
    mask = np.asarray(array, dtype=bool).reshape(-1)
    if mask.size != n:
        raise ValueError(f"{name} must have one flag per sample")
    return mask


def _azimuthal_probe_offsets(offset_lm_rad: ArrayLike) -> NDArray[np.float64]:
    offset = _offset_pairs(offset_lm_rad)
    radii = np.asarray(
        [
            AZIMUTHAL_CORE_ARCMIN,
            2.0 * AZIMUTHAL_CORE_ARCMIN,
            NOMINAL_FIRST_SIDELOBE_ARCMIN,
        ],
        dtype=np.float64,
    ) * ARCMIN_TO_RAD
    angle = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    rings = [
        np.stack([radius * np.sin(angle), radius * np.cos(angle)], axis=1) for radius in radii
    ]
    return np.vstack((offset, *rings))


def shrink_azimuthal_coefficients(
    offset_lm_rad: ArrayLike,
    coef: tuple[float, float, float, float],
    *,
    sidelobe_amplitude: float = 1.0,
) -> tuple[float, float, float, float]:
    """Uniformly shrink the LS envelope so it cannot change sign."""

    offset = _azimuthal_probe_offsets(offset_lm_rad)
    radius = np.hypot(offset[:, 0], offset[:, 1])
    angle = np.arctan2(offset[:, 0], offset[:, 1])
    r0 = NOMINAL_FIRST_SIDELOBE_ARCMIN * ARCMIN_TO_RAD
    sigma = SIDELOBE_BUMP_WIDTH_ARCMIN * ARCMIN_TO_RAD
    core = AZIMUTHAL_CORE_ARCMIN * ARCMIN_TO_RAD
    bump = np.exp(-0.5 * ((radius - r0) / sigma) ** 2)
    bump0 = float(np.exp(-0.5 * (r0 / sigma) ** 2))
    amplitude = float(sidelobe_amplitude)
    sidelobe = (1.0 + (amplitude - 1.0) * bump) / (1.0 + (amplitude - 1.0) * bump0)
    window = 1.0 - np.exp(-((radius / core) ** 2))
    a_c, a_s, b_c, b_s = (float(item) for item in coef)
    harmonic = window * (
        a_c * np.cos(angle)
        + a_s * np.sin(angle)
        + b_c * np.cos(2.0 * angle)
        + b_s * np.sin(2.0 * angle)
    )
    envelope = sidelobe * (1.0 + harmonic)
    if np.all(envelope > ENVELOPE_POSITIVE_FLOOR):
        return (a_c, a_s, b_c, b_s)
    delta = sidelobe * harmonic
    needed = ENVELOPE_POSITIVE_FLOOR - sidelobe
    bad = delta < -1.0e-15
    if not bool(np.any(bad)):
        return (a_c, a_s, b_c, b_s)
    scale = float(np.min(needed[bad] / delta[bad]))
    if not np.isfinite(scale) or scale <= 0.0:
        return (0.0, 0.0, 0.0, 0.0)
    scale = min(1.0, scale)
    return (scale * a_c, scale * a_s, scale * b_c, scale * b_s)


def gaussian_diagonal_lookup(
    peak_r_rad: ArrayLike,
    peak_l_rad: ArrayLike,
    width_rad: float,
) -> FeedFrameLookup:
    """Manufactured feed-frame diagonal beam for tests. Not CASSBEAM."""

    peak_r = np.asarray(peak_r_rad, dtype=np.float64).reshape(2)
    peak_l = np.asarray(peak_l_rad, dtype=np.float64).reshape(2)
    width = float(width_rad)
    if width <= 0.0:
        raise ValueError("manufactured beam width must be positive")

    def lookup(offset_lm_rad: ArrayLike) -> NDArray[np.complex128]:
        offset = _offset_pairs(offset_lm_rad)
        right = np.exp(-0.5 * np.sum((offset - peak_r) ** 2, axis=1) / width**2)
        left = np.exp(-0.5 * np.sum((offset - peak_l) ** 2, axis=1) / width**2)
        jones = np.zeros((offset.shape[0], 2, 2), dtype=np.complex128)
        jones[:, 0, 0] = right
        jones[:, 1, 1] = left
        return jones

    return lookup


def catalog_diagonal_feed_lookup(
    catalog: object,
    frequency_hz: float,
    convention: object | None = None,
) -> FeedFrameLookup:
    """Feed-frame diagonal CASSBEAM lookup used by the frozen HOLORASTER RIME.

    Rounds and deduplicates query coordinates the same way as the comparison
    evaluator. Do not look up each raw row independently.
    """

    from sl1mjax.cassbeam_highres import HighresCassbeamCatalog
    from sl1mjax.holography_highres_cassbeam import DEFAULT_CONVENTION

    if not isinstance(catalog, HighresCassbeamCatalog):
        raise TypeError("catalog_diagonal_feed_lookup requires a HighresCassbeamCatalog")
    conv = DEFAULT_CONVENTION if convention is None else convention
    freqs = np.asarray([float(frequency_hz)], dtype=np.float64)

    def lookup(offset_lm_rad: ArrayLike) -> NDArray[np.complex128]:
        offset = _offset_pairs(offset_lm_rad)
        feed, ok = unique_feed_frame_jones(
            offset, catalog, freqs, conv, off_diagonal=False
        )
        plane = squeeze_sample_jones(feed)
        return np.where(ok[:, 0, None, None], plane, np.nan)

    return lookup


SQUINT_SCALE_GRID = np.linspace(0.85, 1.50, 27)
WIDTH_SCALE_GRID = np.linspace(0.90, 1.12, 23)
POINTING_ARCMIN_GRID = np.linspace(-1.5, 1.5, 13)
SIDELOBE_RADIUS_GRID = np.linspace(0.85, 1.15, 13)
SIDELOBE_AMPLITUDE_GRID = np.linspace(0.70, 1.40, 15)
FROZEN_HYPERPARAMETERS = {
    "cassbeam_rr_peak_arcmin": CASSBEAM_RR_PEAK_ARCMIN,
    "cassbeam_ll_peak_arcmin": CASSBEAM_LL_PEAK_ARCMIN,
    "nominal_first_sidelobe_arcmin": NOMINAL_FIRST_SIDELOBE_ARCMIN,
    "sidelobe_bump_width_arcmin": SIDELOBE_BUMP_WIDTH_ARCMIN,
    "azimuthal_core_arcmin": AZIMUTHAL_CORE_ARCMIN,
    "envelope_positive_floor": ENVELOPE_POSITIVE_FLOOR,
    "squint_scale_grid": tuple(float(item) for item in SQUINT_SCALE_GRID),
    "width_scale_grid": tuple(float(item) for item in WIDTH_SCALE_GRID),
    "pointing_arcmin_grid": tuple(float(item) for item in POINTING_ARCMIN_GRID),
    "sidelobe_radius_grid": tuple(float(item) for item in SIDELOBE_RADIUS_GRID),
    "sidelobe_amplitude_grid": tuple(float(item) for item in SIDELOBE_AMPLITUDE_GRID),
    "paired_delta_bootstrap": PAIRED_DELTA_BOOTSTRAP,
    "bootstrap_seed": 0,
}


@dataclass(frozen=True)
class CorrectionSamples:
    """Frozen HOLORASTER field-10 / SPW-4 / channel-32 correction rows."""

    offset_lm_rad: NDArray[np.float64]
    measured: NDArray[np.complex128]
    baseline: NDArray[np.complex128]
    weight: NDArray[np.float64]
    source: NDArray[np.complex128]
    moving_is_p: NDArray[np.bool_]
    moving_id: NDArray[np.int32]
    reference_id: NDArray[np.int32]
    residual_jones: Mapping[int, NDArray[np.complex128]]
    parallactic_angle_rad: NDArray[np.float64]
    reference_parallactic_angle_rad: NDArray[np.float64]
    antenna_names: tuple[str, ...]
    train: NDArray[np.bool_]
    spatial_holdout: NDArray[np.bool_]
    mover_holdout: NDArray[np.bool_]
    main_lobe: NDArray[np.bool_]
    mid: NDArray[np.bool_]
    outer: NDArray[np.bool_]
    field_id: NDArray[np.int32]
    frequency_hz: float = SPW4_TRAINING_FREQUENCY_HZ
    spectral_window_id: int = 4

    def __post_init__(self) -> None:
        n = _offset_pairs(self.offset_lm_rad).shape[0]
        object.__setattr__(self, "offset_lm_rad", _offset_pairs(self.offset_lm_rad))
        object.__setattr__(
            self, "measured", _one_channel_vis(self.measured, "measured", np.complex128)
        )
        object.__setattr__(
            self, "baseline", _one_channel_vis(self.baseline, "baseline", np.complex128)
        )
        object.__setattr__(self, "weight", _one_channel_vis(self.weight, "weight", np.float64))
        if self.measured.shape[0] != n or self.baseline.shape[0] != n or self.weight.shape[0] != n:
            raise ValueError("visibility planes must have one entry per sample")
        source = np.asarray(self.source, dtype=np.complex128)
        if source.shape == (2, 2):
            pass
        elif source.shape == (n, 2, 2) or source.shape == (n, 1, 2, 2):
            if source.ndim == 4:
                source = source[:, 0]
        else:
            raise ValueError("source must be (2, 2) or one coherency per sample")
        object.__setattr__(self, "source", source)
        for name in ("moving_id", "reference_id"):
            value = np.asarray(getattr(self, name), dtype=np.int32).reshape(-1)
            if value.size != n:
                raise ValueError(f"{name} must have one entry per sample")
            object.__setattr__(self, name, value)
        moving_chi = np.asarray(self.parallactic_angle_rad, dtype=np.float64).reshape(-1)
        reference_chi = np.asarray(self.reference_parallactic_angle_rad, dtype=np.float64).reshape(
            -1
        )
        if moving_chi.size != n or reference_chi.size != n:
            raise ValueError("parallactic angles must have one entry per sample")
        object.__setattr__(self, "parallactic_angle_rad", moving_chi)
        object.__setattr__(self, "reference_parallactic_angle_rad", reference_chi)
        residuals = {
            int(antenna): np.asarray(jones, dtype=np.complex128)
            for antenna, jones in dict(self.residual_jones).items()
        }
        needed = set(int(item) for item in self.moving_id) | set(
            int(item) for item in self.reference_id
        )
        missing = needed - set(residuals)
        if missing:
            raise ValueError(f"residual_jones is missing antennas {sorted(missing)}")
        for antenna, jones in residuals.items():
            if jones.shape != (2, 2):
                raise ValueError(f"residual Jones for antenna {antenna} must be 2×2")
        object.__setattr__(self, "residual_jones", residuals)
        object.__setattr__(self, "moving_is_p", _boolean_mask(self.moving_is_p, n, "moving_is_p"))
        for name in (
            "train",
            "spatial_holdout",
            "mover_holdout",
            "main_lobe",
            "mid",
            "outer",
        ):
            object.__setattr__(self, name, _boolean_mask(getattr(self, name), n, name))
        if not (
            bool(np.any(self.train))
            and bool(np.any(self.spatial_holdout))
            and bool(np.any(self.mover_holdout))
        ):
            raise ValueError("train, spatial holdout, and mover holdout must be non-empty")
        if bool(np.any(self.train & self.spatial_holdout | self.train & self.mover_holdout)):
            raise ValueError("train and holdout masks must be disjoint")
        if bool(np.any(self.spatial_holdout & self.mover_holdout)):
            raise ValueError("spatial and mover holdouts must be disjoint")
        object.__setattr__(self, "antenna_names", tuple(self.antenna_names))
        fields = np.asarray(self.field_id, dtype=np.int32).reshape(-1)
        if fields.size != n:
            raise ValueError("field_id must have one entry per sample")
        object.__setattr__(self, "field_id", fields)
        refuse_c147_training(fields[self.development_mask()])
        if np.any(fields != HOLORASTER_FIELD_ID):
            raise ValueError("correction samples must be HOLORASTER field 10")
        refuse_spw5(
            frequency_hz=self.frequency_hz,
            spectral_window_id=int(self.spectral_window_id),
        )
        if int(self.spectral_window_id) != 4:
            raise ValueError("correction samples must be SPW 4")
        if not np.isclose(
            float(self.frequency_hz), SPW4_TRAINING_FREQUENCY_HZ, rtol=0.0, atol=0.5e6
        ):
            raise ValueError("correction samples must be THOL0001 SPW 4 channel 32")

    def development_mask(self) -> NDArray[np.bool_]:
        return np.asarray(self.train | self.spatial_holdout | self.mover_holdout, dtype=bool)


@dataclass(frozen=True)
class TermEvaluation:
    term: str
    state: CorrectionState
    spatial: PairedLossDifference
    moving: PairedLossDifference
    mover_units: MoverPairedDeltas
    decision: str
    region_residual_power: Mapping[str, float]
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class LadderResult:
    accepted: CorrectionState
    stopped_at: str | None
    identity_matches_baseline: bool
    selection_record: tuple[TermEvaluation, ...]
    development_refit: CorrectionState | None
    holdout_scores_preserved: bool
    hyperparameters: Mapping[str, object] = field(
        default_factory=lambda: dict(FROZEN_HYPERPARAMETERS)
    )
    notes: tuple[str, ...] = (IDENTITY_STATE_NOTE, WARP_NOTE)


def correction_state_to_dict(state: CorrectionState) -> dict[str, object]:
    return {
        "accepted_terms": list(state.accepted_terms),
        "squint_scale": float(state.squint_scale),
        "width_scale": float(state.width_scale),
        "pointing_l_rad": float(state.pointing_l_rad),
        "pointing_m_rad": float(state.pointing_m_rad),
        "sidelobe_radius_scale": float(state.sidelobe_radius_scale),
        "sidelobe_amplitude": float(state.sidelobe_amplitude),
        "azimuthal": [float(item) for item in state.azimuthal],
        "is_identity": state.is_identity(),
    }


def paired_delta_to_dict(delta: PairedLossDifference) -> dict[str, object]:
    return {
        "axis": delta.axis,
        "cluster_kind": delta.cluster_kind,
        "delta": float(delta.delta),
        "delta_lo": float(delta.delta_lo),
        "delta_hi": float(delta.delta_hi),
        "n_clusters": int(delta.n_clusters),
        "n_boot": int(delta.n_boot),
        "improves": bool(delta.improves()),
    }


def mover_units_to_dict(units: MoverPairedDeltas) -> dict[str, object]:
    return {
        "names": list(units.names),
        "delta": [float(item) for item in units.delta],
        "mainlobe_delta": [float(item) for item in units.mainlobe_delta],
        "mainlobe_baseline": [float(item) for item in units.mainlobe_baseline],
        "n_improving": int(units.n_improving),
        "n_material_mainlobe_regression": int(units.n_material_mainlobe_regression),
        "finite": bool(np.all(np.isfinite(units.delta))),
        "passes": bool(units.passes()),
    }


def ladder_result_to_dict(result: LadderResult) -> dict[str, object]:
    return {
        "accepted": correction_state_to_dict(result.accepted),
        "stopped_at": result.stopped_at,
        "identity_matches_baseline": bool(result.identity_matches_baseline),
        "holdout_scores_preserved": bool(result.holdout_scores_preserved),
        "development_refit": (
            None
            if result.development_refit is None
            else correction_state_to_dict(result.development_refit)
        ),
        "hyperparameters": dict(result.hyperparameters),
        "notes": list(result.notes),
        "selection_record": [
            {
                "term": item.term,
                "decision": item.decision,
                "state": correction_state_to_dict(item.state),
                "spatial": paired_delta_to_dict(item.spatial),
                "moving": paired_delta_to_dict(item.moving),
                "mover_units": mover_units_to_dict(item.mover_units),
                "region_residual_power": dict(item.region_residual_power),
                "notes": list(item.notes),
            }
            for item in result.selection_record
        ],
    }


def correction_path_stages(
    offset_lm_rad: ArrayLike,
    lookup: FeedFrameLookup,
    state: CorrectionState,
    *,
    residual_jones: Mapping[int, ArrayLike],
    moving_id: ArrayLike,
    reference_id: ArrayLike,
    moving_is_p: ArrayLike,
    chi_moving: ArrayLike,
    chi_reference: ArrayLike,
    source: ArrayLike,
) -> dict[str, NDArray[np.complex128]]:
    """Feed, sky, and :math:`R_m E_m S R_r^H` from the correction path."""

    feed = apply_feed_frame_correction(offset_lm_rad, lookup, state)
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
            raise ValueError("correction predictions must stay on one native channel")
        vis = vis[:, 0]
    return {"feed": feed, "sky": sky, "visibility": vis}


def predict_from_state(
    samples: CorrectionSamples,
    lookup: FeedFrameLookup,
    state: CorrectionState,
) -> NDArray[np.complex128]:
    """Correct the moving beam, then apply the frozen two-antenna RIME."""

    return correction_path_stages(
        samples.offset_lm_rad,
        lookup,
        state,
        residual_jones=samples.residual_jones,
        moving_id=samples.moving_id,
        reference_id=samples.reference_id,
        moving_is_p=samples.moving_is_p,
        chi_moving=samples.parallactic_angle_rad,
        chi_reference=samples.reference_parallactic_angle_rad,
        source=samples.source,
    )["visibility"]


def region_residual_power(
    measured: ArrayLike,
    predicted: ArrayLike,
    weight: ArrayLike,
    masks: Mapping[str, ArrayLike],
) -> dict[str, float]:
    """Main/mid/outer residual power. Diagnostic; not the acceptance rule."""

    out: dict[str, float] = {}
    for name, mask in masks.items():
        choose = np.asarray(mask, dtype=bool).reshape(-1)
        if not bool(np.any(choose)):
            out[name] = float("nan")
            continue
        out[name] = complex_visibility_loss(
            np.asarray(measured)[choose],
            np.asarray(predicted)[choose],
            np.asarray(weight)[choose],
        )
    return out


def evaluate_candidate(
    samples: CorrectionSamples,
    candidate: ArrayLike,
    baseline: ArrayLike,
    *,
    term: str,
    state: CorrectionState,
    n_boot: int = PAIRED_DELTA_BOOTSTRAP,
    seed: int = 0,
) -> TermEvaluation:
    spatial = score_paired_holdout(
        samples.measured[samples.spatial_holdout],
        np.asarray(candidate)[samples.spatial_holdout],
        np.asarray(baseline)[samples.spatial_holdout],
        samples.weight[samples.spatial_holdout],
        spatial_cluster_ids(samples.offset_lm_rad, samples.spatial_holdout),
        axis="spatial",
        cluster_kind="spatial_cell",
        n_boot=n_boot,
        seed=seed,
    )
    moving = score_paired_holdout(
        samples.measured[samples.mover_holdout],
        np.asarray(candidate)[samples.mover_holdout],
        np.asarray(baseline)[samples.mover_holdout],
        samples.weight[samples.mover_holdout],
        mover_cluster_ids(samples.moving_id, samples.mover_holdout),
        axis="moving",
        cluster_kind="moving_antenna",
        n_boot=n_boot,
        seed=seed + 1,
    )
    units = mover_paired_deltas(
        samples.measured[samples.mover_holdout],
        np.asarray(candidate)[samples.mover_holdout],
        np.asarray(baseline)[samples.mover_holdout],
        samples.weight[samples.mover_holdout],
        samples.moving_id[samples.mover_holdout],
        samples.antenna_names,
        mainlobe_mask=samples.main_lobe[samples.mover_holdout],
    )
    decision = select_nested_correction(spatial, moving, units)
    regions = region_residual_power(
        samples.measured,
        candidate,
        samples.weight,
        {
            "main_lobe": samples.main_lobe,
            "mid": samples.mid,
            "outer_diagnostic": samples.outer,
        },
    )
    return TermEvaluation(
        term=term,
        state=state,
        spatial=spatial,
        moving=moving,
        mover_units=units,
        decision=decision,
        region_residual_power=regions,
        notes=("region residual power is diagnostic",),
    )


def _train_loss(
    samples: CorrectionSamples,
    lookup: FeedFrameLookup,
    state: CorrectionState,
    row_mask: ArrayLike,
) -> float:
    choose = np.asarray(row_mask, dtype=bool).reshape(-1)
    predicted = predict_from_state(samples, lookup, state)
    return complex_visibility_loss(
        samples.measured[choose],
        predicted[choose],
        samples.weight[choose],
    )


def _best_on_grid(
    samples: CorrectionSamples,
    lookup: FeedFrameLookup,
    prefix: CorrectionState,
    updates: Sequence[Mapping[str, float | tuple[float, ...]]],
    row_mask: ArrayLike,
) -> Mapping[str, float | tuple[float, ...]]:
    term = next_term(prefix)
    if term is None:
        raise ValueError("no remaining first-ladder term to evaluate")
    best = updates[0]
    best_loss = np.inf
    for item in updates:
        loss = _train_loss(
            samples,
            lookup,
            with_accepted_term(prefix, term, **dict(item)),
            row_mask,
        )
        if np.isfinite(loss) and loss < best_loss:
            best_loss = float(loss)
            best = item
    return best


def _fit_azimuthal(
    samples: CorrectionSamples,
    prefix_vis: ArrayLike,
    row_mask: ArrayLike,
    *,
    sidelobe_amplitude: float = 1.0,
) -> tuple[float, float, float, float]:
    choose = np.asarray(row_mask, dtype=bool).reshape(-1)
    offset = samples.offset_lm_rad[choose]
    radius = np.hypot(offset[:, 0], offset[:, 1])
    angle = np.arctan2(offset[:, 0], offset[:, 1])
    window = 1.0 - np.exp(-((radius / (AZIMUTHAL_CORE_ARCMIN * ARCMIN_TO_RAD)) ** 2))
    design = np.stack(
        [
            window * np.cos(angle),
            window * np.sin(angle),
            window * np.cos(2.0 * angle),
            window * np.sin(2.0 * angle),
        ],
        axis=1,
    )
    pred = np.asarray(prefix_vis, dtype=np.complex128)[choose]
    meas = samples.measured[choose]
    wgt = samples.weight[choose]
    columns = []
    rhs = []
    for row, col in ((0, 0), (1, 1)):
        prefix = pred[:, row, col]
        ww = np.sqrt(np.maximum(wgt[:, row, col], 0.0))
        finite = np.isfinite(prefix) & np.isfinite(meas[:, row, col]) & (ww > 0.0)
        if not bool(np.any(finite)):
            continue
        basis = prefix[finite, None] * design[finite]
        target = meas[finite, row, col] - prefix[finite]
        scale = ww[finite]
        columns.append(np.vstack((scale[:, None] * basis.real, scale[:, None] * basis.imag)))
        rhs.append(np.concatenate((scale * target.real, scale * target.imag)))
    if not columns:
        return (0.0, 0.0, 0.0, 0.0)
    matrix = np.vstack(columns)
    vector = np.concatenate(rhs)
    coef, *_ = np.linalg.lstsq(matrix, vector, rcond=None)
    return shrink_azimuthal_coefficients(
        samples.offset_lm_rad[choose],
        (float(coef[0]), float(coef[1]), float(coef[2]), float(coef[3])),
        sidelobe_amplitude=float(sidelobe_amplitude),
    )


def fit_term(
    term: str,
    prefix: CorrectionState,
    samples: CorrectionSamples,
    lookup: FeedFrameLookup,
    *,
    row_mask: ArrayLike | None = None,
) -> CorrectionState:
    """Fit one nested term on the supplied rows. Does not score holdouts."""

    refuse_phase_in_first_ladder(term)
    mask = samples.train if row_mask is None else np.asarray(row_mask, dtype=bool)

    def finish(state: CorrectionState) -> CorrectionState:
        require_positive_envelope(samples.offset_lm_rad, state)
        return state

    if term == "rl_squint_scale":
        chosen = _best_on_grid(
            samples,
            lookup,
            prefix,
            tuple({"squint_scale": float(value)} for value in SQUINT_SCALE_GRID),
            mask,
        )
        return finish(with_accepted_term(prefix, term, squint_scale=float(chosen["squint_scale"])))
    if term == "beam_width":
        chosen = _best_on_grid(
            samples,
            lookup,
            prefix,
            tuple({"width_scale": float(value)} for value in WIDTH_SCALE_GRID),
            mask,
        )
        return finish(with_accepted_term(prefix, term, width_scale=float(chosen["width_scale"])))
    if term == "pointing_offset":
        updates = tuple(
            {
                "pointing_l_rad": float(l_arcmin) * ARCMIN_TO_RAD,
                "pointing_m_rad": float(m_arcmin) * ARCMIN_TO_RAD,
            }
            for l_arcmin in POINTING_ARCMIN_GRID
            for m_arcmin in POINTING_ARCMIN_GRID
        )
        chosen = _best_on_grid(samples, lookup, prefix, updates, mask)
        return finish(with_accepted_term(prefix, term, **chosen))
    if term == "first_sidelobe_radius_amplitude":
        updates = tuple(
            {
                "sidelobe_radius_scale": float(radius),
                "sidelobe_amplitude": float(amplitude),
            }
            for radius in SIDELOBE_RADIUS_GRID
            for amplitude in SIDELOBE_AMPLITUDE_GRID
        )
        chosen = _best_on_grid(samples, lookup, prefix, updates, mask)
        return finish(with_accepted_term(prefix, term, **chosen))
    if term == "low_order_azimuthal":
        prefix_vis = predict_from_state(samples, lookup, prefix)
        azimuthal = _fit_azimuthal(
            samples,
            prefix_vis,
            mask,
            sidelobe_amplitude=prefix.sidelobe_amplitude,
        )
        return finish(with_accepted_term(prefix, term, azimuthal=azimuthal))
    raise ValueError(f"unsupported first-ladder term {term!r}")


def refit_frozen_family(
    accepted: CorrectionState,
    samples: CorrectionSamples,
    lookup: FeedFrameLookup,
) -> CorrectionState:
    """Refit accepted coefficients on all SPW-4 development rows."""

    if accepted.is_identity():
        return accepted
    state = IDENTITY_CORRECTION
    development = samples.development_mask()
    for term in accepted.accepted_terms:
        if term == "identity":
            continue
        state = fit_term(term, state, samples, lookup, row_mask=development)
    if state.accepted_terms != accepted.accepted_terms:
        raise ValueError("development refit must preserve the frozen term prefix")
    return state


def run_first_ladder(
    samples: CorrectionSamples,
    lookup: FeedFrameLookup,
    *,
    n_boot: int = PAIRED_DELTA_BOOTSTRAP,
    seed: int = 0,
    refit_development: bool = True,
) -> LadderResult:
    """Add nested terms until the first failed holdout gate. SPW 5 stays closed."""

    refuse_spw5(
        frequency_hz=samples.frequency_hz,
        spectral_window_id=samples.spectral_window_id,
    )
    identity_vis = identity_predictions(samples.baseline)
    require_identity_matches_baseline(identity_vis, samples.baseline)
    model_identity = predict_from_state(samples, lookup, IDENTITY_CORRECTION)
    require_identity_matches_baseline(model_identity, samples.baseline)
    accepted = IDENTITY_CORRECTION
    baseline_vis = samples.baseline
    record: list[TermEvaluation] = []
    stopped: str | None = None
    while True:
        term = next_term(accepted)
        if term is None:
            break
        candidate = fit_term(term, accepted, samples, lookup)
        candidate_vis = predict_from_state(samples, lookup, candidate)
        scored = evaluate_candidate(
            samples,
            candidate_vis,
            baseline_vis,
            term=term,
            state=candidate,
            n_boot=n_boot,
            seed=seed,
        )
        record.append(scored)
        if scored.decision != "accept_candidate":
            stopped = term
            break
        accepted = candidate
        baseline_vis = candidate_vis
    development = (
        refit_frozen_family(accepted, samples, lookup) if refit_development else None
    )
    hyperparameters = dict(FROZEN_HYPERPARAMETERS)
    hyperparameters["paired_delta_bootstrap"] = int(n_boot)
    hyperparameters["bootstrap_seed"] = int(seed)
    return LadderResult(
        accepted=accepted,
        stopped_at=stopped,
        identity_matches_baseline=True,
        selection_record=tuple(record),
        development_refit=development,
        holdout_scores_preserved=True,
        hyperparameters=hyperparameters,
    )


def region_masks_from_voltage(voltage: ArrayLike) -> dict[str, NDArray[np.bool_]]:
    """Publication main/mid/outer bins. Acceptance still uses the holdouts."""

    return voltage_response_region_masks(voltage)
