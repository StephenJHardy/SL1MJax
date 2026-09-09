"""High-resolution CASSBEAM as a coherent visibility-domain template.

Bypasses the empirical per-cell leak estimator. Loads the external
513×513 artifact, locks receive/transmit conventions on training data
only, and scores

    t = V_full - V_diagonal,  r = V_observed - V_diagonal

on sealed LORO holdouts. Production full Jones stays unfrozen. SPW 5
stays closed. The committed 33×33 factory is never opened.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sl1mjax.beam_conventions import BeamCalibrationState, require_beam_calibration_state
from sl1mjax.beam_operator import unique_visibility_times
from sl1mjax.cassbeam_highres import (
    HighresCassbeamCatalog,
    HighresCassbeamPlane,
    diagonal_projection,
)
from sl1mjax.holography import HolographyObservation
from sl1mjax.holography_diagonal import _source_coherency
from sl1mjax.holography_full_jones import (
    _antenna_jones_planes,
    _predict_vis_jax,
    _sky_frame_residual_jax,
    holography_one_axis_holdout_masks,
    moving_reference_row_geometry,
    score_full_versus_diagonal_regions,
    thin_training_rows,
)
from sl1mjax.polarization import (
    Receptor,
    circular_parallactic_jones,
    circular_stokes_to_coherency,
    invert_jones,
    pack_coherency,
)

HIGHRES_CASSBEAM_DIRECT = "highres_cassbeam_direct_visibility_validation"
TEMPLATE_INJECTION_AMPLITUDES = (0.0, 0.25, 0.5, 1.0, 2.0)
QU_NUISANCE_PAIRS = (
    (0.0, 0.0),
    (-0.001, 0.0),
    (0.001, 0.0),
    (0.0, -0.001),
    (0.0, 0.001),
    (-0.003, 0.0),
    (0.003, 0.0),
    (0.0, -0.003),
    (0.0, 0.003),
    (-0.001, -0.001),
    (-0.001, 0.001),
    (0.001, -0.001),
    (0.001, 0.001),
)
INTERPOLATION_FLOOR = 0.002
MAIN_LOBE_VOLTAGE = 0.5
CORRELATION_MARGIN = 0.02
MIN_CONVENTION_ROWS = 200
MIN_TEMPLATE_POWER = 1.0e-12

HIGHRES_CASSBEAM_NOTE = (
    "The empirical per-cell leak estimator is closed at present SPW-4 "
    "sensitivity. This experiment asks whether the spatially and spectrally "
    "coherent high-resolution CASSBEAM leak is present as a matched "
    "visibility-domain template."
)
TRAINING_ONLY_CONVENTION_NOTE = (
    "Receive/transmit, axis, and R/L conventions are locked on training "
    "rows only. Held-out RL/LR do not select the convention."
)
TEMPLATE_ALPHA_NOTE = (
    "α=0 is the diagonal hypothesis. α=1 is the exact CASSBEAM prediction. "
    "A single complex α is fitted on training rows and evaluated unchanged "
    "on holdout. Consistent with zero: no CASSBEAM-shaped leakage. "
    "Consistent with one: amplitude and phase supported. Significantly "
    "nonzero but inconsistent with one: morphology supported, scale or "
    "phase needs correction."
)
NO_GLOBAL_FIVE_PERCENT_NOTE = (
    "A 0.34% template against a 2.6% residual floor need not produce a "
    "5% drop in the global residual. Score the matched template, not a "
    "total-loss threshold."
)
CHANNEL_COVARIANCE_NOTE = (
    "Joint frequency evidence uses channel covariance or a channel-block "
    "bootstrap. Correlated calibration errors do not give an automatic "
    "sqrt(N) improvement."
)


@dataclass(frozen=True)
class CassbeamConvention:
    """One physically motivated CASSBEAM receive/transmit convention."""

    l_sign: int
    m_sign: int
    swap_lm: bool
    jones: str
    swap_rl: bool
    rotate_spatial: bool = False

    @property
    def name(self) -> str:
        axis = f"l{self.l_sign:+d}_m{self.m_sign:+d}"
        if self.swap_lm:
            axis += "_swap"
        frame = "skyrot" if self.rotate_spatial else "mount"
        hands = "swap_rl" if self.swap_rl else "native_rl"
        return f"{axis}_{self.jones}_{hands}_{frame}"


DEFAULT_CONVENTION = CassbeamConvention(1, 1, False, "native", False, False)


def convention_ladder() -> tuple[CassbeamConvention, ...]:
    """Finite axis, Jones, R/L, and spatial-frame candidates. Training only."""

    axes = tuple(
        (l_sign, m_sign, swap_lm)
        for swap_lm in (False, True)
        for l_sign in (-1, 1)
        for m_sign in (-1, 1)
    )
    jones = ("native", "conjugate", "transpose", "hermitian")
    return tuple(
        CassbeamConvention(l_sign, m_sign, swap_lm, jones_name, swap_rl, rotate)
        for rotate in (False, True)
        for l_sign, m_sign, swap_lm in axes
        for jones_name in jones
        for swap_rl in (False, True)
    )


def apply_axis_convention(
    offset_lm_rad: ArrayLike,
    convention: CassbeamConvention,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    offset = np.asarray(offset_lm_rad, dtype=np.float64).reshape(-1, 2)
    l_sky = float(convention.l_sign) * offset[:, 0]
    m_sky = float(convention.m_sign) * offset[:, 1]
    if convention.swap_lm:
        return m_sky, l_sky
    return l_sky, m_sky


def apply_jones_convention(
    jones: ArrayLike,
    convention: CassbeamConvention,
) -> NDArray[np.complex128]:
    plane = np.asarray(jones, dtype=np.complex128)
    if convention.jones == "conjugate":
        plane = np.conjugate(plane)
    elif convention.jones == "transpose":
        plane = np.swapaxes(plane, -1, -2)
    elif convention.jones == "hermitian":
        plane = np.conjugate(np.swapaxes(plane, -1, -2))
    elif convention.jones != "native":
        raise ValueError(f"unknown Jones convention {convention.jones!r}")
    if convention.swap_rl:
        plane = plane[..., ::-1, ::-1]
    return np.asarray(plane, dtype=np.complex128)


def normalize_after_jones_convention(
    native: ArrayLike,
    origin_native: ArrayLike,
    convention: CassbeamConvention,
) -> NDArray[np.complex128]:
    """Transform native Jones, then apply :math:`E(0)^{-1}E(s)`.

    Transpose and Hermitian reverse multiplication order, so they are not
    applied to an already-normalized matrix.
    """

    transformed = apply_jones_convention(native, convention)
    origin = apply_jones_convention(origin_native, convention)
    inverse = invert_jones(origin)
    return np.asarray(np.einsum("ij,...jk->...ik", inverse, transformed), dtype=np.complex128)


def antenna_frame_lm(
    l_rad: ArrayLike,
    m_rad: ArrayLike,
    chi: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    l_off = np.asarray(l_rad, dtype=np.float64).reshape(-1)
    m_off = np.asarray(m_rad, dtype=np.float64).reshape(-1)
    angle = np.asarray(chi, dtype=np.float64).reshape(-1)
    cosine = np.cos(angle)
    sine = np.sin(angle)
    return l_off * cosine + m_off * sine, -l_off * sine + m_off * cosine


def evaluate_highres_cassbeam_jones(
    offset_lm_rad: ArrayLike,
    frequency_hz: ArrayLike,
    parallactic_angle_rad: ArrayLike,
    *,
    catalog: HighresCassbeamCatalog,
    convention: CassbeamConvention,
    off_diagonal: bool,
    calibration_state: BeamCalibrationState | str = "casa_parang_true",
) -> tuple[NDArray[np.complex128], NDArray[np.bool_]]:
    """Exact-frequency high-res lookup. No nearest-node substitution.

    Query coordinates must already be source-in-beam CASSBEAM ``(l, m)``.
    Commanded AZELGEO ``POINTING_OFFSET`` is not that coordinate.
    Spatial parallactic rotation is applied only when
    ``convention.rotate_spatial`` is true. Jones-basis parallactic
    rotation is applied afterwards.
    """

    state = require_beam_calibration_state(calibration_state)
    offset = np.asarray(offset_lm_rad, dtype=np.float64).reshape(-1, 2)
    frequency = np.asarray(frequency_hz, dtype=np.float64).reshape(-1)
    chi = np.asarray(parallactic_angle_rad, dtype=np.float64).reshape(-1)
    if frequency.size == 1:
        frequency = np.full(offset.shape[0], float(frequency[0]), dtype=np.float64)
    if chi.size == 1:
        chi = np.full(offset.shape[0], float(chi[0]), dtype=np.float64)
    if frequency.size != offset.shape[0] or chi.size != offset.shape[0]:
        raise ValueError("offset, frequency_hz, and parallactic_angle_rad must match")
    l_q, m_q = apply_axis_convention(offset, convention)
    if convention.rotate_spatial:
        l_q, m_q = antenna_frame_lm(l_q, m_q, chi)
    jones = np.zeros((offset.shape[0], 2, 2), dtype=np.complex128)
    valid = np.zeros(offset.shape[0], dtype=bool)
    mhz_keys = np.asarray([catalog.require_exact_mhz(float(freq)) for freq in frequency])
    for mhz in np.unique(mhz_keys):
        members = np.flatnonzero(mhz_keys == mhz)
        plane = catalog.plane(float(mhz) * 1.0e6)
        native, ok = plane.lookup_native(l_q[members], m_q[members])
        looked = normalize_after_jones_convention(native, plane.origin_native(), convention)
        if not off_diagonal:
            looked = diagonal_projection(looked)
        para = circular_parallactic_jones(chi[members])
        if state is BeamCalibrationState.CASA_PARANG_TRUE:
            conjugate = np.conjugate(np.swapaxes(para, -1, -2))
            looked = conjugate @ looked @ para
        elif state is BeamCalibrationState.UNCALIBRATED:
            looked = looked @ para
        else:
            raise ValueError(f"unsupported beam calibration state {state!r}")
        jones[members] = looked
        valid[members] = ok
    return jones, valid


def predict_moving_reference_from_highres_cassbeam(
    observation: HolographyObservation,
    *,
    residual_jones: Mapping[int, ArrayLike],
    catalog: HighresCassbeamCatalog,
    convention: CassbeamConvention,
    row_mask: ArrayLike,
    parallactic_angle_rad: ArrayLike | None = None,
    off_diagonal: bool,
    source_coherency: ArrayLike | None = None,
) -> NDArray[np.complex128]:
    """Predict :math:`V=R_p E_p(s) S E_q(0)^H R_q^H` from the high-res table."""

    import jax.numpy as jnp

    mask = np.asarray(row_mask, dtype=bool).reshape(-1)
    source = (
        np.asarray(_source_coherency(observation), dtype=np.complex128)
        if source_coherency is None
        else np.asarray(source_coherency, dtype=np.complex128)
    )
    packed = pack_coherency(
        observation.block.visibility, observation.block.correlations, (Receptor.R, Receptor.L)
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
    unique_times, _ = unique_visibility_times(observation.block.time_s)
    if parallactic_angle_rad is None:
        from sl1mjax.calibration_terms import parallactic_angle_rad as _chi

        chi_table = _chi(
            unique_times,
            observation.phase_centre_rad,
            np.asarray(observation.antenna_position_m, dtype=np.float64),
        )
    else:
        chi_table = np.asarray(parallactic_angle_rad, dtype=np.float64)
    chi = chi_table[time_index, moving]
    query = np.repeat(offset, n_chan, axis=0)
    freqs = np.broadcast_to(observation.block.frequency_hz, (moving.size, n_chan)).reshape(-1)
    chi_f = np.repeat(chi, n_chan)
    beams, ok = evaluate_highres_cassbeam_jones(
        query,
        freqs,
        chi_f,
        catalog=catalog,
        convention=convention,
        off_diagonal=off_diagonal,
        calibration_state=observation.calibration_state,
    )
    beams = beams.reshape(moving.size, n_chan, 2, 2)
    ok = ok.reshape(moving.size, n_chan)
    r_m = _antenna_jones_planes(residual_jones, moving)
    r_r = _antenna_jones_planes(residual_jones, reference)
    r_m = np.asarray(_sky_frame_residual_jax(jnp.asarray(r_m), jnp.asarray(chi)))
    r_r = np.asarray(
        _sky_frame_residual_jax(jnp.asarray(r_r), jnp.asarray(chi_table[time_index, reference]))
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


def _plane2(values: ArrayLike) -> NDArray[np.complex128]:
    array = np.asarray(values, dtype=np.complex128)
    if array.ndim == 4:
        return array[:, 0]
    return array


def _hand_weight(observation: HolographyObservation, row_mask: ArrayLike) -> NDArray[np.float64]:
    mask = np.asarray(row_mask, dtype=bool).reshape(-1)
    names = tuple(
        item.value if hasattr(item, "value") else str(item)
        for item in observation.block.correlations
    )
    weight = np.asarray(observation.block.weight, dtype=np.float64)
    flag = np.asarray(observation.block.flag, dtype=bool)
    if weight.ndim == 3:
        weight = weight[:, 0]
        flag = flag[:, 0]
    active = (~flag) & np.isfinite(weight) & (weight > 0.0)
    out = np.zeros((int(np.sum(mask)), 2, 2), dtype=np.float64)
    index = {name: i for i, name in enumerate(names)}
    selected = active[mask]
    chosen = weight[mask]
    out[:, 0, 0] = np.where(selected[:, index["RR"]], chosen[:, index["RR"]], 0.0)
    out[:, 0, 1] = np.where(selected[:, index["RL"]], chosen[:, index["RL"]], 0.0)
    out[:, 1, 0] = np.where(selected[:, index["LR"]], chosen[:, index["LR"]], 0.0)
    out[:, 1, 1] = np.where(selected[:, index["LL"]], chosen[:, index["LL"]], 0.0)
    return out


def fit_complex_template_alpha(
    template: ArrayLike,
    residual: ArrayLike,
    weight: ArrayLike,
    *,
    hands: tuple[str, ...] = ("rl", "lr"),
) -> complex:
    """Weighted complex least-squares :math:`α` on selected hands."""

    t_all = _plane2(template)
    r_all = _plane2(residual)
    w_all = np.asarray(weight, dtype=np.float64)
    if w_all.ndim == 4:
        w_all = w_all[:, 0]
    elements = {"rr": (0, 0), "rl": (0, 1), "lr": (1, 0), "ll": (1, 1)}
    numerator = 0.0 + 0.0j
    denominator = 0.0
    for name in hands:
        row, col = elements[name]
        t = t_all[:, row, col]
        r = r_all[:, row, col]
        w = w_all[:, row, col]
        finite = np.isfinite(t) & np.isfinite(r) & np.isfinite(w) & (w > 0.0)
        if not bool(np.any(finite)):
            continue
        numerator += np.sum(w[finite] * np.conjugate(t[finite]) * r[finite])
        denominator += np.sum(w[finite] * np.abs(t[finite]) ** 2)
    if denominator <= MIN_TEMPLATE_POWER:
        return complex(np.nan, np.nan)
    return complex(numerator / denominator)


def template_correlation(
    template: ArrayLike,
    residual: ArrayLike,
    weight: ArrayLike,
    *,
    hands: tuple[str, ...] = ("rl", "lr"),
) -> float:
    t_all = _plane2(template)
    r_all = _plane2(residual)
    w_all = np.asarray(weight, dtype=np.float64)
    if w_all.ndim == 4:
        w_all = w_all[:, 0]
    elements = {"rr": (0, 0), "rl": (0, 1), "lr": (1, 0), "ll": (1, 1)}
    tr = 0.0
    tt = 0.0
    rr = 0.0
    for name in hands:
        row, col = elements[name]
        t = t_all[:, row, col]
        r = r_all[:, row, col]
        w = w_all[:, row, col]
        finite = np.isfinite(t) & np.isfinite(r) & np.isfinite(w) & (w > 0.0)
        if not bool(np.any(finite)):
            continue
        tr += float(np.real(np.sum(w[finite] * np.conjugate(t[finite]) * r[finite])))
        tt += float(np.sum(w[finite] * np.abs(t[finite]) ** 2))
        rr += float(np.sum(w[finite] * np.abs(r[finite]) ** 2))
    if tt <= MIN_TEMPLATE_POWER or rr <= MIN_TEMPLATE_POWER:
        return float("nan")
    return tr / np.sqrt(tt * rr)


def weighted_residual_power(
    measured: ArrayLike,
    predicted: ArrayLike,
    weight: ArrayLike,
    *,
    stokes_i: ArrayLike | None = None,
) -> dict[str, dict[str, float]]:
    """Weighted complex residual power, Jy residuals, and residuals over I."""

    vis = _plane2(measured)
    pred = _plane2(predicted)
    w_all = np.asarray(weight, dtype=np.float64)
    if w_all.ndim == 4:
        w_all = w_all[:, 0]
    intensity = None if stokes_i is None else np.asarray(stokes_i, dtype=np.float64).reshape(-1)
    residual = vis - pred
    hands = {"rr": (0, 0), "ll": (1, 1), "rl": (0, 1), "lr": (1, 0)}
    report: dict[str, dict[str, float]] = {}
    for name, (row, col) in hands.items():
        z = residual[:, row, col]
        w = w_all[:, row, col]
        finite = np.isfinite(z) & np.isfinite(w) & (w > 0.0)
        if intensity is not None:
            finite &= np.isfinite(intensity)
        if not bool(np.any(finite)):
            report[name] = {
                "n": 0,
                "weighted_power": float("nan"),
                "median_abs_jy": float("nan"),
                "median_abs_over_i": float("nan"),
            }
            continue
        power = float(np.sum(w[finite] * np.abs(z[finite]) ** 2) / np.sum(w[finite]))
        abs_jy = np.abs(z[finite])
        over_i = (
            abs_jy / np.maximum(np.abs(intensity[finite]), 1.0e-3)
            if intensity is not None
            else np.full(abs_jy.shape, np.nan)
        )
        report[name] = {
            "n": int(np.sum(finite)),
            "weighted_power": power,
            "median_abs_jy": float(np.median(abs_jy)),
            "median_abs_over_i": (
                float(np.median(over_i)) if intensity is not None else float("nan")
            ),
        }
    return report


def _cluster_labels(
    geometry: Mapping[str, ArrayLike],
    row_mask: ArrayLike,
) -> NDArray[np.int64]:
    mask = np.asarray(row_mask, dtype=bool).reshape(-1)
    dwell = np.asarray(geometry["time_index"]).reshape(-1)[mask].astype(np.int64)
    mover = np.asarray(geometry["moving_id"]).reshape(-1)[mask].astype(np.int64)
    reference = np.asarray(geometry["reference_id"]).reshape(-1)[mask].astype(np.int64)
    cell_l = np.asarray(geometry["cell_l"]).reshape(-1)[mask].astype(np.int64)
    cell_m = np.asarray(geometry["cell_m"]).reshape(-1)[mask].astype(np.int64)
    return dwell * 1_000_003 + mover * 10_007 + reference * 1009 + cell_l * 17 + cell_m


def resample_cluster_indices(
    labels: ArrayLike,
    rng: np.random.Generator,
) -> NDArray[np.int64]:
    """Sample clusters with replacement, preserving multiplicities."""

    ids = np.asarray(labels).reshape(-1)
    unique = np.unique(ids)
    if unique.size == 0:
        return np.zeros(0, dtype=np.int64)
    chosen = rng.choice(unique, size=unique.size, replace=True)
    return np.concatenate([np.flatnonzero(ids == cluster) for cluster in chosen])


def _ci95(values: NDArray[np.float64]) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return (float("nan"), float("nan"))
    return (float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975)))


def _contains(interval: tuple[float, float], value: float) -> bool:
    lo, hi = interval
    return bool(np.isfinite(lo) and np.isfinite(hi) and lo <= value <= hi)


def bootstrap_complex_alpha(
    template: ArrayLike,
    residual: ArrayLike,
    weight: ArrayLike,
    cluster_ids: ArrayLike,
    *,
    n_boot: int = 400,
    seed: int = 0,
    hands: tuple[str, ...] = ("rl", "lr"),
) -> dict[str, object]:
    """Clustered bootstrap of the training or holdout template coefficient."""

    labels = np.asarray(cluster_ids).reshape(-1)
    unique = np.unique(labels)
    point = fit_complex_template_alpha(template, residual, weight, hands=hands)
    empty = {
        "real": float(point.real),
        "imag": float(point.imag),
        "abs": float(np.abs(point)),
        "n_clusters": int(unique.size),
        "n_boot": 0,
        "real_std": float("nan"),
        "imag_std": float("nan"),
        "real_ci95": (float("nan"), float("nan")),
        "imag_ci95": (float("nan"), float("nan")),
        "consistent_with_zero": None,
        "consistent_with_one": None,
    }
    if unique.size == 0 or not np.isfinite(point.real):
        return empty
    t_all = _plane2(template)
    r_all = _plane2(residual)
    w_all = np.asarray(weight, dtype=np.float64)
    if w_all.ndim == 4:
        w_all = w_all[:, 0]
    rng = np.random.default_rng(int(seed))
    draws = np.empty(int(n_boot), dtype=np.complex128)
    for boot in range(int(n_boot)):
        index = resample_cluster_indices(labels, rng)
        draws[boot] = fit_complex_template_alpha(
            t_all[index],
            r_all[index],
            w_all[index],
            hands=hands,
        )
    finite = draws[np.isfinite(draws.real) & np.isfinite(draws.imag)]
    real_ci = _ci95(finite.real)
    imag_ci = _ci95(finite.imag)
    return {
        "real": float(point.real),
        "imag": float(point.imag),
        "abs": float(np.abs(point)),
        "n_clusters": int(unique.size),
        "n_boot": int(finite.size),
        "real_std": float(np.std(finite.real, ddof=1)) if finite.size > 1 else float("nan"),
        "imag_std": float(np.std(finite.imag, ddof=1)) if finite.size > 1 else float("nan"),
        "real_ci95": real_ci,
        "imag_ci95": imag_ci,
        "consistent_with_zero": _contains(real_ci, 0.0) and _contains(imag_ci, 0.0),
        "consistent_with_one": _contains(real_ci, 1.0) and _contains(imag_ci, 0.0),
    }


def bootstrap_alpha_increment(
    template: ArrayLike,
    residual_uninjected: ArrayLike,
    residual_injected: ArrayLike,
    weight: ArrayLike,
    cluster_ids: ArrayLike,
    *,
    n_boot: int = 400,
    seed: int = 0,
    hands: tuple[str, ...] = ("rl", "lr"),
) -> dict[str, object]:
    """Clustered bootstrap of :math:`α(a)-α(0)` on identical resamples."""

    labels = np.asarray(cluster_ids).reshape(-1)
    t_all = _plane2(template)
    r0 = _plane2(residual_uninjected)
    ra = _plane2(residual_injected)
    w_all = np.asarray(weight, dtype=np.float64)
    if w_all.ndim == 4:
        w_all = w_all[:, 0]
    point = fit_complex_template_alpha(t_all, ra, w_all, hands=hands) - fit_complex_template_alpha(
        t_all, r0, w_all, hands=hands
    )
    rng = np.random.default_rng(int(seed))
    draws = np.empty(int(n_boot), dtype=np.complex128)
    for boot in range(int(n_boot)):
        index = resample_cluster_indices(labels, rng)
        draws[boot] = fit_complex_template_alpha(
            t_all[index], ra[index], w_all[index], hands=hands
        ) - fit_complex_template_alpha(t_all[index], r0[index], w_all[index], hands=hands)
    finite = draws[np.isfinite(draws.real) & np.isfinite(draws.imag)]
    real_ci = _ci95(finite.real)
    imag_ci = _ci95(finite.imag)
    return {
        "real": float(point.real),
        "imag": float(point.imag),
        "abs": float(np.abs(point)),
        "n_boot": int(finite.size),
        "real_ci95": real_ci,
        "imag_ci95": imag_ci,
        "inconsistent_with_zero": not (_contains(real_ci, 0.0) and _contains(imag_ci, 0.0)),
    }


def software_gates_for_plane(
    plane: HighresCassbeamPlane,
    *,
    measured_lm_rad: ArrayLike | None = None,
    interpolation_floor: float = INTERPOLATION_FLOOR,
) -> dict[str, object]:
    """Identity, finiteness, node lookup, and interpolation invariance."""

    origin = plane.jones_norm[plane.m_origin_index, plane.l_origin_index]
    identity_err = float(np.max(np.abs(origin - np.eye(2))))
    identity_ok = identity_err < 1.0e-10
    finite_ok = bool(np.all(np.isfinite(plane.jones_norm)))
    n_l = int(plane.l_rad.size)
    n_m = int(plane.m_rad.size)
    grid_l = np.broadcast_to(plane.l_rad[None, :], (n_m, n_l)).reshape(-1)
    grid_m = np.broadcast_to(plane.m_rad[:, None], (n_m, n_l)).reshape(-1)
    looked, ok = plane.lookup(grid_l, grid_m, off_diagonal=True)
    table = plane.jones_norm.reshape(-1, 2, 2)
    node_err = float(np.max(np.abs(looked[ok] - table[ok]))) if bool(np.any(ok)) else float("nan")
    node_ok = bool(np.isfinite(node_err) and node_err < 1.0e-12)
    diag = diagonal_projection(plane.jones_norm)
    copolar_ok = bool(
        np.allclose(diag[..., 0, 0], plane.jones_norm[..., 0, 0])
        and np.allclose(diag[..., 1, 1], plane.jones_norm[..., 1, 1])
        and np.allclose(diag[..., 0, 1], 0.0)
        and np.allclose(diag[..., 1, 0], 0.0)
    )
    interp_rel = float("nan")
    interp_ok = True
    if measured_lm_rad is not None:
        measured = np.asarray(measured_lm_rad, dtype=np.float64).reshape(-1, 2)
        if measured.size:
            base, base_ok = plane.lookup(measured[:, 0], measured[:, 1], off_diagonal=True)
            shifted, shift_ok = plane.lookup(
                measured[:, 0] + 1.0e-4 * plane.pixel_scale_rad,
                measured[:, 1],
                off_diagonal=True,
            )
            usable = base_ok & shift_ok & np.isfinite(base[..., 0, 0])
            rr = np.abs(base[:, 0, 0])
            peak = float(np.nanmax(rr[usable])) if bool(np.any(usable)) else float("nan")
            main = usable & np.isfinite(rr) & (rr >= MAIN_LOBE_VOLTAGE * peak)
            if bool(np.any(main)):
                denom = np.maximum(rr[main], 1.0e-12)
                delta = np.max(np.abs(shifted[main] - base[main]), axis=(-2, -1))
                interp_rel = float(np.nanmax(delta / denom))
                interp_ok = interp_rel < float(interpolation_floor)
    passed = bool(identity_ok and finite_ok and node_ok and copolar_ok and interp_ok)
    return {
        "passed": passed,
        "identity_at_origin": {"passed": identity_ok, "max_abs_err": identity_err},
        "finite_full_complex": {"passed": finite_ok},
        "grid_node_lookup": {"passed": node_ok, "max_abs_err": node_err},
        "interpolation_invariance": {
            "passed": interp_ok,
            "max_relative_main_lobe": interp_rel,
            "floor": float(interpolation_floor),
        },
        "diagonal_shares_copolar": {"passed": copolar_ok},
    }


def manufactured_ep_s_eqh_closure(
    plane: HighresCassbeamPlane,
    *,
    convention: CassbeamConvention = DEFAULT_CONVENTION,
) -> dict[str, object]:
    """Closure of :math:`E_p S E_q^H` with the mover on either baseline side."""

    from sl1mjax.data.canonical import VisibilityBlock
    from sl1mjax.holography import AntennaPointingRole, ResolvedAntennaPointing
    from sl1mjax.polarization import Correlation, ReceptorBasis

    flux = 3.7
    source = circular_stokes_to_coherency(flux, 0.0, 0.0, 0.0)
    i_l = min(plane.l_origin_index + 2, int(plane.l_rad.size) - 2)
    i_m = min(plane.m_origin_index + 1, int(plane.m_rad.size) - 2)
    offset = np.array([plane.l_rad[i_l], plane.m_rad[i_m]], dtype=np.float64)
    frequency = float(plane.frequency_hz)
    n_time = 2
    times = np.array([0.0, 10.0], dtype=np.float64)
    antenna1 = np.array([0, 1], dtype=np.int32)
    antenna2 = np.array([1, 0], dtype=np.int32)
    visibility = np.zeros((2, 1, 4), dtype=np.complex128)
    block = VisibilityBlock(
        uvw_m=np.array([[10.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float64),
        frequency_hz=np.array([frequency], dtype=np.float64),
        visibility=visibility,
        weight=np.ones((2, 1, 4), dtype=np.float64),
        flag=np.zeros((2, 1, 4), dtype=bool),
        time_s=times,
        antenna1=antenna1,
        antenna2=antenna2,
        correlations=(Correlation.RR, Correlation.RL, Correlation.LR, Correlation.LL),
        receptor_basis=ReceptorBasis.CIRCULAR,
        phase_centre_rad=(0.0, 0.0),
    )
    offsets = np.zeros((n_time, 2, 2), dtype=np.float64)
    offsets[:, 0] = offset
    role = np.full((n_time, 2), AntennaPointingRole.REFERENCE.value, dtype="U16")
    role[:, 0] = AntennaPointingRole.MOVING.value
    pointing = ResolvedAntennaPointing(
        unique_time_s=times,
        antenna_id=np.arange(2, dtype=np.int32),
        offset_lm_rad=offsets,
        valid=np.ones((n_time, 2), dtype=bool),
        settled=np.ones((n_time, 2), dtype=bool),
        role=role,
        selected_column="DIRECTION",
        offset_sign="commanded_pointing",
        join_rule="interval",
        join_tolerance_s=0.0,
        measure_ref="J2000",
        units="rad",
    )
    observation = HolographyObservation(
        block=block,
        pointing=pointing,
        antenna_position_m=np.array(
            [[-1_601_162.0, -5_042_003.0, 3_553_983.0], [-1_601_100.0, -5_042_100.0, 3_553_900.0]],
            dtype=np.float64,
        ),
        calibration_state="casa_parang_true",
        phase_centre_rad=(0.0, 0.0),
        stokes_i=flux,
        source_coherency_visibility=np.broadcast_to(source, (2, 1, 2, 2)).copy(),
    )
    identity = {0: np.eye(2, dtype=np.complex128), 1: np.eye(2, dtype=np.complex128)}
    chi = np.zeros((n_time, 2), dtype=np.float64)
    catalog = _InMemoryCatalog(plane)
    e_m, ok = evaluate_highres_cassbeam_jones(
        offset[None, :],
        [frequency],
        [0.0],
        catalog=catalog,
        convention=convention,
        off_diagonal=True,
        calibration_state="casa_parang_true",
    )
    expected_p = e_m[0] @ source
    expected_q = source @ e_m[0].conj().T
    pred = predict_moving_reference_from_highres_cassbeam(
        observation,
        residual_jones=identity,
        catalog=catalog,
        convention=convention,
        row_mask=np.ones(2, dtype=bool),
        parallactic_angle_rad=chi,
        off_diagonal=True,
    )
    err_p = float(np.max(np.abs(pred[0, 0] - expected_p)))
    err_q = float(np.max(np.abs(pred[1, 0] - expected_q)))
    passed = bool(ok[0] and err_p < 1.0e-9 and err_q < 1.0e-9)
    return {
        "passed": passed,
        "mover_is_p_max_abs_err": err_p,
        "mover_is_q_max_abs_err": err_q,
    }


class _InMemoryCatalog:
    """Test/gate catalog that serves one already-loaded plane."""

    def __init__(self, plane: HighresCassbeamPlane) -> None:
        self._plane = plane
        self.frozen = False
        self.production_factory = False

    def require_exact_mhz(self, frequency_hz: float) -> int:
        mhz = int(np.round(float(frequency_hz) / 1.0e6))
        if mhz != int(self._plane.frequency_mhz):
            raise ValueError("no exact high-res CASSBEAM plane at this frequency")
        return mhz

    def plane(self, frequency_hz: float) -> HighresCassbeamPlane:
        self.require_exact_mhz(frequency_hz)
        return self._plane


def run_software_gates(
    plane: HighresCassbeamPlane,
    *,
    measured_lm_rad: ArrayLike | None = None,
) -> dict[str, object]:
    plane_gates = software_gates_for_plane(plane, measured_lm_rad=measured_lm_rad)
    closure = manufactured_ep_s_eqh_closure(plane)
    passed = bool(plane_gates["passed"] and closure["passed"])
    return {
        "passed": passed,
        "outcome": "passed" if passed else "software_gate_failed",
        "plane": plane_gates,
        "manufactured_closure": closure,
        "identical_calibration_source_masks": True,
    }


def loro_masks(
    observation: HolographyObservation,
    *,
    held_reference_id: int,
    max_train_per_cell: int = 4,
) -> dict[str, object]:
    from sl1mjax.holography_full_jones import (
        classify_holdout_interpolation_support,
        freeze_interpolation_support_from_rows,
    )

    masks = holography_one_axis_holdout_masks(
        observation,
        axis="leave_one_reference_out",
        held_reference_id=int(held_reference_id),
        reserve_outer_fold=True,
    )
    train = thin_training_rows(observation, masks["train"], max_per_cell=max_train_per_cell)
    geometry = moving_reference_row_geometry(observation)
    support = freeze_interpolation_support_from_rows(observation, train)
    labeled = classify_holdout_interpolation_support(support, geometry, masks["scored"])
    exact = np.asarray(masks["scored"], dtype=bool) & labeled["exact"]
    return {
        "masks": masks,
        "train": train,
        "exact": exact,
        "geometry": geometry,
        "labeled": labeled,
    }


def predict_full_and_diagonal(
    observation: HolographyObservation,
    *,
    residual_jones: Mapping[int, ArrayLike],
    catalog: HighresCassbeamCatalog,
    convention: CassbeamConvention,
    row_mask: ArrayLike,
    parallactic_angle_rad: ArrayLike | None = None,
    source_coherency: ArrayLike | None = None,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    pred_full = predict_moving_reference_from_highres_cassbeam(
        observation,
        residual_jones=residual_jones,
        catalog=catalog,
        convention=convention,
        row_mask=row_mask,
        parallactic_angle_rad=parallactic_angle_rad,
        off_diagonal=True,
        source_coherency=source_coherency,
    )
    pred_diag = predict_moving_reference_from_highres_cassbeam(
        observation,
        residual_jones=residual_jones,
        catalog=catalog,
        convention=convention,
        row_mask=row_mask,
        parallactic_angle_rad=parallactic_angle_rad,
        off_diagonal=False,
        source_coherency=source_coherency,
    )
    return pred_full, pred_diag


def score_template_predictions(
    observation: HolographyObservation,
    pred_full: ArrayLike,
    pred_diag: ArrayLike,
    *,
    row_mask: ArrayLike,
    stokes_i: ArrayLike,
    geometry: Mapping[str, ArrayLike],
    voltage: ArrayLike | None = None,
    fitted_alpha: complex | None = None,
) -> dict[str, object]:
    packed = pack_coherency(
        observation.block.visibility, observation.block.correlations, (Receptor.R, Receptor.L)
    )
    measured = _plane2(packed)
    full = _plane2(pred_full)
    diag = _plane2(pred_diag)
    mask = np.asarray(row_mask, dtype=bool).reshape(-1)
    weight = _hand_weight(observation, mask)
    measured_m = measured[mask]
    full_m = full[mask]
    diag_m = diag[mask]
    t = full_m - diag_m
    r = measured_m - diag_m
    intensity = np.asarray(stokes_i, dtype=np.float64).reshape(-1)
    intensity_m = intensity[mask]
    alpha_coeff = 1.0 + 0.0j if fitted_alpha is None else complex(fitted_alpha)
    pred_fit_m = diag_m + alpha_coeff * t
    pred_fit = np.full_like(diag, np.nan + 1j * np.nan)
    pred_fit[mask] = pred_fit_m
    power_full = weighted_residual_power(measured_m, full_m, weight, stokes_i=intensity_m)
    power_diag = weighted_residual_power(measured_m, diag_m, weight, stokes_i=intensity_m)
    power_fit = weighted_residual_power(measured_m, pred_fit_m, weight, stokes_i=intensity_m)
    clusters = _cluster_labels(geometry, mask)
    alpha = bootstrap_complex_alpha(t, r, weight, clusters)
    corr = template_correlation(t, r, weight)
    cell_ids = (
        np.asarray(geometry["moving_id"], dtype=np.int64) * 2_000_000_003
        + np.asarray(geometry["cell_l"], dtype=np.int64) * 10_007
        + np.asarray(geometry["cell_m"], dtype=np.int64)
    )
    scores = None
    fitted_scores = None
    if voltage is not None:
        scores = score_full_versus_diagonal_regions(
            measured,
            full,
            diag,
            stokes_i=intensity,
            row_mask=mask,
            voltage=voltage,
            dwell_ids=geometry["time_index"],
            mover_ids=geometry["moving_id"],
            reference_ids=geometry["reference_id"],
            cell_ids=cell_ids,
        )
        fitted_scores = score_full_versus_diagonal_regions(
            measured,
            pred_fit,
            diag,
            stokes_i=intensity,
            row_mask=mask,
            voltage=voltage,
            dwell_ids=geometry["time_index"],
            mover_ids=geometry["moving_id"],
            reference_ids=geometry["reference_id"],
            cell_ids=cell_ids,
        )
    return {
        "n": int(np.sum(mask)),
        "alpha": alpha,
        "training_correlation": None,
        "holdout_correlation": corr,
        "residual_power": {"alpha_1": power_full, "alpha_0": power_diag, "alpha_hat": power_fit},
        "paired_scores": scores,
        "fitted_paired_scores": fitted_scores,
        "template_median_abs_rl": (
            float(np.nanmedian(np.abs(t[:, 0, 1]))) if t.size else float("nan")
        ),
        "template_median_abs_lr": (
            float(np.nanmedian(np.abs(t[:, 1, 0]))) if t.size else float("nan")
        ),
    }


def _subsample_mask(mask: ArrayLike, max_rows: int, *, seed: int = 0) -> NDArray[np.bool_]:
    selected = np.asarray(mask, dtype=bool).reshape(-1)
    index = np.flatnonzero(selected)
    if index.size <= int(max_rows):
        return selected
    keep = np.random.default_rng(int(seed)).choice(index, size=int(max_rows), replace=False)
    out = np.zeros_like(selected)
    out[keep] = True
    return out


def lock_cassbeam_convention(
    observation: HolographyObservation,
    *,
    residual_jones: Mapping[int, ArrayLike],
    catalog: HighresCassbeamCatalog,
    train_mask: ArrayLike,
    parallactic_angle_rad: ArrayLike,
    geometry: Mapping[str, ArrayLike],
    candidates: Sequence[CassbeamConvention] | None = None,
    max_rows: int = 8000,
) -> dict[str, object]:
    """Select a convention from training rows only. Never from holdout RL/LR."""

    train = _subsample_mask(train_mask, max_rows)
    packed = pack_coherency(
        observation.block.visibility, observation.block.correlations, (Receptor.R, Receptor.L)
    )
    measured = _plane2(packed)
    weight = _hand_weight(observation, train)
    reference_ids = np.asarray(geometry["reference_id"], dtype=np.int32)
    ladder = tuple(candidates) if candidates is not None else convention_ladder()
    records = []
    for convention in ladder:
        pred_full, pred_diag = predict_full_and_diagonal(
            observation,
            residual_jones=residual_jones,
            catalog=catalog,
            convention=convention,
            row_mask=train,
            parallactic_angle_rad=parallactic_angle_rad,
        )
        t = _plane2(pred_full)[train] - _plane2(pred_diag)[train]
        r = measured[train] - _plane2(pred_diag)[train]
        corr = template_correlation(t, r, weight)
        alpha = fit_complex_template_alpha(t, r, weight)
        records.append(
            {
                "name": convention.name,
                "convention": convention,
                "training_correlation": corr,
                "training_alpha_real": float(alpha.real) if np.isfinite(alpha.real) else None,
                "training_alpha_imag": float(alpha.imag) if np.isfinite(alpha.imag) else None,
                "_t": t,
                "_r": r,
            }
        )
    finite = [
        item
        for item in records
        if item["training_correlation"] is not None and np.isfinite(item["training_correlation"])
    ]
    if not finite:
        return {
            "status": "convention_unresolved",
            "selected": None,
            "candidates": [
                {k: v for k, v in item.items() if not k.startswith("_") and k != "convention"}
                for item in records
            ],
            "reason": "no_finite_training_correlation",
        }
    ranked = sorted(finite, key=lambda item: item["training_correlation"], reverse=True)
    winner = ranked[0]
    runner_up = ranked[1]["training_correlation"] if len(ranked) > 1 else float("-inf")
    unique_ok = float(winner["training_correlation"]) >= float(runner_up) + CORRELATION_MARGIN
    selected = winner["convention"]
    ref_on_train = reference_ids[train]
    per_reference = {}
    agreed = True
    for antenna in np.unique(ref_on_train):
        subset = ref_on_train == int(antenna)
        if int(np.sum(subset)) < MIN_CONVENTION_ROWS:
            continue
        subset_weight = weight[subset]
        best_name = None
        best_corr = float("-inf")
        for item in records:
            t = item["_t"][subset]
            r = item["_r"][subset]
            corr = template_correlation(t, r, subset_weight)
            if np.isfinite(corr) and corr > best_corr:
                best_corr = corr
                best_name = item["name"]
        per_reference[str(int(antenna))] = {"name": best_name, "training_correlation": best_corr}
        if best_name != selected.name:
            agreed = False
    status = "locked" if unique_ok and agreed else "convention_unresolved"
    return {
        "status": status,
        "selected": selected.name if status == "locked" else None,
        "selected_convention": selected if status == "locked" else None,
        "training_correlation": winner["training_correlation"],
        "unique_by_margin": unique_ok,
        "stable_across_training_references": agreed,
        "per_training_reference": per_reference,
        "n_training_rows_used": int(np.sum(train)),
        "candidates": [
            {k: v for k, v in item.items() if not k.startswith("_") and k != "convention"}
            for item in records
        ],
        "notes": (TRAINING_ONLY_CONVENTION_NOTE,),
    }


def inject_cassbeam_template(
    measured: ArrayLike,
    pred_full: ArrayLike,
    pred_diag: ArrayLike,
    amplitude: float,
) -> NDArray[np.complex128]:
    """Add :math:`a(V_{full}-V_{diagonal})`. Already zero in the on-axis gauge."""

    vis = np.array(measured, dtype=np.complex128, copy=True)
    full = np.asarray(pred_full, dtype=np.complex128)
    diag = np.asarray(pred_diag, dtype=np.complex128)
    if vis.ndim != full.ndim:
        full = _plane2(full)
        diag = _plane2(diag)
        if vis.ndim == 4:
            vis = vis[:, 0]
    template = full - diag
    finite = np.isfinite(template)
    vis = vis + float(amplitude) * np.where(finite, template, 0.0)
    return vis


def template_injection_curve(
    observation: HolographyObservation,
    pred_full: ArrayLike,
    pred_diag: ArrayLike,
    *,
    train_mask: ArrayLike,
    holdout_mask: ArrayLike,
    geometry: Mapping[str, ArrayLike],
    amplitudes: Sequence[float] = TEMPLATE_INJECTION_AMPLITUDES,
) -> dict[str, object]:
    """Recover α(a) on training and evaluate the same α on holdout."""

    packed = pack_coherency(
        observation.block.visibility, observation.block.correlations, (Receptor.R, Receptor.L)
    )
    measured = _plane2(packed)
    full = _plane2(pred_full)
    diag = _plane2(pred_diag)
    train = np.asarray(train_mask, dtype=bool).reshape(-1)
    hold = np.asarray(holdout_mask, dtype=bool).reshape(-1)
    t = full - diag
    train_weight = _hand_weight(observation, train)
    hold_weight = _hand_weight(observation, hold)
    train_clusters = _cluster_labels(geometry, train)
    hold_clusters = _cluster_labels(geometry, hold)
    r0_train = measured[train] - diag[train]
    r0_hold = measured[hold] - diag[hold]
    points = []
    for amplitude in amplitudes:
        injected = inject_cassbeam_template(measured, full, diag, amplitude)
        r_train = injected[train] - diag[train]
        r_hold = injected[hold] - diag[hold]
        train_alpha = bootstrap_complex_alpha(t[train], r_train, train_weight, train_clusters)
        hold_alpha = bootstrap_complex_alpha(t[hold], r_hold, hold_weight, hold_clusters)
        train_inc = bootstrap_alpha_increment(
            t[train], r0_train, r_train, train_weight, train_clusters
        )
        hold_inc = bootstrap_alpha_increment(t[hold], r0_hold, r_hold, hold_weight, hold_clusters)
        points.append(
            {
                "a": float(amplitude),
                "train": train_alpha,
                "holdout": hold_alpha,
                "train_increment": train_inc,
                "holdout_increment": hold_inc,
            }
        )
    hold_inc = np.array(
        [
            point["holdout_increment"]["real"]
            for point in points
            if np.isfinite(point["holdout_increment"]["real"])
        ],
        dtype=np.float64,
    )
    a_vals = np.array(
        [point["a"] for point in points if np.isfinite(point["holdout_increment"]["real"])],
        dtype=np.float64,
    )
    slope = float("nan")
    if a_vals.size >= 2:
        slope = float(np.polyfit(a_vals, hold_inc, 1)[0])
    monotonic = bool(a_vals.size >= 2 and np.all(np.diff(hold_inc) >= -0.15))
    a1 = next((point for point in points if abs(point["a"] - 1.0) < 1.0e-12), None)
    detectable = bool(a1 is not None and a1["holdout_increment"].get("inconsistent_with_zero"))
    return {
        "amplitudes": list(amplitudes),
        "points": points,
        "holdout_increment_slope": slope,
        "monotonic_increment": monotonic,
        "a1_detectable": detectable,
        "matched_filter_identity": (
            "Increment α(a)-α(0) uses the same template that was injected. "
            "Detectability is whether that increment is inconsistent with zero, "
            "not whether the raw slope is one."
        ),
    }


def source_coherency_with_qu(
    observation: HolographyObservation,
    q_over_i: float,
    u_over_i: float,
) -> NDArray[np.complex128]:
    source = np.asarray(_source_coherency(observation), dtype=np.complex128)
    intensity = 0.5 * (source[..., 0, 0] + source[..., 1, 1])
    return circular_stokes_to_coherency(
        intensity,
        float(q_over_i) * intensity,
        float(u_over_i) * intensity,
        0.0 * intensity,
    )


def qu_nuisance_robustness(
    observation: HolographyObservation,
    *,
    residual_jones: Mapping[int, ArrayLike],
    catalog: HighresCassbeamCatalog,
    convention: CassbeamConvention,
    row_mask: ArrayLike,
    parallactic_angle_rad: ArrayLike,
    geometry: Mapping[str, ArrayLike],
    pairs: Sequence[tuple[float, float]] = QU_NUISANCE_PAIRS,
) -> dict[str, object]:
    packed = pack_coherency(
        observation.block.visibility, observation.block.correlations, (Receptor.R, Receptor.L)
    )
    measured = _plane2(packed)
    mask = np.asarray(row_mask, dtype=bool).reshape(-1)
    weight = _hand_weight(observation, mask)
    clusters = _cluster_labels(geometry, mask)
    points = []
    for q_over_i, u_over_i in pairs:
        source = source_coherency_with_qu(observation, q_over_i, u_over_i)
        pred_full, pred_diag = predict_full_and_diagonal(
            observation,
            residual_jones=residual_jones,
            catalog=catalog,
            convention=convention,
            row_mask=mask,
            parallactic_angle_rad=parallactic_angle_rad,
            source_coherency=source,
        )
        t = _plane2(pred_full)[mask] - _plane2(pred_diag)[mask]
        r = measured[mask] - _plane2(pred_diag)[mask]
        points.append(
            {
                "q_over_i": float(q_over_i),
                "u_over_i": float(u_over_i),
                "alpha": bootstrap_complex_alpha(t, r, weight, clusters),
            }
        )
    reals = [item["alpha"]["real"] for item in points if np.isfinite(item["alpha"]["real"])]
    imags = [item["alpha"]["imag"] for item in points if np.isfinite(item["alpha"]["imag"])]
    return {
        "points": points,
        "evaluated_on": "training_rows",
        "independent_q_and_u": True,
        "alpha_real_range": (
            (float(np.min(reals)), float(np.max(reals))) if reals else (float("nan"), float("nan"))
        ),
        "alpha_imag_range": (
            (float(np.min(imags)), float(np.max(imags))) if imags else (float("nan"), float("nan"))
        ),
        "stable": bool(reals) and (max(reals) - min(reals)) < 0.4,
    }


def _paired_improves(paired: Mapping[str, object] | None) -> bool:
    if not paired:
        return False
    boot = paired.get("bootstrap_improvement_over_i") or {}
    rl = boot.get("rl") or {}
    lr = boot.get("lr") or {}
    return bool(
        float(rl.get("fraction_positive") or 0.0) > 0.95
        and float(lr.get("fraction_positive") or 0.0) > 0.95
        and not paired.get("rr_ll_regression")
    )


def classify_highres_cassbeam_direct(
    *,
    software_gates_passed: bool,
    convention_status: str,
    injection: Mapping[str, object] | None,
    holdout_alpha: Mapping[str, object] | None,
    holdout_paired: Mapping[str, object] | None = None,
    holdout_fitted_paired: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Five-way scientific classifier. No decision if gates or convention fail."""

    if not software_gates_passed:
        outcome = "software_gate_failed"
    elif convention_status != "locked":
        outcome = "convention_unresolved"
    elif not bool((injection or {}).get("a1_detectable")):
        outcome = "sensitivity_limited"
    else:
        alpha = holdout_alpha or {}
        a1_improves = _paired_improves(holdout_paired)
        hat_improves = _paired_improves(holdout_fitted_paired)
        nonzero = alpha.get("consistent_with_zero") is False
        one = bool(alpha.get("consistent_with_one"))
        if a1_improves and one:
            outcome = "cassbeam_full_jones_supported"
        elif nonzero and not one and hat_improves:
            outcome = "cassbeam_morphology_supported_scale_mismatch"
        else:
            outcome = "cassbeam_offdiagonal_rejected"
    no_science = outcome in {"software_gate_failed", "convention_unresolved"}
    return {
        "gate": HIGHRES_CASSBEAM_DIRECT,
        "status": "pass" if outcome == "cassbeam_full_jones_supported" else "fail",
        "blocking": outcome != "cassbeam_full_jones_supported",
        "outcome": outcome,
        "scientific_decision": not no_science,
        "full_jones_blocked": True,
        "full_jones_frozen": False,
        "spw5_closed": True,
        "most_important_next_artifact": "cassbeam_diagonal_low_order_correction",
        "notes": (
            HIGHRES_CASSBEAM_NOTE,
            TRAINING_ONLY_CONVENTION_NOTE,
            TEMPLATE_ALPHA_NOTE,
            NO_GLOBAL_FIVE_PERCENT_NOTE,
            CHANNEL_COVARIANCE_NOTE,
        ),
    }


def joint_channel_likelihood(
    per_channel: Mapping[str, Mapping[str, ArrayLike]],
    *,
    n_boot: int = 400,
    seed: int = 1,
    hands: tuple[str, ...] = ("rl", "lr"),
) -> dict[str, object]:
    """Channel-block bootstrap: resample clusters across every channel together."""

    channels = []
    for name, item in per_channel.items():
        if not {"template", "residual", "weight", "cluster_ids"} <= set(item):
            return {
                "n_channels": 0,
                "status": "not_a_channel_block",
                "reason": "row-level template, residual, weight, and cluster_ids are required",
                "sqrt_n_not_assumed": True,
                "notes": (CHANNEL_COVARIANCE_NOTE,),
            }
        channels.append(
            (
                str(name),
                _plane2(item["template"]),
                _plane2(item["residual"]),
                np.asarray(item["weight"], dtype=np.float64),
                np.asarray(item["cluster_ids"]).reshape(-1),
            )
        )
    if not channels:
        return {"n_channels": 0, "mean_real": float("nan"), "std": float("nan")}
    all_clusters = np.unique(np.concatenate([item[4] for item in channels]))
    rng = np.random.default_rng(int(seed))
    draws = np.empty(int(n_boot), dtype=np.complex128)
    for boot in range(int(n_boot)):
        chosen = rng.choice(all_clusters, size=all_clusters.size, replace=True)
        t_parts = []
        r_parts = []
        w_parts = []
        for _name, template, residual, weight, labels in channels:
            index = (
                np.concatenate([np.flatnonzero(labels == cluster) for cluster in chosen])
                if chosen.size
                else np.zeros(0, dtype=np.int64)
            )
            t_parts.append(template[index])
            r_parts.append(residual[index])
            w_parts.append(weight[index])
        draws[boot] = fit_complex_template_alpha(
            np.concatenate(t_parts),
            np.concatenate(r_parts),
            np.concatenate(w_parts),
            hands=hands,
        )
    finite = draws[np.isfinite(draws.real) & np.isfinite(draws.imag)]
    point_t = np.concatenate([item[1] for item in channels])
    point_r = np.concatenate([item[2] for item in channels])
    point_w = np.concatenate([item[3] for item in channels])
    point = fit_complex_template_alpha(point_t, point_r, point_w, hands=hands)
    real_ci = _ci95(finite.real)
    imag_ci = _ci95(finite.imag)
    return {
        "n_channels": len(channels),
        "n_clusters": int(all_clusters.size),
        "real": float(point.real),
        "imag": float(point.imag),
        "real_ci95": real_ci,
        "imag_ci95": imag_ci,
        "consistent_with_zero": _contains(real_ci, 0.0) and _contains(imag_ci, 0.0),
        "consistent_with_one": _contains(real_ci, 1.0) and _contains(imag_ci, 0.0),
        "sqrt_n_not_assumed": True,
        "channel_block": True,
        "notes": (CHANNEL_COVARIANCE_NOTE,),
    }


def write_template_plots(
    output_dir: Path,
    *,
    observation: HolographyObservation,
    pred_full: ArrayLike,
    pred_diag: ArrayLike,
    row_mask: ArrayLike,
    geometry: Mapping[str, ArrayLike],
    voltage: ArrayLike,
    alpha_by_group: Mapping[str, Mapping[str, object]] | None = None,
    max_baselines: int = 4,
) -> list[str]:
    """First diagnostic plots for selected mover–reference baselines."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    packed = pack_coherency(
        observation.block.visibility, observation.block.correlations, (Receptor.R, Receptor.L)
    )
    measured = _plane2(packed)
    full = _plane2(pred_full)
    diag = _plane2(pred_diag)
    mask = np.asarray(row_mask, dtype=bool).reshape(-1)
    mover = np.asarray(geometry["moving_id"]).reshape(-1)
    reference = np.asarray(geometry["reference_id"]).reshape(-1)
    offset = np.asarray(geometry["offset_lm_rad"])
    radius = np.asarray(geometry["radius_rad"])
    time_index = np.asarray(geometry["time_index"])
    pairs = []
    for mv, rf in zip(mover[mask], reference[mask], strict=False):
        pairs.append((int(mv), int(rf)))
    if not pairs:
        return []
    unique, counts = np.unique(np.asarray(pairs, dtype=np.int64), axis=0, return_counts=True)
    order = np.argsort(-counts)
    selected = unique[order[: int(max_baselines)]]
    written: list[str] = []
    for mv, rf in selected:
        rows = mask & (mover == int(mv)) & (reference == int(rf))
        if not bool(np.any(rows)):
            continue
        prefix = f"mover{int(mv)}_ref{int(rf)}"
        obs_rl = measured[rows, 0, 1]
        full_rl = full[rows, 0, 1]
        diag_rl = diag[rows, 0, 1]
        obs_lr = measured[rows, 1, 0]
        full_lr = full[rows, 1, 0]
        fig, axes = plt.subplots(2, 2, figsize=(10.0, 8.0))
        axes[0, 0].scatter(full_rl.real, obs_rl.real, s=8, alpha=0.6, label="RL real")
        axes[0, 0].scatter(full_lr.real, obs_lr.real, s=8, alpha=0.6, label="LR real")
        axes[0, 0].set_title("observed vs predicted real")
        axes[0, 1].scatter(full_rl.imag, obs_rl.imag, s=8, alpha=0.6, label="RL imag")
        axes[0, 1].scatter(full_lr.imag, obs_lr.imag, s=8, alpha=0.6, label="LR imag")
        axes[0, 1].set_title("observed vs predicted imag")
        axes[1, 0].scatter(np.abs(full_rl), np.abs(obs_rl), s=8, alpha=0.6, label="full RL")
        axes[1, 0].scatter(np.abs(diag_rl), np.abs(obs_rl), s=8, alpha=0.6, label="diag RL")
        axes[1, 0].set_title("observed vs predicted magnitude")
        sig = (np.abs(obs_rl) > 0.02) & (np.abs(full_rl) > 0.02)
        if bool(np.any(sig)):
            axes[1, 1].scatter(
                np.angle(full_rl[sig]),
                np.angle(obs_rl[sig] * np.conjugate(full_rl[sig])),
                s=8,
                alpha=0.6,
            )
        axes[1, 1].set_title("phase residual where amplitudes are significant")
        for axis in axes.ravel():
            axis.grid(True, alpha=0.3)
        fig.tight_layout()
        path = output_dir / f"{prefix}_complex.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        written.append(str(path))
        fig, axes = plt.subplots(2, 2, figsize=(10.0, 8.0))
        residual = obs_rl - full_rl
        axes[0, 0].scatter(
            offset[rows, 0], offset[rows, 1], c=np.abs(residual), s=10, cmap="viridis"
        )
        axes[0, 0].set_title("RL residual vs l,m")
        axes[0, 1].scatter(radius[rows], np.abs(residual), s=8, alpha=0.6)
        axes[0, 1].set_title("RL residual vs radius")
        axes[1, 0].scatter(time_index[rows], np.abs(residual), s=8, alpha=0.6)
        axes[1, 0].set_title("RL residual vs time")
        axes[1, 1].scatter(np.asarray(voltage).reshape(-1)[rows], np.abs(residual), s=8, alpha=0.6)
        axes[1, 1].set_title("RL residual vs beam voltage")
        fig.tight_layout()
        path = output_dir / f"{prefix}_residual.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        written.append(str(path))
    if alpha_by_group:
        fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
        for axis, key in zip(axes, ("mover", "reference"), strict=True):
            group = alpha_by_group.get(key) or {}
            names = list(group)
            reals = [group[name].get("real", float("nan")) for name in names]
            axis.bar(np.arange(len(names)), reals)
            axis.set_xticks(np.arange(len(names)), names, rotation=45, ha="right")
            axis.set_title(f"fitted α by {key}")
            axis.axhline(0.0, color="k", lw=0.8)
            axis.axhline(1.0, color="C1", lw=0.8, ls="--")
        fig.tight_layout()
        path = output_dir / "alpha_by_group.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        written.append(str(path))
    return written


def artifact_checksum_report(
    catalog: HighresCassbeamCatalog, frequency_hz: float
) -> dict[str, object]:
    plane = catalog.plane(frequency_hz)
    return {
        "root": str(catalog.root),
        "model_id": catalog.expected_model_id,
        "frequency_hz": float(plane.frequency_hz),
        "frequency_mhz": int(plane.frequency_mhz),
        "data_sha256": plane.data_sha256,
        "params_sha256": plane.params_sha256,
        "raster": list(catalog.expected_raster),
        "l_origin_index": plane.l_origin_index,
        "m_origin_index": plane.m_origin_index,
        "production_factory": False,
        "full_jones_frozen": False,
        "committed_33x33_loaded": False,
    }
