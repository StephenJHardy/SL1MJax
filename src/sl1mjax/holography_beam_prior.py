"""SPW-4 multi-channel visibility-domain holography beam prior.

Fits nested off-diagonal models on sealed training splits using batched
sufficient statistics. Conventions are scored jointly across native
channels. Full Jones stays unfrozen. SPW 5 stays closed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sl1mjax.beam_conventions import BeamCalibrationState, require_beam_calibration_state
from sl1mjax.cassbeam_highres import HighresCassbeamCatalog, diagonal_projection
from sl1mjax.holography_full_jones import _antenna_jones_planes
from sl1mjax.holography_highres_cassbeam import (
    CHANNEL_COVARIANCE_NOTE,
    QU_NUISANCE_PAIRS,
    CassbeamConvention,
    _ci95,
    _contains,
    antenna_frame_lm,
    apply_axis_convention,
    convention_ladder,
    normalize_after_jones_convention,
    resample_cluster_indices,
)
from sl1mjax.polarization import circular_parallactic_jones

SPW4_MULTICHANNEL_BEAM_PRIOR = "spw4_multichannel_beam_prior"
INJECTION_VOLTAGE = (0.001, 0.003, 0.01, 0.03)
TEMPLATE_EQUIV_REL = 0.002
SMOOTH_RIDGE = 8.0
SPATIAL_RIDGE_TOWARD_ONE = 16.0
ANTENNA_RIDGE_TOWARD_ARRAY = 32.0

PRIOR_NOTE = (
    "This prior contains only quantities supported by sealed holdouts. "
    "Unsupported off-diagonal structure is an upper limit, not a map."
)
NO_HOLORASTER_IN_DI_NOTE = (
    "HOLORASTER visibilities are never used to fit direction-independent "
    "calibration. Field-10 MODEL_DATA is the source coherency only."
)
INJECTION_NOTE = (
    "Injections are applied only to training and evaluation copies. "
    "Origin rows stay uninjected. Detection is the increment above the "
    "uninjected estimate, with a cluster bootstrap that keeps replacement "
    "multiplicities and joint real/imaginary intervals."
)


@dataclass(frozen=True)
class ComplexMoments:
    """Weighted moments for one complex template coefficient."""

    tt: float
    tr: complex
    rr: float
    n: int
    cluster_id: int = -1
    channel: int = -1
    dwell: int = -1
    mover: int = -1
    reference: int = -1
    cell: int = -1

    @property
    def alpha(self) -> complex:
        if self.tt <= 0.0:
            return complex(np.nan, np.nan)
        return complex(self.tr / self.tt)

    def residual_power(self, alpha: complex) -> float:
        if not np.isfinite(alpha.real):
            return float(self.rr)
        return float(
            self.rr - 2.0 * np.real(np.conjugate(alpha) * self.tr) + abs(alpha) ** 2 * self.tt
        )


@dataclass(frozen=True)
class HolographyBeamPrior:
    """Provisional holography beam-calibration prior. Never frozen here."""

    diagonal_supported: bool
    off_diagonal_mean: complex | None
    off_diagonal_upper_abs: float | None
    frequency_covariance: NDArray[np.complex128] | None
    antenna_scatter: float | None
    support_mask: Mapping[str, object]
    gauge: str
    provenance: Mapping[str, object]
    supported_quantities: tuple[str, ...]
    decision: str
    frozen: bool = False
    full_jones_frozen: bool = False
    production_factory_modified: bool = False
    spw5_opened: bool = False
    notes: tuple[str, ...] = field(
        default_factory=lambda: (PRIOR_NOTE, NO_HOLORASTER_IN_DI_NOTE, CHANNEL_COVARIANCE_NOTE)
    )


def predict_vis_numpy(
    residual_moving: ArrayLike,
    beam: ArrayLike,
    source: ArrayLike,
    residual_reference: ArrayLike,
    moving_is_p: ArrayLike,
) -> NDArray[np.complex128]:
    """Vectorized :math:`V=R_m E_m S R_r^H` without a Python row loop."""

    r_m = np.asarray(residual_moving, dtype=np.complex128)
    e_m = np.asarray(beam, dtype=np.complex128)
    sky = np.asarray(source, dtype=np.complex128)
    r_r = np.asarray(residual_reference, dtype=np.complex128)
    mover_p = np.asarray(moving_is_p, dtype=bool).reshape(-1)
    if e_m.ndim == 3:
        e_m = e_m[:, None, :, :]
    if sky.ndim == 3:
        sky = sky[:, None, :, :]
    if r_m.ndim == 2:
        r_m = r_m[None, ...]
    if r_r.ndim == 2:
        r_r = r_r[None, ...]
    left = r_m[:, None, :, :] @ e_m
    right = np.conjugate(np.swapaxes(r_r[:, None, :, :], -1, -2))
    vis_p = left @ sky @ right
    vis_q = r_r[:, None, :, :] @ sky @ np.conjugate(np.swapaxes(left, -1, -2))
    return np.where(mover_p[:, None, None, None], vis_p, vis_q)


def sky_frame_residual_numpy(feed: ArrayLike, chi: ArrayLike) -> NDArray[np.complex128]:
    jones = np.asarray(feed, dtype=np.complex128)
    angle = np.asarray(chi, dtype=np.float64).reshape(-1)
    rotation = np.exp(-1j * angle)
    para = np.zeros((angle.size, 2, 2), dtype=np.complex128)
    para[:, 0, 0] = rotation
    para[:, 1, 1] = np.conjugate(rotation)
    inverse = np.conjugate(np.swapaxes(para, -1, -2))
    return inverse @ jones @ para


def vis_planes(values: ArrayLike) -> NDArray[np.complex128]:
    """Keep every native channel. Do not collapse to a single visibility."""

    array = np.asarray(values, dtype=np.complex128)
    if array.ndim == 3:
        return array[:, None, :, :]
    if array.ndim != 4:
        raise ValueError("visibilities must have shape (row, [channel,] 2, 2)")
    return array


def flatten_row_channel(
    values: ArrayLike,
    *,
    row_ids: ArrayLike | None = None,
) -> tuple[NDArray, NDArray[np.int64], NDArray[np.int64]]:
    planes = np.asarray(values)
    if planes.ndim == 3:
        planes = planes[:, None, ...]
    n_row, n_chan = int(planes.shape[0]), int(planes.shape[1])
    flat = planes.reshape(n_row * n_chan, *planes.shape[2:])
    rows = (
        np.repeat(np.asarray(row_ids).reshape(-1), n_chan)
        if row_ids is not None
        else np.repeat(np.arange(n_row, dtype=np.int64), n_chan)
    )
    channels = np.tile(np.arange(n_chan, dtype=np.int64), n_row)
    return flat, rows, channels


def hand_weight_cube(observation, row_mask: ArrayLike) -> NDArray[np.float64]:
    """Per-row, per-channel 2×2 weights. Flagged channels stay zero."""

    mask = np.asarray(row_mask, dtype=bool).reshape(-1)
    names = tuple(
        item.value if hasattr(item, "value") else str(item)
        for item in observation.block.correlations
    )
    weight = np.asarray(observation.block.weight, dtype=np.float64)
    flag = np.asarray(observation.block.flag, dtype=bool)
    if weight.ndim == 2:
        weight = weight[:, None, :]
        flag = flag[:, None, :]
    active = (~flag) & np.isfinite(weight) & (weight > 0.0)
    n_chan = int(weight.shape[1])
    out = np.zeros((int(np.sum(mask)), n_chan, 2, 2), dtype=np.float64)
    index = {name: i for i, name in enumerate(names)}
    selected = active[mask]
    chosen = weight[mask]
    out[:, :, 0, 0] = np.where(selected[:, :, index["RR"]], chosen[:, :, index["RR"]], 0.0)
    out[:, :, 0, 1] = np.where(selected[:, :, index["RL"]], chosen[:, :, index["RL"]], 0.0)
    out[:, :, 1, 0] = np.where(selected[:, :, index["LR"]], chosen[:, :, index["LR"]], 0.0)
    out[:, :, 1, 1] = np.where(selected[:, :, index["LL"]], chosen[:, :, index["LL"]], 0.0)
    return out


def accumulate_hand_moments(
    template: ArrayLike,
    residual: ArrayLike,
    weight: ArrayLike,
    *,
    cluster_ids: ArrayLike,
    channel_ids: ArrayLike,
    dwell_ids: ArrayLike,
    mover_ids: ArrayLike,
    reference_ids: ArrayLike,
    cell_ids: ArrayLike,
    hands: tuple[str, ...] = ("rl", "lr"),
) -> list[ComplexMoments]:
    """Batch moments grouped by channel, dwell, mover, reference and cell."""

    t_all = np.asarray(template, dtype=np.complex128)
    r_all = np.asarray(residual, dtype=np.complex128)
    w_all = np.asarray(weight, dtype=np.float64)
    if t_all.ndim == 4:
        t_all, _rows, _chan = flatten_row_channel(t_all)
        r_all, _, _ = flatten_row_channel(r_all)
        w_all, _, _ = flatten_row_channel(w_all)
    elif t_all.ndim != 3:
        raise ValueError("template must have shape (sample, 2, 2) or (row, channel, 2, 2)")
    keys = np.stack(
        [
            np.asarray(channel_ids).reshape(-1),
            np.asarray(dwell_ids).reshape(-1),
            np.asarray(mover_ids).reshape(-1),
            np.asarray(reference_ids).reshape(-1),
            np.asarray(cell_ids).reshape(-1),
            np.asarray(cluster_ids).reshape(-1),
        ],
        axis=1,
    ).astype(np.int64)
    elements = {"rr": (0, 0), "rl": (0, 1), "lr": (1, 0), "ll": (1, 1)}
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    n_group = int(unique.shape[0])
    tt = np.zeros(n_group, dtype=np.float64)
    tr = np.zeros(n_group, dtype=np.complex128)
    rr = np.zeros(n_group, dtype=np.float64)
    count = np.zeros(n_group, dtype=np.int64)
    for name in hands:
        row, col = elements[name]
        t = t_all[:, row, col]
        r = r_all[:, row, col]
        w = w_all[:, row, col]
        finite = np.isfinite(t) & np.isfinite(r) & np.isfinite(w) & (w > 0.0)
        if not bool(np.any(finite)):
            continue
        np.add.at(tt, inverse[finite], w[finite] * np.abs(t[finite]) ** 2)
        np.add.at(tr, inverse[finite], w[finite] * np.conjugate(t[finite]) * r[finite])
        np.add.at(rr, inverse[finite], w[finite] * np.abs(r[finite]) ** 2)
        np.add.at(count, inverse[finite], 1)
    return [
        ComplexMoments(
            tt=float(tt[i]),
            tr=complex(tr[i]),
            rr=float(rr[i]),
            n=int(count[i]),
            cluster_id=int(unique[i, 5]),
            channel=int(unique[i, 0]),
            dwell=int(unique[i, 1]),
            mover=int(unique[i, 2]),
            reference=int(unique[i, 3]),
            cell=int(unique[i, 4]),
        )
        for i in range(n_group)
        if count[i] > 0 and tt[i] > 0.0
    ]


def stack_moments(moments: Sequence[ComplexMoments]) -> ComplexMoments:
    if not moments:
        return ComplexMoments(0.0, 0.0j, 0.0, 0)
    return ComplexMoments(
        tt=float(sum(item.tt for item in moments)),
        tr=complex(sum(item.tr for item in moments)),
        rr=float(sum(item.rr for item in moments)),
        n=int(sum(item.n for item in moments)),
    )


def residual_power_at(moments: Sequence[ComplexMoments], alpha: ArrayLike) -> float:
    coeff = np.asarray(alpha, dtype=np.complex128).reshape(-1)
    if coeff.size == 1:
        return stack_moments(moments).residual_power(complex(coeff[0]))
    by_channel = {}
    for item in moments:
        by_channel.setdefault(int(item.channel), []).append(item)
    total = 0.0
    for channel, group in by_channel.items():
        value = complex(coeff[min(channel, coeff.size - 1)])
        total += stack_moments(group).residual_power(value)
    return float(total)


def fit_unit_and_scalar(moments: Sequence[ComplexMoments]) -> dict[str, object]:
    stacked = stack_moments(moments)
    return {
        "alpha_0": 0.0 + 0.0j,
        "alpha_1": 1.0 + 0.0j,
        "alpha_hat": stacked.alpha,
        "power_0": stacked.residual_power(0.0 + 0.0j),
        "power_1": stacked.residual_power(1.0 + 0.0j),
        "power_hat": stacked.residual_power(stacked.alpha),
        "n": stacked.n,
        "tt": stacked.tt,
    }


def fit_frequency_smooth_alpha(
    moments: Sequence[ComplexMoments],
    *,
    ridge: float = SMOOTH_RIDGE,
) -> dict[str, object]:
    """Per-channel α with first-difference smoothness. Not an average visibility."""

    channels = sorted({int(item.channel) for item in moments})
    if not channels:
        return {"alpha_channel": np.zeros(0, dtype=np.complex128), "power": float("nan")}
    index = {channel: i for i, channel in enumerate(channels)}
    n = len(channels)
    gram = np.zeros((n, n), dtype=np.complex128)
    matched = np.zeros(n, dtype=np.complex128)
    raw_power = 0.0
    for item in moments:
        i = index[int(item.channel)]
        gram[i, i] += item.tt
        matched[i] += item.tr
        raw_power += item.rr
    for i in range(n - 1):
        gram[i, i] += ridge
        gram[i + 1, i + 1] += ridge
        gram[i, i + 1] -= ridge
        gram[i + 1, i] -= ridge
    alpha = np.linalg.solve(gram, matched) if np.all(np.isfinite(gram)) else np.full(n, np.nan)
    mapped = np.zeros(max(channels) + 1, dtype=np.complex128)
    mapped[np.asarray(channels, dtype=np.int64)] = alpha
    power = residual_power_at(moments, mapped)
    return {
        "alpha_channel": mapped,
        "channels": tuple(channels),
        "power": power,
        "ridge": float(ridge),
        "raw_power": float(raw_power),
    }


def fit_spatial_shrinkage(
    moments: Sequence[ComplexMoments],
    *,
    ridge_toward_one: float = SPATIAL_RIDGE_TOWARD_ONE,
) -> dict[str, object]:
    """Array-average per-cell α shrunk toward the unit CASSBEAM template."""

    by_cell: dict[int, list[ComplexMoments]] = {}
    for item in moments:
        by_cell.setdefault(int(item.cell), []).append(item)
    alpha_cell = {}
    power = 0.0
    for cell, group in by_cell.items():
        stacked = stack_moments(group)
        alpha = (stacked.tr + ridge_toward_one) / (stacked.tt + ridge_toward_one)
        alpha_cell[int(cell)] = complex(alpha)
        power += stacked.residual_power(complex(alpha))
    return {
        "alpha_cell": alpha_cell,
        "power": float(power),
        "n_cells": len(alpha_cell),
        "ridge_toward_one": float(ridge_toward_one),
    }


def fit_per_antenna_if_supported(
    moments: Sequence[ComplexMoments],
    *,
    ridge_toward_array: float = ANTENNA_RIDGE_TOWARD_ARRAY,
) -> dict[str, object]:
    array = stack_moments(moments).alpha
    by_mover: dict[int, list[ComplexMoments]] = {}
    for item in moments:
        by_mover.setdefault(int(item.mover), []).append(item)
    alpha = {}
    power = 0.0
    for mover, group in by_mover.items():
        stacked = stack_moments(group)
        value = (stacked.tr + ridge_toward_array * array) / (stacked.tt + ridge_toward_array)
        alpha[int(mover)] = complex(value)
        power += stacked.residual_power(complex(value))
    return {
        "alpha_mover": alpha,
        "array_alpha": complex(array),
        "power": float(power),
        "n_movers": len(alpha),
    }


def convention_equivalence_classes(
    templates: Mapping[str, ArrayLike],
    weight: ArrayLike,
    *,
    relative_floor: float = TEMPLATE_EQUIV_REL,
) -> list[list[str]]:
    """Group conventions whose visibility templates are indistinguishable."""

    names = list(templates)
    planes = {name: vis_planes(templates[name])[:, 0] for name in names}
    w_all = np.asarray(weight, dtype=np.float64)
    if w_all.ndim == 4:
        w_all = np.mean(w_all, axis=1)
    w = w_all[:, 0, 1] + w_all[:, 1, 0]
    assigned = set()
    classes: list[list[str]] = []
    for name in names:
        if name in assigned:
            continue
        group = [name]
        assigned.add(name)
        ref = planes[name]
        denom = np.sum(w * (np.abs(ref[:, 0, 1]) ** 2 + np.abs(ref[:, 1, 0]) ** 2))
        for other in names:
            if other in assigned:
                continue
            cand = planes[other]
            num = np.sum(
                w
                * (
                    np.abs(ref[:, 0, 1] - cand[:, 0, 1]) ** 2
                    + np.abs(ref[:, 1, 0] - cand[:, 1, 0]) ** 2
                )
            )
            if denom > 0.0 and num / denom <= float(relative_floor):
                group.append(other)
                assigned.add(other)
        classes.append(group)
    return classes


def contiguous_channel_block_masks(
    n_channel: int,
    *,
    hold_start: int | None = None,
    hold_width: int = 16,
) -> dict[str, NDArray[np.bool_]]:
    """Hold out one contiguous native-channel block. Do not average channels."""

    n = int(n_channel)
    start = int(n - hold_width) if hold_start is None else int(hold_start)
    start = max(0, min(start, max(n - 1, 0)))
    stop = min(n, start + int(hold_width))
    hold = np.zeros(n, dtype=bool)
    hold[start:stop] = True
    return {"train": ~hold, "holdout": hold, "hold_start": start, "hold_stop": stop}


def inject_voltage_leakage(
    visibilities: ArrayLike,
    template: ArrayLike,
    amplitude: float,
    *,
    origin_mask: ArrayLike,
) -> NDArray[np.complex128]:
    """Add ``amplitude * template`` off origin. Template is already on-axis zero."""

    vis = np.array(visibilities, dtype=np.complex128, copy=True)
    t = np.asarray(template, dtype=np.complex128)
    if vis.ndim == 3:
        vis = vis[:, None, :, :]
    if t.ndim == 3:
        t = t[:, None, :, :]
    if vis.shape != t.shape:
        raise ValueError("injection template must match the visibility copies")
    origin = np.asarray(origin_mask, dtype=bool).reshape(-1)
    if origin.size != vis.shape[0]:
        raise ValueError("origin_mask must be one flag per row")
    scale = np.where(origin, 0.0, float(amplitude)).reshape(-1, 1, 1, 1)
    vis = vis + scale * np.where(np.isfinite(t), t, 0.0)
    return vis


def _moment_arrays(moments: Sequence[ComplexMoments]) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    labels = np.asarray([item.cluster_id for item in moments], dtype=np.int64)
    tt = np.asarray([item.tt for item in moments], dtype=np.float64)
    tr = np.asarray([item.tr for item in moments], dtype=np.complex128)
    rr = np.asarray([item.rr for item in moments], dtype=np.float64)
    return labels, tt, tr, rr


def bootstrap_complex_from_moments(
    moments: Sequence[ComplexMoments],
    *,
    n_boot: int = 400,
    seed: int = 0,
) -> dict[str, object]:
    labels, tt, tr, _rr = _moment_arrays(moments)
    stacked = stack_moments(moments)
    point = stacked.alpha
    rng = np.random.default_rng(int(seed))
    draws = np.empty(int(n_boot), dtype=np.complex128)
    for boot in range(int(n_boot)):
        index = resample_cluster_indices(labels, rng)
        denom = float(np.sum(tt[index]))
        draws[boot] = (np.sum(tr[index]) / denom) if denom > 0.0 else np.nan + 1j * np.nan
    finite = draws[np.isfinite(draws.real) & np.isfinite(draws.imag)]
    real_ci = _ci95(finite.real)
    imag_ci = _ci95(finite.imag)
    abs_draws = np.abs(finite)
    abs_ci = _ci95(abs_draws)
    return {
        "real": float(point.real),
        "imag": float(point.imag),
        "abs": float(np.abs(point)),
        "n_clusters": int(np.unique(labels).size),
        "n_boot": int(finite.size),
        "real_ci95": real_ci,
        "imag_ci95": imag_ci,
        "abs_ci95": abs_ci,
        "abs_ul95": float(np.quantile(abs_draws, 0.95)) if abs_draws.size else float("nan"),
        "consistent_with_zero": _contains(real_ci, 0.0) and _contains(imag_ci, 0.0),
        "consistent_with_one": _contains(real_ci, 1.0) and _contains(imag_ci, 0.0),
    }


def bootstrap_increment_from_moments(
    uninjected: Sequence[ComplexMoments],
    injected: Sequence[ComplexMoments],
    *,
    n_boot: int = 400,
    seed: int = 1,
) -> dict[str, object]:
    labels, tt0, tr0, _rr0 = _moment_arrays(uninjected)
    _lab1, tt1, tr1, _rr1 = _moment_arrays(injected)
    point = stack_moments(injected).alpha - stack_moments(uninjected).alpha
    rng = np.random.default_rng(int(seed))
    draws = np.empty(int(n_boot), dtype=np.complex128)
    for boot in range(int(n_boot)):
        index = resample_cluster_indices(labels, rng)
        d0 = float(np.sum(tt0[index]))
        d1 = float(np.sum(tt1[index]))
        a0 = (np.sum(tr0[index]) / d0) if d0 > 0.0 else np.nan + 1j * np.nan
        a1 = (np.sum(tr1[index]) / d1) if d1 > 0.0 else np.nan + 1j * np.nan
        draws[boot] = a1 - a0
    finite = draws[np.isfinite(draws.real) & np.isfinite(draws.imag)]
    real_ci = _ci95(finite.real)
    imag_ci = _ci95(finite.imag)
    return {
        "real": float(point.real),
        "imag": float(point.imag),
        "abs": float(np.abs(point)),
        "real_ci95": real_ci,
        "imag_ci95": imag_ci,
        "inconsistent_with_zero": not (_contains(real_ci, 0.0) and _contains(imag_ci, 0.0)),
    }


def paired_power_improves(
    moments: Sequence[ComplexMoments],
    alpha_full: complex,
    alpha_diag: complex = 0.0 + 0.0j,
    *,
    n_boot: int = 400,
    seed: int = 3,
) -> dict[str, object]:
    labels, tt, tr, rr = _moment_arrays(moments)
    rng = np.random.default_rng(int(seed))
    wins = 0
    full = complex(alpha_full)
    diag = complex(alpha_diag)

    def _power(index, alpha: complex) -> float:
        return float(
            np.sum(rr[index])
            - 2.0 * np.real(np.conjugate(alpha) * np.sum(tr[index]))
            + abs(alpha) ** 2 * np.sum(tt[index])
        )

    for _boot in range(int(n_boot)):
        index = resample_cluster_indices(labels, rng)
        if _power(index, full) < _power(index, diag):
            wins += 1
    fraction = wins / float(n_boot) if n_boot else float("nan")
    return {
        "fraction_positive": float(fraction),
        "improves": bool(fraction > 0.95),
        "power_full": residual_power_at(moments, alpha_full),
        "power_diag": residual_power_at(moments, alpha_diag),
    }


def classify_beam_prior_decision(
    *,
    software_ok: bool,
    split_clean: bool,
    injection_detectable: Mapping[str, bool],
    holdout_alpha: Mapping[str, object] | None,
    a1_improves: bool,
    hat_improves: bool,
    spatial_improves: bool,
    rr_ll_regression: bool,
    convention_resolved: bool = True,
) -> dict[str, object]:
    """Scientific outcomes are recorded. Process failures fail closed."""

    cassbeam_scale = any(
        bool(injection_detectable.get(key)) for key in ("0.003", "0.01", "0.03", 0.003, 0.01, 0.03)
    )
    if not software_ok:
        decision = "software_gate_failed"
        process_failure = True
    elif not split_clean:
        decision = "contaminated_split"
        process_failure = True
    elif not convention_resolved:
        decision = "convention_unresolved"
        process_failure = False
    elif not cassbeam_scale:
        decision = "upper_limit_only"
        process_failure = False
    elif rr_ll_regression:
        decision = "upper_limit_only"
        process_failure = False
    else:
        alpha = holdout_alpha or {}
        one = bool(alpha.get("consistent_with_one"))
        nonzero = alpha.get("consistent_with_zero") is False
        if a1_improves and one:
            decision = "cassbeam_morphology_supported"
        elif hat_improves and nonzero and not one:
            decision = "rescaled_cassbeam_morphology"
        elif spatial_improves and nonzero:
            decision = "empirical_offdiagonal_morphology"
        else:
            decision = "upper_limit_only"
        process_failure = False
    return {
        "gate": SPW4_MULTICHANNEL_BEAM_PRIOR,
        "decision": decision,
        "process_failure": process_failure,
        "scientific_decision": not process_failure,
        "status": "pass" if decision == "cassbeam_morphology_supported" else "fail",
        "blocking": decision != "cassbeam_morphology_supported",
        "full_jones_frozen": False,
        "spw5_closed": True,
        "most_important_next_artifact": "cassbeam_diagonal_low_order_correction",
        "notes": (PRIOR_NOTE, INJECTION_NOTE, CHANNEL_COVARIANCE_NOTE, NO_HOLORASTER_IN_DI_NOTE),
    }


def supported_prior_from_decision(
    *,
    decision: str,
    scalar: Mapping[str, object],
    smooth: Mapping[str, object],
    spatial: Mapping[str, object],
    antenna: Mapping[str, object],
    holdout_alpha: Mapping[str, object] | None,
    provenance: Mapping[str, object],
    antenna_supported: bool,
) -> HolographyBeamPrior:
    """Keep only quantities the holdouts actually support."""

    supported: list[str] = ["diagonal_beam"]
    off_mean = None
    off_upper = None
    freq_cov = None
    scatter = None
    if decision in {
        "cassbeam_morphology_supported",
        "rescaled_cassbeam_morphology",
        "empirical_offdiagonal_morphology",
    }:
        off_mean = complex(scalar.get("alpha_hat", 0.0))
        supported.append("off_diagonal_mean")
        if decision != "cassbeam_morphology_supported":
            freq_cov = np.asarray(smooth.get("alpha_channel"), dtype=np.complex128)
            supported.append("frequency_smooth_amplitude")
        if decision == "empirical_offdiagonal_morphology":
            supported.append("array_average_spatial")
        if antenna_supported:
            scatter = (
                float(
                    np.mean(
                        [
                            abs(value - antenna.get("array_alpha", 0.0))
                            for value in (antenna.get("alpha_mover") or {}).values()
                        ]
                    )
                )
                if antenna.get("alpha_mover")
                else None
            )
            if scatter is not None:
                supported.append("antenna_scatter")
    else:
        report = holdout_alpha or {}
        abs_hat = float(
            abs(complex(report.get("real", 0.0) or 0.0, report.get("imag", 0.0) or 0.0))
        )
        ci = report.get("real_ci95") or (float("nan"), float("nan"))
        imag_ci = report.get("imag_ci95") or (float("nan"), float("nan"))
        candidates = [abs_hat, abs(ci[0]), abs(ci[1]), abs(imag_ci[0]), abs(imag_ci[1])]
        finite = [value for value in candidates if np.isfinite(value)]
        off_upper = float(max(finite)) if finite else None
        supported.append("off_diagonal_upper_abs")
    return HolographyBeamPrior(
        diagonal_supported=True,
        off_diagonal_mean=off_mean,
        off_diagonal_upper_abs=off_upper,
        frequency_covariance=freq_cov,
        antenna_scatter=scatter,
        support_mask={"quantities": tuple(supported)},
        gauge="on_axis_identity",
        provenance=dict(provenance),
        supported_quantities=tuple(supported),
        decision=decision,
        frozen=False,
        full_jones_frozen=False,
        production_factory_modified=False,
        spw5_opened=False,
    )


def select_convention_class(
    class_scores: Mapping[str, float],
    classes: Sequence[Sequence[str]],
    *,
    margin: float = 0.02,
) -> dict[str, object]:
    """Pick a convention class from joint training residual power. Lower is better."""

    ranked = sorted(
        (
            (names[0], float(class_scores[names[0]]), list(names))
            for names in classes
            if names[0] in class_scores
        ),
        key=lambda item: item[1],
    )
    if not ranked:
        return {
            "status": "convention_unresolved",
            "selected": None,
            "classes": [list(c) for c in classes],
        }
    best_name, best_power, best_class = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else float("inf")
    unique = best_power * (1.0 + margin) < second if np.isfinite(second) else True
    return {
        "status": "locked" if unique else "convention_unresolved",
        "selected": best_name if unique else None,
        "selected_class": best_class if unique else None,
        "training_power": best_power,
        "unique_by_margin": unique,
        "n_classes": len(classes),
        "ranked": [(name, power) for name, power, _group in ranked[:8]],
    }


def qu_pairs() -> tuple[tuple[float, float], ...]:
    return QU_NUISANCE_PAIRS


def all_conventions() -> tuple[CassbeamConvention, ...]:
    return convention_ladder()


def spatial_convention_key(convention: CassbeamConvention) -> tuple[int, int, bool, bool]:
    return (convention.l_sign, convention.m_sign, convention.swap_lm, convention.rotate_spatial)


CASSBEAM_QUERY_DECIMALS = 12


def unique_native_jones(
    offset_lm_rad: ArrayLike,
    chi: ArrayLike,
    frequencies_hz: ArrayLike,
    catalog: HighresCassbeamCatalog,
    convention: CassbeamConvention,
) -> tuple[NDArray[np.complex128], NDArray[np.bool_], NDArray[np.int64]]:
    """Lookup native CASSBEAM Jones at unique query coordinates, all channels."""

    offset = np.asarray(offset_lm_rad, dtype=np.float64).reshape(-1, 2)
    angle = np.asarray(chi, dtype=np.float64).reshape(-1)
    freqs = np.asarray(frequencies_hz, dtype=np.float64).reshape(-1)
    l_q, m_q = apply_axis_convention(offset, convention)
    if convention.rotate_spatial:
        l_q, m_q = antenna_frame_lm(l_q, m_q, angle)
    keys = np.round(np.stack([l_q, m_q], axis=1), CASSBEAM_QUERY_DECIMALS)
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    native = np.zeros((unique.shape[0], freqs.size, 2, 2), dtype=np.complex128)
    valid = np.zeros((unique.shape[0], freqs.size), dtype=bool)
    for ich, freq in enumerate(freqs):
        plane = catalog.plane(float(freq))
        looked, ok = plane.lookup_native(unique[:, 0], unique[:, 1])
        native[:, ich] = looked
        valid[:, ich] = ok
    return native, valid, inverse.astype(np.int64)


def apply_parallactic_to_beams(
    beams: ArrayLike,
    chi: ArrayLike,
    calibration_state: BeamCalibrationState | str,
) -> NDArray[np.complex128]:
    state = require_beam_calibration_state(calibration_state)
    jones = np.asarray(beams, dtype=np.complex128)
    angle = np.asarray(chi, dtype=np.float64).reshape(-1)
    para = circular_parallactic_jones(angle)
    if jones.ndim == 4:
        para = para[:, None, :, :]
    if state is BeamCalibrationState.CASA_PARANG_TRUE:
        conjugate = np.conjugate(np.swapaxes(para, -1, -2))
        return conjugate @ jones @ para
    if state is BeamCalibrationState.UNCALIBRATED:
        return jones @ para
    raise ValueError(f"unsupported beam calibration state {state!r}")


def feed_frame_from_native_unique(
    native_unique: ArrayLike,
    valid_unique: ArrayLike,
    inverse: ArrayLike,
    convention: CassbeamConvention,
    *,
    catalog: HighresCassbeamCatalog,
    frequencies_hz: ArrayLike,
    off_diagonal: bool,
) -> tuple[NDArray[np.complex128], NDArray[np.bool_]]:
    """Normalize, optionally project, and expand unique native Jones. No parallactic."""

    native = np.asarray(native_unique, dtype=np.complex128)
    ok = np.asarray(valid_unique, dtype=bool)
    inv = np.asarray(inverse, dtype=np.int64).reshape(-1)
    freqs = np.asarray(frequencies_hz, dtype=np.float64).reshape(-1)
    looked = np.empty_like(native)
    for ich, freq in enumerate(freqs):
        origin = catalog.plane(float(freq)).origin_native()
        looked[:, ich] = normalize_after_jones_convention(native[:, ich], origin, convention)
    if not off_diagonal:
        looked = diagonal_projection(looked)
    return looked[inv], ok[inv]


def unique_feed_frame_jones(
    offset_lm_rad: ArrayLike,
    catalog: HighresCassbeamCatalog,
    frequencies_hz: ArrayLike,
    convention: CassbeamConvention,
    *,
    chi: ArrayLike | None = None,
    off_diagonal: bool = False,
) -> tuple[NDArray[np.complex128], NDArray[np.bool_]]:
    """Feed-frame Jones after the frozen unique-coordinate CASSBEAM lookup."""

    offset = np.asarray(offset_lm_rad, dtype=np.float64).reshape(-1, 2)
    if chi is None:
        angle = np.zeros(offset.shape[0], dtype=np.float64)
    else:
        angle = np.asarray(chi, dtype=np.float64).reshape(-1)
    native, valid, inverse = unique_native_jones(
        offset, angle, frequencies_hz, catalog, convention
    )
    return feed_frame_from_native_unique(
        native,
        valid,
        inverse,
        convention,
        catalog=catalog,
        frequencies_hz=frequencies_hz,
        off_diagonal=off_diagonal,
    )


def beams_from_native_unique(
    native_unique: ArrayLike,
    valid_unique: ArrayLike,
    inverse: ArrayLike,
    convention: CassbeamConvention,
    *,
    catalog: HighresCassbeamCatalog,
    frequencies_hz: ArrayLike,
    chi: ArrayLike,
    off_diagonal: bool,
    calibration_state: BeamCalibrationState | str,
) -> tuple[NDArray[np.complex128], NDArray[np.bool_]]:
    row_beams, row_ok = feed_frame_from_native_unique(
        native_unique,
        valid_unique,
        inverse,
        convention,
        catalog=catalog,
        frequencies_hz=frequencies_hz,
        off_diagonal=off_diagonal,
    )
    row_beams = apply_parallactic_to_beams(row_beams, chi, calibration_state)
    return row_beams, row_ok


def predict_from_moving_beams(
    residual_jones: Mapping[int, ArrayLike],
    moving_id: ArrayLike,
    reference_id: ArrayLike,
    moving_is_p: ArrayLike,
    chi_moving: ArrayLike,
    chi_reference: ArrayLike,
    beams: ArrayLike,
    source: ArrayLike,
) -> NDArray[np.complex128]:
    r_m = sky_frame_residual_numpy(_antenna_jones_planes(residual_jones, moving_id), chi_moving)
    r_r = sky_frame_residual_numpy(
        _antenna_jones_planes(residual_jones, reference_id), chi_reference
    )
    return predict_vis_numpy(r_m, beams, source, r_r, moving_is_p)


@dataclass(frozen=True)
class HolorasterCassbeamStages:
    """Shared CASSBEAM HOLORASTER stages. Feed frame has no parallactic rotation."""

    feed: NDArray[np.complex128]
    sky: NDArray[np.complex128]
    visibility: NDArray[np.complex128]
    ok: NDArray[np.bool_]

    def as_mapping(self) -> dict[str, NDArray[np.complex128]]:
        return {"feed": self.feed, "sky": self.sky, "visibility": self.visibility}


def squeeze_sample_jones(values: ArrayLike) -> NDArray[np.complex128]:
    """Drop a singleton native-channel axis. Shape becomes ``(sample, 2, 2)``."""

    array = np.asarray(values, dtype=np.complex128)
    if array.ndim == 4:
        if array.shape[1] != 1:
            raise ValueError("expected a single native channel")
        array = array[:, 0]
    if array.ndim != 3 or array.shape[-2:] != (2, 2):
        raise ValueError("Jones/visibilities must have shape (sample, 2, 2)")
    return array


def evaluate_holoraster_cassbeam(
    *,
    catalog: HighresCassbeamCatalog,
    frequencies_hz: ArrayLike,
    convention: CassbeamConvention,
    offset_lm_rad: ArrayLike,
    chi_moving: ArrayLike,
    chi_reference: ArrayLike,
    moving_id: ArrayLike,
    reference_id: ArrayLike,
    moving_is_p: ArrayLike,
    residual_jones: Mapping[int, ArrayLike],
    source: ArrayLike,
    off_diagonal: bool = False,
    calibration_state: BeamCalibrationState | str = "casa_parang_true",
    unique_lookup: tuple[ArrayLike, ArrayLike, ArrayLike] | None = None,
) -> HolorasterCassbeamStages:
    """One CASSBEAM evaluator for comparison and identity correction."""

    if unique_lookup is None:
        native, valid, inverse = unique_native_jones(
            offset_lm_rad, chi_moving, frequencies_hz, catalog, convention
        )
    else:
        native, valid, inverse = unique_lookup
    feed, ok = feed_frame_from_native_unique(
        native,
        valid,
        inverse,
        convention,
        catalog=catalog,
        frequencies_hz=frequencies_hz,
        off_diagonal=off_diagonal,
    )
    sky = apply_parallactic_to_beams(feed, chi_moving, calibration_state)
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
    vis = np.where(ok[..., None, None], vis, np.nan)
    return HolorasterCassbeamStages(feed=feed, sky=sky, visibility=vis, ok=ok)


def copolar_hand_errors(left: ArrayLike, right: ArrayLike) -> dict[str, dict[str, float]]:
    """Maximum absolute and relative RR/LL errors."""

    first = squeeze_sample_jones(left)
    second = squeeze_sample_jones(right)
    if first.shape != second.shape:
        raise ValueError("compared arrays must have the same shape")
    out: dict[str, dict[str, float]] = {}
    for name, i, j in (("RR", 0, 0), ("LL", 1, 1)):
        a = first[:, i, j]
        b = second[:, i, j]
        diff = np.abs(a - b)
        scale = np.maximum(np.abs(a), np.abs(b))
        rel = np.divide(diff, scale, out=np.zeros_like(diff), where=scale > 0.0)
        finite = np.isfinite(diff)
        out[name] = {
            "max_abs": float(np.max(diff[finite])) if bool(np.any(finite)) else float("nan"),
            "max_rel": float(np.max(rel[finite])) if bool(np.any(finite)) else float("nan"),
        }
    return out


def compare_holoraster_stages(
    left: Mapping[str, ArrayLike] | HolorasterCassbeamStages,
    right: Mapping[str, ArrayLike] | HolorasterCassbeamStages,
    *,
    atol: float = 1.0e-12,
) -> dict[str, object]:
    """Report the first stage where RR or LL exceeds ``atol``."""

    def _bundle(values: Mapping[str, ArrayLike] | HolorasterCassbeamStages) -> Mapping[str, ArrayLike]:
        if isinstance(values, HolorasterCassbeamStages):
            return values.as_mapping()
        return values

    first_map = _bundle(left)
    second_map = _bundle(right)
    report: dict[str, object] = {}
    first_stage: str | None = None
    for stage in ("feed", "sky", "visibility"):
        errors = copolar_hand_errors(first_map[stage], second_map[stage])
        report[stage] = errors
        if first_stage is None and (
            float(errors["RR"]["max_abs"]) > float(atol)
            or float(errors["LL"]["max_abs"]) > float(atol)
        ):
            first_stage = stage
    report["first_divergent_stage"] = first_stage
    return report


def template_increment(
    pred_full: ArrayLike,
    pred_diag: ArrayLike,
) -> NDArray[np.complex128]:
    return np.asarray(pred_full, dtype=np.complex128) - np.asarray(pred_diag, dtype=np.complex128)


def assert_split_isolation(train: ArrayLike, holdout: ArrayLike) -> None:
    train_m = np.asarray(train, dtype=bool).reshape(-1)
    hold_m = np.asarray(holdout, dtype=bool).reshape(-1)
    if train_m.size != hold_m.size:
        raise ValueError("train and holdout masks must have the same length")
    if bool(np.any(train_m & hold_m)):
        raise ValueError("contaminated split: train and holdout share rows")


def assert_origin_uninjected(
    origin: ArrayLike,
    injected: ArrayLike,
    original: ArrayLike,
) -> None:
    mask = np.asarray(origin, dtype=bool).reshape(-1)
    inj = np.asarray(injected)
    orig = np.asarray(original)
    if mask.size and not np.allclose(inj[mask], orig[mask], equal_nan=True):
        raise ValueError("origin rows were injected; copies only")


def moments_from_template(
    template: ArrayLike,
    residual: ArrayLike,
    weight: ArrayLike,
    geometry: Mapping[str, ArrayLike],
    row_index: ArrayLike,
    *,
    n_channel: int,
    spatial_groups: bool = True,
) -> list[ComplexMoments]:
    rows = np.asarray(row_index, dtype=np.int64).reshape(-1)
    _flat_t, _row_ids, chan_ids = flatten_row_channel(template, row_ids=rows)
    flat_r, _, _ = flatten_row_channel(residual)
    flat_w, _, _ = flatten_row_channel(weight)
    if _flat_t.shape[0] != rows.size * int(n_channel):
        raise ValueError("template flattening dropped or averaged channels")
    cluster = np.repeat(np.asarray(geometry["time_index"])[rows], n_channel)
    if spatial_groups:
        dwell = cluster
        mover = np.repeat(np.asarray(geometry["moving_id"])[rows], n_channel)
        reference = np.repeat(np.asarray(geometry["reference_id"])[rows], n_channel)
        cell = np.repeat(
            np.asarray(geometry["cell_l"])[rows] * 10_007 + np.asarray(geometry["cell_m"])[rows],
            n_channel,
        )
    else:
        dwell = np.zeros_like(cluster)
        mover = np.zeros_like(cluster)
        reference = np.zeros_like(cluster)
        cell = np.zeros_like(cluster)
    return accumulate_hand_moments(
        _flat_t,
        flat_r,
        flat_w,
        cluster_ids=cluster,
        channel_ids=chan_ids,
        dwell_ids=dwell,
        mover_ids=mover,
        reference_ids=reference,
        cell_ids=cell,
    )


def write_beam_prior_plots(
    output_dir: Path,
    *,
    frequencies_hz: ArrayLike,
    per_channel_alpha: ArrayLike,
    smooth_alpha: ArrayLike,
    measured: ArrayLike,
    predicted: ArrayLike,
    row_mask: ArrayLike,
    geometry: Mapping[str, ArrayLike],
    max_baselines: int = 4,
) -> list[str]:
    """Per-channel coefficients and measured-versus-predicted complex visibilities."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    freqs = np.asarray(frequencies_hz, dtype=np.float64).reshape(-1) / 1.0e6
    per_ch = np.asarray(per_channel_alpha, dtype=np.complex128).reshape(-1)
    smooth = np.asarray(smooth_alpha, dtype=np.complex128).reshape(-1)
    written: list[str] = []
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(freqs[: per_ch.size], per_ch.real, ".", label="per-channel")
    axes[0].plot(freqs[: smooth.size], smooth.real, "-", label="frequency-smooth")
    axes[0].set_ylabel("Re α")
    axes[0].legend()
    axes[1].plot(freqs[: per_ch.size], per_ch.imag, ".", label="per-channel")
    axes[1].plot(freqs[: smooth.size], smooth.imag, "-", label="frequency-smooth")
    axes[1].set_ylabel("Im α")
    axes[1].set_xlabel("Frequency (MHz)")
    path = output_dir / "frequency_smooth_alpha.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    written.append(str(path))

    meas = vis_planes(measured)
    pred = vis_planes(predicted)
    mask = np.asarray(row_mask, dtype=bool).reshape(-1)
    mover = np.asarray(geometry["moving_id"]).reshape(-1)
    reference = np.asarray(geometry["reference_id"]).reshape(-1)
    pairs = np.stack([mover[mask], reference[mask]], axis=1)
    if pairs.size == 0:
        return written
    unique, counts = np.unique(pairs, axis=0, return_counts=True)
    selected = unique[np.argsort(-counts)[: int(max_baselines)]]
    for mv, rf in selected:
        choose = mask & (mover == int(mv)) & (reference == int(rf))
        fig, axes = plt.subplots(2, 2, figsize=(8, 8))
        hands = (("RL", 0, 1), ("LR", 1, 0), ("RR", 0, 0), ("LL", 1, 1))
        for axis, (name, i, j) in zip(axes.ravel(), hands, strict=True):
            m = meas[choose, :, i, j].reshape(-1)
            p = pred[choose, :, i, j].reshape(-1)
            axis.plot(m.real, m.imag, ".", alpha=0.4, label="measured")
            axis.plot(p.real, p.imag, ".", alpha=0.4, label="predicted")
            axis.set_title(f"{name} mover {int(mv)} ref {int(rf)}")
            axis.set_aspect("equal", adjustable="box")
        axes[0, 0].legend()
        path = output_dir / f"vis_complex_mover{int(mv)}_ref{int(rf)}.png"
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        written.append(str(path))
    return written


def prior_to_dict(prior: HolographyBeamPrior) -> dict[str, object]:
    freq = prior.frequency_covariance
    return {
        "diagonal_supported": prior.diagonal_supported,
        "off_diagonal_mean": (
            [prior.off_diagonal_mean.real, prior.off_diagonal_mean.imag]
            if prior.off_diagonal_mean is not None
            else None
        ),
        "off_diagonal_upper_abs": prior.off_diagonal_upper_abs,
        "frequency_covariance": None if freq is None else [[c.real, c.imag] for c in freq],
        "antenna_scatter": prior.antenna_scatter,
        "support_mask": dict(prior.support_mask),
        "gauge": prior.gauge,
        "provenance": dict(prior.provenance),
        "supported_quantities": list(prior.supported_quantities),
        "decision": prior.decision,
        "frozen": prior.frozen,
        "full_jones_frozen": prior.full_jones_frozen,
        "production_factory_modified": prior.production_factory_modified,
        "spw5_opened": prior.spw5_opened,
        "notes": list(prior.notes),
    }
