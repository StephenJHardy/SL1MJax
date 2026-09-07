"""Direct HOLORASTER versus CASSBEAM comparison report.

Every calibrated moving–reference visibility is compared with the locked
mount-frame CASSBEAM prediction at the measured AZELGEO coordinate, time
and native channel. There is no model selection. Diagonal panels are the
scientific comparison. Full-Jones panels are labelled experimental.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sl1mjax.holography_alignment import (
    apparent_voltage_response,
    score_copolar_residuals,
    voltage_response_region_masks,
)
from sl1mjax.holography_beam_prior import flatten_row_channel, vis_planes
from sl1mjax.beam_conventions import evla195_total_squint_rad
from sl1mjax.holography_c147_offset_ring import ns_ew_masks
from sl1mjax.holography_diagonal_diagnostics import SQUINT_MAINLOBE_POWER_FRACTION
from sl1mjax.holography_highres_cassbeam import DEFAULT_CONVENTION

HOLORASTER_CASSBEAM_COMPARISON = "holoraster_cassbeam_comparison_report"
CASSBEAM_DIAGONAL_REFERENCE = "cassbeam_diagonal_cband_reference"
CASSBEAM_DIAGONAL_CORRECTION = "cassbeam_diagonal_low_order_correction"
FULL_JONES_EXPERIMENTAL_LABEL = "experimental prediction"
DIAGONAL_NULL_LABEL = "diagonal prediction (null)"
REPRESENTATIVE_BASELINE_COUNT = 4
SPATIAL_MAP_BINS = 51
MAIN_LOBE_ACCEPTED_MAX = 0.01
MID_BEAM_QUALIFIED_MAX = 0.12
REGION_SUPPORT_CLASS = {
    "main_lobe": "accepted",
    "mid": "qualified",
    "outer_diagnostic": "diagnostic",
}

COMPARISON_NOTE = (
    "HOLORASTER visibilities are compared with CASSBEAM at the measured "
    "antenna, coordinate, time and native channel. No coefficient is fitted "
    "and no model is selected."
)
DIAGONAL_NOTE = (
    "The diagonal CASSBEAM beam is the reference C-band copolar model. "
    "These plots test whether that morphology describes the holography."
)
FULL_JONES_NOTE = (
    "Full-Jones panels use the locked mount-frame convention and are "
    "labelled experimental prediction. They are diagnostics, not a "
    "validated off-diagonal measurement. The 128-member convention "
    "ladder is not searched."
)
SPW5_CLOSED_NOTE = (
    "SPW 5 stays sealed until a small, physically interpretable "
    "correction is frozen on SPW-4 development data."
)


def locked_convention():
    return DEFAULT_CONVENTION


def refuse_convention_search(conventions: Sequence[object] | None) -> None:
    if conventions is not None and len(tuple(conventions)) > 1:
        raise RuntimeError("HOLORASTER comparison does not search CASSBEAM conventions")


def spatial_quadrant_masks(offset_lm_rad: ArrayLike) -> dict[str, NDArray[np.bool_]]:
    """Cardinal quadrants plus the E–W / N–S axes used by the offset ring."""

    offset = np.asarray(offset_lm_rad, dtype=np.float64).reshape(-1, 2)
    l_rad = offset[:, 0]
    m_rad = offset[:, 1]
    finite = np.isfinite(l_rad) & np.isfinite(m_rad)
    return {
        "east": finite & (l_rad > np.abs(m_rad)),
        "west": finite & (l_rad < -np.abs(m_rad)),
        "north": finite & (m_rad > np.abs(l_rad)),
        "south": finite & (m_rad < -np.abs(l_rad)),
        **ns_ew_masks(offset),
    }


def weighted_hand_stats(
    observed: ArrayLike,
    predicted: ArrayLike,
    weight: ArrayLike,
) -> dict[str, float]:
    """Complex correlation, slope and residual power for one visibility hand."""

    obs = np.asarray(observed, dtype=np.complex128).reshape(-1)
    pred = np.asarray(predicted, dtype=np.complex128).reshape(-1)
    wgt = np.asarray(weight, dtype=np.float64).reshape(-1)
    finite = np.isfinite(obs) & np.isfinite(pred) & np.isfinite(wgt) & (wgt > 0.0)
    obs = obs[finite]
    pred = pred[finite]
    wgt = wgt[finite]
    if obs.size == 0:
        return {
            "n": 0,
            "slope_real": float("nan"),
            "slope_imag": float("nan"),
            "correlation_abs": float("nan"),
            "residual_power": float("nan"),
            "median_abs_obs": float("nan"),
            "median_abs_pred": float("nan"),
            "median_abs_ratio": float("nan"),
        }
    tt = float(np.sum(wgt * np.abs(pred) ** 2))
    rr = float(np.sum(wgt * np.abs(obs) ** 2))
    tr = complex(np.sum(wgt * np.conjugate(pred) * obs))
    residual = float(np.sum(wgt * np.abs(obs - pred) ** 2))
    slope = tr / tt if tt > 0.0 else complex(np.nan, np.nan)
    denom = np.sqrt(tt * rr) if tt > 0.0 and rr > 0.0 else np.nan
    med_obs = float(np.median(np.abs(obs)))
    med_pred = float(np.median(np.abs(pred)))
    return {
        "n": int(obs.size),
        "slope_real": float(slope.real),
        "slope_imag": float(slope.imag),
        "correlation_abs": float(np.abs(tr) / denom) if np.isfinite(denom) else float("nan"),
        "residual_power": residual / rr if rr > 0.0 else float("nan"),
        "median_abs_obs": med_obs,
        "median_abs_pred": med_pred,
        "median_abs_ratio": med_obs / med_pred if med_pred > 0.0 else float("nan"),
    }


def visibility_hand_summaries(
    measured: ArrayLike,
    predicted: ArrayLike,
    weight: ArrayLike,
    row_mask: ArrayLike,
) -> dict[str, dict[str, float]]:
    """Per-hand comparison on the selected rows, keeping every native channel."""

    meas = vis_planes(measured)
    pred = vis_planes(predicted)
    wgt = vis_planes(weight).real.astype(np.float64)
    mask = np.asarray(row_mask, dtype=bool).reshape(-1)
    if meas.shape[0] != mask.size:
        raise ValueError("row_mask length must match the visibility row axis")
    meas = meas[mask]
    pred = pred[mask]
    wgt = wgt[mask]
    out: dict[str, dict[str, float]] = {}
    for name, row, col in (("rr", 0, 0), ("ll", 1, 1), ("rl", 0, 1), ("lr", 1, 0)):
        out[name] = weighted_hand_stats(meas[..., row, col], pred[..., row, col], wgt[..., row, col])
    return out


def region_copolar_summaries(
    measured: ArrayLike,
    predicted: ArrayLike,
    weight: ArrayLike,
    intensity: ArrayLike,
    rr_ok: ArrayLike,
    ll_ok: ArrayLike,
    row_mask: ArrayLike,
    voltage: ArrayLike | None = None,
) -> dict[str, object]:
    """Main-lobe, mid-beam and outer-raster copolar scores plus residual power."""

    scored = score_copolar_residuals(
        measured,
        predicted,
        intensity,
        rr_ok,
        ll_ok,
        row_mask,
        voltage=voltage,
    )
    meas = vis_planes(measured)
    pred = vis_planes(predicted)
    wgt = vis_planes(weight).real.astype(np.float64)
    mask = np.asarray(row_mask, dtype=bool).reshape(-1)
    if voltage is None:
        voltage = apparent_voltage_response(meas[:, 0], intensity, rr_ok, ll_ok)
    regions = {"all": mask}
    for name, region in voltage_response_region_masks(voltage).items():
        regions[name] = region & mask
    residual_power = {}
    for name, region in regions.items():
        residual_power[name] = visibility_hand_summaries(meas, pred, wgt, region)
    scored["hand_residual_power"] = residual_power
    return scored


def quadrant_crosshand_summaries(
    measured: ArrayLike,
    predicted_full: ArrayLike,
    predicted_diag: ArrayLike,
    weight: ArrayLike,
    offset_lm_rad: ArrayLike,
    row_mask: ArrayLike,
) -> dict[str, object]:
    """RL/LR correlation of the experimental full-Jones prediction by quadrant."""

    meas = vis_planes(measured)
    full = vis_planes(predicted_full)
    diag = vis_planes(predicted_diag)
    wgt = vis_planes(weight).real.astype(np.float64)
    mask = np.asarray(row_mask, dtype=bool).reshape(-1)
    offset = np.asarray(offset_lm_rad, dtype=np.float64).reshape(-1, 2)
    n_chan = int(meas.shape[1])
    increment = full - diag
    residual = meas - diag
    quadrants = spatial_quadrant_masks(offset)
    out: dict[str, object] = {
        "full_jones_label": FULL_JONES_EXPERIMENTAL_LABEL,
        "diagonal_label": DIAGONAL_NULL_LABEL,
    }
    for name, quadrant in quadrants.items():
        choose = mask & quadrant
        if n_chan > 1:
            choose_flat = np.repeat(choose, n_chan)
            inc, _, _ = flatten_row_channel(increment)
            res, _, _ = flatten_row_channel(residual)
            ww, _, _ = flatten_row_channel(wgt)
            stats = {
                hand: weighted_hand_stats(
                    res[choose_flat, row, col],
                    inc[choose_flat, row, col],
                    ww[choose_flat, row, col],
                )
                for hand, row, col in (("rl", 0, 1), ("lr", 1, 0))
            }
        else:
            stats = {
                hand: weighted_hand_stats(
                    residual[choose, 0, row, col],
                    increment[choose, 0, row, col],
                    wgt[choose, 0, row, col],
                )
                for hand, row, col in (("rl", 0, 1), ("lr", 1, 0))
            }
        out[name] = stats
    return out


def binned_complex_map(
    offset_lm_rad: ArrayLike,
    values: ArrayLike,
    weight: ArrayLike,
    *,
    n_bin: int = SPATIAL_MAP_BINS,
) -> dict[str, NDArray]:
    """Weighted-mean complex map on a regular (l, m) grid. Vectorized."""

    offset = np.asarray(offset_lm_rad, dtype=np.float64).reshape(-1, 2)
    val = np.asarray(values, dtype=np.complex128).reshape(-1)
    wgt = np.asarray(weight, dtype=np.float64).reshape(-1)
    if offset.shape[0] != val.size:
        raise ValueError("offset and values must share the sample axis")
    l_arcmin = offset[:, 0] * (180.0 * 60.0 / np.pi)
    m_arcmin = offset[:, 1] * (180.0 * 60.0 / np.pi)
    finite = (
        np.isfinite(l_arcmin)
        & np.isfinite(m_arcmin)
        & np.isfinite(val)
        & np.isfinite(wgt)
        & (wgt > 0.0)
    )
    span = float(np.nanmax(np.abs(np.concatenate([l_arcmin[finite], m_arcmin[finite]]))))
    span = max(span, 1.0)
    edges = np.linspace(-span, span, int(n_bin) + 1)
    ix = np.clip(np.digitize(l_arcmin, edges) - 1, 0, n_bin - 1)
    iy = np.clip(np.digitize(m_arcmin, edges) - 1, 0, n_bin - 1)
    flat = iy * n_bin + ix
    count = np.zeros(n_bin * n_bin, dtype=np.float64)
    real = np.zeros(n_bin * n_bin, dtype=np.float64)
    imag = np.zeros(n_bin * n_bin, dtype=np.float64)
    np.add.at(count, flat[finite], wgt[finite])
    np.add.at(real, flat[finite], wgt[finite] * val[finite].real)
    np.add.at(imag, flat[finite], wgt[finite] * val[finite].imag)
    safe = count > 0.0
    mean = np.full(n_bin * n_bin, np.nan, dtype=np.complex128)
    mean[safe] = real[safe] / count[safe] + 1j * imag[safe] / count[safe]
    centers = 0.5 * (edges[:-1] + edges[1:])
    return {
        "l_arcmin": centers,
        "m_arcmin": centers,
        "mean": mean.reshape(n_bin, n_bin),
        "weight": count.reshape(n_bin, n_bin),
    }


def mainlobe_power_mask(
    power: ArrayLike,
    weight: ArrayLike,
    *,
    power_fraction: float = SQUINT_MAINLOBE_POWER_FRACTION,
) -> NDArray[np.bool_]:
    """Independent 20%-of-peak mask for one circular hand."""

    pwr = np.asarray(power, dtype=np.float64).reshape(-1)
    wgt = np.asarray(weight, dtype=np.float64).reshape(-1)
    finite = np.isfinite(pwr) & np.isfinite(wgt) & (wgt > 0.0) & (pwr > 0.0)
    if not bool(np.any(finite)):
        return finite
    peak = float(np.max(pwr[finite]))
    return finite & (pwr >= float(power_fraction) * peak)


def power_centroid_lm(
    offset_lm_rad: ArrayLike,
    power: ArrayLike,
    weight: ArrayLike,
    *,
    mask: ArrayLike | None = None,
) -> tuple[float, float]:
    """Power-weighted (l, m) centroid in radians on the supplied mask."""

    offset = np.asarray(offset_lm_rad, dtype=np.float64).reshape(-1, 2)
    pwr = np.asarray(power, dtype=np.float64).reshape(-1)
    wgt = np.asarray(weight, dtype=np.float64).reshape(-1)
    if mask is None:
        choose = np.isfinite(pwr) & np.isfinite(wgt) & (wgt > 0.0) & (pwr > 0.0)
        choose = choose & np.isfinite(offset).all(axis=1)
    else:
        choose = np.asarray(mask, dtype=bool).reshape(-1)
    mass = wgt * pwr
    total = float(np.sum(mass[choose]))
    if total <= 0.0:
        return (float("nan"), float("nan"))
    return (
        float(np.sum(mass[choose] * offset[choose, 0]) / total),
        float(np.sum(mass[choose] * offset[choose, 1]) / total),
    )


def _centroid_record(offset_lm_rad, power, weight, *, power_fraction: float) -> dict[str, float]:
    full = power_centroid_lm(offset_lm_rad, power, weight)
    lobe = mainlobe_power_mask(power, weight, power_fraction=power_fraction)
    centre = power_centroid_lm(offset_lm_rad, power, weight, mask=lobe)
    return {
        "l_rad": centre[0],
        "m_rad": centre[1],
        "l_arcmin": float(centre[0] * 180.0 * 60.0 / np.pi),
        "m_arcmin": float(centre[1] * 180.0 * 60.0 / np.pi),
        "full_raster_l_arcmin": float(full[0] * 180.0 * 60.0 / np.pi),
        "full_raster_m_arcmin": float(full[1] * 180.0 * 60.0 / np.pi),
        "n": int(np.sum(lobe)),
    }


def squint_from_voltage_maps(
    offset_lm_rad: ArrayLike,
    rr_power: ArrayLike,
    ll_power: ArrayLike,
    rr_weight: ArrayLike,
    ll_weight: ArrayLike | None = None,
    *,
    frequency_hz: float | None = None,
    series: str = "measured",
    power_fraction: float = SQUINT_MAINLOBE_POWER_FRACTION,
) -> dict[str, object]:
    """Publication RR/LL squint: 20%-of-peak main-lobe centroids, hands independent.

    ``ll_weight`` may be omitted only in tests that share one positive weight.
    Full-raster centroids are retained as a bias diagnostic and are not the
    published separation.
    """

    if ll_weight is None:
        ll_weight = rr_weight
    rr = _centroid_record(offset_lm_rad, rr_power, rr_weight, power_fraction=power_fraction)
    ll = _centroid_record(offset_lm_rad, ll_power, ll_weight, power_fraction=power_fraction)
    sep = float(np.hypot(rr["l_rad"] - ll["l_rad"], rr["m_rad"] - ll["m_rad"]) * 180.0 * 60.0 / np.pi)
    full_sep = float(
        np.hypot(
            rr["full_raster_l_arcmin"] - ll["full_raster_l_arcmin"],
            rr["full_raster_m_arcmin"] - ll["full_raster_m_arcmin"],
        )
    )
    out: dict[str, object] = {
        "series": str(series),
        "estimator": "mainlobe_20pct_peak",
        "power_fraction": float(power_fraction),
        "publication_estimator": True,
        "rr_l_arcmin": rr["l_arcmin"],
        "rr_m_arcmin": rr["m_arcmin"],
        "ll_l_arcmin": ll["l_arcmin"],
        "ll_m_arcmin": ll["m_arcmin"],
        "separation_arcmin": sep,
        "n_rr": rr["n"],
        "n_ll": ll["n"],
        "full_raster_separation_arcmin": full_sep,
        "full_raster_is_publication_estimator": False,
    }
    if frequency_hz is not None:
        freq = float(frequency_hz)
        out["frequency_hz"] = freq
        memo = float(np.asarray(evla195_total_squint_rad(freq)).reshape(-1)[0] * 180.0 * 60.0 / np.pi)
        out["memo195_separation_arcmin"] = memo
    return out


def rescore_squint_from_comparison_arrays(
    measured: ArrayLike,
    predicted_diag: ArrayLike,
    weight: ArrayLike,
    offset_lm_rad: ArrayLike,
    *,
    frequency_hz: float,
) -> dict[str, object]:
    """Recompute publication squint from saved HOLORASTER arrays. No new predict."""

    meas = vis_planes(measured)
    pred = vis_planes(predicted_diag)
    wgt = vis_planes(weight).real.astype(np.float64)
    offset = np.asarray(offset_lm_rad, dtype=np.float64).reshape(-1, 2)
    return {
        "measured": squint_from_voltage_maps(
            offset,
            np.abs(meas[:, 0, 0, 0]) ** 2,
            np.abs(meas[:, 0, 1, 1]) ** 2,
            wgt[:, 0, 0, 0],
            wgt[:, 0, 1, 1],
            frequency_hz=frequency_hz,
            series="measured",
        ),
        "cassbeam": squint_from_voltage_maps(
            offset,
            np.abs(pred[:, 0, 0, 0]) ** 2,
            np.abs(pred[:, 0, 1, 1]) ** 2,
            wgt[:, 0, 0, 0],
            wgt[:, 0, 1, 1],
            frequency_hz=frequency_hz,
            series="cassbeam",
        ),
        "frequency_hz": float(frequency_hz),
        "predictions_recomputed": False,
    }


def representative_baseline_ids(
    moving_id: ArrayLike,
    reference_id: ArrayLike,
    row_mask: ArrayLike,
    *,
    count: int = REPRESENTATIVE_BASELINE_COUNT,
) -> NDArray[np.int32]:
    """Most-populated mover–reference pairs. No visibility-row Python loop."""

    mask = np.asarray(row_mask, dtype=bool).reshape(-1)
    pairs = np.stack(
        [
            np.asarray(moving_id, dtype=np.int32).reshape(-1)[mask],
            np.asarray(reference_id, dtype=np.int32).reshape(-1)[mask],
        ],
        axis=1,
    )
    if pairs.size == 0:
        return np.zeros((0, 2), dtype=np.int32)
    unique, counts = np.unique(pairs, axis=0, return_counts=True)
    return unique[np.argsort(-counts)[: int(count)]].astype(np.int32)


def residual_geometry_summaries(
    residual: ArrayLike,
    weight: ArrayLike,
    offset_lm_rad: ArrayLike,
    channel_id: ArrayLike,
    row_mask: ArrayLike,
) -> dict[str, object]:
    """Median residual amplitude versus radius, azimuth and channel."""

    res = vis_planes(residual)
    wgt = vis_planes(weight).real.astype(np.float64)
    mask = np.asarray(row_mask, dtype=bool).reshape(-1)
    offset = np.asarray(offset_lm_rad, dtype=np.float64).reshape(-1, 2)
    n_chan = int(res.shape[1])
    radius = np.hypot(offset[:, 0], offset[:, 1]) * (180.0 * 60.0 / np.pi)
    azimuth = np.rad2deg(np.arctan2(offset[:, 0], offset[:, 1]))
    chan = np.asarray(channel_id, dtype=np.int64).reshape(-1)
    if chan.size == 1:
        chan = np.full(n_chan, int(chan[0]), dtype=np.int64)
    out: dict[str, object] = {}
    for name, row, col in (("rr", 0, 0), ("ll", 1, 1), ("rl", 0, 1), ("lr", 1, 0)):
        amp = np.abs(res[:, :, row, col])
        ww = wgt[:, :, row, col]
        finite = mask[:, None] & np.isfinite(amp) & np.isfinite(ww) & (ww > 0.0)
        out[name] = {
            "median_vs_radius": _binned_median(np.broadcast_to(radius[:, None], amp.shape)[finite], amp[finite]),
            "median_vs_azimuth": _binned_median(
                np.broadcast_to(azimuth[:, None], amp.shape)[finite],
                amp[finite],
            ),
            "median_vs_channel": _binned_median(
                np.broadcast_to(chan[None, :], amp.shape)[finite],
                amp[finite],
            ),
        }
    return out


def _binned_median(x: ArrayLike, y: ArrayLike, *, n_bin: int = 24) -> dict[str, list[float]]:
    xx = np.asarray(x, dtype=np.float64).reshape(-1)
    yy = np.asarray(y, dtype=np.float64).reshape(-1)
    finite = np.isfinite(xx) & np.isfinite(yy)
    xx = xx[finite]
    yy = yy[finite]
    if xx.size == 0:
        return {"x": [], "median": []}
    lo, hi = float(np.min(xx)), float(np.max(xx))
    if lo == hi:
        return {"x": [lo], "median": [float(np.median(yy))]}
    edges = np.linspace(lo, hi, int(n_bin) + 1)
    index = np.clip(np.digitize(xx, edges) - 1, 0, n_bin - 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    order = np.argsort(index, kind="mergesort")
    grouped = index[order]
    values = yy[order]
    starts = np.flatnonzero(np.r_[True, grouped[1:] != grouped[:-1]])
    ends = np.r_[starts[1:], values.size]
    medians = np.full(n_bin, np.nan, dtype=np.float64)
    for start, end in zip(starts, ends, strict=True):
        medians[int(grouped[start])] = float(np.median(values[start:end]))
    keep = np.isfinite(medians)
    return {"x": centers[keep].tolist(), "median": medians[keep].tolist()}


def classify_diagonal_region_support(
    hand_residual_power: Mapping[str, Mapping[str, Mapping[str, float]]],
) -> dict[str, object]:
    """Region-qualified diagonal support. Not a single whole-raster label."""

    regions: dict[str, object] = {}
    for name, support in REGION_SUPPORT_CLASS.items():
        hands = hand_residual_power.get(name) or {}
        rr = float((hands.get("rr") or {}).get("residual_power", np.nan))
        ll = float((hands.get("ll") or {}).get("residual_power", np.nan))
        if name == "main_lobe":
            matches = bool(np.isfinite(rr) and np.isfinite(ll) and rr <= MAIN_LOBE_ACCEPTED_MAX and ll <= MAIN_LOBE_ACCEPTED_MAX)
        elif name == "mid":
            matches = bool(np.isfinite(rr) and np.isfinite(ll) and rr <= MID_BEAM_QUALIFIED_MAX and ll <= MID_BEAM_QUALIFIED_MAX)
        else:
            matches = True
        regions[name] = {
            "class": support,
            "residual_power_rr": rr,
            "residual_power_ll": ll,
            "matches_class": matches,
        }
    return {
        "artifact": CASSBEAM_DIAGONAL_REFERENCE,
        "main_lobe": "accepted",
        "mid_beam": "qualified",
        "outer_raster": "diagnostic",
        "regions": regions,
        "full_jones": "experimental_non_detection",
        "spw5_role": "sealed_diagonal_frequency_transfer",
        "convention_search_reopened": False,
    }


def classify_holoraster_comparison_report(
    *,
    software_ok: bool,
    predictions_finite: bool,
    plots_written: bool,
    diagonal_support: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """The report is a diagnostic. It never selects a beam model."""

    if not software_ok:
        decision = "software_gate_failed"
        process_failure = True
    elif not predictions_finite:
        decision = "predictions_not_finite"
        process_failure = True
    elif not plots_written:
        decision = "plots_missing"
        process_failure = True
    else:
        decision = "report_written"
        process_failure = False
    support = dict(diagonal_support or {})
    closed = bool(decision == "report_written" and support.get("main_lobe") == "accepted")
    return {
        "gate": HOLORASTER_CASSBEAM_COMPARISON,
        "decision": decision,
        "diagonal_reference": CASSBEAM_DIAGONAL_REFERENCE if closed else None,
        "diagonal_support": support,
        "model_selected": False,
        "process_failure": process_failure,
        "scientific_decision": not process_failure,
        "status": "pass" if decision == "report_written" else "fail",
        "blocking": process_failure,
        "full_jones_label": FULL_JONES_EXPERIMENTAL_LABEL,
        "full_jones_frozen": False,
        "production_factory_modified": False,
        "spw5_closed": True,
        "spw5_opened": False,
        "spw5_full_jones_sealed": True,
        "convention_search_reopened": False,
        "most_important_next_artifact": (
            CASSBEAM_DIAGONAL_CORRECTION if closed else HOLORASTER_CASSBEAM_COMPARISON
        ),
        "notes": (COMPARISON_NOTE, DIAGONAL_NOTE, FULL_JONES_NOTE, SPW5_CLOSED_NOTE),
    }


def _one_to_one_limits(x: NDArray, y: NDArray) -> tuple[float, float]:
    finite = np.isfinite(x) & np.isfinite(y)
    if not bool(np.any(finite)):
        return (-1.0, 1.0)
    lo = float(np.min([np.min(x[finite]), np.min(y[finite])]))
    hi = float(np.max([np.max(x[finite]), np.max(y[finite])]))
    if lo == hi:
        return (lo - 1.0, hi + 1.0)
    pad = 0.05 * (hi - lo)
    return (lo - pad, hi + pad)


def _scatter_sample(values: NDArray, *, max_points: int = 25_000) -> NDArray:
    flat = np.asarray(values).reshape(-1)
    if flat.size <= max_points:
        return flat
    choose = np.linspace(0, flat.size - 1, max_points, dtype=np.int64)
    return flat[choose]


def _scatter_one_to_one(axis, observed: NDArray, predicted: NDArray, *, xlabel: str, ylabel: str) -> None:
    pred = _scatter_sample(predicted)
    obs = _scatter_sample(observed)
    axis.scatter(pred, obs, s=4, alpha=0.2, linewidths=0)
    lo, hi = _one_to_one_limits(np.asarray(predicted).reshape(-1), np.asarray(observed).reshape(-1))
    axis.plot([lo, hi], [lo, hi], "k-", lw=0.8, label="one-to-one")
    axis.set_xlim(lo, hi)
    axis.set_ylim(lo, hi)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_aspect("equal", adjustable="box")


def _imshow_map(axis, mapped: Mapping[str, NDArray], *, title: str, cmap: str) -> None:
    grid = np.asarray(mapped["mean"])
    l_ax = np.asarray(mapped["l_arcmin"])
    m_ax = np.asarray(mapped["m_arcmin"])
    extent = (float(l_ax[0]), float(l_ax[-1]), float(m_ax[0]), float(m_ax[-1]))
    image = axis.imshow(
        np.abs(grid),
        origin="lower",
        extent=extent,
        cmap=cmap,
        aspect="equal",
    )
    axis.set_title(title)
    axis.set_xlabel("l (arcmin)")
    axis.set_ylabel("m (arcmin)")
    axis.figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)


def write_holoraster_comparison_plots(
    output_dir: Path,
    *,
    measured: ArrayLike,
    predicted_diag: ArrayLike,
    predicted_full: ArrayLike,
    weight: ArrayLike,
    offset_lm_rad: ArrayLike,
    moving_id: ArrayLike,
    reference_id: ArrayLike,
    row_mask: ArrayLike,
    channel_id: ArrayLike,
    squint_records: Sequence[Mapping[str, float]] | None = None,
) -> list[str]:
    """Write the eight HOLORASTER comparison families. No model selection."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    meas = vis_planes(measured)
    diag = vis_planes(predicted_diag)
    full = vis_planes(predicted_full)
    wgt = vis_planes(weight).real.astype(np.float64)
    mask = np.asarray(row_mask, dtype=bool).reshape(-1)
    offset = np.asarray(offset_lm_rad, dtype=np.float64).reshape(-1, 2)
    mover = np.asarray(moving_id, dtype=np.int32).reshape(-1)
    reference = np.asarray(reference_id, dtype=np.int32).reshape(-1)

    fig, axes = plt.subplots(2, 4, figsize=(14.0, 7.0))
    for col, (hand, i, j) in enumerate((("RR", 0, 0), ("LL", 1, 1))):
        m = meas[mask, :, i, j].reshape(-1)
        p = diag[mask, :, i, j].reshape(-1)
        _scatter_one_to_one(
            axes[0, col],
            m.real,
            p.real,
            xlabel=f"CASSBEAM {hand} Re (Jy)",
            ylabel=f"observed {hand} Re (Jy)",
        )
        _scatter_one_to_one(
            axes[1, col],
            m.imag,
            p.imag,
            xlabel=f"CASSBEAM {hand} Im (Jy)",
            ylabel=f"observed {hand} Im (Jy)",
        )
        axes[0, col].set_title(f"{hand} real")
        axes[1, col].set_title(f"{hand} imag")
    for col, (hand, i, j) in enumerate((("RR", 0, 0), ("LL", 1, 1))):
        m = meas[mask, :, i, j].reshape(-1)
        p = diag[mask, :, i, j].reshape(-1)
        _scatter_one_to_one(
            axes[0, col + 2],
            np.abs(m),
            np.abs(p),
            xlabel=f"CASSBEAM |{hand}| (Jy)",
            ylabel=f"observed |{hand}| (Jy)",
        )
        _scatter_one_to_one(
            axes[1, col + 2],
            np.angle(m, deg=True),
            np.angle(p, deg=True),
            xlabel=f"CASSBEAM {hand} phase (deg)",
            ylabel=f"observed {hand} phase (deg)",
        )
        axes[0, col + 2].set_title(f"{hand} amplitude")
        axes[1, col + 2].set_title(f"{hand} phase")
    axes[0, 0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Observed versus diagonal CASSBEAM")
    fig.tight_layout()
    path = root / "obs_vs_diag_rr_ll.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    written.append(str(path))

    for hand, i, j, cmap in (("RR", 0, 0, "viridis"), ("LL", 1, 1, "viridis")):
        fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0))
        w_hand = wgt[mask, 0, i, j]
        maps = (
            ("measured", meas[mask, 0, i, j]),
            ("CASSBEAM", diag[mask, 0, i, j]),
            ("residual", meas[mask, 0, i, j] - diag[mask, 0, i, j]),
        )
        for axis, (title, values) in zip(axes, maps, strict=True):
            _imshow_map(
                axis,
                binned_complex_map(offset[mask], values, w_hand),
                title=f"{hand} {title}",
                cmap=cmap if title != "residual" else "magma",
            )
        fig.tight_layout()
        path = root / f"spatial_{hand.lower()}_measured_cassbeam_residual.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        written.append(str(path))

    residual = meas - diag
    geometry = residual_geometry_summaries(residual, wgt, offset, channel_id, mask)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0))
    for axis, key, xlabel in (
        (axes[0], "median_vs_radius", "radius (arcmin)"),
        (axes[1], "median_vs_azimuth", "azimuth (deg)"),
        (axes[2], "median_vs_channel", "channel"),
    ):
        for hand, color in (("rr", "C0"), ("ll", "C1")):
            series = geometry[hand][key]
            axis.plot(series["x"], series["median"], "o-", color=color, label=hand.upper())
        axis.set_xlabel(xlabel)
        axis.set_ylabel("median |residual| (Jy)")
        axis.legend(fontsize=8)
    fig.tight_layout()
    path = root / "residual_vs_radius_azimuth_channel.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    written.append(str(path))

    pairs = representative_baseline_ids(mover, reference, mask)
    for mv, rf in pairs:
        choose = mask & (mover == int(mv)) & (reference == int(rf))
        fig, axes = plt.subplots(2, 2, figsize=(8.5, 8.0))
        for axis, (hand, i, j) in zip(
            axes.ravel(),
            (("RR", 0, 0), ("LL", 1, 1), ("RL", 0, 1), ("LR", 1, 0)),
            strict=True,
        ):
            m = meas[choose, :, i, j].reshape(-1)
            p = diag[choose, :, i, j].reshape(-1) if hand in {"RR", "LL"} else full[choose, :, i, j].reshape(-1)
            label = "CASSBEAM" if hand in {"RR", "LL"} else FULL_JONES_EXPERIMENTAL_LABEL
            _scatter_one_to_one(
                axis,
                m.real,
                p.real,
                xlabel=f"{label} Re (Jy)",
                ylabel=f"observed {hand} Re (Jy)",
            )
            axis.set_title(f"{hand} mover {int(mv)} ref {int(rf)}")
        fig.tight_layout()
        path = root / f"baseline_mover{int(mv)}_ref{int(rf)}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        written.append(str(path))

    fig, axes = plt.subplots(2, 2, figsize=(8.5, 8.0))
    for axis, (hand, i, j) in zip(
        axes[0],
        (("RL", 0, 1), ("LR", 1, 0)),
        strict=True,
    ):
        m = meas[mask, :, i, j].reshape(-1)
        p_full = full[mask, :, i, j].reshape(-1)
        p_diag = diag[mask, :, i, j].reshape(-1)
        axis.scatter(
            _scatter_sample(p_full.real),
            _scatter_sample(m.real),
            s=4,
            alpha=0.2,
            label=FULL_JONES_EXPERIMENTAL_LABEL,
        )
        axis.scatter(
            _scatter_sample(p_diag.real),
            _scatter_sample(m.real),
            s=4,
            alpha=0.2,
            label=DIAGONAL_NULL_LABEL,
        )
        lo, hi = _one_to_one_limits(p_full.real, m.real)
        axis.plot([lo, hi], [lo, hi], "k-", lw=0.8)
        axis.set_title(f"{hand} real")
        axis.set_xlabel("predicted Re (Jy)")
        axis.set_ylabel("observed Re (Jy)")
        axis.set_aspect("equal", adjustable="box")
    for axis, (hand, i, j) in zip(
        axes[1],
        (("RL", 0, 1), ("LR", 1, 0)),
        strict=True,
    ):
        m = meas[mask, :, i, j].reshape(-1)
        p_full = full[mask, :, i, j].reshape(-1)
        p_diag = diag[mask, :, i, j].reshape(-1)
        axis.scatter(
            _scatter_sample(p_full.imag),
            _scatter_sample(m.imag),
            s=4,
            alpha=0.2,
            label=FULL_JONES_EXPERIMENTAL_LABEL,
        )
        axis.scatter(
            _scatter_sample(p_diag.imag),
            _scatter_sample(m.imag),
            s=4,
            alpha=0.2,
            label=DIAGONAL_NULL_LABEL,
        )
        lo, hi = _one_to_one_limits(p_full.imag, m.imag)
        axis.plot([lo, hi], [lo, hi], "k-", lw=0.8)
        axis.set_title(f"{hand} imag")
        axis.set_xlabel("predicted Im (Jy)")
        axis.set_ylabel("observed Im (Jy)")
        axis.set_aspect("equal", adjustable="box")
    axes[0, 0].legend(loc="upper left", fontsize=7)
    fig.suptitle("Observed RL/LR versus experimental full Jones")
    fig.tight_layout()
    path = root / "obs_vs_full_jones_rl_lr.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    written.append(str(path))

    quadrants = spatial_quadrant_masks(offset)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.2))
    increment = full - diag
    residual = meas - diag
    labels = ("east_west", "north_south", "east", "west", "north", "south")
    for axis, (hand, i, j) in zip(axes, (("RL", 0, 1), ("LR", 1, 0)), strict=True):
        xs = []
        corrs = []
        alphas = []
        for name in labels:
            choose = mask & quadrants[name]
            stats = weighted_hand_stats(
                residual[choose, :, i, j],
                increment[choose, :, i, j],
                wgt[choose, :, i, j],
            )
            xs.append(name.replace("_", "\n"))
            corrs.append(stats["correlation_abs"])
            alphas.append(np.hypot(stats["slope_real"], stats["slope_imag"]))
        axis.bar(np.arange(len(xs)) - 0.18, corrs, width=0.36, label="|correlation|")
        axis.bar(np.arange(len(xs)) + 0.18, alphas, width=0.36, label="|α|")
        axis.set_xticks(np.arange(len(xs)), xs, fontsize=8)
        axis.set_title(f"{hand} experimental full Jones")
        axis.set_ylim(0.0, 1.05)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    path = root / "crosshand_quadrant_correlation.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    written.append(str(path))

    return written


def write_squint_publication_plot(
    output_dir: Path,
    *,
    measured: Mapping[str, object],
    cassbeam: Mapping[str, object],
) -> list[str]:
    """Measured and CASSBEAM 20%-of-peak squint as separate series."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.2))
    for axis, record, title in (
        (axes[0], measured, "measured"),
        (axes[1], cassbeam, "CASSBEAM"),
    ):
        axis.scatter(
            [float(record["rr_l_arcmin"])],
            [float(record["rr_m_arcmin"])],
            marker="o",
            label="RR",
        )
        axis.scatter(
            [float(record["ll_l_arcmin"])],
            [float(record["ll_m_arcmin"])],
            marker="s",
            label="LL",
        )
        axis.plot(
            [float(record["rr_l_arcmin"]), float(record["ll_l_arcmin"])],
            [float(record["rr_m_arcmin"]), float(record["ll_m_arcmin"])],
            "k-",
            lw=0.8,
        )
        axis.set_title(
            f"{title} 20%-of-peak  {float(record['separation_arcmin']):.3f}′"
        )
        axis.set_xlabel("l (arcmin)")
        axis.set_ylabel("m (arcmin)")
        axis.set_aspect("equal", adjustable="box")
        axis.axhline(0.0, color="k", lw=0.4)
        axis.axvline(0.0, color="k", lw=0.4)
        axis.legend(fontsize=8)
    fig.suptitle("Publication squint estimator (full-raster centroid is not shown)")
    fig.tight_layout()
    path = root / "squint_mainlobe_20pct.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return [str(path)]
