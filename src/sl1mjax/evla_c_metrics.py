"""Visibility-domain scores for the EVLA-C validation refresh.

Residual power uses ``sum w |V-hat|^2 / sum w |V|^2``. Relative RMS is
the square root of that ratio. Jy residuals and ``|ΔV|/I`` are reported
separately so a power denominator is never confused with unattenuated I.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sl1mjax.evla_c_independent_rime import HAND_INDEX
from sl1mjax.evla_c_validation_refresh import (
    MAIN_LOBE_NONREGRESSION_ABS,
    MAIN_LOBE_NONREGRESSION_FRAC,
)

HANDS = ("RR", "RL", "LR", "LL")


def _planes(values: ArrayLike, name: str) -> NDArray:
    array = np.asarray(values)
    if array.ndim == 4:
        if array.shape[1] != 1:
            raise ValueError(f"{name} must be one native channel, got {array.shape}")
        array = array[:, 0]
    if array.ndim != 3 or array.shape[-2:] != (2, 2):
        raise ValueError(f"{name} must have shape (sample, 2, 2)")
    return array


def hand_ok(
    measured: ArrayLike,
    predicted: ArrayLike,
    weight: ArrayLike,
    hand: str,
) -> NDArray[np.bool_]:
    meas = _planes(measured, "measured")
    pred = _planes(predicted, "predicted")
    wgt = _planes(weight, "weight").real
    row, col = HAND_INDEX[hand]
    return (
        np.isfinite(meas[:, row, col])
        & np.isfinite(pred[:, row, col])
        & np.isfinite(wgt[:, row, col])
        & (wgt[:, row, col] > 0.0)
    )


def residual_power(
    measured: ArrayLike,
    predicted: ArrayLike,
    weight: ArrayLike,
    hand: str,
    *,
    multiplicity: ArrayLike | None = None,
) -> float:
    meas = _planes(measured, "measured")
    pred = _planes(predicted, "predicted")
    wgt = _planes(weight, "weight").real
    row, col = HAND_INDEX[hand]
    ok = hand_ok(meas, pred, wgt, hand)
    if not bool(np.any(ok)):
        return float("nan")
    sample_w = wgt[ok, row, col]
    if multiplicity is not None:
        extra = np.asarray(multiplicity, dtype=np.float64).reshape(-1)
        if extra.size != meas.shape[0]:
            raise ValueError("multiplicity must have one entry per sample")
        sample_w = sample_w * extra[ok]
    denom = float(np.sum(sample_w * np.abs(meas[ok, row, col]) ** 2))
    if denom <= 0.0:
        return float("nan")
    residual = np.abs(meas[ok, row, col] - pred[ok, row, col]) ** 2
    return float(np.sum(sample_w * residual) / denom)


def residual_rms(power: float) -> float:
    if not np.isfinite(power) or power < 0.0:
        return float("nan")
    return float(np.sqrt(power))


def median_abs_residual_jy(
    measured: ArrayLike,
    predicted: ArrayLike,
    weight: ArrayLike,
    hand: str,
) -> float:
    meas = _planes(measured, "measured")
    pred = _planes(predicted, "predicted")
    wgt = _planes(weight, "weight").real
    row, col = HAND_INDEX[hand]
    ok = hand_ok(meas, pred, wgt, hand)
    if not bool(np.any(ok)):
        return float("nan")
    return float(np.median(np.abs(meas[ok, row, col] - pred[ok, row, col])))


def median_abs_over_i(
    measured: ArrayLike,
    predicted: ArrayLike,
    weight: ArrayLike,
    hand: str,
    source_i_jy: float,
) -> float:
    if not np.isfinite(source_i_jy) or source_i_jy <= 0.0:
        raise ValueError("source_i_jy must be a positive finite scale")
    return median_abs_residual_jy(measured, predicted, weight, hand) / float(source_i_jy)


def complex_slope(
    measured: ArrayLike,
    predicted: ArrayLike,
    weight: ArrayLike,
    hand: str,
) -> complex:
    meas = _planes(measured, "measured")
    pred = _planes(predicted, "predicted")
    wgt = _planes(weight, "weight").real
    row, col = HAND_INDEX[hand]
    ok = hand_ok(meas, pred, wgt, hand)
    if not bool(np.any(ok)):
        return complex("nan")
    denom = float(np.sum(wgt[ok, row, col] * np.abs(pred[ok, row, col]) ** 2))
    if denom <= 0.0:
        return complex("nan")
    return complex(
        np.sum(wgt[ok, row, col] * np.conjugate(pred[ok, row, col]) * meas[ok, row, col]) / denom
    )


def complex_correlation(
    measured: ArrayLike,
    predicted: ArrayLike,
    weight: ArrayLike,
    hand: str,
) -> float:
    meas = _planes(measured, "measured")
    pred = _planes(predicted, "predicted")
    wgt = _planes(weight, "weight").real
    row, col = HAND_INDEX[hand]
    ok = hand_ok(meas, pred, wgt, hand)
    if not bool(np.any(ok)):
        return float("nan")
    num = np.abs(np.sum(wgt[ok, row, col] * np.conjugate(pred[ok, row, col]) * meas[ok, row, col]))
    denom = np.sqrt(
        np.sum(wgt[ok, row, col] * np.abs(pred[ok, row, col]) ** 2)
        * np.sum(wgt[ok, row, col] * np.abs(meas[ok, row, col]) ** 2)
    )
    if denom <= 0.0:
        return float("nan")
    return float(num / denom)


def rr_minus_ll_residual_power(
    measured: ArrayLike,
    predicted: ArrayLike,
    weight: ArrayLike,
) -> float:
    meas = _planes(measured, "measured")
    pred = _planes(predicted, "predicted")
    wgt = _planes(weight, "weight").real
    obs = meas[:, 0, 0] - meas[:, 1, 1]
    hat = pred[:, 0, 0] - pred[:, 1, 1]
    ww = 0.5 * (wgt[:, 0, 0] + wgt[:, 1, 1])
    ok = np.isfinite(obs) & np.isfinite(hat) & (ww > 0.0)
    if not bool(np.any(ok)):
        return float("nan")
    denom = float(np.sum(ww[ok] * np.abs(obs[ok]) ** 2))
    if denom <= 0.0:
        return float("nan")
    return float(np.sum(ww[ok] * np.abs(obs[ok] - hat[ok]) ** 2) / denom)


def hand_scores(
    measured: ArrayLike,
    predicted: ArrayLike,
    weight: ArrayLike,
    *,
    source_i_jy: float,
    hands: tuple[str, ...] = HANDS,
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for hand in hands:
        power = residual_power(measured, predicted, weight, hand)
        slope = complex_slope(measured, predicted, weight, hand)
        out[hand] = {
            "n": int(np.sum(hand_ok(measured, predicted, weight, hand))),
            "residual_power": power,
            "residual_rms": residual_rms(power),
            "median_abs_jy": median_abs_residual_jy(measured, predicted, weight, hand),
            "median_abs_over_i": median_abs_over_i(
                measured, predicted, weight, hand, source_i_jy
            ),
            "correlation_abs": complex_correlation(measured, predicted, weight, hand),
            "slope_real": float(np.real(slope)) if np.isfinite(slope) else float("nan"),
            "slope_imag": float(np.imag(slope)) if np.isfinite(slope) else float("nan"),
        }
    out["RR_minus_LL"] = {
        "n": int(
            np.sum(
                hand_ok(measured, predicted, weight, "RR")
                & hand_ok(measured, predicted, weight, "LL")
            )
        ),
        "residual_power": rr_minus_ll_residual_power(measured, predicted, weight),
    }
    return out


def paired_delta(candidate_loss: float, baseline_loss: float) -> float:
    return float(candidate_loss - baseline_loss)


def main_lobe_nonregression_limit(diagonal_loss: float) -> float:
    if not np.isfinite(diagonal_loss):
        return float("nan")
    return float(max(MAIN_LOBE_NONREGRESSION_ABS, MAIN_LOBE_NONREGRESSION_FRAC * diagonal_loss))


def copolar_nonregression(
    full_scores: Mapping[str, Mapping[str, float]],
    diagonal_scores: Mapping[str, Mapping[str, float]],
    *,
    hands: tuple[str, ...] = ("RR", "LL"),
) -> dict[str, object]:
    out: dict[str, object] = {}
    passed = True
    for hand in hands:
        diag = float(diagonal_scores[hand]["residual_power"])
        full = float(full_scores[hand]["residual_power"])
        limit = main_lobe_nonregression_limit(diag)
        delta = paired_delta(full, diag)
        ok = bool(np.isfinite(delta) and np.isfinite(limit) and delta <= limit)
        out[hand] = {
            "diagonal_loss": diag,
            "full_loss": full,
            "delta": delta,
            "limit": limit,
            "passes": ok,
        }
        passed = passed and ok
    out["passes"] = passed
    return out
