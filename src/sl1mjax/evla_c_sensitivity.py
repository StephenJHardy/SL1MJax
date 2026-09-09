"""Predeclared full-Jones sensitivity and classification.

Paired loss is candidate minus diagonal; negative is better. Absence of
improvement is not ``disfavoured``. Production acceptance is never set
here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sl1mjax.evla_c_metrics import copolar_nonregression, residual_power
from sl1mjax.evla_c_validation_refresh import (
    BOOTSTRAP_N,
    BOOTSTRAP_SEED,
    INJECTION_SEED,
    OUTCOMES,
    RESIDUAL_JONES_FREQUENCY_POLICY,
)

INJECTION_AMPLITUDES = (0.0, 0.3, 1.0, 3.0)
INJECTION_PHASES_DEG = (0.0, 90.0, 180.0)
CROSS_HANDS = ("RL", "LR")


def inject_offdiagonal_visibility(
    visibility: ArrayLike,
    template: ArrayLike,
    *,
    amplitude: float,
    phase_rad: float = 0.0,
) -> NDArray[np.complex128]:
    """Add ``amplitude * e^{iφ} * (V_full - V_diag)`` to a copied visibility."""

    vis = np.array(visibility, dtype=np.complex128, copy=True)
    delta = np.asarray(template, dtype=np.complex128)
    if vis.shape != delta.shape:
        raise ValueError("injection template must match the visibility shape")
    vis = vis + float(amplitude) * np.exp(1j * float(phase_rad)) * delta
    return vis


def manufactured_nontemplate_leakage(
    measured: ArrayLike,
    *,
    seed: int = INJECTION_SEED,
    amplitude: float = 0.05,
) -> NDArray[np.complex128]:
    """A leakage increment that is not the EVLA-C full-minus-diagonal template."""

    vis = np.asarray(measured, dtype=np.complex128)
    rng = np.random.default_rng(int(seed))
    noise = (rng.normal(size=vis.shape) + 1j * rng.normal(size=vis.shape)) / np.sqrt(2.0)
    leak = np.zeros_like(vis)
    leak[..., 0, 1] = amplitude * (1.0 + 0.2j) * noise[..., 0, 1]
    leak[..., 1, 0] = amplitude * (0.7 - 0.3j) * noise[..., 1, 0]
    return vis + leak


def bootstrap_delta(
    measured: ArrayLike,
    candidate: ArrayLike,
    baseline: ArrayLike,
    weight: ArrayLike,
    cluster_ids: ArrayLike,
    hand: str,
    *,
    n_boot: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float]:
    meas = np.asarray(measured)
    cand = np.asarray(candidate)
    base = np.asarray(baseline)
    wgt = np.asarray(weight)
    clusters = np.asarray(cluster_ids, dtype=np.int64).reshape(-1)
    if clusters.size != meas.shape[0]:
        raise ValueError("cluster_ids must have one entry per sample")
    unique = np.unique(clusters)
    if unique.size == 0:
        return {
            "delta": float("nan"),
            "delta_lo": float("nan"),
            "delta_hi": float("nan"),
            "n_clusters": 0,
        }
    point = residual_power(meas, cand, wgt, hand) - residual_power(meas, base, wgt, hand)
    rng = np.random.default_rng(int(seed))
    draws = np.empty(int(n_boot), dtype=np.float64)
    _, inverse = np.unique(clusters, return_inverse=True)
    n_cluster = int(unique.size)
    for iboot in range(int(n_boot)):
        chosen = rng.integers(0, n_cluster, size=n_cluster)
        multiplicity = np.bincount(chosen, minlength=n_cluster)[inverse]
        draws[iboot] = residual_power(
            meas, cand, wgt, hand, multiplicity=multiplicity
        ) - residual_power(meas, base, wgt, hand, multiplicity=multiplicity)
    finite = draws[np.isfinite(draws)]
    if finite.size == 0:
        lo = hi = float("nan")
    else:
        lo = float(np.quantile(finite, 0.025))
        hi = float(np.quantile(finite, 0.975))
    return {
        "delta": float(point),
        "delta_lo": lo,
        "delta_hi": hi,
        "n_clusters": int(unique.size),
        "improves": bool(np.isfinite(lo) and hi < 0.0),
    }


def _mean_crosshand_delta(
    measured: ArrayLike,
    candidate: ArrayLike,
    baseline: ArrayLike,
    weight: ArrayLike,
) -> float:
    deltas = [
        residual_power(measured, candidate, weight, hand)
        - residual_power(measured, baseline, weight, hand)
        for hand in CROSS_HANDS
    ]
    return float(np.nanmean(deltas))


def pooled_crosshand_delta(
    measured: ArrayLike,
    candidate: ArrayLike,
    baseline: ArrayLike,
    weight: ArrayLike,
    cluster_ids: ArrayLike,
    *,
    n_boot: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, object]:
    per_hand = {
        hand: bootstrap_delta(
            measured,
            candidate,
            baseline,
            weight,
            cluster_ids,
            hand,
            n_boot=n_boot,
            seed=seed + i,
        )
        for i, hand in enumerate(CROSS_HANDS)
    }
    meas = np.asarray(measured)
    cand = np.asarray(candidate)
    base = np.asarray(baseline)
    wgt = np.asarray(weight)
    clusters = np.asarray(cluster_ids, dtype=np.int64).reshape(-1)
    unique = np.unique(clusters)
    point = _mean_crosshand_delta(meas, cand, base, wgt)
    if unique.size == 0:
        pooled = {
            "delta": float("nan"),
            "delta_lo": float("nan"),
            "delta_hi": float("nan"),
            "n_clusters": 0,
            "improves": False,
        }
        return {"RL": per_hand["RL"], "LR": per_hand["LR"], "pooled_RL_LR": pooled}
    rng = np.random.default_rng(int(seed) + 17)
    draws = np.empty(int(n_boot), dtype=np.float64)
    _, inverse = np.unique(clusters, return_inverse=True)
    n_cluster = int(unique.size)
    for iboot in range(int(n_boot)):
        chosen = rng.integers(0, n_cluster, size=n_cluster)
        multiplicity = np.bincount(chosen, minlength=n_cluster)[inverse]
        deltas = [
            residual_power(meas, cand, wgt, hand, multiplicity=multiplicity)
            - residual_power(meas, base, wgt, hand, multiplicity=multiplicity)
            for hand in CROSS_HANDS
        ]
        draws[iboot] = float(np.nanmean(deltas))
    finite = draws[np.isfinite(draws)]
    lo = float(np.quantile(finite, 0.025)) if finite.size else float("nan")
    hi = float(np.quantile(finite, 0.975)) if finite.size else float("nan")
    pooled = {
        "delta": float(point),
        "delta_lo": lo,
        "delta_hi": hi,
        "n_clusters": int(unique.size),
        "improves": bool(np.isfinite(lo) and hi < 0.0),
    }
    return {"RL": per_hand["RL"], "LR": per_hand["LR"], "pooled_RL_LR": pooled}


def mover_improvement_count(
    measured: ArrayLike,
    candidate: ArrayLike,
    baseline: ArrayLike,
    weight: ArrayLike,
    moving_id: ArrayLike,
    holdout_ids: Sequence[int],
    *,
    hands: tuple[str, ...] = CROSS_HANDS,
) -> dict[str, object]:
    meas = np.asarray(measured)
    cand = np.asarray(candidate)
    base = np.asarray(baseline)
    wgt = np.asarray(weight)
    movers = np.asarray(moving_id, dtype=np.int32).reshape(-1)
    rows = []
    n_improve = 0
    for antenna in holdout_ids:
        choose = movers == int(antenna)
        if not bool(np.any(choose)):
            rows.append({"moving_id": int(antenna), "improves": False, "n": 0})
            continue
        deltas = [
            residual_power(meas[choose], cand[choose], wgt[choose], hand)
            - residual_power(meas[choose], base[choose], wgt[choose], hand)
            for hand in hands
        ]
        mean_delta = float(np.nanmean(deltas))
        improves = bool(np.isfinite(mean_delta) and mean_delta < 0.0)
        n_improve += int(improves)
        rows.append(
            {
                "moving_id": int(antenna),
                "delta": mean_delta,
                "improves": improves,
                "n": int(np.sum(choose)),
            }
        )
    return {
        "movers": rows,
        "n_improving": n_improve,
        "n_holdout": len(tuple(holdout_ids)),
        "at_least_four": n_improve >= 4,
    }


def classify_full_jones(
    *,
    spatial: Mapping[str, object],
    mover: Mapping[str, object],
    mover_units: Mapping[str, object],
    copolar: Mapping[str, object],
    injection_detects_unit: bool,
    injection_zero_is_null: bool,
    numerical_ok: bool,
    residual_jones_policy: str = RESIDUAL_JONES_FREQUENCY_POLICY,
) -> dict[str, object]:
    """Return one of the four predeclared experiment outcomes."""

    if not numerical_ok:
        outcome = "blocked_implementation_or_contract"
        reason = "numerical or provenance failure"
    elif not injection_detects_unit:
        outcome = "inconclusive_sensitivity"
        reason = "unit-model injection was not recovered on holdout copies"
    else:
        spatial_pool = dict(spatial.get("pooled_RL_LR") or {})
        mover_pool = dict(mover.get("pooled_RL_LR") or {})
        spatial_ok = bool(spatial_pool.get("improves"))
        mover_ok = bool(mover_pool.get("improves"))
        hands_ok = bool(
            dict(spatial.get("RL") or {}).get("improves")
            and dict(spatial.get("LR") or {}).get("improves")
            and dict(mover.get("RL") or {}).get("improves")
            and dict(mover.get("LR") or {}).get("improves")
        )
        movers_ok = bool(mover_units.get("at_least_four"))
        copolar_ok = bool(copolar.get("passes"))
        if spatial_ok and mover_ok and hands_ok and movers_ok and copolar_ok:
            outcome = "supported_on_spw4_development"
            reason = "paired RL/LR intervals below zero on both partitions"
        elif (
            injection_detects_unit
            and injection_zero_is_null
            and spatial_ok is False
            and mover_ok is False
        ):
            # Detectable unit model, real data disagrees on both partitions.
            spatial_hi = spatial_pool.get("delta_lo")
            mover_hi = mover_pool.get("delta_lo")
            both_above = bool(
                np.isfinite(spatial_hi)
                and np.isfinite(mover_hi)
                and float(spatial_pool.get("delta_lo")) > 0.0
                and float(mover_pool.get("delta_lo")) > 0.0
            )
            if both_above:
                outcome = "disfavoured_on_tested_support"
                reason = "unit prediction is detectable but real-data intervals lie above zero"
            else:
                outcome = "inconclusive_sensitivity"
                reason = "no consistent paired improvement; not a positive disfavour"
        else:
            outcome = "inconclusive_sensitivity"
            reason = "the experiment cannot distinguish unit full Jones from diagonal"
    if outcome not in OUTCOMES:
        raise RuntimeError(f"invalid outcome {outcome!r}")
    return {
        "outcome": outcome,
        "reason": reason,
        "production_accepted": False,
        "residual_jones_frequency_policy": residual_jones_policy,
        "spatial_improves": bool(dict(spatial.get("pooled_RL_LR") or {}).get("improves")),
        "mover_improves": bool(dict(mover.get("pooled_RL_LR") or {}).get("improves")),
        "mover_units": dict(mover_units),
        "copolar": dict(copolar),
        "injection_detects_unit": bool(injection_detects_unit),
        "injection_zero_is_null": bool(injection_zero_is_null),
    }


def require_copolar_nonregression(
    full_scores: Mapping[str, Mapping[str, float]],
    diagonal_scores: Mapping[str, Mapping[str, float]],
) -> dict[str, object]:
    return copolar_nonregression(full_scores, diagonal_scores)
