"""Cross-validated recovery tests for temporal and spectral sky components."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from sl1mjax.data.canonical import VisibilityBlock


@dataclass(frozen=True)
class BaselineHoldout:
    """Disjoint discovery and evaluation samples split by whole baselines."""

    discovery_mask: np.ndarray
    evaluation_mask: np.ndarray
    discovery_baselines: tuple[tuple[int, int], ...]
    evaluation_baselines: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class RecoveryScore:
    """Weighted residual statistics on one fixed sample set."""

    sample_count: int
    weight_sum: float
    residual_power: float
    weighted_complex_mse: float


@dataclass(frozen=True)
class ComponentRecoveryFit:
    """A real sky coefficient fitted on discovery baselines and scored elsewhere."""

    coefficient: float
    information_scale: float
    discovery_sample_count: int
    evaluation_sample_count: int
    null_evaluation: RecoveryScore
    component_evaluation: RecoveryScore
    null_supported_evaluation: RecoveryScore
    component_supported_evaluation: RecoveryScore
    prediction: np.ndarray

    @property
    def evaluation_relative_improvement(self) -> float:
        before = self.null_evaluation.residual_power
        if before <= 0:
            return float("nan")
        return 1.0 - self.component_evaluation.residual_power / before

    @property
    def supported_evaluation_relative_improvement(self) -> float:
        before = self.null_supported_evaluation.residual_power
        if before <= 0:
            return float("nan")
        return 1.0 - self.component_supported_evaluation.residual_power / before


def split_native_baselines(
    block: VisibilityBlock,
    *,
    evaluation_fraction: float = 0.25,
    seed: int = 0,
) -> BaselineHoldout:
    """Split complete antenna baselines, preserving all times and frequencies.

    A component is fitted on ``discovery_baselines`` and must predict different
    antenna pairs in ``evaluation_baselines``. This prevents baseline-local
    interference from validating itself as a sky component.
    """

    if not 0 < evaluation_fraction < 1:
        raise ValueError("evaluation_fraction must be between zero and one")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    first = np.minimum(block.antenna1, block.antenna2)
    second = np.maximum(block.antenna1, block.antenna2)
    baselines = tuple(
        sorted({(int(a), int(b)) for a, b in zip(first, second, strict=True)})
    )
    if len(baselines) < 2:
        raise ValueError("at least two baselines are required for a holdout")
    evaluation_count = int(round(evaluation_fraction * len(baselines)))
    evaluation_count = min(max(evaluation_count, 1), len(baselines) - 1)
    shuffled = np.random.default_rng(seed).permutation(len(baselines))
    evaluation_indices = set(int(value) for value in shuffled[:evaluation_count])
    evaluation_baselines = tuple(
        baseline for index, baseline in enumerate(baselines) if index in evaluation_indices
    )
    discovery_baselines = tuple(
        baseline for index, baseline in enumerate(baselines) if index not in evaluation_indices
    )
    evaluation_lookup = set(evaluation_baselines)
    evaluation_rows = np.fromiter(
        (
            (int(a), int(b)) in evaluation_lookup
            for a, b in zip(first, second, strict=True)
        ),
        dtype=bool,
        count=block.shape[0],
    )
    evaluation = np.broadcast_to(evaluation_rows[:, None, None], block.shape) & block.active
    discovery = ~np.broadcast_to(evaluation_rows[:, None, None], block.shape) & block.active
    return BaselineHoldout(
        discovery_mask=discovery,
        evaluation_mask=evaluation,
        discovery_baselines=discovery_baselines,
        evaluation_baselines=evaluation_baselines,
    )


def temporal_support_mask(
    block: VisibilityBlock,
    *,
    start_s: float,
    duration_s: float,
) -> np.ndarray:
    """Select native integrations whose centres lie in a half-open time interval."""

    if not np.isfinite(start_s):
        raise ValueError("start_s must be finite")
    if not np.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("duration_s must be finite and positive")
    rows = (block.time_s >= start_s) & (block.time_s < start_s + duration_s)
    return np.broadcast_to(rows[:, None, None], block.shape).copy()


def spectral_support_mask(
    block: VisibilityBlock,
    *,
    first_channel: int,
    channel_count: int,
) -> np.ndarray:
    """Select a contiguous native channel interval."""

    if first_channel < 0:
        raise ValueError("first_channel must be non-negative")
    if channel_count < 1:
        raise ValueError("channel_count must be positive")
    stop = first_channel + channel_count
    if stop > block.shape[1]:
        raise ValueError("spectral support exceeds the visibility block")
    selected = np.zeros(block.shape, dtype=bool)
    selected[:, first_channel:stop, :] = True
    return selected


def inject_sky_component(
    block: VisibilityBlock,
    unit_sky_response: np.ndarray,
    support_mask: np.ndarray,
    amplitude_jy: float,
) -> VisibilityBlock:
    """Add a known sky component without changing flags, weights, or the frozen model."""

    response, support = _validate_response_and_support(block, unit_sky_response, support_mask)
    if not np.isfinite(amplitude_jy):
        raise ValueError("amplitude_jy must be finite")
    injected = block.visibility + amplitude_jy * np.where(support, response, 0.0)
    return replace(block, visibility=injected)


def score_recovery_residual(
    residual: np.ndarray,
    weight: np.ndarray,
    mask: np.ndarray,
) -> RecoveryScore:
    """Score a complex residual on an explicit active mask."""

    values = np.asarray(residual, dtype=np.complex128)
    sample_weight = np.asarray(weight, dtype=np.float64)
    selected = np.asarray(mask, dtype=bool)
    if values.shape != sample_weight.shape or values.shape != selected.shape:
        raise ValueError("residual, weight, and mask must have equal shapes")
    usable = (
        selected
        & np.isfinite(values.real)
        & np.isfinite(values.imag)
        & np.isfinite(sample_weight)
        & (sample_weight > 0)
    )
    count = int(np.count_nonzero(usable))
    if count == 0:
        return RecoveryScore(0, 0.0, 0.0, float("nan"))
    weight_sum = float(np.sum(sample_weight[usable]))
    power = float(np.sum(sample_weight[usable] * np.abs(values[usable]) ** 2))
    return RecoveryScore(count, weight_sum, power, power / weight_sum)


def fit_real_sky_component(
    block: VisibilityBlock,
    unit_sky_response: np.ndarray,
    support_mask: np.ndarray,
    discovery_mask: np.ndarray,
    evaluation_mask: np.ndarray,
    *,
    nonnegative: bool = False,
) -> ComponentRecoveryFit:
    """Fit one supported real coefficient and validate it on disjoint samples.

    ``block.model_visibility`` is treated as a frozen static sky. The fitted
    coefficient therefore describes only the temporal or spectral extension.
    """

    if block.model_visibility is None:
        raise ValueError("component recovery requires a frozen model_visibility")
    response, support = _validate_response_and_support(block, unit_sky_response, support_mask)
    discovery = np.asarray(discovery_mask, dtype=bool)
    evaluation = np.asarray(evaluation_mask, dtype=bool)
    if discovery.shape != block.shape or evaluation.shape != block.shape:
        raise ValueError("discovery_mask and evaluation_mask must match the block")
    if np.any(discovery & evaluation):
        raise ValueError("discovery_mask and evaluation_mask must not overlap")
    usable = (
        block.active
        & support
        & np.isfinite(response.real)
        & np.isfinite(response.imag)
    )
    fit = discovery & usable
    test = evaluation & block.active
    test_supported = test & support
    residual = block.visibility - block.model_visibility
    denominator = float(np.sum(block.weight[fit] * np.abs(response[fit]) ** 2))
    if denominator <= 0:
        raise ValueError("the discovery support has no positive response weight")
    numerator = float(
        np.sum(block.weight[fit] * np.real(np.conj(response[fit]) * residual[fit]))
    )
    coefficient = numerator / denominator
    if nonnegative:
        coefficient = max(coefficient, 0.0)
    prediction = coefficient * np.where(support, response, 0.0)
    corrected = residual - prediction
    return ComponentRecoveryFit(
        coefficient=coefficient,
        information_scale=denominator**-0.5,
        discovery_sample_count=int(np.count_nonzero(fit)),
        evaluation_sample_count=int(np.count_nonzero(test_supported)),
        null_evaluation=score_recovery_residual(residual, block.weight, test),
        component_evaluation=score_recovery_residual(corrected, block.weight, test),
        null_supported_evaluation=score_recovery_residual(
            residual, block.weight, test_supported
        ),
        component_supported_evaluation=score_recovery_residual(
            corrected, block.weight, test_supported
        ),
        prediction=prediction,
    )


def _validate_response_and_support(
    block: VisibilityBlock,
    unit_sky_response: np.ndarray,
    support_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    response = np.asarray(unit_sky_response, dtype=np.complex128)
    support = np.asarray(support_mask, dtype=bool)
    if response.shape != block.shape or support.shape != block.shape:
        raise ValueError("unit_sky_response and support_mask must match the block")
    if np.any(support & (~np.isfinite(response.real) | ~np.isfinite(response.imag))):
        raise ValueError("unit_sky_response must be finite on its support")
    return response, support
