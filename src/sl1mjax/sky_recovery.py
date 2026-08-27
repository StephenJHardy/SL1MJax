"""Cross-validated recovery tests for temporal and spectral sky components."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

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


VariationKind = Literal[
    "temporal_interval",
    "spectral_interval",
    "spectral_slope",
]


@dataclass(frozen=True)
class BaselineSearchSplit:
    """Discovery, model-selection, and sealed-evaluation baseline cohorts."""

    discovery_mask: np.ndarray
    selection_mask: np.ndarray
    evaluation_mask: np.ndarray
    discovery_baselines: tuple[tuple[int, int], ...]
    selection_baselines: tuple[tuple[int, int], ...]
    evaluation_baselines: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class SkyVariationCandidate:
    """One compact temporal or spectral refinement of a static sky atom."""

    name: str
    kind: VariationKind
    start_index: int | None = None
    bin_count: int | None = None
    coordinate_start: float | None = None
    coordinate_stop: float | None = None


@dataclass(frozen=True)
class SkyVariationCandidateScore:
    """Discovery fit and optional selection score for one candidate."""

    candidate: SkyVariationCandidate
    discovery_static_coefficient: float
    discovery_variation_coefficient: float
    discovery_incremental_power: float
    discovery_incremental_weighted_mse: float
    discovery_relative_improvement: float
    selection_incremental_power: float | None = None
    selection_incremental_weighted_mse: float | None = None
    selection_relative_improvement: float | None = None


@dataclass(frozen=True)
class BlindSkyVariationSearchResult:
    """Nested blind search with a sealed baseline evaluation."""

    candidate_count: int
    shortlist_size: int
    discovery_ranking: tuple[SkyVariationCandidateScore, ...]
    shortlist: tuple[SkyVariationCandidateScore, ...]
    selected_candidate: SkyVariationCandidate | None
    refit_static_coefficient: float
    refit_variation_coefficient: float
    selection_incremental_weighted_mse: float
    evaluation_static_weighted_mse: float
    evaluation_candidate_weighted_mse: float
    evaluation_incremental_weighted_mse: float
    evaluation_relative_improvement: float
    accepted: bool


@dataclass(frozen=True)
class _VariationSufficientStatistics:
    """Matched-filter sufficient statistics for one baseline cohort."""

    time_s: np.ndarray
    frequency_hz: np.ndarray
    matched_total: float
    response_power_total: float
    residual_power_total: float
    weight_total: float
    sample_count: int
    matched_by_time: np.ndarray
    response_power_by_time: np.ndarray
    matched_by_channel: np.ndarray
    response_power_by_channel: np.ndarray


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


def split_search_baselines(
    block: VisibilityBlock,
    *,
    selection_fraction: float = 0.2,
    evaluation_fraction: float = 0.2,
    seed: int = 0,
) -> BaselineSearchSplit:
    """Partition complete baselines for discovery, selection, and evaluation."""

    if not 0 < selection_fraction < 1:
        raise ValueError("selection_fraction must be between zero and one")
    if not 0 < evaluation_fraction < 1:
        raise ValueError("evaluation_fraction must be between zero and one")
    if selection_fraction + evaluation_fraction >= 1:
        raise ValueError("selection and evaluation fractions must sum to less than one")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    first = np.minimum(block.antenna1, block.antenna2)
    second = np.maximum(block.antenna1, block.antenna2)
    baselines = tuple(
        sorted({(int(a), int(b)) for a, b in zip(first, second, strict=True)})
    )
    if len(baselines) < 3:
        raise ValueError("at least three baselines are required for a blind search")
    selection_count = max(1, int(round(selection_fraction * len(baselines))))
    evaluation_count = max(1, int(round(evaluation_fraction * len(baselines))))
    if selection_count + evaluation_count >= len(baselines):
        raise ValueError("baseline fractions leave no discovery baselines")
    shuffled = np.random.default_rng(seed).permutation(len(baselines))
    selection_indices = set(int(value) for value in shuffled[:selection_count])
    evaluation_indices = set(
        int(value)
        for value in shuffled[selection_count : selection_count + evaluation_count]
    )
    selection_baselines = tuple(
        baseline for index, baseline in enumerate(baselines) if index in selection_indices
    )
    evaluation_baselines = tuple(
        baseline for index, baseline in enumerate(baselines) if index in evaluation_indices
    )
    discovery_baselines = tuple(
        baseline
        for index, baseline in enumerate(baselines)
        if index not in selection_indices and index not in evaluation_indices
    )

    def sample_mask(selected: tuple[tuple[int, int], ...]) -> np.ndarray:
        lookup = set(selected)
        rows = np.fromiter(
            (
                (int(a), int(b)) in lookup
                for a, b in zip(first, second, strict=True)
            ),
            dtype=bool,
            count=block.shape[0],
        )
        return np.asarray(
            np.broadcast_to(rows[:, None, None], block.shape) & block.active,
            dtype=bool,
        )

    return BaselineSearchSplit(
        discovery_mask=sample_mask(discovery_baselines),
        selection_mask=sample_mask(selection_baselines),
        evaluation_mask=sample_mask(evaluation_baselines),
        discovery_baselines=discovery_baselines,
        selection_baselines=selection_baselines,
        evaluation_baselines=evaluation_baselines,
    )


def temporal_interval_candidates(
    block: VisibilityBlock,
    widths: tuple[int, ...],
) -> tuple[SkyVariationCandidate, ...]:
    """Generate sliding native-time intervals without crossing observation gaps."""

    selected_widths = _validated_widths(widths, name="temporal widths")
    time_s = np.unique(block.time_s)
    if block.interval_s is None:
        raise ValueError("temporal candidates require integration intervals")
    cadence = float(np.median(block.interval_s))
    if not np.isfinite(cadence) or cadence <= 0:
        raise ValueError("native integration cadence must be finite and positive")
    breaks = np.flatnonzero(np.diff(time_s) > 1.5 * cadence) + 1
    run_boundaries = np.concatenate(([0], breaks, [time_s.size]))
    candidates: list[SkyVariationCandidate] = []
    for run_start, run_stop in zip(run_boundaries[:-1], run_boundaries[1:], strict=True):
        run_size = int(run_stop - run_start)
        for width in selected_widths:
            if width > run_size:
                continue
            for local_start in range(run_size - width + 1):
                start = int(run_start + local_start)
                stop = start + width
                candidates.append(
                    SkyVariationCandidate(
                        name=f"time_{start:04d}_w{width:03d}",
                        kind="temporal_interval",
                        start_index=start,
                        bin_count=width,
                        coordinate_start=float(time_s[start]),
                        coordinate_stop=float(time_s[stop - 1] + cadence),
                    )
                )
    return tuple(candidates)


def spectral_interval_candidates(
    block: VisibilityBlock,
    widths: tuple[int, ...],
) -> tuple[SkyVariationCandidate, ...]:
    """Generate sliding compact spectral intervals below the full band width."""

    selected_widths = _validated_widths(widths, name="spectral widths")
    channel_count = block.frequency_hz.size
    if channel_count > 1:
        channel_width = float(np.median(np.diff(block.frequency_hz)))
    else:
        channel_width = 0.0
    candidates: list[SkyVariationCandidate] = []
    for width in selected_widths:
        if width >= channel_count:
            continue
        for start in range(channel_count - width + 1):
            stop = start + width
            candidates.append(
                SkyVariationCandidate(
                    name=f"frequency_{start:04d}_w{width:03d}",
                    kind="spectral_interval",
                    start_index=start,
                    bin_count=width,
                    coordinate_start=float(block.frequency_hz[start]),
                    coordinate_stop=float(block.frequency_hz[stop - 1] + channel_width),
                )
            )
    return tuple(candidates)


def native_variation_candidates(
    block: VisibilityBlock,
    *,
    temporal_widths: tuple[int, ...] = (1, 3, 6),
    spectral_widths: tuple[int, ...] = (1, 2, 4, 8),
    include_spectral_slope: bool = True,
) -> tuple[SkyVariationCandidate, ...]:
    """Build the first factored time/frequency candidate bank."""

    candidates = list(temporal_interval_candidates(block, temporal_widths))
    candidates.extend(spectral_interval_candidates(block, spectral_widths))
    if include_spectral_slope and block.frequency_hz.size > 1:
        candidates.append(
            SkyVariationCandidate(
                name="spectral_log_slope",
                kind="spectral_slope",
                coordinate_start=float(np.min(block.frequency_hz)),
                coordinate_stop=float(np.max(block.frequency_hz)),
            )
        )
    names = [candidate.name for candidate in candidates]
    if len(names) != len(set(names)):
        raise RuntimeError("variation candidate names must be unique")
    return tuple(candidates)


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


def blind_search_sky_variation(
    block: VisibilityBlock,
    unit_sky_response: np.ndarray,
    candidates: tuple[SkyVariationCandidate, ...],
    split: BaselineSearchSplit,
    *,
    shortlist_size: int = 16,
    minimum_selection_mse_gain: float = 0.0,
    minimum_evaluation_mse_gain: float = 0.0,
) -> BlindSkyVariationSearchResult:
    """Discover, select, and evaluate one nested temporal or spectral refinement.

    Candidate ranking uses only discovery baselines. The best discovery
    candidates are compared on selection baselines. Only the selected model is
    refitted on discovery plus selection and exposed to the sealed evaluation
    baselines.
    """

    if block.model_visibility is None:
        raise ValueError("blind variation search requires a frozen model_visibility")
    response = np.asarray(unit_sky_response, dtype=np.complex128)
    if response.shape != block.shape:
        raise ValueError("unit_sky_response must match the visibility block")
    if not candidates:
        raise ValueError("candidates must contain at least one variation")
    if shortlist_size < 1:
        raise ValueError("shortlist_size must be positive")
    if not np.isfinite(minimum_selection_mse_gain) or minimum_selection_mse_gain < 0:
        raise ValueError("minimum_selection_mse_gain must be finite and non-negative")
    if not np.isfinite(minimum_evaluation_mse_gain) or minimum_evaluation_mse_gain < 0:
        raise ValueError("minimum_evaluation_mse_gain must be finite and non-negative")
    _validate_search_split(block, split)
    names = [candidate.name for candidate in candidates]
    if len(names) != len(set(names)):
        raise ValueError("candidate names must be unique")

    discovery = _variation_statistics(block, response, split.discovery_mask)
    selection = _variation_statistics(block, response, split.selection_mask)
    evaluation = _variation_statistics(block, response, split.evaluation_mask)
    discovery_static = _fit_static_coefficient(discovery)
    static_discovery_power = _residual_power(discovery, (discovery_static, 0.0), None)
    ranking: list[SkyVariationCandidateScore] = []
    coefficients: dict[str, tuple[float, float]] = {}
    for candidate in candidates:
        fitted = _fit_nested_coefficients(discovery, candidate)
        coefficients[candidate.name] = fitted
        candidate_power = _residual_power(discovery, fitted, candidate)
        gain = static_discovery_power - candidate_power
        ranking.append(
            SkyVariationCandidateScore(
                candidate=candidate,
                discovery_static_coefficient=fitted[0],
                discovery_variation_coefficient=fitted[1],
                discovery_incremental_power=float(gain),
                discovery_incremental_weighted_mse=float(gain / discovery.weight_total),
                discovery_relative_improvement=_relative_gain(gain, static_discovery_power),
            )
        )
    ranking.sort(
        key=lambda item: (
            -item.discovery_incremental_weighted_mse,
            item.candidate.name,
        )
    )
    shortlist_count = min(shortlist_size, len(ranking))
    selection_static_power = _residual_power(
        selection, (discovery_static, 0.0), None
    )
    shortlisted: list[SkyVariationCandidateScore] = []
    for score in ranking[:shortlist_count]:
        candidate = score.candidate
        candidate_power = _residual_power(
            selection,
            coefficients[candidate.name],
            candidate,
        )
        gain = selection_static_power - candidate_power
        shortlisted.append(
            replace(
                score,
                selection_incremental_power=float(gain),
                selection_incremental_weighted_mse=float(gain / selection.weight_total),
                selection_relative_improvement=_relative_gain(gain, selection_static_power),
            )
        )
    shortlisted.sort(
        key=lambda item: (
            -_required_selection_gain(item),
            item.candidate.name,
        )
    )
    best = shortlisted[0]
    best_selection_gain = _required_selection_gain(best)
    selected_candidate = (
        best.candidate
        if best_selection_gain > minimum_selection_mse_gain
        else None
    )
    combined = _add_variation_statistics(discovery, selection)
    refit_static = _fit_static_coefficient(combined)
    evaluation_static_power = _residual_power(
        evaluation, (refit_static, 0.0), None
    )
    if selected_candidate is None:
        refit = (refit_static, 0.0)
        evaluation_candidate_power = evaluation_static_power
    else:
        refit = _fit_nested_coefficients(combined, selected_candidate)
        evaluation_candidate_power = _residual_power(
            evaluation,
            refit,
            selected_candidate,
        )
    evaluation_gain = evaluation_static_power - evaluation_candidate_power
    evaluation_mse_gain = evaluation_gain / evaluation.weight_total
    accepted = (
        selected_candidate is not None
        and evaluation_mse_gain > minimum_evaluation_mse_gain
    )
    return BlindSkyVariationSearchResult(
        candidate_count=len(candidates),
        shortlist_size=shortlist_count,
        discovery_ranking=tuple(ranking),
        shortlist=tuple(shortlisted),
        selected_candidate=selected_candidate,
        refit_static_coefficient=float(refit[0]),
        refit_variation_coefficient=float(refit[1]),
        selection_incremental_weighted_mse=(
            best_selection_gain if selected_candidate is not None else 0.0
        ),
        evaluation_static_weighted_mse=float(
            evaluation_static_power / evaluation.weight_total
        ),
        evaluation_candidate_weighted_mse=float(
            evaluation_candidate_power / evaluation.weight_total
        ),
        evaluation_incremental_weighted_mse=float(evaluation_mse_gain),
        evaluation_relative_improvement=_relative_gain(
            evaluation_gain, evaluation_static_power
        ),
        accepted=bool(accepted),
    )


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


def _validated_widths(widths: tuple[int, ...], *, name: str) -> tuple[int, ...]:
    if not widths:
        raise ValueError(f"{name} must contain at least one width")
    selected = tuple(int(width) for width in widths)
    if any(width < 1 for width in selected):
        raise ValueError(f"{name} must be positive")
    if len(selected) != len(set(selected)):
        raise ValueError(f"{name} must not contain duplicates")
    return selected


def _validate_search_split(block: VisibilityBlock, split: BaselineSearchSplit) -> None:
    masks = (split.discovery_mask, split.selection_mask, split.evaluation_mask)
    for mask in masks:
        selected = np.asarray(mask, dtype=bool)
        if selected.shape != block.shape:
            raise ValueError("every search mask must match the visibility block")
        if np.any(selected & ~block.active):
            raise ValueError("search masks must contain only active samples")
    if (
        np.any(masks[0] & masks[1])
        or np.any(masks[0] & masks[2])
        or np.any(masks[1] & masks[2])
    ):
        raise ValueError("discovery, selection, and evaluation masks must be disjoint")
    if any(not np.any(mask) for mask in masks):
        raise ValueError("every search cohort must contain active samples")


def _variation_statistics(
    block: VisibilityBlock,
    unit_sky_response: np.ndarray,
    mask: np.ndarray,
) -> _VariationSufficientStatistics:
    assert block.model_visibility is not None
    response = np.asarray(unit_sky_response, dtype=np.complex128)
    residual = block.visibility - block.model_visibility
    selected = (
        np.asarray(mask, dtype=bool)
        & block.active
        & np.isfinite(response.real)
        & np.isfinite(response.imag)
        & np.isfinite(residual.real)
        & np.isfinite(residual.imag)
    )
    sample_weight = np.where(selected, block.weight, 0.0)
    safe_response = np.where(selected, response, 0.0)
    safe_residual = np.where(selected, residual, 0.0)
    matched = sample_weight * np.real(np.conj(safe_response) * safe_residual)
    response_power = sample_weight * np.abs(safe_response) ** 2
    residual_power = sample_weight * np.abs(safe_residual) ** 2
    row_matched = np.sum(matched, axis=(1, 2))
    row_response_power = np.sum(response_power, axis=(1, 2))
    time_s, time_inverse = np.unique(block.time_s, return_inverse=True)
    matched_by_time = np.bincount(
        time_inverse, weights=row_matched, minlength=time_s.size
    )
    response_power_by_time = np.bincount(
        time_inverse, weights=row_response_power, minlength=time_s.size
    )
    return _VariationSufficientStatistics(
        time_s=time_s,
        frequency_hz=np.asarray(block.frequency_hz, dtype=np.float64),
        matched_total=float(np.sum(matched)),
        response_power_total=float(np.sum(response_power)),
        residual_power_total=float(np.sum(residual_power)),
        weight_total=float(np.sum(sample_weight)),
        sample_count=int(np.count_nonzero(selected)),
        matched_by_time=np.asarray(matched_by_time, dtype=np.float64),
        response_power_by_time=np.asarray(response_power_by_time, dtype=np.float64),
        matched_by_channel=np.asarray(np.sum(matched, axis=(0, 2)), dtype=np.float64),
        response_power_by_channel=np.asarray(
            np.sum(response_power, axis=(0, 2)), dtype=np.float64
        ),
    )


def _candidate_moments(
    statistics: _VariationSufficientStatistics,
    candidate: SkyVariationCandidate,
) -> tuple[float, float, float]:
    """Return matched candidate, static cross-power, and candidate power."""

    if candidate.kind == "temporal_interval":
        assert candidate.start_index is not None and candidate.bin_count is not None
        selected = slice(candidate.start_index, candidate.start_index + candidate.bin_count)
        matched = float(np.sum(statistics.matched_by_time[selected]))
        power = float(np.sum(statistics.response_power_by_time[selected]))
        return matched, power, power
    if candidate.kind == "spectral_interval":
        assert candidate.start_index is not None and candidate.bin_count is not None
        selected = slice(candidate.start_index, candidate.start_index + candidate.bin_count)
        matched = float(np.sum(statistics.matched_by_channel[selected]))
        power = float(np.sum(statistics.response_power_by_channel[selected]))
        return matched, power, power
    if candidate.kind == "spectral_slope":
        if np.any(statistics.frequency_hz <= 0):
            raise ValueError("spectral slope requires positive frequencies")
        reference = float(
            np.exp(np.mean(np.log(statistics.frequency_hz)))
        )
        multiplier = np.log(statistics.frequency_hz / reference)
        matched = float(np.sum(statistics.matched_by_channel * multiplier))
        cross_power = float(
            np.sum(statistics.response_power_by_channel * multiplier)
        )
        power = float(
            np.sum(statistics.response_power_by_channel * multiplier * multiplier)
        )
        return matched, cross_power, power
    raise ValueError(f"unsupported variation kind {candidate.kind!r}")


def _fit_static_coefficient(statistics: _VariationSufficientStatistics) -> float:
    if statistics.response_power_total <= 0:
        raise ValueError("search cohort has no positive response weight")
    return statistics.matched_total / statistics.response_power_total


def _fit_nested_coefficients(
    statistics: _VariationSufficientStatistics,
    candidate: SkyVariationCandidate,
) -> tuple[float, float]:
    matched_variation, cross_power, variation_power = _candidate_moments(
        statistics, candidate
    )
    normal = np.asarray(
        [
            [statistics.response_power_total, cross_power],
            [cross_power, variation_power],
        ],
        dtype=np.float64,
    )
    matched = np.asarray(
        [statistics.matched_total, matched_variation], dtype=np.float64
    )
    if variation_power <= 0:
        return _fit_static_coefficient(statistics), 0.0
    coefficients, _, rank, _ = np.linalg.lstsq(normal, matched, rcond=None)
    if rank < 2:
        return _fit_static_coefficient(statistics), 0.0
    return float(coefficients[0]), float(coefficients[1])


def _residual_power(
    statistics: _VariationSufficientStatistics,
    coefficients: tuple[float, float],
    candidate: SkyVariationCandidate | None,
) -> float:
    static, variation = coefficients
    if candidate is None:
        matched_variation = 0.0
        cross_power = 0.0
        variation_power = 0.0
    else:
        matched_variation, cross_power, variation_power = _candidate_moments(
            statistics, candidate
        )
    return float(
        statistics.residual_power_total
        - 2.0 * (static * statistics.matched_total + variation * matched_variation)
        + static * static * statistics.response_power_total
        + 2.0 * static * variation * cross_power
        + variation * variation * variation_power
    )


def _add_variation_statistics(
    first: _VariationSufficientStatistics,
    second: _VariationSufficientStatistics,
) -> _VariationSufficientStatistics:
    if not np.array_equal(first.time_s, second.time_s) or not np.array_equal(
        first.frequency_hz, second.frequency_hz
    ):
        raise ValueError("variation statistics have different coordinates")
    return _VariationSufficientStatistics(
        time_s=first.time_s,
        frequency_hz=first.frequency_hz,
        matched_total=first.matched_total + second.matched_total,
        response_power_total=first.response_power_total + second.response_power_total,
        residual_power_total=first.residual_power_total + second.residual_power_total,
        weight_total=first.weight_total + second.weight_total,
        sample_count=first.sample_count + second.sample_count,
        matched_by_time=first.matched_by_time + second.matched_by_time,
        response_power_by_time=(
            first.response_power_by_time + second.response_power_by_time
        ),
        matched_by_channel=first.matched_by_channel + second.matched_by_channel,
        response_power_by_channel=(
            first.response_power_by_channel + second.response_power_by_channel
        ),
    )


def _relative_gain(gain: float, reference_power: float) -> float:
    return float(gain / reference_power) if reference_power > 0 else float("nan")


def _required_selection_gain(score: SkyVariationCandidateScore) -> float:
    value = score.selection_incremental_weighted_mse
    if value is None:  # pragma: no cover - guarded by shortlist construction
        raise RuntimeError("shortlisted candidate has no selection score")
    return float(value)
