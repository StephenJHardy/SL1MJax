"""Non-mutating residual-handling policies with transient protection."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np


class ResidualHandlingMode(StrEnum):
    """How a robust residual score may affect a visibility sample."""

    REPORT_ONLY = "report_only"
    ROBUST_WEIGHTS = "robust_weights"
    STATIC_SKY = "static_sky"
    TRANSIENT_SAFE = "transient_safe"


@dataclass(frozen=True)
class ResidualHandlingResult:
    """A proposed hard mask and continuous weight multiplier."""

    mode: ResidualHandlingMode
    proposed_flag: np.ndarray
    weight_multiplier: np.ndarray
    sky_protected: np.ndarray
    instrumental_flag: np.ndarray


@dataclass(frozen=True)
class ExistingFlagAudit:
    """Four-way comparison between existing flags and residual scores."""

    sample_count: int
    existing_flag_count: int
    existing_unflagged_count: int
    flagged_residual_tail_count: int
    flagged_residual_bulk_count: int
    unflagged_residual_tail_count: int
    unflagged_residual_bulk_count: int
    flagged_residual_bulk_fraction: float
    unflagged_residual_tail_fraction: float


@dataclass(frozen=True)
class SkyCoherenceGroup:
    """Cross-validated fit of one grouped real sky-template coefficient."""

    group_id: int
    coefficient: float
    discovery_sample_count: int
    evaluation_sample_count: int
    evaluation_relative_improvement: float
    protected: bool


@dataclass(frozen=True)
class SkyCoherenceResult:
    """A temporal/spectral sky extension and samples it protects."""

    prediction: np.ndarray
    protected_mask: np.ndarray
    groups: tuple[SkyCoherenceGroup, ...]


def apply_residual_handling(
    robust_score: np.ndarray,
    *,
    mode: ResidualHandlingMode | str,
    threshold: float = 6.0,
    sky_coherent: np.ndarray | None = None,
    instrumental_flag: np.ndarray | None = None,
) -> ResidualHandlingResult:
    """Turn residual scores into an inspectable proposal.

    ``STATIC_SKY`` may hard-flag samples that disagree with a static model.
    ``TRANSIENT_SAFE`` applies the same rule only outside ``sky_coherent``.
    A protected sample represents residual power explained by an additional
    temporal or spectral sky component. Instrumental flags always take
    precedence over sky protection.

    ``ROBUST_WEIGHTS`` applies a Huber-like multiplier but never proposes a new
    hard flag. ``REPORT_ONLY`` changes neither flags nor weights.
    """

    selected_mode = ResidualHandlingMode(mode)
    score = np.asarray(robust_score, dtype=np.float64)
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be finite and positive")
    finite = np.isfinite(score)
    instrumental = (
        np.zeros(score.shape, dtype=bool)
        if instrumental_flag is None
        else np.asarray(instrumental_flag, dtype=bool)
    )
    if instrumental.shape != score.shape:
        raise ValueError("instrumental_flag must match robust_score")
    if sky_coherent is None:
        protected = np.zeros(score.shape, dtype=bool)
    else:
        protected = np.asarray(sky_coherent, dtype=bool)
        if protected.shape != score.shape:
            raise ValueError("sky_coherent must match robust_score")
    if selected_mode is ResidualHandlingMode.TRANSIENT_SAFE and sky_coherent is None:
        raise ValueError("transient_safe mode requires a sky_coherent mask")

    excess = np.maximum(score, 0.0)
    robust_multiplier = np.ones(score.shape, dtype=np.float64)
    high = finite & (excess > threshold)
    robust_multiplier[high] = threshold / excess[high]
    robust_multiplier[~finite] = 0.0
    proposed = instrumental.copy()
    multiplier = np.ones(score.shape, dtype=np.float64)

    if selected_mode is ResidualHandlingMode.ROBUST_WEIGHTS:
        multiplier = robust_multiplier
    elif selected_mode is ResidualHandlingMode.STATIC_SKY:
        proposed |= high
        multiplier[proposed] = 0.0
    elif selected_mode is ResidualHandlingMode.TRANSIENT_SAFE:
        proposed |= high & ~protected
        multiplier = np.where(protected, 1.0, robust_multiplier)
        multiplier[proposed] = 0.0
    elif selected_mode is not ResidualHandlingMode.REPORT_ONLY:  # pragma: no cover
        raise AssertionError(f"unhandled residual mode {selected_mode}")

    multiplier[instrumental] = 0.0
    return ResidualHandlingResult(
        mode=selected_mode,
        proposed_flag=proposed,
        weight_multiplier=multiplier,
        sky_protected=protected & ~instrumental,
        instrumental_flag=instrumental,
    )


def audit_existing_flags(
    robust_score: np.ndarray,
    existing_flag: np.ndarray,
    *,
    threshold: float = 6.0,
) -> ExistingFlagAudit:
    """Count supported, questionable, missed, and quiet flag decisions."""

    score = np.asarray(robust_score, dtype=np.float64)
    flagged = np.asarray(existing_flag, dtype=bool)
    if flagged.shape != score.shape:
        raise ValueError("existing_flag must match robust_score")
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be finite and positive")
    usable = np.isfinite(score)
    flagged &= usable
    unflagged = ~flagged & usable
    tail = score > threshold
    flagged_tail = int(np.count_nonzero(flagged & tail))
    flagged_bulk = int(np.count_nonzero(flagged & ~tail))
    unflagged_tail = int(np.count_nonzero(unflagged & tail))
    unflagged_bulk = int(np.count_nonzero(unflagged & ~tail))
    flagged_count = flagged_tail + flagged_bulk
    unflagged_count = unflagged_tail + unflagged_bulk
    return ExistingFlagAudit(
        sample_count=flagged_count + unflagged_count,
        existing_flag_count=flagged_count,
        existing_unflagged_count=unflagged_count,
        flagged_residual_tail_count=flagged_tail,
        flagged_residual_bulk_count=flagged_bulk,
        unflagged_residual_tail_count=unflagged_tail,
        unflagged_residual_bulk_count=unflagged_bulk,
        flagged_residual_bulk_fraction=(
            float(flagged_bulk / flagged_count) if flagged_count else float("nan")
        ),
        unflagged_residual_tail_fraction=(
            float(unflagged_tail / unflagged_count) if unflagged_count else float("nan")
        ),
    )


def fit_grouped_real_sky_component(
    residual: np.ndarray,
    unit_sky_response: np.ndarray,
    weight: np.ndarray,
    group_id: np.ndarray,
    discovery_mask: np.ndarray,
    evaluation_mask: np.ndarray,
    *,
    minimum_evaluation_relative_improvement: float = 0.25,
    minimum_discovery_samples: int = 16,
    minimum_evaluation_samples: int = 16,
) -> SkyCoherenceResult:
    """Fit and validate a grouped real coefficient for a sky response.

    A time-variable point source is one use: ``unit_sky_response`` is the
    visibility response of one Jy at its position and ``group_id`` identifies
    time bins. Coefficients are fitted only on ``discovery_mask`` baselines.
    A group is protected only when the same coefficient reduces residual power
    on disjoint evaluation baselines. Baseline-local interference therefore
    cannot protect itself merely by being large in the discovery samples.
    """

    values = np.asarray(residual, dtype=np.complex128)
    response = np.asarray(unit_sky_response, dtype=np.complex128)
    sample_weight = np.asarray(weight, dtype=np.float64)
    discovery = np.asarray(discovery_mask, dtype=bool)
    evaluation = np.asarray(evaluation_mask, dtype=bool)
    groups_by_row = np.asarray(group_id)
    if not (
        response.shape
        == sample_weight.shape
        == discovery.shape
        == evaluation.shape
        == values.shape
    ):
        raise ValueError("residual, response, weight, and masks must have equal shapes")
    if groups_by_row.shape != (values.shape[0],):
        raise ValueError("group_id must contain one value per visibility row")
    if np.any(discovery & evaluation):
        raise ValueError("discovery_mask and evaluation_mask must not overlap")
    if not 0 <= minimum_evaluation_relative_improvement <= 1:
        raise ValueError("minimum evaluation improvement must be between zero and one")
    if minimum_discovery_samples < 1 or minimum_evaluation_samples < 1:
        raise ValueError("minimum sample counts must be positive")
    usable = (
        np.isfinite(values.real)
        & np.isfinite(values.imag)
        & np.isfinite(response.real)
        & np.isfinite(response.imag)
        & np.isfinite(sample_weight)
        & (sample_weight > 0)
    )
    prediction = np.zeros(values.shape, dtype=np.complex128)
    protected_mask = np.zeros(values.shape, dtype=bool)
    summaries: list[SkyCoherenceGroup] = []
    for selected_group in np.unique(groups_by_row):
        rows = groups_by_row == selected_group
        group_samples = np.broadcast_to(rows[:, None, None], values.shape)
        fit = discovery & usable & group_samples
        test = evaluation & usable & group_samples
        fit_count = int(np.count_nonzero(fit))
        test_count = int(np.count_nonzero(test))
        denominator = float(np.sum(sample_weight[fit] * np.abs(response[fit]) ** 2))
        coefficient = (
            float(
                np.sum(
                    sample_weight[fit]
                    * np.real(np.conj(response[fit]) * values[fit])
                )
                / denominator
            )
            if fit_count >= minimum_discovery_samples and denominator > 0
            else 0.0
        )
        group_prediction = coefficient * response
        before = float(np.sum(sample_weight[test] * np.abs(values[test]) ** 2))
        after = float(
            np.sum(sample_weight[test] * np.abs(values[test] - group_prediction[test]) ** 2)
        )
        improvement = (
            1.0 - after / before
            if test_count >= minimum_evaluation_samples and before > 0
            else float("nan")
        )
        protected = (
            fit_count >= minimum_discovery_samples
            and test_count >= minimum_evaluation_samples
            and np.isfinite(improvement)
            and improvement >= minimum_evaluation_relative_improvement
        )
        prediction[group_samples] = group_prediction[group_samples]
        if protected:
            protected_mask[group_samples & usable] = True
        summaries.append(
            SkyCoherenceGroup(
                group_id=int(selected_group),
                coefficient=coefficient,
                discovery_sample_count=fit_count,
                evaluation_sample_count=test_count,
                evaluation_relative_improvement=float(improvement),
                protected=protected,
            )
        )
    return SkyCoherenceResult(
        prediction=prediction,
        protected_mask=protected_mask,
        groups=tuple(summaries),
    )
