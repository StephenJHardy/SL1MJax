"""Leakage-safe, model-residual audits for visibility data quality."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from sl1mjax.data.canonical import VisibilityBlock


@dataclass(frozen=True)
class RobustResidualScale:
    """Amplitude-residual location and scale fitted on discovery samples."""

    pointing: int
    channel: int
    correlation: int
    centre: float
    scale: float
    sample_count: int


@dataclass(frozen=True)
class ResidualPartitionSummary:
    """Residual-tail metrics for one sample partition."""

    sample_count: int
    weight_sum: float
    weighted_complex_mse: float
    normalized_residual_power: float
    median_robust_score: float
    score_p99: float
    maximum_robust_score: float
    outlier_count: int
    outlier_fraction: float
    outlier_residual_power_fraction: float
    top_one_percent_residual_power_fraction: float
    top_point_one_percent_residual_power_fraction: float


@dataclass(frozen=True)
class ResidualGroupSummary:
    """Discovery and sealed-evaluation metrics for a physical data group."""

    kind: str
    key: tuple[int, ...]
    label: str
    discovery: ResidualPartitionSummary
    evaluation: ResidualPartitionSummary
    candidate: bool
    validated: bool


@dataclass(frozen=True)
class VisibilityResidualAudit:
    """Complete residual audit with train-fitted scales and grouped rankings."""

    score_threshold: float
    minimum_group_samples: int
    minimum_group_outlier_fraction: float
    scales: tuple[RobustResidualScale, ...]
    discovery: ResidualPartitionSummary
    evaluation: ResidualPartitionSummary
    groups: tuple[ResidualGroupSummary, ...]


GroupKind = Literal["pointing", "baseline", "antenna", "channel", "correlation", "scan"]


def _validate_inputs(
    blocks: tuple[VisibilityBlock, ...],
    predictions: tuple[np.ndarray, ...],
    discovery_masks: tuple[np.ndarray, ...],
    evaluation_masks: tuple[np.ndarray, ...],
) -> None:
    if not blocks:
        raise ValueError("blocks must contain at least one visibility block")
    if not (len(predictions) == len(discovery_masks) == len(evaluation_masks) == len(blocks)):
        raise ValueError("blocks, predictions, and masks must have equal lengths")
    for index, (block, prediction, discovery, evaluation) in enumerate(
        zip(blocks, predictions, discovery_masks, evaluation_masks, strict=True)
    ):
        if prediction.shape != block.shape:
            raise ValueError(f"prediction {index} must match its visibility block")
        if discovery.shape != block.shape or evaluation.shape != block.shape:
            raise ValueError(f"masks {index} must match their visibility block")
        if np.any(discovery & evaluation):
            raise ValueError(f"discovery and evaluation masks {index} overlap")
        if np.any(discovery & ~block.active) or np.any(evaluation & ~block.active):
            raise ValueError(f"masks {index} contain inactive samples")


def robust_residual_scores(
    blocks: tuple[VisibilityBlock, ...],
    predictions: tuple[np.ndarray, ...],
    discovery_masks: tuple[np.ndarray, ...],
    *,
    minimum_relative_scale: float = 1e-6,
) -> tuple[tuple[np.ndarray, ...], tuple[RobustResidualScale, ...]]:
    """Fit robust scales on discovery data and score every active sample.

    The quantity being scored is ``sqrt(weight) * abs(observed - predicted)``.
    A separate median and Gaussian-consistent MAD scale is fitted for every
    pointing, channel, and correlation. Evaluation samples never influence the
    fitted location or scale.
    """

    if not blocks or len(blocks) != len(predictions) or len(blocks) != len(discovery_masks):
        raise ValueError("blocks, predictions, and discovery_masks must have equal lengths")
    if not np.isfinite(minimum_relative_scale) or minimum_relative_scale <= 0:
        raise ValueError("minimum_relative_scale must be finite and positive")
    scores: list[np.ndarray] = []
    scales: list[RobustResidualScale] = []
    for pointing, (block, prediction, discovery) in enumerate(
        zip(blocks, predictions, discovery_masks, strict=True)
    ):
        if prediction.shape != block.shape or discovery.shape != block.shape:
            raise ValueError(f"prediction and mask {pointing} must match their block")
        if np.any(discovery & ~block.active):
            raise ValueError(f"discovery mask {pointing} contains inactive samples")
        weighted_amplitude = np.abs(block.visibility - prediction) * np.sqrt(
            np.maximum(block.weight, 0.0)
        )
        block_scores = np.full(block.shape, np.nan, dtype=np.float64)
        for channel in range(block.shape[1]):
            for correlation in range(block.shape[2]):
                selected = discovery[:, channel, correlation]
                values = weighted_amplitude[selected, channel, correlation]
                if values.size == 0:
                    raise ValueError(
                        "every pointing/channel/correlation stratum needs discovery samples"
                    )
                centre = float(np.median(values))
                raw_scale = 1.4826 * float(np.median(np.abs(values - centre)))
                scale = max(
                    raw_scale,
                    minimum_relative_scale * max(centre, 1.0),
                    np.finfo(np.float64).eps,
                )
                active = block.active[:, channel, correlation]
                block_scores[active, channel, correlation] = (
                    weighted_amplitude[active, channel, correlation] - centre
                ) / scale
                scales.append(
                    RobustResidualScale(
                        pointing=pointing,
                        channel=channel,
                        correlation=correlation,
                        centre=centre,
                        scale=scale,
                        sample_count=int(values.size),
                    )
                )
        scores.append(block_scores)
    return tuple(scores), tuple(scales)


def apply_robust_residual_scales(
    blocks: tuple[VisibilityBlock, ...],
    predictions: tuple[np.ndarray, ...],
    scales: tuple[RobustResidualScale, ...],
) -> tuple[np.ndarray, ...]:
    """Score residuals with a previously fitted location and scale."""

    if not blocks or len(blocks) != len(predictions):
        raise ValueError("blocks and predictions must have equal non-zero lengths")
    scale_by_stratum = {
        (scale.pointing, scale.channel, scale.correlation): scale for scale in scales
    }
    if len(scale_by_stratum) != len(scales):
        raise ValueError("robust scales must contain unique strata")
    expected_count = sum(block.shape[1] * block.shape[2] for block in blocks)
    if len(scales) != expected_count:
        raise ValueError("robust scales must cover every pointing/channel/correlation")

    result: list[np.ndarray] = []
    for pointing, (block, prediction) in enumerate(zip(blocks, predictions, strict=True)):
        if prediction.shape != block.shape:
            raise ValueError(f"prediction {pointing} must match its block")
        weighted_amplitude = np.abs(block.visibility - prediction) * np.sqrt(
            np.maximum(block.weight, 0.0)
        )
        scores = np.full(block.shape, np.nan, dtype=np.float64)
        for channel in range(block.shape[1]):
            for correlation in range(block.shape[2]):
                scale = scale_by_stratum.get((pointing, channel, correlation))
                if scale is None or not np.isfinite(scale.scale) or scale.scale <= 0:
                    raise ValueError("robust scales contain a missing or invalid stratum")
                active = block.active[:, channel, correlation]
                scores[active, channel, correlation] = (
                    weighted_amplitude[active, channel, correlation] - scale.centre
                ) / scale.scale
        result.append(scores)
    return tuple(result)


def _empty_partition() -> ResidualPartitionSummary:
    return ResidualPartitionSummary(
        sample_count=0,
        weight_sum=0.0,
        weighted_complex_mse=float("nan"),
        normalized_residual_power=float("nan"),
        median_robust_score=float("nan"),
        score_p99=float("nan"),
        maximum_robust_score=float("nan"),
        outlier_count=0,
        outlier_fraction=float("nan"),
        outlier_residual_power_fraction=float("nan"),
        top_one_percent_residual_power_fraction=float("nan"),
        top_point_one_percent_residual_power_fraction=float("nan"),
    )


def _partition_summary(
    blocks: tuple[VisibilityBlock, ...],
    predictions: tuple[np.ndarray, ...],
    scores: tuple[np.ndarray, ...],
    masks: tuple[np.ndarray, ...],
    *,
    score_threshold: float,
) -> ResidualPartitionSummary:
    residual_power_parts: list[np.ndarray] = []
    score_parts: list[np.ndarray] = []
    residual_power_sum = 0.0
    signal_power_sum = 0.0
    weight_sum = 0.0
    sample_count = 0
    for block, prediction, score, mask in zip(blocks, predictions, scores, masks, strict=True):
        selected = mask & block.active & np.isfinite(score)
        if not np.any(selected):
            continue
        residual_power = (
            block.weight[selected] * np.abs(block.visibility[selected] - prediction[selected]) ** 2
        )
        residual_power_parts.append(residual_power)
        score_parts.append(score[selected])
        residual_power_sum += float(np.sum(residual_power))
        signal_power_sum += float(
            np.sum(block.weight[selected] * np.abs(block.visibility[selected]) ** 2)
        )
        weight_sum += float(np.sum(block.weight[selected]))
        sample_count += int(np.count_nonzero(selected))
    if sample_count == 0:
        return _empty_partition()
    power = np.concatenate(residual_power_parts)
    score = np.concatenate(score_parts)
    outlier = score > score_threshold
    ordered_power = np.sort(power)[::-1]
    tiny = np.finfo(np.float64).tiny

    def top_fraction(fraction: float) -> float:
        count = max(1, int(np.ceil(fraction * sample_count)))
        return float(np.sum(ordered_power[:count]) / max(residual_power_sum, tiny))

    return ResidualPartitionSummary(
        sample_count=sample_count,
        weight_sum=weight_sum,
        weighted_complex_mse=residual_power_sum / max(weight_sum, tiny),
        normalized_residual_power=residual_power_sum / max(signal_power_sum, tiny),
        median_robust_score=float(np.median(score)),
        score_p99=float(np.quantile(score, 0.99)),
        maximum_robust_score=float(np.max(score)),
        outlier_count=int(np.count_nonzero(outlier)),
        outlier_fraction=float(np.mean(outlier)),
        outlier_residual_power_fraction=float(
            np.sum(power[outlier]) / max(residual_power_sum, tiny)
        ),
        top_one_percent_residual_power_fraction=top_fraction(0.01),
        top_point_one_percent_residual_power_fraction=top_fraction(0.001),
    )


def _group_keys(
    blocks: tuple[VisibilityBlock, ...], kind: GroupKind
) -> tuple[tuple[tuple[int, ...], str], ...]:
    if kind == "pointing":
        return tuple(((index,), f"C{index + 1}") for index in range(len(blocks)))
    if kind == "baseline":
        pairs = {
            (int(min(first, second)), int(max(first, second)))
            for block in blocks
            for first, second in zip(block.antenna1, block.antenna2, strict=True)
        }
        return tuple((pair, f"{pair[0]}-{pair[1]}") for pair in sorted(pairs))
    if kind == "antenna":
        antennas = sorted(
            {
                int(antenna)
                for block in blocks
                for antenna in np.concatenate((block.antenna1, block.antenna2))
            }
        )
        return tuple(((antenna,), str(antenna)) for antenna in antennas)
    if kind == "channel":
        channels = sorted({channel for block in blocks for channel in range(block.shape[1])})
        return tuple(((channel,), str(channel)) for channel in channels)
    if kind == "correlation":
        correlations = sorted(
            {correlation for block in blocks for correlation in range(block.shape[2])}
        )
        return tuple(((correlation,), str(correlation)) for correlation in correlations)
    if kind == "scan":
        scans = {
            (pointing, int(scan))
            for pointing, block in enumerate(blocks)
            if block.scan_id is not None
            for scan in np.unique(block.scan_id)
        }
        return tuple((key, f"C{key[0] + 1}:scan{key[1]}") for key in sorted(scans))
    raise ValueError(f"unsupported group kind {kind!r}")


def _group_masks(
    blocks: tuple[VisibilityBlock, ...], kind: GroupKind, key: tuple[int, ...]
) -> tuple[np.ndarray, ...]:
    masks: list[np.ndarray] = []
    for pointing, block in enumerate(blocks):
        if kind == "pointing":
            selected = np.full(block.shape, pointing == key[0], dtype=bool)
        elif kind == "baseline":
            first = np.minimum(block.antenna1, block.antenna2)
            second = np.maximum(block.antenna1, block.antenna2)
            rows = (first == key[0]) & (second == key[1])
            selected = np.broadcast_to(rows[:, None, None], block.shape)
        elif kind == "antenna":
            rows = (block.antenna1 == key[0]) | (block.antenna2 == key[0])
            selected = np.broadcast_to(rows[:, None, None], block.shape)
        elif kind == "channel":
            selected = np.zeros(block.shape, dtype=bool)
            if key[0] < block.shape[1]:
                selected[:, key[0], :] = True
        elif kind == "correlation":
            selected = np.zeros(block.shape, dtype=bool)
            if key[0] < block.shape[2]:
                selected[:, :, key[0]] = True
        elif kind == "scan":
            rows = (
                np.zeros(block.shape[0], dtype=bool)
                if block.scan_id is None
                else (pointing == key[0]) & (block.scan_id == key[1])
            )
            selected = np.broadcast_to(rows[:, None, None], block.shape)
        else:  # pragma: no cover - guarded by the public literal API
            raise ValueError(f"unsupported group kind {kind!r}")
        masks.append(selected)
    return tuple(masks)


def audit_visibility_residuals(
    blocks: tuple[VisibilityBlock, ...],
    predictions: tuple[np.ndarray, ...],
    discovery_masks: tuple[np.ndarray, ...],
    evaluation_masks: tuple[np.ndarray, ...],
    *,
    score_threshold: float = 6.0,
    group_kinds: tuple[GroupKind, ...] = (
        "pointing",
        "baseline",
        "antenna",
        "channel",
        "correlation",
        "scan",
    ),
    minimum_group_samples: int = 128,
    minimum_group_outlier_fraction: float = 0.05,
    fixed_scales: tuple[RobustResidualScale, ...] | None = None,
) -> VisibilityResidualAudit:
    """Audit model residuals without using evaluation samples for selection."""

    _validate_inputs(blocks, predictions, discovery_masks, evaluation_masks)
    if not np.isfinite(score_threshold) or score_threshold <= 0:
        raise ValueError("score_threshold must be finite and positive")
    if minimum_group_samples < 1:
        raise ValueError("minimum_group_samples must be positive")
    if not 0 <= minimum_group_outlier_fraction <= 1:
        raise ValueError("minimum_group_outlier_fraction must be between zero and one")
    if len(group_kinds) != len(set(group_kinds)):
        raise ValueError("group_kinds must not contain duplicates")
    if fixed_scales is None:
        scores, scales = robust_residual_scores(blocks, predictions, discovery_masks)
    else:
        scales = fixed_scales
        scores = apply_robust_residual_scales(blocks, predictions, scales)
    discovery = _partition_summary(
        blocks,
        predictions,
        scores,
        discovery_masks,
        score_threshold=score_threshold,
    )
    evaluation = _partition_summary(
        blocks,
        predictions,
        scores,
        evaluation_masks,
        score_threshold=score_threshold,
    )
    groups: list[ResidualGroupSummary] = []
    for kind in group_kinds:
        for key, label in _group_keys(blocks, kind):
            group_masks = _group_masks(blocks, kind, key)
            group_discovery = tuple(
                mask & group for mask, group in zip(discovery_masks, group_masks, strict=True)
            )
            group_evaluation = tuple(
                mask & group for mask, group in zip(evaluation_masks, group_masks, strict=True)
            )
            discovery_summary = _partition_summary(
                blocks,
                predictions,
                scores,
                group_discovery,
                score_threshold=score_threshold,
            )
            evaluation_summary = _partition_summary(
                blocks,
                predictions,
                scores,
                group_evaluation,
                score_threshold=score_threshold,
            )
            candidate = (
                discovery_summary.sample_count >= minimum_group_samples
                and discovery_summary.outlier_fraction >= minimum_group_outlier_fraction
            )
            validated = (
                candidate
                and evaluation_summary.sample_count >= minimum_group_samples
                and evaluation_summary.outlier_fraction >= minimum_group_outlier_fraction
            )
            groups.append(
                ResidualGroupSummary(
                    kind=kind,
                    key=key,
                    label=label,
                    discovery=discovery_summary,
                    evaluation=evaluation_summary,
                    candidate=candidate,
                    validated=validated,
                )
            )
    groups.sort(
        key=lambda group: (
            group.kind,
            -group.discovery.outlier_fraction
            if np.isfinite(group.discovery.outlier_fraction)
            else np.inf,
            group.key,
        )
    )
    return VisibilityResidualAudit(
        score_threshold=score_threshold,
        minimum_group_samples=minimum_group_samples,
        minimum_group_outlier_fraction=minimum_group_outlier_fraction,
        scales=scales,
        discovery=discovery,
        evaluation=evaluation,
        groups=tuple(groups),
    )


def masks_excluding_groups(
    blocks: tuple[VisibilityBlock, ...],
    masks: tuple[np.ndarray, ...],
    groups: tuple[ResidualGroupSummary, ...],
) -> tuple[np.ndarray, ...]:
    """Remove the union of selected physical groups from existing masks."""

    if len(blocks) != len(masks):
        raise ValueError("blocks and masks must have equal lengths")
    result = tuple(np.asarray(mask, dtype=bool).copy() for mask in masks)
    for index, (block, mask) in enumerate(zip(blocks, result, strict=True)):
        if mask.shape != block.shape:
            raise ValueError(f"mask {index} must match its visibility block")
    for group in groups:
        if group.kind not in {"pointing", "baseline", "antenna", "channel", "correlation", "scan"}:
            raise ValueError(f"unsupported group kind {group.kind!r}")
        selected = _group_masks(blocks, group.kind, group.key)  # type: ignore[arg-type]
        result = tuple(mask & ~remove for mask, remove in zip(result, selected, strict=True))
    return result
