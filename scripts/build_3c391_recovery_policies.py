#!/usr/bin/env python3
"""Build non-mutating 3C391 visibility-recovery policy fixtures."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from compare_3c391_composite_existing_flags import _load_predictions
from image_3c391_target import _concatenate

from sl1mjax.data.canonical import VisibilityBlock, VisibilityDataset, read_dataset, write_dataset
from sl1mjax.flagging import (
    ResidualHandlingMode,
    apply_residual_handling,
    baseline_group_masks,
)
from sl1mjax.residual_audit import (
    apply_robust_residual_scales,
    audit_visibility_residuals,
    robust_residual_scores,
)
from sl1mjax.split import interleaved_time_folds

POLICIES = ("active_only", "whole_baseline", "supported_tail", "robust_weights")


def _or_masks(*groups: tuple[np.ndarray, ...]) -> tuple[np.ndarray, ...]:
    if not groups or any(len(group) != len(groups[0]) for group in groups):
        raise ValueError("mask groups must have equal non-zero lengths")
    return tuple(
        np.logical_or.reduce([group[index] for group in groups])
        for index in range(len(groups[0]))
    )


def _transfer_folds(
    reference: tuple[VisibilityBlock, ...],
    candidate: tuple[VisibilityBlock, ...],
    reference_folds: tuple[tuple[np.ndarray, ...], ...],
    *,
    bin_seconds: float,
) -> tuple[tuple[np.ndarray, ...], ...]:
    """Transfer reference fold labels to candidate rows with shared time bins."""

    if len(reference) != len(candidate):
        raise ValueError("reference and candidate must have equal block counts")
    fold_count = len(reference_folds)
    result: list[list[np.ndarray]] = [[] for _ in range(fold_count)]
    for block_index, (source, target) in enumerate(zip(reference, candidate, strict=True)):
        source_bin = np.floor(source.time_s / bin_seconds).astype(np.int64)
        target_bin = np.floor(target.time_s / bin_seconds).astype(np.int64)
        fold_by_bin: dict[int, int] = {}
        for fold in range(fold_count):
            selected_rows = np.any(reference_folds[fold][block_index], axis=(1, 2))
            for value in np.unique(source_bin[selected_rows]):
                previous = fold_by_bin.setdefault(int(value), fold)
                if previous != fold:
                    raise ValueError("a reference time bin belongs to multiple folds")
        row_fold = np.asarray([fold_by_bin.get(int(value), -1) for value in target_bin])
        for fold in range(fold_count):
            result[fold].append(target.active & (row_fold == fold)[:, None, None])
    return tuple(tuple(masks) for masks in result)


def _mask_unshared_bins(
    blocks: tuple[VisibilityBlock, ...],
    folds: tuple[tuple[np.ndarray, ...], ...],
) -> tuple[VisibilityBlock, ...]:
    output = []
    for index, block in enumerate(blocks):
        shared = np.logical_or.reduce([fold[index] for fold in folds])
        output.append(replace(block, flag=block.flag | ~shared))
    return tuple(output)


def _policy_recovery_blocks(
    flagged: tuple[VisibilityBlock, ...],
    scores: tuple[np.ndarray, ...],
    support: tuple[np.ndarray, ...],
    policy: str,
    *,
    threshold: float,
) -> tuple[VisibilityBlock, ...]:
    if policy not in POLICIES[1:]:
        raise ValueError(f"unsupported recovery policy {policy!r}")
    output = []
    for block, score, instrumental_support in zip(flagged, scores, support, strict=True):
        if policy == "whole_baseline":
            output.append(replace(block, flag=block.flag | instrumental_support))
        elif policy == "supported_tail":
            output.append(
                replace(
                    block,
                    flag=block.flag | (instrumental_support & (score > threshold)),
                )
            )
        else:
            handling = apply_residual_handling(
                score,
                mode=ResidualHandlingMode.ROBUST_WEIGHTS,
                threshold=threshold,
            )
            output.append(
                replace(block, weight=block.weight * handling.weight_multiplier)
            )
    return tuple(output)


def _write_masks(
    path: Path,
    folds: tuple[tuple[np.ndarray, ...], ...],
) -> None:
    np.savez_compressed(
        path,
        **{
            f"fold{fold}_C{pointing + 1}": mask
            for fold, fold_masks in enumerate(folds)
            for pointing, mask in enumerate(fold_masks)
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--active-fixture",
        type=Path,
        default=Path("outputs/3c391_gain_time_model_sweep/selected_native_fixture.zarr"),
    )
    parser.add_argument(
        "--flagged-fixture",
        type=Path,
        default=Path("outputs/3c391_matched_existing_flag_audit/flagged_fixture.zarr"),
    )
    parser.add_argument(
        "--active-predictions",
        type=Path,
        default=Path("outputs/3c391_composite_catalogue_stage3/full_lambda_0.0003.npz"),
    )
    parser.add_argument(
        "--flagged-predictions",
        type=Path,
        default=Path(
            "outputs/3c391_calibration_flag_audit/composite_flagged_predictions.npz"
        ),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/3c391_recovery_policies")
    )
    parser.add_argument("--time-bin-seconds", type=float, default=60.0)
    parser.add_argument("--score-threshold", type=float, default=6.0)
    parser.add_argument("--minimum-group-samples", type=int, default=128)
    parser.add_argument("--minimum-group-outlier-fraction", type=float, default=0.2)
    arguments = parser.parse_args()

    active = read_dataset(arguments.active_fixture).blocks
    flagged = read_dataset(arguments.flagged_fixture).blocks
    active_predictions = _load_predictions(arguments.active_predictions, active)
    flagged_predictions = _load_predictions(arguments.flagged_predictions, flagged)
    active_folds = interleaved_time_folds(
        active, bin_seconds=arguments.time_bin_seconds
    )
    group_discovery = _or_masks(active_folds[0], active_folds[1])
    group_confirmation = active_folds[2]
    audit = audit_visibility_residuals(
        active,
        active_predictions,
        group_discovery,
        group_confirmation,
        score_threshold=arguments.score_threshold,
        group_kinds=("baseline",),
        minimum_group_samples=arguments.minimum_group_samples,
        minimum_group_outlier_fraction=arguments.minimum_group_outlier_fraction,
    )
    baselines = tuple(
        group.key for group in audit.groups if group.kind == "baseline" and group.validated
    )
    support = baseline_group_masks(flagged, baselines)
    _, scales = robust_residual_scores(active, active_predictions, group_discovery)
    scores = apply_robust_residual_scales(flagged, flagged_predictions, scales)
    flagged_folds = _transfer_folds(
        active,
        flagged,
        active_folds,
        bin_seconds=arguments.time_bin_seconds,
    )
    flagged = _mask_unshared_bins(flagged, flagged_folds)
    flagged_folds = _transfer_folds(
        active,
        flagged,
        active_folds,
        bin_seconds=arguments.time_bin_seconds,
    )

    arguments.output.mkdir(parents=True, exist_ok=True)
    policy_summaries = {}
    for policy in POLICIES:
        if policy == "active_only":
            blocks = active
            folds = active_folds
            recovered_count = 0
            effective_weight = 0.0
        else:
            recovery = _policy_recovery_blocks(
                flagged,
                scores,
                support,
                policy,
                threshold=arguments.score_threshold,
            )
            blocks = tuple(
                _concatenate([original, added], label=f"recovery_{policy}")
                for original, added in zip(active, recovery, strict=True)
            )
            folds = tuple(
                tuple(
                    np.concatenate((active_folds[fold][index], flagged_folds[fold][index]), axis=0)
                    & blocks[index].active
                    for index in range(len(blocks))
                )
                for fold in range(len(active_folds))
            )
            recovered_count = int(sum(np.count_nonzero(block.active) for block in recovery))
            effective_weight = float(sum(np.sum(block.weight[block.active]) for block in recovery))
        fixture = arguments.output / f"{policy}.zarr"
        write_dataset(
            VisibilityDataset(
                blocks,
                provenance={
                    "experiment": "3C391 non-mutating flagged-sample recovery",
                    "policy": policy,
                    "validated_baselines": [list(value) for value in baselines],
                },
            ),
            fixture,
        )
        _write_masks(arguments.output / f"{policy}_folds.npz", folds)
        policy_summaries[policy] = {
            "fixture": str(fixture),
            "folds": str(arguments.output / f"{policy}_folds.npz"),
            "active_sample_count": int(sum(np.count_nonzero(block.active) for block in blocks)),
            "recovered_sample_count": recovered_count,
            "recovered_effective_weight": effective_weight,
            "fold_sample_counts": [
                int(sum(np.count_nonzero(mask) for mask in fold)) for fold in folds
            ],
        }

    summary = {
        "schema_version": 1,
        "protocol": {
            "active_fixture": str(arguments.active_fixture),
            "flagged_fixture": str(arguments.flagged_fixture),
            "active_predictions": str(arguments.active_predictions),
            "flagged_predictions": str(arguments.flagged_predictions),
            "group_discovery_folds": [0, 1],
            "group_confirmation_fold": 2,
            "policy_selection_fold": 3,
            "sealed_test_fold": 4,
            "unshared_flagged_time_bins": "excluded",
            "score_threshold": arguments.score_threshold,
            "minimum_group_samples": arguments.minimum_group_samples,
            "minimum_group_outlier_fraction": arguments.minimum_group_outlier_fraction,
            "measurement_set_modified": False,
        },
        "validated_baselines": [list(value) for value in baselines],
        "baseline_audit": asdict(audit),
        "policies": policy_summaries,
    }
    (arguments.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "validated_baseline_count": len(baselines),
                "policies": policy_summaries,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
