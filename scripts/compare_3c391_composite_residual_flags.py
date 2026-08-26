#!/usr/bin/env python3
"""Compare residual-tail flag evidence before and after wide-field sky repair."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from sl1mjax.data.canonical import VisibilityBlock, read_dataset
from sl1mjax.residual_audit import VisibilityResidualAudit, audit_visibility_residuals
from sl1mjax.split import interleaved_time_fold_masks


def _load_predictions(
    path: Path,
    blocks: tuple[VisibilityBlock, ...],
) -> tuple[np.ndarray, ...]:
    if not path.exists():
        raise ValueError(f"prediction checkpoint does not exist: {path}")
    with np.load(path) as stored:
        predictions = tuple(
            np.asarray(stored[f"prediction_C{index + 1}"]) for index in range(len(blocks))
        )
    for index, (block, prediction) in enumerate(zip(blocks, predictions, strict=True)):
        if prediction.shape != block.shape or np.any(~np.isfinite(prediction)):
            raise ValueError(f"prediction_C{index + 1} is incompatible with its block")
    return predictions


def _validated_baselines(audit: VisibilityResidualAudit) -> set[tuple[int, ...]]:
    return {group.key for group in audit.groups if group.kind == "baseline" and group.validated}


def _paired_group_changes(
    reference: VisibilityResidualAudit,
    candidate: VisibilityResidualAudit,
) -> list[dict[str, Any]]:
    candidate_by_key = {(group.kind, group.key): group for group in candidate.groups}
    records: list[dict[str, Any]] = []
    for before in reference.groups:
        after = candidate_by_key.get((before.kind, before.key))
        if after is None:
            raise ValueError(f"candidate audit lacks group {(before.kind, before.key)}")
        records.append(
            {
                "kind": before.kind,
                "key": list(before.key),
                "label": before.label,
                "reference_validated": before.validated,
                "candidate_validated": after.validated,
                "discovery_outlier_fraction_before": before.discovery.outlier_fraction,
                "discovery_outlier_fraction_after": after.discovery.outlier_fraction,
                "evaluation_outlier_fraction_before": before.evaluation.outlier_fraction,
                "evaluation_outlier_fraction_after": after.evaluation.outlier_fraction,
                "evaluation_outlier_fraction_change": (
                    after.evaluation.outlier_fraction - before.evaluation.outlier_fraction
                ),
                "evaluation_normalized_residual_power_before": (
                    before.evaluation.normalized_residual_power
                ),
                "evaluation_normalized_residual_power_after": (
                    after.evaluation.normalized_residual_power
                ),
                "evaluation_normalized_residual_power_change": (
                    after.evaluation.normalized_residual_power
                    - before.evaluation.normalized_residual_power
                ),
            }
        )
    return records


def _write_baseline_csv(path: Path, records: list[dict[str, Any]]) -> None:
    baselines = [record for record in records if record["kind"] == "baseline"]
    baselines.sort(key=lambda item: item["evaluation_outlier_fraction_change"])
    fields = tuple(key for key in baselines[0] if key != "key") if baselines else ()
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("key", *fields))
        writer.writeheader()
        for record in baselines:
            writer.writerow(
                {
                    **record,
                    "key": ":".join(map(str, record["key"])),
                }
            )


def _relative_change(before: float, after: float) -> float:
    if not np.isfinite(before) or before <= 0 or not np.isfinite(after):
        return float("nan")
    return after / before - 1.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "fixture",
        type=Path,
        nargs="?",
        default=Path("outputs/3c391_mosaic_imaging_fixture.zarr"),
    )
    parser.add_argument("reference_predictions", type=Path)
    parser.add_argument("candidate_predictions", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/3c391_composite_residual_flag_audit"),
    )
    parser.add_argument("--time-bin-seconds", type=float, default=60.0)
    parser.add_argument("--score-threshold", type=float, default=6.0)
    parser.add_argument("--minimum-baseline-samples", type=int, default=128)
    parser.add_argument("--minimum-baseline-outlier-fraction", type=float, default=0.2)
    arguments = parser.parse_args()

    blocks = read_dataset(arguments.fixture).blocks
    reference_predictions = _load_predictions(arguments.reference_predictions, blocks)
    candidate_predictions = _load_predictions(arguments.candidate_predictions, blocks)
    discovery_masks, _, evaluation_masks = interleaved_time_fold_masks(
        blocks,
        bin_seconds=arguments.time_bin_seconds,
    )
    audit_arguments = {
        "score_threshold": arguments.score_threshold,
        "minimum_group_samples": arguments.minimum_baseline_samples,
        "minimum_group_outlier_fraction": arguments.minimum_baseline_outlier_fraction,
    }
    reference = audit_visibility_residuals(
        blocks,
        reference_predictions,
        discovery_masks,
        evaluation_masks,
        **audit_arguments,
    )
    candidate = audit_visibility_residuals(
        blocks,
        candidate_predictions,
        discovery_masks,
        evaluation_masks,
        **audit_arguments,
    )
    candidate_on_reference_scale = audit_visibility_residuals(
        blocks,
        candidate_predictions,
        discovery_masks,
        evaluation_masks,
        fixed_scales=reference.scales,
        **audit_arguments,
    )
    paired = _paired_group_changes(reference, candidate)
    before_baselines = _validated_baselines(reference)
    after_baselines = _validated_baselines(candidate)
    arguments.output.mkdir(parents=True, exist_ok=True)
    _write_baseline_csv(arguments.output / "baseline_changes.csv", paired)
    summary = {
        "schema_version": 1,
        "fixture": str(arguments.fixture),
        "reference_predictions": str(arguments.reference_predictions),
        "candidate_predictions": str(arguments.candidate_predictions),
        "protocol": {
            "time_bin_seconds": arguments.time_bin_seconds,
            "discovery_folds": [0, 1, 2],
            "excluded_validation_fold": 3,
            "sealed_evaluation_fold": 4,
            "score_threshold": arguments.score_threshold,
            "minimum_baseline_samples": arguments.minimum_baseline_samples,
            "minimum_baseline_outlier_fraction": (arguments.minimum_baseline_outlier_fraction),
        },
        "reference": asdict(reference),
        "candidate": asdict(candidate),
        "candidate_on_reference_scale": asdict(candidate_on_reference_scale),
        "change": {
            "discovery_normalized_residual_power_relative": _relative_change(
                reference.discovery.normalized_residual_power,
                candidate.discovery.normalized_residual_power,
            ),
            "evaluation_normalized_residual_power_relative": _relative_change(
                reference.evaluation.normalized_residual_power,
                candidate.evaluation.normalized_residual_power,
            ),
            "discovery_outlier_fraction_relative": _relative_change(
                reference.discovery.outlier_fraction,
                candidate.discovery.outlier_fraction,
            ),
            "evaluation_outlier_fraction_relative": _relative_change(
                reference.evaluation.outlier_fraction,
                candidate.evaluation.outlier_fraction,
            ),
            "common_scale_discovery_outlier_fraction_relative": _relative_change(
                reference.discovery.outlier_fraction,
                candidate_on_reference_scale.discovery.outlier_fraction,
            ),
            "common_scale_evaluation_outlier_fraction_relative": _relative_change(
                reference.evaluation.outlier_fraction,
                candidate_on_reference_scale.evaluation.outlier_fraction,
            ),
            "reference_validated_baseline_count": len(before_baselines),
            "candidate_validated_baseline_count": len(after_baselines),
            "validated_baselines_removed": [
                list(key) for key in sorted(before_baselines - after_baselines)
            ],
            "validated_baselines_added": [
                list(key) for key in sorted(after_baselines - before_baselines)
            ],
        },
        "paired_groups": paired,
    }
    (arguments.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "reference_discovery": asdict(reference.discovery),
                "reference_evaluation": asdict(reference.evaluation),
                "candidate_discovery": asdict(candidate.discovery),
                "candidate_evaluation": asdict(candidate.evaluation),
                "candidate_on_reference_scale_discovery": asdict(
                    candidate_on_reference_scale.discovery
                ),
                "candidate_on_reference_scale_evaluation": asdict(
                    candidate_on_reference_scale.evaluation
                ),
                "change": summary["change"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
