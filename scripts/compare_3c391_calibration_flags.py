#!/usr/bin/env python3
"""Compare CASA and SL1MJax calibration in matched residual-flag audits."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
from compare_3c391_composite_existing_flags import (
    _cohort_audit,
    _components_from_checkpoint,
    _load_predictions,
    _predict_flagged,
)

from sl1mjax.beam import VLABeamCatalog, VLAPrimaryBeam
from sl1mjax.data.canonical import VisibilityBlock, read_dataset
from sl1mjax.direct_operator import DirectDFTConfig
from sl1mjax.residual_audit import VisibilityResidualAudit, audit_visibility_residuals
from sl1mjax.split import interleaved_time_fold_masks


def _assert_aligned(
    reference: tuple[VisibilityBlock, ...],
    candidate: tuple[VisibilityBlock, ...],
    cohort: str,
) -> None:
    if len(reference) != len(candidate):
        raise ValueError(f"{cohort} fixtures have different pointing counts")
    for index, (first, second) in enumerate(zip(reference, candidate, strict=True), start=1):
        if first.shape != second.shape:
            raise ValueError(f"{cohort} C{index} shapes differ")
        for name in ("antenna1", "antenna2", "field_id", "scan_id"):
            if not np.array_equal(getattr(first, name), getattr(second, name)):
                raise ValueError(f"{cohort} C{index} differs on {name}")
        for name, atol in (("time_s", 1e-6), ("uvw_m", 1e-9), ("frequency_hz", 1e-6)):
            if not np.allclose(
                getattr(first, name), getattr(second, name), rtol=1e-12, atol=atol
            ):
                raise ValueError(f"{cohort} C{index} differs on {name}")


def _matched_active(
    reference: tuple[VisibilityBlock, ...],
    candidate: tuple[VisibilityBlock, ...],
) -> tuple[tuple[VisibilityBlock, ...], tuple[VisibilityBlock, ...]]:
    """Give aligned fixtures exactly the same active visibility samples."""

    _assert_aligned(reference, candidate, "active")
    first_result = []
    second_result = []
    for first, second in zip(reference, candidate, strict=True):
        common = first.active & second.active
        first_result.append(replace(first, flag=~common))
        second_result.append(replace(second, flag=~common))
    return tuple(first_result), tuple(second_result)


def _matched_flagged(
    reference: tuple[VisibilityBlock, ...],
    candidate: tuple[VisibilityBlock, ...],
) -> tuple[tuple[VisibilityBlock, ...], tuple[VisibilityBlock, ...]]:
    """Restrict originally flagged cohorts to samples usable by both calibrations."""

    _assert_aligned(reference, candidate, "flagged")
    first_result = []
    second_result = []
    for first, second in zip(reference, candidate, strict=True):
        common = first.active & second.active
        first_result.append(replace(first, flag=~common))
        second_result.append(replace(second, flag=~common))
    return tuple(first_result), tuple(second_result)


def _audit_active(
    blocks: tuple[VisibilityBlock, ...],
    predictions: tuple[np.ndarray, ...],
    discovery: tuple[np.ndarray, ...],
    evaluation: tuple[np.ndarray, ...],
    *,
    score_threshold: float,
    fixed_scales: tuple[Any, ...] | None = None,
) -> VisibilityResidualAudit:
    return audit_visibility_residuals(
        blocks,
        predictions,
        discovery,
        evaluation,
        score_threshold=score_threshold,
        group_kinds=("pointing", "baseline", "antenna", "channel", "correlation", "scan"),
        minimum_group_samples=128,
        minimum_group_outlier_fraction=0.2,
        fixed_scales=fixed_scales,
    )


def _relative_change(reference: float, candidate: float) -> float:
    if not np.isfinite(reference) or reference <= 0 or not np.isfinite(candidate):
        return float("nan")
    return candidate / reference - 1.0


def _validated_groups(audit: VisibilityResidualAudit) -> set[tuple[str, tuple[int, ...]]]:
    return {(group.kind, group.key) for group in audit.groups if group.validated}


def _comparison(reference: VisibilityResidualAudit, candidate: VisibilityResidualAudit) -> dict:
    before = _validated_groups(reference)
    after = _validated_groups(candidate)
    return {
        "discovery_residual_power_relative": _relative_change(
            reference.discovery.normalized_residual_power,
            candidate.discovery.normalized_residual_power,
        ),
        "evaluation_residual_power_relative": _relative_change(
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
        "reference_validated_group_count": len(before),
        "candidate_validated_group_count": len(after),
        "validated_groups_removed": [
            {"kind": kind, "key": list(key)} for kind, key in sorted(before - after)
        ],
        "validated_groups_added": [
            {"kind": kind, "key": list(key)} for kind, key in sorted(after - before)
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("casa_fixture", type=Path)
    parser.add_argument("jax_fixture", type=Path)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--casa-flagged-fixture", type=Path)
    parser.add_argument("--jax-flagged-fixture", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/3c391_calibration_flag_audit")
    )
    parser.add_argument("--time-bin-seconds", type=float, default=60.0)
    parser.add_argument("--score-threshold", type=float, default=6.0)
    parser.add_argument("--visibility-tile-size", type=int, default=256)
    parser.add_argument("--pixel-tile-size", type=int, default=1024)
    parser.add_argument("--precision", choices=("float32", "float64"), default="float32")
    arguments = parser.parse_args()

    casa_active, jax_active = _matched_active(
        read_dataset(arguments.casa_fixture).blocks,
        read_dataset(arguments.jax_fixture).blocks,
    )
    if (arguments.casa_flagged_fixture is None) != (
        arguments.jax_flagged_fixture is None
    ):
        raise ValueError("both flagged fixtures must be provided together")

    active_predictions = _load_predictions(arguments.checkpoint, casa_active)
    discovery, _, evaluation = interleaved_time_fold_masks(
        casa_active, bin_seconds=arguments.time_bin_seconds
    )
    discovery = tuple(
        mask & block.active for mask, block in zip(discovery, casa_active, strict=True)
    )
    evaluation = tuple(
        mask & block.active for mask, block in zip(evaluation, casa_active, strict=True)
    )

    casa_active_audit = _audit_active(
        casa_active,
        active_predictions,
        discovery,
        evaluation,
        score_threshold=arguments.score_threshold,
    )
    jax_active_audit = _audit_active(
        jax_active,
        active_predictions,
        discovery,
        evaluation,
        score_threshold=arguments.score_threshold,
    )
    jax_active_common_scale = _audit_active(
        jax_active,
        active_predictions,
        discovery,
        evaluation,
        score_threshold=arguments.score_threshold,
        fixed_scales=casa_active_audit.scales,
    )

    arguments.output.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": 1,
        "protocol": {
            "sky_checkpoint": str(arguments.checkpoint),
            "sky_protocol": str(arguments.protocol),
            "active_samples": "intersection of CASA and SL1MJax post-calibration masks",
            "flagged_samples": "intersection of usable originally flagged samples",
            "discovery_folds": [0, 1, 2],
            "excluded_validation_fold": 3,
            "sealed_evaluation_fold": 4,
            "score_threshold": arguments.score_threshold,
            "flags_changed": False,
        },
        "fixtures": {
            "casa_active": str(arguments.casa_fixture),
            "jax_active": str(arguments.jax_fixture),
            "casa_flagged": (
                None
                if arguments.casa_flagged_fixture is None
                else str(arguments.casa_flagged_fixture)
            ),
            "jax_flagged": (
                None
                if arguments.jax_flagged_fixture is None
                else str(arguments.jax_flagged_fixture)
            ),
        },
        "active": {
            "casa": asdict(casa_active_audit),
            "jax": asdict(jax_active_audit),
            "jax_on_casa_scale": asdict(jax_active_common_scale),
            "change": _comparison(casa_active_audit, jax_active_audit),
            "common_scale_change": _comparison(casa_active_audit, jax_active_common_scale),
        },
    }
    printed = {
        "active_casa": asdict(casa_active_audit.evaluation),
        "active_jax": asdict(jax_active_audit.evaluation),
        "active_jax_on_casa_scale": asdict(jax_active_common_scale.evaluation),
        "active_change": summary["active"]["change"],
    }
    if arguments.casa_flagged_fixture is not None:
        assert arguments.jax_flagged_fixture is not None
        casa_flagged, jax_flagged = _matched_flagged(
            read_dataset(arguments.casa_flagged_fixture).blocks,
            read_dataset(arguments.jax_flagged_fixture).blocks,
        )
        protocol = json.loads(arguments.protocol.read_text(encoding="utf-8"))
        mosaic_phase_centre = casa_active[0].phase_centre_rad
        components = _components_from_checkpoint(
            arguments.checkpoint, protocol, mosaic_phase_centre
        )
        beam = VLAPrimaryBeam(
            kind="airy",
            catalog=replace(
                VLABeamCatalog(),
                airy_max_radius_rad_at_1ghz=np.deg2rad(
                    float(protocol["airy_max_radius_deg_at_1ghz"])
                ),
            ),
        )
        flagged_predictions = _predict_flagged(
            arguments.output,
            "composite",
            casa_flagged,
            components,
            mosaic_phase_centre,
            beam=beam,
            direct=DirectDFTConfig(
                visibility_chunk_size=arguments.visibility_tile_size,
                pixel_chunk_size=arguments.pixel_tile_size,
                precision=arguments.precision,
            ),
        )
        casa_flagged_audit = _cohort_audit(
            casa_active,
            active_predictions,
            casa_flagged,
            flagged_predictions,
            discovery,
            score_threshold=arguments.score_threshold,
        )
        jax_flagged_audit = _cohort_audit(
            jax_active,
            active_predictions,
            jax_flagged,
            flagged_predictions,
            discovery,
            score_threshold=arguments.score_threshold,
        )
        jax_flagged_common_scale = _cohort_audit(
            jax_active,
            active_predictions,
            jax_flagged,
            flagged_predictions,
            discovery,
            score_threshold=arguments.score_threshold,
            fixed_scales=casa_flagged_audit.scales,
        )
        summary["existing_flags"] = {
            "casa": asdict(casa_flagged_audit),
            "jax": asdict(jax_flagged_audit),
            "jax_on_casa_scale": asdict(jax_flagged_common_scale),
            "change": _comparison(casa_flagged_audit, jax_flagged_audit),
            "common_scale_change": _comparison(
                casa_flagged_audit, jax_flagged_common_scale
            ),
        }
        printed.update(
            {
                "flagged_casa": asdict(casa_flagged_audit.evaluation),
                "flagged_jax": asdict(jax_flagged_audit.evaluation),
                "flagged_jax_on_casa_scale": asdict(
                    jax_flagged_common_scale.evaluation
                ),
                "flagged_change": summary["existing_flags"]["change"],
            }
        )
    (arguments.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(printed, indent=2, sort_keys=True),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
