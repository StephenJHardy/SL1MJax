#!/usr/bin/env python3
"""Re-audit existing 3C391 flags with central and composite sky models."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

from sl1mjax.beam import VLABeamCatalog, VLAPrimaryBeam
from sl1mjax.composite import (
    MosaicPointComponent,
    MosaicQuadtreeComponent,
    MosaicSkyComponent,
    predict_mosaic_composite,
)
from sl1mjax.coordinates import radec_to_lmn
from sl1mjax.data.canonical import VisibilityBlock, read_dataset
from sl1mjax.direct_operator import DirectDFTConfig
from sl1mjax.quadtree import (
    QuadtreeGrid,
    QuadtreeLeaf,
    QuadtreeTopology,
    quadtree_sky_from_regular_grid,
)
from sl1mjax.residual_audit import (
    RobustResidualScale,
    VisibilityResidualAudit,
    audit_visibility_residuals,
)
from sl1mjax.split import interleaved_time_fold_masks


def _load_topology(
    path: Path,
    *,
    root_size: int,
    root_pixel_size_rad: float,
) -> QuadtreeTopology:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = tuple(csv.DictReader(stream))
    leaves = tuple(QuadtreeLeaf(int(row["level"]), int(row["iy"]), int(row["ix"])) for row in rows)
    if not leaves:
        raise ValueError("topology CSV must contain at least one leaf")
    return QuadtreeTopology(QuadtreeGrid(root_size, root_pixel_size_rad), leaves)


def _load_predictions(
    path: Path,
    blocks: tuple[VisibilityBlock, ...],
) -> tuple[np.ndarray, ...]:
    with np.load(path) as stored:
        predictions = tuple(
            np.asarray(stored[f"prediction_C{index + 1}"]) for index in range(len(blocks))
        )
    for index, (block, prediction) in enumerate(zip(blocks, predictions, strict=True)):
        if prediction.shape != block.shape or np.any(~np.isfinite(prediction)):
            raise ValueError(f"prediction_C{index + 1} is incompatible with its block")
    return predictions


def _relative_change(before: float, after: float) -> float:
    if not np.isfinite(before) or before <= 0 or not np.isfinite(after):
        return float("nan")
    return after / before - 1.0


def _combine_cohorts(
    active_blocks: tuple[VisibilityBlock, ...],
    active_predictions: tuple[np.ndarray, ...],
    flagged_blocks: tuple[VisibilityBlock, ...],
    flagged_predictions: tuple[np.ndarray, ...],
    active_discovery_masks: tuple[np.ndarray, ...],
) -> tuple[
    tuple[VisibilityBlock, ...],
    tuple[np.ndarray, ...],
    tuple[np.ndarray, ...],
    tuple[np.ndarray, ...],
]:
    if not (
        len(active_blocks)
        == len(active_predictions)
        == len(flagged_blocks)
        == len(flagged_predictions)
        == len(active_discovery_masks)
    ):
        raise ValueError("cohort arrays and masks must have equal lengths")
    blocks = []
    predictions = []
    discovery_masks = []
    evaluation_masks = []
    for active, active_prediction, flagged, flagged_prediction, active_discovery in zip(
        active_blocks,
        active_predictions,
        flagged_blocks,
        flagged_predictions,
        active_discovery_masks,
        strict=True,
    ):
        if active.frequency_hz.shape != flagged.frequency_hz.shape or not np.allclose(
            active.frequency_hz, flagged.frequency_hz
        ):
            raise ValueError("active and flagged cohorts must use the same frequencies")
        if active_discovery.shape != active.shape or np.any(active_discovery & ~active.active):
            raise ValueError("active discovery mask must select active samples only")
        row_fields = (
            "uvw_m",
            "visibility",
            "weight",
            "flag",
            "time_s",
            "antenna1",
            "antenna2",
            "scan_id",
            "field_id",
            "state_id",
            "observation_id",
            "feed1",
            "feed2",
            "interval_s",
        )
        updates: dict[str, Any] = {}
        for name in row_fields:
            first = getattr(active, name)
            second = getattr(flagged, name)
            if first is None or second is None:
                if first is not None or second is not None:
                    raise ValueError(f"cohort metadata {name} is present in only one fixture")
                updates[name] = None
            else:
                updates[name] = np.concatenate((first, second), axis=0)
        combined = replace(
            active,
            **updates,
            provenance={**dict(active.provenance), "cohort": "flag_audit"},
        )
        prediction = np.concatenate((active_prediction, flagged_prediction), axis=0)
        discovery = np.zeros(combined.shape, dtype=bool)
        evaluation = np.zeros(combined.shape, dtype=bool)
        discovery[: active.shape[0]] = active_discovery
        evaluation[active.shape[0] :] = flagged.active
        blocks.append(combined)
        predictions.append(prediction)
        discovery_masks.append(discovery)
        evaluation_masks.append(evaluation)
    return tuple(blocks), tuple(predictions), tuple(discovery_masks), tuple(evaluation_masks)


def _components_from_checkpoint(
    checkpoint: Path,
    protocol: dict[str, Any],
    mosaic_phase_centre_rad: tuple[float, float],
) -> tuple[MosaicSkyComponent, ...]:
    frozen_directory = Path(protocol["frozen_directory"])
    frozen_summary = json.loads((frozen_directory / "summary.json").read_text(encoding="utf-8"))
    central_topology = _load_topology(
        frozen_directory / "consensus_topology.csv",
        root_size=int(frozen_summary["root_size"]),
        root_pixel_size_rad=np.deg2rad(float(frozen_summary["root_pixel_arcsec"]) / 3600.0),
    )
    with np.load(checkpoint) as stored:
        components: list[MosaicSkyComponent] = [
            MosaicQuadtreeComponent(
                "central",
                central_topology,
                np.asarray(stored["flux_central"]),
            )
        ]
        if "flux_coarse" in stored:
            coarse = quadtree_sky_from_regular_grid(
                int(protocol["coarse_size"]),
                np.deg2rad(float(protocol["coarse_pixel_arcsec"]) / 3600.0),
                np.asarray(stored["flux_coarse"]),
            )
            components.append(MosaicQuadtreeComponent("coarse", coarse.topology, coarse.flux))
        if "flux_catalogue" in stored:
            atom_metadata = protocol["catalog_atoms"]
            ra = np.deg2rad([float(atom["ra_deg"]) for atom in atom_metadata])
            dec = np.deg2rad([float(atom["dec_deg"]) for atom in atom_metadata])
            l_rad, m_rad, _ = radec_to_lmn(
                mosaic_phase_centre_rad[0],
                mosaic_phase_centre_rad[1],
                ra,
                dec,
            )
            components.append(
                MosaicPointComponent(
                    "catalogue",
                    l_rad,
                    m_rad,
                    np.asarray(stored["flux_catalogue"]),
                )
            )
    return tuple(components)


def _predict_flagged(
    output: Path,
    label: str,
    flagged_blocks: tuple[VisibilityBlock, ...],
    components: tuple[MosaicSkyComponent, ...],
    mosaic_phase_centre_rad: tuple[float, float],
    *,
    beam: VLAPrimaryBeam,
    direct: DirectDFTConfig,
) -> tuple[np.ndarray, ...]:
    path = output / f"{label}_flagged_predictions.npz"
    if path.exists():
        return _load_predictions(path, flagged_blocks)
    predictions = predict_mosaic_composite(
        flagged_blocks,
        components,
        mosaic_phase_centre_rad,
        primary_beam=beam,
        config=direct,
    )
    np.savez(
        path,
        **{f"prediction_C{index + 1}": prediction for index, prediction in enumerate(predictions)},
    )
    return predictions


def _cohort_audit(
    active_blocks: tuple[VisibilityBlock, ...],
    active_predictions: tuple[np.ndarray, ...],
    flagged_blocks: tuple[VisibilityBlock, ...],
    flagged_predictions: tuple[np.ndarray, ...],
    train_masks: tuple[np.ndarray, ...],
    *,
    score_threshold: float,
    fixed_scales: tuple[RobustResidualScale, ...] | None = None,
) -> VisibilityResidualAudit:
    blocks, predictions, discovery, evaluation = _combine_cohorts(
        active_blocks,
        active_predictions,
        flagged_blocks,
        flagged_predictions,
        train_masks,
    )
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "fixture",
        type=Path,
        nargs="?",
        default=Path("outputs/3c391_mosaic_imaging_fixture.zarr"),
    )
    parser.add_argument("flagged_fixture", type=Path)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("reference_checkpoint", type=Path)
    parser.add_argument("candidate_checkpoint", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/3c391_composite_existing_flag_audit"),
    )
    parser.add_argument("--time-bin-seconds", type=float, default=60.0)
    parser.add_argument("--score-threshold", type=float, default=6.0)
    parser.add_argument("--visibility-tile-size", type=int, default=256)
    parser.add_argument("--pixel-tile-size", type=int, default=1024)
    parser.add_argument("--precision", choices=("float32", "float64"), default="float32")
    arguments = parser.parse_args()

    active_blocks = read_dataset(arguments.fixture).blocks
    flagged_blocks = read_dataset(arguments.flagged_fixture).blocks
    if len(active_blocks) != len(flagged_blocks):
        raise ValueError("active and flagged fixtures must have equal pointing counts")
    protocol = json.loads(arguments.protocol.read_text(encoding="utf-8"))
    mosaic_phase_centre = active_blocks[0].phase_centre_rad
    beam = VLAPrimaryBeam(
        kind="airy",
        catalog=replace(
            VLABeamCatalog(),
            airy_max_radius_rad_at_1ghz=np.deg2rad(float(protocol["airy_max_radius_deg_at_1ghz"])),
        ),
    )
    direct = DirectDFTConfig(
        visibility_chunk_size=arguments.visibility_tile_size,
        pixel_chunk_size=arguments.pixel_tile_size,
        precision=arguments.precision,
    )
    reference_components = _components_from_checkpoint(
        arguments.reference_checkpoint,
        protocol,
        mosaic_phase_centre,
    )
    candidate_components = _components_from_checkpoint(
        arguments.candidate_checkpoint,
        protocol,
        mosaic_phase_centre,
    )
    arguments.output.mkdir(parents=True, exist_ok=True)
    reference_active = _load_predictions(arguments.reference_checkpoint, active_blocks)
    candidate_active = _load_predictions(arguments.candidate_checkpoint, active_blocks)
    reference_flagged = _predict_flagged(
        arguments.output,
        "reference",
        flagged_blocks,
        reference_components,
        mosaic_phase_centre,
        beam=beam,
        direct=direct,
    )
    candidate_flagged = _predict_flagged(
        arguments.output,
        "candidate",
        flagged_blocks,
        candidate_components,
        mosaic_phase_centre,
        beam=beam,
        direct=direct,
    )
    train_masks, _, _ = interleaved_time_fold_masks(
        active_blocks,
        bin_seconds=arguments.time_bin_seconds,
    )
    reference = _cohort_audit(
        active_blocks,
        reference_active,
        flagged_blocks,
        reference_flagged,
        train_masks,
        score_threshold=arguments.score_threshold,
    )
    candidate = _cohort_audit(
        active_blocks,
        candidate_active,
        flagged_blocks,
        candidate_flagged,
        train_masks,
        score_threshold=arguments.score_threshold,
    )
    candidate_on_reference_scale = _cohort_audit(
        active_blocks,
        candidate_active,
        flagged_blocks,
        candidate_flagged,
        train_masks,
        score_threshold=arguments.score_threshold,
        fixed_scales=reference.scales,
    )
    summary = {
        "schema_version": 1,
        "fixture": str(arguments.fixture),
        "flagged_fixture": str(arguments.flagged_fixture),
        "protocol": str(arguments.protocol),
        "reference_checkpoint": str(arguments.reference_checkpoint),
        "candidate_checkpoint": str(arguments.candidate_checkpoint),
        "score_threshold": arguments.score_threshold,
        "reference": asdict(reference),
        "candidate": asdict(candidate),
        "candidate_on_reference_scale": asdict(candidate_on_reference_scale),
        "change": {
            "unflagged_residual_power_relative": _relative_change(
                reference.discovery.normalized_residual_power,
                candidate.discovery.normalized_residual_power,
            ),
            "flagged_residual_power_relative": _relative_change(
                reference.evaluation.normalized_residual_power,
                candidate.evaluation.normalized_residual_power,
            ),
            "flagged_outlier_fraction_relative": _relative_change(
                reference.evaluation.outlier_fraction,
                candidate.evaluation.outlier_fraction,
            ),
            "common_scale_flagged_outlier_fraction_relative": _relative_change(
                reference.evaluation.outlier_fraction,
                candidate_on_reference_scale.evaluation.outlier_fraction,
            ),
        },
    }
    (arguments.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "reference_unflagged": asdict(reference.discovery),
                "reference_flagged": asdict(reference.evaluation),
                "candidate_unflagged": asdict(candidate.discovery),
                "candidate_flagged": asdict(candidate.evaluation),
                "candidate_on_reference_scale_flagged": asdict(
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
