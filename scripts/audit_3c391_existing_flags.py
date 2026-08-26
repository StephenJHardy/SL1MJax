#!/usr/bin/env python3
"""Test existing 3C391 flags against a sky fitted without flagged samples."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
from audit_3c391_residual_flags import _load_sky
from casacore import tables
from image_3c391_target import _block, _concatenate

from sl1mjax.beam import primary_beam_from_name
from sl1mjax.calibration import CalibrationSolution, apply_calibration, read_calibration
from sl1mjax.data.averaging import average_frequency_bins, average_time_bins
from sl1mjax.data.canonical import (
    VisibilityBlock,
    VisibilityDataset,
    read_dataset,
    write_dataset,
)
from sl1mjax.direct_operator import DirectDFTConfig
from sl1mjax.flagging import audit_existing_flags
from sl1mjax.inference import predict_mosaic_quadtree
from sl1mjax.residual_audit import audit_visibility_residuals, robust_residual_scores
from sl1mjax.sky import GaussianApproximation


def _flag_cohort(block: VisibilityBlock) -> VisibilityBlock:
    """Select currently flagged finite samples without changing their values."""

    return replace(
        block,
        flag=~block.flag,
        provenance={**dict(block.provenance), "cohort": "currently_flagged"},
    )


def _prepare_flagged_cohort(
    block: VisibilityBlock,
    calibration: CalibrationSolution | None,
) -> VisibilityBlock:
    """Activate original flags, then optionally calibrate their raw data values."""

    cohort = _flag_cohort(block)
    if calibration is None:
        return cohort
    return apply_calibration(cohort, calibration, extrapolate=True)


def _match_active_samples(
    reference: VisibilityBlock,
    candidate: VisibilityBlock,
) -> tuple[VisibilityBlock, VisibilityBlock]:
    """Apply a common pre-averaging active mask to aligned raw cohorts."""

    if reference.shape != candidate.shape:
        raise ValueError("matched cohorts must have the same shape")
    common = reference.active & candidate.active
    return replace(reference, flag=~common), replace(candidate, flag=~common)


def _extract_flagged_blocks(
    measurement_set: Path,
    *,
    field_ids: tuple[int, ...],
    frequency_bins: int,
    time_bin_s: float,
    chunk_rows: int,
    calibration: CalibrationSolution | None = None,
) -> tuple[tuple[VisibilityBlock, ...], tuple[VisibilityBlock, ...] | None]:
    result: list[VisibilityBlock] = []
    reference_result: list[VisibilityBlock] | None = [] if calibration is not None else None
    with (
        tables.table(str(measurement_set), readonly=True, ack=False) as main,
        tables.table(
            str(measurement_set / "SPECTRAL_WINDOW"), readonly=True, ack=False
        ) as spectral_window,
        tables.table(str(measurement_set / "FIELD"), readonly=True, ack=False) as field,
    ):
        frequency_hz = np.asarray(spectral_window.getcell("CHAN_FREQ", 0), dtype=np.float64)
        for field_id in field_ids:
            selected = main.query(f"FIELD_ID=={field_id}")
            chunks: list[VisibilityBlock] = []
            reference_chunks: list[VisibilityBlock] | None = (
                [] if calibration is not None else None
            )
            try:
                if selected.nrows() == 0:
                    raise ValueError(f"field {field_id} has no rows")
                direction = np.asarray(
                    field.getcell("PHASE_DIR", field_id), dtype=np.float64
                ).reshape(-1, 2)[0]
                phase_centre = (float(direction[0]), float(direction[1]))
                for start in range(0, selected.nrows(), chunk_rows):
                    count = min(chunk_rows, selected.nrows() - start)
                    flag = np.asarray(
                        selected.getcol("FLAG", startrow=start, nrow=count),
                        dtype=bool,
                    )
                    flag_row = np.asarray(
                        selected.getcol("FLAG_ROW", startrow=start, nrow=count),
                        dtype=bool,
                    )
                    source_column = "CORRECTED_DATA" if calibration is None else "DATA"
                    source = _block(
                        visibility=np.asarray(
                            selected.getcol(source_column, startrow=start, nrow=count)
                        ),
                        flag=flag,
                        flag_row=flag_row,
                        weight=np.asarray(
                            selected.getcol("WEIGHT", startrow=start, nrow=count),
                            dtype=np.float64,
                        ),
                        uvw_m=np.asarray(
                            selected.getcol("UVW", startrow=start, nrow=count),
                            dtype=np.float64,
                        ),
                        frequency_hz=frequency_hz,
                        time_s=np.asarray(
                            selected.getcol("TIME", startrow=start, nrow=count),
                            dtype=np.float64,
                        ),
                        antenna1=np.asarray(
                            selected.getcol("ANTENNA1", startrow=start, nrow=count),
                            dtype=np.int32,
                        ),
                        antenna2=np.asarray(
                            selected.getcol("ANTENNA2", startrow=start, nrow=count),
                            dtype=np.int32,
                        ),
                        scan_id=np.asarray(
                            selected.getcol("SCAN_NUMBER", startrow=start, nrow=count),
                            dtype=np.int32,
                        ),
                        field_id=field_id,
                        interval_s=np.asarray(
                            selected.getcol("INTERVAL", startrow=start, nrow=count),
                            dtype=np.float64,
                        ),
                        phase_centre_rad=phase_centre,
                        column=f"{source_column}/currently_flagged",
                    )
                    cohort = _prepare_flagged_cohort(source, calibration)
                    if calibration is not None:
                        reference = _block(
                            visibility=np.asarray(
                                selected.getcol(
                                    "CORRECTED_DATA", startrow=start, nrow=count
                                )
                            ),
                            flag=flag,
                            flag_row=flag_row,
                            weight=np.asarray(
                                selected.getcol("WEIGHT", startrow=start, nrow=count),
                                dtype=np.float64,
                            ),
                            uvw_m=np.asarray(
                                selected.getcol("UVW", startrow=start, nrow=count),
                                dtype=np.float64,
                            ),
                            frequency_hz=frequency_hz,
                            time_s=np.asarray(
                                selected.getcol("TIME", startrow=start, nrow=count),
                                dtype=np.float64,
                            ),
                            antenna1=np.asarray(
                                selected.getcol("ANTENNA1", startrow=start, nrow=count),
                                dtype=np.int32,
                            ),
                            antenna2=np.asarray(
                                selected.getcol("ANTENNA2", startrow=start, nrow=count),
                                dtype=np.int32,
                            ),
                            scan_id=np.asarray(
                                selected.getcol("SCAN_NUMBER", startrow=start, nrow=count),
                                dtype=np.int32,
                            ),
                            field_id=field_id,
                            interval_s=np.asarray(
                                selected.getcol("INTERVAL", startrow=start, nrow=count),
                                dtype=np.float64,
                            ),
                            phase_centre_rad=phase_centre,
                            column="CORRECTED_DATA/currently_flagged/matched",
                        )
                        reference, cohort = _match_active_samples(
                            _flag_cohort(reference), cohort
                        )
                        assert reference_chunks is not None
                        reference_chunks.append(
                            average_frequency_bins(reference, bin_count=frequency_bins)
                        )
                    chunks.append(
                        average_frequency_bins(cohort, bin_count=frequency_bins)
                    )
            finally:
                selected.close()
            result.append(
                average_time_bins(
                    _concatenate(chunks, label="currently_flagged"),
                    bin_seconds=time_bin_s,
                )
            )
            if reference_chunks is not None:
                assert reference_result is not None
                reference_result.append(
                    average_time_bins(
                        _concatenate(reference_chunks, label="currently_flagged_matched_casa"),
                        bin_seconds=time_bin_s,
                    )
                )
    return tuple(result), None if reference_result is None else tuple(reference_result)


def _combine_cohorts(
    active_blocks: tuple[VisibilityBlock, ...],
    active_predictions: tuple[np.ndarray, ...],
    flagged_blocks: tuple[VisibilityBlock, ...],
    flagged_predictions: tuple[np.ndarray, ...],
    active_discovery_masks: tuple[np.ndarray, ...] | None = None,
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
    ):
        raise ValueError("cohort blocks and predictions must have equal lengths")
    if active_discovery_masks is not None and len(active_discovery_masks) != len(active_blocks):
        raise ValueError("active_discovery_masks must contain one mask per active block")
    blocks: list[VisibilityBlock] = []
    predictions: list[np.ndarray] = []
    discovery_masks: list[np.ndarray] = []
    evaluation_masks: list[np.ndarray] = []
    selected_discovery = (
        tuple(block.active for block in active_blocks)
        if active_discovery_masks is None
        else active_discovery_masks
    )
    for active, active_prediction, flagged, flagged_prediction, active_discovery in zip(
        active_blocks,
        active_predictions,
        flagged_blocks,
        flagged_predictions,
        selected_discovery,
        strict=True,
    ):
        if active.frequency_hz.shape != flagged.frequency_hz.shape or not np.allclose(
            active.frequency_hz, flagged.frequency_hz
        ):
            raise ValueError("active and flagged cohorts must use the same frequencies")
        block = _concatenate([active, flagged], label="flag_audit")
        prediction = np.concatenate((active_prediction, flagged_prediction), axis=0)
        discovery = np.zeros(block.shape, dtype=bool)
        evaluation = np.zeros(block.shape, dtype=bool)
        if active_discovery.shape != active.shape or np.any(active_discovery & ~active.active):
            raise ValueError("active discovery mask must select active samples only")
        discovery[: active.shape[0]] = active_discovery
        evaluation[active.shape[0] :] = flagged.active
        blocks.append(block)
        predictions.append(prediction)
        discovery_masks.append(discovery)
        evaluation_masks.append(evaluation)
    return (
        tuple(blocks),
        tuple(predictions),
        tuple(discovery_masks),
        tuple(evaluation_masks),
    )


def _reason_masks(
    blocks: tuple[VisibilityBlock, ...],
    evaluation_masks: tuple[np.ndarray, ...],
    *,
    tutorial_flagged_antennas: frozenset[int],
) -> dict[str, tuple[np.ndarray, ...]]:
    result: dict[str, list[np.ndarray]] = {
        "flagged_antenna": [],
        "scan_1": [],
        "quack_or_overlap": [],
    }
    for block, evaluation in zip(blocks, evaluation_masks, strict=True):
        antenna_rows = np.isin(block.antenna1, tuple(tutorial_flagged_antennas)) | np.isin(
            block.antenna2, tuple(tutorial_flagged_antennas)
        )
        scan_rows = block.scan_id == 1
        result["flagged_antenna"].append(evaluation & antenna_rows[:, None, None])
        result["scan_1"].append(evaluation & scan_rows[:, None, None])
        result["quack_or_overlap"].append(evaluation & ~(antenna_rows | scan_rows)[:, None, None])
    return {key: tuple(value) for key, value in result.items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("measurement_set", type=Path)
    parser.add_argument(
        "--fixture",
        type=Path,
        default=Path("outputs/3c391_mosaic_imaging_fixture.zarr"),
    )
    parser.add_argument(
        "--frozen-directory",
        type=Path,
        default=Path("outputs/3c391_mosaic_hierarchical_frozen_104"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/3c391_existing_flag_audit"),
    )
    parser.add_argument("--frequency-bins", type=int, default=4)
    parser.add_argument("--time-bin-s", type=float, default=60.0)
    parser.add_argument("--chunk-rows", type=int, default=2048)
    parser.add_argument(
        "--calibration",
        type=Path,
        help="Optional SL1MJax calibration applied to raw DATA for the flagged cohort.",
    )
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Create or validate the flagged fixture, then stop before sky prediction.",
    )
    parser.add_argument("--score-threshold", type=float, default=6.0)
    parser.add_argument("--visibility-tile-size", type=int, default=256)
    parser.add_argument("--pixel-tile-size", type=int, default=1024)
    parser.add_argument("--precision", choices=("float32", "float64"), default="float32")
    arguments = parser.parse_args()

    dataset = read_dataset(arguments.fixture)
    active_blocks = dataset.blocks
    calibration = (
        None if arguments.calibration is None else read_calibration(arguments.calibration)
    )
    field_ids = tuple(int(block.field_id[0]) for block in active_blocks)
    arguments.output.mkdir(parents=True, exist_ok=True)
    flagged_fixture = arguments.output / "flagged_fixture.zarr"
    matched_reference_fixture = arguments.output / "matched_casa_flagged_fixture.zarr"
    if flagged_fixture.exists():
        print("loading cached currently flagged cohort", flush=True)
        flagged_blocks = read_dataset(flagged_fixture).blocks
        if calibration is not None:
            if not matched_reference_fixture.exists():
                raise ValueError(
                    "cached calibrated cohort lacks its matched CASA reference; "
                    "use a new output directory"
                )
            matched_reference_blocks = read_dataset(matched_reference_fixture).blocks
        else:
            matched_reference_blocks = None
    else:
        print("extracting currently flagged cohort", flush=True)
        flagged_blocks, matched_reference_blocks = _extract_flagged_blocks(
            arguments.measurement_set,
            field_ids=field_ids,
            frequency_bins=arguments.frequency_bins,
            time_bin_s=arguments.time_bin_s,
            chunk_rows=arguments.chunk_rows,
            calibration=calibration,
        )
        write_dataset(
            VisibilityDataset(
                flagged_blocks,
                provenance={
                    "measurement_set": arguments.measurement_set.name,
                    "cohort": (
                        "currently_flagged_CORRECTED_DATA"
                        if calibration is None
                        else "currently_flagged_DATA_with_SL1MJax_calibration"
                    ),
                    "calibration": (
                        None if arguments.calibration is None else str(arguments.calibration)
                    ),
                    "field_ids": list(field_ids),
                    "frequency_bins": arguments.frequency_bins,
                    "time_bin_s": arguments.time_bin_s,
                },
            ),
            flagged_fixture,
        )
        print(f"cached flagged cohort at {flagged_fixture}", flush=True)
        if matched_reference_blocks is not None:
            write_dataset(
                VisibilityDataset(
                    matched_reference_blocks,
                    provenance={
                        "measurement_set": arguments.measurement_set.name,
                        "cohort": "currently_flagged_CORRECTED_DATA_matched_before_averaging",
                        "matched_calibration": str(arguments.calibration),
                        "field_ids": list(field_ids),
                        "frequency_bins": arguments.frequency_bins,
                        "time_bin_s": arguments.time_bin_s,
                    },
                ),
                matched_reference_fixture,
            )
            print(
                f"cached matched CASA cohort at {matched_reference_fixture}",
                flush=True,
            )
    if arguments.extract_only:
        print(f"flagged fixture ready at {flagged_fixture}", flush=True)
        return 0
    with np.load(arguments.frozen_directory / "predictions.npz") as stored:
        active_predictions = tuple(
            stored[f"consensus_C{index + 1}"] for index in range(len(active_blocks))
        )
    frozen_summary = json.loads(
        (arguments.frozen_directory / "summary.json").read_text(encoding="utf-8")
    )
    sky = _load_sky(
        arguments.frozen_directory / "consensus_topology.csv",
        root_size=int(frozen_summary["root_size"]),
        root_pixel_size_rad=np.deg2rad(float(frozen_summary["root_pixel_arcsec"]) / 3600.0),
    )
    prediction_path = arguments.output / "flagged_predictions.npz"
    if prediction_path.exists():
        print("loading cached flagged predictions", flush=True)
        with np.load(prediction_path) as stored:
            flagged_predictions = tuple(
                stored[f"flagged_C{index + 1}"] for index in range(len(flagged_blocks))
            )
    else:
        print(
            "predicting frozen sky at currently flagged visibility coordinates",
            flush=True,
        )
        flagged_predictions = predict_mosaic_quadtree(
            flagged_blocks,
            sky.topology,
            sky.flux,
            active_blocks[0].phase_centre_rad,
            primary_beam=primary_beam_from_name("airy"),
            approximation=GaussianApproximation.WIDE_FIELD,
            config=DirectDFTConfig(
                visibility_chunk_size=arguments.visibility_tile_size,
                pixel_chunk_size=arguments.pixel_tile_size,
                precision=arguments.precision,
            ),
        )
        np.savez(
            prediction_path,
            **{
                f"flagged_C{index + 1}": prediction
                for index, prediction in enumerate(flagged_predictions)
            },
        )
    blocks, predictions, discovery_masks, evaluation_masks = _combine_cohorts(
        active_blocks,
        active_predictions,
        flagged_blocks,
        flagged_predictions,
    )
    audit = audit_visibility_residuals(
        blocks,
        predictions,
        discovery_masks,
        evaluation_masks,
        score_threshold=arguments.score_threshold,
        group_kinds=("pointing", "baseline", "antenna", "channel", "correlation", "scan"),
        minimum_group_samples=128,
        minimum_group_outlier_fraction=0.2,
    )
    score_arrays, _ = robust_residual_scores(blocks, predictions, discovery_masks)
    flattened_scores = np.concatenate([score[np.isfinite(score)] for score in score_arrays])
    flattened_existing_flags = np.concatenate(
        [
            evaluation[np.isfinite(score)]
            for evaluation, score in zip(evaluation_masks, score_arrays, strict=True)
        ]
    )
    threshold_sensitivity = {
        str(threshold): asdict(
            audit_existing_flags(
                flattened_scores,
                flattened_existing_flags,
                threshold=threshold,
            )
        )
        for threshold in (3.0, 6.0, 10.0, audit.discovery.score_p99)
    }
    with tables.table(
        str(arguments.measurement_set / "ANTENNA"), readonly=True, ack=False
    ) as antennas:
        antenna_names = tuple(map(str, antennas.getcol("NAME")))
    tutorial_names = frozenset(("ea05", "ea13", "ea15"))
    tutorial_antennas = frozenset(
        index for index, name in enumerate(antenna_names) if name in tutorial_names
    )
    reason_summaries: dict[str, Any] = {}
    for reason, reason_evaluation in _reason_masks(
        blocks,
        evaluation_masks,
        tutorial_flagged_antennas=tutorial_antennas,
    ).items():
        if not any(np.any(mask) for mask in reason_evaluation):
            continue
        reason_audit = audit_visibility_residuals(
            blocks,
            predictions,
            discovery_masks,
            reason_evaluation,
            score_threshold=arguments.score_threshold,
            group_kinds=(),
        )
        reason_summaries[reason] = asdict(reason_audit.evaluation)
    summary = {
        "schema_version": 1,
        "measurement_set": str(arguments.measurement_set),
        "fixture": str(arguments.fixture),
        "frozen_directory": str(arguments.frozen_directory),
        "interpretation": {
            "discovery": "currently unflagged samples; robust scales are fitted here",
            "evaluation": "currently flagged samples; none influence the sky or scales",
            "bulk_definition": f"robust residual score <= {arguments.score_threshold}",
        },
        "currently_unflagged": asdict(audit.discovery),
        "currently_flagged": {
            **asdict(audit.evaluation),
            "bulk_count": audit.evaluation.sample_count - audit.evaluation.outlier_count,
            "bulk_fraction": 1.0 - audit.evaluation.outlier_fraction,
        },
        "flag_reason_approximations": reason_summaries,
        "threshold_sensitivity": threshold_sensitivity,
        "tutorial_flagged_antennas": [
            {"id": antenna, "name": antenna_names[antenna]} for antenna in sorted(tutorial_antennas)
        ],
        "group_audit": asdict(audit),
    }
    (arguments.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "currently_unflagged": summary["currently_unflagged"],
                "currently_flagged": summary["currently_flagged"],
                "flag_reason_approximations": reason_summaries,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
