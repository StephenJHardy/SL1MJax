#!/usr/bin/env python3
"""Fit one 3C391 gain-calibrator solution per complete calibrator scan."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

from sl1mjax.calibration import read_calibration, write_calibration
from sl1mjax.calibration_diagnostics import evaluate_fixed_sky_calibration
from sl1mjax.calibration_inference import (
    CalibrationSolveConfig,
    flux_scale_solution,
    solve_time_gains,
)
from sl1mjax.data.canonical import (
    VisibilityBlock,
    VisibilityDataset,
    read_dataset,
    write_dataset,
)
from sl1mjax.polarization import Correlation, ReceptorBasis
from sl1mjax.split import calibration_split, interleaved_time_fold_masks


def _saved_flags(flag_table: Any, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    selected = flag_table.selectrows(rows)
    try:
        flag = np.asarray(selected.getcol("FLAG"), dtype=bool)
        flag_row = (
            np.asarray(selected.getcol("FLAG_ROW"), dtype=bool)
            if "FLAG_ROW" in selected.colnames()
            else np.zeros(rows.size, dtype=bool)
        )
        return flag, flag_row
    finally:
        selected.close()


def _extract_calibrator(measurement_set: Path, *, field_id: int) -> VisibilityBlock:
    """Read all parallel-hand calibrator rows with pre-calibration flags."""

    from casacore import tables

    flag_path = Path(str(measurement_set) + ".flagversions") / "flags.sl1mjax_calibration_input"
    correlations = np.asarray([0, 3])
    with (
        tables.table(str(measurement_set), readonly=True, ack=False) as main,
        tables.table(str(flag_path), readonly=True, ack=False) as input_flags,
        tables.table(
            str(measurement_set / "SPECTRAL_WINDOW"), readonly=True, ack=False
        ) as spectral_window,
        tables.table(str(measurement_set / "FIELD"), readonly=True, ack=False) as field,
    ):
        selected = main.query(f"FIELD_ID=={field_id}")
        try:
            if selected.nrows() == 0:
                raise ValueError(f"field {field_id} has no rows")
            rows = np.asarray(selected.rownumbers(), dtype=np.int64)
            flag, flag_row = _saved_flags(input_flags, rows)
            visibility = np.asarray(selected.getcol("DATA"))[:, :, correlations]
            model = np.asarray(selected.getcol("MODEL_DATA"))[:, :, correlations]
            row_weight = np.asarray(selected.getcol("WEIGHT"), dtype=np.float64)[:, correlations]
            weight = np.broadcast_to(row_weight[:, None, :], visibility.shape).copy()
            direction = np.asarray(field.getcell("PHASE_DIR", field_id), dtype=np.float64).reshape(
                -1, 2
            )[0]
            return VisibilityBlock(
                uvw_m=np.asarray(selected.getcol("UVW"), dtype=np.float64),
                frequency_hz=np.asarray(spectral_window.getcell("CHAN_FREQ", 0), dtype=np.float64),
                visibility=visibility,
                model_visibility=model,
                weight=weight,
                flag=flag[:, :, correlations] | flag_row[:, None, None],
                time_s=np.asarray(selected.getcol("TIME"), dtype=np.float64),
                antenna1=np.asarray(selected.getcol("ANTENNA1"), dtype=np.int32),
                antenna2=np.asarray(selected.getcol("ANTENNA2"), dtype=np.int32),
                field_id=np.full(selected.nrows(), field_id, dtype=np.int32),
                scan_id=np.asarray(selected.getcol("SCAN_NUMBER"), dtype=np.int32),
                interval_s=np.asarray(selected.getcol("INTERVAL"), dtype=np.float64),
                correlations=(Correlation.RR, Correlation.LL),
                receptor_basis=ReceptorBasis.CIRCULAR,
                phase_centre_rad=(float(direction[0]), float(direction[1])),
                provenance={
                    "source": measurement_set.name,
                    "source_column": "DATA",
                    "model_column": "MODEL_DATA",
                    "field_id": field_id,
                    "flag_version": "sl1mjax_calibration_input",
                    "averaging": "none",
                },
            )
        finally:
            selected.close()


def _scan_gain_coordinates(block: VisibilityBlock) -> np.ndarray:
    """Assign each row the active-weight centroid of its calibrator scan."""

    row_weight = np.sum(np.where(block.active, block.weight, 0.0), axis=(1, 2))
    coordinates = np.empty(block.shape[0], dtype=np.float64)
    for scan in np.unique(block.scan_id):
        rows = block.scan_id == scan
        total = float(np.sum(row_weight[rows]))
        if not np.isfinite(total) or total <= 0:
            raise ValueError(f"calibrator scan {scan} has no active weight")
        coordinates[rows] = np.sum(block.time_s[rows] * row_weight[rows]) / total
    return coordinates


def _assert_aligned(reference: VisibilityBlock, candidate: VisibilityBlock) -> None:
    if reference.shape != candidate.shape:
        raise ValueError("candidate target shape differs from frozen reference")
    for name in ("antenna1", "antenna2", "field_id", "scan_id"):
        if not np.array_equal(getattr(reference, name), getattr(candidate, name)):
            raise ValueError(f"candidate target does not align on {name}")
    if not np.allclose(reference.time_s, candidate.time_s, rtol=0.0, atol=1e-6):
        raise ValueError("candidate target does not align on time")
    if not np.allclose(reference.uvw_m, candidate.uvw_m, rtol=1e-12, atol=1e-9):
        raise ValueError("candidate target does not align on UVW")


def _load_predictions(
    path: Path,
    blocks: tuple[VisibilityBlock, ...],
) -> tuple[np.ndarray, ...]:
    with np.load(path) as stored:
        predictions = tuple(
            np.asarray(stored[f"prediction_C{index + 1}"]) for index in range(len(blocks))
        )
    for index, (block, prediction) in enumerate(zip(blocks, predictions, strict=True), start=1):
        if prediction.shape != block.shape or np.any(~np.isfinite(prediction)):
            raise ValueError(f"prediction_C{index} is incompatible with its block")
    return predictions


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "measurement_set",
        type=Path,
        nargs="?",
        default=Path("data/3c391_work_v2/3c391_ctm_mosaic_10s_spw0.ms"),
    )
    parser.add_argument(
        "--calibrator-fixture",
        type=Path,
        default=Path("outputs/3c391_full_gain_calibrator_fixture.zarr"),
    )
    parser.add_argument(
        "--base-calibration",
        type=Path,
        default=Path(
            "outputs/3c391_calibration_composite_time_complexity/selected_calibration.npz"
        ),
    )
    parser.add_argument(
        "--calibration-summary",
        type=Path,
        default=Path("outputs/3c391_calibration_composite_time_complexity/summary.json"),
    )
    parser.add_argument(
        "--reference-fixture",
        type=Path,
        default=Path("outputs/3c391_mosaic_imaging_fixture.zarr"),
    )
    parser.add_argument(
        "--composite-checkpoint",
        type=Path,
        default=Path("outputs/3c391_composite_catalogue_stage3/full_lambda_0.0003.npz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/3c391_full_scan_gain_baseline"),
    )
    parser.add_argument("--field-id", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--max-chunk-rows", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--frequency-bins", type=int, default=4)
    parser.add_argument("--time-bin-s", type=float, default=60.0)
    parser.add_argument("--chunk-rows", type=int, default=2048)
    parser.add_argument("--time-fold-seconds", type=float, default=60.0)
    parser.add_argument("--solve-only", action="store_true")
    parser.add_argument("--open-sealed", action="store_true")
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)

    if arguments.calibrator_fixture.exists():
        dataset = read_dataset(arguments.calibrator_fixture)
        if len(dataset.blocks) != 1:
            raise ValueError("calibrator fixture must contain exactly one block")
        calibrator = dataset.blocks[0]
    else:
        calibrator = _extract_calibrator(
            arguments.measurement_set,
            field_id=arguments.field_id,
        )
        write_dataset(
            VisibilityDataset(
                (calibrator,),
                provenance={"experiment": "3C391 full-row gain calibration"},
            ),
            arguments.calibrator_fixture,
        )

    gain_time_s = _scan_gain_coordinates(calibrator)
    gain_knots = np.unique(gain_time_s)
    base = read_calibration(arguments.base_calibration)
    split = calibration_split(calibrator, seed=arguments.seed)
    config = CalibrationSolveConfig(
        iterations=arguments.iterations,
        learning_rate=arguments.learning_rate,
        max_chunk_rows=arguments.max_chunk_rows,
        seed=arguments.seed,
    )
    started = time.perf_counter()
    fit = solve_time_gains(
        calibrator,
        base,
        split=split,
        config=config,
        gain_time_s=gain_time_s,
    )
    solve_seconds = time.perf_counter() - started
    calibration_summary = json.loads(arguments.calibration_summary.read_text(encoding="utf-8"))
    flux_jy = float(calibration_summary["calibrator_metrics"]["secondary_flux_jy"])
    solution = replace(
        flux_scale_solution(fit.solution, flux_jy),
        interpolation="linear",
        provenance={
            **fit.solution.provenance,
            "gain_group": "calibrator scan",
            "gain_time_statistic": "active-weight centroid",
            "target_gain_interpolation": "linear",
        },
    )
    write_calibration(solution, arguments.output / "full_scan_calibration.npz")

    summary: dict[str, Any] = {
        "schema_version": 1,
        "measurement_set": arguments.measurement_set.name,
        "calibrator": {
            "field_id": arguments.field_id,
            "row_count": calibrator.shape[0],
            "active_sample_count": int(np.count_nonzero(calibrator.active)),
            "scan_ids": np.unique(calibrator.scan_id).tolist(),
            "gain_knot_count": int(gain_knots.size),
            "gain_time_s": gain_knots.tolist(),
            "train_rms": fit.train_rms,
            "holdout_rms": fit.holdout_rms,
            "flux_jy": flux_jy,
        },
        "optimizer": {
            "iterations": arguments.iterations,
            "learning_rate": arguments.learning_rate,
            "max_chunk_rows": arguments.max_chunk_rows,
            "solve_seconds": solve_seconds,
            "initial_loss": fit.losses[0],
            "final_loss": fit.losses[-1],
            "minimum_loss": min(fit.losses),
        },
    }

    if not arguments.solve_only:
        from image_3c391_target import _extract_target

        reference_blocks = read_dataset(arguments.reference_fixture).blocks
        predictions = _load_predictions(arguments.composite_checkpoint, reference_blocks)
        candidate_blocks = []
        for index, reference in enumerate(reference_blocks, start=1):
            field_ids = np.unique(reference.field_id)
            if field_ids.size != 1:
                raise ValueError(f"reference C{index} mixes field IDs")
            print(f"extracting full-scan target C{index}", flush=True)
            _, candidate = _extract_target(
                arguments.measurement_set,
                solution,
                field_id=int(field_ids[0]),
                frequency_bins=arguments.frequency_bins,
                time_bin_s=arguments.time_bin_s,
                chunk_rows=arguments.chunk_rows,
                raw_flag_source="post_application",
            )
            _assert_aligned(reference, candidate)
            candidate_blocks.append(candidate)
        candidates = tuple(candidate_blocks)
        common_active = tuple(
            reference.active & candidate.active
            for reference, candidate in zip(reference_blocks, candidates, strict=True)
        )
        development, validation, sealed = interleaved_time_fold_masks(
            reference_blocks,
            bin_seconds=arguments.time_fold_seconds,
        )
        development = tuple(
            mask & active for mask, active in zip(development, common_active, strict=True)
        )
        validation = tuple(
            mask & active for mask, active in zip(validation, common_active, strict=True)
        )
        selection = evaluate_fixed_sky_calibration(
            "full_scan_native_linear",
            candidates,
            predictions,
            development,
            validation,
        )
        summary["fixed_sky_validation"] = asdict(selection)
        if arguments.open_sealed:
            sealed = tuple(
                mask & active for mask, active in zip(sealed, common_active, strict=True)
            )
            test = evaluate_fixed_sky_calibration(
                "full_scan_native_linear",
                candidates,
                predictions,
                development,
                sealed,
            )
            summary["fixed_sky_sealed_test"] = asdict(test)

    (arguments.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
