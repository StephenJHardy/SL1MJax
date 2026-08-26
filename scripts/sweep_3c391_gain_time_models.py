#!/usr/bin/env python3
"""Select a constrained 3C391 gain-time model on the frozen-sky fold."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

from sl1mjax.calibration import (
    CalibrationSolution,
    apply_calibration,
    baseline_jones,
    read_calibration,
    write_calibration,
)
from sl1mjax.calibration_diagnostics import evaluate_fixed_sky_calibration
from sl1mjax.data.averaging import average_frequency_bins, average_time_bins
from sl1mjax.data.canonical import (
    VisibilityBlock,
    VisibilityDataset,
    read_dataset,
    write_dataset,
)
from sl1mjax.gain_time_models import circular_gp_gain_solution, smooth_gain_solution
from sl1mjax.split import interleaved_time_fold_masks


def _dense_time_grid(solution: CalibrationSolution, step_s: float) -> np.ndarray:
    if not np.isfinite(step_s) or step_s <= 0:
        raise ValueError("dense time-grid step must be finite and positive")
    start = float(solution.gain_time_s[0])
    stop = float(solution.gain_time_s[-1])
    regular = np.arange(start, stop, step_s, dtype=np.float64)
    return np.unique(np.concatenate((regular, solution.gain_time_s, np.array([stop]))))


def _candidate_specs(
    solution: CalibrationSolution,
    *,
    smoothing_strengths: tuple[float, ...],
    gp_length_scales_s: tuple[float, ...],
    gp_noise_variances: tuple[float, ...],
    dense_step_s: float,
) -> list[tuple[str, CalibrationSolution, dict[str, Any]]]:
    native = replace(
        solution,
        interpolation="linear",
        provenance={**solution.provenance, "gain_time_model": "native"},
    )
    candidates: list[tuple[str, CalibrationSolution, dict[str, Any]]] = [
        (
            "native_linear",
            native,
            {"family": "native", "parameter_count": int(np.count_nonzero(native.gain_valid))},
        )
    ]
    for strength in smoothing_strengths:
        label = f"smooth_{strength:g}"
        modeled = replace(
            smooth_gain_solution(native, strength=strength),
            interpolation="linear",
        )
        candidates.append(
            (
                label,
                modeled,
                {
                    "family": "second_derivative",
                    "smoothing_strength": strength,
                    "parameter_count": int(np.count_nonzero(native.gain_valid)),
                },
            )
        )
    evaluation_time = _dense_time_grid(native, dense_step_s)
    for length_scale_s in gp_length_scales_s:
        for noise_variance in gp_noise_variances:
            label = f"circular_gp_l{length_scale_s:g}_n{noise_variance:g}"
            modeled = replace(
                circular_gp_gain_solution(
                    native,
                    evaluation_time,
                    length_scale_s=length_scale_s,
                    noise_variance=noise_variance,
                ),
                interpolation="linear",
            )
            candidates.append(
                (
                    label,
                    modeled,
                    {
                        "family": "circular_rbf_gp",
                        "length_scale_s": length_scale_s,
                        "noise_variance": noise_variance,
                        "dense_step_s": dense_step_s,
                        "parameter_count": int(np.count_nonzero(native.gain_valid)),
                    },
                )
            )
    if len({label for label, _, _ in candidates}) != len(candidates):
        raise ValueError("candidate labels are not unique")
    return candidates


def _concatenate(blocks: list[VisibilityBlock], *, label: str) -> VisibilityBlock:
    if not blocks:
        raise ValueError("cannot concatenate an empty block list")
    first = blocks[0]

    def rows(name: str) -> np.ndarray:
        return np.concatenate([np.asarray(getattr(block, name)) for block in blocks], axis=0)

    return VisibilityBlock(
        uvw_m=rows("uvw_m"),
        frequency_hz=first.frequency_hz,
        visibility=rows("visibility"),
        weight=rows("weight"),
        flag=rows("flag"),
        time_s=rows("time_s"),
        antenna1=rows("antenna1"),
        antenna2=rows("antenna2"),
        field_id=rows("field_id"),
        scan_id=rows("scan_id"),
        state_id=rows("state_id"),
        observation_id=rows("observation_id"),
        feed1=rows("feed1"),
        feed2=rows("feed2"),
        interval_s=rows("interval_s"),
        correlations=first.correlations,
        receptor_basis=first.receptor_basis,
        phase_centre_rad=first.phase_centre_rad,
        provenance={"experiment": label, "chunk_count": len(blocks)},
    )


def _assert_shared_fixed_terms(base: CalibrationSolution, candidate: CalibrationSolution) -> None:
    if base.correlations != candidate.correlations:
        raise ValueError("gain candidates have different correlations")
    for name in (
        "delays_s",
        "delay_valid",
        "bandpass",
        "bandpass_frequency_hz",
        "bandpass_valid",
        "antenna_position_offset_m",
    ):
        if not np.array_equal(getattr(base, name), getattr(candidate, name)):
            raise ValueError(f"gain candidates differ in fixed term {name}")


def _gain_only_baseline(
    solution: CalibrationSolution,
    block: VisibilityBlock,
) -> tuple[np.ndarray, np.ndarray]:
    frequency = np.asarray([solution.reference_frequency_hz], dtype=np.float64)
    gain_only = replace(
        solution,
        delays_s=np.zeros_like(solution.delays_s),
        delay_valid=np.ones_like(solution.delay_valid),
        bandpass=np.ones(
            (solution.antenna_count, 1, solution.receptor_count),
            dtype=np.complex128,
        ),
        bandpass_frequency_hz=frequency,
        bandpass_valid=np.ones((solution.antenna_count, 1, solution.receptor_count), dtype=bool),
        reference_frequency_hz=float(frequency[0]),
        antenna_position_offset_m=None,
    )
    baseline, valid = baseline_jones(
        gain_only,
        block.time_s,
        frequency,
        block.antenna1,
        block.antenna2,
        extrapolate=True,
    )
    return np.asarray(baseline), np.asarray(valid)


def _apply_shared_fixed_calibration(
    raw: VisibilityBlock,
    base_solution: CalibrationSolution,
    candidate_solution: CalibrationSolution,
    base_calibrated: VisibilityBlock | None = None,
) -> VisibilityBlock:
    """Apply a gain candidate by an exact ratio around shared fixed terms."""

    _assert_shared_fixed_terms(base_solution, candidate_solution)
    calibrated = (
        apply_calibration(raw, base_solution, extrapolate=True)
        if base_calibrated is None
        else base_calibrated
    )
    base_gain, base_valid = _gain_only_baseline(base_solution, raw)
    candidate_gain, candidate_valid = _gain_only_baseline(candidate_solution, raw)
    valid = (
        base_valid
        & candidate_valid
        & np.isfinite(base_gain)
        & np.isfinite(candidate_gain)
        & (np.abs(candidate_gain) > 0)
    )
    ratio = np.divide(
        base_gain,
        candidate_gain,
        out=np.ones_like(base_gain),
        where=valid,
    )
    return replace(
        calibrated,
        visibility=calibrated.visibility * ratio,
        flag=calibrated.flag | ~valid,
        provenance={
            **calibrated.provenance,
            "gain_time_candidate": candidate_solution.provenance.get("gain_time_model", "unknown"),
            "application": "exact gain ratio around shared fixed Jones terms",
        },
    )


def _extract_candidate_targets(
    measurement_set: Path,
    candidates: list[tuple[str, CalibrationSolution, dict[str, Any]]],
    *,
    field_id: int,
    frequency_bins: int,
    time_bin_s: float,
    chunk_rows: int,
) -> dict[str, VisibilityBlock]:
    """Apply every candidate in one pass over a target field."""

    from casacore import tables
    from image_3c391_target import _block

    if not candidates:
        raise ValueError("at least one gain candidate is required")
    chunks: dict[str, list[VisibilityBlock]] = {label: [] for label, _, _ in candidates}
    base_solution = candidates[0][1]
    for _, solution, _ in candidates[1:]:
        _assert_shared_fixed_terms(base_solution, solution)
    with (
        tables.table(str(measurement_set), readonly=True, ack=False) as main,
        tables.table(
            str(measurement_set / "SPECTRAL_WINDOW"), readonly=True, ack=False
        ) as spectral_window,
        tables.table(str(measurement_set / "FIELD"), readonly=True, ack=False) as field,
    ):
        selected = main.query(f"FIELD_ID=={field_id}")
        try:
            if selected.nrows() == 0:
                raise ValueError(f"field {field_id} has no rows")
            frequency_hz = np.asarray(spectral_window.getcell("CHAN_FREQ", 0), dtype=np.float64)
            direction = np.asarray(field.getcell("PHASE_DIR", field_id), dtype=np.float64).reshape(
                -1, 2
            )[0]
            phase_centre = (float(direction[0]), float(direction[1]))
            for start in range(0, selected.nrows(), chunk_rows):
                count = min(chunk_rows, selected.nrows() - start)
                flag = np.asarray(selected.getcol("FLAG", startrow=start, nrow=count), dtype=bool)
                flag_row = np.asarray(
                    selected.getcol("FLAG_ROW", startrow=start, nrow=count), dtype=bool
                )
                weight = np.asarray(
                    selected.getcol("WEIGHT", startrow=start, nrow=count),
                    dtype=np.float64,
                )
                uvw_m = np.asarray(
                    selected.getcol("UVW", startrow=start, nrow=count), dtype=np.float64
                )
                time_s = np.asarray(
                    selected.getcol("TIME", startrow=start, nrow=count), dtype=np.float64
                )
                antenna1 = np.asarray(
                    selected.getcol("ANTENNA1", startrow=start, nrow=count), dtype=np.int32
                )
                antenna2 = np.asarray(
                    selected.getcol("ANTENNA2", startrow=start, nrow=count), dtype=np.int32
                )
                scan_id = np.asarray(
                    selected.getcol("SCAN_NUMBER", startrow=start, nrow=count),
                    dtype=np.int32,
                )
                interval_s = np.asarray(
                    selected.getcol("INTERVAL", startrow=start, nrow=count), dtype=np.float64
                )
                raw = _block(
                    visibility=np.asarray(selected.getcol("DATA", startrow=start, nrow=count)),
                    flag=flag,
                    flag_row=flag_row,
                    weight=weight,
                    uvw_m=uvw_m,
                    frequency_hz=frequency_hz,
                    time_s=time_s,
                    antenna1=antenna1,
                    antenna2=antenna2,
                    scan_id=scan_id,
                    field_id=field_id,
                    interval_s=interval_s,
                    phase_centre_rad=phase_centre,
                    column="DATA",
                )
                base_calibrated = apply_calibration(raw, base_solution, extrapolate=True)
                for label, solution, _ in candidates:
                    calibrated = _apply_shared_fixed_calibration(
                        raw,
                        base_solution,
                        solution,
                        base_calibrated,
                    )
                    chunks[label].append(
                        average_frequency_bins(calibrated, bin_count=frequency_bins)
                    )
        finally:
            selected.close()
    return {
        label: average_time_bins(_concatenate(chunks[label], label=label), bin_seconds=time_bin_s)
        for label, _, _ in candidates
    }


def _assert_aligned(reference: VisibilityBlock, candidate: VisibilityBlock, label: str) -> None:
    if reference.shape != candidate.shape:
        raise ValueError(f"{label} shape does not match the frozen reference")
    for name in ("antenna1", "antenna2", "field_id", "scan_id"):
        if not np.array_equal(getattr(reference, name), getattr(candidate, name)):
            raise ValueError(f"{label} does not align on {name}")
    if not np.allclose(reference.time_s, candidate.time_s, rtol=0.0, atol=1e-6):
        raise ValueError(f"{label} does not align on time")
    if not np.allclose(reference.uvw_m, candidate.uvw_m, rtol=1e-12, atol=1e-9):
        raise ValueError(f"{label} does not align on UVW")


def _load_predictions(path: Path, blocks: tuple[VisibilityBlock, ...]) -> tuple[np.ndarray, ...]:
    with np.load(path) as stored:
        predictions = tuple(
            np.asarray(stored[f"prediction_C{index + 1}"]) for index in range(len(blocks))
        )
    for index, (block, prediction) in enumerate(zip(blocks, predictions, strict=True), start=1):
        if prediction.shape != block.shape or np.any(~np.isfinite(prediction)):
            raise ValueError(f"prediction_C{index} is incompatible with its block")
    return predictions


def _write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = tuple(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _selection_run(arguments: argparse.Namespace) -> dict[str, Any]:
    base = read_calibration(arguments.base_calibration)
    candidates = _candidate_specs(
        base,
        smoothing_strengths=tuple(arguments.smoothing_strengths),
        gp_length_scales_s=tuple(arguments.gp_length_scales_s),
        gp_noise_variances=tuple(arguments.gp_noise_variances),
        dense_step_s=arguments.dense_step_s,
    )
    calibration_directory = arguments.output / "calibrations"
    calibration_directory.mkdir(parents=True, exist_ok=True)
    for label, solution, _ in candidates:
        write_calibration(solution, calibration_directory / f"{label}.npz")

    reference_blocks = read_dataset(arguments.reference_fixture).blocks
    predictions = _load_predictions(arguments.composite_checkpoint, reference_blocks)
    candidate_blocks: dict[str, list[VisibilityBlock]] = {label: [] for label, _, _ in candidates}
    candidate_labels = [label for label, _, _ in candidates]
    cache_directory = arguments.output / "candidate_blocks"
    cache_directory.mkdir(parents=True, exist_ok=True)
    for index, reference in enumerate(reference_blocks, start=1):
        field_ids = np.unique(reference.field_id)
        if field_ids.size != 1:
            raise ValueError(f"reference C{index} mixes field IDs")
        cache_path = cache_directory / f"C{index}.zarr"
        if cache_path.exists():
            cached = read_dataset(cache_path)
            if list(cached.provenance.get("candidate_labels", [])) != candidate_labels:
                raise ValueError(f"cached C{index} candidate grid differs from this run")
            if len(cached.blocks) != len(candidate_labels):
                raise ValueError(f"cached C{index} has the wrong block count")
            extracted = dict(zip(candidate_labels, cached.blocks, strict=True))
            print(f"loaded cached C{index} for {len(candidates)} candidates", flush=True)
        else:
            print(f"extracting C{index} for {len(candidates)} candidates", flush=True)
            extracted = _extract_candidate_targets(
                arguments.measurement_set,
                candidates,
                field_id=int(field_ids[0]),
                frequency_bins=arguments.frequency_bins,
                time_bin_s=arguments.time_bin_s,
                chunk_rows=arguments.chunk_rows,
            )
            write_dataset(
                VisibilityDataset(
                    tuple(extracted[label] for label in candidate_labels),
                    provenance={
                        "experiment": "3C391 gain-time model selection cache",
                        "candidate_labels": candidate_labels,
                        "pointing": f"C{index}",
                    },
                ),
                cache_path,
            )
        for label, _, _ in candidates:
            _assert_aligned(reference, extracted[label], f"{label} C{index}")
            candidate_blocks[label].append(extracted[label])

    common_active = tuple(
        reference.active
        & np.logical_and.reduce(
            [candidate_blocks[label][index].active for label, _, _ in candidates]
        )
        for index, reference in enumerate(reference_blocks)
    )
    development, validation, _ = interleaved_time_fold_masks(
        reference_blocks, bin_seconds=arguments.time_fold_seconds
    )
    development = tuple(
        mask & active for mask, active in zip(development, common_active, strict=True)
    )
    validation = tuple(
        mask & active for mask, active in zip(validation, common_active, strict=True)
    )
    casa = evaluate_fixed_sky_calibration(
        "casa_corrected", reference_blocks, predictions, development, validation
    )
    metadata = {label: values for label, _, values in candidates}
    evaluations = []
    rows = []
    for label, _, _ in candidates:
        evaluation = evaluate_fixed_sky_calibration(
            label,
            tuple(candidate_blocks[label]),
            predictions,
            development,
            validation,
        )
        evaluations.append({"label": label, **metadata[label], "selection": asdict(evaluation)})
        rows.append(
            {
                "label": label,
                **metadata[label],
                "development_power": evaluation.train["normalized_residual_power"],
                "validation_power": evaluation.holdout["normalized_residual_power"],
                "validation_rms": evaluation.holdout["normalized_rms"],
            }
        )
    rows.sort(key=lambda row: float(row["validation_power"]))
    _write_table(arguments.output / "ranking.csv", rows)
    return {
        "schema_version": 1,
        "protocol": {
            "selection_fold": 3,
            "sealed_fold": 4,
            "sealed_opened": False,
            "common_active_samples": int(sum(np.count_nonzero(x) for x in common_active)),
            "target_flags": "post-application",
        },
        "base_calibration": str(arguments.base_calibration),
        "casa_corrected_selection": asdict(casa),
        "candidates": evaluations,
        "ranking": [row["label"] for row in rows],
        "selected_candidate": rows[0]["label"],
    }


def _sealed_run(arguments: argparse.Namespace, summary: dict[str, Any]) -> dict[str, Any]:
    if summary["protocol"].get("sealed_opened"):
        raise ValueError("sealed fold has already been opened for this selection")
    label = str(summary["selected_candidate"])
    solution = read_calibration(arguments.output / "calibrations" / f"{label}.npz")
    candidate = [(label, solution, {})]
    reference_blocks = read_dataset(arguments.reference_fixture).blocks
    predictions = _load_predictions(arguments.composite_checkpoint, reference_blocks)
    blocks = []
    for index, reference in enumerate(reference_blocks, start=1):
        field_ids = np.unique(reference.field_id)
        if field_ids.size != 1:
            raise ValueError(f"reference C{index} mixes field IDs")
        print(f"extracting sealed C{index} for {label}", flush=True)
        extracted = _extract_candidate_targets(
            arguments.measurement_set,
            candidate,
            field_id=int(field_ids[0]),
            frequency_bins=arguments.frequency_bins,
            time_bin_s=arguments.time_bin_s,
            chunk_rows=arguments.chunk_rows,
        )[label]
        _assert_aligned(reference, extracted, f"{label} C{index}")
        blocks.append(extracted)
    common_active = tuple(
        reference.active & block.active
        for reference, block in zip(reference_blocks, blocks, strict=True)
    )
    development, _, sealed = interleaved_time_fold_masks(
        reference_blocks, bin_seconds=arguments.time_fold_seconds
    )
    development = tuple(
        mask & active for mask, active in zip(development, common_active, strict=True)
    )
    sealed = tuple(mask & active for mask, active in zip(sealed, common_active, strict=True))
    selected_test = evaluate_fixed_sky_calibration(
        label, tuple(blocks), predictions, development, sealed
    )
    casa_test = evaluate_fixed_sky_calibration(
        "casa_corrected", reference_blocks, predictions, development, sealed
    )
    summary["protocol"]["sealed_opened"] = True
    summary["selected_sealed_test"] = asdict(selected_test)
    summary["casa_corrected_sealed_test"] = asdict(casa_test)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "measurement_set",
        type=Path,
        nargs="?",
        default=Path("data/3c391_work_v2/3c391_ctm_mosaic_10s_spw0.ms"),
    )
    parser.add_argument(
        "--base-calibration",
        type=Path,
        default=Path("outputs/3c391_full_scan_gain_baseline/full_scan_calibration.npz"),
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
        default=Path("outputs/3c391_gain_time_model_sweep"),
    )
    parser.add_argument(
        "--smoothing-strengths",
        type=float,
        nargs="+",
        default=(0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0),
    )
    parser.add_argument(
        "--gp-length-scales-s",
        type=float,
        nargs="+",
        default=(900.0, 1800.0, 3600.0, 7200.0),
    )
    parser.add_argument(
        "--gp-noise-variances",
        type=float,
        nargs="+",
        default=(0.001, 0.01, 0.1),
    )
    parser.add_argument("--dense-step-s", type=float, default=30.0)
    parser.add_argument("--frequency-bins", type=int, default=4)
    parser.add_argument("--time-bin-s", type=float, default=60.0)
    parser.add_argument("--chunk-rows", type=int, default=2048)
    parser.add_argument("--time-fold-seconds", type=float, default=60.0)
    parser.add_argument("--open-sealed", action="store_true")
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    summary_path = arguments.output / "summary.json"
    if arguments.open_sealed:
        if not summary_path.exists():
            raise ValueError("selection summary must exist before opening the sealed fold")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary = _sealed_run(arguments, summary)
    else:
        if summary_path.exists():
            raise ValueError("selection output already exists; use a new output directory")
        summary = _selection_run(arguments)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "ranking": summary["ranking"],
                "selected_candidate": summary["selected_candidate"],
                "sealed_opened": summary["protocol"]["sealed_opened"],
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
