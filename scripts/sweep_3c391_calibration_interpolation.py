#!/usr/bin/env python3
"""Select gain time complexity against a frozen composite 3C391 sky."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

from sl1mjax.calibration import write_calibration
from sl1mjax.calibration_diagnostics import evaluate_fixed_sky_calibration
from sl1mjax.calibration_inference import CalibrationSolveConfig
from sl1mjax.data.canonical import (
    VisibilityBlock,
    VisibilityDataset,
    read_dataset,
    write_dataset,
)
from sl1mjax.residual_audit import audit_visibility_residuals
from sl1mjax.split import interleaved_time_fold_masks

_TIME_MODEL_DEGREE = {
    "constant": 0,
    "linear_trend": 1,
    "quadratic_trend": 2,
}


def _assert_aligned(reference: VisibilityBlock, candidate: VisibilityBlock, label: str) -> None:
    """Require row/channel alignment before applying frozen masks or predictions."""

    if candidate.shape != reference.shape:
        raise ValueError(f"{label} shape {candidate.shape} != reference {reference.shape}")
    for name in ("antenna1", "antenna2", "scan_id"):
        expected = getattr(reference, name)
        actual = getattr(candidate, name)
        if expected is None or actual is None or not np.array_equal(actual, expected):
            raise ValueError(f"{label} does not align on {name}")
    if not np.allclose(candidate.time_s, reference.time_s, rtol=0.0, atol=1e-6):
        raise ValueError(f"{label} does not align on time")
    if not np.allclose(candidate.frequency_hz, reference.frequency_hz, rtol=1e-12):
        raise ValueError(f"{label} does not align on frequency")
    if not np.allclose(candidate.uvw_m, reference.uvw_m, rtol=1e-12, atol=1e-9):
        raise ValueError(f"{label} does not align on UVW coordinates")


def _evaluation_payload(evaluation: Any) -> dict[str, Any]:
    return asdict(evaluation)


def _load_composite_predictions(
    path: Path,
    blocks: tuple[VisibilityBlock, ...],
) -> tuple[np.ndarray, ...]:
    """Load finite per-pointing predictions from a composite checkpoint."""

    if not path.exists():
        raise ValueError(f"composite checkpoint does not exist: {path}")
    with np.load(path) as stored:
        predictions = tuple(
            np.asarray(stored[f"prediction_C{index + 1}"]) for index in range(len(blocks))
        )
    for index, (block, prediction) in enumerate(zip(blocks, predictions, strict=True), start=1):
        if prediction.shape != block.shape:
            raise ValueError(f"prediction_C{index} shape {prediction.shape} != {block.shape}")
        if np.any(~np.isfinite(prediction)):
            raise ValueError(f"prediction_C{index} contains non-finite values")
    return predictions


def _polynomial_time_model(solution: Any, model: str) -> Any:
    """Project solved complex gains onto a low-order time model.

    Log amplitude and unwrapped phase are fitted separately. Missing antenna
    solutions are inferred from the fitted trend when at least one knot is
    valid. The native model is returned unchanged.
    """

    if model == "native":
        return replace(
            solution,
            provenance={
                **solution.provenance,
                "gain_time_model": "native",
                "gain_time_native_knot_count": int(solution.gain_time_s.size),
            },
        )
    if model not in _TIME_MODEL_DEGREE:
        raise ValueError(f"unknown time model {model!r}")
    degree = _TIME_MODEL_DEGREE[model]
    times = np.asarray(solution.gain_time_s, dtype=np.float64)
    if times.size == 0:
        raise ValueError("gain solution has no time knots")
    centre = float(np.mean(times))
    scale = float(np.max(np.abs(times - centre)))
    coordinate = np.zeros_like(times) if scale == 0 else (times - centre) / scale
    gains = np.ones_like(solution.gains)
    valid = np.zeros_like(solution.gain_valid)
    for antenna in range(solution.antenna_count):
        for receptor in range(solution.receptor_count):
            selected = (
                solution.gain_valid[:, antenna, receptor]
                & np.isfinite(solution.gains[:, antenna, receptor])
                & (np.abs(solution.gains[:, antenna, receptor]) > 0)
            )
            count = int(np.count_nonzero(selected))
            if count == 0:
                continue
            effective_degree = min(degree, count - 1)
            values = solution.gains[selected, antenna, receptor]
            log_amplitude = np.log(np.abs(values))
            phase = np.unwrap(np.angle(values))
            amplitude_coefficients = np.polynomial.polynomial.polyfit(
                coordinate[selected], log_amplitude, effective_degree
            )
            phase_coefficients = np.polynomial.polynomial.polyfit(
                coordinate[selected], phase, effective_degree
            )
            fitted_amplitude = np.polynomial.polynomial.polyval(coordinate, amplitude_coefficients)
            fitted_phase = np.polynomial.polynomial.polyval(coordinate, phase_coefficients)
            gains[:, antenna, receptor] = np.exp(fitted_amplitude + 1j * fitted_phase)
            valid[:, antenna, receptor] = True
    reference_phase = np.angle(gains[:, solution.reference_antenna, :])
    gains *= np.exp(-1j * reference_phase[:, None, :])
    return replace(
        solution,
        gains=gains,
        gain_valid=valid,
        provenance={
            **solution.provenance,
            "gain_time_model": model,
            "gain_time_polynomial_degree": degree,
            "gain_time_native_knot_count": int(times.size),
        },
    )


def _candidate_specs(
    solution: Any,
    time_models: list[str],
    interpolations: list[str],
) -> list[tuple[str, Any, str, str]]:
    """Build unique time-model/interpolation calibration candidates."""

    candidates: list[tuple[str, Any, str, str]] = []
    for model in dict.fromkeys(time_models):
        modeled = _polynomial_time_model(solution, model)
        selected_interpolations = (
            ("nearest",) if model == "constant" else tuple(dict.fromkeys(interpolations))
        )
        for interpolation in selected_interpolations:
            label = f"{model}_{interpolation}"
            candidates.append(
                (
                    label,
                    replace(
                        modeled,
                        interpolation=interpolation,
                        provenance={
                            **modeled.provenance,
                            "target_gain_interpolation": interpolation,
                        },
                    ),
                    model,
                    interpolation,
                )
            )
    if not candidates:
        raise ValueError("at least one calibration candidate is required")
    return candidates


def _write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Solve the external calibrators once, project the gains onto "
            "several time models, apply them to raw 3C391 DATA, and select "
            "complexity against a frozen composite sky."
        )
    )
    parser.add_argument("measurement_set", type=Path)
    parser.add_argument(
        "--golden",
        type=Path,
        default=Path("tests/fixtures/3c391_calibration_golden.npz"),
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
        default=Path("outputs/3c391_calibration_interpolation_sweep"),
    )
    parser.add_argument(
        "--interpolations",
        choices=("nearest", "linear"),
        nargs="+",
        default=("nearest", "linear"),
    )
    parser.add_argument(
        "--time-models",
        choices=("constant", "linear_trend", "quadratic_trend", "native"),
        nargs="+",
        default=("constant", "linear_trend", "quadratic_trend", "native"),
    )
    parser.add_argument("--frequency-bins", type=int, default=4)
    parser.add_argument("--time-bin-s", type=float, default=60.0)
    parser.add_argument("--chunk-rows", type=int, default=2048)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--time-fold-seconds", type=float, default=60.0)
    arguments = parser.parse_args()

    # These 3C391-specific extraction helpers stream the MS and preserve the
    # frozen pre-calibration flags. Importing here keeps --help usable without
    # the optional python-casacore dependency.
    from image_3c391_target import _extract_target, _solve_calibration

    reference_blocks = read_dataset(arguments.reference_fixture).blocks
    predictions = _load_composite_predictions(arguments.composite_checkpoint, reference_blocks)

    solve_config = CalibrationSolveConfig(
        iterations=arguments.iterations,
        learning_rate=arguments.learning_rate,
        seed=arguments.seed,
    )
    solution, calibrator_metrics = _solve_calibration(arguments.golden, solve_config)
    candidate_specs = _candidate_specs(
        solution, list(arguments.time_models), list(arguments.interpolations)
    )
    candidates: dict[str, tuple[VisibilityBlock, ...]] = {}
    candidate_metadata: dict[str, dict[str, str]] = {}
    for label, selected_solution, time_model, interpolation in candidate_specs:
        blocks = []
        print(f"extracting {label} target calibration", flush=True)
        for index, reference in enumerate(reference_blocks, start=1):
            if reference.field_id is None:
                raise ValueError(f"reference pointing C{index} has no field ID")
            field_ids = np.unique(reference.field_id)
            if field_ids.size != 1:
                raise ValueError(f"reference pointing C{index} mixes field IDs")
            _, calibrated = _extract_target(
                arguments.measurement_set,
                selected_solution,
                field_id=int(field_ids[0]),
                frequency_bins=arguments.frequency_bins,
                time_bin_s=arguments.time_bin_s,
                chunk_rows=arguments.chunk_rows,
                raw_flag_source="post_application",
            )
            _assert_aligned(reference, calibrated, f"{label} C{index}")
            blocks.append(calibrated)
        candidates[label] = tuple(blocks)
        candidate_metadata[label] = {
            "time_model": time_model,
            "interpolation": interpolation,
        }

    common_active = tuple(
        reference.active
        & np.logical_and.reduce([blocks[index].active for blocks in candidates.values()])
        for index, reference in enumerate(reference_blocks)
    )
    development, validation, sealed_test = interleaved_time_fold_masks(
        reference_blocks,
        bin_seconds=arguments.time_fold_seconds,
    )
    development_masks = tuple(
        mask & active for mask, active in zip(development, common_active, strict=True)
    )
    validation_masks = tuple(
        mask & active for mask, active in zip(validation, common_active, strict=True)
    )
    test_masks = tuple(
        mask & active for mask, active in zip(sealed_test, common_active, strict=True)
    )
    reference_selection = evaluate_fixed_sky_calibration(
        "casa_corrected",
        reference_blocks,
        predictions,
        development_masks,
        validation_masks,
    )
    reference_test = evaluate_fixed_sky_calibration(
        "casa_corrected",
        reference_blocks,
        predictions,
        development_masks,
        test_masks,
    )
    evaluations = []
    table_rows = []
    for label, blocks in candidates.items():
        selection = evaluate_fixed_sky_calibration(
            label,
            blocks,
            predictions,
            development_masks,
            validation_masks,
        )
        test = evaluate_fixed_sky_calibration(
            label,
            blocks,
            predictions,
            development_masks,
            test_masks,
        )
        evaluations.append(
            {
                "label": label,
                **candidate_metadata[label],
                "selection": _evaluation_payload(selection),
                "sealed_test": _evaluation_payload(test),
            }
        )
        table_rows.append(
            {
                "label": label,
                **candidate_metadata[label],
                "development_power": selection.train["normalized_residual_power"],
                "validation_power": selection.holdout["normalized_residual_power"],
                "sealed_test_power": test.holdout["normalized_residual_power"],
                "validation_rms": selection.holdout["normalized_rms"],
                "sealed_test_rms": test.holdout["normalized_rms"],
                "common_development_samples": selection.train["active_count"],
                "common_validation_samples": selection.holdout["active_count"],
                "common_test_samples": test.holdout["active_count"],
            }
        )
    table_rows.sort(key=lambda row: float(row["validation_power"]))
    selected_label = str(table_rows[0]["label"])
    reference_audit = audit_visibility_residuals(
        reference_blocks,
        predictions,
        development_masks,
        test_masks,
    )
    selected_audit = audit_visibility_residuals(
        candidates[selected_label],
        predictions,
        development_masks,
        test_masks,
    )
    selected_on_reference_scale = audit_visibility_residuals(
        candidates[selected_label],
        predictions,
        development_masks,
        test_masks,
        fixed_scales=reference_audit.scales,
    )
    summary = {
        "schema_version": 1,
        "measurement_set": arguments.measurement_set.name,
        "reference_fixture": str(arguments.reference_fixture),
        "composite_checkpoint": str(arguments.composite_checkpoint),
        "sky_model": "frozen central hierarchy plus coarse field and catalogue atoms",
        "selection_metric": "fold-3 fixed-sky normalized residual power",
        "sealed_metric": "fold-4 fixed-sky normalized residual power",
        "time_split": {
            "bin_seconds": arguments.time_fold_seconds,
            "development_folds": [0, 1, 2],
            "validation_fold": 3,
            "sealed_test_fold": 4,
        },
        "normalization": "weighted frozen-model power",
        "target_flag_source": "frozen post-application flags for matched averaging geometry",
        "calibrator_metrics": calibrator_metrics,
        "native_gain_knot_count": int(solution.gain_time_s.size),
        "native_gain_times_s": solution.gain_time_s.tolist(),
        "casa_corrected_selection": _evaluation_payload(reference_selection),
        "casa_corrected_sealed_test": _evaluation_payload(reference_test),
        "candidates": evaluations,
        "ranking": [row["label"] for row in table_rows],
        "selected_candidate": selected_label,
        "selected_residual_tail_audit": {
            "casa_corrected": asdict(reference_audit),
            "candidate": asdict(selected_audit),
            "candidate_on_casa_scale": asdict(selected_on_reference_scale),
        },
    }
    arguments.output.mkdir(parents=True, exist_ok=True)
    selected_solution = next(
        candidate_solution
        for label, candidate_solution, _, _ in candidate_specs
        if label == selected_label
    )
    write_calibration(selected_solution, arguments.output / "selected_calibration.npz")
    write_dataset(
        VisibilityDataset(
            candidates[selected_label],
            provenance={
                "experiment": "3C391 composite-sky calibration complexity sweep",
                "selected_candidate": selected_label,
                "selection_fold": 3,
                "sealed_test_fold": 4,
                "measurement_set": arguments.measurement_set.name,
            },
        ),
        arguments.output / "selected_fixture.zarr",
    )
    (arguments.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_table(arguments.output / "ranking.csv", table_rows)
    print(json.dumps({"ranking": summary["ranking"], "rows": table_rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
