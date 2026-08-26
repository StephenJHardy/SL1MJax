#!/usr/bin/env python3
"""Cross-validate low-complexity target self-calibration across time halves."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

from sl1mjax.calibration import apply_calibration, identity_solution
from sl1mjax.calibration_inference import CalibrationSolveConfig, solve_time_gains
from sl1mjax.data.canonical import VisibilityBlock, read_dataset
from sl1mjax.split import VisibilitySplit, calibration_split


def time_half_masks(block: VisibilityBlock) -> tuple[np.ndarray, np.ndarray, float]:
    active_rows = np.any(block.active, axis=(1, 2))
    times = np.unique(block.time_s[active_rows])
    if times.size < 2:
        raise ValueError("block needs at least two active integration times")
    boundary = float(np.median(times))
    first = block.active & (block.time_s <= boundary)[:, None, None]
    second = block.active & (block.time_s > boundary)[:, None, None]
    if not np.any(first) or not np.any(second):
        raise ValueError("time split produced an empty half")
    return first, second, boundary


def static_calibration_block(
    block: VisibilityBlock,
    prediction: np.ndarray,
    selected: np.ndarray,
) -> VisibilityBlock:
    """Collapse one half to a single gain interval without averaging samples."""

    if prediction.shape != block.shape or selected.shape != block.shape:
        raise ValueError("prediction and selected mask must match the visibility block")
    active = selected & block.active
    if not np.any(active):
        raise ValueError("selected calibration half contains no active samples")
    return replace(
        block,
        flag=~active,
        time_s=np.zeros(block.shape[0], dtype=np.float64),
        model_visibility=prediction,
        provenance={
            **dict(block.provenance),
            "selfcal_time_model": "one static gain per antenna for selected half",
        },
    )


def heldout_baseline_mask(
    block: VisibilityBlock,
    split: VisibilitySplit,
    selected: np.ndarray,
) -> np.ndarray:
    held_rows = np.any(split.holdout, axis=(1, 2))
    held_pairs = {
        tuple(sorted((int(block.antenna1[row]), int(block.antenna2[row]))))
        for row in np.flatnonzero(held_rows)
    }
    row_selected = np.asarray(
        [
            tuple(sorted((int(first), int(second)))) in held_pairs
            for first, second in zip(block.antenna1, block.antenna2, strict=True)
        ]
    )
    return selected & block.active & row_selected[:, None, None]


def _metric(
    block: VisibilityBlock,
    prediction: np.ndarray,
    selected: np.ndarray,
    *,
    visibility: np.ndarray | None = None,
) -> dict[str, float | int]:
    mask = selected & block.active
    observed = block.visibility if visibility is None else visibility
    if not np.any(mask):
        raise ValueError("metric mask contains no active samples")
    weights = block.weight[mask]
    residual_power = float(np.sum(weights * np.abs(observed[mask] - prediction[mask]) ** 2))
    model_power = float(np.sum(weights * np.abs(prediction[mask]) ** 2))
    weight_sum = float(np.sum(weights))
    return {
        "sample_count": int(np.count_nonzero(mask)),
        "weighted_complex_mse": residual_power / weight_sum,
        "normalized_residual_power": residual_power / model_power,
    }


def _comparison(
    block: VisibilityBlock,
    prediction: np.ndarray,
    selected: np.ndarray,
    corrected_visibility: np.ndarray,
) -> dict[str, Any]:
    base = _metric(block, prediction, selected)
    candidate = _metric(
        block,
        prediction,
        selected,
        visibility=corrected_visibility,
    )
    return {
        "base": base,
        "candidate": candidate,
        "relative_weighted_complex_mse_change": (
            candidate["weighted_complex_mse"] / base["weighted_complex_mse"] - 1.0
        ),
    }


def fit_direction(
    block: VisibilityBlock,
    prediction: np.ndarray,
    train_half: np.ndarray,
    evaluation_half: np.ndarray,
    *,
    phase_only: bool,
    config: CalibrationSolveConfig,
) -> dict[str, Any]:
    calibration_block = static_calibration_block(block, prediction, train_half)
    split = calibration_split(
        calibration_block,
        holdout_fraction=config.holdout_fraction,
        seed=config.seed,
    )
    active_antennas = np.unique(
        np.concatenate(
            (
                calibration_block.antenna1[np.any(calibration_block.active, axis=(1, 2))],
                calibration_block.antenna2[np.any(calibration_block.active, axis=(1, 2))],
            )
        )
    )
    reference_antenna = int(active_antennas[0])
    initial = identity_solution(
        antenna_count=block.antenna_count,
        correlations=block.correlations,
        frequency_hz=block.frequency_hz,
        time_s=(0.0,),
        reference_antenna=reference_antenna,
    )
    fit = solve_time_gains(
        calibration_block,
        initial,
        split=split,
        config=config,
        phase_only=phase_only,
    )
    corrected = apply_calibration(block, fit.solution, extrapolate=True)
    train_holdout = split.holdout & train_half & block.active
    evaluation_holdout = heldout_baseline_mask(block, split, evaluation_half)
    gains = fit.solution.gains[0, active_antennas, :]
    return {
        "phase_only": phase_only,
        "reference_antenna": reference_antenna,
        "active_antenna_count": int(active_antennas.size),
        "iterations": len(fit.losses),
        "final_objective": float(fit.losses[-1]),
        "gain_log_amplitude_rms": float(np.sqrt(np.mean(np.log(np.abs(gains)) ** 2))),
        "gain_phase_rms_rad": float(np.sqrt(np.mean(np.angle(gains) ** 2))),
        "training_heldout_baselines": _comparison(
            block,
            prediction,
            train_holdout,
            corrected.visibility,
        ),
        "cross_time_heldout_baselines": _comparison(
            block,
            prediction,
            evaluation_holdout,
            corrected.visibility,
        ),
        "cross_time_all_baselines": _comparison(
            block,
            prediction,
            evaluation_half,
            corrected.visibility,
        ),
    }


def _write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "fixture",
        type=Path,
        nargs="?",
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
        default=Path("outputs/3c391_time_half_selfcal"),
    )
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=19)
    arguments = parser.parse_args()

    blocks = read_dataset(arguments.fixture).blocks
    with np.load(arguments.frozen_directory / "predictions.npz") as stored:
        predictions = tuple(
            np.asarray(stored[f"consensus_C{index + 1}"])
            for index in range(len(blocks))
        )
    config = CalibrationSolveConfig(
        iterations=arguments.iterations,
        learning_rate=arguments.learning_rate,
        holdout_fraction=arguments.holdout_fraction,
        seed=arguments.seed,
    )
    pointings = []
    rows = []
    for index, (block, prediction) in enumerate(
        zip(blocks, predictions, strict=True), start=1
    ):
        first, second, boundary = time_half_masks(block)
        directions = {}
        for direction, train, evaluation in (
            ("first_to_second", first, second),
            ("second_to_first", second, first),
        ):
            modes = {}
            for mode, phase_only in (("phase_only", True), ("amplitude_phase", False)):
                print(f"C{index} {direction} {mode}", flush=True)
                result = fit_direction(
                    block,
                    prediction,
                    train,
                    evaluation,
                    phase_only=phase_only,
                    config=config,
                )
                modes[mode] = result
                rows.append(
                    {
                        "pointing": f"C{index}",
                        "direction": direction,
                        "mode": mode,
                        "gain_log_amplitude_rms": result["gain_log_amplitude_rms"],
                        "gain_phase_rms_rad": result["gain_phase_rms_rad"],
                        "train_holdout_mse_change": result[
                            "training_heldout_baselines"
                        ]["relative_weighted_complex_mse_change"],
                        "cross_time_holdout_mse_change": result[
                            "cross_time_heldout_baselines"
                        ]["relative_weighted_complex_mse_change"],
                        "cross_time_all_mse_change": result[
                            "cross_time_all_baselines"
                        ]["relative_weighted_complex_mse_change"],
                    }
                )
            directions[direction] = modes
        pointings.append(
            {
                "label": f"C{index}",
                "time_boundary_s": boundary,
                "directions": directions,
            }
        )
    summary = {
        "schema_version": 1,
        "fixture": str(arguments.fixture),
        "frozen_directory": str(arguments.frozen_directory),
        "calibration_model": (
            "one static residual complex gain per antenna, pointing, and time half"
        ),
        "baseline_split": "connected baseline groups; same held-out baselines used across time",
        "config": asdict(config),
        "pointings": pointings,
    }
    arguments.output.mkdir(parents=True, exist_ok=True)
    (arguments.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_table(arguments.output / "ranking.csv", rows)
    print(json.dumps(rows, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
