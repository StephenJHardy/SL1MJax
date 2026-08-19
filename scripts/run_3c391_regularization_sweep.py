"""Run a resumable 3C391 regularization and validation sweep."""

from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Case:
    pixel_model: str
    split_strategy: str
    sparsity_weight: float
    smoothness_weight: float

    @property
    def name(self) -> str:
        return (
            f"{self.split_strategy}/{self.pixel_model}/"
            f"sparsity_{self.sparsity_weight:g}_"
            f"smoothness_{self.smoothness_weight:g}"
        )


def _strings(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _floats(value: str) -> tuple[float, ...]:
    return tuple(float(item) for item in _strings(value))


def _row(case: Case, summary: dict[str, Any], status: str) -> dict[str, Any]:
    casa = summary.get("casa_corrected", {})
    calibrated = summary.get("jax_calibrated", {})
    comparison = summary.get("image_comparison", {})
    return {
        "status": status,
        "pixel_model": case.pixel_model,
        "split_strategy": case.split_strategy,
        "sparsity_weight": case.sparsity_weight,
        "smoothness_weight": case.smoothness_weight,
        "casa_best_step": casa.get("best_step"),
        "casa_train_loss": casa.get("train_loss"),
        "casa_holdout_loss": casa.get("holdout_loss"),
        "casa_train_normalized_loss": casa.get("train_normalized_loss"),
        "casa_holdout_normalized_loss": casa.get(
            "holdout_normalized_loss"
        ),
        "casa_total_flux": casa.get("total_flux"),
        "jax_best_step": calibrated.get("best_step"),
        "jax_train_loss": calibrated.get("train_loss"),
        "jax_holdout_loss": calibrated.get("holdout_loss"),
        "jax_train_normalized_loss": calibrated.get(
            "train_normalized_loss"
        ),
        "jax_holdout_normalized_loss": calibrated.get(
            "holdout_normalized_loss"
        ),
        "jax_total_flux": calibrated.get("total_flux"),
        "calibration_image_correlation": comparison.get("correlation"),
        "calibration_image_normalized_rms": comparison.get(
            "normalized_rms"
        ),
    }


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("fixture", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/3c391_regularization_sweep"),
    )
    parser.add_argument(
        "--models",
        type=_strings,
        default=("delta", "gaussian-wide-field"),
    )
    parser.add_argument(
        "--split-strategies",
        type=_strings,
        default=("uv_cell", "random_row"),
    )
    parser.add_argument(
        "--sparsity-weights",
        type=_floats,
        default=(0.0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3),
    )
    parser.add_argument(
        "--smoothness-weights",
        type=_floats,
        default=(0.0, 1e3, 1e4, 1e5),
    )
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--validation-interval", type=int, default=20)
    parser.add_argument("--visibility-tile-size", type=int, default=4096)
    parser.add_argument("--pixel-tile-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260819)
    arguments = parser.parse_args()

    cases = [
        Case(model, split, sparsity, smoothness)
        for model in arguments.models
        for split in arguments.split_strategies
        for sparsity in arguments.sparsity_weights
        for smoothness in arguments.smoothness_weights
    ]
    random.Random(arguments.seed).shuffle(cases)
    arguments.output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    failures = 0
    for index, case in enumerate(cases, start=1):
        output = arguments.output / case.name
        summary_path = output / "summary.json"
        cached_summary = (
            json.loads(summary_path.read_text(encoding="utf-8"))
            if summary_path.exists()
            else None
        )
        cached_model = (
            cached_summary.get("configuration", {}).get("pixel_model")
            if cached_summary is not None
            else None
        )
        status = "cached" if cached_model == case.pixel_model else "stale"
        if status != "cached":
            output.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable,
                str(
                    Path(__file__).with_name(
                        "image_3c391_portable.py"
                    )
                ),
                str(arguments.fixture),
                "--output",
                str(output),
                "--size",
                str(arguments.size),
                "--steps",
                str(arguments.steps),
                "--learning-rate",
                str(arguments.learning_rate),
                "--pixel-model",
                case.pixel_model,
                "--sparsity-weight",
                str(case.sparsity_weight),
                "--smoothness-weight",
                str(case.smoothness_weight),
                "--split-strategy",
                case.split_strategy,
                "--validation-interval",
                str(arguments.validation_interval),
                "--precision",
                "float32",
                "--visibility-tile-size",
                str(arguments.visibility_tile_size),
                "--pixel-tile-size",
                str(arguments.pixel_tile_size),
            ]
            print(f"[{index}/{len(cases)}] {case.name}", flush=True)
            with (output / "run.log").open(
                "w", encoding="utf-8"
            ) as stream:
                completed = subprocess.run(
                    command,
                    check=False,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                )
            status = "ok" if completed.returncode == 0 else "error"
            if status == "error":
                failures += 1
        summary = (
            json.loads(summary_path.read_text(encoding="utf-8"))
            if summary_path.exists()
            else {}
        )
        rows.append(_row(case, summary, status))
        _write_rows(arguments.output / "results.csv", rows)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
