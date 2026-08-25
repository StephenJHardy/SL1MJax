#!/usr/bin/env python3
"""Summarize repeated-fold FISTA regularization and stopping-time sweeps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def _run_directories(root: Path) -> list[Path]:
    return sorted(
        directory
        for stage in ("coarse", "refined")
        for directory in (root / stage).glob("lambda_*/seed_*")
        if (directory / "fit.npz").is_file()
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root",
        type=Path,
        nargs="?",
        default=Path("outputs/3c391_airy_fista_selection"),
    )
    arguments = parser.parse_args()
    grouped: dict[float, list[Path]] = {}
    for directory in _run_directories(arguments.root):
        sparsity_weight = float(directory.parent.name.removeprefix("lambda_"))
        grouped.setdefault(sparsity_weight, []).append(directory)
    if not grouped:
        raise ValueError(f"no sweep outputs found under {arguments.root}")

    results: dict[str, Any] = {}
    curves: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    best: tuple[float, float, int] | None = None
    for sparsity_weight, directories in sorted(grouped.items()):
        holdouts = []
        stationarity = []
        common_steps: np.ndarray | None = None
        for directory in directories:
            with np.load(directory / "fit.npz") as fit:
                steps = np.asarray(fit["holdout_steps"], dtype=np.int64)
                if common_steps is None:
                    common_steps = steps
                elif not np.array_equal(steps, common_steps):
                    raise ValueError(f"holdout steps differ in {directory}")
                holdouts.append(np.asarray(fit["holdout_history"], dtype=np.float64))
                stationarity.append(np.asarray(fit["stationarity_history"], dtype=np.float64))
        assert common_steps is not None
        holdout_array = np.stack(holdouts)
        stationarity_array = np.stack(stationarity)
        mean = np.mean(holdout_array, axis=0)
        sample_std = np.std(holdout_array, axis=0, ddof=1)
        best_index = int(np.argmin(mean))
        selected_step = int(common_steps[best_index])
        selected_mean = float(mean[best_index])
        key = f"{sparsity_weight:.12g}"
        results[key] = {
            "fold_count": len(directories),
            "folds": [directory.name for directory in directories],
            "selected_step": selected_step,
            "selected_mean_holdout": selected_mean,
            "selected_sample_std_holdout": float(sample_std[best_index]),
            "selected_mean_kkt_residual": float(np.mean(stationarity_array[:, best_index])),
            "final_step": int(common_steps[-1]),
            "final_mean_holdout": float(mean[-1]),
            "final_sample_std_holdout": float(sample_std[-1]),
            "final_mean_kkt_residual": float(np.mean(stationarity_array[:, -1])),
            "holdout_steps": common_steps.tolist(),
            "mean_holdout": mean.tolist(),
            "sample_std_holdout": sample_std.tolist(),
            "mean_kkt_residual": np.mean(stationarity_array, axis=0).tolist(),
        }
        curves[sparsity_weight] = (common_steps, mean, sample_std)
        candidate = (selected_mean, -sparsity_weight, selected_step)
        if best is None or candidate < best:
            best = candidate

    assert best is not None
    selected_mean, negative_weight, selected_step = best
    selected_weight = -negative_weight
    summary = {
        "selection": {
            "sparsity_weight": selected_weight,
            "fista_steps": selected_step,
            "mean_holdout": selected_mean,
            "rule": (
                "minimum mean holdout across matched UV-cell folds; prefer larger lambda on ties"
            ),
        },
        "results": results,
    }
    summary_path = arguments.root / "selection_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    figure, (loss_axis, kkt_axis) = plt.subplots(
        1,
        2,
        figsize=(13, 5),
        constrained_layout=True,
    )
    color_map = plt.get_cmap("viridis")
    weights = sorted(curves)
    for index, sparsity_weight in enumerate(weights):
        steps, mean, sample_std = curves[sparsity_weight]
        color = color_map(index / max(len(weights) - 1, 1))
        label = f"λ={sparsity_weight:g}"
        loss_axis.plot(steps, mean, color=color, label=label)
        loss_axis.fill_between(
            steps,
            mean - sample_std,
            mean + sample_std,
            color=color,
            alpha=0.10,
        )
        kkt_axis.plot(
            steps,
            results[f"{sparsity_weight:.12g}"]["mean_kkt_residual"],
            color=color,
            label=label,
        )
    loss_axis.set_xlabel("FISTA step")
    loss_axis.set_ylabel("Mean held-out weighted complex MSE")
    loss_axis.set_title("Prediction across three UV-cell folds")
    loss_axis.legend(fontsize="small")
    kkt_axis.set_xlabel("FISTA step")
    kkt_axis.set_ylabel("Mean positive-L1 KKT residual")
    kkt_axis.set_yscale("log")
    kkt_axis.set_title("Fixed-topology stationarity")
    kkt_axis.legend(fontsize="small")
    figure_path = arguments.root / "selection_curves.png"
    figure.savefig(figure_path, dpi=170)
    plt.close(figure)
    print(json.dumps({"summary": str(summary_path), "figure": str(figure_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
