#!/usr/bin/env python3
"""Summarize and plot the 3C391 solver/primary-beam comparison matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

SOLVERS = ("softplus_adam", "fista", "proximal_sgd", "hybrid")
SOLVER_LABELS = {
    "softplus_adam": "Softplus Adam",
    "fista": "FISTA",
    "proximal_sgd": "Proximal SGD",
    "hybrid": "SGD → FISTA",
}


def _load_image(path: Path) -> np.ndarray:
    return np.asarray(fits.getdata(path), dtype=np.float64)


def _edge_fraction(image: np.ndarray) -> float:
    edge_flux = float(
        np.sum(image[0, :])
        + np.sum(image[-1, :])
        + np.sum(image[1:-1, 0])
        + np.sum(image[1:-1, -1])
    )
    return edge_flux / float(np.sum(image))


def _max_corner(image: np.ndarray) -> float:
    return float(max(image[0, 0], image[0, -1], image[-1, 0], image[-1, -1]))


def _normalized_rms(actual: np.ndarray, reference: np.ndarray) -> float:
    denominator = max(float(np.sum(np.square(reference))), np.finfo(float).tiny)
    return float(np.sqrt(np.sum(np.square(actual - reference)) / denominator))


def _correlation(actual: np.ndarray, reference: np.ndarray) -> float:
    return float(np.corrcoef(actual.ravel(), reference.ravel())[0, 1])


def _load_run(root: Path, beam: str, solver: str) -> dict[str, Any]:
    directory = root / beam / solver
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    intrinsic = _load_image(directory / "intrinsic_sky.fits")
    apparent = _load_image(directory / "apparent_sky_reference_frequency.fits")
    return {"summary": summary, "intrinsic": intrinsic, "apparent": apparent}


def _plot(root: Path, runs: dict[tuple[str, str], dict[str, Any]]) -> Path:
    displayed = [
        runs[(beam, solver)][kind]
        for solver in SOLVERS
        for beam, kind in (("none", "intrinsic"), ("airy", "intrinsic"), ("airy", "apparent"))
    ]
    image_limit = float(np.percentile(np.concatenate([image.ravel() for image in displayed]), 99.9))
    differences = [
        runs[("airy", solver)]["intrinsic"] - runs[("none", solver)]["intrinsic"]
        for solver in SOLVERS
    ]
    flattened_differences = np.concatenate([image.ravel() for image in differences])
    difference_limit = max(
        float(np.percentile(np.abs(flattened_differences), 99.5)),
        np.finfo(float).tiny,
    )
    figure, axes = plt.subplots(4, 4, figsize=(15, 14), constrained_layout=True)
    columns = (
        "No beam: intrinsic I",
        "Airy beam: intrinsic I",
        "Airy beam: apparent BI",
        "Intrinsic difference: Airy − none",
    )
    for column, title in enumerate(columns):
        axes[0, column].set_title(title)
    positive_artist = None
    difference_artist = None
    for row, solver in enumerate(SOLVERS):
        images = (
            runs[("none", solver)]["intrinsic"],
            runs[("airy", solver)]["intrinsic"],
            runs[("airy", solver)]["apparent"],
            differences[row],
        )
        axes[row, 0].set_ylabel(SOLVER_LABELS[solver])
        for column, image in enumerate(images):
            axis = axes[row, column]
            if column < 3:
                positive_artist = axis.imshow(
                    image,
                    origin="lower",
                    cmap="inferno",
                    vmin=0.0,
                    vmax=image_limit,
                )
            else:
                difference_artist = axis.imshow(
                    image,
                    origin="lower",
                    cmap="coolwarm",
                    vmin=-difference_limit,
                    vmax=difference_limit,
                )
            axis.set_xticks([])
            axis.set_yticks([])
    assert positive_artist is not None
    assert difference_artist is not None
    figure.colorbar(positive_artist, ax=axes[:, :3], shrink=0.75, label="Jy/pixel")
    figure.colorbar(difference_artist, ax=axes[:, 3], shrink=0.75, label="Jy/pixel")
    reference_hz = runs[("airy", "fista")]["summary"]["configuration"][
        "apparent_reference_frequency_hz"
    ]
    figure.suptitle(f"3C391 C1 solver and VLA Airy-beam comparison ({reference_hz / 1e9:.3f} GHz)")
    output = root / "comparison.png"
    figure.savefig(output, dpi=170)
    plt.close(figure)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root",
        type=Path,
        nargs="?",
        default=Path("outputs/3c391_solver_beam_matrix"),
    )
    arguments = parser.parse_args()
    runs = {
        (beam, solver): _load_run(arguments.root, beam, solver)
        for beam in ("none", "airy")
        for solver in SOLVERS
    }
    summary: dict[str, Any] = {"runs": {}, "comparisons": {}}
    for beam in ("none", "airy"):
        fista = runs[(beam, "fista")]["intrinsic"]
        for solver in SOLVERS:
            run = runs[(beam, solver)]
            fit = run["summary"]["fit"]
            key = f"{beam}/{solver}"
            summary["runs"][key] = {
                "elapsed_s": fit["elapsed_s"],
                "training_objective": fit["training_objective"],
                "holdout_data": fit["holdout_data"],
                "kkt_residual": fit["solver_kkt_residual"],
                "intrinsic_total_flux_jy": fit["total_flux"],
                "apparent_total_flux_jy": fit["apparent_total_flux"],
                "intrinsic_peak_jy": fit["peak_flux"],
                "apparent_peak_jy": fit["apparent_peak_flux"],
                "peak_iy_ix": fit["peak_iy_ix"],
                "intrinsic_edge_fraction": _edge_fraction(run["intrinsic"]),
                "apparent_edge_fraction": _edge_fraction(run["apparent"]),
                "intrinsic_max_corner_jy": _max_corner(run["intrinsic"]),
                "apparent_max_corner_jy": _max_corner(run["apparent"]),
                "intrinsic_correlation_to_fista": _correlation(run["intrinsic"], fista),
                "intrinsic_normalized_rms_to_fista": _normalized_rms(run["intrinsic"], fista),
            }
    for solver in SOLVERS:
        no_beam = runs[("none", solver)]["intrinsic"]
        airy = runs[("airy", solver)]["intrinsic"]
        summary["comparisons"][solver] = {
            "airy_intrinsic_correlation_to_no_beam": _correlation(airy, no_beam),
            "airy_intrinsic_normalized_rms_to_no_beam": _normalized_rms(airy, no_beam),
            "airy_intrinsic_flux_ratio": float(np.sum(airy) / np.sum(no_beam)),
        }
    summary_path = arguments.root / "comparison_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    image_path = _plot(arguments.root, runs)
    print(json.dumps({"summary": str(summary_path), "figure": str(image_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
