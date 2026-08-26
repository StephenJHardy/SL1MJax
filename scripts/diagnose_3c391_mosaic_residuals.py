#!/usr/bin/env python3
"""Form common-grid residual images for all-data 3C391 mosaic fits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

from sl1mjax.beam import primary_beam_from_name
from sl1mjax.data.canonical import read_dataset
from sl1mjax.diagnostics import MosaicResidualEvaluation, evaluate_mosaic_residuals
from sl1mjax.direct_operator import DirectDFTConfig
from sl1mjax.sky import RegularGrid


def _header(
    shape: tuple[int, int],
    pixel_size_rad: float,
    phase_centre_rad: tuple[float, float],
    *,
    unit: str,
) -> fits.Header:
    header = fits.Header()
    header["BUNIT"] = unit
    header["CTYPE1"] = "RA---SIN"
    header["CTYPE2"] = "DEC--SIN"
    header["CRPIX1"] = (shape[1] + 1) / 2
    header["CRPIX2"] = (shape[0] + 1) / 2
    header["CRVAL1"] = np.rad2deg(phase_centre_rad[0])
    header["CRVAL2"] = np.rad2deg(phase_centre_rad[1])
    header["CDELT1"] = -np.rad2deg(pixel_size_rad)
    header["CDELT2"] = np.rad2deg(pixel_size_rad)
    return header


def _write_evaluation(
    directory: Path,
    name: str,
    evaluation: MosaicResidualEvaluation,
    header: fits.Header,
) -> None:
    for suffix, image, unit in (
        ("natural", evaluation.natural_dirty, "Jy/beam"),
        ("sensitivity_corrected", evaluation.sensitivity_corrected_dirty, "Jy/beam"),
        ("sensitivity_fraction", evaluation.sensitivity_fraction, "1"),
    ):
        selected_header = header.copy()
        selected_header["BUNIT"] = unit
        fits.PrimaryHDU(image, selected_header).writeto(
            directory / f"{name}_residual_{suffix}.fits",
            overwrite=True,
        )


def _finite_percentile(image: np.ndarray, percentile: float) -> float:
    values = np.abs(image[np.isfinite(image)])
    if values.size == 0:
        raise ValueError("image contains no finite values")
    return float(np.percentile(values, percentile))


def _plot(
    baseline: MosaicResidualEvaluation,
    consensus: MosaicResidualEvaluation,
    path: Path,
) -> None:
    first = baseline.sensitivity_corrected_dirty
    second = consensus.sensitivity_corrected_dirty
    valid = np.isfinite(first) & np.isfinite(second)
    difference = np.where(valid, second - first, np.nan)
    limit = _finite_percentile(np.concatenate((first, second), axis=0), 99.5)
    difference_limit = _finite_percentile(difference, 99.5)
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for axis, image, title in zip(
        axes[:2],
        (first, second),
        ("16-arcsec base residual", "Consensus residual"),
        strict=True,
    ):
        shown = axis.imshow(
            image,
            origin="lower",
            cmap="coolwarm",
            vmin=-limit,
            vmax=limit,
        )
        finite = image[np.isfinite(image)]
        rms = float(np.sqrt(np.mean(np.square(finite))))
        axis.set_title(f"{title}\nimage RMS {rms:.4g} Jy/beam")
        figure.colorbar(shown, ax=axis, fraction=0.046, label="Jy/beam")
    shown = axes[2].imshow(
        difference,
        origin="lower",
        cmap="coolwarm",
        vmin=-difference_limit,
        vmax=difference_limit,
    )
    axes[2].set_title("Consensus − base residual")
    figure.colorbar(shown, ax=axes[2], fraction=0.046, label="Jy/beam")
    figure.suptitle("3C391 all-data joint-mosaic residual adjoints")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "fixture",
        type=Path,
        nargs="?",
        default=Path("outputs/3c391_mosaic_imaging_fixture.zarr"),
    )
    parser.add_argument(
        "--fit-directory",
        type=Path,
        default=Path("outputs/3c391_mosaic_hierarchical_consensus_all"),
    )
    parser.add_argument("--grid-size", type=int, default=208)
    parser.add_argument("--pixel-arcsec", type=float, default=8.0)
    parser.add_argument("--minimum-sensitivity-fraction", type=float, default=0.1)
    parser.add_argument("--visibility-tile-size", type=int, default=256)
    parser.add_argument("--pixel-tile-size", type=int, default=1024)
    parser.add_argument("--precision", choices=("float32", "float64"), default="float32")
    arguments = parser.parse_args()

    blocks = read_dataset(arguments.fixture).blocks
    summary = json.loads((arguments.fit_directory / "summary.json").read_text(encoding="utf-8"))
    phase_centre = tuple(np.deg2rad(summary["mosaic_phase_centre_deg"]))
    grid = RegularGrid(
        arguments.grid_size,
        np.deg2rad(arguments.pixel_arcsec / 3600.0),
    )
    with np.load(arguments.fit_directory / "predictions.npz") as stored:
        baseline_predictions = tuple(
            stored[f"baseline_C{index + 1}"] for index in range(len(blocks))
        )
        consensus_predictions = tuple(
            stored[f"consensus_C{index + 1}"] for index in range(len(blocks))
        )
    config = DirectDFTConfig(
        visibility_chunk_size=arguments.visibility_tile_size,
        pixel_chunk_size=arguments.pixel_tile_size,
        precision=arguments.precision,
    )
    beam = primary_beam_from_name("airy")
    print("joint residual adjoint: baseline", flush=True)
    baseline = evaluate_mosaic_residuals(
        blocks,
        baseline_predictions,
        grid,
        phase_centre,
        primary_beam=beam,
        config=config,
        minimum_sensitivity_fraction=arguments.minimum_sensitivity_fraction,
    )
    print("joint residual adjoint: consensus", flush=True)
    consensus = evaluate_mosaic_residuals(
        blocks,
        consensus_predictions,
        grid,
        phase_centre,
        primary_beam=beam,
        config=config,
        minimum_sensitivity_fraction=arguments.minimum_sensitivity_fraction,
    )
    header = _header(
        baseline.natural_dirty.shape,
        grid.pixel_size_rad,
        phase_centre,
        unit="Jy/beam",
    )
    _write_evaluation(arguments.fit_directory, "baseline", baseline, header)
    _write_evaluation(arguments.fit_directory, "consensus", consensus, header)
    _plot(
        baseline,
        consensus,
        arguments.fit_directory / "mosaic_residual_comparison.jpg",
    )
    diagnostics = {
        "schema_version": 1,
        "grid_size": arguments.grid_size,
        "pixel_arcsec": arguments.pixel_arcsec,
        "minimum_sensitivity_fraction": arguments.minimum_sensitivity_fraction,
        "baseline": baseline.diagnostics,
        "consensus": consensus.diagnostics,
    }
    (arguments.fit_directory / "residual_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(diagnostics["consensus"]["images"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
