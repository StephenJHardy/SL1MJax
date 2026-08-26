#!/usr/bin/env python3
"""Refit a validated joint-mosaic topology on every active visibility."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

from sl1mjax.beam import primary_beam_from_name
from sl1mjax.data.canonical import VisibilityBlock, read_dataset
from sl1mjax.direct_operator import DirectDFTConfig
from sl1mjax.inference import (
    InferenceConfig,
    MosaicQuadtreeInferenceResult,
    infer_mosaic_quadtree,
)
from sl1mjax.quadtree import (
    QuadtreeGrid,
    QuadtreeLeaf,
    QuadtreeSky,
    QuadtreeTopology,
    quadtree_sky_from_regular_grid,
)
from sl1mjax.refinement import render_quadtree_surface_brightness
from sl1mjax.sky import GaussianApproximation


def _load_sky(
    path: Path,
    *,
    root_size: int,
    root_pixel_size_rad: float,
) -> QuadtreeSky:
    """Load a topology and aligned warm-start flux from its CSV product."""

    with path.open(newline="", encoding="utf-8") as stream:
        rows = tuple(csv.DictReader(stream))
    if not rows:
        raise ValueError("topology CSV must contain at least one leaf")
    flux_by_leaf = {
        QuadtreeLeaf(int(row["level"]), int(row["iy"]), int(row["ix"])): float(row["flux_jy"])
        for row in rows
    }
    if len(flux_by_leaf) != len(rows):
        raise ValueError("topology CSV contains duplicate leaves")
    if any(not np.isfinite(value) or value < 0 for value in flux_by_leaf.values()):
        raise ValueError("topology flux must be finite and non-negative")
    grid = QuadtreeGrid(root_size, root_pixel_size_rad)
    topology = QuadtreeTopology(grid, tuple(flux_by_leaf))
    flux = np.asarray([flux_by_leaf[leaf] for leaf in topology.leaves])
    return QuadtreeSky(grid, topology.leaves, flux)


def _metrics(
    blocks: tuple[VisibilityBlock, ...],
    fit: MosaicQuadtreeInferenceResult,
) -> dict[str, Any]:
    residual_numerator = 0.0
    signal_numerator = 0.0
    weight_sum = 0.0
    per_pointing = []
    for index, (block, prediction) in enumerate(zip(blocks, fit.predictions, strict=True), start=1):
        weight = np.where(block.active, block.weight, 0.0)
        residual = np.where(block.active, prediction - block.visibility, 0.0)
        residual_power = float(np.sum(weight * np.abs(residual) ** 2))
        signal_power = float(np.sum(weight * np.abs(block.visibility) ** 2))
        active_weight = float(np.sum(weight))
        residual_numerator += residual_power
        signal_numerator += signal_power
        weight_sum += active_weight
        per_pointing.append(
            {
                "label": f"C{index}",
                "field_id": (None if block.field_id is None else int(block.field_id[0])),
                "active_samples": int(np.count_nonzero(block.active)),
                "weighted_complex_mse": residual_power / active_weight,
                "normalized_residual_power": residual_power / signal_power,
            }
        )
    return {
        "leaf_count": len(fit.topology.leaves),
        "total_flux_jy": float(np.sum(fit.flux)),
        "steps": fit.steps,
        "best_step": fit.best_step,
        "converged": fit.converged,
        "kkt_residual": fit.kkt_residual,
        "weighted_complex_mse": residual_numerator / weight_sum,
        "normalized_residual_power": residual_numerator / signal_numerator,
        "per_pointing": per_pointing,
    }


def _write_reconstruction(
    path: Path,
    fit: MosaicQuadtreeInferenceResult,
) -> None:
    level = max(leaf.level for leaf in fit.topology.leaves)
    pixel_size = fit.topology.grid.root_pixel_size_rad / 2**level
    brightness = render_quadtree_surface_brightness(
        fit.topology,
        fit.flux,
        level=level,
    )
    image = brightness * pixel_size**2
    header = fits.Header()
    header["BUNIT"] = "Jy/pixel"
    header["CTYPE1"] = "RA---SIN"
    header["CTYPE2"] = "DEC--SIN"
    header["CRPIX1"] = (image.shape[1] + 1) / 2
    header["CRPIX2"] = (image.shape[0] + 1) / 2
    header["CRVAL1"] = np.rad2deg(fit.mosaic_phase_centre_rad[0])
    header["CRVAL2"] = np.rad2deg(fit.mosaic_phase_centre_rad[1])
    header["CDELT1"] = -np.rad2deg(pixel_size)
    header["CDELT2"] = np.rad2deg(pixel_size)
    fits.PrimaryHDU(image, header=header).writeto(path, overwrite=True)


def _write_topology(path: Path, fit: MosaicQuadtreeInferenceResult) -> None:
    l, m = fit.topology.centers()
    widths = fit.topology.widths_rad()
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("level", "iy", "ix", "flux_jy", "l_rad", "m_rad", "width_rad"))
        for leaf, flux, leaf_l, leaf_m, width in zip(
            fit.topology.leaves,
            fit.flux,
            l,
            m,
            widths,
            strict=True,
        ):
            writer.writerow((leaf.level, leaf.iy, leaf.ix, flux, leaf_l, leaf_m, width))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "fixture",
        type=Path,
        nargs="?",
        default=Path("outputs/3c391_mosaic_imaging_fixture.zarr"),
    )
    parser.add_argument(
        "--validated-directory",
        type=Path,
        default=Path("outputs/3c391_mosaic_hierarchical_frozen_104"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/3c391_mosaic_hierarchical_consensus_all"),
    )
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lambda-l1", type=float, default=3e-4)
    parser.add_argument("--kkt-tolerance", type=float, default=3e-5)
    parser.add_argument("--visibility-tile-size", type=int, default=256)
    parser.add_argument("--pixel-tile-size", type=int, default=1024)
    parser.add_argument("--precision", choices=("float32", "float64"), default="float32")
    parser.add_argument("--maximum-pointings", type=int)
    arguments = parser.parse_args()

    validation_summary = json.loads(
        (arguments.validated_directory / "summary.json").read_text(encoding="utf-8")
    )
    root_size = int(validation_summary["root_size"])
    root_pixel_arcsec = float(validation_summary["root_pixel_arcsec"])
    root_pixel_size_rad = np.deg2rad(root_pixel_arcsec / 3600.0)
    hierarchy = _load_sky(
        arguments.validated_directory / "consensus_topology.csv",
        root_size=root_size,
        root_pixel_size_rad=root_pixel_size_rad,
    )
    root = quadtree_sky_from_regular_grid(
        root_size,
        root_pixel_size_rad,
        np.zeros(root_size**2),
    )
    blocks = read_dataset(arguments.fixture).blocks
    if arguments.maximum_pointings is not None:
        if not 1 <= arguments.maximum_pointings <= len(blocks):
            raise ValueError("maximum-pointings must be between one and the block count")
        blocks = blocks[: arguments.maximum_pointings]
    mosaic_phase_centre = blocks[0].phase_centre_rad
    masks = tuple(block.active for block in blocks)
    config = InferenceConfig(
        solver="fista",
        steps=arguments.steps,
        sparsity_weight=arguments.lambda_l1,
        kkt_tolerance=arguments.kkt_tolerance,
        validation_interval=25,
        operator_mode="explicit",
        direct_dft=DirectDFTConfig(
            visibility_chunk_size=arguments.visibility_tile_size,
            pixel_chunk_size=arguments.pixel_tile_size,
            precision=arguments.precision,
        ),
    )
    beam = primary_beam_from_name("airy")
    arguments.output.mkdir(parents=True, exist_ok=True)

    print(f"all-data baseline: {len(root.leaves)} leaves", flush=True)
    baseline = infer_mosaic_quadtree(
        blocks,
        root.topology,
        masks,
        mosaic_phase_centre,
        config,
        primary_beam=beam,
        approximation=GaussianApproximation.WIDE_FIELD,
        initial_flux=root.flux,
    )
    print(
        f"baseline complete: steps={baseline.steps}, KKT={baseline.kkt_residual:.3g}",
        flush=True,
    )
    print(f"all-data consensus: {len(hierarchy.leaves)} leaves", flush=True)
    consensus = infer_mosaic_quadtree(
        blocks,
        hierarchy.topology,
        masks,
        mosaic_phase_centre,
        config,
        primary_beam=beam,
        approximation=GaussianApproximation.WIDE_FIELD,
        initial_flux=hierarchy.flux,
    )
    print(
        f"consensus complete: steps={consensus.steps}, KKT={consensus.kkt_residual:.3g}",
        flush=True,
    )

    baseline_metrics = _metrics(blocks, baseline)
    consensus_metrics = _metrics(blocks, consensus)
    baseline_power = float(baseline_metrics["normalized_residual_power"])
    consensus_power = float(consensus_metrics["normalized_residual_power"])
    summary = {
        "schema_version": 1,
        "fixture": str(arguments.fixture),
        "validated_directory": str(arguments.validated_directory),
        "fit_scope": "all_active_visibilities",
        "pointing_count": len(blocks),
        "active_sample_count": int(sum(np.count_nonzero(mask) for mask in masks)),
        "mosaic_phase_centre_deg": [
            float(np.rad2deg(mosaic_phase_centre[0])),
            float(np.rad2deg(mosaic_phase_centre[1])),
        ],
        "root_size": root_size,
        "root_pixel_arcsec": root_pixel_arcsec,
        "field_of_view_arcmin": root_size * root_pixel_arcsec / 60.0,
        "lambda_l1": arguments.lambda_l1,
        "primary_beam": "airy",
        "approximation": "wide_field",
        "baseline": baseline_metrics,
        "consensus": consensus_metrics,
        "improvement_over_unrefined": {
            "absolute_normalized_residual_power": baseline_power - consensus_power,
            "relative": 1.0 - consensus_power / baseline_power,
        },
    }
    _write_reconstruction(arguments.output / "baseline_reconstruction.fits", baseline)
    _write_reconstruction(arguments.output / "consensus_reconstruction.fits", consensus)
    _write_topology(arguments.output / "consensus_topology.csv", consensus)
    np.savez(
        arguments.output / "predictions.npz",
        **{
            f"baseline_C{index + 1}": prediction
            for index, prediction in enumerate(baseline.predictions)
        },
        **{
            f"consensus_C{index + 1}": prediction
            for index, prediction in enumerate(consensus.predictions)
        },
    )
    (arguments.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
