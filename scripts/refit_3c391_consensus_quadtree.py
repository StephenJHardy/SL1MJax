#!/usr/bin/env python3
"""Refit a cross-fold consensus quadtree on every active visibility."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

from sl1mjax.beam import primary_beam_from_name
from sl1mjax.data.canonical import VisibilityBlock, read_dataset
from sl1mjax.direct_operator import DirectDFTConfig
from sl1mjax.hierarchical_imaging import consensus_split_leaves
from sl1mjax.inference import (
    InferenceConfig,
    QuadtreeInferenceResult,
    infer_quadtree,
    save_checkpoint,
)
from sl1mjax.quadtree import QuadtreeLeaf, quadtree_sky_from_regular_grid
from sl1mjax.refinement import (
    quadtree_objective_metrics,
    render_quadtree_surface_brightness,
)
from sl1mjax.sky import GaussianApproximation


def _accepted_splits(summary: dict[str, Any]) -> tuple[QuadtreeLeaf, ...]:
    accepted: list[QuadtreeLeaf] = []
    for round_record in summary["rounds"]:
        accepted_attempts = [attempt for attempt in round_record["attempts"] if attempt["accepted"]]
        if not accepted_attempts:
            continue
        split_count = int(accepted_attempts[0]["split_count"])
        selected = round_record["selected"][:split_count]
        accepted.extend(QuadtreeLeaf(*map(int, leaf)) for leaf in selected)
    if len(set(accepted)) != len(accepted):
        raise ValueError("a fold summary contains a duplicate accepted split")
    return tuple(accepted)


def _load_fold_splits(paths: tuple[Path, ...]) -> tuple[tuple[QuadtreeLeaf, ...], ...]:
    summaries = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    geometries = {
        (
            int(summary["configuration"]["root_size"]),
            float(summary["configuration"]["root_pixel_arcsec"]),
        )
        for summary in summaries
    }
    if len(geometries) != 1:
        raise ValueError("fold summaries must use the same root geometry")
    return tuple(_accepted_splits(summary) for summary in summaries)


def _write_products(
    output: Path,
    block: VisibilityBlock,
    fit: QuadtreeInferenceResult,
    config: InferenceConfig,
    fold_paths: tuple[Path, ...],
    fold_splits: tuple[tuple[QuadtreeLeaf, ...], ...],
    consensus: tuple[QuadtreeLeaf, ...],
    minimum_support: int,
    root_pixel_size_rad: float,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    deepest_level = max(leaf.level for leaf in fit.topology.leaves)
    surface_brightness = render_quadtree_surface_brightness(
        fit.topology,
        fit.flux,
        level=deepest_level,
    )
    render_pixel_size = root_pixel_size_rad / 2**deepest_level
    image = surface_brightness * render_pixel_size**2
    header = fits.Header()
    header["BUNIT"] = "Jy/pixel"
    header["CTYPE1"] = "RA---SIN"
    header["CTYPE2"] = "DEC--SIN"
    header["CRPIX1"] = (image.shape[1] + 1) / 2
    header["CRPIX2"] = (image.shape[0] + 1) / 2
    header["CRVAL1"] = np.rad2deg(block.phase_centre_rad[0])
    header["CRVAL2"] = np.rad2deg(block.phase_centre_rad[1])
    header["CDELT1"] = -np.rad2deg(render_pixel_size)
    header["CDELT2"] = np.rad2deg(render_pixel_size)
    fits.PrimaryHDU(image, header=header).writeto(
        output / "consensus_reconstruction.fits",
        overwrite=True,
    )

    l, m = fit.topology.centers()
    widths = fit.topology.widths_rad()
    with (output / "consensus_topology.csv").open("w", newline="", encoding="utf-8") as stream:
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

    support = Counter(leaf for splits in fold_splits for leaf in set(splits))
    with (output / "consensus_splits.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("level", "iy", "ix", "fold_support"))
        for leaf in consensus:
            writer.writerow((leaf.level, leaf.iy, leaf.ix, support[leaf]))

    np.savez(
        output / "consensus_residuals.npz",
        prediction=fit.prediction,
        residual=fit.residual,
        active_mask=block.active,
    )
    save_checkpoint(output / "consensus_reconstruction.checkpoint.npz", fit)
    metrics = quadtree_objective_metrics(block, fit, block.active, config)
    summary = {
        "fold_summaries": [str(path) for path in fold_paths],
        "fold_count": len(fold_splits),
        "minimum_support": minimum_support,
        "accepted_splits_per_fold": [len(splits) for splits in fold_splits],
        "consensus_split_count": len(consensus),
        "consensus_split_count_by_level": {
            str(level): sum(leaf.level == level for leaf in consensus)
            for level in sorted({leaf.level for leaf in consensus})
        },
        "leaf_count": len(fit.topology.leaves),
        "leaf_count_by_level": {
            str(level): sum(leaf.level == level for leaf in fit.topology.leaves)
            for level in sorted({leaf.level for leaf in fit.topology.leaves})
        },
        "deepest_level": deepest_level,
        "render_shape": list(image.shape),
        "total_flux_jy": float(np.sum(fit.flux)),
        "peak_render_pixel_jy": float(np.max(image)),
        "training_objective": metrics.objective,
        "training_data_loss": metrics.training_data,
        "sparsity": metrics.sparsity,
        "optimizer_solver": fit.solver,
        "optimizer_steps": fit.steps,
        "optimizer_best_step": fit.best_step,
        "optimizer_converged": fit.converged,
        "optimizer_kkt_residual": fit.kkt_residual,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("fixture", type=Path)
    parser.add_argument("fold_summaries", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block", type=int, default=0)
    parser.add_argument("--minimum-support", type=int)
    parser.add_argument("--root-size", type=int, default=64)
    parser.add_argument("--root-pixel-arcsec", type=float, default=16.0)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--sparsity-weight", type=float, default=3e-4)
    parser.add_argument("--kkt-tolerance", type=float, default=3e-5)
    parser.add_argument("--primary-beam", choices=("none", "gaussian", "airy"), default="airy")
    parser.add_argument(
        "--approximation",
        choices=("paraxial", "wide-field"),
        default="wide-field",
    )
    parser.add_argument("--precision", choices=("float32", "float64"), default="float32")
    parser.add_argument("--visibility-tile-size", type=int, default=256)
    parser.add_argument("--pixel-tile-size", type=int, default=1024)
    arguments = parser.parse_args()

    fold_paths = tuple(arguments.fold_summaries)
    fold_splits = _load_fold_splits(fold_paths)
    required = (
        len(fold_splits) // 2 + 1
        if arguments.minimum_support is None
        else arguments.minimum_support
    )
    consensus = consensus_split_leaves(fold_splits, minimum_support=required)
    root_pixel_size_rad = np.deg2rad(arguments.root_pixel_arcsec / 3600.0)
    sky = quadtree_sky_from_regular_grid(
        arguments.root_size,
        root_pixel_size_rad,
        np.zeros(arguments.root_size**2),
    )
    for leaf in consensus:
        if leaf not in sky.topology.leaves:
            raise ValueError(f"consensus split {leaf} is not active after its ancestors")
        sky = sky.split(leaf)

    dataset = read_dataset(arguments.fixture)
    if not 0 <= arguments.block < len(dataset.blocks):
        raise ValueError(f"block must be between 0 and {len(dataset.blocks) - 1}")
    block = dataset.blocks[arguments.block]
    approximation = (
        GaussianApproximation.PARAXIAL
        if arguments.approximation == "paraxial"
        else GaussianApproximation.WIDE_FIELD
    )
    primary_beam = primary_beam_from_name(arguments.primary_beam)
    config = InferenceConfig(
        solver="fista",
        steps=arguments.steps,
        sparsity_weight=arguments.sparsity_weight,
        kkt_tolerance=arguments.kkt_tolerance,
        validation_interval=25,
        operator_mode="explicit",
        direct_dft=DirectDFTConfig(
            visibility_chunk_size=arguments.visibility_tile_size,
            pixel_chunk_size=arguments.pixel_tile_size,
            precision=arguments.precision,
        ),
    )
    print(
        f"consensus topology: {len(consensus)} splits, {len(sky.leaves)} leaves; "
        f"fitting {block.shape[0]} rows",
        flush=True,
    )
    fit = infer_quadtree(
        block,
        sky.topology,
        block.active,
        config,
        primary_beam=primary_beam,
        approximation=approximation,
        initial_flux=sky.flux,
    )
    print(
        f"fit complete: steps={fit.steps}, KKT={fit.kkt_residual:.3g}, "
        f"objective={fit.objective_history[-1]:.8g}",
        flush=True,
    )
    _write_products(
        arguments.output,
        block,
        fit,
        config,
        fold_paths,
        fold_splits,
        consensus,
        required,
        root_pixel_size_rad,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
