#!/usr/bin/env python3
"""Run a nested, frozen-protocol hierarchical evaluation on one visibility block.

The outer test set contains complete scans selected without inspecting visibility
values. All topology discovery happens on the remaining rows. Three inner
validation runs vote on a consensus topology, then leaf fluxes are fitted on the
outer training set and scored once on the sealed outer test set.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

from sl1mjax.beam import primary_beam_from_name
from sl1mjax.data.canonical import VisibilityBlock, read_dataset
from sl1mjax.diagnostics import evaluate_residuals
from sl1mjax.direct_operator import DirectDFTConfig
from sl1mjax.hierarchical_imaging import (
    AdaptiveRefinementConfig,
    HierarchicalImagingResult,
    consensus_split_leaves,
    reconstruct_hierarchical,
)
from sl1mjax.inference import InferenceConfig, QuadtreeInferenceResult, infer_quadtree
from sl1mjax.objective import normalized_weighted_complex_mse, weighted_complex_mse
from sl1mjax.quadtree import QuadtreeLeaf, quadtree_sky_from_regular_grid
from sl1mjax.refinement import render_quadtree_surface_brightness
from sl1mjax.sky import GaussianApproximation, RegularGrid

FROZEN_PROTOCOL_VERSION = 1
FROZEN_INNER_SEEDS = (17, 29, 43)


def select_outer_test_scans(
    block: VisibilityBlock,
    *,
    fraction: float,
    seed: int,
) -> tuple[int, ...]:
    """Select whole scans close to a target active-sample fraction.

    Selection uses only scan identifiers and active-sample counts. The shuffled
    scan order is fixed by ``seed``. A scan is added when doing so moves the
    selected count closer to the requested target.
    """

    if not 0 < fraction < 1:
        raise ValueError("fraction must be between zero and one")
    assert block.scan_id is not None
    scans = np.unique(block.scan_id)
    if scans.size < 2:
        raise ValueError("at least two scans are required for an outer split")
    counts = {
        int(scan): int(np.count_nonzero(block.active[block.scan_id == scan])) for scan in scans
    }
    target = fraction * np.count_nonzero(block.active)
    selected: list[int] = []
    selected_count = 0
    for scan_value in np.random.default_rng(seed).permutation(scans):
        scan = int(scan_value)
        candidate_count = selected_count + counts[scan]
        if abs(candidate_count - target) < abs(selected_count - target):
            selected.append(scan)
            selected_count = candidate_count
    if not selected or len(selected) == scans.size:
        raise ValueError("outer scan selection must leave non-empty train and test sets")
    return tuple(sorted(selected))


def outer_scan_masks(
    block: VisibilityBlock,
    test_scans: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Return disjoint active train and test masks for complete held-out scans."""

    if not test_scans:
        raise ValueError("test_scans must not be empty")
    assert block.scan_id is not None
    known = set(map(int, np.unique(block.scan_id)))
    unknown = set(test_scans) - known
    if unknown:
        raise ValueError(f"unknown test scans: {sorted(unknown)}")
    test_rows = np.isin(block.scan_id, test_scans)
    test = test_rows[:, None, None] & block.active
    train = ~test_rows[:, None, None] & block.active
    if not np.any(train) or not np.any(test):
        raise ValueError("outer train and test masks must both contain active samples")
    return train, test


def subset_rows(block: VisibilityBlock, rows: np.ndarray) -> VisibilityBlock:
    """Return a visibility block containing the selected complete rows."""

    selected = np.asarray(rows, dtype=np.int64)
    if selected.ndim != 1 or selected.size == 0:
        raise ValueError("rows must be a non-empty one-dimensional array")
    assert block.field_id is not None
    assert block.scan_id is not None
    assert block.state_id is not None
    assert block.observation_id is not None
    assert block.feed1 is not None
    assert block.feed2 is not None
    assert block.interval_s is not None
    return replace(
        block,
        uvw_m=block.uvw_m[selected],
        visibility=block.visibility[selected],
        weight=block.weight[selected],
        flag=block.flag[selected],
        time_s=block.time_s[selected],
        antenna1=block.antenna1[selected],
        antenna2=block.antenna2[selected],
        model_visibility=(
            None if block.model_visibility is None else block.model_visibility[selected]
        ),
        field_id=block.field_id[selected],
        scan_id=block.scan_id[selected],
        state_id=block.state_id[selected],
        observation_id=block.observation_id[selected],
        feed1=block.feed1[selected],
        feed2=block.feed2[selected],
        interval_s=block.interval_s[selected],
    )


def accepted_split_leaves(result: HierarchicalImagingResult) -> tuple[QuadtreeLeaf, ...]:
    """Return the split operations accepted during an adaptive reconstruction."""

    accepted: list[QuadtreeLeaf] = []
    for round_record in result.rounds:
        validation = round_record.validation
        attempt = None if validation is None else validation.accepted_attempt
        if attempt is not None:
            accepted.extend(attempt.selected)
    if len(accepted) != len(set(accepted)):
        raise ValueError("adaptive result contains a duplicate accepted split")
    return tuple(accepted)


def _frozen_config(
    *,
    seed: int,
    visibility_tile_size: int,
    pixel_tile_size: int,
    precision: str,
) -> AdaptiveRefinementConfig:
    approximation = GaussianApproximation.WIDE_FIELD
    return AdaptiveRefinementConfig(
        root_size=64,
        root_pixel_size_rad=np.deg2rad(16.0 / 3600.0),
        inference=InferenceConfig(
            solver="fista",
            steps=500,
            sparsity_weight=3e-4,
            kkt_tolerance=3e-5,
            validation_interval=25,
            operator_mode="explicit",
            direct_dft=DirectDFTConfig(
                visibility_chunk_size=visibility_tile_size,
                pixel_chunk_size=pixel_tile_size,
                precision=precision,  # type: ignore[arg-type]
            ),
        ),
        holdout_fraction=0.2,
        split_seed=seed,
        split_strategy="uv_cell",
        uv_cells_per_axis=8,
        max_rounds=8,
        max_depth=2,
        leaf_penalty=0.0,
        target_improvement_fraction=0.7,
        max_split_fraction=0.05,
        max_splits_per_round=256,
        score_candidate_batch_size=32,
        score_row_batch_size=1024,
        minimum_holdout_relative_improvement=0.001,
        max_refits_per_round=4,
        approximation=approximation,
        allow_approximate_curvature=True,
        enable_merging=False,
    )


def _leaf_payload(leaves: tuple[QuadtreeLeaf, ...]) -> list[list[int]]:
    return [[leaf.level, leaf.iy, leaf.ix] for leaf in leaves]


def _leaves_from_payload(payload: list[list[int]]) -> tuple[QuadtreeLeaf, ...]:
    return tuple(QuadtreeLeaf(*map(int, item)) for item in payload)


def _round_summary(result: HierarchicalImagingResult) -> list[dict[str, Any]]:
    records = []
    for round_record in result.rounds:
        validation = round_record.validation
        accepted = None if validation is None else validation.accepted_attempt
        records.append(
            {
                "index": round_record.index,
                "leaf_count_before": round_record.leaf_count_before,
                "selected_count": len(round_record.selection.selected),
                "accepted_split_count": 0 if accepted is None else len(accepted.selected),
                "accepted": accepted is not None,
                "baseline_holdout_loss": (
                    None if validation is None else validation.baseline.holdout_data
                ),
                "accepted_holdout_loss": (
                    None if accepted is None else accepted.metrics.holdout_data
                ),
                "accepted_training_objective": (
                    None if accepted is None else accepted.metrics.objective
                ),
            }
        )
    return records


def _write_fold(
    directory: Path,
    result: HierarchicalImagingResult,
    *,
    seed: int,
) -> tuple[QuadtreeLeaf, ...]:
    directory.mkdir(parents=True, exist_ok=True)
    splits = accepted_split_leaves(result)
    payload = {
        "schema_version": 1,
        "inner_seed": seed,
        "stop_reason": result.stop_reason,
        "elapsed_s": result.elapsed_s,
        "accepted_split_count": len(splits),
        "accepted_splits": _leaf_payload(splits),
        "final_leaf_count": len(result.inference.topology.leaves),
        "final_kkt_residual": result.inference.kkt_residual,
        "rounds": _round_summary(result),
    }
    (directory / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    np.savez(
        directory / "masks.npz",
        train=result.train_mask,
        holdout=result.holdout_mask,
    )
    return splits


def _load_fold(directory: Path, *, seed: int) -> tuple[QuadtreeLeaf, ...] | None:
    path = directory / "summary.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload["inner_seed"]) != seed:
        raise ValueError(f"existing fold {directory} has a different seed")
    return _leaves_from_payload(payload["accepted_splits"])


def _fit_metrics(
    block: VisibilityBlock,
    fit: QuadtreeInferenceResult,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
) -> dict[str, Any]:
    return {
        "leaf_count": len(fit.topology.leaves),
        "steps": fit.steps,
        "best_step": fit.best_step,
        "converged": fit.converged,
        "kkt_residual": fit.kkt_residual,
        "total_flux_jy": float(np.sum(fit.flux)),
        "train_weighted_complex_mse": float(
            weighted_complex_mse(fit.prediction, block.visibility, block.weight, ~train_mask)
        ),
        "test_weighted_complex_mse": float(
            weighted_complex_mse(fit.prediction, block.visibility, block.weight, ~test_mask)
        ),
        "train_normalized_residual_power": float(
            normalized_weighted_complex_mse(
                fit.prediction, block.visibility, block.weight, ~train_mask
            )
        ),
        "test_normalized_residual_power": float(
            normalized_weighted_complex_mse(
                fit.prediction, block.visibility, block.weight, ~test_mask
            )
        ),
    }


def _image_header(
    block: VisibilityBlock,
    *,
    size: int,
    pixel_size_rad: float,
    unit: str,
) -> fits.Header:
    header = fits.Header()
    header["BUNIT"] = unit
    header["CTYPE1"] = "RA---SIN"
    header["CTYPE2"] = "DEC--SIN"
    header["CRPIX1"] = (size + 1) / 2
    header["CRPIX2"] = (size + 1) / 2
    header["CRVAL1"] = np.rad2deg(block.phase_centre_rad[0])
    header["CRVAL2"] = np.rad2deg(block.phase_centre_rad[1])
    header["CDELT1"] = -np.rad2deg(pixel_size_rad)
    header["CDELT2"] = np.rad2deg(pixel_size_rad)
    return header


def _write_final_products(
    output: Path,
    block: VisibilityBlock,
    fit: QuadtreeInferenceResult,
    baseline: QuadtreeInferenceResult,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    config: AdaptiveRefinementConfig,
    fold_splits: tuple[tuple[QuadtreeLeaf, ...], ...],
    consensus: tuple[QuadtreeLeaf, ...],
    test_scans: tuple[int, ...],
    outer_seed: int,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    deepest_level = max(leaf.level for leaf in fit.topology.leaves)
    render_pixel_size = config.root_pixel_size_rad / 2**deepest_level
    surface_brightness = render_quadtree_surface_brightness(
        fit.topology, fit.flux, level=deepest_level
    )
    image = surface_brightness * render_pixel_size**2
    fits.PrimaryHDU(
        image,
        header=_image_header(
            block,
            size=image.shape[0],
            pixel_size_rad=render_pixel_size,
            unit="Jy/pixel",
        ),
    ).writeto(output / "frozen_reconstruction.fits", overwrite=True)

    diagnostic_grid = RegularGrid(128, np.deg2rad(4.0 / 3600.0))
    evaluation = evaluate_residuals(
        block,
        fit.prediction,
        diagnostic_grid,
        train_mask,
        test_mask,
        config=config.inference.direct_dft,
    )
    residual_header = _image_header(
        block,
        size=diagnostic_grid.size,
        pixel_size_rad=diagnostic_grid.pixel_size_rad,
        unit="Jy/beam",
    )
    for name, residual_image in (
        ("train", evaluation.train_dirty),
        ("test", evaluation.holdout_dirty),
        ("full", evaluation.full_dirty),
    ):
        fits.PrimaryHDU(residual_image, header=residual_header).writeto(
            output / f"frozen_{name}_residual_dirty.fits", overwrite=True
        )

    topology_path = output / "frozen_topology.csv"
    l, m = fit.topology.centers()
    widths = fit.topology.widths_rad()
    with topology_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("level", "iy", "ix", "flux_jy", "l_rad", "m_rad", "width_rad"))
        for leaf, flux, leaf_l, leaf_m, width in zip(
            fit.topology.leaves, fit.flux, l, m, widths, strict=True
        ):
            writer.writerow((leaf.level, leaf.iy, leaf.ix, flux, leaf_l, leaf_m, width))

    fit_metrics = _fit_metrics(block, fit, train_mask, test_mask)
    baseline_metrics = _fit_metrics(block, baseline, train_mask, test_mask)
    summary = {
        "schema_version": 1,
        "frozen_protocol_version": FROZEN_PROTOCOL_VERSION,
        "outer_split": {
            "strategy": "whole_scan_seeded_count_match",
            "seed": outer_seed,
            "test_scans": list(test_scans),
            "train_active_count": int(np.count_nonzero(train_mask)),
            "test_active_count": int(np.count_nonzero(test_mask)),
            "test_active_fraction": float(
                np.count_nonzero(test_mask) / np.count_nonzero(block.active)
            ),
        },
        "inner_seeds": list(FROZEN_INNER_SEEDS),
        "fold_split_counts": [len(splits) for splits in fold_splits],
        "consensus_split_count": len(consensus),
        "consensus_splits": _leaf_payload(consensus),
        "configuration": {
            "root_size": config.root_size,
            "root_pixel_arcsec": float(np.rad2deg(config.root_pixel_size_rad) * 3600),
            "sparsity_weight": config.inference.sparsity_weight,
            "solver": config.inference.solver,
            "steps": config.inference.steps,
            "kkt_tolerance": config.inference.kkt_tolerance,
            "inner_holdout_fraction": config.holdout_fraction,
            "max_rounds": config.max_rounds,
            "max_depth": config.max_depth,
            "primary_beam": "airy",
            "approximation": config.approximation.value,
        },
        "baseline": baseline_metrics,
        "consensus": fit_metrics,
        "test_improvement_over_unrefined": {
            "absolute_normalized_residual_power": (
                baseline_metrics["test_normalized_residual_power"]
                - fit_metrics["test_normalized_residual_power"]
            ),
            "relative": 1.0
            - fit_metrics["test_normalized_residual_power"]
            / baseline_metrics["test_normalized_residual_power"],
        },
        "residual_diagnostics": evaluation.diagnostics,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    np.savez(
        output / "frozen_predictions.npz",
        prediction=fit.prediction,
        residual=fit.residual,
        train_mask=train_mask,
        test_mask=test_mask,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "fixture",
        type=Path,
        nargs="?",
        default=Path("outputs/3c391_imaging_fixture.zarr"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/3c391_hierarchical_frozen_protocol"),
    )
    parser.add_argument("--block", type=int, default=0)
    parser.add_argument("--outer-test-fraction", type=float, default=0.2)
    parser.add_argument("--outer-seed", type=int, default=101)
    parser.add_argument("--visibility-tile-size", type=int, default=256)
    parser.add_argument("--pixel-tile-size", type=int, default=1024)
    parser.add_argument("--precision", choices=("float32", "float64"), default="float32")
    parser.add_argument("--no-resume", action="store_true")
    arguments = parser.parse_args()

    dataset = read_dataset(arguments.fixture)
    if not 0 <= arguments.block < len(dataset.blocks):
        raise ValueError(f"block must be between 0 and {len(dataset.blocks) - 1}")
    block = dataset.blocks[arguments.block]
    test_scans = select_outer_test_scans(
        block,
        fraction=arguments.outer_test_fraction,
        seed=arguments.outer_seed,
    )
    outer_train, outer_test = outer_scan_masks(block, test_scans)
    train_rows = np.flatnonzero(np.any(outer_train, axis=(1, 2)))
    inner_block = subset_rows(block, train_rows)
    arguments.output.mkdir(parents=True, exist_ok=True)
    protocol = {
        "frozen_protocol_version": FROZEN_PROTOCOL_VERSION,
        "fixture": str(arguments.fixture),
        "block": arguments.block,
        "outer_seed": arguments.outer_seed,
        "outer_test_fraction_target": arguments.outer_test_fraction,
        "outer_test_scans": list(test_scans),
        "inner_seeds": list(FROZEN_INNER_SEEDS),
    }
    protocol_path = arguments.output / "protocol.json"
    if protocol_path.exists() and not arguments.no_resume:
        existing = json.loads(protocol_path.read_text(encoding="utf-8"))
        if existing != protocol:
            raise ValueError("existing frozen protocol does not match requested protocol")
    else:
        protocol_path.write_text(
            json.dumps(protocol, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    primary_beam = primary_beam_from_name("airy")
    fold_splits: list[tuple[QuadtreeLeaf, ...]] = []
    for seed in FROZEN_INNER_SEEDS:
        fold_directory = arguments.output / f"inner_seed_{seed}"
        splits = None if arguments.no_resume else _load_fold(fold_directory, seed=seed)
        if splits is None:
            print(f"inner fold {seed}: starting topology discovery", flush=True)
            config = _frozen_config(
                seed=seed,
                visibility_tile_size=arguments.visibility_tile_size,
                pixel_tile_size=arguments.pixel_tile_size,
                precision=arguments.precision,
            )
            result = reconstruct_hierarchical(
                inner_block,
                config,
                primary_beam=primary_beam,
                progress=lambda message, fold_seed=seed: print(
                    f"inner fold {fold_seed}: {message}", flush=True
                ),
            )
            splits = _write_fold(fold_directory, result, seed=seed)
            print(f"inner fold {seed}: accepted {len(splits)} splits", flush=True)
        else:
            print(f"inner fold {seed}: resuming {len(splits)} accepted splits", flush=True)
        fold_splits.append(splits)

    consensus = consensus_split_leaves(tuple(fold_splits))
    config = _frozen_config(
        seed=FROZEN_INNER_SEEDS[0],
        visibility_tile_size=arguments.visibility_tile_size,
        pixel_tile_size=arguments.pixel_tile_size,
        precision=arguments.precision,
    )
    root = quadtree_sky_from_regular_grid(
        config.root_size,
        config.root_pixel_size_rad,
        np.zeros(config.root_size**2),
    )
    sky = root
    for leaf in consensus:
        if leaf not in sky.topology.leaves:
            raise ValueError(f"consensus split {leaf} is not active after its ancestors")
        sky = sky.split(leaf)

    print(
        f"outer fit: {len(consensus)} consensus splits, {len(sky.leaves)} leaves; "
        f"sealed scans={test_scans}",
        flush=True,
    )
    baseline = infer_quadtree(
        block,
        root.topology,
        outer_train,
        config.inference,
        holdout_mask=outer_test,
        primary_beam=primary_beam,
        approximation=config.approximation,
        initial_flux=root.flux,
    )
    fit = infer_quadtree(
        block,
        sky.topology,
        outer_train,
        config.inference,
        holdout_mask=outer_test,
        primary_beam=primary_beam,
        approximation=config.approximation,
        initial_flux=sky.flux,
    )
    _write_final_products(
        arguments.output,
        block,
        fit,
        baseline,
        outer_train,
        outer_test,
        config,
        tuple(fold_splits),
        consensus,
        test_scans,
        arguments.outer_seed,
    )
    final_summary = json.loads((arguments.output / "summary.json").read_text())
    print(
        json.dumps(
            {
                "test_scans": list(test_scans),
                "baseline": final_summary["baseline"],
                "consensus": final_summary["consensus"],
                "test_improvement_over_unrefined": final_summary["test_improvement_over_unrefined"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
