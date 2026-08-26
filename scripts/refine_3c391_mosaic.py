#!/usr/bin/env python3
"""Run repeated-fold joint hierarchical imaging on the seven-field mosaic."""

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
from sl1mjax.direct_operator import DirectDFTConfig
from sl1mjax.hierarchical_imaging import (
    AdaptiveRefinementConfig,
    consensus_split_leaves,
)
from sl1mjax.inference import MosaicQuadtreeInferenceResult, infer_mosaic_quadtree
from sl1mjax.mosaic_refinement import (
    MosaicHierarchicalImagingResult,
    reconstruct_mosaic_hierarchical,
)
from sl1mjax.quadtree import QuadtreeLeaf, quadtree_sky_from_regular_grid
from sl1mjax.refinement import render_quadtree_surface_brightness
from sl1mjax.sky import GaussianApproximation
from sl1mjax.split import uv_cell_split


def select_outer_test_scans(
    block: VisibilityBlock,
    *,
    fraction: float,
    seed: int,
) -> tuple[int, ...]:
    """Select whole scans close to a target active-sample fraction."""

    if not 0 < fraction < 1:
        raise ValueError("fraction must be between zero and one")
    if block.scan_id is None:
        raise ValueError("scan identifiers are required for the frozen split")
    scans = np.unique(block.scan_id)
    if scans.size < 2:
        raise ValueError("at least two scans are required")
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
        raise ValueError("outer split must leave non-empty train and test sets")
    return tuple(sorted(selected))


def outer_scan_masks(
    block: VisibilityBlock,
    scans: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Return disjoint active train and sealed-test masks."""

    if block.scan_id is None:
        raise ValueError("scan identifiers are required")
    test_rows = np.isin(block.scan_id, scans)
    test = test_rows[:, None, None] & block.active
    train = ~test_rows[:, None, None] & block.active
    if not np.any(train) or not np.any(test):
        raise ValueError("outer train and test masks must both be non-empty")
    return train, test


def subset_rows(block: VisibilityBlock, rows: np.ndarray) -> VisibilityBlock:
    """Return one block restricted to complete selected rows."""

    selected = np.asarray(rows, dtype=np.int64)
    if selected.ndim != 1 or selected.size == 0:
        raise ValueError("rows must be a non-empty one-dimensional array")
    updates: dict[str, Any] = {
        "uvw_m": block.uvw_m[selected],
        "visibility": block.visibility[selected],
        "weight": block.weight[selected],
        "flag": block.flag[selected],
        "time_s": block.time_s[selected],
        "antenna1": block.antenna1[selected],
        "antenna2": block.antenna2[selected],
    }
    for name in (
        "model_visibility",
        "field_id",
        "scan_id",
        "state_id",
        "observation_id",
        "feed1",
        "feed2",
        "interval_s",
    ):
        value = getattr(block, name)
        updates[name] = None if value is None else value[selected]
    return replace(block, **updates)


def accepted_split_leaves(
    result: MosaicHierarchicalImagingResult,
) -> tuple[QuadtreeLeaf, ...]:
    """Return all split operations accepted by one inner fold."""

    accepted: list[QuadtreeLeaf] = []
    for round_record in result.rounds:
        validation = round_record.validation
        attempt = None if validation is None else validation.accepted_attempt
        if attempt is not None:
            accepted.extend(attempt.selected)
    if len(accepted) != len(set(accepted)):
        raise ValueError("adaptive result contains a duplicate split")
    return tuple(accepted)


def _leaf_payload(leaves: tuple[QuadtreeLeaf, ...]) -> list[list[int]]:
    return [[leaf.level, leaf.iy, leaf.ix] for leaf in leaves]


def _leaves_from_payload(payload: list[list[int]]) -> tuple[QuadtreeLeaf, ...]:
    return tuple(QuadtreeLeaf(*map(int, item)) for item in payload)


def _round_summary(result: MosaicHierarchicalImagingResult) -> list[dict[str, Any]]:
    records = []
    for round_record in result.rounds:
        validation = round_record.validation
        accepted = None if validation is None else validation.accepted_attempt
        records.append(
            {
                "index": round_record.index,
                "leaf_count_before": round_record.leaf_count_before,
                "screened_count": len(round_record.screening_scores),
                "exact_rescore_count": len(round_record.exact_screening_scores),
                "selected_count": len(round_record.selection.selected),
                "accepted_split_count": 0 if accepted is None else len(accepted.selected),
                "accepted": accepted is not None,
                "numerical_failure_count": (0 if validation is None else len(validation.failures)),
                "numerical_failures": (
                    []
                    if validation is None
                    else [
                        {
                            "split_count": len(failure.selected),
                            "error": failure.error,
                        }
                        for failure in validation.failures
                    ]
                ),
                "baseline_holdout_loss": (
                    None if validation is None else validation.baseline.holdout_data
                ),
                "accepted_holdout_loss": (
                    None if accepted is None else accepted.metrics.holdout_data
                ),
                "training_relative_improvement": (
                    None if accepted is None else accepted.training_relative_improvement
                ),
                "holdout_relative_improvement": (
                    None if accepted is None else accepted.holdout_relative_improvement
                ),
            }
        )
    return records


def _write_fold(
    directory: Path,
    result: MosaicHierarchicalImagingResult,
    *,
    seed: int,
) -> tuple[QuadtreeLeaf, ...]:
    directory.mkdir(parents=True, exist_ok=True)
    splits = accepted_split_leaves(result)
    summary = {
        "schema_version": 1,
        "inner_seed": seed,
        "stop_reason": result.stop_reason,
        "elapsed_s": result.elapsed_s,
        "accepted_split_count": len(splits),
        "accepted_splits": _leaf_payload(splits),
        "final_leaf_count": len(result.inference.topology.leaves),
        "final_kkt_residual": result.inference.kkt_residual,
        "effective_max_depth": result.effective_max_depth,
        "resolution_max_depth": result.resolution_max_depth,
        "synthesized_beam": (
            None
            if result.synthesized_beam is None
            else {
                "major_fwhm_arcsec": (np.rad2deg(result.synthesized_beam.major_fwhm_rad) * 3600.0),
                "minor_fwhm_arcsec": (np.rad2deg(result.synthesized_beam.minor_fwhm_rad) * 3600.0),
                "position_angle_deg": np.rad2deg(result.synthesized_beam.position_angle_rad),
                "method": result.synthesized_beam.method,
            }
        ),
        "rounds": _round_summary(result),
    }
    (directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    np.savez(
        directory / "masks.npz",
        **{f"train_C{index + 1}": mask for index, mask in enumerate(result.train_masks)},
        **{f"holdout_C{index + 1}": mask for index, mask in enumerate(result.holdout_masks)},
    )
    return splits


def _load_fold(directory: Path, *, seed: int) -> tuple[QuadtreeLeaf, ...] | None:
    path = directory / "summary.json"
    if not path.exists():
        return None
    summary = json.loads(path.read_text(encoding="utf-8"))
    if int(summary["inner_seed"]) != seed:
        raise ValueError(f"existing fold {directory} has a different seed")
    return _leaves_from_payload(summary["accepted_splits"])


def _config(arguments: argparse.Namespace, *, seed: int) -> AdaptiveRefinementConfig:
    return AdaptiveRefinementConfig(
        root_size=arguments.root_size,
        root_pixel_size_rad=np.deg2rad(arguments.root_pixel_arcsec / 3600.0),
        inference=replace(
            AdaptiveRefinementConfig().inference,
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
        ),
        holdout_fraction=arguments.inner_holdout_fraction,
        split_seed=seed,
        split_strategy="uv_cell",
        uv_cells_per_axis=arguments.uv_cells_per_axis,
        max_rounds=arguments.max_rounds,
        max_depth=arguments.max_depth,
        maximum_pixels_per_beam=(
            None if arguments.disable_resolution_depth_cap else arguments.maximum_pixels_per_beam
        ),
        target_improvement_fraction=arguments.target_improvement_fraction,
        max_split_fraction=arguments.max_split_fraction,
        max_splits_per_round=arguments.max_splits_per_round,
        score_candidate_batch_size=arguments.score_candidate_batch_size,
        score_row_batch_size=arguments.score_row_batch_size,
        minimum_holdout_relative_improvement=(arguments.minimum_holdout_relative_improvement),
        max_refits_per_round=arguments.max_refits_per_round,
        approximation=GaussianApproximation.WIDE_FIELD,
        allow_approximate_curvature=True,
        enable_merging=False,
    )


def _fit_metrics(
    blocks: tuple[VisibilityBlock, ...],
    fit: MosaicQuadtreeInferenceResult,
    train_masks: tuple[np.ndarray, ...],
    test_masks: tuple[np.ndarray, ...],
) -> dict[str, Any]:
    partitions: dict[str, Any] = {}
    for label, masks in (("train", train_masks), ("test", test_masks)):
        residual_numerator = 0.0
        signal_numerator = 0.0
        weight_sum = 0.0
        per_pointing = []
        for index, (block, prediction, mask) in enumerate(
            zip(blocks, fit.predictions, masks, strict=True), start=1
        ):
            weight = np.where(mask, block.weight, 0.0)
            residual = np.where(mask, prediction - block.visibility, 0.0)
            residual_power = float(np.sum(weight * np.abs(residual) ** 2))
            signal_power = float(np.sum(weight * np.abs(block.visibility) ** 2))
            active_weight = float(np.sum(weight))
            residual_numerator += residual_power
            signal_numerator += signal_power
            weight_sum += active_weight
            per_pointing.append(
                {
                    "label": f"C{index}",
                    "weighted_complex_mse": residual_power / active_weight,
                    "normalized_residual_power": residual_power / signal_power,
                }
            )
        partitions[label] = {
            "weighted_complex_mse": residual_numerator / weight_sum,
            "normalized_residual_power": residual_numerator / signal_numerator,
            "per_pointing": per_pointing,
        }
    return {
        "leaf_count": len(fit.topology.leaves),
        "steps": fit.steps,
        "best_step": fit.best_step,
        "converged": fit.converged,
        "kkt_residual": fit.kkt_residual,
        "total_flux_jy": float(np.sum(fit.flux)),
        **partitions,
    }


def _write_reconstruction(
    path: Path,
    fit: MosaicQuadtreeInferenceResult,
) -> None:
    level = max(leaf.level for leaf in fit.topology.leaves)
    pixel_size = fit.topology.grid.root_pixel_size_rad / 2**level
    brightness = render_quadtree_surface_brightness(fit.topology, fit.flux, level=level)
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
            fit.topology.leaves, fit.flux, l, m, widths, strict=True
        ):
            writer.writerow((leaf.level, leaf.iy, leaf.ix, flux, leaf_l, leaf_m, width))


def _parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item) for item in value.split(",") if item.strip())
    if not seeds or len(seeds) != len(set(seeds)):
        raise argparse.ArgumentTypeError("inner seeds must be a unique comma-separated list")
    return seeds


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "fixture",
        type=Path,
        nargs="?",
        default=Path("outputs/3c391_mosaic_imaging_fixture.zarr"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/3c391_mosaic_hierarchical_frozen_protocol"),
    )
    parser.add_argument("--inner-seeds", type=_parse_seeds, default=(17, 29, 43))
    parser.add_argument("--outer-test-fraction", type=float, default=0.2)
    parser.add_argument("--outer-seed", type=int, default=101)
    parser.add_argument("--inner-holdout-fraction", type=float, default=0.2)
    parser.add_argument("--root-size", type=int, default=104)
    parser.add_argument("--root-pixel-arcsec", type=float, default=16.0)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lambda-l1", type=float, default=3e-4)
    parser.add_argument("--kkt-tolerance", type=float, default=3e-5)
    parser.add_argument("--max-rounds", type=int, default=8)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--maximum-pixels-per-beam", type=float, default=5.0)
    parser.add_argument("--disable-resolution-depth-cap", action="store_true")
    parser.add_argument("--uv-cells-per-axis", type=int, default=64)
    parser.add_argument("--target-improvement-fraction", type=float, default=0.7)
    parser.add_argument("--max-split-fraction", type=float, default=0.05)
    parser.add_argument("--max-splits-per-round", type=int, default=256)
    parser.add_argument("--score-candidate-batch-size", type=int, default=32)
    parser.add_argument("--score-row-batch-size", type=int, default=1024)
    parser.add_argument("--minimum-holdout-relative-improvement", type=float, default=0.001)
    parser.add_argument("--max-refits-per-round", type=int, default=4)
    parser.add_argument("--visibility-tile-size", type=int, default=256)
    parser.add_argument("--pixel-tile-size", type=int, default=1024)
    parser.add_argument("--precision", choices=("float32", "float64"), default="float32")
    parser.add_argument("--maximum-pointings", type=int)
    parser.add_argument("--no-resume", action="store_true")
    arguments = parser.parse_args()

    blocks = read_dataset(arguments.fixture).blocks
    if arguments.maximum_pointings is not None:
        if not 1 <= arguments.maximum_pointings <= len(blocks):
            raise ValueError("maximum-pointings must be between one and the block count")
        blocks = blocks[: arguments.maximum_pointings]
    test_scans = tuple(
        select_outer_test_scans(
            block,
            fraction=arguments.outer_test_fraction,
            seed=arguments.outer_seed + index,
        )
        for index, block in enumerate(blocks)
    )
    outer_pairs = tuple(
        outer_scan_masks(block, scans) for block, scans in zip(blocks, test_scans, strict=True)
    )
    outer_train = tuple(pair[0] for pair in outer_pairs)
    outer_test = tuple(pair[1] for pair in outer_pairs)
    inner_blocks = tuple(
        subset_rows(block, np.flatnonzero(np.any(mask, axis=(1, 2))))
        for block, mask in zip(blocks, outer_train, strict=True)
    )
    arguments.output.mkdir(parents=True, exist_ok=True)
    protocol = {
        "schema_version": 1,
        "fixture": str(arguments.fixture),
        "pointing_count": len(blocks),
        "outer_seed": arguments.outer_seed,
        "outer_test_fraction": arguments.outer_test_fraction,
        "outer_test_scans": [list(scans) for scans in test_scans],
        "inner_seeds": list(arguments.inner_seeds),
        "root_size": arguments.root_size,
        "root_pixel_arcsec": arguments.root_pixel_arcsec,
        "lambda_l1": arguments.lambda_l1,
        "steps": arguments.steps,
        "kkt_tolerance": arguments.kkt_tolerance,
        "max_rounds": arguments.max_rounds,
        "max_depth": arguments.max_depth,
        "maximum_pixels_per_beam": (
            None if arguments.disable_resolution_depth_cap else arguments.maximum_pixels_per_beam
        ),
        "inner_holdout_fraction": arguments.inner_holdout_fraction,
        "uv_cells_per_axis": arguments.uv_cells_per_axis,
        "target_improvement_fraction": arguments.target_improvement_fraction,
        "max_split_fraction": arguments.max_split_fraction,
        "max_splits_per_round": arguments.max_splits_per_round,
        "minimum_holdout_relative_improvement": (arguments.minimum_holdout_relative_improvement),
        "primary_beam": "airy",
        "approximation": "wide_field",
    }
    protocol_path = arguments.output / "protocol.json"
    if protocol_path.exists() and not arguments.no_resume:
        if json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
            raise ValueError("existing protocol does not match the requested run")
    else:
        protocol_path.write_text(
            json.dumps(protocol, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    beam = primary_beam_from_name("airy")
    mosaic_phase_centre = blocks[0].phase_centre_rad
    fold_splits = []
    for seed in arguments.inner_seeds:
        fold_directory = arguments.output / f"inner_seed_{seed}"
        splits = None if arguments.no_resume else _load_fold(fold_directory, seed=seed)
        if splits is None:
            split_pairs = tuple(
                uv_cell_split(
                    block,
                    holdout_fraction=arguments.inner_holdout_fraction,
                    cells_per_axis=arguments.uv_cells_per_axis,
                    seed=seed + index,
                )
                for index, block in enumerate(inner_blocks)
            )
            train_masks = tuple(split.train for split in split_pairs)
            holdout_masks = tuple(split.holdout for split in split_pairs)
            config = _config(arguments, seed=seed)
            print(f"inner fold {seed}: starting joint topology discovery", flush=True)
            result = reconstruct_mosaic_hierarchical(
                inner_blocks,
                train_masks,
                holdout_masks,
                mosaic_phase_centre,
                config,
                primary_beam=beam,
                progress=lambda message, fold_seed=seed: print(
                    f"inner fold {fold_seed}: {message}", flush=True
                ),
            )
            splits = _write_fold(fold_directory, result, seed=seed)
            print(f"inner fold {seed}: accepted {len(splits)} splits", flush=True)
        else:
            print(f"inner fold {seed}: resumed {len(splits)} splits", flush=True)
        fold_splits.append(splits)

    consensus = consensus_split_leaves(tuple(fold_splits))
    config = _config(arguments, seed=arguments.inner_seeds[0])
    root = quadtree_sky_from_regular_grid(
        config.root_size,
        config.root_pixel_size_rad,
        np.zeros(config.root_size**2),
    )
    hierarchy = root
    for leaf in consensus:
        if leaf not in hierarchy.topology.leaves:
            raise ValueError(f"consensus split {leaf} is not active after its ancestors")
        hierarchy = hierarchy.split(leaf)
    print(
        f"outer fit: {len(consensus)} consensus splits, {len(hierarchy.leaves)} leaves",
        flush=True,
    )
    baseline = infer_mosaic_quadtree(
        blocks,
        root.topology,
        outer_train,
        mosaic_phase_centre,
        config.inference,
        holdout_masks=outer_test,
        primary_beam=beam,
        approximation=config.approximation,
        initial_flux=root.flux,
    )
    baseline_by_root = dict(zip(root.leaves, baseline.flux, strict=True))
    hierarchy_initial = np.asarray(
        [
            baseline_by_root[QuadtreeLeaf(0, leaf.iy // 2**leaf.level, leaf.ix // 2**leaf.level)]
            / 4**leaf.level
            for leaf in hierarchy.leaves
        ]
    )
    fit = infer_mosaic_quadtree(
        blocks,
        hierarchy.topology,
        outer_train,
        mosaic_phase_centre,
        config.inference,
        holdout_masks=outer_test,
        primary_beam=beam,
        approximation=config.approximation,
        initial_flux=hierarchy_initial,
    )
    baseline_metrics = _fit_metrics(blocks, baseline, outer_train, outer_test)
    fit_metrics = _fit_metrics(blocks, fit, outer_train, outer_test)
    baseline_test = baseline_metrics["test"]["normalized_residual_power"]
    fit_test = fit_metrics["test"]["normalized_residual_power"]
    summary = {
        **protocol,
        "fold_split_counts": [len(splits) for splits in fold_splits],
        "consensus_split_count": len(consensus),
        "consensus_splits": _leaf_payload(consensus),
        "baseline": baseline_metrics,
        "consensus": fit_metrics,
        "sealed_test_improvement_over_unrefined": {
            "absolute_normalized_residual_power": baseline_test - fit_test,
            "relative": 1.0 - fit_test / baseline_test,
        },
    }
    _write_reconstruction(arguments.output / "baseline_reconstruction.fits", baseline)
    _write_reconstruction(arguments.output / "consensus_reconstruction.fits", fit)
    _write_topology(arguments.output / "consensus_topology.csv", fit)
    np.savez(
        arguments.output / "predictions.npz",
        **{
            f"baseline_C{index + 1}": prediction
            for index, prediction in enumerate(baseline.predictions)
        },
        **{
            f"consensus_C{index + 1}": prediction
            for index, prediction in enumerate(fit.predictions)
        },
        **{f"outer_train_C{index + 1}": mask for index, mask in enumerate(outer_train)},
        **{f"outer_test_C{index + 1}": mask for index, mask in enumerate(outer_test)},
    )
    (arguments.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "baseline": baseline_metrics,
                "consensus": fit_metrics,
                "sealed_test_improvement_over_unrefined": summary[
                    "sealed_test_improvement_over_unrefined"
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
