#!/usr/bin/env python3
"""Run adaptive quadtree imaging on one block of the portable 3C391 fixture."""

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
    AdaptiveRefinementRound,
    HierarchicalImagingResult,
    reconstruct_hierarchical,
)
from sl1mjax.inference import InferenceConfig, save_checkpoint
from sl1mjax.refinement import (
    quadtree_objective_metrics,
    render_quadtree_surface_brightness,
)
from sl1mjax.sky import GaussianApproximation


def _select_even_rows(
    block: VisibilityBlock,
    maximum_rows: int | None,
) -> VisibilityBlock:
    if maximum_rows is None or maximum_rows >= block.shape[0]:
        return block
    if maximum_rows < 2:
        raise ValueError("maximum_rows must be at least two")
    rows = np.linspace(0, block.shape[0] - 1, maximum_rows, dtype=np.int64)
    assert block.field_id is not None
    assert block.scan_id is not None
    assert block.state_id is not None
    assert block.observation_id is not None
    assert block.feed1 is not None
    assert block.feed2 is not None
    assert block.interval_s is not None
    model_visibility = None if block.model_visibility is None else block.model_visibility[rows]
    return replace(
        block,
        uvw_m=block.uvw_m[rows],
        visibility=block.visibility[rows],
        weight=block.weight[rows],
        flag=block.flag[rows],
        time_s=block.time_s[rows],
        antenna1=block.antenna1[rows],
        antenna2=block.antenna2[rows],
        model_visibility=model_visibility,
        field_id=block.field_id[rows],
        scan_id=block.scan_id[rows],
        state_id=block.state_id[rows],
        observation_id=block.observation_id[rows],
        feed1=block.feed1[rows],
        feed2=block.feed2[rows],
        interval_s=block.interval_s[rows],
    )


def _round_diagnostics(result: HierarchicalImagingResult) -> list[dict[str, Any]]:
    diagnostics = []
    for round_result in result.rounds:
        validation = round_result.validation
        accepted = None if validation is None else validation.accepted_attempt
        grid = result.inference.topology.grid
        boundary_selected = sum(
            leaf.ix in {0, grid.leaves_per_axis(leaf.level) - 1}
            or leaf.iy in {0, grid.leaves_per_axis(leaf.level) - 1}
            for leaf in round_result.selection.selected
        )
        diagnostics.append(
            {
                "index": round_result.index,
                "leaf_count_before": round_result.leaf_count_before,
                "candidate_count": len(round_result.screening_scores),
                "eligible_count": sum(score.eligible for score in round_result.screening_scores),
                "selected": [
                    [leaf.level, leaf.iy, leaf.ix] for leaf in round_result.selection.selected
                ],
                "available_predicted_improvement": (round_result.selection.available_improvement),
                "selected_predicted_improvement": (round_result.selection.selected_improvement),
                "boundary_selected_fraction": (
                    boundary_selected / len(round_result.selection.selected)
                    if round_result.selection.selected
                    else 0.0
                ),
                "score_fraction": round_result.selection.covered_fraction,
                "attempts": []
                if validation is None
                else [
                    {
                        "split_count": len(attempt.selected),
                        "leaf_count": len(attempt.fit.topology.leaves),
                        "training_objective": attempt.metrics.objective,
                        "holdout_loss": attempt.metrics.holdout_data,
                        "training_relative_improvement": (attempt.training_relative_improvement),
                        "holdout_relative_improvement": (attempt.holdout_relative_improvement),
                        "accepted": attempt.accepted,
                        "optimizer_steps": attempt.fit.steps,
                        "optimizer_best_step": attempt.fit.best_step,
                        "optimizer_converged": attempt.fit.converged,
                    }
                    for attempt in validation.attempts
                ],
                "baseline_training_objective": (
                    None if validation is None else validation.baseline.objective
                ),
                "baseline_holdout_loss": (
                    None if validation is None else validation.baseline.holdout_data
                ),
                "accepted_split_count": 0 if accepted is None else len(accepted.selected),
                **_merge_diagnostics(round_result),
            }
        )
    return diagnostics


def _merge_diagnostics(round_result: AdaptiveRefinementRound) -> dict[str, Any]:
    merge_validation = round_result.merge_validation
    merge_accepted = None if merge_validation is None else merge_validation.accepted_attempt
    merge_selection = round_result.merge_selection
    return {
        "merge_candidate_count": len(round_result.merge_evaluations),
        "merge_evaluations": [
            {
                "leaf": [evaluation.leaf.level, evaluation.leaf.iy, evaluation.leaf.ix],
                "parent_flux_jy": evaluation.parent_flux,
                "predicted_improvement": evaluation.predicted_improvement,
                "holdout_change": evaluation.holdout_change,
            }
            for evaluation in round_result.merge_evaluations
        ],
        "merge_selected": []
        if merge_selection is None
        else [[leaf.level, leaf.iy, leaf.ix] for leaf in merge_selection.selected],
        "merge_selected_predicted_improvement": (
            None if merge_selection is None else merge_selection.selected_improvement
        ),
        "merge_attempts": []
        if merge_validation is None
        else [
            {
                "merge_count": len(attempt.selected),
                "leaf_count": len(attempt.fit.topology.leaves),
                "training_objective": attempt.metrics.objective,
                "holdout_loss": attempt.metrics.holdout_data,
                "training_relative_improvement": attempt.training_relative_improvement,
                "holdout_relative_improvement": attempt.holdout_relative_improvement,
                "accepted": attempt.accepted,
            }
            for attempt in merge_validation.attempts
        ],
        "accepted_merge_count": 0 if merge_accepted is None else len(merge_accepted.selected),
    }


def _write_products(
    output: Path,
    block: VisibilityBlock,
    result: HierarchicalImagingResult,
    config: AdaptiveRefinementConfig,
    fixture: Path,
    block_index: int,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    final_metrics = quadtree_objective_metrics(
        block,
        result.inference,
        result.train_mask,
        config.inference,
        holdout_mask=result.holdout_mask,
    )
    deepest_level = max(leaf.level for leaf in result.inference.topology.leaves)
    surface_brightness = render_quadtree_surface_brightness(
        result.inference.topology,
        result.inference.flux,
        level=deepest_level,
    )
    render_pixel_size = config.root_pixel_size_rad / 2**deepest_level
    image = surface_brightness * render_pixel_size**2
    image_path = output / "hierarchical_reconstruction.fits"
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
    fits.PrimaryHDU(image, header=header).writeto(image_path, overwrite=True)

    topology_path = output / "hierarchical_topology.csv"
    l, m = result.inference.topology.centers()
    widths = result.inference.topology.widths_rad()
    with topology_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("level", "iy", "ix", "flux_jy", "l_rad", "m_rad", "width_rad"))
        for leaf, flux, leaf_l, leaf_m, width in zip(
            result.inference.topology.leaves,
            result.inference.flux,
            l,
            m,
            widths,
            strict=True,
        ):
            writer.writerow((leaf.level, leaf.iy, leaf.ix, flux, leaf_l, leaf_m, width))

    scores_path = output / "hierarchical_scores.csv"
    with scores_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "round",
                "rank",
                "level",
                "iy",
                "ix",
                "parent_flux_jy",
                "gradient_horizontal",
                "gradient_vertical",
                "gradient_diagonal",
                "eigenvalue_ratio",
                "raw_predicted_improvement",
                "predicted_improvement",
                "eligible",
                "selected",
                "curvature_mode",
            )
        )
        for round_result in result.rounds:
            selected = set(round_result.selection.selected)
            ranked = sorted(
                round_result.screening_scores,
                key=lambda score: (-score.predicted_improvement, score.leaf),
            )
            for rank, score in enumerate(ranked, start=1):
                writer.writerow(
                    (
                        round_result.index,
                        rank,
                        score.leaf.level,
                        score.leaf.iy,
                        score.leaf.ix,
                        score.parent_flux,
                        *score.gradient,
                        score.eigenvalue_ratio,
                        score.raw_predicted_improvement,
                        score.predicted_improvement,
                        score.eligible,
                        score.leaf in selected,
                        score.curvature_mode,
                    )
                )

    np.savez(
        output / "hierarchical_residuals.npz",
        prediction=result.inference.prediction,
        residual=result.inference.residual,
        train_mask=result.train_mask,
        holdout_mask=result.holdout_mask,
    )
    save_checkpoint(output / "hierarchical_reconstruction.checkpoint.npz", result.inference)
    summary = {
        "fixture": str(fixture),
        "block_index": block_index,
        "block_shape": list(block.shape),
        "stop_reason": result.stop_reason,
        "elapsed_s": result.elapsed_s,
        "initial_leaf_count": config.root_size**2,
        "final_leaf_count": len(result.inference.topology.leaves),
        "accepted_merge_rounds": sum(
            1
            for round_result in result.rounds
            if round_result.merge_validation is not None
            and round_result.merge_validation.accepted_attempt is not None
        ),
        "final_merge_eligible_streak": [
            [leaf.level, leaf.iy, leaf.ix, streak]
            for leaf, streak in sorted(result.merge_hysteresis.eligible_streak.items())
        ],
        "final_merge_split_cooldown": [
            [leaf.level, leaf.iy, leaf.ix, cooldown]
            for leaf, cooldown in sorted(result.merge_hysteresis.split_cooldown.items())
        ],
        "deepest_level": deepest_level,
        "render_shape": list(image.shape),
        "total_flux_jy": float(np.sum(result.inference.flux)),
        "peak_render_pixel_jy": float(np.max(image)),
        "final_training_objective": final_metrics.objective,
        "final_training_data_loss": final_metrics.training_data,
        "final_holdout_loss": final_metrics.holdout_data,
        "final_optimizer_steps": result.inference.steps,
        "final_optimizer_best_step": result.inference.best_step,
        "final_optimizer_converged": result.inference.converged,
        "configuration": {
            "root_size": config.root_size,
            "root_pixel_arcsec": np.rad2deg(config.root_pixel_size_rad) * 3600,
            "steps": config.inference.steps,
            "patience": config.inference.patience,
            "initial_intensity": config.inference.initial_intensity,
            "learning_rate": config.inference.learning_rate,
            "sparsity_weight": config.inference.sparsity_weight,
            "leaf_penalty": config.leaf_penalty,
            "holdout_fraction": config.holdout_fraction,
            "split_strategy": config.split_strategy,
            "max_rounds": config.max_rounds,
            "max_depth": config.max_depth,
            "target_improvement_fraction": config.target_improvement_fraction,
            "max_split_fraction": config.max_split_fraction,
            "max_splits_per_round": config.max_splits_per_round,
            "minimum_holdout_relative_improvement": (config.minimum_holdout_relative_improvement),
            "approximation": config.approximation.value,
            "precision": config.inference.direct_dft.precision,
            "enable_merging": config.enable_merging,
            "max_merge_fraction": config.max_merge_fraction,
            "max_merges_per_round": config.max_merges_per_round,
            "merge_target_improvement_fraction": (config.merge_target_improvement_fraction),
            "merge_required_streak": config.merge_required_streak,
            "merge_cooldown_rounds": config.merge_cooldown_rounds,
        },
        "rounds": _round_diagnostics(result),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("fixture", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/3c391_hierarchical"))
    parser.add_argument("--block", type=int, default=0)
    parser.add_argument("--maximum-rows", type=int)
    parser.add_argument("--root-size", type=int, default=128)
    parser.add_argument("--root-pixel-arcsec", type=float, default=4.0)
    parser.add_argument(
        "--steps",
        type=int,
        default=1000,
        help="Maximum steps per fixed-topology solve; early stopping normally ends sooner.",
    )
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--initial-intensity", type=float, default=1e-3)
    parser.add_argument("--sparsity-weight", type=float, default=1e-4)
    parser.add_argument("--leaf-penalty", type=float, default=1e-7)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--split-strategy", choices=("uv_cell", "random_row"), default="uv_cell")
    parser.add_argument("--split-seed", type=int, default=17)
    parser.add_argument("--uv-cells-per-axis", type=int, default=8)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--target-improvement-fraction", type=float, default=0.7)
    parser.add_argument("--max-split-fraction", type=float, default=0.05)
    parser.add_argument("--max-splits-per-round", type=int)
    parser.add_argument("--max-refits-per-round", type=int, default=4)
    parser.add_argument("--minimum-holdout-relative-improvement", type=float, default=0.0)
    parser.add_argument("--minimum-parent-flux", type=float, default=0.0)
    parser.add_argument(
        "--approximation",
        choices=("paraxial", "wide-field"),
        default="wide-field",
        help="Square-pixel kernel. Shared Haar curvature is exact only for "
        "paraxial pixels without a primary beam.",
    )
    parser.add_argument(
        "--allow-approximate-curvature",
        action="store_true",
        help="Opt into shared per-level Haar curvature when it is not exact. "
        "reconstruct_hierarchical already does this for wide-field or beamed runs.",
    )
    parser.add_argument(
        "--disable-merging",
        action="store_true",
        help="Skip coarsening (shrinkage) entirely and only run split rounds.",
    )
    parser.add_argument("--max-merge-fraction", type=float, default=0.05)
    parser.add_argument("--max-merges-per-round", type=int)
    parser.add_argument("--merge-target-improvement-fraction", type=float, default=0.7)
    parser.add_argument("--merge-required-streak", type=int, default=2)
    parser.add_argument("--merge-cooldown-rounds", type=int, default=1)
    parser.add_argument("--primary-beam", choices=("none", "gaussian", "airy"), default="none")
    parser.add_argument("--beam-squint", action="store_true")
    parser.add_argument("--precision", choices=("float32", "float64"), default="float32")
    parser.add_argument("--visibility-tile-size", type=int, default=256)
    parser.add_argument("--pixel-tile-size", type=int, default=1024)
    arguments = parser.parse_args()

    dataset = read_dataset(arguments.fixture)
    if not 0 <= arguments.block < len(dataset.blocks):
        raise ValueError(f"block must be between 0 and {len(dataset.blocks) - 1}")
    block = _select_even_rows(dataset.blocks[arguments.block], arguments.maximum_rows)
    approximation = (
        GaussianApproximation.PARAXIAL
        if arguments.approximation == "paraxial"
        else GaussianApproximation.WIDE_FIELD
    )
    direct = DirectDFTConfig(
        visibility_chunk_size=arguments.visibility_tile_size,
        pixel_chunk_size=arguments.pixel_tile_size,
        precision=arguments.precision,
    )
    config = AdaptiveRefinementConfig(
        root_size=arguments.root_size,
        root_pixel_size_rad=np.deg2rad(arguments.root_pixel_arcsec / 3600.0),
        inference=InferenceConfig(
            steps=arguments.steps,
            learning_rate=arguments.learning_rate,
            sparsity_weight=arguments.sparsity_weight,
            initial_intensity=arguments.initial_intensity,
            patience=arguments.patience,
            validation_interval=10,
            operator_mode="explicit",
            direct_dft=direct,
        ),
        holdout_fraction=arguments.holdout_fraction,
        split_seed=arguments.split_seed,
        split_strategy=arguments.split_strategy,
        uv_cells_per_axis=arguments.uv_cells_per_axis,
        max_rounds=arguments.max_rounds,
        max_depth=arguments.max_depth,
        leaf_penalty=arguments.leaf_penalty,
        target_improvement_fraction=arguments.target_improvement_fraction,
        max_split_fraction=arguments.max_split_fraction,
        max_splits_per_round=arguments.max_splits_per_round,
        min_parent_flux=arguments.minimum_parent_flux,
        minimum_holdout_relative_improvement=(arguments.minimum_holdout_relative_improvement),
        max_refits_per_round=arguments.max_refits_per_round,
        approximation=approximation,
        allow_approximate_curvature=arguments.allow_approximate_curvature,
        enable_merging=not arguments.disable_merging,
        max_merge_fraction=arguments.max_merge_fraction,
        max_merges_per_round=arguments.max_merges_per_round,
        merge_target_improvement_fraction=arguments.merge_target_improvement_fraction,
        merge_required_streak=arguments.merge_required_streak,
        merge_cooldown_rounds=arguments.merge_cooldown_rounds,
    )
    primary_beam = primary_beam_from_name(
        arguments.primary_beam,
        apply_squint=arguments.beam_squint,
    )
    result = reconstruct_hierarchical(
        block,
        config,
        primary_beam=primary_beam,
    )
    _write_products(
        arguments.output,
        block,
        result,
        config,
        arguments.fixture,
        arguments.block,
    )
    print(
        json.dumps(
            {
                "stop_reason": result.stop_reason,
                "rounds": len(result.rounds),
                "initial_leaves": config.root_size**2,
                "final_leaves": len(result.inference.topology.leaves),
                "accepted_merge_rounds": sum(
                    1
                    for round_result in result.rounds
                    if round_result.merge_validation is not None
                    and round_result.merge_validation.accepted_attempt is not None
                ),
                "elapsed_s": result.elapsed_s,
                "output": str(arguments.output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
