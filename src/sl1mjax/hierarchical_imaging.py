"""End-to-end adaptive quadtree imaging with held-out split validation."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Literal

import numpy as np

from sl1mjax.beam import VLAPrimaryBeam
from sl1mjax.data.canonical import VisibilityBlock
from sl1mjax.inference import InferenceConfig, QuadtreeInferenceResult, infer_quadtree
from sl1mjax.quadtree import quadtree_sky_from_regular_grid
from sl1mjax.refinement import (
    BulkSplitSelection,
    RefinementBatchResult,
    ResidualHaarScore,
    batched_residual_haar_scores,
    refine_quadtree_batch,
    residual_haar_scores,
    select_bulk_splits,
)
from sl1mjax.sky import GaussianApproximation
from sl1mjax.split import random_row_split, uv_cell_split


@dataclass(frozen=True)
class AdaptiveRefinementConfig:
    """Static geometry, optimization, marking, and validation controls."""

    root_size: int = 64
    root_pixel_size_rad: float = np.deg2rad(4.0 / 3600.0)
    inference: InferenceConfig = InferenceConfig(operator_mode="explicit")
    holdout_fraction: float = 0.2
    split_seed: int = 0
    split_strategy: Literal["uv_cell", "random_row"] = "uv_cell"
    uv_cells_per_axis: int = 8
    max_rounds: int = 3
    max_depth: int = 3
    leaf_penalty: float = 0.0
    target_improvement_fraction: float = 0.7
    max_split_fraction: float = 0.05
    max_splits_per_round: int | None = None
    min_parent_flux: float = 0.0
    min_curvature: float = 0.0
    min_eigenvalue_ratio: float = 1e-8
    ridge_relative: float = 1e-8
    minimum_training_relative_improvement: float = 0.0
    minimum_holdout_relative_improvement: float = 0.0
    max_refits_per_round: int = 4
    approximation: GaussianApproximation = GaussianApproximation.PARAXIAL
    allow_approximate_curvature: bool = False


@dataclass(frozen=True)
class AdaptiveRefinementRound:
    """Screening, marking, and validation record for one topology round."""

    index: int
    leaf_count_before: int
    screening_scores: tuple[ResidualHaarScore, ...]
    selection: BulkSplitSelection
    validation: RefinementBatchResult | None


@dataclass(frozen=True)
class HierarchicalImagingResult:
    """Accepted quadtree fit and the complete adaptive decision history."""

    inference: QuadtreeInferenceResult
    rounds: tuple[AdaptiveRefinementRound, ...]
    train_mask: np.ndarray
    holdout_mask: np.ndarray
    stop_reason: str
    elapsed_s: float


def _validate_config(config: AdaptiveRefinementConfig) -> None:
    if config.root_size < 1:
        raise ValueError("root_size must be positive")
    if not np.isfinite(config.root_pixel_size_rad) or config.root_pixel_size_rad <= 0:
        raise ValueError("root_pixel_size_rad must be finite and positive")
    if not 0 < config.holdout_fraction < 1:
        raise ValueError("holdout_fraction must be between zero and one")
    if config.max_rounds < 0:
        raise ValueError("max_rounds must be non-negative")
    if config.max_depth < 1:
        raise ValueError("max_depth must be positive")
    if config.inference.operator_mode != "explicit":
        raise ValueError("adaptive quadtree imaging requires operator_mode='explicit'")


def _split_masks(
    block: VisibilityBlock,
    config: AdaptiveRefinementConfig,
) -> tuple[np.ndarray, np.ndarray]:
    if config.split_strategy == "uv_cell":
        split = uv_cell_split(
            block,
            holdout_fraction=config.holdout_fraction,
            cells_per_axis=config.uv_cells_per_axis,
            seed=config.split_seed,
        )
    elif config.split_strategy == "random_row":
        split = random_row_split(
            block,
            holdout_fraction=config.holdout_fraction,
            seed=config.split_seed,
        )
    else:
        raise ValueError("split_strategy must be uv_cell or random_row")
    return split.train, split.holdout


def reconstruct_hierarchical(
    block: VisibilityBlock,
    config: AdaptiveRefinementConfig | None = None,
    *,
    fixed_gains: np.ndarray | None = None,
    primary_beam: VLAPrimaryBeam | None = None,
) -> HierarchicalImagingResult:
    """Fit an initial regular quadtree and run validated bulk refinement.

    The fast score evaluates every active leaf. A leaf-count cost of
    ``3 * leaf_penalty`` is subtracted before marking because each split adds
    three leaves. When shared curvature is approximate, the marked shortlist
    is rescored with the exact per-parent calculation before any topology
    change is attempted.
    """

    selected_config = config or AdaptiveRefinementConfig()
    _validate_config(selected_config)
    approximation = GaussianApproximation(selected_config.approximation)
    train_mask, holdout_mask = _split_masks(block, selected_config)
    initial_sky = quadtree_sky_from_regular_grid(
        selected_config.root_size,
        selected_config.root_pixel_size_rad,
        np.zeros(selected_config.root_size**2),
    )
    started = perf_counter()
    current_fit = infer_quadtree(
        block,
        initial_sky.topology,
        train_mask,
        selected_config.inference,
        holdout_mask=holdout_mask,
        fixed_gains=fixed_gains,
        primary_beam=primary_beam,
        approximation=approximation,
        leaf_penalty=selected_config.leaf_penalty,
    )

    rounds: list[AdaptiveRefinementRound] = []
    stop_reason = "maximum_rounds"
    for round_index in range(selected_config.max_rounds):
        scores = batched_residual_haar_scores(
            block,
            current_fit,
            train_mask,
            selected_config.inference,
            max_depth=selected_config.max_depth,
            fixed_gains=fixed_gains,
            primary_beam=primary_beam,
            approximation=approximation,
            min_parent_flux=selected_config.min_parent_flux,
            min_curvature=selected_config.min_curvature,
            min_eigenvalue_ratio=selected_config.min_eigenvalue_ratio,
            ridge_relative=selected_config.ridge_relative,
            allow_approximate_curvature=selected_config.allow_approximate_curvature,
        )
        selection = select_bulk_splits(
            scores,
            len(current_fit.topology.leaves),
            target_improvement_fraction=(
                selected_config.target_improvement_fraction
            ),
            max_split_fraction=selected_config.max_split_fraction,
            max_splits=selected_config.max_splits_per_round,
            split_cost=3.0 * selected_config.leaf_penalty,
        )
        if selection.selected and any(
            score.curvature_mode.startswith("per_level_approximate")
            for score in scores
        ):
            exact_scores = residual_haar_scores(
                block,
                current_fit,
                train_mask,
                selected_config.inference,
                candidates=selection.selected,
                fixed_gains=fixed_gains,
                primary_beam=primary_beam,
                approximation=approximation,
                min_parent_flux=selected_config.min_parent_flux,
                min_curvature=selected_config.min_curvature,
                min_eigenvalue_ratio=selected_config.min_eigenvalue_ratio,
                ridge_relative=selected_config.ridge_relative,
            )
            selection = select_bulk_splits(
                exact_scores,
                len(current_fit.topology.leaves),
                target_improvement_fraction=(
                    selected_config.target_improvement_fraction
                ),
                max_split_fraction=selected_config.max_split_fraction,
                max_splits=selected_config.max_splits_per_round,
                split_cost=3.0 * selected_config.leaf_penalty,
            )
        if not selection.selected:
            rounds.append(
                AdaptiveRefinementRound(
                    index=round_index,
                    leaf_count_before=len(current_fit.topology.leaves),
                    screening_scores=scores,
                    selection=selection,
                    validation=None,
                )
            )
            stop_reason = "no_eligible_splits"
            break

        validation = refine_quadtree_batch(
            block,
            current_fit,
            train_mask,
            holdout_mask,
            selected_config.inference,
            selection.selected,
            fixed_gains=fixed_gains,
            primary_beam=primary_beam,
            approximation=approximation,
            minimum_training_relative_improvement=(
                selected_config.minimum_training_relative_improvement
            ),
            minimum_holdout_relative_improvement=(
                selected_config.minimum_holdout_relative_improvement
            ),
            max_refits=selected_config.max_refits_per_round,
        )
        rounds.append(
            AdaptiveRefinementRound(
                index=round_index,
                leaf_count_before=len(current_fit.topology.leaves),
                screening_scores=scores,
                selection=selection,
                validation=validation,
            )
        )
        accepted = validation.accepted_attempt
        if accepted is None:
            stop_reason = "validation_rejected"
            break
        current_fit = accepted.fit

    return HierarchicalImagingResult(
        inference=current_fit,
        rounds=tuple(rounds),
        train_mask=train_mask,
        holdout_mask=holdout_mask,
        stop_reason=stop_reason,
        elapsed_s=perf_counter() - started,
    )
