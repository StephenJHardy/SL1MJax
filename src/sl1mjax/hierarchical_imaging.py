"""End-to-end adaptive quadtree imaging with held-out split validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Literal

import numpy as np

from sl1mjax.beam import VLAPrimaryBeam
from sl1mjax.data.canonical import VisibilityBlock
from sl1mjax.inference import InferenceConfig, QuadtreeInferenceResult, infer_quadtree
from sl1mjax.quadtree import QuadtreeLeaf, quadtree_sky_from_regular_grid
from sl1mjax.refinement import (
    BulkMergeSelection,
    BulkSplitSelection,
    LocalMergeEvaluation,
    MergeHysteresisState,
    RefinementBatchResult,
    ResidualHaarScore,
    advance_merge_hysteresis,
    batched_residual_haar_scores,
    local_four_sibling_merge_lookahead,
    merge_quadtree_batch,
    mergeable_parents,
    refine_quadtree_batch,
    residual_haar_scores,
    select_bulk_merges,
    select_bulk_splits,
)
from sl1mjax.sky import GaussianApproximation
from sl1mjax.split import random_row_split, uv_cell_split


@dataclass(frozen=True)
class AdaptiveRefinementConfig:
    """Static geometry, optimization, marking, and validation controls.

    The default square kernel is wide-field. Shared Haar curvature is treated
    as exact only for paraxial pixels without a primary beam.
    """

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
    approximation: GaussianApproximation = GaussianApproximation.WIDE_FIELD
    allow_approximate_curvature: bool = False
    enable_merging: bool = True
    max_merge_fraction: float = 0.05
    max_merges_per_round: int | None = None
    merge_target_improvement_fraction: float = 0.7
    merge_required_streak: int = 2
    merge_cooldown_rounds: int = 1


@dataclass(frozen=True)
class AdaptiveRefinementRound:
    """Screening, marking, and validation record for one topology round.

    Splits (growth) and merges (shrinkage) run within the same round with
    independent budgets: the merge fields reflect whichever complete
    sibling groups existed once this round's split (if any) was applied,
    scored with the exact reverse local lookahead and gated by the
    persistent hysteresis state carried across rounds.
    """

    index: int
    leaf_count_before: int
    screening_scores: tuple[ResidualHaarScore, ...]
    selection: BulkSplitSelection
    validation: RefinementBatchResult | None
    merge_evaluations: tuple[LocalMergeEvaluation, ...] = ()
    merge_selection: BulkMergeSelection | None = None
    merge_validation: RefinementBatchResult | None = None


@dataclass(frozen=True)
class HierarchicalImagingResult:
    """Accepted quadtree fit and the complete adaptive decision history."""

    inference: QuadtreeInferenceResult
    rounds: tuple[AdaptiveRefinementRound, ...]
    train_mask: np.ndarray
    holdout_mask: np.ndarray
    stop_reason: str
    elapsed_s: float
    merge_hysteresis: MergeHysteresisState = field(default_factory=MergeHysteresisState.empty)


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
    three leaves. Shared per-level curvature is exact only for paraxial
    squares without a primary beam. Wide-field or beam-weighted screens use
    that shared Gram as an approximation and rescore the marked shortlist
    with the exact per-parent calculation before any topology change.
    """

    selected_config = config or AdaptiveRefinementConfig()
    _validate_config(selected_config)
    approximation = GaussianApproximation(selected_config.approximation)
    curvature_is_exact = (
        approximation is GaussianApproximation.PARAXIAL and primary_beam is None
    )
    allow_approximate_curvature = (
        selected_config.allow_approximate_curvature or not curvature_is_exact
    )
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
    merge_hysteresis = MergeHysteresisState.empty()
    for round_index in range(selected_config.max_rounds):
        leaf_count_before = len(current_fit.topology.leaves)

        # --- Growth: screen, mark, and validate a split batch. ---
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
            allow_approximate_curvature=allow_approximate_curvature,
        )
        selection = select_bulk_splits(
            scores,
            leaf_count_before,
            target_improvement_fraction=(selected_config.target_improvement_fraction),
            max_split_fraction=selected_config.max_split_fraction,
            max_splits=selected_config.max_splits_per_round,
            split_cost=3.0 * selected_config.leaf_penalty,
        )
        if selection.selected and any(
            score.curvature_mode.startswith("per_level_approximate") for score in scores
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
                leaf_count_before,
                target_improvement_fraction=(selected_config.target_improvement_fraction),
                max_split_fraction=selected_config.max_split_fraction,
                max_splits=selected_config.max_splits_per_round,
                split_cost=3.0 * selected_config.leaf_penalty,
            )

        split_validation: RefinementBatchResult | None = None
        split_accepted = False
        just_split: tuple[QuadtreeLeaf, ...] = ()
        if selection.selected:
            split_validation = refine_quadtree_batch(
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
            split_accepted_attempt = split_validation.accepted_attempt
            if split_accepted_attempt is not None:
                current_fit = split_accepted_attempt.fit
                just_split = split_accepted_attempt.selected
                split_accepted = True

        # --- Shrinkage: score, gate by hysteresis, mark, and validate a merge
        # batch. Runs on whatever topology growth left behind this round, so
        # split and merge share one adaptive epoch with independent budgets. ---
        merge_candidates: tuple[QuadtreeLeaf, ...] = ()
        merge_evaluations: tuple[LocalMergeEvaluation, ...] = ()
        merge_selection: BulkMergeSelection | None = None
        merge_validation: RefinementBatchResult | None = None
        merge_accepted = False
        if selected_config.enable_merging:
            merge_candidates = mergeable_parents(current_fit.topology)
            if merge_candidates:
                merge_evaluations = local_four_sibling_merge_lookahead(
                    block,
                    current_fit,
                    train_mask,
                    selected_config.inference,
                    holdout_mask=holdout_mask,
                    candidates=merge_candidates,
                    fixed_gains=fixed_gains,
                    primary_beam=primary_beam,
                    approximation=approximation,
                ).evaluations
            merge_hysteresis = advance_merge_hysteresis(
                merge_hysteresis,
                merge_evaluations,
                just_split=just_split,
                cooldown_rounds=selected_config.merge_cooldown_rounds,
            )
            if merge_evaluations:
                merge_selection = select_bulk_merges(
                    merge_evaluations,
                    merge_hysteresis,
                    len(current_fit.topology.leaves),
                    required_streak=selected_config.merge_required_streak,
                    target_improvement_fraction=(selected_config.merge_target_improvement_fraction),
                    max_merge_fraction=selected_config.max_merge_fraction,
                    max_merges=selected_config.max_merges_per_round,
                )
                if merge_selection.selected:
                    merge_validation = merge_quadtree_batch(
                        block,
                        current_fit,
                        train_mask,
                        holdout_mask,
                        selected_config.inference,
                        merge_selection.selected,
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
                    merge_accepted_attempt = merge_validation.accepted_attempt
                    if merge_accepted_attempt is not None:
                        current_fit = merge_accepted_attempt.fit
                        merge_accepted = True

        rounds.append(
            AdaptiveRefinementRound(
                index=round_index,
                leaf_count_before=leaf_count_before,
                screening_scores=scores,
                selection=selection,
                validation=split_validation,
                merge_evaluations=merge_evaluations,
                merge_selection=merge_selection,
                merge_validation=merge_validation,
            )
        )

        if selection.selected and not split_accepted:
            stop_reason = "split_validation_rejected"
            break
        if merge_selection is not None and merge_selection.selected and not merge_accepted:
            stop_reason = "merge_validation_rejected"
            break
        if not selection.selected and not merge_candidates:
            stop_reason = "no_eligible_changes"
            break

    return HierarchicalImagingResult(
        inference=current_fit,
        rounds=tuple(rounds),
        train_mask=train_mask,
        holdout_mask=holdout_mask,
        stop_reason=stop_reason,
        elapsed_s=perf_counter() - started,
        merge_hysteresis=merge_hysteresis,
    )
