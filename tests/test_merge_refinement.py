from __future__ import annotations

import numpy as np
import pytest

from sl1mjax.data.canonical import VisibilityBlock
from sl1mjax.inference import InferenceConfig, infer_quadtree
from sl1mjax.polarization import Correlation, ReceptorBasis
from sl1mjax.quadtree import (
    QuadtreeLeaf,
    QuadtreeSky,
    predict_quadtree_stokes_i,
    quadtree_sky_from_regular_grid,
)
from sl1mjax.refinement import (
    LocalMergeEvaluation,
    MergeHysteresisState,
    QuadtreeObjectiveMetrics,
    advance_merge_hysteresis,
    compare_merge_lookahead_to_oracle,
    exhaustive_single_merge_oracle,
    local_four_sibling_merge_lookahead,
    merge_quadtree_batch,
    mergeable_parents,
    select_bulk_merges,
)
from sl1mjax.sky import GaussianApproximation


def _block_from_sky(
    sky: QuadtreeSky,
    *,
    rows: int = 96,
    channels: int = 2,
    seed: int = 13,
) -> VisibilityBlock:
    rng = np.random.default_rng(seed)
    uvw_m = rng.uniform(-6_000.0, 6_000.0, size=(rows, 3))
    uvw_m[:, 2] *= 0.2
    frequency_hz = 1.15e9 + np.arange(channels) * 8e6
    antenna1 = rng.integers(0, 5, rows, dtype=np.int32)
    antenna2 = (antenna1 + rng.integers(1, 5, rows, dtype=np.int32)) % 5
    correlations = (Correlation.I,)
    visibility = np.asarray(
        predict_quadtree_stokes_i(
            sky.flux,
            sky.topology,
            uvw_m,
            frequency_hz,
            antenna1,
            antenna2,
            correlations,
            approximation=GaussianApproximation.PARAXIAL,
        )
    )
    return VisibilityBlock(
        uvw_m=uvw_m,
        frequency_hz=frequency_hz,
        visibility=visibility,
        weight=np.ones_like(visibility.real),
        flag=np.zeros_like(visibility.real, dtype=bool),
        time_s=np.arange(rows, dtype=np.float64),
        antenna1=antenna1,
        antenna2=antenna2,
        correlations=correlations,
        receptor_basis=ReceptorBasis.STOKES,
    )


def _smooth_fit(seed: int = 13) -> tuple[VisibilityBlock, np.ndarray, np.ndarray, object]:
    """A parent split into four flux-equal (i.e. truly smooth) children."""

    root_sky = quadtree_sky_from_regular_grid(2, 2e-4, [1.0, 0.3, 0.02, 0.01])
    truth = root_sky.split(QuadtreeLeaf(0, 0, 0), child_flux=[0.25, 0.25, 0.25, 0.25])
    block = _block_from_sky(truth, seed=seed)
    holdout = np.zeros(block.shape, dtype=bool)
    holdout[:16] = True
    train = block.active & ~holdout
    config = InferenceConfig(
        steps=180,
        learning_rate=0.1,
        sparsity_weight=1e-8,
        initial_intensity=0.05,
        patience=220,
        chunk_size=41,
    )
    fit = infer_quadtree(
        block,
        truth.topology,
        train,
        config,
        holdout_mask=holdout,
        approximation=GaussianApproximation.PARAXIAL,
        leaf_penalty=1e-6,
        initial_flux=truth.flux,
    )
    return block, train, holdout, fit


def _structured_fit(seed: int = 13) -> tuple[VisibilityBlock, np.ndarray, np.ndarray, object]:
    """A parent split into four flux-anisotropic (genuinely detailed) children."""

    root_sky = quadtree_sky_from_regular_grid(2, 2e-4, [1.0, 0.3, 0.02, 0.01])
    truth = root_sky.split(QuadtreeLeaf(0, 0, 0), child_flux=[0.7, 0.1, 0.15, 0.05])
    block = _block_from_sky(truth, seed=seed)
    holdout = np.zeros(block.shape, dtype=bool)
    holdout[:16] = True
    train = block.active & ~holdout
    config = InferenceConfig(
        steps=180,
        learning_rate=0.1,
        sparsity_weight=1e-8,
        initial_intensity=0.05,
        patience=220,
        chunk_size=41,
    )
    fit = infer_quadtree(
        block,
        truth.topology,
        train,
        config,
        holdout_mask=holdout,
        approximation=GaussianApproximation.PARAXIAL,
        leaf_penalty=1e-6,
        initial_flux=truth.flux,
    )
    return block, train, holdout, fit


def test_mergeable_parents_requires_a_complete_sibling_group() -> None:
    sky = quadtree_sky_from_regular_grid(2, 2e-4, [1.0, 0.3, 0.02, 0.01])
    parent = QuadtreeLeaf(0, 0, 0)
    complete = sky.split(parent)
    grandchild_parent = QuadtreeLeaf(1, 0, 0)
    # splitting one child further breaks the original group: `parent` no
    # longer has all four children active, but `grandchild_parent` (that
    # child) now has its own complete sibling group underneath it.
    incomplete = complete.split(grandchild_parent)

    assert mergeable_parents(complete.topology) == (parent,)
    assert mergeable_parents(incomplete.topology) == (grandchild_parent,)
    assert parent not in mergeable_parents(incomplete.topology)
    # candidates filter accepts non-mergeable identities and drops them silently
    assert mergeable_parents(complete.topology, candidates=(QuadtreeLeaf(0, 1, 1),)) == ()
    with pytest.raises(ValueError, match="unique"):
        mergeable_parents(complete.topology, candidates=(parent, parent))


def test_local_merge_lookahead_conserves_flux_for_a_smooth_region() -> None:
    block, train, holdout, fit = _smooth_fit()

    lookahead = local_four_sibling_merge_lookahead(
        block,
        fit,
        train,
        InferenceConfig(sparsity_weight=1e-8),
        holdout_mask=holdout,
        approximation=GaussianApproximation.PARAXIAL,
    )

    assert lookahead.best is not None
    evaluation = lookahead.best
    np.testing.assert_allclose(
        sum(evaluation.child_flux), evaluation.parent_flux, rtol=0.0, atol=5e-3
    )
    assert evaluation.predicted_improvement > 0


def test_local_merge_lookahead_rejects_genuine_child_detail() -> None:
    block, train, holdout, fit = _structured_fit()

    lookahead = local_four_sibling_merge_lookahead(
        block,
        fit,
        train,
        InferenceConfig(sparsity_weight=1e-8),
        holdout_mask=holdout,
        approximation=GaussianApproximation.PARAXIAL,
    )

    assert lookahead.best is None
    assert lookahead.evaluations[0].predicted_improvement < 0


def test_merge_lookahead_matches_exhaustive_oracle_on_smooth_region() -> None:
    block, train, holdout, fit = _smooth_fit()
    config = InferenceConfig(
        steps=180,
        learning_rate=0.1,
        sparsity_weight=1e-8,
        initial_intensity=0.05,
        patience=220,
        chunk_size=41,
    )

    lookahead = local_four_sibling_merge_lookahead(
        block,
        fit,
        train,
        config,
        holdout_mask=holdout,
        approximation=GaussianApproximation.PARAXIAL,
    )
    oracle = exhaustive_single_merge_oracle(
        block,
        fit,
        train,
        config,
        holdout_mask=holdout,
        approximation=GaussianApproximation.PARAXIAL,
    )
    comparison = compare_merge_lookahead_to_oracle(lookahead, oracle)

    assert oracle.best is not None
    assert oracle.best.leaf == QuadtreeLeaf(0, 0, 0)
    assert comparison.top1_match
    assert comparison.spearman_rho == pytest.approx(1.0)
    np.testing.assert_allclose(oracle.best.child_flux, (0.25, 0.25, 0.25, 0.25), atol=5e-3)


def test_merge_hysteresis_requires_two_consecutive_eligible_rounds() -> None:
    block, train, holdout, fit = _smooth_fit()
    config = InferenceConfig(sparsity_weight=1e-8)
    lookahead = local_four_sibling_merge_lookahead(
        block,
        fit,
        train,
        config,
        holdout_mask=holdout,
        approximation=GaussianApproximation.PARAXIAL,
    )
    assert lookahead.best is not None
    leaf_count = len(fit.topology.leaves)

    state = MergeHysteresisState.empty()
    state = advance_merge_hysteresis(state, lookahead.evaluations)
    assert state.eligible_streak == {QuadtreeLeaf(0, 0, 0): 1}
    first_round_selection = select_bulk_merges(lookahead.evaluations, state, leaf_count)
    assert first_round_selection.selected == ()

    state = advance_merge_hysteresis(state, lookahead.evaluations)
    assert state.eligible_streak == {QuadtreeLeaf(0, 0, 0): 2}
    second_round_selection = select_bulk_merges(lookahead.evaluations, state, leaf_count)
    assert second_round_selection.selected == (QuadtreeLeaf(0, 0, 0),)


def test_merge_hysteresis_cooldown_blocks_immediate_remerge_after_split() -> None:
    parent = QuadtreeLeaf(0, 0, 0)
    favorable = LocalMergeEvaluation(
        leaf=parent,
        children=parent.children(),
        child_flux=(0.25, 0.25, 0.25, 0.25),
        parent_flux=1.0,
        metrics=QuadtreeObjectiveMetrics(0.0, 0.0, 0.0, 0.0, None),
        objective_change=-1.0,
        predicted_improvement=1.0,
        holdout_change=None,
    )

    state = MergeHysteresisState.empty()
    state = advance_merge_hysteresis(state, (favorable,), just_split=(parent,), cooldown_rounds=2)
    assert state.eligible_streak == {}
    assert state.split_cooldown == {parent: 2}

    state = advance_merge_hysteresis(state, (favorable,))
    assert state.eligible_streak == {}
    assert state.split_cooldown == {parent: 1}

    state = advance_merge_hysteresis(state, (favorable,))
    assert state.eligible_streak == {parent: 1}
    assert state.split_cooldown == {}

    # a favorable score alone, on the first post-cooldown round, is not enough
    selection = select_bulk_merges((favorable,), state, current_leaf_count=10)
    assert selection.selected == ()


def test_select_bulk_merges_orders_deterministically_within_shrink_budget() -> None:
    parents = tuple(QuadtreeLeaf(0, 0, column) for column in range(4))
    evaluations = tuple(
        LocalMergeEvaluation(
            leaf=parent,
            children=parent.children(),
            child_flux=(0.25, 0.25, 0.25, 0.25),
            parent_flux=1.0,
            metrics=QuadtreeObjectiveMetrics(0.0, 0.0, 0.0, 0.0, None),
            objective_change=-improvement,
            predicted_improvement=improvement,
            holdout_change=None,
        )
        for parent, improvement in zip(parents, (4.0, 3.0, 2.0, 1.0), strict=True)
    )
    state = MergeHysteresisState(
        eligible_streak={parent: 2 for parent in parents}, split_cooldown={}
    )

    selection = select_bulk_merges(
        evaluations,
        state,
        current_leaf_count=40,
        target_improvement_fraction=0.7,
        max_merge_fraction=0.05,
    )

    assert selection.selected == parents[:2]
    assert selection.available_improvement == pytest.approx(10.0)
    assert selection.selected_improvement == pytest.approx(7.0)
    assert selection.covered_fraction == pytest.approx(0.7)
    assert selection.merge_budget == 2
    assert selection.removed_leaf_count == 6

    # identical improvements must still resolve deterministically by leaf order
    tied = tuple(
        LocalMergeEvaluation(
            leaf=parent,
            children=parent.children(),
            child_flux=(0.25, 0.25, 0.25, 0.25),
            parent_flux=1.0,
            metrics=QuadtreeObjectiveMetrics(0.0, 0.0, 0.0, 0.0, None),
            objective_change=-1.0,
            predicted_improvement=1.0,
            holdout_change=None,
        )
        for parent in reversed(parents)
    )
    tied_selection = select_bulk_merges(
        tied,
        state,
        current_leaf_count=40,
        target_improvement_fraction=1.0,
        max_merge_fraction=1.0,
        max_merges=4,
    )
    assert tied_selection.selected == parents


def test_select_bulk_merges_gates_on_cooldown_directly() -> None:
    """A positive split_cooldown must block selection even if eligible_streak
    is (inconsistently) already at or above the required streak. Normal use
    through advance_merge_hysteresis never produces that combination, but
    select_bulk_merges must not rely on callers preserving that invariant."""

    parent = QuadtreeLeaf(0, 0, 0)
    evaluation = LocalMergeEvaluation(
        leaf=parent,
        children=parent.children(),
        child_flux=(0.25, 0.25, 0.25, 0.25),
        parent_flux=1.0,
        metrics=QuadtreeObjectiveMetrics(0.0, 0.0, 0.0, 0.0, None),
        objective_change=-1.0,
        predicted_improvement=1.0,
        holdout_change=None,
    )
    inconsistent_state = MergeHysteresisState(
        eligible_streak={parent: 5}, split_cooldown={parent: 1}
    )

    selection = select_bulk_merges(
        (evaluation,), inconsistent_state, current_leaf_count=10, required_streak=2
    )

    assert selection.selected == ()


def test_merge_quadtree_batch_accepts_and_conserves_flux() -> None:
    block, train, holdout, fit = _smooth_fit()
    config = InferenceConfig(
        steps=180,
        learning_rate=0.1,
        sparsity_weight=1e-8,
        initial_intensity=0.05,
        patience=220,
        chunk_size=41,
    )
    total_flux_before = float(np.sum(fit.flux))

    batch = merge_quadtree_batch(
        block,
        fit,
        train,
        holdout,
        config,
        (QuadtreeLeaf(0, 0, 0),),
        approximation=GaussianApproximation.PARAXIAL,
    )

    accepted = batch.accepted_attempt
    assert accepted is not None
    assert accepted.selected == (QuadtreeLeaf(0, 0, 0),)
    assert len(accepted.fit.topology.leaves) == len(fit.topology.leaves) - 3
    np.testing.assert_allclose(float(np.sum(accepted.fit.flux)), total_flux_before, rtol=1e-2)


def test_merge_quadtree_batch_rejects_structured_region() -> None:
    block, train, holdout, fit = _structured_fit()
    config = InferenceConfig(
        steps=180,
        learning_rate=0.1,
        sparsity_weight=1e-8,
        initial_intensity=0.05,
        patience=220,
        chunk_size=41,
    )

    batch = merge_quadtree_batch(
        block,
        fit,
        train,
        holdout,
        config,
        (QuadtreeLeaf(0, 0, 0),),
        approximation=GaussianApproximation.PARAXIAL,
        minimum_training_relative_improvement=0.0,
        minimum_holdout_relative_improvement=0.0,
    )

    assert batch.accepted_attempt is None
    assert len(batch.attempts) == 1


def test_merge_quadtree_batch_rejects_invalid_or_incomplete_candidates() -> None:
    block, train, holdout, fit = _smooth_fit()
    config = InferenceConfig(sparsity_weight=1e-8)

    with pytest.raises(ValueError, match="not mergeable"):
        merge_quadtree_batch(
            block,
            fit,
            train,
            holdout,
            config,
            (QuadtreeLeaf(0, 1, 1),),
        )
    with pytest.raises(ValueError, match="unique"):
        merge_quadtree_batch(
            block,
            fit,
            train,
            holdout,
            config,
            (QuadtreeLeaf(0, 0, 0), QuadtreeLeaf(0, 0, 0)),
        )
