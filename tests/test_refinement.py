from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from sl1mjax.data.canonical import VisibilityBlock
from sl1mjax.direct_operator import DirectDFTConfig
from sl1mjax.inference import (
    InferenceConfig,
    QuadtreeInferenceResult,
    infer_quadtree,
)
from sl1mjax.polarization import Correlation, ReceptorBasis
from sl1mjax.quadtree import (
    QuadtreeLeaf,
    QuadtreeSky,
    predict_quadtree_stokes_i,
    quadtree_sky_from_regular_grid,
)
from sl1mjax.refinement import (
    ResidualHaarScore,
    _solve_nonnegative_quadratic,
    baseline_split_scores,
    batched_residual_haar_scores,
    compare_haar_to_oracle,
    compare_lookahead_to_oracle,
    exhaustive_single_split_oracle,
    local_four_child_lookahead,
    refine_quadtree_batch,
    render_quadtree_surface_brightness,
    residual_haar_scores,
    select_bulk_splits,
    solve_quadtree_flux_active_set,
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


def test_surface_brightness_render_conserves_flux_across_levels() -> None:
    width = 2e-3
    area = width**2
    sky = quadtree_sky_from_regular_grid(2, width, np.full(4, area))
    sky = sky.split(QuadtreeLeaf(0, 0, 0), child_flux=np.full(4, area / 4))

    image = render_quadtree_surface_brightness(sky.topology, sky.flux, level=2)
    render_area = sky.grid.leaf_width_rad(2) ** 2

    np.testing.assert_allclose(image, 1.0)
    np.testing.assert_allclose(image.sum() * render_area, sky.flux.sum())
    scores = baseline_split_scores(sky.topology, sky.flux, render_level=2)
    assert all(score.gradient == pytest.approx(0.0) for score in scores)
    assert all(score.laplacian == pytest.approx(0.0) for score in scores)


def test_baseline_scores_filter_candidates_and_detect_an_edge() -> None:
    sky = quadtree_sky_from_regular_grid(2, 2e-3, [1.0, 0.05, 1.0, 0.05])
    selected = (QuadtreeLeaf(0, 0, 0), QuadtreeLeaf(0, 0, 1))

    scores = baseline_split_scores(
        sky.topology,
        sky.flux,
        candidates=selected,
        max_depth=1,
    )

    assert tuple(score.leaf for score in scores) == selected
    assert scores[0].flux == pytest.approx(1.0)
    assert scores[0].surface_brightness > scores[1].surface_brightness
    assert all(score.gradient > 0 for score in scores)
    assert all(score.laplacian > 0 for score in scores)

    assert baseline_split_scores(
        sky.topology,
        sky.flux,
        candidates=selected,
        max_depth=0,
    ) == ()
    with pytest.raises(ValueError, match="not active"):
        baseline_split_scores(
            sky.topology,
            sky.flux,
            candidates=(QuadtreeLeaf(1, 0, 0),),
        )


def test_four_child_active_set_solver_enforces_positivity_and_total_flux() -> None:
    hessian = 2.0 * np.eye(4)
    linear = np.asarray([-2.0, 1.0, -4.0, 0.5])

    unconstrained_total = _solve_nonnegative_quadratic(
        hessian,
        linear,
        total=None,
    )
    fixed_total = _solve_nonnegative_quadratic(
        hessian,
        linear,
        total=1.0,
    )

    np.testing.assert_allclose(unconstrained_total, [1.0, 0.0, 2.0, 0.0])
    np.testing.assert_allclose(fixed_total, [0.0, 0.0, 1.0, 0.0], atol=1e-12)
    assert fixed_total.sum() == pytest.approx(1.0)


def test_bulk_split_marking_covers_score_mass_with_growth_budget() -> None:
    leaves = tuple(QuadtreeLeaf(0, 0, column) for column in range(4))
    scores = tuple(
        ResidualHaarScore(
            leaf=leaf,
            parent_flux=1.0,
            gradient=(1.0, 0.0, 0.0),
            gram=((1.0, 0.0, 0.0),) * 3,
            eigenvalues=(1.0, 1.0, 1.0),
            eigenvalue_ratio=1.0,
            ridge=0.0,
            raw_predicted_improvement=improvement,
            predicted_improvement=improvement,
            eligible=True,
        )
        for leaf, improvement in zip(leaves, (4.0, 3.0, 2.0, 1.0), strict=True)
    )

    selection = select_bulk_splits(
        scores,
        current_leaf_count=40,
        target_improvement_fraction=0.7,
        max_split_fraction=0.05,
    )

    assert selection.selected == leaves[:2]
    assert selection.available_improvement == pytest.approx(10.0)
    assert selection.selected_improvement == pytest.approx(7.0)
    assert selection.covered_fraction == pytest.approx(0.7)
    assert selection.split_budget == 2
    assert selection.added_leaf_count == 6

    penalized = select_bulk_splits(
        scores,
        current_leaf_count=40,
        max_split_fraction=1.0,
        split_cost=3.5,
    )
    assert penalized.selected == (leaves[0],)
    assert penalized.available_improvement == pytest.approx(0.5)


def test_batched_haar_scores_scale_to_4096_initial_leaves() -> None:
    leaf_count = 64 * 64
    sky = quadtree_sky_from_regular_grid(
        64,
        2e-5,
        np.zeros(leaf_count),
    )
    rows = 6
    visibility = np.linspace(0.1, 0.6, rows)[:, None, None].astype(np.complex128)
    block = VisibilityBlock(
        uvw_m=np.column_stack(
            (
                np.linspace(-4_000.0, 4_000.0, rows),
                np.linspace(3_000.0, -3_000.0, rows),
                np.zeros(rows),
            )
        ),
        frequency_hz=np.asarray([1.2e9]),
        visibility=visibility,
        weight=np.ones_like(visibility.real),
        flag=np.zeros_like(visibility.real, dtype=bool),
        time_s=np.arange(rows, dtype=np.float64),
        antenna1=np.zeros(rows, dtype=np.int32),
        antenna2=np.ones(rows, dtype=np.int32),
        correlations=(Correlation.I,),
        receptor_basis=ReceptorBasis.STOKES,
    )
    fit = QuadtreeInferenceResult(
        topology=sky.topology,
        flux=sky.flux,
        prediction=np.zeros_like(visibility),
        residual=-visibility,
        raw_parameters=np.zeros(leaf_count),
        optimizer_state=None,
        objective_history=(),
        data_history=(),
        prior_history=(),
        holdout_history=(),
        holdout_steps=(),
        leaf_penalty=0.0,
        topology_penalty=0.0,
        best_step=0,
        steps=0,
        converged=True,
    )
    config = InferenceConfig(
        operator_mode="explicit",
        direct_dft=DirectDFTConfig(
            visibility_chunk_size=6,
            pixel_chunk_size=1024,
            max_response_bytes=2 * 1024**2,
        ),
    )

    scores = batched_residual_haar_scores(
        block,
        fit,
        block.active,
        config,
        max_depth=1,
        approximation=GaussianApproximation.PARAXIAL,
    )

    assert len(scores) == leaf_count
    assert all(np.isfinite(score.predicted_improvement) for score in scores)
    selection = select_bulk_splits(
        scores,
        current_leaf_count=leaf_count,
        max_split_fraction=0.01,
    )
    assert selection.split_budget == 41
    assert selection.added_leaf_count <= 3 * 41


def test_active_set_oracle_rejects_splits_of_an_exact_smooth_model() -> None:
    sky = quadtree_sky_from_regular_grid(2, 2e-4, [1.0, 0.3, 0.08, 0.02])
    block = _block_from_sky(sky, rows=48, channels=1)
    config = InferenceConfig(sparsity_weight=0.0)
    fit = solve_quadtree_flux_active_set(
        block,
        sky.topology,
        block.active,
        config,
        approximation=GaussianApproximation.PARAXIAL,
        leaf_penalty=1e-6,
    )

    np.testing.assert_allclose(fit.flux, sky.flux, rtol=1e-11, atol=1e-12)
    assert fit.data_history[0] < 1e-20
    oracle = exhaustive_single_split_oracle(
        block,
        fit,
        block.active,
        config,
        max_depth=1,
        approximation=GaussianApproximation.PARAXIAL,
        solver="active_set",
    )

    assert oracle.best is None
    assert max(
        evaluation.predicted_improvement for evaluation in oracle.evaluations
    ) == pytest.approx(-3e-6, abs=1e-11)
    with pytest.raises(ValueError, match="at most 3 leaves"):
        solve_quadtree_flux_active_set(
            block,
            sky.topology,
            block.active,
            config,
            max_leaves=3,
        )


def test_exhaustive_oracle_finds_faint_structure_over_bright_smooth_flux() -> None:
    bright_smooth = QuadtreeLeaf(0, 0, 0)
    faint_structured = QuadtreeLeaf(0, 0, 1)
    root_sky = quadtree_sky_from_regular_grid(2, 2e-4, [1.0, 0.3, 0.02, 0.01])
    truth = root_sky.split(
        faint_structured,
        child_flux=[0.24, 0.03, 0.02, 0.01],
    )
    block = _block_from_sky(truth)
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
    root_fit = infer_quadtree(
        block,
        root_sky.topology,
        train,
        config,
        approximation=GaussianApproximation.PARAXIAL,
        leaf_penalty=1e-6,
    )

    baseline = baseline_split_scores(root_fit.topology, root_fit.flux)
    strongest = max(baseline, key=lambda score: score.flux)
    assert strongest.leaf == bright_smooth
    assert max(baseline, key=lambda score: score.gradient).leaf == bright_smooth
    assert max(baseline, key=lambda score: score.laplacian).leaf == bright_smooth

    oracle = exhaustive_single_split_oracle(
        block,
        root_fit,
        train,
        config,
        holdout_mask=holdout,
        max_depth=1,
        approximation=GaussianApproximation.PARAXIAL,
    )
    haar = residual_haar_scores(
        block,
        root_fit,
        train,
        config,
        max_depth=1,
        approximation=GaussianApproximation.PARAXIAL,
    )
    comparison = compare_haar_to_oracle(haar, oracle)
    lookahead = local_four_child_lookahead(
        block,
        root_fit,
        train,
        config,
        holdout_mask=holdout,
        max_depth=1,
        approximation=GaussianApproximation.PARAXIAL,
    )
    lookahead_comparison = compare_lookahead_to_oracle(lookahead, oracle)

    assert len(oracle.evaluations) == 4
    assert oracle.best is not None
    assert oracle.best.leaf == faint_structured
    assert oracle.best.predicted_improvement > 0
    assert oracle.best.holdout_change is not None
    assert oracle.best.holdout_change < 0
    assert oracle.ranked[0].leaf == faint_structured
    assert oracle.ranked[0].objective_change == pytest.approx(
        -oracle.ranked[0].predicted_improvement
    )
    bright_evaluation = next(
        evaluation for evaluation in oracle.evaluations if evaluation.leaf == bright_smooth
    )
    assert oracle.best.predicted_improvement > bright_evaluation.predicted_improvement
    assert oracle.best.fit.topology.leaves != root_fit.topology.leaves
    assert max(haar, key=lambda score: score.predicted_improvement).leaf == faint_structured
    assert all(score.eligible for score in haar)
    assert all(score.eigenvalues[0] >= -1e-12 for score in haar)
    assert comparison.top1_match
    assert comparison.spearman_rho > 0
    assert comparison.entries[0].leaf == faint_structured
    assert comparison.entries[0].haar_rank == 1
    assert comparison.entries[0].oracle_rank == 1
    assert lookahead.best is not None
    assert lookahead.best.leaf == faint_structured
    assert all(flux >= 0 for flux in lookahead.best.child_flux)
    assert lookahead.best.objective_change == pytest.approx(
        -lookahead.best.predicted_improvement
    )
    assert lookahead.best.metrics.topology - lookahead.baseline.topology == pytest.approx(
        3.0 * root_fit.leaf_penalty
    )
    assert lookahead.best.metrics.objective == pytest.approx(
        lookahead.best.metrics.training_data
        + lookahead.best.metrics.sparsity
        + lookahead.best.metrics.topology
    )
    assert lookahead.best.predicted_improvement <= oracle.best.predicted_improvement + 1e-12
    assert lookahead.best.holdout_change is not None
    assert lookahead.best.holdout_change < 0
    np.testing.assert_allclose(
        lookahead.best.child_flux,
        [0.24, 0.03, 0.02, 0.01],
        rtol=0,
        atol=5e-3,
    )
    assert lookahead_comparison.top1_match
    assert lookahead_comparison.spearman_rho == pytest.approx(1.0)
    assert lookahead_comparison.spearman_rho >= comparison.spearman_rho
    assert lookahead_comparison.entries[0].lookahead_rank == 1

    proposed_flux = dict(
        zip(root_fit.topology.leaves, root_fit.flux, strict=True)
    )
    del proposed_flux[faint_structured]
    proposed_flux.update(
        zip(lookahead.best.children, lookahead.best.child_flux, strict=True)
    )
    proposed_sky = QuadtreeSky(
        root_fit.topology.grid,
        tuple(proposed_flux),
        np.asarray(list(proposed_flux.values())),
    )
    proposed_prediction = np.asarray(
        predict_quadtree_stokes_i(
            proposed_sky.flux,
            proposed_sky.topology,
            block.uvw_m,
            block.frequency_hz,
            block.antenna1,
            block.antenna2,
            block.correlations,
            approximation=GaussianApproximation.PARAXIAL,
        )
    )
    training_weight = np.where(train, block.weight, 0.0)
    independently_evaluated_data = float(
        np.sum(training_weight * np.abs(proposed_prediction - block.visibility) ** 2)
        / np.sum(training_weight)
    )
    assert lookahead.best.metrics.training_data == pytest.approx(
        independently_evaluated_data,
        rel=1e-12,
    )

    conserving = local_four_child_lookahead(
        block,
        root_fit,
        train,
        config,
        candidates=(faint_structured,),
        approximation=GaussianApproximation.PARAXIAL,
        conserve_parent_flux=True,
    )
    assert conserving.flux_conserving
    assert sum(conserving.evaluations[0].child_flux) == pytest.approx(
        conserving.evaluations[0].parent_flux,
        rel=0,
        abs=1e-12,
    )
    assert conserving.evaluations[0].metrics.sparsity == pytest.approx(
        conserving.baseline.sparsity
    )

    scaled_weight_scores = residual_haar_scores(
        replace(block, weight=7.0 * block.weight),
        root_fit,
        train,
        config,
        max_depth=1,
        approximation=GaussianApproximation.PARAXIAL,
    )
    np.testing.assert_allclose(
        [score.predicted_improvement for score in scaled_weight_scores],
        [score.predicted_improvement for score in haar],
        rtol=1e-12,
    )
    explicit_scores = residual_haar_scores(
        block,
        root_fit,
        train,
        replace(config, operator_mode="explicit"),
        max_depth=1,
        approximation=GaussianApproximation.PARAXIAL,
    )
    np.testing.assert_allclose(
        [score.predicted_improvement for score in explicit_scores],
        [score.predicted_improvement for score in haar],
        rtol=1e-12,
        atol=1e-14,
    )
    batched_scores = batched_residual_haar_scores(
        block,
        root_fit,
        train,
        replace(config, operator_mode="explicit"),
        max_depth=1,
        approximation=GaussianApproximation.PARAXIAL,
    )
    assert tuple(score.leaf for score in batched_scores) == tuple(
        score.leaf for score in explicit_scores
    )
    np.testing.assert_allclose(
        [score.gradient for score in batched_scores],
        [score.gradient for score in explicit_scores],
        rtol=1e-12,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        [score.gram for score in batched_scores],
        [score.gram for score in explicit_scores],
        rtol=1e-12,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        [score.predicted_improvement for score in batched_scores],
        [score.predicted_improvement for score in explicit_scores],
        rtol=1e-12,
        atol=1e-14,
    )
    assert all(score.curvature_mode == "per_level_exact:0" for score in batched_scores)
    with pytest.raises(ValueError, match="shared per-level curvature"):
        batched_residual_haar_scores(
            block,
            root_fit,
            train,
            replace(config, operator_mode="explicit"),
        )

    refinement = refine_quadtree_batch(
        block,
        root_fit,
        train,
        holdout,
        config,
        (faint_structured,),
        approximation=GaussianApproximation.PARAXIAL,
        max_refits=1,
    )
    assert refinement.accepted_attempt is not None
    assert refinement.accepted_attempt.selected == (faint_structured,)
    assert refinement.accepted_attempt.training_relative_improvement > 0
    assert refinement.accepted_attempt.holdout_relative_improvement > 0
    assert len(refinement.accepted_attempt.fit.topology.leaves) == 7

    rejected = refine_quadtree_batch(
        block,
        replace(root_fit, leaf_penalty=1.0, topology_penalty=4.0),
        train,
        holdout,
        replace(config, steps=30, patience=40),
        (bright_smooth, faint_structured),
        approximation=GaussianApproximation.PARAXIAL,
        max_refits=2,
    )
    assert rejected.accepted_attempt is None
    assert tuple(len(attempt.selected) for attempt in rejected.attempts) == (2, 1)
    assert all(not attempt.accepted for attempt in rejected.attempts)

    gated = residual_haar_scores(
        block,
        root_fit,
        train,
        config,
        candidates=(faint_structured,),
        approximation=GaussianApproximation.PARAXIAL,
        min_parent_flux=float(root_fit.flux.max() + 1.0),
    )
    assert gated[0].raw_predicted_improvement > 0
    assert gated[0].predicted_improvement == 0
    assert not gated[0].eligible
