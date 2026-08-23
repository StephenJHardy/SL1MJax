from __future__ import annotations

from dataclasses import replace

import numpy as np

from sl1mjax.data.canonical import VisibilityBlock
from sl1mjax.direct_operator import DirectDFTConfig
from sl1mjax.hierarchical_imaging import (
    AdaptiveRefinementConfig,
    reconstruct_hierarchical,
)
from sl1mjax.inference import InferenceConfig
from sl1mjax.polarization import Correlation, ReceptorBasis
from sl1mjax.quadtree import (
    QuadtreeLeaf,
    predict_quadtree_stokes_i,
    quadtree_sky_from_regular_grid,
)
from sl1mjax.sky import GaussianApproximation


def test_adaptive_refinement_defaults_to_wide_field_kernel() -> None:
    config = AdaptiveRefinementConfig()
    assert config.approximation is GaussianApproximation.WIDE_FIELD
    assert config.allow_approximate_curvature is False


def test_hierarchical_reconstruction_runs_one_validated_refinement_round() -> None:
    target = QuadtreeLeaf(0, 0, 1)
    root = quadtree_sky_from_regular_grid(2, 2e-4, [1.0, 0.3, 0.02, 0.01])
    truth = root.split(target, child_flux=[0.24, 0.03, 0.02, 0.01])
    rng = np.random.default_rng(13)
    rows = 96
    uvw_m = rng.uniform(-6_000.0, 6_000.0, size=(rows, 3))
    uvw_m[:, 2] *= 0.2
    frequency_hz = np.asarray([1.15e9, 1.158e9])
    antenna1 = rng.integers(0, 5, rows, dtype=np.int32)
    antenna2 = (antenna1 + rng.integers(1, 5, rows, dtype=np.int32)) % 5
    correlations = (Correlation.I,)
    visibility = np.asarray(
        predict_quadtree_stokes_i(
            truth.flux,
            truth.topology,
            uvw_m,
            frequency_hz,
            antenna1,
            antenna2,
            correlations,
            approximation=GaussianApproximation.PARAXIAL,
        )
    )
    block = VisibilityBlock(
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
    config = AdaptiveRefinementConfig(
        root_size=2,
        root_pixel_size_rad=2e-4,
        inference=InferenceConfig(
            steps=180,
            learning_rate=0.1,
            sparsity_weight=1e-8,
            initial_intensity=0.05,
            patience=220,
            validation_interval=10,
            operator_mode="explicit",
            direct_dft=DirectDFTConfig(
                visibility_chunk_size=41,
                pixel_chunk_size=32,
            ),
        ),
        split_strategy="random_row",
        split_seed=3,
        max_rounds=1,
        max_depth=1,
        leaf_penalty=1e-6,
        max_split_fraction=0.25,
        max_splits_per_round=1,
        approximation=GaussianApproximation.PARAXIAL,
    )

    result = reconstruct_hierarchical(block, config)

    assert result.stop_reason == "maximum_rounds"
    assert len(result.rounds) == 1
    assert result.rounds[0].selection.selected == (target,)
    assert result.rounds[0].validation is not None
    assert result.rounds[0].validation.accepted_attempt is not None
    assert len(result.inference.topology.leaves) == 7
    assert not np.any(result.train_mask & result.holdout_mask)
    assert np.any(result.train_mask)
    assert np.any(result.holdout_mask)
    # merging defaults on, but a genuinely detailed split has nothing to merge yet
    assert result.rounds[0].merge_selection is not None
    assert result.rounds[0].merge_selection.selected == ()


def test_hierarchical_reconstruction_runs_split_and_merge_in_the_same_round_loop() -> None:
    target = QuadtreeLeaf(0, 0, 1)
    root = quadtree_sky_from_regular_grid(2, 2e-4, [1.0, 0.3, 0.02, 0.01])
    truth = root.split(target, child_flux=[0.24, 0.03, 0.02, 0.01])
    rng = np.random.default_rng(13)
    rows = 96
    uvw_m = rng.uniform(-6_000.0, 6_000.0, size=(rows, 3))
    uvw_m[:, 2] *= 0.2
    frequency_hz = np.asarray([1.15e9, 1.158e9])
    antenna1 = rng.integers(0, 5, rows, dtype=np.int32)
    antenna2 = (antenna1 + rng.integers(1, 5, rows, dtype=np.int32)) % 5
    correlations = (Correlation.I,)
    visibility = np.asarray(
        predict_quadtree_stokes_i(
            truth.flux,
            truth.topology,
            uvw_m,
            frequency_hz,
            antenna1,
            antenna2,
            correlations,
            approximation=GaussianApproximation.PARAXIAL,
        )
    )
    block = VisibilityBlock(
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
    config = AdaptiveRefinementConfig(
        root_size=2,
        root_pixel_size_rad=2e-4,
        inference=InferenceConfig(
            steps=180,
            learning_rate=0.1,
            sparsity_weight=1e-8,
            initial_intensity=0.05,
            patience=220,
            validation_interval=10,
            operator_mode="explicit",
            direct_dft=DirectDFTConfig(
                visibility_chunk_size=41,
                pixel_chunk_size=32,
            ),
        ),
        split_strategy="random_row",
        split_seed=3,
        max_rounds=3,
        max_depth=1,
        leaf_penalty=1e-6,
        max_split_fraction=0.25,
        max_splits_per_round=1,
        enable_merging=True,
        merge_required_streak=2,
        merge_cooldown_rounds=1,
        approximation=GaussianApproximation.PARAXIAL,
    )

    result = reconstruct_hierarchical(block, config)

    assert len(result.rounds) == 3
    first_round = result.rounds[0]
    assert first_round.selection.selected == (target,)
    # the freshly split leaf's sibling group is evaluated immediately but
    # must not be merge-selectable in the same round it was created
    assert first_round.merge_selection is not None
    assert first_round.merge_selection.selected == ()
    assert result.merge_hysteresis is not None

    # disabling merging entirely must skip merge evaluation and selection
    disabled = reconstruct_hierarchical(
        block,
        replace(config, enable_merging=False),
    )
    for round_record in disabled.rounds:
        assert round_record.merge_evaluations == ()
        assert round_record.merge_selection is None
        assert round_record.merge_validation is None


def test_hierarchical_reconstruction_accepts_a_merge() -> None:
    """A single genuine point of detail can still cause the bulk split screen
    to also flag an unrelated, truly flat leaf (sparse uv coverage makes a
    coarse quadtree basis prone to this kind of leakage). Splitting that flat
    leaf is a false positive: it initially clears validation riding along
    with the genuine split, but once the hysteresis streak completes, the
    merge path must coarsen it back while leaving the genuinely detailed
    leaf split."""

    base_flux = [0.0] * 16
    base_flux[0] = 1.0
    strong_leaf = QuadtreeLeaf(0, 0, 0)
    spurious_leaf = QuadtreeLeaf(0, 3, 3)
    root = quadtree_sky_from_regular_grid(4, 2e-4, base_flux)
    truth = root.split(strong_leaf, child_flux=[0.94, 0.03, 0.02, 0.01])
    rng = np.random.default_rng(13)
    rows = 200
    uvw_m = rng.uniform(-6_000.0, 6_000.0, size=(rows, 3))
    uvw_m[:, 2] *= 0.2
    frequency_hz = np.asarray([1.15e9, 1.158e9])
    antenna1 = rng.integers(0, 5, rows, dtype=np.int32)
    antenna2 = (antenna1 + rng.integers(1, 5, rows, dtype=np.int32)) % 5
    correlations = (Correlation.I,)
    visibility = np.asarray(
        predict_quadtree_stokes_i(
            truth.flux,
            truth.topology,
            uvw_m,
            frequency_hz,
            antenna1,
            antenna2,
            correlations,
            approximation=GaussianApproximation.PARAXIAL,
        )
    )
    block = VisibilityBlock(
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
    config = AdaptiveRefinementConfig(
        root_size=4,
        root_pixel_size_rad=2e-4,
        inference=InferenceConfig(
            steps=180,
            learning_rate=0.1,
            sparsity_weight=1e-8,
            initial_intensity=0.05,
            patience=220,
            validation_interval=10,
            operator_mode="explicit",
            direct_dft=DirectDFTConfig(
                visibility_chunk_size=64,
                pixel_chunk_size=32,
            ),
        ),
        split_strategy="random_row",
        split_seed=3,
        max_rounds=3,
        max_depth=1,
        leaf_penalty=1e-4,
        target_improvement_fraction=0.5,
        max_split_fraction=0.5,
        max_splits_per_round=2,
        enable_merging=True,
        merge_required_streak=2,
        merge_cooldown_rounds=1,
        approximation=GaussianApproximation.PARAXIAL,
    )

    result = reconstruct_hierarchical(block, config)

    assert len(result.rounds) == 3
    first_round = result.rounds[0]
    assert set(first_round.selection.selected) == {strong_leaf, spurious_leaf}
    assert first_round.validation is not None
    assert first_round.validation.accepted_attempt is not None

    merge_round = result.rounds[2]
    assert merge_round.merge_selection is not None
    assert merge_round.merge_selection.selected == (spurious_leaf,)
    assert merge_round.merge_validation is not None
    assert merge_round.merge_validation.accepted_attempt is not None

    leaves = set(result.inference.topology.leaves)
    assert spurious_leaf in leaves
    assert all(child not in leaves for child in spurious_leaf.children())
    assert strong_leaf not in leaves
    assert all(child in leaves for child in strong_leaf.children())
