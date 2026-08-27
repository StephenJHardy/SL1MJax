from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from sl1mjax.data.canonical import VisibilityBlock
from sl1mjax.direct_operator import DirectDFTConfig
from sl1mjax.polarization import Correlation, ReceptorBasis
from sl1mjax.quadtree import (
    QuadtreeGrid,
    QuadtreeTopology,
    predict_quadtree_stokes_i_explicit,
)
from sl1mjax.sky_recovery import (
    blind_search_quadtree_sky_variation,
    blind_search_sky_variation,
    blind_search_spatial_sky_variation,
    fit_real_sky_component,
    inject_sky_component,
    native_variation_candidates,
    spectral_support_mask,
    split_native_baselines,
    split_search_baselines,
    temporal_interval_candidates,
    temporal_support_mask,
)


def _block() -> tuple[VisibilityBlock, np.ndarray]:
    baselines = [(0, 1), (0, 2), (1, 2), (1, 3)]
    times = np.arange(6, dtype=np.float64) * 10.0
    antenna1 = np.tile([pair[0] for pair in baselines], times.size)
    antenna2 = np.tile([pair[1] for pair in baselines], times.size)
    time_s = np.repeat(times, len(baselines))
    rows = time_s.size
    channels = 4
    phase = (
        np.arange(rows, dtype=np.float64)[:, None, None] * 0.13
        + np.arange(channels, dtype=np.float64)[None, :, None] * 0.31
    )
    response = np.exp(1j * phase)
    shape = (rows, channels, 1)
    block = VisibilityBlock(
        uvw_m=np.zeros((rows, 3)),
        frequency_hz=4.5e9 + np.arange(channels) * 2e6,
        visibility=np.zeros(shape, dtype=np.complex128),
        model_visibility=np.zeros(shape, dtype=np.complex128),
        weight=np.ones(shape),
        flag=np.zeros(shape, dtype=bool),
        time_s=time_s,
        interval_s=np.full(rows, 10.0),
        antenna1=antenna1,
        antenna2=antenna2,
        correlations=(Correlation.RR,),
        receptor_basis=ReceptorBasis.CIRCULAR,
    )
    return block, response


def test_baseline_holdout_is_disjoint_complete_and_deterministic() -> None:
    block, _ = _block()
    first = split_native_baselines(block, evaluation_fraction=0.25, seed=17)
    second = split_native_baselines(block, evaluation_fraction=0.25, seed=17)

    assert set(first.discovery_baselines).isdisjoint(first.evaluation_baselines)
    assert len(first.discovery_baselines) == 3
    assert len(first.evaluation_baselines) == 1
    np.testing.assert_array_equal(first.discovery_mask | first.evaluation_mask, block.active)
    assert not np.any(first.discovery_mask & first.evaluation_mask)
    np.testing.assert_array_equal(first.evaluation_mask, second.evaluation_mask)


def test_temporal_injection_recovers_amplitude_on_unseen_baseline() -> None:
    block, response = _block()
    holdout = split_native_baselines(block, evaluation_fraction=0.25, seed=3)
    support = temporal_support_mask(block, start_s=20.0, duration_s=20.0)
    injected = inject_sky_component(block, response, support, 2.5)

    fit = fit_real_sky_component(
        injected,
        response,
        support,
        holdout.discovery_mask,
        holdout.evaluation_mask,
    )

    assert fit.coefficient == pytest.approx(2.5)
    assert fit.discovery_sample_count == 2 * 3 * 4
    assert fit.evaluation_sample_count == 2 * 1 * 4
    assert fit.component_supported_evaluation.weighted_complex_mse == pytest.approx(0.0)
    assert fit.supported_evaluation_relative_improvement == pytest.approx(1.0)


def test_spectral_injection_beats_static_component_on_held_out_data() -> None:
    block, response = _block()
    holdout = split_native_baselines(block, evaluation_fraction=0.5, seed=5)
    spectral = spectral_support_mask(block, first_channel=1, channel_count=1)
    injected = inject_sky_component(block, response, spectral, 1.75)

    recovered = fit_real_sky_component(
        injected,
        response,
        spectral,
        holdout.discovery_mask,
        holdout.evaluation_mask,
    )
    static = fit_real_sky_component(
        injected,
        response,
        np.ones(block.shape, dtype=bool),
        holdout.discovery_mask,
        holdout.evaluation_mask,
    )

    assert recovered.coefficient == pytest.approx(1.75)
    assert recovered.component_evaluation.weighted_complex_mse == pytest.approx(0.0)
    assert static.coefficient == pytest.approx(1.75 / 4.0)
    assert (
        static.component_evaluation.weighted_complex_mse
        > recovered.component_evaluation.weighted_complex_mse
    )


def test_signed_component_can_recover_a_decrement() -> None:
    block, response = _block()
    holdout = split_native_baselines(block, evaluation_fraction=0.25, seed=0)
    support = temporal_support_mask(block, start_s=0.0, duration_s=10.0)
    injected = inject_sky_component(block, response, support, -0.4)

    signed = fit_real_sky_component(
        injected,
        response,
        support,
        holdout.discovery_mask,
        holdout.evaluation_mask,
    )
    positive = fit_real_sky_component(
        injected,
        response,
        support,
        holdout.discovery_mask,
        holdout.evaluation_mask,
        nonnegative=True,
    )

    assert signed.coefficient == pytest.approx(-0.4)
    assert positive.coefficient == 0.0


def test_support_validation_rejects_bad_ranges() -> None:
    block, _ = _block()
    with pytest.raises(ValueError, match="exceeds"):
        spectral_support_mask(block, first_channel=3, channel_count=2)
    with pytest.raises(ValueError, match="duration"):
        temporal_support_mask(block, start_s=0.0, duration_s=0.0)


def test_three_way_baseline_search_split_is_disjoint_and_complete() -> None:
    block, _ = _block()
    split = split_search_baselines(
        block,
        selection_fraction=0.25,
        evaluation_fraction=0.25,
        seed=12,
    )

    assert len(split.discovery_baselines) == 2
    assert len(split.selection_baselines) == 1
    assert len(split.evaluation_baselines) == 1
    assert not np.any(split.discovery_mask & split.selection_mask)
    assert not np.any(split.discovery_mask & split.evaluation_mask)
    assert not np.any(split.selection_mask & split.evaluation_mask)
    np.testing.assert_array_equal(
        split.discovery_mask | split.selection_mask | split.evaluation_mask,
        block.active,
    )


def test_temporal_candidates_do_not_cross_observation_gap() -> None:
    block, _ = _block()
    gapped_times = np.asarray([0.0, 10.0, 20.0, 100.0, 110.0, 120.0])
    gapped = replace(block, time_s=np.repeat(gapped_times, 4))

    candidates = temporal_interval_candidates(gapped, (1, 3, 4))

    assert len([item for item in candidates if item.bin_count == 3]) == 2
    assert not any(item.bin_count == 4 for item in candidates)


def test_blind_search_recovers_temporal_support_on_sealed_baselines() -> None:
    block, response = _block()
    support = temporal_support_mask(block, start_s=20.0, duration_s=20.0)
    injected = inject_sky_component(block, response, support, 1.5)
    split = split_search_baselines(
        injected,
        selection_fraction=0.25,
        evaluation_fraction=0.25,
        seed=4,
    )
    candidates = native_variation_candidates(
        injected,
        temporal_widths=(1, 2, 3),
        spectral_widths=(1, 2),
    )

    result = blind_search_sky_variation(
        injected,
        response,
        candidates,
        split,
        shortlist_size=12,
    )

    assert result.selected_candidate is not None
    assert result.selected_candidate.kind == "temporal_interval"
    assert result.selected_candidate.coordinate_start == 20.0
    assert result.selected_candidate.coordinate_stop == 40.0
    assert result.refit_variation_coefficient == pytest.approx(1.5)
    assert result.evaluation_candidate_weighted_mse == pytest.approx(0.0, abs=1e-12)
    assert result.accepted


def test_blind_search_recovers_spectral_support_and_slope() -> None:
    block, response = _block()
    split = split_search_baselines(
        block,
        selection_fraction=0.25,
        evaluation_fraction=0.25,
        seed=7,
    )
    candidates = native_variation_candidates(
        block,
        temporal_widths=(1, 2),
        spectral_widths=(1, 2),
    )
    line = spectral_support_mask(block, first_channel=1, channel_count=2)
    line_result = blind_search_sky_variation(
        inject_sky_component(block, response, line, 0.8),
        response,
        candidates,
        split,
        shortlist_size=12,
    )

    assert line_result.selected_candidate is not None
    assert line_result.selected_candidate.kind == "spectral_interval"
    assert line_result.selected_candidate.start_index == 1
    assert line_result.selected_candidate.bin_count == 2
    assert line_result.refit_variation_coefficient == pytest.approx(0.8)

    reference = float(np.exp(np.mean(np.log(block.frequency_hz))))
    multiplier = np.log(block.frequency_hz / reference)[None, :, None]
    sloped = replace(block, visibility=block.visibility + 0.7 * response * multiplier)
    slope_result = blind_search_sky_variation(
        sloped,
        response,
        candidates,
        split,
        shortlist_size=12,
    )
    assert slope_result.selected_candidate is not None
    assert slope_result.selected_candidate.kind == "spectral_slope"
    assert slope_result.refit_variation_coefficient == pytest.approx(0.7)


def test_evaluation_data_cannot_change_blind_candidate_selection() -> None:
    block, response = _block()
    support = temporal_support_mask(block, start_s=10.0, duration_s=20.0)
    injected = inject_sky_component(block, response, support, 1.0)
    split = split_search_baselines(
        injected,
        selection_fraction=0.25,
        evaluation_fraction=0.25,
        seed=19,
    )
    candidates = native_variation_candidates(
        injected,
        temporal_widths=(1, 2, 3),
        spectral_widths=(1, 2),
    )
    reference = blind_search_sky_variation(
        injected, response, candidates, split, shortlist_size=10
    )
    evaluation_corruption = np.where(
        split.evaluation_mask,
        100.0 * response * spectral_support_mask(block, first_channel=3, channel_count=1),
        0.0,
    )
    corrupted = replace(
        injected,
        visibility=injected.visibility + evaluation_corruption,
    )
    changed = blind_search_sky_variation(
        corrupted, response, candidates, split, shortlist_size=10
    )

    assert changed.selected_candidate == reference.selected_candidate
    assert changed.selection_incremental_weighted_mse == pytest.approx(
        reference.selection_incremental_weighted_mse
    )
    assert changed.evaluation_candidate_weighted_mse != pytest.approx(
        reference.evaluation_candidate_weighted_mse
    )


def test_blind_search_rejects_exact_null() -> None:
    block, response = _block()
    split = split_search_baselines(
        block,
        selection_fraction=0.25,
        evaluation_fraction=0.25,
        seed=2,
    )
    candidates = native_variation_candidates(
        block,
        temporal_widths=(1, 2),
        spectral_widths=(1, 2),
    )

    result = blind_search_sky_variation(
        block, response, candidates, split, shortlist_size=8
    )

    assert result.selected_candidate is None
    assert not result.accepted
    assert result.evaluation_incremental_weighted_mse == 0.0


def test_spatial_search_recovers_leaf_and_interval_without_evaluation_leakage() -> None:
    block, response = _block()
    leaves = QuadtreeGrid(2, 1e-3).root_leaves()
    row_phase = np.arange(block.shape[0], dtype=np.float64)[:, None, None]
    channel_phase = np.arange(block.shape[1], dtype=np.float64)[None, :, None]
    responses = tuple(
        response * np.exp(1j * index * (0.071 * row_phase + 0.19 * channel_phase))
        for index in range(len(leaves))
    )
    variations = native_variation_candidates(
        block,
        temporal_widths=(1, 2, 3),
        spectral_widths=(1, 2),
    )
    true_variation = next(
        candidate
        for candidate in variations
        if candidate.kind == "temporal_interval"
        and candidate.coordinate_start == 20.0
        and candidate.bin_count == 2
    )
    support = temporal_support_mask(block, start_s=20.0, duration_s=20.0)
    injected = inject_sky_component(block, responses[2], support, 1.5)
    split = split_search_baselines(
        injected,
        selection_fraction=0.25,
        evaluation_fraction=0.25,
        seed=31,
    )

    result = blind_search_spatial_sky_variation(
        injected,
        leaves,
        responses,
        variations,
        split,
        spatial_shortlist_size=3,
        discovery_shortlist_size=12,
    )

    assert result.selected_leaf == leaves[2]
    assert result.selected_variation == true_variation
    assert result.refit_variation_coefficient == pytest.approx(1.5)
    assert result.accepted

    corruption = np.where(split.evaluation_mask, 100.0 * responses[0], 0.0)
    corrupted = replace(injected, visibility=injected.visibility + corruption)
    changed = blind_search_spatial_sky_variation(
        corrupted,
        leaves,
        responses,
        variations,
        split,
        spatial_shortlist_size=3,
        discovery_shortlist_size=12,
    )
    assert changed.selected_leaf == result.selected_leaf
    assert changed.selected_variation == result.selected_variation
    assert changed.selection_incremental_weighted_mse == pytest.approx(
        result.selection_incremental_weighted_mse
    )
    assert changed.evaluation_candidate_weighted_mse != pytest.approx(
        result.evaluation_candidate_weighted_mse
    )


def test_streamed_quadtree_search_matches_materialized_responses() -> None:
    block, _ = _block()
    rows = block.shape[0]
    uvw_m = np.column_stack(
        (
            np.linspace(-120.0, 180.0, rows),
            np.linspace(90.0, -70.0, rows),
            np.linspace(-30.0, 45.0, rows),
        )
    )
    block = replace(block, uvw_m=uvw_m)
    grid = QuadtreeGrid(2, 8e-4)
    topology = QuadtreeTopology(grid, grid.root_leaves())
    direct = DirectDFTConfig(
        visibility_chunk_size=8,
        pixel_chunk_size=4,
        precision="float64",
    )
    responses = tuple(
        np.asarray(
            predict_quadtree_stokes_i_explicit(
                np.eye(len(topology.leaves))[index],
                topology,
                block.uvw_m,
                block.frequency_hz,
                block.antenna1,
                block.antenna2,
                block.correlations,
                config=direct,
            )
        )
        for index in range(len(topology.leaves))
    )
    variations = native_variation_candidates(
        block,
        temporal_widths=(1, 2),
        spectral_widths=(1, 2),
    )
    true_variation = next(
        candidate
        for candidate in variations
        if candidate.kind == "spectral_interval"
        and candidate.start_index == 1
        and candidate.bin_count == 2
    )
    support = spectral_support_mask(block, first_channel=1, channel_count=2)
    injected = inject_sky_component(block, responses[3], support, 0.9)
    split = split_search_baselines(
        injected,
        selection_fraction=0.25,
        evaluation_fraction=0.25,
        seed=9,
    )
    materialized = blind_search_spatial_sky_variation(
        injected,
        topology.leaves,
        responses,
        variations,
        split,
        spatial_shortlist_size=4,
        discovery_shortlist_size=16,
    )
    streamed = blind_search_quadtree_sky_variation(
        injected,
        topology,
        topology.leaves,
        variations,
        split,
        direct_config=direct,
        spatial_shortlist_size=4,
        discovery_shortlist_size=16,
        candidate_batch_size=3,
        row_batch_size=7,
    )

    assert streamed.selected_leaf == materialized.selected_leaf == topology.leaves[3]
    assert streamed.selected_variation == materialized.selected_variation == true_variation
    assert streamed.refit_static_coefficient == pytest.approx(
        materialized.refit_static_coefficient,
        abs=1e-10,
    )
    assert streamed.refit_variation_coefficient == pytest.approx(
        materialized.refit_variation_coefficient,
        abs=1e-10,
    )
    assert streamed.evaluation_candidate_weighted_mse == pytest.approx(
        materialized.evaluation_candidate_weighted_mse,
        abs=1e-10,
    )
