from __future__ import annotations

import numpy as np
import pytest

import sl1mjax.mosaic_refinement as mosaic_refinement
from sl1mjax.beam import VLAPrimaryBeam, predict_beam_weights
from sl1mjax.coordinates import lmn_to_radec, radec_to_lmn
from sl1mjax.data.canonical import VisibilityBlock
from sl1mjax.direct_operator import DirectDFTConfig
from sl1mjax.inference import (
    InferenceConfig,
    infer_mosaic_quadtree,
    infer_quadtree,
    predict_mosaic_quadtree,
)
from sl1mjax.mosaic_refinement import (
    batched_exact_mosaic_residual_haar_scores,
    mosaic_quadtree_objective_metrics,
    refine_mosaic_quadtree_batch,
)
from sl1mjax.polarization import Correlation, ReceptorBasis
from sl1mjax.quadtree import (
    QuadtreeLeaf,
    QuadtreeSky,
    predict_quadtree_stokes_i,
    quadtree_sky_from_regular_grid,
)
from sl1mjax.refinement import batched_exact_residual_haar_scores


def _block(
    sky: QuadtreeSky,
    mosaic_phase_centre: tuple[float, float],
    pointing_phase_centre: tuple[float, float],
    *,
    seed: int,
    beam: VLAPrimaryBeam | None = None,
) -> VisibilityBlock:
    rng = np.random.default_rng(seed)
    rows = 48
    uvw_m = rng.uniform(-2500.0, 2500.0, size=(rows, 3))
    uvw_m[:, 2] *= 0.2
    frequency_hz = np.asarray([1.1e9, 1.105e9])
    antenna1 = rng.integers(0, 4, rows, dtype=np.int32)
    antenna2 = (antenna1 + rng.integers(1, 4, rows, dtype=np.int32)) % 4
    global_l, global_m = sky.centers()
    ra, dec = lmn_to_radec(*mosaic_phase_centre, global_l, global_m)
    local_l, local_m, _ = radec_to_lmn(*pointing_phase_centre, ra, dec)
    beam_i, _, _ = predict_beam_weights(beam, local_l, local_m, frequency_hz)
    visibility = np.asarray(
        predict_quadtree_stokes_i(
            sky.flux,
            sky.topology,
            uvw_m,
            frequency_hz,
            antenna1,
            antenna2,
            (Correlation.I,),
            centers_lm=(local_l, local_m),
            beam_weights=beam_i,
        )
    )
    return VisibilityBlock(
        uvw_m=uvw_m,
        frequency_hz=frequency_hz,
        visibility=visibility,
        weight=np.ones(visibility.shape),
        flag=np.zeros(visibility.shape, dtype=bool),
        time_s=np.arange(rows, dtype=np.float64),
        antenna1=antenna1,
        antenna2=antenna2,
        correlations=(Correlation.I,),
        receptor_basis=ReceptorBasis.STOKES,
        phase_centre_rad=pointing_phase_centre,
    )


def _config() -> InferenceConfig:
    return InferenceConfig(
        solver="fista",
        steps=180,
        sparsity_weight=1e-8,
        validation_interval=10,
        kkt_tolerance=2e-5,
        operator_mode="explicit",
        direct_dft=DirectDFTConfig(
            visibility_chunk_size=13,
            pixel_chunk_size=8,
        ),
    )


def test_fixed_mosaic_prediction_does_not_require_a_fit() -> None:
    sky = quadtree_sky_from_regular_grid(2, 2e-4, [0.8, 0.2, 0.1, 0.05])
    phase_centre = (1.2, -0.3)
    beam = VLAPrimaryBeam(kind="gaussian")
    blocks = (
        _block(sky, phase_centre, phase_centre, seed=20, beam=beam),
        _block(
            sky,
            phase_centre,
            (phase_centre[0] + 7e-4, phase_centre[1] - 5e-4),
            seed=21,
            beam=beam,
        ),
    )

    predictions = predict_mosaic_quadtree(
        blocks,
        sky.topology,
        sky.flux,
        phase_centre,
        primary_beam=beam,
        config=DirectDFTConfig(visibility_chunk_size=13, pixel_chunk_size=8),
    )

    for block, prediction in zip(blocks, predictions, strict=True):
        np.testing.assert_allclose(prediction, block.visibility, rtol=2e-12, atol=2e-12)


def test_exact_mosaic_scores_match_a_duplicated_single_pointing() -> None:
    parent = QuadtreeLeaf(0, 0, 0)
    base = quadtree_sky_from_regular_grid(2, 2e-4, [0.8, 0.2, 0.1, 0.05])
    truth = base.split(parent, child_flux=[0.55, 0.15, 0.08, 0.02])
    phase_centre = (1.2, -0.3)
    block = _block(truth, phase_centre, phase_centre, seed=8)
    config = _config()
    single = infer_quadtree(block, base.topology, block.active, config)
    mosaic = infer_mosaic_quadtree(
        (block, block),
        base.topology,
        (block.active, block.active),
        phase_centre,
        config,
    )

    single_scores = batched_exact_residual_haar_scores(
        block,
        single,
        block.active,
        config,
        candidate_batch_size=2,
        row_batch_size=17,
    )
    mosaic_scores = batched_exact_mosaic_residual_haar_scores(
        (block, block),
        mosaic,
        (block.active, block.active),
        config,
        candidate_batch_size=2,
        row_batch_size=17,
    )

    np.testing.assert_allclose(mosaic.flux, single.flux, rtol=2e-5, atol=2e-6)
    for single_score, mosaic_score in zip(single_scores, mosaic_scores, strict=True):
        assert mosaic_score.leaf == single_score.leaf
        np.testing.assert_allclose(mosaic_score.gradient, single_score.gradient, rtol=2e-5)
        np.testing.assert_allclose(mosaic_score.gram, single_score.gram, rtol=2e-5)
        np.testing.assert_allclose(
            mosaic_score.predicted_improvement,
            single_score.predicted_improvement,
            rtol=5e-5,
            atol=1e-10,
        )


def test_joint_refit_accepts_a_real_split_seen_by_two_beams() -> None:
    parent = QuadtreeLeaf(0, 0, 0)
    base = quadtree_sky_from_regular_grid(2, 2e-4, [0.8, 0.2, 0.1, 0.05])
    truth = base.split(parent, child_flux=[0.55, 0.15, 0.08, 0.02])
    phase_centre = (1.2, -0.3)
    beam = VLAPrimaryBeam(kind="gaussian")
    blocks = (
        _block(truth, phase_centre, phase_centre, seed=10, beam=beam),
        _block(
            truth,
            phase_centre,
            (phase_centre[0] + 7e-4, phase_centre[1] - 5e-4),
            seed=11,
            beam=beam,
        ),
    )
    train_masks = []
    holdout_masks = []
    for block in blocks:
        train = block.active.copy()
        train[::4] = False
        train_masks.append(train)
        holdout_masks.append(block.active & ~train)
    train_masks_tuple = tuple(train_masks)
    holdout_masks_tuple = tuple(holdout_masks)
    config = _config()
    fit = infer_mosaic_quadtree(
        blocks,
        base.topology,
        train_masks_tuple,
        phase_centre,
        config,
        holdout_masks=holdout_masks_tuple,
        primary_beam=beam,
    )

    result = refine_mosaic_quadtree_batch(
        blocks,
        fit,
        train_masks_tuple,
        holdout_masks_tuple,
        config,
        (parent,),
        primary_beam=beam,
    )

    accepted = result.accepted_attempt
    assert accepted is not None
    assert accepted.training_relative_improvement > 0
    assert accepted.holdout_relative_improvement > 0
    assert len(accepted.fit.topology.leaves) == len(base.leaves) + 3
    metrics = mosaic_quadtree_objective_metrics(
        blocks,
        accepted.fit,
        train_masks_tuple,
        config,
        holdout_masks=holdout_masks_tuple,
    )
    assert metrics.objective < result.baseline.objective
    assert metrics.holdout_data is not None
    assert result.baseline.holdout_data is not None
    assert metrics.holdout_data < result.baseline.holdout_data


def test_joint_refit_backtracks_after_one_numerical_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_parent = QuadtreeLeaf(0, 0, 0)
    second_parent = QuadtreeLeaf(0, 0, 1)
    base = quadtree_sky_from_regular_grid(2, 2e-4, [0.8, 0.2, 0.1, 0.05])
    truth = base.split(first_parent, child_flux=[0.55, 0.15, 0.08, 0.02])
    phase_centre = (1.2, -0.3)
    block = _block(truth, phase_centre, phase_centre, seed=14)
    train = block.active.copy()
    train[::4] = False
    holdout = block.active & ~train
    config = _config()
    fit = infer_mosaic_quadtree(
        (block,),
        base.topology,
        (train,),
        phase_centre,
        config,
        holdout_masks=(holdout,),
    )

    original_infer = mosaic_refinement.infer_mosaic_quadtree
    call_count = 0

    def fail_once(*args: object, **kwargs: object):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("synthetic backtracking failure")
        return original_infer(*args, **kwargs)

    monkeypatch.setattr(mosaic_refinement, "infer_mosaic_quadtree", fail_once)
    result = refine_mosaic_quadtree_batch(
        (block,),
        fit,
        (train,),
        (holdout,),
        config,
        (first_parent, second_parent),
    )

    assert len(result.failures) == 1
    assert result.failures[0].selected == (first_parent, second_parent)
    assert "synthetic backtracking failure" in result.failures[0].error
    assert call_count == 2
    assert result.attempts[0].selected == (first_parent,)
    assert result.accepted_attempt is not None
