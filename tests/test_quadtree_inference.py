from __future__ import annotations

import numpy as np
import pytest

from sl1mjax.beam import VLAPrimaryBeam, predict_beam_weights
from sl1mjax.data.canonical import VisibilityBlock
from sl1mjax.direct_operator import DirectDFTConfig
from sl1mjax.inference import InferenceConfig, infer_quadtree, infer_regular_grid
from sl1mjax.polarization import Correlation, ReceptorBasis
from sl1mjax.quadtree import (
    QuadtreeLeaf,
    QuadtreeSky,
    predict_quadtree_stokes_i,
    quadtree_sky_from_regular_grid,
)
from sl1mjax.sky import (
    GaussianApproximation,
    RegularGrid,
    SquarePixelBasis,
    raw_from_intensity,
)


def _visibility_block(
    sky: QuadtreeSky,
    *,
    correlations: tuple[Correlation, ...] = (Correlation.I,),
    receptor_basis: ReceptorBasis = ReceptorBasis.STOKES,
    primary_beam: VLAPrimaryBeam | None = None,
    rows: int = 48,
    channels: int = 2,
    seed: int = 3,
) -> VisibilityBlock:
    rng = np.random.default_rng(seed)
    uvw_m = rng.uniform(-3_000.0, 3_000.0, size=(rows, 3))
    uvw_m[:, 2] *= 0.25
    frequency_hz = 1.1e9 + np.arange(channels) * 5e6
    antenna1 = rng.integers(0, 4, rows, dtype=np.int32)
    antenna2 = (antenna1 + rng.integers(1, 4, rows, dtype=np.int32)) % 4
    l, m = sky.centers()
    beam_i, beam_rr, beam_ll = predict_beam_weights(primary_beam, l, m, frequency_hz)
    visibility = np.asarray(
        predict_quadtree_stokes_i(
            sky.flux,
            sky.topology,
            uvw_m,
            frequency_hz,
            antenna1,
            antenna2,
            correlations,
            beam_weights=beam_i,
            beam_weights_rr=beam_rr,
            beam_weights_ll=beam_ll,
        )
    )
    shape = visibility.shape
    return VisibilityBlock(
        uvw_m=uvw_m,
        frequency_hz=frequency_hz,
        visibility=visibility,
        weight=np.ones(shape),
        flag=np.zeros(shape, dtype=bool),
        time_s=np.arange(rows, dtype=np.float64),
        antenna1=antenna1,
        antenna2=antenna2,
        correlations=correlations,
        receptor_basis=receptor_basis,
    )


def test_level_zero_fit_matches_regular_square_grid() -> None:
    sky = quadtree_sky_from_regular_grid(2, 2e-4, [0.7, 0.2, 0.1, 0.4])
    block = _visibility_block(sky, rows=20, channels=1)
    config = InferenceConfig(
        steps=8,
        learning_rate=0.08,
        sparsity_weight=1e-5,
        patience=20,
        chunk_size=17,
    )
    initial_flux = np.asarray([0.12, 0.09, 0.06, 0.15])
    initial_raw = np.asarray(raw_from_intensity(initial_flux))

    regular = infer_regular_grid(
        block,
        RegularGrid(2, 2e-4),
        block.active,
        config,
        pixel_basis=SquarePixelBasis(1.0, GaussianApproximation.WIDE_FIELD),
        initial_raw=initial_raw,
    )
    quadtree = infer_quadtree(
        block,
        sky.topology,
        block.active,
        config,
        initial_flux=initial_flux,
    )

    np.testing.assert_allclose(quadtree.flux, regular.image.ravel(), rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(
        quadtree.objective_history,
        regular.objective_history,
        rtol=1e-13,
        atol=1e-13,
    )
    np.testing.assert_allclose(quadtree.residual, quadtree.prediction - block.visibility)


def test_mixed_depth_fit_recovers_unequal_child_flux() -> None:
    sky = quadtree_sky_from_regular_grid(2, 2e-4, [1.0, 0.15, 0.08, 0.03])
    sky = sky.split(QuadtreeLeaf(0, 0, 0), child_flux=[0.65, 0.2, 0.1, 0.05])
    block = _visibility_block(sky, rows=96, channels=2, seed=5)
    config = InferenceConfig(
        steps=200,
        learning_rate=0.1,
        sparsity_weight=1e-8,
        initial_intensity=0.05,
        patience=220,
        chunk_size=37,
    )

    result = infer_quadtree(
        block,
        sky.topology,
        block.active,
        config,
        leaf_penalty=2e-4,
    )

    assert result.topology == sky.topology
    assert result.topology_penalty == pytest.approx(2e-4 * len(sky.leaves))
    assert result.topology_penalty - 2e-4 * sky.grid.root_size**2 == pytest.approx(3 * 2e-4)
    assert result.data_history[-1] < result.data_history[0] * 1e-3
    assert np.linalg.norm(result.residual) < np.linalg.norm(block.visibility) * 0.02
    np.testing.assert_allclose(result.flux, sky.flux, atol=0.035)
    assert all(value >= result.topology_penalty for value in result.prior_history)


def test_explicit_fit_supports_squinted_beam_and_holdout() -> None:
    sky = quadtree_sky_from_regular_grid(2, 2e-4, [0.8, 0.2, 0.1, 0.05])
    sky = sky.split(QuadtreeLeaf(0, 0, 0), child_flux=[0.4, 0.2, 0.15, 0.05])
    beam = VLAPrimaryBeam(kind="gaussian", apply_squint=True)
    block = _visibility_block(
        sky,
        correlations=(Correlation.RR, Correlation.LL),
        receptor_basis=ReceptorBasis.CIRCULAR,
        primary_beam=beam,
        rows=16,
        channels=2,
        seed=9,
    )
    holdout = np.zeros(block.shape, dtype=bool)
    holdout[:4] = True
    train = block.active & ~holdout
    config = InferenceConfig(
        steps=4,
        learning_rate=0.08,
        patience=8,
        validation_interval=2,
        operator_mode="explicit",
        direct_dft=DirectDFTConfig(
            visibility_chunk_size=5,
            pixel_chunk_size=3,
            precision="float32",
        ),
    )

    result = infer_quadtree(
        block,
        sky.topology,
        train,
        config,
        holdout_mask=holdout,
        primary_beam=beam,
        initial_flux=np.full(len(sky.leaves), 0.1),
    )

    assert result.raw_parameters.dtype == np.float32
    assert result.holdout_steps == (2, 4)
    assert len(result.holdout_history) == 2
    assert np.all(np.isfinite(result.objective_history))
    assert np.all(np.isfinite(result.holdout_history))
    assert result.prediction.shape == block.shape


def test_quadtree_inference_validates_tree_specific_inputs() -> None:
    sky = quadtree_sky_from_regular_grid(2, 2e-4, [1.0, 0.0, 0.0, 0.0])
    block = _visibility_block(sky, rows=4, channels=1)

    with pytest.raises(ValueError, match="smoothness_weight"):
        infer_quadtree(
            block,
            sky.topology,
            block.active,
            InferenceConfig(steps=1, smoothness_weight=1e-3),
        )
    with pytest.raises(ValueError, match="either initial_flux or initial_raw"):
        infer_quadtree(
            block,
            sky.topology,
            block.active,
            InferenceConfig(steps=1),
            initial_flux=sky.flux,
            initial_raw=np.zeros(len(sky.leaves)),
        )
    with pytest.raises(ValueError, match="one value per topology leaf"):
        infer_quadtree(
            block,
            sky.topology,
            block.active,
            InferenceConfig(steps=1),
            initial_flux=np.ones(len(sky.leaves) - 1),
        )
    with pytest.raises(ValueError, match="leaf_penalty"):
        infer_quadtree(
            block,
            sky.topology,
            block.active,
            InferenceConfig(steps=1),
            leaf_penalty=-1.0,
        )
