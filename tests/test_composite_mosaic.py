from __future__ import annotations

import numpy as np
import pytest

from sl1mjax.beam import VLAPrimaryBeam
from sl1mjax.composite import (
    MosaicPointComponent,
    MosaicQuadtreeComponent,
    infer_mosaic_composite,
    mosaic_beam_sensitivity_weights,
    predict_mosaic_composite,
)
from sl1mjax.data.canonical import VisibilityBlock
from sl1mjax.direct_operator import DirectDFTConfig
from sl1mjax.inference import InferenceConfig
from sl1mjax.polarization import Correlation, ReceptorBasis
from sl1mjax.quadtree import quadtree_sky_from_regular_grid


def _empty_block(
    phase_centre_rad: tuple[float, float],
    *,
    seed: int,
    rows: int = 72,
) -> VisibilityBlock:
    rng = np.random.default_rng(seed)
    uvw_m = rng.uniform(-2_500.0, 2_500.0, size=(rows, 3))
    uvw_m[:, 2] *= 0.2
    frequency_hz = np.asarray([1.05e9, 1.12e9])
    antenna1 = rng.integers(0, 5, rows, dtype=np.int32)
    antenna2 = (antenna1 + rng.integers(1, 5, rows, dtype=np.int32)) % 5
    shape = (rows, frequency_hz.size, 1)
    return VisibilityBlock(
        uvw_m=uvw_m,
        frequency_hz=frequency_hz,
        visibility=np.zeros(shape, dtype=np.complex128),
        weight=np.ones(shape),
        flag=np.zeros(shape, dtype=bool),
        time_s=np.arange(rows, dtype=np.float64),
        antenna1=antenna1,
        antenna2=antenna2,
        correlations=(Correlation.I,),
        receptor_basis=ReceptorBasis.STOKES,
        phase_centre_rad=phase_centre_rad,
    )


def _with_visibility(block: VisibilityBlock, visibility: np.ndarray) -> VisibilityBlock:
    return VisibilityBlock(
        **{
            **block.__dict__,
            "visibility": visibility,
        }
    )


def _truth_components() -> tuple[
    MosaicQuadtreeComponent, MosaicQuadtreeComponent, MosaicPointComponent
]:
    central = quadtree_sky_from_regular_grid(1, 1.4e-4, [0.7])
    outer = quadtree_sky_from_regular_grid(
        3,
        7e-4,
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.22, 0.0, 0.0],
    )
    return (
        MosaicQuadtreeComponent("central", central.topology, central.flux),
        MosaicQuadtreeComponent("outer", outer.topology, outer.flux),
        MosaicPointComponent(
            "catalogue",
            l_rad=np.asarray([-1.35e-3]),
            m_rad=np.asarray([8.5e-4]),
            flux=np.asarray([0.31]),
        ),
    )


def test_composite_prediction_is_additive_and_accepts_fixed_background() -> None:
    phase_centre = (4.2, -0.3)
    blocks = (_empty_block(phase_centre, seed=2, rows=12),)
    components = _truth_components()
    direct = DirectDFTConfig(visibility_chunk_size=7, pixel_chunk_size=3)

    joint = predict_mosaic_composite(
        blocks,
        components,
        phase_centre,
        config=direct,
    )[0]
    separate = sum(
        (
            predict_mosaic_composite(
                blocks,
                (component,),
                phase_centre,
                config=direct,
            )[0]
            for component in components
        ),
        np.zeros(blocks[0].shape, dtype=np.complex128),
    )
    np.testing.assert_allclose(joint, separate, rtol=2e-13, atol=2e-13)

    background = np.full(blocks[0].shape, 0.03 + 0.02j)
    with_background = predict_mosaic_composite(
        blocks,
        components,
        phase_centre,
        config=direct,
        fixed_predictions=(background,),
    )[0]
    np.testing.assert_allclose(with_background, joint + background)


def test_composite_fista_recovers_quadtree_and_point_groups_with_holdout() -> None:
    phase_centre = (4.2, -0.3)
    pointings = (
        phase_centre,
        (phase_centre[0] + 3.5e-4, phase_centre[1] - 2e-4),
    )
    empty_blocks = tuple(
        _empty_block(pointing, seed=seed) for pointing, seed in zip(pointings, (3, 5), strict=True)
    )
    truth = _truth_components()
    beam = VLAPrimaryBeam(kind="gaussian")
    direct = DirectDFTConfig(visibility_chunk_size=19, pixel_chunk_size=4)
    truth_predictions = predict_mosaic_composite(
        empty_blocks,
        truth,
        phase_centre,
        primary_beam=beam,
        config=direct,
    )
    blocks = tuple(
        _with_visibility(block, prediction)
        for block, prediction in zip(empty_blocks, truth_predictions, strict=True)
    )
    train_masks = []
    holdout_masks = []
    for block in blocks:
        train_rows = np.arange(block.shape[0]) % 4 != 0
        train_masks.append(block.active & train_rows[:, None, None])
        holdout_masks.append(block.active & ~train_rows[:, None, None])

    zero_components = tuple(
        MosaicPointComponent(
            component.name,
            component.l_rad,
            component.m_rad,
            np.zeros_like(component.flux),
        )
        if isinstance(component, MosaicPointComponent)
        else MosaicQuadtreeComponent(
            component.name,
            component.topology,
            np.zeros_like(component.flux),
        )
        for component in truth
    )
    config = InferenceConfig(
        solver="fista",
        operator_mode="explicit",
        steps=220,
        learning_rate=0.04,
        sparsity_weight=1e-9,
        validation_interval=10,
        kkt_tolerance=2e-5,
        direct_dft=direct,
    )
    result = infer_mosaic_composite(
        blocks,
        zero_components,
        tuple(train_masks),
        phase_centre,
        config,
        holdout_masks=tuple(holdout_masks),
        primary_beam=beam,
    )

    for recovered, expected in zip(result.components, truth, strict=True):
        np.testing.assert_allclose(recovered.flux, expected.flux, atol=4e-3)
    assert result.holdout_history
    assert np.isfinite(result.holdout_history[-1])
    assert len(result.component_predictions) == len(truth)
    for block, residual in zip(blocks, result.residuals, strict=True):
        assert np.linalg.norm(residual) < 0.004 * np.linalg.norm(block.visibility)


def test_beam_sensitivity_weights_are_global_and_downweight_outer_atoms() -> None:
    phase_centre = (4.2, -0.3)
    block = _empty_block(phase_centre, seed=8, rows=10)
    components = (
        MosaicPointComponent(
            "central",
            np.asarray([0.0]),
            np.asarray([0.0]),
            np.asarray([0.0]),
        ),
        MosaicPointComponent(
            "outer",
            np.asarray([0.012]),
            np.asarray([0.0]),
            np.asarray([0.0]),
        ),
    )
    weights = mosaic_beam_sensitivity_weights(
        (block,),
        components,
        (block.active,),
        phase_centre,
        primary_beam=VLAPrimaryBeam(kind="gaussian"),
    )

    assert weights[0][0] == pytest.approx(1.0)
    assert 0 < weights[1][0] < 0.2


def test_composite_validates_component_and_mask_shapes() -> None:
    phase_centre = (4.2, -0.3)
    block = _empty_block(phase_centre, seed=4, rows=4)
    component = MosaicPointComponent(
        "source",
        np.asarray([0.0]),
        np.asarray([0.0]),
        np.asarray([0.1]),
    )
    config = InferenceConfig(solver="fista", operator_mode="explicit", steps=1)

    with pytest.raises(ValueError, match="one mask per block"):
        infer_mosaic_composite((block,), (component,), (), phase_centre, config)
    with pytest.raises(ValueError, match="component names must be unique"):
        predict_mosaic_composite(
            (block,),
            (component, component),
            phase_centre,
        )
    with pytest.raises(ValueError, match="one value per atom"):
        predict_mosaic_composite(
            (block,),
            (
                MosaicPointComponent(
                    "bad",
                    np.asarray([0.0, 0.1]),
                    np.asarray([0.0]),
                    np.asarray([0.1, 0.2]),
                ),
            ),
            phase_centre,
        )
