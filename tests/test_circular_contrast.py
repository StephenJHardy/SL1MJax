from __future__ import annotations

import numpy as np
import pytest

from sl1mjax.circular_contrast import (
    CIRCULAR_CONTRAST_LIMIT,
    apply_global_circular_contrast,
    circular_contrast_response,
    clip_circular_contrast,
    correlation_residual_power,
    fit_global_circular_contrast,
    parallel_hand_intensities,
)
from sl1mjax.composite import MosaicPointComponent, infer_mosaic_composite, predict_mosaic_composite
from sl1mjax.data.canonical import VisibilityBlock
from sl1mjax.direct_operator import DirectDFTConfig, predict_stokes_i_explicit
from sl1mjax.inference import InferenceConfig
from sl1mjax.polarization import Correlation, ReceptorBasis
from sl1mjax.quadtree import predict_quadtree_stokes_i_explicit, quadtree_sky_from_regular_grid
from sl1mjax.rime import predict_stokes_i
from sl1mjax.sky import GaussianApproximation, RegularGrid, SquarePixelBasis


def _circular_geometry() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[Correlation, ...],
]:
    grid = RegularGrid(3, 2e-3)
    l, m = grid.coordinates
    intensity = np.asarray(
        [0.4, 0.0, 0.2, 0.0, 1.1, 0.0, 0.3, 0.0, 0.5], dtype=np.float64
    )
    uvw_m = np.asarray(
        [
            [12.0, -9.0, 3.0],
            [-18.0, 7.0, 11.0],
            [25.0, 14.0, -8.0],
            [-6.0, -21.0, 5.0],
        ]
    )
    frequency_hz = np.asarray([1.42e9, 1.428e9])
    antenna1 = np.asarray([0, 0, 1, 2], dtype=np.int32)
    antenna2 = np.asarray([1, 2, 3, 3], dtype=np.int32)
    correlations = (Correlation.RR, Correlation.LL)
    return intensity, l, m, uvw_m, frequency_hz, antenna1, antenna2, correlations


def test_default_solver_is_hybrid() -> None:
    assert InferenceConfig().solver == "hybrid"


def test_parallel_hand_intensities_are_non_negative_inside_the_unit_box() -> None:
    intensity = np.asarray([2.0, 0.5, 0.0])
    rr, ll = parallel_hand_intensities(intensity, 0.4)
    np.testing.assert_allclose(np.asarray(rr), [2.8, 0.7, 0.0])
    np.testing.assert_allclose(np.asarray(ll), [1.2, 0.3, 0.0])
    clipped = clip_circular_contrast(np.asarray([1.5, -2.0, 0.2]))
    np.testing.assert_allclose(np.asarray(clipped), [1.0, -1.0, 0.2])
    rr_edge, ll_edge = parallel_hand_intensities(intensity, CIRCULAR_CONTRAST_LIMIT)
    np.testing.assert_allclose(np.asarray(rr_edge), [4.0, 1.0, 0.0])
    np.testing.assert_allclose(np.asarray(ll_edge), [0.0, 0.0, 0.0])


def test_zero_circular_contrast_matches_stokes_i_predict() -> None:
    intensity, l, m, uvw_m, frequency_hz, antenna1, antenna2, correlations = (
        _circular_geometry()
    )
    kwargs = dict(
        intensity=intensity,
        l=l,
        m=m,
        uvw_m=uvw_m,
        frequency_hz=frequency_hz,
        antenna1=antenna1,
        antenna2=antenna2,
        correlations=correlations,
    )
    baseline = predict_stokes_i(**kwargs)
    explicit = predict_stokes_i_explicit(**kwargs)
    with_zero = predict_stokes_i(**kwargs, circular_contrast=0.0)
    explicit_zero = predict_stokes_i_explicit(**kwargs, circular_contrast=0.0)
    np.testing.assert_allclose(with_zero, baseline, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(explicit_zero, explicit, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(explicit, baseline, rtol=2e-13, atol=2e-13)


def test_omitted_contrast_uses_a_partial_hand_beam() -> None:
    intensity, l, m, uvw_m, frequency_hz, antenna1, antenna2, correlations = (
        _circular_geometry()
    )
    rr_beam = np.full((intensity.size, frequency_hz.size), 2.0)
    kwargs = dict(
        intensity=intensity,
        l=l,
        m=m,
        uvw_m=uvw_m,
        frequency_hz=frequency_hz,
        antenna1=antenna1,
        antenna2=antenna2,
        correlations=correlations,
        beam_weights_rr=rr_beam,
    )
    omitted = np.asarray(predict_stokes_i(**kwargs))
    explicit_zero = np.asarray(predict_stokes_i(**kwargs, circular_contrast=0.0))
    explicit = np.asarray(predict_stokes_i_explicit(**kwargs))
    np.testing.assert_allclose(omitted, explicit_zero, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(explicit, explicit_zero, rtol=1e-12, atol=1e-12)
    no_beam = np.asarray(predict_stokes_i(**{**kwargs, "beam_weights_rr": None}))
    np.testing.assert_allclose(omitted[..., 0], 2.0 * no_beam[..., 0])
    np.testing.assert_allclose(omitted[..., 1], no_beam[..., 1])


def test_packed_stokes_v_is_the_rr_ll_difference() -> None:
    intensity, l, m, uvw_m, frequency_hz, antenna1, antenna2, _correlations = (
        _circular_geometry()
    )
    v = 0.25
    correlations = (Correlation.RR, Correlation.LL, Correlation.V)
    predicted = np.asarray(
        predict_stokes_i(
            intensity,
            l,
            m,
            uvw_m,
            frequency_hz,
            antenna1,
            antenna2,
            correlations,
            circular_contrast=v,
        )
    )
    np.testing.assert_allclose(
        predicted[..., 2],
        0.5 * (predicted[..., 0] - predicted[..., 1]),
        rtol=1e-12,
        atol=1e-12,
    )
    unpolarised = np.asarray(
        predict_stokes_i(
            intensity,
            l,
            m,
            uvw_m,
            frequency_hz,
            antenna1,
            antenna2,
            correlations,
        )
    )
    np.testing.assert_allclose(unpolarised[..., 2], 0.0, atol=1e-12)


def test_global_circular_contrast_scales_rr_and_ll_oppositely() -> None:
    intensity, l, m, uvw_m, frequency_hz, antenna1, antenna2, correlations = (
        _circular_geometry()
    )
    v = 0.35
    baseline = np.asarray(
        predict_stokes_i(
            intensity,
            l,
            m,
            uvw_m,
            frequency_hz,
            antenna1,
            antenna2,
            correlations,
        )
    )
    predicted = np.asarray(
        predict_stokes_i(
            intensity,
            l,
            m,
            uvw_m,
            frequency_hz,
            antenna1,
            antenna2,
            correlations,
            circular_contrast=v,
        )
    )
    np.testing.assert_allclose(predicted[..., 0], (1.0 + v) * baseline[..., 0])
    np.testing.assert_allclose(predicted[..., 1], (1.0 - v) * baseline[..., 1])
    packed = apply_global_circular_contrast(baseline, correlations, v)
    np.testing.assert_allclose(predicted, packed)


def test_zero_flux_leaves_keep_zero_parallel_hands() -> None:
    rr, ll = parallel_hand_intensities(np.asarray([0.0, 0.0]), np.asarray([0.9, -0.8]))
    np.testing.assert_array_equal(np.asarray(rr), [0.0, 0.0])
    np.testing.assert_array_equal(np.asarray(ll), [0.0, 0.0])


def test_fit_recovers_injected_global_circular_contrast_on_held_out_baselines() -> None:
    intensity, l, m, uvw_m, frequency_hz, antenna1, antenna2, correlations = (
        _circular_geometry()
    )
    truth = 0.22
    model = np.asarray(
        predict_stokes_i_explicit(
            intensity,
            l,
            m,
            uvw_m,
            frequency_hz,
            antenna1,
            antenna2,
            correlations,
            config=DirectDFTConfig(visibility_chunk_size=3, pixel_chunk_size=4),
        )
    )
    observed = apply_global_circular_contrast(model, correlations, truth)
    discovery = np.zeros(model.shape, dtype=bool)
    discovery[:3] = True
    evaluation = np.zeros(model.shape, dtype=bool)
    evaluation[3:] = True
    contrast, fit, _statistics = fit_global_circular_contrast(
        observed - model,
        np.ones(model.shape),
        discovery,
        model,
        correlations,
    )
    assert contrast == pytest.approx(truth, abs=1e-12)
    predicted = apply_global_circular_contrast(model, correlations, contrast)
    eval_weight = np.ones(model.shape)
    before = float(np.sum(eval_weight[evaluation] * np.abs((observed - model)[evaluation]) ** 2))
    after = float(
        np.sum(eval_weight[evaluation] * np.abs((observed - predicted)[evaluation]) ** 2)
    )
    assert after < 1e-18
    assert after < before
    assert fit.coefficients.shape == (1,)


def test_pol_dependent_gain_is_not_a_single_sky_contrast() -> None:
    intensity, l, m, uvw_m, frequency_hz, antenna1, antenna2, correlations = (
        _circular_geometry()
    )
    model = np.asarray(
        predict_stokes_i(
            intensity,
            l,
            m,
            uvw_m,
            frequency_hz,
            antenna1,
            antenna2,
            correlations,
        )
    )
    observed = model.copy()
    observed[..., 0] *= 1.15
    observed[..., 1] *= 0.97
    contrast, _fit, _statistics = fit_global_circular_contrast(
        observed - model,
        np.ones(model.shape),
        np.ones(model.shape, dtype=bool),
        model,
        correlations,
    )
    predicted = apply_global_circular_contrast(model, correlations, contrast)
    sky_power = float(np.sum(np.abs(observed - predicted) ** 2))
    gain_only_rr = float(np.sum(np.abs(observed[..., 1] - model[..., 1]) ** 2))
    assert sky_power > 0.0
    assert abs(contrast) < 0.2
    powers = correlation_residual_power(
        observed - predicted,
        np.ones(model.shape),
        np.ones(model.shape, dtype=bool),
        correlations,
    )
    assert powers["RR"] > 0.0 or gain_only_rr == 0.0


def test_circular_contrast_rejects_stokes_i_only_products() -> None:
    with pytest.raises(ValueError, match="RR, LL, and/or Stokes V"):
        circular_contrast_response(np.ones((4, 2, 1)), (Correlation.I,))


def test_quadtree_circular_contrast_matches_flat_operator() -> None:
    intensity, l, m, uvw_m, frequency_hz, antenna1, antenna2, correlations = (
        _circular_geometry()
    )
    sky = quadtree_sky_from_regular_grid(3, 2e-3, intensity)
    v = np.zeros(intensity.size)
    v[4] = 0.5
    expected = predict_stokes_i_explicit(
        intensity,
        l,
        m,
        uvw_m,
        frequency_hz,
        antenna1,
        antenna2,
        correlations,
        circular_contrast=v,
        pixel_basis=SquarePixelBasis(1.0, GaussianApproximation.WIDE_FIELD),
        pixel_size_rad=2e-3,
    )
    actual = predict_quadtree_stokes_i_explicit(
        sky.flux,
        sky.topology,
        uvw_m,
        frequency_hz,
        antenna1,
        antenna2,
        correlations,
        circular_contrast=v,
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_composite_hybrid_solver_runs_on_a_point_source() -> None:
    rng = np.random.default_rng(3)
    rows = 24
    uvw_m = rng.uniform(-800.0, 800.0, size=(rows, 3))
    frequency_hz = np.asarray([1.1e9])
    antenna1 = rng.integers(0, 4, rows, dtype=np.int32)
    antenna2 = (antenna1 + rng.integers(1, 4, rows, dtype=np.int32)) % 4
    component = MosaicPointComponent(
        "source",
        np.asarray([0.0]),
        np.asarray([0.0]),
        np.asarray([0.0]),
    )
    truth = MosaicPointComponent(
        "source",
        np.asarray([0.0]),
        np.asarray([0.0]),
        np.asarray([0.6]),
    )
    phase_centre = (0.1, -0.2)
    template = VisibilityBlock(
        uvw_m=uvw_m,
        frequency_hz=frequency_hz,
        visibility=np.zeros((rows, 1, 1), dtype=np.complex128),
        weight=np.ones((rows, 1, 1)),
        flag=np.zeros((rows, 1, 1), dtype=bool),
        time_s=np.arange(rows, dtype=np.float64),
        antenna1=antenna1,
        antenna2=antenna2,
        correlations=(Correlation.I,),
        receptor_basis=ReceptorBasis.STOKES,
        phase_centre_rad=phase_centre,
    )
    (visibility,) = predict_mosaic_composite((template,), (truth,), phase_centre)
    block = VisibilityBlock(
        **{**template.__dict__, "visibility": visibility},
    )
    result = infer_mosaic_composite(
        (block,),
        (component,),
        (block.active,),
        phase_centre,
        InferenceConfig(
            solver="hybrid",
            operator_mode="explicit",
            steps=40,
            learning_rate=0.05,
            sparsity_weight=0.0,
            kkt_tolerance=1e-4,
            validation_interval=5,
        ),
    )
    assert result.solver == "hybrid"
    np.testing.assert_allclose(result.components[0].flux, [0.6], atol=2e-2)
