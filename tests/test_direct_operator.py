from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from sl1mjax.data.synthetic import simulate_dataset
from sl1mjax.direct_operator import (
    DirectDFTConfig,
    direct_scalar_visibility,
    predict_stokes_i_explicit,
)
from sl1mjax.inference import InferenceConfig, infer_regular_grid
from sl1mjax.polarization import Correlation, ReceptorBasis
from sl1mjax.rime import SPEED_OF_LIGHT_M_S, predict_stokes_i
from sl1mjax.sky import PIXEL_MODEL_NAMES, RegularGrid, pixel_basis_from_name


def _problem() -> tuple[
    Array,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[Correlation, ...],
]:
    grid = RegularGrid(4, 2.5e-3)
    l, m = grid.coordinates
    intensity = jnp.asarray(
        [0.1, 0.3, 0.2, 0.7, 0.4, 1.2, 0.5, 0.8, 0.9, 0.2, 0.6, 0.3, 0.1, 0.4, 0.2, 0.5]
    )
    uvw_m = np.asarray(
        [
            [13.0, -27.0, 4.0],
            [-19.0, 7.0, 31.0],
            [41.0, 17.0, -23.0],
            [-5.0, -11.0, 9.0],
            [29.0, -37.0, 15.0],
        ]
    )
    frequency_hz = np.asarray([0.91e9, 1.07e9, 1.31e9])
    antenna1 = np.asarray([0, 0, 1, 1, 2], dtype=np.int32)
    antenna2 = np.asarray([1, 2, 2, 3, 3], dtype=np.int32)
    correlations = (
        Correlation.RR,
        Correlation.RL,
        Correlation.LR,
        Correlation.LL,
    )
    return (
        intensity,
        l,
        m,
        uvw_m,
        frequency_hz,
        antenna1,
        antenna2,
        correlations,
    )


@pytest.mark.parametrize("pixel_model", PIXEL_MODEL_NAMES)
def test_dual_chunked_forward_matches_materialized_autodiff_operator(
    pixel_model: str,
) -> None:
    intensity, l, m, uvw_m, frequency_hz, antenna1, antenna2, correlations = _problem()
    basis = pixel_basis_from_name(pixel_model, gaussian_sigma_pixels=0.65)
    expected = predict_stokes_i(
        intensity,
        l,
        m,
        uvw_m,
        frequency_hz,
        antenna1,
        antenna2,
        correlations,
        chunk_size=7,
        pixel_basis=basis,
        pixel_size_rad=2.5e-3,
    )
    actual = predict_stokes_i_explicit(
        intensity,
        l,
        m,
        uvw_m,
        frequency_hz,
        antenna1,
        antenna2,
        correlations,
        pixel_basis=basis,
        pixel_size_rad=2.5e-3,
        config=DirectDFTConfig(
            visibility_chunk_size=4,
            pixel_chunk_size=6,
        ),
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-13)


@pytest.mark.parametrize("pixel_model", PIXEL_MODEL_NAMES)
def test_explicit_vjp_matches_native_autodiff_for_every_pixel_basis(
    pixel_model: str,
) -> None:
    intensity, l, m, uvw_m, frequency_hz, antenna1, antenna2, correlations = _problem()
    basis = pixel_basis_from_name(pixel_model, gaussian_sigma_pixels=0.65)
    gains = jnp.asarray([1.1 + 0.2j, 0.9 - 0.1j, 1.05 + 0.05j, 0.8 + 0.15j])
    rng = np.random.default_rng(4)
    cotangent = jnp.asarray(
        rng.normal(size=(5, 3, 4)) + 1j * rng.normal(size=(5, 3, 4))
    )

    def native_loss(values: Array) -> Array:
        prediction = predict_stokes_i(
            values,
            l,
            m,
            uvw_m,
            frequency_hz,
            antenna1,
            antenna2,
            correlations,
            fixed_gains=gains,
            chunk_size=7,
            pixel_basis=basis,
            pixel_size_rad=2.5e-3,
        )
        return jnp.real(jnp.sum(prediction * cotangent)) + 0.2 * jnp.sum(
            jnp.abs(prediction) ** 2
        )

    def explicit_loss(values: Array) -> Array:
        prediction = predict_stokes_i_explicit(
            values,
            l,
            m,
            uvw_m,
            frequency_hz,
            antenna1,
            antenna2,
            correlations,
            fixed_gains=gains,
            pixel_basis=basis,
            pixel_size_rad=2.5e-3,
            config=DirectDFTConfig(
                visibility_chunk_size=4,
                pixel_chunk_size=6,
            ),
        )
        return jnp.real(jnp.sum(prediction * cotangent)) + 0.2 * jnp.sum(
            jnp.abs(prediction) ** 2
        )

    native_gradient = jax.grad(native_loss)(intensity)
    explicit_gradient = jax.jit(jax.grad(explicit_loss))(intensity)
    np.testing.assert_allclose(
        explicit_gradient,
        native_gradient,
        rtol=3e-12,
        atol=3e-12,
    )


def test_explicit_adjoint_obeys_real_transpose_identity() -> None:
    intensity, l, m, uvw_m, frequency_hz, *_ = _problem()
    uvw_wavelengths = (
        uvw_m[:, None, :] * frequency_hz[None, :, None] / SPEED_OF_LIGHT_M_S
    ).reshape(-1, 3)
    config = DirectDFTConfig(visibility_chunk_size=4, pixel_chunk_size=6)
    rng = np.random.default_rng(8)
    cotangent = jnp.asarray(
        rng.normal(size=uvw_wavelengths.shape[0])
        + 1j * rng.normal(size=uvw_wavelengths.shape[0])
    )

    def operator(values: Array) -> Array:
        return direct_scalar_visibility(
            values,
            l,
            m,
            uvw_wavelengths,
            pixel_size_rad=2.5e-3,
            config=config,
        )

    output, pullback = jax.vjp(operator, intensity)
    (adjoint,) = pullback(cotangent)
    left = jnp.real(jnp.sum(output * cotangent))
    right = jnp.vdot(intensity, adjoint)
    np.testing.assert_allclose(left, right, rtol=2e-13, atol=2e-13)


@pytest.mark.parametrize("pixel_model", PIXEL_MODEL_NAMES)
def test_float32_forward_and_vjp_track_float64_reference(
    pixel_model: str,
) -> None:
    intensity, l, m, uvw_m, frequency_hz, antenna1, antenna2, correlations = _problem()
    basis = pixel_basis_from_name(pixel_model, gaussian_sigma_pixels=0.65)
    float32_config = DirectDFTConfig(
        visibility_chunk_size=4,
        pixel_chunk_size=6,
        precision="float32",
    )
    float64_config = DirectDFTConfig(
        visibility_chunk_size=4,
        pixel_chunk_size=6,
        precision="float64",
    )

    def prediction(values: Array, config: DirectDFTConfig) -> Array:
        return predict_stokes_i_explicit(
            values,
            l,
            m,
            uvw_m,
            frequency_hz,
            antenna1,
            antenna2,
            correlations,
            pixel_basis=basis,
            pixel_size_rad=2.5e-3,
            config=config,
        )

    float32_intensity = jnp.asarray(intensity, dtype=jnp.float32)
    actual = prediction(float32_intensity, float32_config)
    expected = prediction(intensity, float64_config)
    assert actual.dtype == jnp.complex64
    np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-5)

    def loss(values: Array, config: DirectDFTConfig) -> Array:
        predicted = prediction(values, config)
        return jnp.mean(jnp.square(jnp.abs(predicted)))

    actual_gradient = jax.grad(loss)(float32_intensity, float32_config)
    expected_gradient = jax.grad(loss)(intensity, float64_config)
    assert actual_gradient.dtype == jnp.float32
    np.testing.assert_allclose(
        actual_gradient,
        expected_gradient,
        rtol=8e-5,
        atol=8e-5,
    )


def test_response_tile_memory_budget_is_enforced_before_execution() -> None:
    def apply(intensity: Array) -> Array:
        return direct_scalar_visibility(
            intensity,
            jnp.zeros(16),
            jnp.zeros(16),
            jnp.zeros((15, 3)),
            config=DirectDFTConfig(
                visibility_chunk_size=8,
                pixel_chunk_size=8,
                max_response_bytes=100,
            ),
        )

    with pytest.raises(ValueError, match="response tile requires"):
        apply(jnp.ones(16))


def test_explicit_operator_runs_inside_jitted_optax_inference() -> None:
    grid = RegularGrid(3, np.deg2rad(12 / 3600))
    block = simulate_dataset(
        grid,
        basis=ReceptorBasis.LINEAR,
        rows=12,
        channels=1,
        noise_std=0.0,
        seed=7,
    ).blocks[0]
    result = infer_regular_grid(
        block,
        grid,
        block.active,
        InferenceConfig(
            steps=3,
            learning_rate=0.1,
            patience=4,
            operator_mode="explicit",
            direct_dft=DirectDFTConfig(
                visibility_chunk_size=5,
                pixel_chunk_size=4,
                precision="float32",
            ),
        ),
    )
    assert result.steps == 3
    assert result.raw_parameters.dtype == np.float32
    assert np.all(np.isfinite(result.objective_history))
