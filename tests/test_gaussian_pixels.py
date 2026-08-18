from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.polynomial.hermite import hermgauss

from sl1mjax.polarization import Correlation
from sl1mjax.rime import _delta_kernel, _gaussian_kernel, predict_stokes_i
from sl1mjax.sky import (
    COMPOUND_N4_BASIS,
    CompoundPixelBasis,
    GaussianApproximation,
    GaussianPixelBasis,
)


def _spherical_gaussian_quadrature(
    uvw_wavelengths: np.ndarray,
    l0: float,
    m0: float,
    sigma_rad: float,
    *,
    order: int = 80,
) -> complex:
    nodes, weights = hermgauss(order)
    delta = np.sqrt(2) * sigma_rad * nodes
    l = l0 + delta[:, None]
    m = m0 + delta[None, :]
    n = np.sqrt(1 - l * l - m * m)
    phase = 2j * np.pi * (
        uvw_wavelengths[0] * l
        + uvw_wavelengths[1] * m
        + uvw_wavelengths[2] * (n - 1)
    )
    return complex(np.sum(weights[:, None] * weights[None, :] * np.exp(phase)) / np.pi)


def test_gaussian_zero_w_matches_normalized_fourier_transform() -> None:
    uvw = jnp.asarray([[120.0, -45.0, 0.0]])
    l = jnp.asarray([0.03])
    m = jnp.asarray([-0.02])
    sigma = 8e-4
    response = _gaussian_kernel(
        uvw,
        l,
        m,
        sigma,
        GaussianApproximation.PARAXIAL,
        include_projection=False,
    )
    expected = np.exp(
        2j * np.pi * (120 * 0.03 - 45 * -0.02)
        - 2 * np.pi**2 * sigma**2 * (120**2 + 45**2)
    )
    np.testing.assert_allclose(response[0, 0], expected, rtol=1e-13, atol=1e-13)


def test_zero_width_limits_are_paraxial_and_exact_delta_phases() -> None:
    uvw = jnp.asarray([[13.0, -21.0, 47.0]])
    l = jnp.asarray([0.22])
    m = jnp.asarray([-0.17])
    radial_squared = float(l[0] ** 2 + m[0] ** 2)
    paraxial = _gaussian_kernel(
        uvw,
        l,
        m,
        0.0,
        GaussianApproximation.PARAXIAL,
        include_projection=False,
    )
    expected_paraxial = np.exp(
        2j
        * np.pi
        * (13 * float(l[0]) - 21 * float(m[0]) - 0.5 * 47 * radial_squared)
    )
    wide = _gaussian_kernel(
        uvw,
        l,
        m,
        0.0,
        GaussianApproximation.WIDE_FIELD,
        include_projection=False,
    )
    exact = _delta_kernel(uvw, l, m, include_projection=False)
    np.testing.assert_allclose(paraxial[0, 0], expected_paraxial, atol=1e-13)
    np.testing.assert_allclose(wide, exact, atol=1e-13)


def test_wide_field_correction_tracks_spherical_quadrature() -> None:
    uvw = np.asarray([20.0, 15.0, 50.0])
    l0, m0, sigma = 0.2, 0.15, 0.003
    oracle = _spherical_gaussian_quadrature(uvw, l0, m0, sigma)
    predictions = {
        approximation: complex(
            _gaussian_kernel(
                jnp.asarray(uvw[None, :]),
                jnp.asarray([l0]),
                jnp.asarray([m0]),
                sigma,
                approximation,
                include_projection=False,
            )[0, 0]
        )
        for approximation in GaussianApproximation
    }
    paraxial_error = abs(predictions[GaussianApproximation.PARAXIAL] - oracle)
    wide_error = abs(predictions[GaussianApproximation.WIDE_FIELD] - oracle)
    assert wide_error < 2e-3
    assert wide_error < 0.02 * paraxial_error


def test_compound_response_is_weighted_sum_of_gaussian_responses() -> None:
    uvw_m = np.asarray([[120.0, -70.0, 35.0], [-50.0, 90.0, -20.0]])
    common = (
        np.asarray([0.8, 0.3]),
        np.asarray([0.01, -0.02]),
        np.asarray([-0.03, 0.04]),
        uvw_m,
        np.asarray([1.1e9]),
        np.asarray([0, 0]),
        np.asarray([1, 1]),
        (Correlation.I,),
    )
    pixel_size = 2e-4
    compound = np.asarray(
        predict_stokes_i(
            *common,
            pixel_basis=COMPOUND_N4_BASIS,
            pixel_size_rad=pixel_size,
        )
    )
    expected = np.zeros_like(compound)
    for weight, sigma in zip(
        COMPOUND_N4_BASIS.integrated_weights,
        COMPOUND_N4_BASIS.sigma_pixels,
        strict=True,
    ):
        expected += weight * np.asarray(
            predict_stokes_i(
                *common,
                pixel_basis=GaussianPixelBasis(sigma),
                pixel_size_rad=pixel_size,
            )
        )
    np.testing.assert_allclose(compound, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(COMPOUND_N4_BASIS.integrated_weights.sum(), 1.0)


@pytest.mark.parametrize(
    "basis",
    [
        GaussianPixelBasis(0.45, GaussianApproximation.PARAXIAL),
        GaussianPixelBasis(0.45, GaussianApproximation.WIDE_FIELD),
        COMPOUND_N4_BASIS,
    ],
    ids=["gaussian-paraxial", "gaussian-wide-field", "compound-wide-field"],
)
def test_gaussian_intensity_gradient_matches_finite_difference(
    basis: GaussianPixelBasis | CompoundPixelBasis,
) -> None:
    intensity = jnp.asarray([0.7, 0.2])
    common = (
        jnp.asarray([0.01, -0.015]),
        jnp.asarray([-0.02, 0.025]),
        jnp.asarray([[500.0, -300.0, 200.0]]),
        jnp.asarray([1.4e9]),
        jnp.asarray([0]),
        jnp.asarray([1]),
        (Correlation.I,),
    )

    def loss(values: jax.Array) -> jax.Array:
        prediction = predict_stokes_i(
            values,
            *common,
            pixel_basis=basis,
            pixel_size_rad=1e-4,
        )
        return jnp.real(jnp.sum(prediction * jnp.conj(prediction)))

    automatic = np.asarray(jax.grad(loss)(intensity))
    epsilon = 1e-6
    finite = np.empty(2)
    for index in range(2):
        offset = np.zeros(2)
        offset[index] = epsilon
        finite[index] = (
            float(loss(intensity + offset)) - float(loss(intensity - offset))
        ) / (2 * epsilon)
    np.testing.assert_allclose(automatic, finite, rtol=2e-6, atol=2e-7)


def test_invalid_compound_normalization_is_rejected() -> None:
    with pytest.raises(ValueError, match="unit integrated flux"):
        CompoundPixelBasis((1.0,), (1.0,))
