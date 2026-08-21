from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss

from sl1mjax.polarization import Correlation
from sl1mjax.rime import (
    _delta_kernel,
    _square_kernel,
    predict_stokes_i,
    square_wide_field_error_bound,
)
from sl1mjax.sky import GaussianApproximation, SquarePixelBasis


def _spherical_square_quadrature(
    uvw_wavelengths: np.ndarray,
    l0: float,
    m0: float,
    width_rad: float,
    *,
    order: int = 80,
) -> complex:
    """Numerically integrate the exact curved-sky top-hat response."""

    nodes, weights = leggauss(order)
    offset = 0.5 * width_rad * nodes
    l = l0 + offset[:, None]
    m = m0 + offset[None, :]
    n = np.sqrt(1 - l * l - m * m)
    phase = 2j * np.pi * (
        uvw_wavelengths[0] * l
        + uvw_wavelengths[1] * m
        + uvw_wavelengths[2] * (n - 1)
    )
    jacobian = (0.5 * width_rad) ** 2 / width_rad**2
    return complex(
        np.sum(weights[:, None] * weights[None, :] * np.exp(phase)) * jacobian
    )


def test_square_zero_w_matches_flat_sinc_fourier_transform() -> None:
    uvw = jnp.asarray([[120.0, -45.0, 0.0]])
    l = jnp.asarray([0.03])
    m = jnp.asarray([-0.02])
    width = 4e-3
    response = _square_kernel(
        uvw,
        l,
        m,
        width,
        GaussianApproximation.PARAXIAL,
        include_projection=False,
    )
    expected = (
        np.exp(2j * np.pi * (120 * 0.03 - 45 * -0.02))
        * np.sinc(120 * width)
        * np.sinc(-45 * width)
    )
    np.testing.assert_allclose(response[0, 0], expected, rtol=1e-13, atol=1e-13)


def test_zero_width_limits_are_paraxial_and_exact_delta_phases() -> None:
    uvw = jnp.asarray([[13.0, -21.0, 47.0]])
    l = jnp.asarray([0.22])
    m = jnp.asarray([-0.17])
    paraxial = _square_kernel(
        uvw,
        l,
        m,
        0.0,
        GaussianApproximation.PARAXIAL,
        include_projection=False,
    )
    expected_paraxial = np.exp(
        2j * np.pi * (13 * float(l[0]) - 21 * float(m[0]))
    )
    wide = _square_kernel(
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
    l0, m0, width = 0.2, 0.15, 6e-3
    oracle = _spherical_square_quadrature(uvw, l0, m0, width)
    predictions = {
        approximation: complex(
            _square_kernel(
                jnp.asarray(uvw[None, :]),
                jnp.asarray([l0]),
                jnp.asarray([m0]),
                width,
                approximation,
                include_projection=False,
            )[0, 0]
        )
        for approximation in GaussianApproximation
    }
    paraxial_error = abs(predictions[GaussianApproximation.PARAXIAL] - oracle)
    wide_error = abs(predictions[GaussianApproximation.WIDE_FIELD] - oracle)
    assert wide_error < 0.05
    assert wide_error < 0.05 * paraxial_error


def test_wide_field_tilt_shift_beats_naive_centroid_correction() -> None:
    """The sinc-argument tilt shift must not regress vs. a plain centroid fix."""

    def naive_centroid_correction(uvw: np.ndarray, l0: float, m0: float, width: float) -> complex:
        u, v, w = uvw
        n0 = np.sqrt(1 - l0 * l0 - m0 * m0)
        flat = np.sinc(u * width) * np.sinc(v * width) * np.exp(
            2j * np.pi * (u * l0 + v * m0)
        )
        return complex(flat * np.exp(2j * np.pi * w * (n0 - 1)))

    uvw = np.asarray([20.0, 15.0, 200.0])
    l0, m0, width = 0.2, 0.15, 8e-3
    oracle = _spherical_square_quadrature(uvw, l0, m0, width, order=120)
    naive = naive_centroid_correction(uvw, l0, m0, width)
    tilted = complex(
        _square_kernel(
            jnp.asarray(uvw[None, :]),
            jnp.asarray([l0]),
            jnp.asarray([m0]),
            width,
            GaussianApproximation.WIDE_FIELD,
            include_projection=False,
        )[0, 0]
    )
    naive_error = abs(naive - oracle)
    tilted_error = abs(tilted - oracle)
    assert tilted_error < naive_error


def test_wide_field_error_bound_holds_against_spherical_quadrature() -> None:
    """The analytic bound must never fall short of the true oracle error."""

    rng = np.random.default_rng(1)
    for _ in range(200):
        l0 = float(rng.uniform(-0.5, 0.5))
        m0 = float(rng.uniform(-0.5, 0.5))
        if l0**2 + m0**2 > 0.6:
            continue
        width = float(10 ** rng.uniform(-4, -1.3))
        w = float(rng.uniform(-500, 500))
        u = float(rng.uniform(-200, 200))
        v = float(rng.uniform(-200, 200))
        uvw = np.asarray([u, v, w])
        oracle = _spherical_square_quadrature(uvw, l0, m0, width, order=100)
        approx = complex(
            _square_kernel(
                jnp.asarray(uvw[None, :]),
                jnp.asarray([l0]),
                jnp.asarray([m0]),
                width,
                GaussianApproximation.WIDE_FIELD,
                include_projection=False,
            )[0, 0]
        )
        actual_error = abs(approx - oracle)
        bound = float(square_wide_field_error_bound(width, l0, m0, w))
        assert actual_error <= bound * 1.05, (l0, m0, width, w, actual_error, bound)


def test_wide_field_error_bound_scales_with_width_squared_and_w() -> None:
    l0, m0 = 0.2, 0.15
    base = float(square_wide_field_error_bound(4e-3, l0, m0, 100.0))
    doubled_width = float(square_wide_field_error_bound(8e-3, l0, m0, 100.0))
    doubled_w = float(square_wide_field_error_bound(4e-3, l0, m0, 200.0))
    np.testing.assert_allclose(doubled_width, base * 4, rtol=1e-12)
    np.testing.assert_allclose(doubled_w, base * 2, rtol=1e-12)


def test_square_split_conserves_flux_and_reproduces_parent_response() -> None:
    """Four equal-flux quarter-width children exactly tile a parent square."""

    uvw_m = np.asarray([[120.0, -70.0, 35.0], [-50.0, 90.0, -20.0]])
    frequency_hz = np.asarray([1.1e9])
    antenna1 = np.asarray([0, 0])
    antenna2 = np.asarray([1, 1])
    correlations = (Correlation.I,)
    pixel_size = 2e-4
    parent_flux = 1.6
    parent_width = 2.0
    parent_l, parent_m = 0.01, -0.03

    parent = np.asarray(
        predict_stokes_i(
            [parent_flux],
            [parent_l],
            [parent_m],
            uvw_m,
            frequency_hz,
            antenna1,
            antenna2,
            correlations,
            pixel_basis=SquarePixelBasis(parent_width, GaussianApproximation.PARAXIAL),
            pixel_size_rad=pixel_size,
        )
    )

    child_width = parent_width / 2
    child_offset = child_width * pixel_size / 2
    child_l = parent_l + np.asarray([1, 1, -1, -1]) * child_offset
    child_m = parent_m + np.asarray([1, -1, 1, -1]) * child_offset
    child_flux = np.full(4, parent_flux / 4)

    children = np.asarray(
        predict_stokes_i(
            child_flux,
            child_l,
            child_m,
            uvw_m,
            frequency_hz,
            antenna1,
            antenna2,
            correlations,
            pixel_basis=SquarePixelBasis(child_width, GaussianApproximation.PARAXIAL),
            pixel_size_rad=pixel_size,
        )
    )
    np.testing.assert_allclose(float(child_flux.sum()), parent_flux, rtol=1e-14)
    np.testing.assert_allclose(children, parent, rtol=1e-12, atol=1e-12)


def test_square_intensity_gradient_matches_finite_difference() -> None:
    basis = SquarePixelBasis(0.8, GaussianApproximation.WIDE_FIELD)
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


def test_invalid_square_width_is_rejected() -> None:
    with pytest.raises(ValueError, match="width_pixels"):
        SquarePixelBasis(0.0)
    with pytest.raises(ValueError, match="width_pixels"):
        SquarePixelBasis(-1.0)
