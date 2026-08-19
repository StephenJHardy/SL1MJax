"""Polarization-aware direct radio-interferometric measurement equation."""

from __future__ import annotations

import math

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from sl1mjax.polarization import Correlation, stokes_i_to_correlations
from sl1mjax.sky import (
    CompoundPixelBasis,
    DeltaPixelBasis,
    GaussianApproximation,
    GaussianPixelBasis,
    PixelBasis,
)

SPEED_OF_LIGHT_M_S = 299_792_458.0


def _delta_kernel(
    uvw_wavelengths: Array,
    l: Array,
    m: Array,
    *,
    include_projection: bool,
) -> Array:
    n = jnp.sqrt(1.0 - l * l - m * m)
    phase = 2j * jnp.pi * (
        uvw_wavelengths[:, 0, None] * l[None, :]
        + uvw_wavelengths[:, 1, None] * m[None, :]
        + uvw_wavelengths[:, 2, None] * (n[None, :] - 1.0)
    )
    response = jnp.exp(phase)
    return response / n[None, :] if include_projection else response


def _gaussian_kernel(
    uvw_wavelengths: Array,
    l: Array,
    m: Array,
    sigma_rad: float | Array,
    approximation: GaussianApproximation,
    *,
    include_projection: bool,
) -> Array:
    """Hardy (2013) normalized circular-Gaussian visibility response.

    ``sigma_rad`` is the ordinary standard deviation. The paper's scale is
    ``sqrt(2π) * sigma_rad``.
    """

    n = jnp.sqrt(1.0 - l * l - m * m)
    u = uvw_wavelengths[:, 0, None]
    v = uvw_wavelengths[:, 1, None]
    w = uvw_wavelengths[:, 2, None]
    source_l = l[None, :]
    source_m = m[None, :]
    radial_squared = source_l * source_l + source_m * source_m
    paper_sigma_squared = 2.0 * jnp.pi * jnp.square(
        jnp.asarray(sigma_rad, dtype=uvw_wavelengths.real.dtype)
    )
    denominator = 1.0 + 1j * w * paper_sigma_squared
    response = (
        jnp.exp(2j * jnp.pi * (u * source_l + v * source_m) / denominator)
        * jnp.exp(-1j * jnp.pi * w * radial_squared / denominator)
        * jnp.exp(
            -jnp.pi
            * paper_sigma_squared
            * (u * u + v * v)
            / denominator
        )
        / denominator
    )
    if approximation is GaussianApproximation.WIDE_FIELD:
        response *= jnp.exp(
            2j
            * jnp.pi
            * w
            * (n[None, :] - 1.0 + 0.5 * radial_squared)
        )
    if include_projection:
        response /= n[None, :]
    return response


def _pixel_basis_kernel(
    uvw_wavelengths: Array,
    l: Array,
    m: Array,
    pixel_basis: PixelBasis,
    pixel_size_rad: float | None,
    *,
    include_projection: bool,
) -> Array:
    if isinstance(pixel_basis, DeltaPixelBasis):
        return _delta_kernel(
            uvw_wavelengths, l, m, include_projection=include_projection
        )
    if (
        pixel_size_rad is None
        or not math.isfinite(pixel_size_rad)
        or pixel_size_rad <= 0
    ):
        raise ValueError("finite positive pixel_size_rad is required for Gaussian pixels")
    if isinstance(pixel_basis, GaussianPixelBasis):
        return _gaussian_kernel(
            uvw_wavelengths,
            l,
            m,
            pixel_basis.sigma_pixels * pixel_size_rad,
            pixel_basis.approximation,
            include_projection=include_projection,
        )
    if isinstance(pixel_basis, CompoundPixelBasis):
        response = jnp.zeros(
            (uvw_wavelengths.shape[0], l.size),
            dtype=(
                jnp.complex64
                if uvw_wavelengths.dtype == jnp.float32
                else jnp.complex128
            ),
        )
        for weight, sigma_pixels in zip(
            pixel_basis.integrated_weights,
            pixel_basis.sigma_pixels,
            strict=True,
        ):
            response += jnp.asarray(
                weight, dtype=uvw_wavelengths.real.dtype
            ) * _gaussian_kernel(
                uvw_wavelengths,
                l,
                m,
                sigma_pixels * pixel_size_rad,
                pixel_basis.approximation,
                include_projection=include_projection,
            )
        return response
    raise TypeError(f"unsupported pixel basis {type(pixel_basis).__name__}")


def predict_stokes_i(
    intensity: ArrayLike,
    l: ArrayLike,
    m: ArrayLike,
    uvw_m: ArrayLike,
    frequency_hz: ArrayLike,
    antenna1: ArrayLike,
    antenna2: ArrayLike,
    correlations: tuple[Correlation, ...],
    *,
    fixed_gains: ArrayLike | None = None,
    chunk_size: int = 4096,
    include_projection: bool = False,
    pixel_basis: PixelBasis | None = None,
    pixel_size_rad: float | None = None,
) -> Array:
    """Predict correlations for integrated Stokes-I pixel/component fluxes.

    ``intensity`` is integrated Jy per component/pixel, so no ``1/n`` Jacobian
    is applied by default. Set ``include_projection`` only for a brightness
    density explicitly integrated in direction-cosine coordinates. Non-delta
    pixel widths are standard deviations measured in grid pixel spacings.
    """

    intensity_array = jnp.asarray(intensity, dtype=jnp.float64).ravel()
    l_array = jnp.asarray(l, dtype=jnp.float64).ravel()
    m_array = jnp.asarray(m, dtype=jnp.float64).ravel()
    uvw_array = jnp.asarray(uvw_m, dtype=jnp.float64)
    frequency_array = jnp.asarray(frequency_hz, dtype=jnp.float64)
    if not (intensity_array.size == l_array.size == m_array.size):
        raise ValueError("intensity, l, and m must contain the same number of components")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    selected_basis = pixel_basis or DeltaPixelBasis()
    rows, channels = uvw_array.shape[0], frequency_array.size
    scale = frequency_array / SPEED_OF_LIGHT_M_S
    uvw_samples = (uvw_array[:, None, :] * scale[None, :, None]).reshape(-1, 3)
    pieces = []
    for start in range(0, rows * channels, chunk_size):
        response = _pixel_basis_kernel(
            uvw_samples[start : start + chunk_size],
            l_array,
            m_array,
            selected_basis,
            pixel_size_rad,
            include_projection=include_projection,
        )
        pieces.append(response @ intensity_array)
    scalar = jnp.concatenate(pieces).reshape(rows, channels)
    if fixed_gains is not None:
        gains = jnp.asarray(fixed_gains, dtype=jnp.complex128)
        baseline_gain = gains[jnp.asarray(antenna1)] * jnp.conj(
            gains[jnp.asarray(antenna2)]
        )
        scalar = scalar * baseline_gain[:, None]
    return stokes_i_to_correlations(scalar, correlations)
