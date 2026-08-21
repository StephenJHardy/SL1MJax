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
    SquarePixelBasis,
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


def _square_kernel(
    uvw_wavelengths: Array,
    l: Array,
    m: Array,
    width_rad: float | Array,
    approximation: GaussianApproximation,
    *,
    include_projection: bool,
) -> Array:
    """Analytic uniform-brightness square-pixel visibility response.

    The paraxial response is the exact 2-D Fourier transform of a flat-sky
    top-hat: a separable product of two ``sinc`` factors, so no numerical
    integration is required. ``width_rad`` is the square's side length,
    measured in the paraxial ``(l, m)`` plane and carries no curvature term.

    Unlike the Gaussian kernel, the top-hat integral has no closed form once
    a quadratic w-term phase is folded in, so the wide-field correction here
    is a plain centroid correction: it multiplies by the exact single-point
    curvature phase ``w·(n(l, m) - 1)`` evaluated at the pixel center. This
    ignores curvature variation across the pixel's own extent, unlike the
    Gaussian kernel's wide-field term, which swaps out an already-integrated
    quadratic approximation for the exact centroid value.
    """

    n = jnp.sqrt(1.0 - l * l - m * m)
    u = uvw_wavelengths[:, 0, None]
    v = uvw_wavelengths[:, 1, None]
    w = uvw_wavelengths[:, 2, None]
    source_l = l[None, :]
    source_m = m[None, :]
    width = jnp.asarray(width_rad, dtype=uvw_wavelengths.real.dtype)
    response = (
        jnp.exp(2j * jnp.pi * (u * source_l + v * source_m))
        * jnp.sinc(u * width)
        * jnp.sinc(v * width)
    )
    if approximation is GaussianApproximation.WIDE_FIELD:
        response = response * jnp.exp(2j * jnp.pi * w * (n[None, :] - 1.0))
    if include_projection:
        response = response / n[None, :]
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
        raise ValueError("finite positive pixel_size_rad is required for non-delta pixels")
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
    if isinstance(pixel_basis, SquarePixelBasis):
        return _square_kernel(
            uvw_wavelengths,
            l,
            m,
            pixel_basis.width_pixels * pixel_size_rad,
            pixel_basis.approximation,
            include_projection=include_projection,
        )
    raise TypeError(f"unsupported pixel basis {type(pixel_basis).__name__}")


def _scalar_from_intensity(
    intensity: Array,
    l: Array,
    m: Array,
    uvw_m: Array,
    frequency_hz: Array,
    *,
    chunk_size: int,
    include_projection: bool,
    pixel_basis: PixelBasis,
    pixel_size_rad: float | None,
    beam_weights: Array | None,
) -> Array:
    rows, channels = uvw_m.shape[0], frequency_hz.size
    if beam_weights is None:
        scale = frequency_hz / SPEED_OF_LIGHT_M_S
        uvw_samples = (uvw_m[:, None, :] * scale[None, :, None]).reshape(-1, 3)
        pieces = []
        for start in range(0, rows * channels, chunk_size):
            response = _pixel_basis_kernel(
                uvw_samples[start : start + chunk_size],
                l,
                m,
                pixel_basis,
                pixel_size_rad,
                include_projection=include_projection,
            )
            pieces.append(response @ intensity)
        return jnp.concatenate(pieces).reshape(rows, channels)
    pieces = []
    for channel in range(channels):
        scaled = intensity * beam_weights[:, channel]
        uvw_samples = uvw_m * (frequency_hz[channel] / SPEED_OF_LIGHT_M_S)
        channel_pieces = []
        for start in range(0, rows, chunk_size):
            response = _pixel_basis_kernel(
                uvw_samples[start : start + chunk_size],
                l,
                m,
                pixel_basis,
                pixel_size_rad,
                include_projection=include_projection,
            )
            channel_pieces.append(response @ scaled)
        pieces.append(jnp.concatenate(channel_pieces))
    return jnp.stack(pieces, axis=1)


def _pack_parallel_correlations(
    rr: Array, ll: Array, correlations: tuple[Correlation, ...]
) -> Array:
    """Pack RR/LL scalar visibilities into the requested correlation order."""

    columns = []
    for correlation in correlations:
        if correlation is Correlation.RR:
            columns.append(rr)
        elif correlation is Correlation.LL:
            columns.append(ll)
        elif correlation is Correlation.I:
            columns.append(0.5 * (rr + ll))
        else:
            columns.append(jnp.zeros_like(rr))
    return jnp.stack(columns, axis=-1)


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
    beam_weights: ArrayLike | None = None,
    beam_weights_rr: ArrayLike | None = None,
    beam_weights_ll: ArrayLike | None = None,
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
    baseline_gain = None
    if fixed_gains is not None:
        gains = jnp.asarray(fixed_gains, dtype=jnp.complex128)
        baseline_gain = gains[jnp.asarray(antenna1)] * jnp.conj(
            gains[jnp.asarray(antenna2)]
        )

    def _apply(weights: Array | None) -> Array:
        scalar = _scalar_from_intensity(
            intensity_array,
            l_array,
            m_array,
            uvw_array,
            frequency_array,
            chunk_size=chunk_size,
            include_projection=include_projection,
            pixel_basis=selected_basis,
            pixel_size_rad=pixel_size_rad,
            beam_weights=weights,
        )
        if baseline_gain is not None:
            return scalar * baseline_gain[:, None]
        return scalar

    if beam_weights_rr is None or beam_weights_ll is None:
        weights = None if beam_weights is None else jnp.asarray(beam_weights)
        return stokes_i_to_correlations(_apply(weights), correlations)
    return _pack_parallel_correlations(
        _apply(jnp.asarray(beam_weights_rr)),
        _apply(jnp.asarray(beam_weights_ll)),
        correlations,
    )
