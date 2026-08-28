"""Polarization-aware direct radio-interferometric measurement equation."""

from __future__ import annotations

import math

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from sl1mjax.circular_contrast import (
    parallel_hand_intensities,
    requires_circular_parallel_hands,
    uses_split_parallel_operator,
)
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
    the curved-sky phase ``w·(n(l, m) - 1)`` is folded in, so the wide-field
    response instead Taylor-expands that phase around the pixel center. The
    linear term is absorbed exactly by shifting the sinc arguments by
    ``w·(l, m)/n`` (the standard tilted-plane/faceting correction used in
    wide-field imaging); only the quadratic curvature term across the
    pixel's own footprint is dropped. Its magnitude is bounded by
    ``square_wide_field_error_bound``.
    """

    n = jnp.sqrt(1.0 - l * l - m * m)
    u = uvw_wavelengths[:, 0, None]
    v = uvw_wavelengths[:, 1, None]
    w = uvw_wavelengths[:, 2, None]
    source_l = l[None, :]
    source_m = m[None, :]
    source_n = n[None, :]
    width = jnp.asarray(width_rad, dtype=uvw_wavelengths.real.dtype)
    if approximation is GaussianApproximation.WIDE_FIELD:
        u_eff = u - w * source_l / source_n
        v_eff = v - w * source_m / source_n
    else:
        u_eff = u
        v_eff = v
    response = (
        jnp.exp(2j * jnp.pi * (u * source_l + v * source_m))
        * jnp.sinc(u_eff * width)
        * jnp.sinc(v_eff * width)
    )
    if approximation is GaussianApproximation.WIDE_FIELD:
        response = response * jnp.exp(2j * jnp.pi * w * (source_n - 1.0))
    if include_projection:
        response = response / source_n
    return response


def square_wide_field_error_bound(
    width_rad: ArrayLike,
    l: ArrayLike,
    m: ArrayLike,
    max_w_wavelengths: ArrayLike,
) -> Array:
    """Upper bound on the wide-field square kernel's neglected curvature error.

    ``_square_kernel`` with ``GaussianApproximation.WIDE_FIELD`` removes the
    *linear* term of the curved-sky phase ``w·(n(l, m) - 1)`` around each
    pixel's center exactly, via a sinc-argument shift. The leading source of
    remaining error is therefore the *quadratic* curvature term neglected
    across the pixel's own footprint. Bounding ``|exp(iθ) - 1| <= |θ|`` for
    that quadratic phase over the pixel's ``[-width/2, width/2]^2`` extent
    gives, to leading Taylor order,

        error <= (pi/4) * |w| * width_rad**2 * (1 + n**2 + 2*|l*m|) / n**3

    where ``n = sqrt(1 - l**2 - m**2)``. This bound was checked against a
    high-order spherical-quadrature oracle across randomized pixel and
    baseline configurations (widths up to ~5e-2 rad, ``|w|`` up to 500
    wavelengths): the true error never exceeded roughly a third of the
    bound, so it is a genuine but not wildly conservative estimate in that
    regime; it degrades like any truncated Taylor series far outside it.

    ``max_w_wavelengths`` should be the largest ``|w|`` (in wavelengths)
    among the visibilities that will see this pixel — a dataset-wide maximum
    gives a simple, conservative, per-pixel refinement gate that needs no
    visibility evaluation at all, only pixel geometry.
    """

    l_array = jnp.asarray(l)
    m_array = jnp.asarray(m)
    width_array = jnp.asarray(width_rad)
    n = jnp.sqrt(1.0 - l_array * l_array - m_array * m_array)
    curvature_scale = (1.0 + n * n + 2.0 * jnp.abs(l_array * m_array)) / (n**3)
    return (
        0.25
        * jnp.pi
        * jnp.abs(jnp.asarray(max_w_wavelengths))
        * jnp.square(width_array)
        * curvature_scale
    )


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
        elif correlation is Correlation.V:
            columns.append(0.5 * (rr - ll))
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
    circular_contrast: ArrayLike | None = None,
) -> Array:
    """Predict correlations for integrated Stokes-I pixel/component fluxes.

    ``intensity`` is integrated Jy per component/pixel, so no ``1/n`` Jacobian
    is applied by default. Set ``include_projection`` only for a brightness
    density explicitly integrated in direction-cosine coordinates. Non-delta
    pixel widths are standard deviations measured in grid pixel spacings.
    ``circular_contrast`` is the shared-support Stokes-V fraction v with
    I_RR = I(1+v) and I_LL = I(1-v).
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
    if circular_contrast is not None:
        requires_circular_parallel_hands(correlations)
    selected_basis = pixel_basis or DeltaPixelBasis()
    baseline_gain = None
    if fixed_gains is not None:
        gains = jnp.asarray(fixed_gains, dtype=jnp.complex128)
        baseline_gain = gains[jnp.asarray(antenna1)] * jnp.conj(
            gains[jnp.asarray(antenna2)]
        )

    def _apply(selected_intensity: Array, weights: Array | None) -> Array:
        scalar = _scalar_from_intensity(
            selected_intensity,
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

    rr_beam = None if beam_weights_rr is None else jnp.asarray(beam_weights_rr)
    ll_beam = None if beam_weights_ll is None else jnp.asarray(beam_weights_ll)
    stokes_beam = None if beam_weights is None else jnp.asarray(beam_weights)
    if not uses_split_parallel_operator(circular_contrast, rr_beam, ll_beam):
        return stokes_i_to_correlations(_apply(intensity_array, stokes_beam), correlations)
    rr_intensity, ll_intensity = parallel_hand_intensities(
        intensity_array, circular_contrast
    )
    return _pack_parallel_correlations(
        _apply(rr_intensity, stokes_beam if rr_beam is None else rr_beam),
        _apply(ll_intensity, stokes_beam if ll_beam is None else ll_beam),
        correlations,
    )
