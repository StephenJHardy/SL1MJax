"""Dual-chunked exact DFT with an explicit streamed adjoint."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from typing import Any, Literal, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from sl1mjax.polarization import Correlation, stokes_i_to_correlations
from sl1mjax.rime import (
    SPEED_OF_LIGHT_M_S,
    _pack_parallel_correlations,
    _pixel_basis_kernel,
)
from sl1mjax.sky import DeltaPixelBasis, PixelBasis


@dataclass(frozen=True)
class DirectDFTConfig:
    """Static tile sizes and an explicit response-tile memory budget."""

    visibility_chunk_size: int = 256
    pixel_chunk_size: int = 1024
    max_response_bytes: int = 512 * 1024**2
    precision: Literal["float32", "float64"] = "float64"

    def __post_init__(self) -> None:
        if self.visibility_chunk_size < 1 or self.pixel_chunk_size < 1:
            raise ValueError("direct DFT chunk sizes must be positive")
        if self.max_response_bytes < 1:
            raise ValueError("max_response_bytes must be positive")
        if self.precision not in {"float32", "float64"}:
            raise ValueError("precision must be float32 or float64")

    @property
    def real_dtype(self) -> Any:
        return jnp.float32 if self.precision == "float32" else jnp.float64

    @property
    def complex_dtype(self) -> Any:
        return jnp.complex64 if self.precision == "float32" else jnp.complex128

    def response_bytes(self, visibility_count: int, pixel_count: int) -> int:
        visibility_tile = min(visibility_count, self.visibility_chunk_size)
        pixel_tile = min(pixel_count, self.pixel_chunk_size)
        return (
            visibility_tile
            * pixel_tile
            * np.dtype(
                np.complex64 if self.precision == "float32" else np.complex128
            ).itemsize
        )

    def validate_problem(self, visibility_count: int, pixel_count: int) -> None:
        response_bytes = self.response_bytes(visibility_count, pixel_count)
        if response_bytes > self.max_response_bytes:
            raise ValueError(
                "direct DFT response tile requires "
                f"{response_bytes} bytes, exceeding max_response_bytes="
                f"{self.max_response_bytes}"
            )


def _pad_rows(array: Array, target_rows: int) -> Array:
    return jnp.pad(
        array,
        ((0, target_rows - array.shape[0]),)
        + ((0, 0),) * (array.ndim - 1),
    )


def _forward_tiled(
    intensity: Array,
    l: Array,
    m: Array,
    uvw_wavelengths: Array,
    *,
    pixel_basis: PixelBasis,
    pixel_size_rad: float | None,
    visibility_chunk_size: int,
    pixel_chunk_size: int,
) -> Array:
    visibility_count = uvw_wavelengths.shape[0]
    pixel_count = intensity.size
    visibility_tile = min(visibility_count, visibility_chunk_size)
    pixel_tile = min(pixel_count, pixel_chunk_size)
    visibility_tiles = (visibility_count + visibility_tile - 1) // visibility_tile
    pixel_tiles = (pixel_count + pixel_tile - 1) // pixel_tile
    padded_visibility_count = visibility_tiles * visibility_tile
    padded_pixel_count = pixel_tiles * pixel_tile
    padded_uvw = _pad_rows(uvw_wavelengths, padded_visibility_count)
    padded_intensity = _pad_rows(intensity, padded_pixel_count)
    padded_l = _pad_rows(l, padded_pixel_count)
    padded_m = _pad_rows(m, padded_pixel_count)
    complex_dtype = (
        jnp.complex64 if intensity.dtype == jnp.float32 else jnp.complex128
    )
    output = jnp.zeros(padded_visibility_count, dtype=complex_dtype)

    def visibility_body(index: int, values: Array) -> Array:
        visibility_start = index * visibility_tile
        uvw_tile = jax.lax.dynamic_slice(
            padded_uvw,
            (visibility_start, 0),
            (visibility_tile, 3),
        )

        def pixel_body(pixel_index: int, prediction: Array) -> Array:
            pixel_start = pixel_index * pixel_tile
            intensity_tile = jax.lax.dynamic_slice(
                padded_intensity, (pixel_start,), (pixel_tile,)
            )
            l_tile = jax.lax.dynamic_slice(
                padded_l, (pixel_start,), (pixel_tile,)
            )
            m_tile = jax.lax.dynamic_slice(
                padded_m, (pixel_start,), (pixel_tile,)
            )
            response = _pixel_basis_kernel(
                uvw_tile,
                l_tile,
                m_tile,
                pixel_basis,
                pixel_size_rad,
                include_projection=False,
            )
            return prediction + response @ intensity_tile

        prediction = jax.lax.fori_loop(
            0,
            pixel_tiles,
            pixel_body,
            jnp.zeros(visibility_tile, dtype=complex_dtype),
        )
        return jax.lax.dynamic_update_slice(
            values, prediction, (visibility_start,)
        )

    output = jax.lax.fori_loop(
        0, visibility_tiles, visibility_body, output
    )
    return cast(Array, output[:visibility_count])


def _adjoint_tiled(
    cotangent: Array,
    l: Array,
    m: Array,
    uvw_wavelengths: Array,
    *,
    pixel_basis: PixelBasis,
    pixel_size_rad: float | None,
    visibility_chunk_size: int,
    pixel_chunk_size: int,
) -> Array:
    visibility_count = uvw_wavelengths.shape[0]
    pixel_count = l.size
    visibility_tile = min(visibility_count, visibility_chunk_size)
    pixel_tile = min(pixel_count, pixel_chunk_size)
    visibility_tiles = (visibility_count + visibility_tile - 1) // visibility_tile
    pixel_tiles = (pixel_count + pixel_tile - 1) // pixel_tile
    padded_visibility_count = visibility_tiles * visibility_tile
    padded_pixel_count = pixel_tiles * pixel_tile
    padded_uvw = _pad_rows(uvw_wavelengths, padded_visibility_count)
    padded_cotangent = _pad_rows(cotangent, padded_visibility_count)
    padded_l = _pad_rows(l, padded_pixel_count)
    padded_m = _pad_rows(m, padded_pixel_count)
    gradient = jnp.zeros(padded_pixel_count, dtype=l.dtype)

    def pixel_body(pixel_index: int, pixel_values: Array) -> Array:
        pixel_start = pixel_index * pixel_tile
        l_tile = jax.lax.dynamic_slice(
            padded_l, (pixel_start,), (pixel_tile,)
        )
        m_tile = jax.lax.dynamic_slice(
            padded_m, (pixel_start,), (pixel_tile,)
        )

        def visibility_body(index: int, contribution: Array) -> Array:
            visibility_start = index * visibility_tile
            uvw_tile = jax.lax.dynamic_slice(
                padded_uvw,
                (visibility_start, 0),
                (visibility_tile, 3),
            )
            cotangent_tile = jax.lax.dynamic_slice(
                padded_cotangent,
                (visibility_start,),
                (visibility_tile,),
            )
            response = _pixel_basis_kernel(
                uvw_tile,
                l_tile,
                m_tile,
                pixel_basis,
                pixel_size_rad,
                include_projection=False,
            )
            # JAX complex VJPs use a transpose cotangent convention. For a
            # real-valued intensity vector the exact pullback is Re(Aᵀ g);
            # the cotangent already carries the required conjugation.
            return contribution + jnp.real(response.T @ cotangent_tile)

        contribution = jax.lax.fori_loop(
            0,
            visibility_tiles,
            visibility_body,
            jnp.zeros(pixel_tile, dtype=l.dtype),
        )
        return jax.lax.dynamic_update_slice(
            pixel_values,
            contribution,
            (pixel_start,),
        )

    gradient = jax.lax.fori_loop(
        0, pixel_tiles, pixel_body, gradient
    )
    return cast(Array, gradient[:pixel_count])


@cache
def _operator_factory(
    pixel_basis: PixelBasis,
    pixel_size_rad: float | None,
    visibility_chunk_size: int,
    pixel_chunk_size: int,
) -> Callable[[Array, Array, Array, Array], Array]:
    @jax.custom_vjp
    def apply(
        intensity: Array, l: Array, m: Array, uvw_wavelengths: Array
    ) -> Array:
        return _forward_tiled(
            intensity,
            l,
            m,
            uvw_wavelengths,
            pixel_basis=pixel_basis,
            pixel_size_rad=pixel_size_rad,
            visibility_chunk_size=visibility_chunk_size,
            pixel_chunk_size=pixel_chunk_size,
        )

    def forward(
        intensity: Array, l: Array, m: Array, uvw_wavelengths: Array
    ) -> tuple[Array, tuple[Array, Array, Array]]:
        result = _forward_tiled(
            intensity,
            l,
            m,
            uvw_wavelengths,
            pixel_basis=pixel_basis,
            pixel_size_rad=pixel_size_rad,
            visibility_chunk_size=visibility_chunk_size,
            pixel_chunk_size=pixel_chunk_size,
        )
        return result, (l, m, uvw_wavelengths)

    def backward(
        residual: tuple[Array, Array, Array], cotangent: Array
    ) -> tuple[Array, None, None, None]:
        l, m, uvw_wavelengths = residual
        gradient = _adjoint_tiled(
            cotangent,
            l,
            m,
            uvw_wavelengths,
            pixel_basis=pixel_basis,
            pixel_size_rad=pixel_size_rad,
            visibility_chunk_size=visibility_chunk_size,
            pixel_chunk_size=pixel_chunk_size,
        )
        return gradient, None, None, None

    apply.defvjp(forward, backward)
    return cast(Callable[[Array, Array, Array, Array], Array], apply)


def direct_scalar_visibility(
    intensity: ArrayLike,
    l: ArrayLike,
    m: ArrayLike,
    uvw_wavelengths: ArrayLike,
    *,
    pixel_basis: PixelBasis | None = None,
    pixel_size_rad: float | None = None,
    config: DirectDFTConfig | None = None,
) -> Array:
    """Apply the exact matrix-free scalar RIME using an explicit adjoint VJP."""

    selected_basis = pixel_basis or DeltaPixelBasis()
    selected_config = config or DirectDFTConfig()
    intensity_array = jnp.asarray(
        intensity, dtype=selected_config.real_dtype
    ).ravel()
    l_array = jnp.asarray(l, dtype=selected_config.real_dtype).ravel()
    m_array = jnp.asarray(m, dtype=selected_config.real_dtype).ravel()
    uvw_array = jnp.asarray(
        uvw_wavelengths, dtype=selected_config.real_dtype
    ).reshape(-1, 3)
    if not (intensity_array.size == l_array.size == m_array.size):
        raise ValueError("intensity, l and m must have equal sizes")
    if intensity_array.size == 0 or uvw_array.shape[0] == 0:
        raise ValueError("direct DFT requires at least one pixel and visibility")
    selected_config.validate_problem(uvw_array.shape[0], intensity_array.size)
    operator = _operator_factory(
        selected_basis,
        pixel_size_rad,
        selected_config.visibility_chunk_size,
        selected_config.pixel_chunk_size,
    )
    return operator(intensity_array, l_array, m_array, uvw_array)


def direct_scalar_adjoint(
    visibility: ArrayLike,
    l: ArrayLike,
    m: ArrayLike,
    uvw_wavelengths: ArrayLike,
    *,
    pixel_basis: PixelBasis | None = None,
    pixel_size_rad: float | None = None,
    config: DirectDFTConfig | None = None,
) -> Array:
    """Apply the real adjoint, ``Re(Aᴴ visibility)``, without materializing A."""

    selected_basis = pixel_basis or DeltaPixelBasis()
    selected_config = config or DirectDFTConfig()
    l_array = jnp.asarray(l, dtype=selected_config.real_dtype).ravel()
    m_array = jnp.asarray(m, dtype=selected_config.real_dtype).ravel()
    uvw_array = jnp.asarray(
        uvw_wavelengths, dtype=selected_config.real_dtype
    ).reshape(-1, 3)
    visibility_array = jnp.asarray(
        visibility, dtype=selected_config.complex_dtype
    ).ravel()
    if l_array.size != m_array.size:
        raise ValueError("l and m must have equal sizes")
    if visibility_array.size != uvw_array.shape[0]:
        raise ValueError("visibility and uvw_wavelengths must have equal sizes")
    if l_array.size == 0 or visibility_array.size == 0:
        raise ValueError("direct DFT requires at least one pixel and visibility")
    selected_config.validate_problem(uvw_array.shape[0], l_array.size)
    return _adjoint_tiled(
        jnp.conj(visibility_array),
        l_array,
        m_array,
        uvw_array,
        pixel_basis=selected_basis,
        pixel_size_rad=pixel_size_rad,
        visibility_chunk_size=selected_config.visibility_chunk_size,
        pixel_chunk_size=selected_config.pixel_chunk_size,
    )


def _explicit_scalar(
    intensity: ArrayLike,
    l: ArrayLike,
    m: ArrayLike,
    uvw: Array,
    frequency: Array,
    *,
    pixel_basis: PixelBasis | None,
    pixel_size_rad: float | None,
    config: DirectDFTConfig,
    beam_weights: ArrayLike | None,
) -> Array:
    rows, channels = uvw.shape[0], frequency.size
    if beam_weights is None:
        uvw_samples = (
            uvw[:, None, :] * frequency[None, :, None] / SPEED_OF_LIGHT_M_S
        ).reshape(-1, 3)
        return direct_scalar_visibility(
            intensity,
            l,
            m,
            uvw_samples,
            pixel_basis=pixel_basis,
            pixel_size_rad=pixel_size_rad,
            config=config,
        ).reshape(rows, channels)
    weights = jnp.asarray(beam_weights, dtype=config.real_dtype)
    intensity_array = jnp.asarray(intensity, dtype=config.real_dtype).ravel()
    pieces = []
    for channel in range(channels):
        uvw_samples = uvw * (frequency[channel] / SPEED_OF_LIGHT_M_S)
        pieces.append(
            direct_scalar_visibility(
                intensity_array * weights[:, channel],
                l,
                m,
                uvw_samples,
                pixel_basis=pixel_basis,
                pixel_size_rad=pixel_size_rad,
                config=config,
            )
        )
    return jnp.stack(pieces, axis=1)


def predict_stokes_i_explicit(
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
    pixel_basis: PixelBasis | None = None,
    pixel_size_rad: float | None = None,
    config: DirectDFTConfig | None = None,
    beam_weights: ArrayLike | None = None,
    beam_weights_rr: ArrayLike | None = None,
    beam_weights_ll: ArrayLike | None = None,
) -> Array:
    """Predict Stokes-I correlations with the explicit exact DFT operator."""

    selected_config = config or DirectDFTConfig()
    uvw = jnp.asarray(uvw_m, dtype=selected_config.real_dtype)
    frequency = jnp.asarray(
        frequency_hz, dtype=selected_config.real_dtype
    )
    baseline_gain = None
    if fixed_gains is not None:
        gains = jnp.asarray(fixed_gains, dtype=selected_config.complex_dtype)
        baseline_gain = gains[jnp.asarray(antenna1)] * jnp.conj(
            gains[jnp.asarray(antenna2)]
        )

    def _apply(weights: ArrayLike | None) -> Array:
        scalar = _explicit_scalar(
            intensity,
            l,
            m,
            uvw,
            frequency,
            pixel_basis=pixel_basis,
            pixel_size_rad=pixel_size_rad,
            config=selected_config,
            beam_weights=weights,
        )
        if baseline_gain is not None:
            return scalar * baseline_gain[:, None]
        return scalar

    if beam_weights_rr is None or beam_weights_ll is None:
        return stokes_i_to_correlations(_apply(beam_weights), correlations)
    return _pack_parallel_correlations(
        _apply(beam_weights_rr),
        _apply(beam_weights_ll),
        correlations,
    )
