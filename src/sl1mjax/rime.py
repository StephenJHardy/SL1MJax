"""Polarization-aware direct radio-interferometric measurement equation."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from sl1mjax.polarization import Correlation, stokes_i_to_correlations

SPEED_OF_LIGHT_M_S = 299_792_458.0


def _kernel(
    uvw_wavelengths: Array,
    l: Array,
    m: Array,
    *,
    include_projection: bool,
) -> Array:
    n = jnp.sqrt(1.0 - l * l - m * m)
    phase = -2j * jnp.pi * (
        uvw_wavelengths[:, 0, None] * l[None, :]
        + uvw_wavelengths[:, 1, None] * m[None, :]
        + uvw_wavelengths[:, 2, None] * (n[None, :] - 1.0)
    )
    response = jnp.exp(phase)
    return response / n[None, :] if include_projection else response


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
    include_projection: bool = True,
) -> Array:
    """Predict all ordered correlations for a positive Stokes-I sky."""

    intensity_array = jnp.asarray(intensity, dtype=jnp.float64).ravel()
    l_array = jnp.asarray(l, dtype=jnp.float64).ravel()
    m_array = jnp.asarray(m, dtype=jnp.float64).ravel()
    uvw_array = jnp.asarray(uvw_m, dtype=jnp.float64)
    frequency_array = jnp.asarray(frequency_hz, dtype=jnp.float64)
    if not (intensity_array.size == l_array.size == m_array.size):
        raise ValueError("intensity, l, and m must contain the same number of components")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    rows, channels = uvw_array.shape[0], frequency_array.size
    scale = frequency_array / SPEED_OF_LIGHT_M_S
    uvw_samples = (uvw_array[:, None, :] * scale[None, :, None]).reshape(-1, 3)
    pieces = []
    for start in range(0, rows * channels, chunk_size):
        response = _kernel(
            uvw_samples[start : start + chunk_size],
            l_array,
            m_array,
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
