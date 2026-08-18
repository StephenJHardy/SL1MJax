"""Independent direct-DFT diagnostic imaging products."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from sl1mjax.data.canonical import VisibilityBlock
from sl1mjax.polarization import stokes_i_to_correlations
from sl1mjax.rime import SPEED_OF_LIGHT_M_S
from sl1mjax.sky import RegularGrid


def dirty_image_and_psf(
    block: VisibilityBlock,
    grid: RegularGrid,
    *,
    chunk_size: int = 256,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Form naturally weighted dirty image and PSF by a direct adjoint DFT."""

    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    factors = np.asarray(
        stokes_i_to_correlations(1.0, block.correlations), dtype=np.complex128
    )
    active_weight = np.where(block.active, block.weight, 0.0)
    weighted_visibility = np.sum(
        active_weight * np.conj(factors)[None, None, :] * block.visibility,
        axis=2,
    )
    weighted_response = np.sum(
        active_weight * np.abs(factors)[None, None, :] ** 2,
        axis=2,
    )
    normalization = float(np.sum(weighted_response))
    if normalization <= 0:
        raise ValueError("visibility block has no active positive-weight samples")

    uvw_wavelengths = (
        block.uvw_m[:, None, :]
        * block.frequency_hz[None, :, None]
        / SPEED_OF_LIGHT_M_S
    )
    l, m = grid.coordinates
    dirty = np.empty(l.size, dtype=np.float64)
    psf = np.empty(l.size, dtype=np.float64)
    for start in range(0, l.size, chunk_size):
        stop = min(start + chunk_size, l.size)
        source_l = l[start:stop]
        source_m = m[start:stop]
        source_n = np.sqrt(1 - source_l * source_l - source_m * source_m)
        phase = 2j * np.pi * (
            uvw_wavelengths[..., 0, None] * source_l
            + uvw_wavelengths[..., 1, None] * source_m
            + uvw_wavelengths[..., 2, None] * (source_n - 1)
        )
        adjoint = np.exp(-phase)
        dirty[start:stop] = (
            np.real(np.sum(adjoint * weighted_visibility[..., None], axis=(0, 1)))
            / normalization
        )
        psf[start:stop] = (
            np.real(np.sum(adjoint * weighted_response[..., None], axis=(0, 1)))
            / normalization
        )
    shape = (grid.size, grid.size)
    return dirty.reshape(shape), psf.reshape(shape)
