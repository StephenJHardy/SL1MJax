"""Optional primary-beam models kept separate from the geometric RIME."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sl1mjax.rime import SPEED_OF_LIGHT_M_S


def gaussian_primary_beam(
    l: ArrayLike,
    m: ArrayLike,
    frequency_hz: ArrayLike,
    *,
    dish_diameter_m: float,
    fwhm_factor: float = 1.02,
) -> NDArray[np.float64]:
    """Return an idealized circular Gaussian power-beam attenuation."""

    if dish_diameter_m <= 0 or fwhm_factor <= 0:
        raise ValueError("dish diameter and FWHM factor must be positive")
    l_array, m_array, frequency = np.broadcast_arrays(
        np.asarray(l, dtype=np.float64),
        np.asarray(m, dtype=np.float64),
        np.asarray(frequency_hz, dtype=np.float64),
    )
    if np.any(frequency <= 0):
        raise ValueError("frequency must be positive")
    radius = np.sqrt(l_array * l_array + m_array * m_array)
    if np.any(radius >= 1):
        raise ValueError("direction cosines must lie inside the visible hemisphere")
    angular_radius = np.arcsin(radius)
    fwhm = fwhm_factor * SPEED_OF_LIGHT_M_S / (frequency * dish_diameter_m)
    return np.asarray(
        np.exp(-4 * np.log(2) * (angular_radius / fwhm) ** 2),
        dtype=np.float64,
    )
