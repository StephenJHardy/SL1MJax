"""Exact spherical phase-centre and direction-cosine transforms."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def radec_to_lmn(
    phase_centre_ra_rad: float,
    phase_centre_dec_rad: float,
    ra_rad: ArrayLike,
    dec_rad: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Convert celestial coordinates to exact phase-centred direction cosines."""

    ra = np.asarray(ra_rad, dtype=np.float64)
    dec = np.asarray(dec_rad, dtype=np.float64)
    delta_ra = ra - phase_centre_ra_rad
    l = np.cos(dec) * np.sin(delta_ra)
    m = (
        np.sin(dec) * np.cos(phase_centre_dec_rad)
        - np.cos(dec) * np.sin(phase_centre_dec_rad) * np.cos(delta_ra)
    )
    n = (
        np.sin(dec) * np.sin(phase_centre_dec_rad)
        + np.cos(dec) * np.cos(phase_centre_dec_rad) * np.cos(delta_ra)
    )
    return l, m, n


def lmn_to_radec(
    phase_centre_ra_rad: float,
    phase_centre_dec_rad: float,
    l: ArrayLike,
    m: ArrayLike,
    n: ArrayLike | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Convert valid direction cosines back to celestial coordinates."""

    l_array = np.asarray(l, dtype=np.float64)
    m_array = np.asarray(m, dtype=np.float64)
    n_array = (
        np.sqrt(1.0 - l_array * l_array - m_array * m_array)
        if n is None
        else np.asarray(n, dtype=np.float64)
    )
    sin_ra0, cos_ra0 = np.sin(phase_centre_ra_rad), np.cos(phase_centre_ra_rad)
    sin_dec0, cos_dec0 = (
        np.sin(phase_centre_dec_rad),
        np.cos(phase_centre_dec_rad),
    )
    x = n_array * cos_dec0 * cos_ra0 - l_array * sin_ra0 - m_array * sin_dec0 * cos_ra0
    y = n_array * cos_dec0 * sin_ra0 + l_array * cos_ra0 - m_array * sin_dec0 * sin_ra0
    z = n_array * sin_dec0 + m_array * cos_dec0
    ra = np.mod(np.arctan2(y, x), 2 * np.pi)
    dec = np.arcsin(np.clip(z, -1.0, 1.0))
    return ra, dec
