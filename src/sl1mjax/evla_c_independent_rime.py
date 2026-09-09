"""Independent NumPy RIME used to audit the production assembly helpers.

This module reimplements Stokes packing, Jones application, parallactic
rotation, HOLORASTER moving-reference visibilities, and the C147 geometric
fringe without calling the production prediction helpers. Tests compare
the two. A matching round-trip is not sufficient; deliberate defects must
fail.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

SPEED_OF_LIGHT_M_S = 299_792_458.0
HAND_INDEX = {"RR": (0, 0), "RL": (0, 1), "LR": (1, 0), "LL": (1, 1)}


def as_jones(values: ArrayLike) -> NDArray[np.complex128]:
    array = np.asarray(values, dtype=np.complex128)
    if array.shape[-2:] != (2, 2):
        raise ValueError("Jones/coherency trailing axes must be 2×2")
    return array


def independent_circular_stokes(
    stokes_i: ArrayLike,
    stokes_q: ArrayLike = 0.0,
    stokes_u: ArrayLike = 0.0,
    stokes_v: ArrayLike = 0.0,
) -> NDArray[np.complex128]:
    """IEEE/CASA circular packing: RR=I+V, LL=I-V, RL=Q+iU, LR=Q-iU."""

    intensity = np.asarray(stokes_i, dtype=np.complex128)
    q_val = np.asarray(stokes_q, dtype=np.complex128)
    u_val = np.asarray(stokes_u, dtype=np.complex128)
    v_val = np.asarray(stokes_v, dtype=np.complex128)
    shape = np.broadcast_shapes(intensity.shape, q_val.shape, u_val.shape, v_val.shape)
    coherency = np.zeros(shape + (2, 2), dtype=np.complex128)
    intensity = np.broadcast_to(intensity, shape)
    q_val = np.broadcast_to(q_val, shape)
    u_val = np.broadcast_to(u_val, shape)
    v_val = np.broadcast_to(v_val, shape)
    coherency[..., 0, 0] = intensity + v_val
    coherency[..., 0, 1] = q_val + 1j * u_val
    coherency[..., 1, 0] = q_val - 1j * u_val
    coherency[..., 1, 1] = intensity - v_val
    return coherency


def independent_apply_jones(
    jones_p: ArrayLike,
    coherency: ArrayLike,
    jones_q: ArrayLike,
) -> NDArray[np.complex128]:
    """Form ``J_p C J_q^H``."""

    left = as_jones(jones_p)
    sky = as_jones(coherency)
    right = np.conjugate(np.swapaxes(as_jones(jones_q), -1, -2))
    return left @ sky @ right


def independent_parallactic_jones(chi_rad: ArrayLike) -> NDArray[np.complex128]:
    """``P(χ) = diag(e^{-iχ}, e^{+iχ})``."""

    angle = np.asarray(chi_rad, dtype=np.float64).reshape(-1)
    rotation = np.exp(-1j * angle)
    para = np.zeros((angle.size, 2, 2), dtype=np.complex128)
    para[:, 0, 0] = rotation
    para[:, 1, 1] = np.conjugate(rotation)
    return para


def independent_sky_from_feed(feed: ArrayLike, chi_rad: ArrayLike) -> NDArray[np.complex128]:
    """``E_sky = P^H E_feed P``."""

    jones = as_jones(feed)
    para = independent_parallactic_jones(chi_rad)
    if jones.ndim == 4:
        para = para[:, None, :, :]
    conjugate = np.conjugate(np.swapaxes(para, -1, -2))
    return conjugate @ jones @ para


def independent_holoraster_visibility(
    residual_moving: ArrayLike,
    beam_moving: ArrayLike,
    source: ArrayLike,
    residual_reference: ArrayLike,
    moving_is_p: ArrayLike,
) -> NDArray[np.complex128]:
    """Compact-calibrator HOLORASTER: ``V = R_m E_m S R_r^H`` when the mover is p."""

    r_m = as_jones(residual_moving)
    e_m = as_jones(beam_moving)
    sky = as_jones(source)
    r_r = as_jones(residual_reference)
    mover_p = np.asarray(moving_is_p, dtype=bool).reshape(-1)
    if e_m.ndim == 3:
        e_m = e_m[:, None, :, :]
    if sky.ndim == 3:
        sky = sky[:, None, :, :]
    if r_m.ndim == 3:
        r_m = r_m[:, None, :, :]
    if r_r.ndim == 3:
        r_r = r_r[:, None, :, :]
    left = r_m @ e_m
    vis_p = left @ sky @ np.conjugate(np.swapaxes(r_r, -1, -2))
    vis_q = r_r @ sky @ np.conjugate(np.swapaxes(left, -1, -2))
    return np.where(mover_p[:, None, None, None], vis_p, vis_q)


def independent_dual_antenna_visibility(
    residual_p: ArrayLike,
    beam_p: ArrayLike,
    source: ArrayLike,
    beam_q: ArrayLike,
    residual_q: ArrayLike,
) -> NDArray[np.complex128]:
    """Offset-field RIME without fringe: ``V = R_p E_p S E_q^H R_q^H``."""

    r_p = as_jones(residual_p)
    e_p = as_jones(beam_p)
    sky = as_jones(source)
    e_q = as_jones(beam_q)
    r_q = as_jones(residual_q)
    right = np.conjugate(np.swapaxes(e_q, -1, -2)) @ np.conjugate(np.swapaxes(r_q, -1, -2))
    return r_p @ e_p @ sky @ right


def independent_geometric_fringe(
    uvw_m: ArrayLike,
    frequency_hz: ArrayLike,
    lm_rad: ArrayLike,
) -> NDArray[np.complex128]:
    """CASA geometric phase ``exp(+2πi [u l + v m + w(n-1)])``."""

    uvw = np.asarray(uvw_m, dtype=np.float64)
    freq = np.asarray(frequency_hz, dtype=np.float64).reshape(-1)
    lm = np.asarray(lm_rad, dtype=np.float64)
    if uvw.ndim != 2 or uvw.shape[-1] != 3:
        raise ValueError("uvw_m must have shape (row, 3)")
    if lm.shape == (2,):
        lm = np.broadcast_to(lm, (uvw.shape[0], 2))
    if lm.shape != (uvw.shape[0], 2):
        raise ValueError("lm_rad must have shape (2,) or (row, 2)")
    l_rad = lm[:, 0]
    m_rad = lm[:, 1]
    n_rad = np.sqrt(np.maximum(0.0, 1.0 - l_rad * l_rad - m_rad * m_rad))
    waves = uvw[:, :, None] * (freq[None, None, :] / SPEED_OF_LIGHT_M_S)
    phase = (
        waves[:, 0, :] * l_rad[:, None]
        + waves[:, 1, :] * m_rad[:, None]
        + waves[:, 2, :] * (n_rad[:, None] - 1.0)
    )
    return np.exp(2j * np.pi * phase)


def independent_offset_field_visibility(
    residual_p: ArrayLike,
    beam_p: ArrayLike,
    source: ArrayLike,
    beam_q: ArrayLike,
    residual_q: ArrayLike,
    *,
    uvw_m: ArrayLike,
    frequency_hz: ArrayLike,
    sky_lm_rad: ArrayLike,
) -> NDArray[np.complex128]:
    vis = independent_dual_antenna_visibility(residual_p, beam_p, source, beam_q, residual_q)
    phase = independent_geometric_fringe(uvw_m, frequency_hz, sky_lm_rad)
    if vis.ndim == 3:
        return vis * phase[:, 0, None, None]
    return vis * phase[..., None, None]


def pack_cassbeam_native_columns(columns: ArrayLike) -> NDArray[np.complex128]:
    """Map CASSBEAM ``RR,LR,RL,LL`` columns onto ``[[RR, RL], [LR, LL]]``."""

    raw = np.asarray(columns, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[-1] != 8:
        raise ValueError("native columns must have shape (sample, 8)")
    jones = np.empty((raw.shape[0], 2, 2), dtype=np.complex128)
    jones[:, 0, 0] = raw[:, 0] + 1j * raw[:, 1]
    jones[:, 1, 0] = raw[:, 2] + 1j * raw[:, 3]
    jones[:, 0, 1] = raw[:, 4] + 1j * raw[:, 5]
    jones[:, 1, 1] = raw[:, 6] + 1j * raw[:, 7]
    return jones


def manufactured_asymmetric_jones(
    *,
    scale: complex = 1.0 + 0.2j,
) -> NDArray[np.complex128]:
    """Known non-Hermitian Jones used by the independent algebra tests."""

    return scale * np.array([[1.1 + 0.3j, 0.04 - 0.02j], [0.03 + 0.05j, 0.9 - 0.1j]])


def source_lm_is_negative_commanded(commanded_azelgeo: ArrayLike) -> NDArray[np.float64]:
    """HOLORASTER source-in-beam coordinate after the locked no-swap map."""

    return -np.asarray(commanded_azelgeo, dtype=np.float64)
