"""Polarization metadata, feed receptors, and Jones coherency packing."""

from __future__ import annotations

from enum import StrEnum

import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike
from numpy.typing import ArrayLike as NumpyArrayLike, NDArray


class ReceptorBasis(StrEnum):
    LINEAR = "linear"
    CIRCULAR = "circular"
    STOKES = "stokes"


class Receptor(StrEnum):
    """A Jones axis: a feed, or scalar Stokes I.

    Circular feeds are ``R`` and ``L``; linear feeds are ``X`` and ``Y``.
    Packed visibilities such as RR or RL are products of two feeds.
    ``I`` is the 1×1 Stokes-I Jones used when the data are already packed
    as Stokes I, not a third physical feed.
    """

    R = "R"
    L = "L"
    X = "X"
    Y = "Y"
    I = "I"


class Correlation(StrEnum):
    I = "I"
    Q = "Q"
    U = "U"
    V = "V"
    XX = "XX"
    XY = "XY"
    YX = "YX"
    YY = "YY"
    RR = "RR"
    RL = "RL"
    LR = "LR"
    LL = "LL"


_ALLOWED = {
    ReceptorBasis.LINEAR: {Correlation.XX, Correlation.XY, Correlation.YX, Correlation.YY},
    ReceptorBasis.CIRCULAR: {Correlation.RR, Correlation.RL, Correlation.LR, Correlation.LL},
    ReceptorBasis.STOKES: {Correlation.I, Correlation.Q, Correlation.U, Correlation.V},
}


def validate_correlations(
    basis: ReceptorBasis, correlations: tuple[Correlation, ...]
) -> None:
    if not correlations:
        raise ValueError("at least one correlation is required")
    if len(set(correlations)) != len(correlations):
        raise ValueError("correlations must be unique")
    invalid = set(correlations) - _ALLOWED[basis]
    if invalid:
        raise ValueError(f"{basis} basis does not support {sorted(invalid)}")


def stokes_i_to_correlations(
    visibility_i: ArrayLike, correlations: tuple[Correlation, ...]
) -> Array:
    """Map an ideal unpolarized Stokes-I visibility into ordered products."""

    value = jnp.asarray(visibility_i)
    parallel = {Correlation.I, Correlation.XX, Correlation.YY, Correlation.RR, Correlation.LL}
    factors = jnp.asarray(
        [1.0 if correlation in parallel else 0.0 for correlation in correlations],
        dtype=value.real.dtype,
    )
    return value[..., None] * factors


_CIRCULAR_PRODUCTS: dict[Correlation, tuple[Receptor, Receptor]] = {
    Correlation.RR: (Receptor.R, Receptor.R),
    Correlation.RL: (Receptor.R, Receptor.L),
    Correlation.LR: (Receptor.L, Receptor.R),
    Correlation.LL: (Receptor.L, Receptor.L),
}
_LINEAR_PRODUCTS: dict[Correlation, tuple[Receptor, Receptor]] = {
    Correlation.XX: (Receptor.X, Receptor.X),
    Correlation.XY: (Receptor.X, Receptor.Y),
    Correlation.YX: (Receptor.Y, Receptor.X),
    Correlation.YY: (Receptor.Y, Receptor.Y),
}
_CANONICAL_RECEPTORS = {
    ReceptorBasis.CIRCULAR: (Receptor.R, Receptor.L),
    ReceptorBasis.LINEAR: (Receptor.X, Receptor.Y),
}


def correlation_receptor_pair(correlation: Correlation) -> tuple[Receptor, Receptor]:
    """Return the ordered feed pair that forms a correlation product."""

    if correlation is Correlation.I:
        return (Receptor.I, Receptor.I)
    if correlation in _CIRCULAR_PRODUCTS:
        return _CIRCULAR_PRODUCTS[correlation]
    if correlation in _LINEAR_PRODUCTS:
        return _LINEAR_PRODUCTS[correlation]
    raise ValueError(
        f"{correlation} is a Stokes product, not a feed coherency; "
        "Jones apply uses RR/RL/LR/LL, XX/XY/YX/YY, or Stokes I"
    )


def receptors_for_correlations(
    correlations: tuple[Correlation, ...],
) -> tuple[Receptor, ...]:
    """Canonical Jones receptors required by a set of correlation products."""

    if not correlations:
        raise ValueError("at least one correlation is required")
    stokes = {Correlation.I, Correlation.Q, Correlation.U, Correlation.V}
    if any(correlation in stokes for correlation in correlations):
        if all(correlation is Correlation.I for correlation in correlations):
            return (Receptor.I,)
        raise ValueError(
            "Stokes Q/U/V are not feed coherencies; Jones apply uses "
            "RR/RL/LR/LL, XX/XY/YX/YY, or Stokes I"
        )
    pairs = [correlation_receptor_pair(correlation) for correlation in correlations]
    bases = {
        ReceptorBasis.CIRCULAR if pair[0] in {Receptor.R, Receptor.L} else ReceptorBasis.LINEAR
        for pair in pairs
    }
    if len(bases) != 1:
        raise ValueError("correlations mix linear and circular feeds")
    canonical = _CANONICAL_RECEPTORS[next(iter(bases))]
    present = {receptor for pair in pairs for receptor in pair}
    return tuple(receptor for receptor in canonical if receptor in present)


def pack_coherency(
    visibility: NumpyArrayLike,
    correlations: tuple[Correlation, ...],
    receptors: tuple[Receptor, ...],
) -> NDArray[np.complex128]:
    """Pack ordered products into a receptor×receptor coherency matrix.

    Missing matrix slots are zero.  This is the feed coherency used by
    :math:`J_p C J_q^H`, not the sky Stokes-V packing
    :math:`V=(RR-LL)/2`.
    """

    vis = np.asarray(visibility)
    if vis.shape[-1] != len(correlations):
        raise ValueError("visibility last axis must match correlations")
    if not receptors:
        raise ValueError("at least one receptor is required")
    if len(set(receptors)) != len(receptors):
        raise ValueError("receptors must be unique")
    index = {receptor: i for i, receptor in enumerate(receptors)}
    packed = np.zeros(
        vis.shape[:-1] + (len(receptors), len(receptors)),
        dtype=np.result_type(vis, np.complex128),
    )
    for slot, correlation in enumerate(correlations):
        first, second = correlation_receptor_pair(correlation)
        try:
            packed[..., index[first], index[second]] = vis[..., slot]
        except KeyError as exc:
            raise ValueError(
                f"{correlation} needs receptors {(first, second)} inside {receptors}"
            ) from exc
    return packed


def unpack_coherency(
    coherency: NumpyArrayLike,
    correlations: tuple[Correlation, ...],
    receptors: tuple[Receptor, ...],
) -> NDArray[np.complex128]:
    """Extract ordered correlation products from a packed coherency matrix."""

    packed = np.asarray(coherency)
    expected = (len(receptors), len(receptors))
    if packed.shape[-2:] != expected:
        raise ValueError(f"coherency last axes must have shape {expected}")
    index = {receptor: i for i, receptor in enumerate(receptors)}
    visibility = np.empty(packed.shape[:-2] + (len(correlations),), dtype=packed.dtype)
    for slot, correlation in enumerate(correlations):
        first, second = correlation_receptor_pair(correlation)
        try:
            visibility[..., slot] = packed[..., index[first], index[second]]
        except KeyError as exc:
            raise ValueError(
                f"{correlation} needs receptors {(first, second)} inside {receptors}"
            ) from exc
    return visibility


def diagonal_jones_matrices(antenna_jones: NumpyArrayLike) -> NDArray[np.complex128]:
    """Promote per-receptor scalar Jones values to diagonal 2×2 (or 1×1) matrices."""

    values = np.asarray(antenna_jones)
    n_receptor = values.shape[-1]
    matrices = np.zeros(values.shape + (n_receptor,), dtype=np.result_type(values, np.complex128))
    for receptor in range(n_receptor):
        matrices[..., receptor, receptor] = values[..., receptor]
    return matrices


def invert_jones(jones: NumpyArrayLike) -> NDArray[np.complex128]:
    """Invert 1×1 or 2×2 Jones matrices along the last two axes."""

    matrices = np.asarray(jones)
    n_receptor = matrices.shape[-1]
    if matrices.shape[-2] != n_receptor:
        raise ValueError("Jones matrices must be square")
    if n_receptor == 1:
        inverse = np.full(
            matrices.shape, np.nan, dtype=np.result_type(matrices, np.complex128)
        )
        np.divide(1.0, matrices, out=inverse, where=matrices != 0)
        return inverse
    if n_receptor != 2:
        raise ValueError("only 1×1 and 2×2 Jones inverses are supported")
    a = matrices[..., 0, 0]
    b = matrices[..., 0, 1]
    c = matrices[..., 1, 0]
    d = matrices[..., 1, 1]
    det = a * d - b * c
    inverse = np.full(
        matrices.shape, np.nan, dtype=np.result_type(matrices, np.complex128)
    )
    finite = np.isfinite(det) & (det != 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        np.divide(d, det, out=inverse[..., 0, 0], where=finite)
        np.divide(-b, det, out=inverse[..., 0, 1], where=finite)
        np.divide(-c, det, out=inverse[..., 1, 0], where=finite)
        np.divide(a, det, out=inverse[..., 1, 1], where=finite)
    return inverse


def apply_jones_to_coherency(
    coherency: NumpyArrayLike,
    jones_p: NumpyArrayLike,
    jones_q: NumpyArrayLike,
) -> NDArray[np.complex128]:
    """Form ``J_p C J_q^H`` for packed feed coherencies."""

    sky = np.asarray(coherency)
    left = np.asarray(jones_p)
    right = np.conjugate(np.swapaxes(np.asarray(jones_q), -1, -2))
    return np.matmul(left, np.matmul(sky, right))


def circular_stokes_from_correlations(
    visibility: NumpyArrayLike,
    correlations: tuple[Correlation, ...],
) -> tuple[NDArray[np.complex128], NDArray[np.complex128], NDArray[np.complex128], NDArray[np.complex128]]:
    """Unpack Stokes I, Q, U, V from circular products.

    CASA circular convention: ``RR=I+V``, ``LL=I-V``, ``RL=Q+iU``,
    ``LR=Q-iU``.
    """

    vis = np.asarray(visibility)
    index = {correlation: slot for slot, correlation in enumerate(correlations)}
    required = (Correlation.RR, Correlation.RL, Correlation.LR, Correlation.LL)
    missing = [correlation for correlation in required if correlation not in index]
    if missing:
        raise ValueError(f"circular Stokes unpack needs {required}, missing {missing}")
    rr = vis[..., index[Correlation.RR]]
    rl = vis[..., index[Correlation.RL]]
    lr = vis[..., index[Correlation.LR]]
    ll = vis[..., index[Correlation.LL]]
    stokes_i = 0.5 * (rr + ll)
    stokes_q = 0.5 * (rl + lr)
    stokes_u = 0.5 * (rl - lr) / 1j
    stokes_v = 0.5 * (rr - ll)
    return stokes_i, stokes_q, stokes_u, stokes_v


def fractional_linear_polarisation(
    stokes_q: NumpyArrayLike, stokes_u: NumpyArrayLike, stokes_i: NumpyArrayLike
) -> NDArray[np.float64]:
    """Return ``sqrt(Q^2+U^2)/I`` from Stokes values."""

    intensity = np.real(np.asarray(stokes_i))
    return np.abs(
        np.hypot(np.real(np.asarray(stokes_q)), np.real(np.asarray(stokes_u))) / intensity
    )


def electric_vector_position_angle_rad(
    stokes_q: NumpyArrayLike, stokes_u: NumpyArrayLike
) -> NDArray[np.float64]:
    """Return EVPA ``χ = (1/2) arg(Q + iU)`` in radians."""

    return 0.5 * np.arctan2(np.real(np.asarray(stokes_u)), np.real(np.asarray(stokes_q)))
