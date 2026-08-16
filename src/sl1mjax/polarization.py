"""Polarization metadata and ideal Stokes-I correlation mapping."""

from __future__ import annotations

from enum import StrEnum

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike


class ReceptorBasis(StrEnum):
    LINEAR = "linear"
    CIRCULAR = "circular"
    STOKES = "stokes"


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
