"""Regular-grid sky parameterization and physical transforms."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike


@dataclass(frozen=True)
class RegularGrid:
    size: int
    pixel_size_rad: float

    def __post_init__(self) -> None:
        if self.size < 2:
            raise ValueError("size must be at least two")
        if not np.isfinite(self.pixel_size_rad) or self.pixel_size_rad <= 0:
            raise ValueError("pixel_size_rad must be finite and positive")
        edge = (self.size - 1) * self.pixel_size_rad / 2
        if edge >= 1 / np.sqrt(2):
            raise ValueError("grid extends outside valid direction cosines")

    @property
    def coordinates(self) -> tuple[np.ndarray, np.ndarray]:
        axis = (
            np.arange(self.size, dtype=np.float64) - (self.size - 1) / 2
        ) * self.pixel_size_rad
        l = np.broadcast_to(axis[None, :], (self.size, self.size))
        m = np.broadcast_to(axis[:, None], (self.size, self.size))
        return l.ravel(), m.ravel()


def physical_intensity(raw_intensity: ArrayLike) -> Array:
    """Map unconstrained optimizer parameters to positive flux."""

    return jnp.logaddexp(jnp.asarray(raw_intensity, dtype=jnp.float64), 0.0)


def raw_from_intensity(intensity: ArrayLike) -> Array:
    """Stable inverse softplus for strictly positive initialization."""

    value = jnp.maximum(jnp.asarray(intensity, dtype=jnp.float64), 1e-12)
    return value + jnp.log(-jnp.expm1(-value))
