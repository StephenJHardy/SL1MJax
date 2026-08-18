"""Regular-grid sky parameterization and physical transforms."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike


class GaussianApproximation(StrEnum):
    """Analytic Gaussian visibility approximations from Hardy (2013)."""

    PARAXIAL = "paraxial"
    WIDE_FIELD = "wide_field"


@dataclass(frozen=True)
class DeltaPixelBasis:
    """Integrated point-source flux at each grid location."""

    kind: str = field(default="delta", init=False)


@dataclass(frozen=True)
class GaussianPixelBasis:
    """A normalized circular Gaussian of fixed standard deviation per pixel."""

    sigma_pixels: float
    approximation: GaussianApproximation = GaussianApproximation.WIDE_FIELD
    kind: str = field(default="gaussian", init=False)

    def __post_init__(self) -> None:
        if not np.isfinite(self.sigma_pixels) or self.sigma_pixels <= 0:
            raise ValueError("sigma_pixels must be finite and positive")
        object.__setattr__(
            self, "approximation", GaussianApproximation(self.approximation)
        )


@dataclass(frozen=True)
class CompoundPixelBasis:
    """Fixed positive radial Gaussian mixture carrying one fitted pixel flux."""

    amplitudes: tuple[float, ...]
    sigma_pixels: tuple[float, ...]
    approximation: GaussianApproximation = GaussianApproximation.WIDE_FIELD
    kind: str = field(default="compound", init=False)

    def __post_init__(self) -> None:
        amplitudes = tuple(float(value) for value in self.amplitudes)
        sigmas = tuple(float(value) for value in self.sigma_pixels)
        if not amplitudes or len(amplitudes) != len(sigmas):
            raise ValueError("compound amplitudes and sigmas must have equal nonzero length")
        if not np.all(np.isfinite(amplitudes)):
            raise ValueError("compound amplitudes must be finite")
        if not np.all(np.isfinite(sigmas)) or np.any(np.asarray(sigmas) <= 0):
            raise ValueError("compound sigmas must be finite and positive")
        normalization = 2 * np.pi * np.dot(amplitudes, np.square(sigmas))
        if not np.isclose(normalization, 1.0, rtol=1e-8, atol=1e-10):
            raise ValueError("compound radial kernel must have unit integrated flux")
        object.__setattr__(self, "amplitudes", amplitudes)
        object.__setattr__(self, "sigma_pixels", sigmas)
        object.__setattr__(
            self, "approximation", GaussianApproximation(self.approximation)
        )

    @property
    def integrated_weights(self) -> np.ndarray:
        """Signed unit-sum flux carried by each normalized Gaussian."""

        return (
            2
            * np.pi
            * np.asarray(self.amplitudes)
            * np.square(np.asarray(self.sigma_pixels))
        )


PixelBasis = DeltaPixelBasis | GaussianPixelBasis | CompoundPixelBasis


COMPOUND_N4_BASIS = CompoundPixelBasis(
    amplitudes=(-0.4819388811, 1.4796429201, -2.3399303030, 2.1791470337),
    sigma_pixels=(0.1383215145, 0.1870230973, 0.2611714362, 0.3560340736),
)

PIXEL_MODEL_NAMES = (
    "delta",
    "gaussian-paraxial",
    "gaussian-wide-field",
    "compound-paraxial",
    "compound-wide-field",
)


def pixel_basis_from_name(
    name: str, *, gaussian_sigma_pixels: float = 0.5
) -> PixelBasis:
    """Construct a public pixel basis from its CLI/configuration name."""

    if name == "delta":
        return DeltaPixelBasis()
    if name.startswith("gaussian-"):
        approximation = GaussianApproximation(name.removeprefix("gaussian-").replace("-", "_"))
        return GaussianPixelBasis(gaussian_sigma_pixels, approximation)
    if name.startswith("compound-"):
        approximation = GaussianApproximation(name.removeprefix("compound-").replace("-", "_"))
        return CompoundPixelBasis(
            COMPOUND_N4_BASIS.amplitudes,
            COMPOUND_N4_BASIS.sigma_pixels,
            approximation,
        )
    raise ValueError(f"unknown pixel model {name!r}")


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
        # FITS celestial images conventionally have CDELT1 < 0: increasing
        # array columns move toward decreasing RA, hence decreasing eastward l.
        l = np.broadcast_to(-axis[None, :], (self.size, self.size))
        m = np.broadcast_to(axis[:, None], (self.size, self.size))
        return l.ravel(), m.ravel()


def physical_intensity(raw_intensity: ArrayLike) -> Array:
    """Map unconstrained optimizer parameters to positive flux."""

    return jnp.logaddexp(jnp.asarray(raw_intensity, dtype=jnp.float64), 0.0)


def raw_from_intensity(intensity: ArrayLike) -> Array:
    """Stable inverse softplus for strictly positive initialization."""

    value = jnp.maximum(jnp.asarray(intensity, dtype=jnp.float64), 1e-12)
    return value + jnp.log(-jnp.expm1(-value))
