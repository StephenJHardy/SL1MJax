"""VLA primary-beam catalog and on-grid power patterns.

The Airy dish, blockage, and maximum radius match CASA ``vpmanager.setpbairy``
defaults for the VLA. Optional beam squint displaces each receptor by
``VLA_SQUINT_FWHM_FRACTION`` of the Gaussian FWHM, in opposite directions,
scaling as ``1/frequency``. That stored value is a receptor half-offset, so
the total RCP–LCP separation is twice as large. It is not evidence-grade;
see ``beam_conventions``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sl1mjax.rime import SPEED_OF_LIGHT_M_S

VLA_DISH_DIAMETER_M = 25.0
VLA_BLOCKAGE_DIAMETER_M = 2.5
VLA_GAUSSIAN_FWHM_FACTOR = 1.02
VLA_AIRY_MAX_RADIUS_RAD_AT_1GHZ = np.deg2rad(0.8)
# Receptor half-offset as a fraction of Gaussian FWHM. Opposite hands
# therefore have a total RCP–LCP separation of 0.12 FWHM. Published totals
# are ~0.05–0.06 FWHM (EVLA Memo 195: 2.4 arcmin·GHz). Do not enable for
# evidence-grade work; see sl1mjax.beam_conventions.
VLA_SQUINT_FWHM_FRACTION = 0.06
VLA_SQUINT_REFERENCE_HZ = 1.0e9

PrimaryBeamKind = Literal["gaussian", "airy"]
BeamReceptor = Literal["I", "RR", "LL"]


def _j1(x: ArrayLike) -> NDArray[np.float64]:
    """Bessel J1 from the power series, accurate for the VLA Airy argument range."""

    argument = np.asarray(x, dtype=np.float64)
    half = 0.5 * argument
    half_squared = half * half
    term = np.ones_like(argument)
    total = np.ones_like(argument)
    for order in range(1, 28):
        term *= -half_squared / (order * (order + 1))
        total += term
    return np.asarray(half * total, dtype=np.float64)


def _jinc(x: ArrayLike) -> NDArray[np.float64]:
    argument = np.asarray(x, dtype=np.float64)
    return np.divide(
        2.0 * _j1(argument),
        argument,
        out=np.ones_like(argument),
        where=np.abs(argument) >= 1e-12,
    )


def _angular_radius(l: np.ndarray, m: np.ndarray) -> np.ndarray:
    radius = np.sqrt(l * l + m * m)
    if np.any(radius >= 1):
        raise ValueError("direction cosines must lie inside the visible hemisphere")
    return np.asarray(np.arcsin(radius), dtype=np.float64)


def _rotated_offset(
    offset_rad: float, parallactic_angle_rad: float
) -> tuple[float, float]:
    return (
        offset_rad * np.cos(parallactic_angle_rad),
        offset_rad * np.sin(parallactic_angle_rad),
    )


@dataclass(frozen=True)
class VLABeamCatalog:
    """Pinned VLA geometric beam constants."""

    dish_diameter_m: float = VLA_DISH_DIAMETER_M
    blockage_diameter_m: float = VLA_BLOCKAGE_DIAMETER_M
    gaussian_fwhm_factor: float = VLA_GAUSSIAN_FWHM_FACTOR
    airy_max_radius_rad_at_1ghz: float = VLA_AIRY_MAX_RADIUS_RAD_AT_1GHZ
    squint_fwhm_fraction: float = VLA_SQUINT_FWHM_FRACTION
    squint_reference_hz: float = VLA_SQUINT_REFERENCE_HZ

    def __post_init__(self) -> None:
        if self.dish_diameter_m <= 0 or self.blockage_diameter_m < 0:
            raise ValueError("dish and blockage diameters must be non-negative")
        if self.blockage_diameter_m >= self.dish_diameter_m:
            raise ValueError("blockage must be smaller than the dish")
        if self.gaussian_fwhm_factor <= 0:
            raise ValueError("gaussian_fwhm_factor must be positive")
        if self.airy_max_radius_rad_at_1ghz <= 0:
            raise ValueError("airy_max_radius_rad_at_1ghz must be positive")
        if self.squint_fwhm_fraction < 0:
            raise ValueError("squint_fwhm_fraction must be non-negative")
        if self.squint_reference_hz <= 0:
            raise ValueError("squint_reference_hz must be positive")

    def gaussian_fwhm_rad(self, frequency_hz: ArrayLike) -> NDArray[np.float64]:
        frequency = np.asarray(frequency_hz, dtype=np.float64)
        if np.any(frequency <= 0):
            raise ValueError("frequency must be positive")
        return (
            self.gaussian_fwhm_factor
            * SPEED_OF_LIGHT_M_S
            / (frequency * self.dish_diameter_m)
        )

    def squint_offset_rad(self, frequency_hz: ArrayLike) -> NDArray[np.float64]:
        return self.squint_fwhm_fraction * self.gaussian_fwhm_rad(frequency_hz)

    def airy_max_radius_rad(self, frequency_hz: ArrayLike) -> NDArray[np.float64]:
        frequency = np.asarray(frequency_hz, dtype=np.float64)
        if np.any(frequency <= 0):
            raise ValueError("frequency must be positive")
        return self.airy_max_radius_rad_at_1ghz * (
            self.squint_reference_hz / frequency
        )


@dataclass(frozen=True)
class VLAPrimaryBeam:
    """Analytic VLA power beam, optionally squinted for RR/LL."""

    kind: PrimaryBeamKind = "airy"
    apply_squint: bool = False
    pointing_lm: tuple[float, float] = (0.0, 0.0)
    parallactic_angle_rad: float = 0.0
    catalog: VLABeamCatalog = VLABeamCatalog()

    def __post_init__(self) -> None:
        if self.kind not in {"gaussian", "airy"}:
            raise ValueError("kind must be gaussian or airy")
        if not np.all(np.isfinite(self.pointing_lm)):
            raise ValueError("pointing_lm must be finite")

    def power(
        self,
        l: ArrayLike,
        m: ArrayLike,
        frequency_hz: ArrayLike,
        *,
        receptor: BeamReceptor = "I",
    ) -> NDArray[np.float64]:
        """Return the power beam on broadcastable ``(l, m, frequency)`` samples."""

        if receptor not in {"I", "RR", "LL"}:
            raise ValueError("receptor must be I, RR, or LL")
        l_array, m_array, frequency = np.broadcast_arrays(
            np.asarray(l, dtype=np.float64),
            np.asarray(m, dtype=np.float64),
            np.asarray(frequency_hz, dtype=np.float64),
        )
        if receptor == "I" and not self.apply_squint:
            return self._centered_power(
                l_array - self.pointing_lm[0],
                m_array - self.pointing_lm[1],
                frequency,
            )
        rr = self._receptor_power(l_array, m_array, frequency, sign=1.0)
        ll = self._receptor_power(l_array, m_array, frequency, sign=-1.0)
        if receptor == "RR":
            return rr
        if receptor == "LL":
            return ll
        return 0.5 * (rr + ll)

    def power_weights(
        self,
        l: ArrayLike,
        m: ArrayLike,
        frequency_hz: ArrayLike,
        *,
        receptor: BeamReceptor = "I",
    ) -> NDArray[np.float64]:
        """Return ``(pixel, channel)`` weights for a Stokes-I grid."""

        l_array = np.asarray(l, dtype=np.float64).ravel()
        m_array = np.asarray(m, dtype=np.float64).ravel()
        frequency = np.asarray(frequency_hz, dtype=np.float64).ravel()
        if l_array.size != m_array.size:
            raise ValueError("l and m must have the same size")
        if frequency.size == 0:
            raise ValueError("frequency_hz must contain at least one channel")
        return self.power(
            l_array[:, None],
            m_array[:, None],
            frequency[None, :],
            receptor=receptor,
        )

    def _receptor_power(
        self,
        l: np.ndarray,
        m: np.ndarray,
        frequency: np.ndarray,
        *,
        sign: float,
    ) -> NDArray[np.float64]:
        if not self.apply_squint:
            return self._centered_power(
                l - self.pointing_lm[0],
                m - self.pointing_lm[1],
                frequency,
            )
        offset = self.catalog.squint_offset_rad(frequency)
        east, north = _rotated_offset(1.0, self.parallactic_angle_rad)
        return self._centered_power(
            l - self.pointing_lm[0] - sign * offset * east,
            m - self.pointing_lm[1] - sign * offset * north,
            frequency,
        )

    def _centered_power(
        self, l: np.ndarray, m: np.ndarray, frequency: np.ndarray
    ) -> NDArray[np.float64]:
        if self.kind == "gaussian":
            return _gaussian_power(l, m, frequency, self.catalog)
        return _airy_power(l, m, frequency, self.catalog)


def _gaussian_power(
    l: np.ndarray,
    m: np.ndarray,
    frequency: np.ndarray,
    catalog: VLABeamCatalog,
) -> NDArray[np.float64]:
    angular_radius = _angular_radius(l, m)
    fwhm = catalog.gaussian_fwhm_rad(frequency)
    return np.asarray(
        np.exp(-4 * np.log(2) * np.square(angular_radius / fwhm)),
        dtype=np.float64,
    )


def _airy_power(
    l: np.ndarray,
    m: np.ndarray,
    frequency: np.ndarray,
    catalog: VLABeamCatalog,
) -> NDArray[np.float64]:
    angular_radius = _angular_radius(l, m)
    wavelength = SPEED_OF_LIGHT_M_S / frequency
    argument = (
        np.pi * catalog.dish_diameter_m * np.sin(angular_radius) / wavelength
    )
    blockage_ratio = catalog.blockage_diameter_m / catalog.dish_diameter_m
    voltage = (
        _jinc(argument) - blockage_ratio**2 * _jinc(blockage_ratio * argument)
    ) / (1.0 - blockage_ratio**2)
    power = np.square(voltage)
    max_radius = catalog.airy_max_radius_rad(frequency)
    return np.asarray(np.where(angular_radius <= max_radius, power, 0.0), dtype=np.float64)


def gaussian_primary_beam(
    l: ArrayLike,
    m: ArrayLike,
    frequency_hz: ArrayLike,
    *,
    dish_diameter_m: float,
    fwhm_factor: float = VLA_GAUSSIAN_FWHM_FACTOR,
) -> NDArray[np.float64]:
    """Return an idealized circular Gaussian power-beam attenuation."""

    beam = VLAPrimaryBeam(
        kind="gaussian",
        catalog=VLABeamCatalog(
            dish_diameter_m=dish_diameter_m,
            gaussian_fwhm_factor=fwhm_factor,
        ),
    )
    return beam.power(l, m, frequency_hz)


def predict_beam_weights(
    beam: VLAPrimaryBeam | None,
    l: ArrayLike,
    m: ArrayLike,
    frequency_hz: ArrayLike,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Return I or RR/LL power weights for the visibility predict path."""

    if beam is None:
        return None, None, None
    if beam.apply_squint:
        return (
            None,
            beam.power_weights(l, m, frequency_hz, receptor="RR"),
            beam.power_weights(l, m, frequency_hz, receptor="LL"),
        )
    return beam.power_weights(l, m, frequency_hz, receptor="I"), None, None


def primary_beam_from_name(
    name: str,
    *,
    apply_squint: bool = False,
    pointing_lm: tuple[float, float] = (0.0, 0.0),
) -> VLAPrimaryBeam | None:
    """Construct a public VLA beam, or ``None`` for an unattenuated image."""

    if name in {"none", "off"}:
        return None
    if name == "gaussian":
        return VLAPrimaryBeam(
            kind="gaussian", apply_squint=apply_squint, pointing_lm=pointing_lm
        )
    if name == "airy":
        return VLAPrimaryBeam(
            kind="airy", apply_squint=apply_squint, pointing_lm=pointing_lm
        )
    raise ValueError(f"unknown primary beam {name!r}")
