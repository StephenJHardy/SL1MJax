"""Voltage-Jones beam contract and scalar C-band backends.

Phase 2 wraps the accepted Airy power path as a diagonal voltage Jones.
Phase 3 evaluates the Perley 2016 C-band polynomial as a fail-closed
main-beam backend. The composite is an in-band spatial ablation only:
Airy may cover directions outside the Perley 5% radius, never frequencies
outside the Table 5 interval. The main/outer handover must be chosen
explicitly.

Phase 5 adds ``DiagonalSquintVoltageBeam``: Memo 195 R/L offsets rotated
by ``χ``. The unused Airy ``0.06`` FWHM half-offset is refused.

The visibility predict path still uses ``VLAPrimaryBeam``. These backends
are the reference evaluator, not a cache and not the streamed operator.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sl1mjax.beam import VLABeamCatalog, _airy_voltage
from sl1mjax.beam_conventions import (
    CIRCULAR_P_JONES,
    CIRCULAR_STOKES,
    JONES_RECEPTOR_ORDER,
    ON_AXIS_DI_JONES_ORDER,
    PERLEY2016_CATALOG_VERSION,
    PERLEY2016_MINIMUM_VALID_POWER,
    PERLEY2016_SOURCE_URL,
    BeamCalibrationState,
    PerleyFrequencyPolicy,
    SquintMagnitudePolicy,
    beam_requires_identity_on_axis,
    perley2016_frequency_is_supported,
    perley2016_frequency_support_hz,
    perley2016_stokes_i_power,
    perley2016_stokes_i_validity,
    receptor_squint_offset_lm_rad,
    require_beam_calibration_state,
    select_perley2016_cband_window,
    squint_receptor_half_offset_rad,
)
from sl1mjax.polarization import Receptor

JONES_AXES = ("antenna", "direction", "channel", "receptor_out", "receptor_in")
PERLEY_VOLTAGE_PHASE_CONVENTION = "real_nonnegative_sqrt_power"
AIRY_VOLTAGE_PHASE_CONVENTION = "signed_blocked_aperture_voltage"


class CompositeHandoverPolicy(StrEnum):
    """How the C-band main beam meets the analytic outer ablation.

    The outer Airy is only a spatial fallback inside the Perley frequency
    support. Out-of-band frequencies stay unsupported.

    ``match_power`` scales the Airy voltage so power is continuous at the
    Perley 5% radius. That scale is an amplitude match, not a physical
    far-sidelobe model. ``hard_splice`` is the discontinuous join and must
    be requested explicitly.
    """

    MATCH_POWER = "match_power"
    HARD_SPLICE = "hard_splice"


@dataclass(frozen=True)
class BeamCoordinates:
    """Sky-frame samples for one voltage-beam evaluation.

    ``l_rad`` and ``m_rad`` are direction cosines, not small-angle
    approximations. They are offsets from the pointing centre after
    ``pointing_offset_lm_rad`` is subtracted. Array-average backends ignore
    ``antenna_id`` and ``parallactic_angle_rad`` and must say so.
    """

    l_rad: np.ndarray
    m_rad: np.ndarray
    frequency_hz: np.ndarray
    parallactic_angle_rad: np.ndarray
    antenna_id: np.ndarray | None = None
    pointing_offset_lm_rad: np.ndarray | None = None
    elevation_rad: np.ndarray | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "l_rad", _as_1d(self.l_rad, "l_rad"))
        object.__setattr__(self, "m_rad", _as_1d(self.m_rad, "m_rad"))
        object.__setattr__(
            self, "frequency_hz", _as_1d(self.frequency_hz, "frequency_hz")
        )
        object.__setattr__(
            self,
            "parallactic_angle_rad",
            _as_1d(self.parallactic_angle_rad, "parallactic_angle_rad"),
        )
        if self.l_rad.size != self.m_rad.size:
            raise ValueError("l_rad and m_rad must have the same size")
        if self.frequency_hz.size == 0:
            raise ValueError("frequency_hz must contain at least one channel")
        if self.antenna_id is not None:
            antenna_id = np.asarray(self.antenna_id)
            if antenna_id.ndim != 1 or antenna_id.size == 0:
                raise ValueError("antenna_id must be a nonempty 1-d array")
            object.__setattr__(self, "antenna_id", antenna_id)
        if self.pointing_offset_lm_rad is not None:
            pointing = np.asarray(self.pointing_offset_lm_rad, dtype=np.float64)
            if pointing.shape != (2,):
                raise ValueError("array-average pointing_offset_lm_rad must have shape (2,)")
            if not np.all(np.isfinite(pointing)):
                raise ValueError("pointing_offset_lm_rad must be finite")
            object.__setattr__(self, "pointing_offset_lm_rad", pointing)
        if self.elevation_rad is not None:
            object.__setattr__(
                self, "elevation_rad", _as_1d(self.elevation_rad, "elevation_rad")
            )


@dataclass(frozen=True)
class BeamEvaluation:
    """Voltage Jones on the contract axes ``(antenna, direction, channel, 2, 2)``.

    ``off_diagonal_valid`` is the leakage support mask. It defaults to
    ``valid``, which is correct for diagonal models whose zeros are the
    model. Full Jones must clear it where leakage is unsupported so a
    later Stokes operator cannot treat those zeros as known leakage.
    """

    jones: np.ndarray
    valid: np.ndarray
    provenance: dict[str, object]
    off_diagonal_valid: np.ndarray | None = None

    def __post_init__(self) -> None:
        jones = np.asarray(self.jones)
        valid = np.asarray(self.valid, dtype=bool)
        if jones.ndim != 5 or jones.shape[-2:] != (2, 2):
            raise ValueError("jones must have shape (antenna, direction, channel, 2, 2)")
        if valid.shape != jones.shape[:3]:
            raise ValueError("valid must match jones (antenna, direction, channel)")
        if self.off_diagonal_valid is None:
            off_diagonal_valid = valid
        else:
            off_diagonal_valid = np.asarray(self.off_diagonal_valid, dtype=bool)
            if off_diagonal_valid.shape != valid.shape:
                raise ValueError(
                    "off_diagonal_valid must match valid "
                    "(antenna, direction, channel)"
                )
        object.__setattr__(self, "jones", jones)
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "off_diagonal_valid", off_diagonal_valid)


class VoltageBeamModel(Protocol):
    """Public voltage-beam evaluator."""

    model_id: str

    def evaluate(
        self,
        coordinates: BeamCoordinates,
        *,
        calibration_state: BeamCalibrationState | str,
    ) -> BeamEvaluation: ...


@dataclass(frozen=True)
class AnalyticAiryVoltageBeam:
    """Diagonal voltage Jones with exact Stokes-I parity to ``VLAPrimaryBeam``."""

    model_id: str = "analytic_blocked_airy"
    catalog: VLABeamCatalog = VLABeamCatalog()

    def evaluate(
        self,
        coordinates: BeamCoordinates,
        *,
        calibration_state: BeamCalibrationState | str,
    ) -> BeamEvaluation:
        state = require_beam_calibration_state(calibration_state)
        l_off, m_off = _pointing_relative_lm(coordinates)
        frequency = coordinates.frequency_hz
        voltage, visible = _airy_voltage_field(
            l_off[:, None], m_off[:, None], frequency[None, :], self.catalog
        )
        jones = _diagonal_voltage_jones(voltage[None, ...])
        valid = visible[None, ...]
        if beam_requires_identity_on_axis(state):
            _assert_identity_on_axis(jones, valid, l_off, m_off)
        return BeamEvaluation(
            jones=jones,
            valid=valid,
            provenance=_scalar_provenance(
                model_id=self.model_id,
                artifact_id="sl1mjax_airy",
                kind="analytic",
                support_class="analytic",
                calibration_state=state,
                frequency_policy=None,
                voltage_phase_convention=AIRY_VOLTAGE_PHASE_CONVENTION,
                ignored_coordinates=("antenna_id", "parallactic_angle_rad", "elevation_rad"),
                catalog={
                    "dish_diameter_m": self.catalog.dish_diameter_m,
                    "blockage_diameter_m": self.catalog.blockage_diameter_m,
                    "airy_max_radius_rad_at_1ghz": float(
                        self.catalog.airy_max_radius_rad_at_1ghz
                    ),
                },
            ),
        )


@dataclass(frozen=True)
class Perley2016CBandVoltageBeam:
    """Empirical C-band Stokes-I voltage beam from EVLA Memo 195 Table 5."""

    model_id: str = "perley2016_cband_stokes_i"
    frequency_policy: PerleyFrequencyPolicy = PerleyFrequencyPolicy.CASA_NEAREST

    def __post_init__(self) -> None:
        if self.frequency_policy is PerleyFrequencyPolicy.INTERPOLATED:
            raise ValueError(
                "Perley frequency interpolation is not implemented; refuse to invent a spectrum"
            )
        if self.frequency_policy is not PerleyFrequencyPolicy.CASA_NEAREST:
            raise ValueError(f"unknown Perley frequency policy {self.frequency_policy!r}")

    def evaluate(
        self,
        coordinates: BeamCoordinates,
        *,
        calibration_state: BeamCalibrationState | str,
    ) -> BeamEvaluation:
        state = require_beam_calibration_state(calibration_state)
        l_off, m_off = _pointing_relative_lm(coordinates)
        frequency = coordinates.frequency_hz
        voltage, valid, selected_hz = _perley_voltage_field(
            l_off[:, None],
            m_off[:, None],
            frequency[None, :],
            self.frequency_policy,
        )
        jones = _diagonal_voltage_jones(voltage[None, ...])
        if beam_requires_identity_on_axis(state):
            _assert_identity_on_axis(jones, valid[None, ...], l_off, m_off)
        low, high = perley2016_frequency_support_hz()
        return BeamEvaluation(
            jones=jones,
            valid=valid[None, ...],
            provenance=_scalar_provenance(
                model_id=self.model_id,
                artifact_id="perley2016_cband_stokes_i",
                kind="empirical",
                support_class="measured",
                calibration_state=state,
                frequency_policy=self.frequency_policy.value,
                voltage_phase_convention=PERLEY_VOLTAGE_PHASE_CONVENTION,
                ignored_coordinates=("antenna_id", "parallactic_angle_rad", "elevation_rad"),
                catalog={
                    "catalog_version": PERLEY2016_CATALOG_VERSION,
                    "source_url": PERLEY2016_SOURCE_URL,
                    "minimum_valid_power": PERLEY2016_MINIMUM_VALID_POWER,
                    "frequency_support_hz": [low, high],
                    "selected_window_hz": selected_hz,
                    "coefficient_source": (
                        "EVLA Memo 195 Table 5; CASA PBMath1DEVLA uses the same table"
                    ),
                },
            ),
        )


@dataclass(frozen=True)
class CompositeScalarVoltageBeam:
    """Perley C-band main beam with an analytic Airy outer-field ablation.

    The Airy fallback is spatial and in-band only. Frequencies outside the
    Perley catalog remain unsupported. ``handover`` must be chosen explicitly.
    """

    main: Perley2016CBandVoltageBeam
    outer: AnalyticAiryVoltageBeam
    handover: CompositeHandoverPolicy
    model_id: str = "perley2016_main_airy_outer"

    def evaluate(
        self,
        coordinates: BeamCoordinates,
        *,
        calibration_state: BeamCalibrationState | str,
    ) -> BeamEvaluation:
        state = require_beam_calibration_state(calibration_state)
        main = self.main.evaluate(coordinates, calibration_state=state)
        outer = self.outer.evaluate(coordinates, calibration_state=state)
        in_band = perley2016_frequency_is_supported(coordinates.frequency_hz)
        in_band_grid = in_band[None, None, :]
        use_main = main.valid
        use_outer = in_band_grid & ~use_main & outer.valid
        scales = np.ones(coordinates.frequency_hz.size, dtype=np.float64)
        outer_jones = outer.jones
        if self.handover is CompositeHandoverPolicy.MATCH_POWER:
            scales = _airy_power_match_scales(coordinates, self.outer, state)
            usable_scale = np.isfinite(scales) & (scales > 0.0)
            use_outer = use_outer & usable_scale[None, None, :]
            outer_jones = outer.jones * scales[None, None, :, None, None]
        elif self.handover is not CompositeHandoverPolicy.HARD_SPLICE:
            raise ValueError(f"unknown composite handover {self.handover!r}")
        jones = np.where(use_main[..., None, None], main.jones, 0.0)
        jones = np.where(use_outer[..., None, None], outer_jones, jones)
        valid = use_main | use_outer
        provenance = _scalar_provenance(
            model_id=self.model_id,
            artifact_id="perley2016_main_airy_outer",
            kind="empirical",
            support_class="measured_with_analytic_outer",
            calibration_state=state,
            frequency_policy=self.main.frequency_policy.value,
            voltage_phase_convention={
                "main": PERLEY_VOLTAGE_PHASE_CONVENTION,
                "outer": AIRY_VOLTAGE_PHASE_CONVENTION,
            },
            ignored_coordinates=("antenna_id", "parallactic_angle_rad", "elevation_rad"),
            catalog={
                "main": main.provenance,
                "outer": outer.provenance,
                "handover": self.handover.value,
                "outer_rule": (
                    "analytic Airy only for in-band directions outside Perley "
                    "spatial support"
                ),
                "edge_power_scale": scales.tolist(),
            },
        )
        return BeamEvaluation(jones=jones, valid=valid, provenance=provenance)


@dataclass(frozen=True)
class DiagonalSquintVoltageBeam:
    """Diagonal R/L voltage Jones with Memo 195 squint and χ rotation.

    The unused Airy ``0.06`` FWHM half-offset is refused. Feed-frame PA and
    R/L sign stay physically unverified. Imaging still uses the static
    unsquinted Airy path.
    """

    shape: (
        AnalyticAiryVoltageBeam
        | Perley2016CBandVoltageBeam
        | CompositeScalarVoltageBeam
    )
    magnitude: SquintMagnitudePolicy = SquintMagnitudePolicy.EVLA195
    model_id: str = "evla195_diagonal_squint"
    antenna_planes_from_parallactic: bool = True

    def __post_init__(self) -> None:
        squint_receptor_half_offset_rad(1.0e9, policy=self.magnitude)

    def evaluate(
        self,
        coordinates: BeamCoordinates,
        *,
        calibration_state: BeamCalibrationState | str,
    ) -> BeamEvaluation:
        state = require_beam_calibration_state(calibration_state)
        l_off, m_off = _pointing_relative_lm(coordinates)
        frequency = coordinates.frequency_hz
        half = squint_receptor_half_offset_rad(frequency, policy=self.magnitude)
        chi = _squint_parallactic_angles(coordinates)
        dl_r, dm_r = receptor_squint_offset_lm_rad(
            half, chi, receptor=Receptor.R
        )
        dl_l, dm_l = receptor_squint_offset_lm_rad(
            half, chi, receptor=Receptor.L
        )
        frequency_grid = frequency[None, None, :]
        l_grid = l_off[None, :, None]
        m_grid = m_off[None, :, None]
        voltage_r, valid_r = _shape_voltage_field(
            self.shape,
            l_grid - dl_r[:, None, :],
            m_grid - dm_r[:, None, :],
            frequency_grid,
            calibration_state=state,
        )
        voltage_l, valid_l = _shape_voltage_field(
            self.shape,
            l_grid - dl_l[:, None, :],
            m_grid - dm_l[:, None, :],
            frequency_grid,
            calibration_state=state,
        )
        valid = valid_r & valid_l
        if beam_requires_identity_on_axis(state):
            voltage_r, voltage_l, valid = _normalize_squint_on_axis(
                self.shape,
                voltage_r,
                voltage_l,
                valid,
                dl_r,
                dm_r,
                dl_l,
                dm_l,
                frequency,
                state,
            )
        jones = _diagonal_receptor_jones(voltage_r, voltage_l)
        if beam_requires_identity_on_axis(state):
            _assert_identity_on_axis(jones, valid, l_off, m_off)
        return BeamEvaluation(
            jones=jones,
            valid=valid,
            provenance=_scalar_provenance(
                model_id=self.model_id,
                artifact_id="evla195_diagonal_squint",
                kind=_shape_kind(self.shape),
                support_class=_shape_support(self.shape),
                calibration_state=state,
                frequency_policy=_shape_frequency_policy(self.shape),
                voltage_phase_convention=_shape_phase_convention(self.shape),
                ignored_coordinates=("elevation_rad",),
                creates_i_to_v=True,
                catalog={
                    "shape_model_id": self.shape.model_id,
                    "magnitude_policy": self.magnitude.value,
                    "magnitude_quantity": "total_rcp_lcp_separation",
                    "receptor_offset_quantity": "receptor_half_offset",
                    "total_separation_arcmin_ghz": 2.4,
                    "rotation": "internal_chi_zero_places_r_at_plus_l",
                    "feed_frame_polarization": "physically_unverified",
                    "legacy_analytic_half_offset_enabled": False,
                },
            ),
        )


def beam_coordinates(
    l_rad: ArrayLike,
    m_rad: ArrayLike,
    frequency_hz: ArrayLike,
    *,
    parallactic_angle_rad: ArrayLike = 0.0,
    pointing_offset_lm_rad: ArrayLike | None = None,
    antenna_id: ArrayLike | None = None,
    elevation_rad: ArrayLike | None = None,
) -> BeamCoordinates:
    """Build sky-frame beam coordinates for a scalar array-average evaluation."""

    pointing = (
        None
        if pointing_offset_lm_rad is None
        else np.asarray(pointing_offset_lm_rad, dtype=np.float64)
    )
    return BeamCoordinates(
        l_rad=np.asarray(l_rad, dtype=np.float64),
        m_rad=np.asarray(m_rad, dtype=np.float64),
        frequency_hz=np.asarray(frequency_hz, dtype=np.float64),
        parallactic_angle_rad=np.asarray(parallactic_angle_rad, dtype=np.float64),
        antenna_id=None if antenna_id is None else np.asarray(antenna_id),
        pointing_offset_lm_rad=pointing,
        elevation_rad=(
            None if elevation_rad is None else np.asarray(elevation_rad, dtype=np.float64)
        ),
    )


def stokes_i_power_from_jones(jones: ArrayLike) -> NDArray[np.float64]:
    """Return apparent Stokes I for an unpolarised source, ``½ tr(E Eᴴ)``."""

    matrices = np.asarray(jones)
    return np.asarray(0.5 * np.sum(np.square(np.abs(matrices)), axis=(-2, -1)), dtype=np.float64)


def _as_1d(values: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} must contain at least one sample")
    return array


def _pointing_relative_lm(coordinates: BeamCoordinates) -> tuple[np.ndarray, np.ndarray]:
    pointing = (
        np.zeros(2, dtype=np.float64)
        if coordinates.pointing_offset_lm_rad is None
        else coordinates.pointing_offset_lm_rad
    )
    return coordinates.l_rad - pointing[0], coordinates.m_rad - pointing[1]


def _diagonal_voltage_jones(voltage: np.ndarray) -> NDArray[np.complex128]:
    """Pack a real voltage as ``diag(E, E)`` on the contract axes."""

    return _diagonal_receptor_jones(voltage, voltage)


def _diagonal_receptor_jones(
    voltage_r: np.ndarray, voltage_l: np.ndarray
) -> NDArray[np.complex128]:
    """Pack receptor voltages as ``diag(E_R, E_L)`` on the contract axes."""

    jones = np.zeros(voltage_r.shape + (2, 2), dtype=np.complex128)
    jones[..., 0, 0] = voltage_r
    jones[..., 1, 1] = voltage_l
    return jones


def _airy_voltage_field(
    l: np.ndarray,
    m: np.ndarray,
    frequency: np.ndarray,
    catalog: VLABeamCatalog,
) -> tuple[np.ndarray, np.ndarray]:
    l_grid, m_grid, frequency_grid = np.broadcast_arrays(l, m, frequency)
    visible = (
        np.isfinite(l_grid)
        & np.isfinite(m_grid)
        & np.isfinite(frequency_grid)
        & (frequency_grid > 0.0)
        & (np.hypot(l_grid, m_grid) < 1.0)
    )
    l_safe = np.where(visible, l_grid, 0.0)
    m_safe = np.where(visible, m_grid, 0.0)
    voltage = np.where(
        visible, _airy_voltage(l_safe, m_safe, frequency_grid, catalog), 0.0
    )
    return np.asarray(voltage, dtype=np.float64), np.asarray(visible, dtype=bool)


def _perley_voltage_field(
    l: np.ndarray,
    m: np.ndarray,
    frequency: np.ndarray,
    policy: PerleyFrequencyPolicy,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    l_grid, m_grid, frequency_grid = np.broadcast_arrays(l, m, frequency)
    voltage = np.zeros(l_grid.shape, dtype=np.float64)
    valid = np.zeros(l_grid.shape, dtype=bool)
    selected_hz: list[float] = []
    channel_count = frequency_grid.shape[-1]
    for channel in range(channel_count):
        frequency_hz = float(np.reshape(frequency_grid[..., channel], -1)[0])
        try:
            window = select_perley2016_cband_window(frequency_hz, policy=policy)
        except ValueError:
            selected_hz.append(float("nan"))
            continue
        selected_hz.append(window.frequency_hz)
        radius = np.hypot(l_grid[..., channel], m_grid[..., channel])
        offset_arcmin = np.rad2deg(np.arcsin(np.minimum(radius, 1.0))) * 60.0
        hemisphere = radius < 1.0
        channel_valid = (
            perley2016_stokes_i_validity(offset_arcmin, frequency_hz, window)
            & hemisphere
            & np.isfinite(l_grid[..., channel])
            & np.isfinite(m_grid[..., channel])
        )
        power = perley2016_stokes_i_power(
            offset_arcmin, frequency_hz, window, require_valid=False
        )
        voltage[..., channel] = np.where(
            channel_valid, np.sqrt(np.maximum(power, 0.0)), 0.0
        )
        valid[..., channel] = channel_valid
    return voltage, valid, selected_hz


def _shape_voltage_field(
    shape: (
        AnalyticAiryVoltageBeam
        | Perley2016CBandVoltageBeam
        | CompositeScalarVoltageBeam
    ),
    l: np.ndarray,
    m: np.ndarray,
    frequency: np.ndarray,
    *,
    calibration_state: BeamCalibrationState,
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(shape, AnalyticAiryVoltageBeam):
        return _airy_voltage_field(l, m, frequency, shape.catalog)
    if isinstance(shape, Perley2016CBandVoltageBeam):
        voltage, valid, _selected = _perley_voltage_field(
            l, m, frequency, shape.frequency_policy
        )
        return voltage, valid
    return _composite_voltage_field(shape, l, m, frequency, calibration_state)


def _composite_voltage_field(
    shape: CompositeScalarVoltageBeam,
    l: np.ndarray,
    m: np.ndarray,
    frequency: np.ndarray,
    state: BeamCalibrationState,
) -> tuple[np.ndarray, np.ndarray]:
    main_voltage, main_valid, _selected = _perley_voltage_field(
        l, m, frequency, shape.main.frequency_policy
    )
    outer_voltage, outer_valid = _airy_voltage_field(
        l, m, frequency, shape.outer.catalog
    )
    channel_frequency = np.asarray(frequency, dtype=np.float64)
    freqs = channel_frequency.reshape(-1, channel_frequency.shape[-1])[0]
    in_band = perley2016_frequency_is_supported(freqs)
    in_band_grid = in_band.reshape((1,) * (main_valid.ndim - 1) + (-1,))
    use_main = main_valid
    use_outer = in_band_grid & ~use_main & outer_valid
    if shape.handover is CompositeHandoverPolicy.MATCH_POWER:
        scales = _airy_power_match_scales(
            beam_coordinates([0.0], [0.0], freqs), shape.outer, state
        )
        scale_grid = scales.reshape((1,) * (outer_voltage.ndim - 1) + (-1,))
        usable = np.isfinite(scales) & (scales > 0.0)
        use_outer = use_outer & usable.reshape(scale_grid.shape)
        outer_voltage = outer_voltage * scale_grid
    elif shape.handover is not CompositeHandoverPolicy.HARD_SPLICE:
        raise ValueError(f"unknown composite handover {shape.handover!r}")
    voltage = np.where(use_main, main_voltage, 0.0)
    voltage = np.where(use_outer, outer_voltage, voltage)
    return np.asarray(voltage, dtype=np.float64), np.asarray(
        use_main | use_outer, dtype=bool
    )


def _shape_kind(
    shape: AnalyticAiryVoltageBeam | Perley2016CBandVoltageBeam | CompositeScalarVoltageBeam,
) -> str:
    return "analytic" if isinstance(shape, AnalyticAiryVoltageBeam) else "empirical"


def _shape_support(
    shape: AnalyticAiryVoltageBeam | Perley2016CBandVoltageBeam | CompositeScalarVoltageBeam,
) -> str:
    if isinstance(shape, AnalyticAiryVoltageBeam):
        return "analytic"
    if isinstance(shape, CompositeScalarVoltageBeam):
        return "measured_with_analytic_outer"
    return "measured"


def _shape_frequency_policy(
    shape: AnalyticAiryVoltageBeam | Perley2016CBandVoltageBeam | CompositeScalarVoltageBeam,
) -> str | None:
    if isinstance(shape, AnalyticAiryVoltageBeam):
        return None
    if isinstance(shape, CompositeScalarVoltageBeam):
        return shape.main.frequency_policy.value
    return shape.frequency_policy.value


def _shape_phase_convention(
    shape: AnalyticAiryVoltageBeam | Perley2016CBandVoltageBeam | CompositeScalarVoltageBeam,
) -> str | dict[str, str]:
    if isinstance(shape, AnalyticAiryVoltageBeam):
        return AIRY_VOLTAGE_PHASE_CONVENTION
    if isinstance(shape, CompositeScalarVoltageBeam):
        return {
            "main": PERLEY_VOLTAGE_PHASE_CONVENTION,
            "outer": AIRY_VOLTAGE_PHASE_CONVENTION,
        }
    return PERLEY_VOLTAGE_PHASE_CONVENTION


def _squint_parallactic_angles(coordinates: BeamCoordinates) -> np.ndarray:
    chi = np.asarray(coordinates.parallactic_angle_rad, dtype=np.float64)
    if coordinates.antenna_id is None:
        return chi if chi.size == 1 else np.asarray([float(chi[0])], dtype=np.float64)
    n_antenna = coordinates.antenna_id.size
    if chi.size == 1:
        return np.full(n_antenna, float(chi[0]), dtype=np.float64)
    if chi.size != n_antenna:
        raise ValueError("parallactic_angle_rad must be scalar or one value per antenna")
    return chi


def _normalize_squint_on_axis(
    shape: (
        AnalyticAiryVoltageBeam
        | Perley2016CBandVoltageBeam
        | CompositeScalarVoltageBeam
    ),
    voltage_r: np.ndarray,
    voltage_l: np.ndarray,
    valid: np.ndarray,
    dl_r: np.ndarray,
    dm_r: np.ndarray,
    dl_l: np.ndarray,
    dm_l: np.ndarray,
    frequency: np.ndarray,
    state: BeamCalibrationState,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frequency_grid = frequency[None, None, :]
    e0_r, ok_r = _shape_voltage_field(
        shape,
        -dl_r[:, None, :],
        -dm_r[:, None, :],
        frequency_grid,
        calibration_state=state,
    )
    e0_l, ok_l = _shape_voltage_field(
        shape,
        -dl_l[:, None, :],
        -dm_l[:, None, :],
        frequency_grid,
        calibration_state=state,
    )
    usable = (
        ok_r[:, 0, :]
        & ok_l[:, 0, :]
        & np.isfinite(e0_r[:, 0, :])
        & np.isfinite(e0_l[:, 0, :])
        & (np.abs(e0_r[:, 0, :]) > 1.0e-12)
        & (np.abs(e0_l[:, 0, :]) > 1.0e-12)
    )
    if not np.all(usable):
        raise ValueError("cannot normalize squinted beam to E(0)=I")
    scale_r = e0_r[:, 0, :][:, None, :]
    scale_l = e0_l[:, 0, :][:, None, :]
    return voltage_r / scale_r, voltage_l / scale_l, valid


def _assert_identity_on_axis(
    jones: np.ndarray,
    valid: np.ndarray,
    l_off: np.ndarray,
    m_off: np.ndarray,
) -> None:
    on_axis = (np.abs(l_off) <= 1e-15) & (np.abs(m_off) <= 1e-15)
    if not np.any(on_axis):
        return
    selected = jones[:, on_axis, ...]
    selected_valid = valid[:, on_axis, ...]
    if not np.any(selected_valid):
        return
    identity = np.eye(2, dtype=np.complex128)
    if not np.allclose(selected[selected_valid], identity, atol=1e-12):
        raise ValueError("casa_parang_true requires E(0)=I")


def _airy_power_match_scales(
    coordinates: BeamCoordinates,
    outer: AnalyticAiryVoltageBeam,
    state: BeamCalibrationState,
) -> NDArray[np.float64]:
    """Return per-channel Airy voltage scales that match Perley at 5% power."""

    scales = np.full(coordinates.frequency_hz.size, np.nan, dtype=np.float64)
    for channel, frequency_hz in enumerate(coordinates.frequency_hz):
        if not bool(perley2016_frequency_is_supported(frequency_hz)):
            continue
        window = select_perley2016_cband_window(
            float(frequency_hz), policy=PerleyFrequencyPolicy.CASA_NEAREST
        )
        edge_l = np.sin(np.deg2rad(window.support_radius_arcmin(float(frequency_hz)) / 60.0))
        edge = outer.evaluate(
            beam_coordinates([edge_l], [0.0], [frequency_hz]),
            calibration_state=state,
        )
        airy_power = float(stokes_i_power_from_jones(edge.jones)[0, 0, 0])
        if bool(edge.valid[0, 0, 0]) and airy_power > 1.0e-12:
            scales[channel] = float(np.sqrt(PERLEY2016_MINIMUM_VALID_POWER / airy_power))
    return scales


def _scalar_provenance(
    *,
    model_id: str,
    artifact_id: str,
    kind: str,
    support_class: str,
    calibration_state: BeamCalibrationState,
    frequency_policy: str | None,
    voltage_phase_convention: str | dict[str, str],
    ignored_coordinates: tuple[str, ...],
    catalog: dict[str, object],
    creates_i_to_v: bool = False,
) -> dict[str, object]:
    return {
        "model_id": model_id,
        "artifact_id": artifact_id,
        "kind": kind,
        "support_class": support_class,
        "array_average": True,
        "jones_axes": list(JONES_AXES),
        "receptors": [receptor.value for receptor in JONES_RECEPTOR_ORDER],
        "receptor_basis": "circular",
        "direction_frame": "sky_direction_cosines",
        "voltage_phase_convention": voltage_phase_convention,
        "on_axis_normalization": "E(0)=I",
        "calibration_state": calibration_state.value,
        "on_axis_di_jones_order": ON_AXIS_DI_JONES_ORDER,
        "circular_stokes": CIRCULAR_STOKES,
        "circular_p_jones": CIRCULAR_P_JONES,
        "frequency_policy": frequency_policy,
        "ignored_coordinates": list(ignored_coordinates),
        "diagonal": True,
        "creates_i_to_qu": False,
        "creates_i_to_v": creates_i_to_v,
        "catalog": catalog,
    }
