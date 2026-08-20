"""Calibration-grade telescope and observation metadata records."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class CalibratorRole(StrEnum):
    TARGET = "target"
    FLUX = "flux"
    BANDPASS = "bandpass"
    PHASE = "phase"
    AMPLITUDE = "amplitude"
    POLARIZATION = "polarization"
    POINTING = "pointing"


@dataclass(frozen=True)
class AntennaRecord:
    antenna_id: int
    name: str
    station: str
    position_m: tuple[float, float, float]
    dish_diameter_m: float
    mount: str


@dataclass(frozen=True)
class FieldRecord:
    field_id: int
    name: str
    source_id: int
    phase_direction_rad: tuple[float, float]
    delay_direction_rad: tuple[float, float]
    reference_direction_rad: tuple[float, float]
    roles: tuple[CalibratorRole, ...] = ()


@dataclass(frozen=True)
class StateRecord:
    state_id: int
    observation_mode: str
    intents: tuple[str, ...] = ()
    sig: bool | None = None
    ref: bool | None = None
    cal: bool | None = None
    load: bool | None = None


@dataclass(frozen=True)
class ObservationRecord:
    observation_id: int
    telescope_name: str
    observer: str
    project: str
    time_range_s: tuple[float, float]


@dataclass(frozen=True)
class FeedRecord:
    antenna_id: int
    feed_id: int
    spectral_window_id: int
    receptor_types: tuple[str, ...]
    receptor_angles_rad: tuple[float, ...]


@dataclass(frozen=True)
class SpectralWindowRecord:
    spectral_window_id: int
    name: str
    reference_frequency_hz: float
    channel_frequencies_hz: tuple[float, ...]
    channel_widths_hz: tuple[float, ...]
    effective_bandwidths_hz: tuple[float, ...] = ()
    resolutions_hz: tuple[float, ...] = ()
    total_bandwidth_hz: float | None = None


@dataclass(frozen=True)
class DataDescriptionRecord:
    data_description_id: int
    spectral_window_id: int
    polarization_id: int


@dataclass(frozen=True)
class WeatherRecord:
    time_s: float
    interval_s: float
    antenna_id: int
    temperature_k: float | None = None
    dew_point_k: float | None = None
    pressure_pa: float | None = None
    relative_humidity: float | None = None
    wind_speed_m_s: float | None = None
    wind_direction_rad: float | None = None


@dataclass(frozen=True)
class SwitchedPowerRecord:
    time_s: float
    interval_s: float
    antenna_id: int
    feed_id: int
    spectral_window_id: int
    switched_diff: tuple[float, ...]
    switched_sum: tuple[float, ...]
    requantizer_gain: tuple[float, ...]


@dataclass(frozen=True)
class CalibrationDeviceRecord:
    time_s: float
    interval_s: float
    antenna_id: int
    feed_id: int
    spectral_window_id: int
    noise_cal_k: tuple[float, ...]
    calibration_efficiency: tuple[float, ...] = ()
    load_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class ObservationMetadata:
    antennas: tuple[AntennaRecord, ...] = ()
    fields: tuple[FieldRecord, ...] = ()
    states: tuple[StateRecord, ...] = ()
    observations: tuple[ObservationRecord, ...] = ()
    feeds: tuple[FeedRecord, ...] = ()
    spectral_windows: tuple[SpectralWindowRecord, ...] = ()
    data_descriptions: tuple[DataDescriptionRecord, ...] = ()
    weather: tuple[WeatherRecord, ...] = ()
    switched_power: tuple[SwitchedPowerRecord, ...] = ()
    calibration_devices: tuple[CalibrationDeviceRecord, ...] = ()
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> ObservationMetadata | None:
        if value is None:
            return None
        return cls(
            antennas=tuple(
                AntennaRecord(
                    **{
                        **record,
                        "position_m": tuple(record["position_m"]),
                    }
                )
                for record in value.get("antennas", ())
            ),
            fields=tuple(
                FieldRecord(
                    **{
                        **record,
                        "phase_direction_rad": tuple(record["phase_direction_rad"]),
                        "delay_direction_rad": tuple(record["delay_direction_rad"]),
                        "reference_direction_rad": tuple(
                            record["reference_direction_rad"]
                        ),
                        "roles": tuple(
                            CalibratorRole(role) for role in record.get("roles", ())
                        ),
                    }
                )
                for record in value.get("fields", ())
            ),
            states=tuple(
                StateRecord(
                    **{
                        **record,
                        "intents": tuple(record.get("intents", ())),
                    }
                )
                for record in value.get("states", ())
            ),
            observations=tuple(
                ObservationRecord(
                    **{
                        **record,
                        "time_range_s": tuple(record["time_range_s"]),
                    }
                )
                for record in value.get("observations", ())
            ),
            feeds=tuple(
                FeedRecord(
                    **{
                        **record,
                        "receptor_types": tuple(record["receptor_types"]),
                        "receptor_angles_rad": tuple(record["receptor_angles_rad"]),
                    }
                )
                for record in value.get("feeds", ())
            ),
            spectral_windows=tuple(
                SpectralWindowRecord(
                    **{
                        **record,
                        "channel_frequencies_hz": tuple(
                            record["channel_frequencies_hz"]
                        ),
                        "channel_widths_hz": tuple(record["channel_widths_hz"]),
                        "effective_bandwidths_hz": tuple(
                            record.get("effective_bandwidths_hz", ())
                        ),
                        "resolutions_hz": tuple(record.get("resolutions_hz", ())),
                    }
                )
                for record in value.get("spectral_windows", ())
            ),
            data_descriptions=tuple(
                DataDescriptionRecord(**record)
                for record in value.get("data_descriptions", ())
            ),
            weather=tuple(
                WeatherRecord(**record) for record in value.get("weather", ())
            ),
            switched_power=tuple(
                SwitchedPowerRecord(
                    **{
                        **record,
                        "switched_diff": tuple(record["switched_diff"]),
                        "switched_sum": tuple(record["switched_sum"]),
                        "requantizer_gain": tuple(record["requantizer_gain"]),
                    }
                )
                for record in value.get("switched_power", ())
            ),
            calibration_devices=tuple(
                CalibrationDeviceRecord(
                    **{
                        **record,
                        "noise_cal_k": tuple(record["noise_cal_k"]),
                        "calibration_efficiency": tuple(
                            record.get("calibration_efficiency", ())
                        ),
                        "load_names": tuple(record.get("load_names", ())),
                    }
                )
                for record in value.get("calibration_devices", ())
            ),
            extras=dict(value.get("extras", {})),
        )

    def field_by_name(self, name: str) -> FieldRecord:
        matches = [field for field in self.fields if field.name == name]
        if len(matches) != 1:
            raise ValueError(f"expected one field named {name!r}; found {len(matches)}")
        return matches[0]
