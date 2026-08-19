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
class ObservationMetadata:
    antennas: tuple[AntennaRecord, ...] = ()
    fields: tuple[FieldRecord, ...] = ()
    states: tuple[StateRecord, ...] = ()
    observations: tuple[ObservationRecord, ...] = ()
    feeds: tuple[FeedRecord, ...] = ()
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
            states=tuple(StateRecord(**record) for record in value.get("states", ())),
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
            extras=dict(value.get("extras", {})),
        )

    def field_by_name(self, name: str) -> FieldRecord:
        matches = [field for field in self.fields if field.name == name]
        if len(matches) != 1:
            raise ValueError(f"expected one field named {name!r}; found {len(matches)}")
        return matches[0]
