"""Independent generators for VLA a-priori calibration terms."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from sl1mjax.calibration_terms import (
    GainCurveTerm,
    OpacityTerm,
    RequantizerTerm,
    generate_requantizer_gain,
)
from sl1mjax.data.metadata import (
    ObservationMetadata,
    SpectralWindowRecord,
    WeatherRecord,
)

GAIN_CURVE_CATALOG_VERSION = "casa-data-65a746f9e666"
OPACITY_MODEL_VERSION = "evla-memo-143-plotweather-2024"


@dataclass(frozen=True)
class GainCurveCatalogEntry:
    """One date- and frequency-bounded observatory gain-curve record."""

    start_mjd: float
    stop_mjd: float
    minimum_frequency_hz: float
    maximum_frequency_hz: float
    coefficients: tuple[
        tuple[float, float, float, float],
        tuple[float, float, float, float],
    ]
    source: str
    antenna: str = "0"
    band: str = ""


@lru_cache(maxsize=1)
def load_vla_gain_curve_catalog() -> tuple[GainCurveCatalogEntry, ...]:
    """Load the pinned NRAO/CASA VLA coefficient release."""

    path = Path(__file__).with_name("data") / "vla_gain_curves.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("catalog_version") != GAIN_CURVE_CATALOG_VERSION:
        raise ValueError("unexpected VLA gain-curve catalog version")
    source = (
        f"{payload['source']}@{payload['source_commit']}"
    )

    def coefficients(
        value: list[list[float]],
    ) -> tuple[
        tuple[float, float, float, float],
        tuple[float, float, float, float],
    ]:
        if len(value) != 2 or any(len(receptor) != 4 for receptor in value):
            raise ValueError("invalid VLA gain-curve coefficient shape")
        return (
            (
                float(value[0][0]),
                float(value[0][1]),
                float(value[0][2]),
                float(value[0][3]),
            ),
            (
                float(value[1][0]),
                float(value[1][1]),
                float(value[1][2]),
                float(value[1][3]),
            ),
        )

    return tuple(
        GainCurveCatalogEntry(
            start_mjd=row["start_mjd"],
            stop_mjd=row["stop_mjd"],
            minimum_frequency_hz=row["minimum_frequency_hz"],
            maximum_frequency_hz=row["maximum_frequency_hz"],
            coefficients=coefficients(row["coefficients"]),
            source=source,
            antenna=row["antenna"],
            band=row["band"],
        )
        for row in payload["rows"]
    )


def generate_vla_gain_curve(
    metadata: ObservationMetadata,
    *,
    observation_time_s: float,
    receptor_count: int = 2,
    catalog: tuple[GainCurveCatalogEntry, ...] | None = None,
) -> GainCurveTerm:
    """Select versioned VLA coefficients by date and SPW frequency."""

    if not metadata.antennas or not metadata.spectral_windows:
        raise ValueError("gain-curve generation requires antenna and SPW metadata")
    if receptor_count != 2:
        raise ValueError("the VLA gain-curve catalog contains exactly two receptors")
    selected_catalog = load_vla_gain_curve_catalog() if catalog is None else catalog
    mjd = observation_time_s / 86400.0
    coefficients = np.ones(
        (
            len(metadata.antennas),
            len(metadata.spectral_windows),
            receptor_count,
            4,
        ),
        dtype=np.float64,
    )
    valid = np.zeros(coefficients.shape[:-1], dtype=bool)
    selected_sources: set[str] = set()
    for antenna_index, antenna in enumerate(metadata.antennas):
        match_name = re.sub(r"^[A-Za-z]+0*", "", antenna.name) or "0"
        for spw_index, spw in enumerate(metadata.spectral_windows):
            frequency = spw.reference_frequency_hz
            matches = [
                entry
                for entry in selected_catalog
                if entry.start_mjd <= mjd < entry.stop_mjd
                and entry.minimum_frequency_hz
                <= frequency
                < entry.maximum_frequency_hz
                and entry.antenna in {match_name, "0"}
            ]
            match = next(
                (entry for entry in matches if entry.antenna == match_name),
                next((entry for entry in matches if entry.antenna == "0"), None),
            )
            if match is None:
                continue
            coefficients[antenna_index, spw_index, :, :] = match.coefficients
            valid[antenna_index, spw_index, :] = True
            selected_sources.add(match.source)
    return GainCurveTerm(
        coefficients,
        np.asarray(
            [spw.spectral_window_id for spw in metadata.spectral_windows],
            dtype=np.int32,
        ),
        valid,
        {
            "generator": "sl1mjax",
            "catalog_version": GAIN_CURVE_CATALOG_VERSION,
            "sources": sorted(selected_sources),
            "observation_mjd": mjd,
        },
    )


def _mjd_seconds_to_datetime(value: float) -> datetime:
    return datetime(1858, 11, 17, tzinfo=UTC) + timedelta(seconds=value)


def _seasonal_pwv_mm(time_s: float) -> float:
    timestamp = _mjd_seconds_to_datetime(time_s)
    day = 30.0 * (timestamp.month - 1) + timestamp.day
    day += 5.0 * day / 365.0
    if day > 199.0:
        day -= 365.0
    modified_day = day + 165.0
    tau_22_percent = 22.1 - 0.178 * modified_day + 0.00044 * modified_day**2
    return -1.71 + 1.3647 * tau_22_percent


def _weather_pwv_mm(sample: WeatherRecord) -> float | None:
    if sample.temperature_k is None:
        return None
    temperature_c = sample.temperature_k - 273.15
    if sample.dew_point_k is not None:
        dew_point_c = sample.dew_point_k - 273.15
    elif sample.relative_humidity is not None and sample.relative_humidity > 0:
        humidity_percent = (
            sample.relative_humidity * 100.0
            if sample.relative_humidity <= 1.0
            else sample.relative_humidity
        )
        gamma = np.log(humidity_percent / 100.0) + (
            17.27 * temperature_c / (237.3 + temperature_c)
        )
        dew_point_c = 237.3 * gamma / (17.27 - gamma)
    else:
        return None
    vapor_pressure = np.exp(
        1.81 + 17.27 * dew_point_c / (dew_point_c + 237.3)
    )
    return float(324.7 * vapor_pressure / (temperature_c + 273.15))


def _opacity_from_pwv(frequency_hz: np.ndarray, pwv_mm: float) -> np.ndarray:
    """Portable approximation to the VLA atmospheric profile in EVLA Memo 143."""

    frequency_ghz = np.asarray(frequency_hz, dtype=np.float64) / 1e9
    # Dry continuum plus pressure-broadened O2 (60 GHz) and H2O (22.235 GHz).
    dry = 0.0051 + 1.5e-4 * frequency_ghz
    oxygen = 0.012 * (frequency_ghz / 60.0) ** 2 / (
        1.0 + ((frequency_ghz - 60.0) / 12.0) ** 2
    )
    water = (
        max(pwv_mm, 0.0)
        * 0.001
        / (1.0 + ((frequency_ghz - 22.235) / 3.5) ** 2)
    )
    return dry + oxygen + water


def estimate_vla_zenith_opacity(
    metadata: ObservationMetadata,
    *,
    observation_time_s: float,
    seasonal_weight: float = 0.5,
    receptor_count: int = 2,
) -> OpacityTerm:
    """Estimate per-SPW opacity from WEATHER and the VLA seasonal PWV model."""

    if not 0.0 <= seasonal_weight <= 1.0:
        raise ValueError("seasonal_weight must be between zero and one")
    if not metadata.antennas or not metadata.spectral_windows:
        raise ValueError("opacity generation requires antenna and SPW metadata")
    measured = [
        value
        for sample in metadata.weather
        if (value := _weather_pwv_mm(sample)) is not None and np.isfinite(value)
    ]
    seasonal_pwv = _seasonal_pwv_mm(observation_time_s)
    measured_pwv = float(np.median(measured)) if measured else None
    used_weight = seasonal_weight if measured_pwv is not None else 1.0
    pwv = (
        seasonal_pwv
        if measured_pwv is None
        else (1.0 - used_weight) * measured_pwv + used_weight * seasonal_pwv
    )
    spw_tau = np.asarray(
        [
            float(
                np.mean(
                    _opacity_from_pwv(
                        np.asarray(spw.channel_frequencies_hz)
                        if spw.channel_frequencies_hz
                        else np.asarray([spw.reference_frequency_hz]),
                        pwv,
                    )
                )
            )
            for spw in metadata.spectral_windows
        ]
    )
    tau = np.broadcast_to(
        spw_tau[None, :, None],
        (len(metadata.antennas), spw_tau.size, receptor_count),
    ).copy()
    valid = np.isfinite(tau) & (tau >= 0)
    return OpacityTerm(
        tau,
        np.asarray(
            [spw.spectral_window_id for spw in metadata.spectral_windows],
            dtype=np.int32,
        ),
        valid,
        {
            "generator": "sl1mjax",
            "model_version": OPACITY_MODEL_VERSION,
            "seasonal_weight_requested": seasonal_weight,
            "seasonal_weight_used": used_weight,
            "seasonal_pwv_mm": seasonal_pwv,
            "measured_pwv_mm": measured_pwv,
            "combined_pwv_mm": pwv,
            "uncertainty": "rough VLA weather/seasonal model; not radiometer measured",
        },
    )


def generate_vla_requantizer(
    metadata: ObservationMetadata,
    *,
    chunk_size: int = 65_536,
    provenance: dict[str, Any] | None = None,
) -> RequantizerTerm:
    """Generate RQ-only voltage gains in bounded-memory chunks."""

    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    records = metadata.switched_power
    if not records:
        raise ValueError("requantizer generation requires SYSPOWER records")
    if not metadata.calibration_devices:
        raise ValueError("requantizer generation requires the CALDEVICE subtable")
    receptor_count = max(len(record.requantizer_gain) for record in records)
    if receptor_count < 1:
        raise ValueError("SYSPOWER has no REQUANTIZER_GAIN samples")
    gains: list[np.ndarray] = []
    validity: list[np.ndarray] = []
    for start in range(0, len(records), chunk_size):
        chunk = records[start : start + chunk_size]
        values = np.full((len(chunk), receptor_count), np.nan)
        for row, record in enumerate(chunk):
            values[row, : len(record.requantizer_gain)] = record.requantizer_gain
        gains.append(values)
        validity.append(np.isfinite(values) & (values > 0))
    return generate_requantizer_gain(
        np.concatenate(gains),
        time_s=np.asarray([record.time_s for record in records]),
        interval_s=np.asarray([record.interval_s for record in records]),
        antenna_id=np.asarray([record.antenna_id for record in records]),
        spectral_window_id=np.asarray(
            [record.spectral_window_id for record in records]
        ),
        valid=np.concatenate(validity),
        provenance={
            "source": "MeasurementSet/SYSPOWER.REQUANTIZER_GAIN",
            "caldevice_available": True,
            "chunk_size": chunk_size,
            "convention": "CASA 6.5 RQ-only voltage gain; calwt=False",
            **({} if provenance is None else provenance),
        },
    )


def generate_vla_requantizer_from_ms(
    path: str | Path, *, chunk_size: int = 65_536
) -> RequantizerTerm:
    """Read an RQ-only term directly from large MS subtables in chunks."""

    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    try:
        from casacore import tables
    except ImportError as exc:  # pragma: no cover - optional native dependency
        raise RuntimeError("MeasurementSet RQ generation requires the `ms` extra") from exc
    source = Path(path)
    with tables.table(
        str(source / "CALDEVICE"), readonly=True, ack=False
    ) as caldevice:
        if caldevice.nrows() == 0:
            raise ValueError("requantizer generation requires CALDEVICE rows")
    columns: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "gain",
            "time_s",
            "interval_s",
            "antenna_id",
            "spectral_window_id",
        )
    }
    with tables.table(str(source / "SYSPOWER"), readonly=True, ack=False) as syspower:
        for start in range(0, syspower.nrows(), chunk_size):
            count = min(chunk_size, syspower.nrows() - start)
            columns["gain"].append(
                np.asarray(
                    syspower.getcol(
                        "REQUANTIZER_GAIN", startrow=start, nrow=count
                    ),
                    dtype=np.float64,
                )
            )
            for output, name, dtype in (
                ("time_s", "TIME", np.float64),
                ("interval_s", "INTERVAL", np.float64),
                ("antenna_id", "ANTENNA_ID", np.int32),
                ("spectral_window_id", "SPECTRAL_WINDOW_ID", np.int32),
            ):
                columns[output].append(
                    np.asarray(
                        syspower.getcol(name, startrow=start, nrow=count),
                        dtype=dtype,
                    )
                )
    gain = np.concatenate(columns["gain"])
    return generate_requantizer_gain(
        gain,
        time_s=np.concatenate(columns["time_s"]),
        interval_s=np.concatenate(columns["interval_s"]),
        antenna_id=np.concatenate(columns["antenna_id"]),
        spectral_window_id=np.concatenate(columns["spectral_window_id"]),
        provenance={
            "source": str((source / "SYSPOWER").resolve()),
            "caldevice": str((source / "CALDEVICE").resolve()),
            "chunk_size": chunk_size,
            "convention": "CASA 6.5 RQ-only voltage gain; calwt=False",
        },
    )


def spectral_window_centres(
    spectral_windows: tuple[SpectralWindowRecord, ...],
) -> np.ndarray:
    """Return stable SPW reference frequencies for validation and plotting."""

    return np.asarray(
        [record.reference_frequency_hz for record in spectral_windows],
        dtype=np.float64,
    )
