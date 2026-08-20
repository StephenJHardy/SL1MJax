"""Composable diagonal a-priori Jones terms for VLA calibration."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class CalibrationCoordinates:
    """Coordinates required to evaluate antenna-based prior terms."""

    time_s: np.ndarray
    frequency_hz: np.ndarray
    spectral_window_id: int
    phase_centre_rad: tuple[float, float]
    antenna_position_m: np.ndarray
    receptor_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "time_s", np.asarray(self.time_s, dtype=np.float64))
        object.__setattr__(
            self, "frequency_hz", np.asarray(self.frequency_hz, dtype=np.float64)
        )
        object.__setattr__(
            self,
            "antenna_position_m",
            np.asarray(self.antenna_position_m, dtype=np.float64),
        )
        if self.time_s.ndim != 1 or self.frequency_hz.ndim != 1:
            raise ValueError("time and frequency coordinates must be one-dimensional")
        if (
            self.antenna_position_m.ndim != 2
            or self.antenna_position_m.shape[1] != 3
        ):
            raise ValueError("antenna_position_m must have shape (antenna, 3)")
        if self.receptor_count < 1:
            raise ValueError("receptor_count must be positive")


class PriorJonesTerm(Protocol):
    """Protocol for one diagonal antenna Jones term."""

    @property
    def kind(self) -> str: ...

    @property
    def provenance(self) -> dict[str, Any]: ...

    def evaluate(
        self, coordinates: CalibrationCoordinates
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return Jones and validity with shape row, channel, antenna, receptor."""


def elevation_rad(coordinates: CalibrationCoordinates) -> np.ndarray:
    """Geometric source elevation for every row and antenna."""

    mjd = coordinates.time_s / 86400.0
    julian_date = mjd + 2_400_000.5
    gmst = np.deg2rad(
        np.mod(
            280.46061837 + 360.98564736629 * (julian_date - 2_451_545.0),
            360.0,
        )
    )
    right_ascension, declination = coordinates.phase_centre_rad
    hour_angle = right_ascension - gmst
    source_ecef = np.stack(
        (
            np.cos(declination) * np.cos(hour_angle),
            np.cos(declination) * np.sin(hour_angle),
            np.full(coordinates.time_s.shape, np.sin(declination)),
        ),
        axis=1,
    )
    position = coordinates.antenna_position_m
    up = position / np.linalg.norm(position, axis=1, keepdims=True)
    return np.asarray(np.arcsin(np.clip(source_ecef @ up.T, -1.0, 1.0)))


def airmass_from_elevation(
    elevation: np.ndarray, *, minimum_elevation_rad: float = 0.0
) -> np.ndarray:
    """Plane-parallel airmass, optionally floored for non-CASA applications."""

    selected = np.maximum(elevation, minimum_elevation_rad)
    return np.asarray(
        1.0 / np.maximum(np.sin(selected), np.finfo(np.float64).tiny)
    )


def _spw_index(spectral_window_ids: np.ndarray, selected: int) -> int:
    matches = np.flatnonzero(spectral_window_ids == selected)
    if matches.size != 1:
        raise ValueError(f"spectral window {selected} is outside the prior term")
    return int(matches[0])


@dataclass(frozen=True)
class GainCurveTerm:
    """VLA voltage-gain polynomial in zenith angle (degrees)."""

    coefficients: np.ndarray
    spectral_window_ids: np.ndarray
    valid: np.ndarray
    provenance: dict[str, Any] = field(default_factory=dict)
    kind: str = field(default="gain_curve", init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "coefficients", np.asarray(self.coefficients, dtype=np.float64)
        )
        object.__setattr__(
            self,
            "spectral_window_ids",
            np.asarray(self.spectral_window_ids, dtype=np.int32),
        )
        object.__setattr__(self, "valid", np.asarray(self.valid, dtype=bool))
        if self.coefficients.ndim != 4:
            raise ValueError(
                "gain-curve coefficients must have shape antenna, spw, receptor, coefficient"
            )
        if self.valid.shape != self.coefficients.shape[:-1]:
            raise ValueError("gain-curve validity must match antenna/spw/receptor")

    def evaluate(
        self, coordinates: CalibrationCoordinates
    ) -> tuple[np.ndarray, np.ndarray]:
        index = _spw_index(self.spectral_window_ids, coordinates.spectral_window_id)
        coefficients = self.coefficients[:, index, :, :]
        if coefficients.shape[1] != coordinates.receptor_count:
            raise ValueError("gain-curve receptors do not match visibility receptors")
        zenith_angle_deg = np.rad2deg(np.pi / 2 - elevation_rad(coordinates))
        powers = np.power(
            zenith_angle_deg[..., None],
            np.arange(coefficients.shape[-1], dtype=np.float64),
        )
        values = np.einsum("rac,apc->rap", powers, coefficients)
        jones = np.broadcast_to(
            values[:, None, :, :],
            (
                coordinates.time_s.size,
                coordinates.frequency_hz.size,
                coefficients.shape[0],
                coefficients.shape[1],
            ),
        ).astype(np.complex128)
        valid = np.broadcast_to(
            self.valid[:, index, :][None, None, :, :], jones.shape
        ).copy()
        valid &= np.isfinite(jones) & (np.abs(jones) > 0)
        return jones, valid


@dataclass(frozen=True)
class OpacityTerm:
    """Tropospheric attenuation from zenith optical depth per SPW."""

    zenith_opacity: np.ndarray
    spectral_window_ids: np.ndarray
    valid: np.ndarray
    provenance: dict[str, Any] = field(default_factory=dict)
    kind: str = field(default="opacity", init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "zenith_opacity", np.asarray(self.zenith_opacity, dtype=np.float64)
        )
        object.__setattr__(
            self,
            "spectral_window_ids",
            np.asarray(self.spectral_window_ids, dtype=np.int32),
        )
        object.__setattr__(self, "valid", np.asarray(self.valid, dtype=bool))
        if self.zenith_opacity.ndim != 3:
            raise ValueError("opacity must have shape antenna, spw, receptor")
        if self.valid.shape != self.zenith_opacity.shape:
            raise ValueError("opacity validity must match opacity values")

    def evaluate(
        self, coordinates: CalibrationCoordinates
    ) -> tuple[np.ndarray, np.ndarray]:
        index = _spw_index(self.spectral_window_ids, coordinates.spectral_window_id)
        tau = self.zenith_opacity[:, index, :]
        if tau.shape[1] != coordinates.receptor_count:
            raise ValueError("opacity receptors do not match visibility receptors")
        elevation = elevation_rad(coordinates)
        airmass = airmass_from_elevation(elevation)
        # This is the corrupting voltage Jones. Dividing by it corrects attenuation.
        amplitude = np.exp(-0.5 * airmass[:, :, None] * tau[None, :, :])
        jones = np.broadcast_to(
            amplitude[:, None, :, :],
            (
                coordinates.time_s.size,
                coordinates.frequency_hz.size,
                tau.shape[0],
                tau.shape[1],
            ),
        ).astype(np.complex128)
        valid = np.broadcast_to(
            self.valid[:, index, :][None, None, :, :], jones.shape
        ).copy()
        valid &= np.broadcast_to(
            (elevation > 0)[:, None, :, None], jones.shape
        )
        valid &= np.isfinite(jones)
        return jones, valid


@dataclass(frozen=True)
class RequantizerTerm:
    """Nearest-time RQ-only voltage gains from VLA SYSPOWER/CALDEVICE data."""

    gain: np.ndarray
    time_s: np.ndarray
    interval_s: np.ndarray
    antenna_id: np.ndarray
    spectral_window_id: np.ndarray
    valid: np.ndarray
    provenance: dict[str, Any] = field(default_factory=dict)
    kind: str = field(default="requantizer", init=False)

    def __post_init__(self) -> None:
        for name, dtype in (
            ("gain", np.float64),
            ("time_s", np.float64),
            ("interval_s", np.float64),
            ("antenna_id", np.int32),
            ("spectral_window_id", np.int32),
            ("valid", bool),
        ):
            object.__setattr__(self, name, np.asarray(getattr(self, name), dtype=dtype))
        if self.gain.ndim != 2 or self.valid.shape != self.gain.shape:
            raise ValueError("requantizer gain/valid must have shape sample, receptor")
        samples = self.gain.shape[0]
        for name in ("time_s", "interval_s", "antenna_id", "spectral_window_id"):
            if getattr(self, name).shape != (samples,):
                raise ValueError(f"{name} must match the requantizer sample axis")

    def evaluate(
        self, coordinates: CalibrationCoordinates
    ) -> tuple[np.ndarray, np.ndarray]:
        antennas = coordinates.antenna_position_m.shape[0]
        receptors = coordinates.receptor_count
        if self.gain.shape[1] != receptors:
            raise ValueError("requantizer receptors do not match visibility receptors")
        output = np.ones(
            (coordinates.time_s.size, antennas, receptors), dtype=np.float64
        )
        validity = np.zeros(output.shape, dtype=bool)
        for antenna in range(antennas):
            selected = (self.antenna_id == antenna) & (
                self.spectral_window_id == coordinates.spectral_window_id
            )
            if not np.any(selected):
                continue
            order = np.argsort(self.time_s[selected], kind="stable")
            times = self.time_s[selected][order]
            gains = self.gain[selected][order]
            valid = self.valid[selected][order]
            intervals = self.interval_s[selected][order]
            right = np.searchsorted(times, coordinates.time_s, side="left")
            right = np.clip(right, 0, times.size - 1)
            left = np.maximum(right - 1, 0)
            choose_left = np.abs(coordinates.time_s - times[left]) <= np.abs(
                coordinates.time_s - times[right]
            )
            nearest = np.where(choose_left, left, right)
            output[:, antenna, :] = gains[nearest]
            domain = (intervals[nearest] <= 0) | (
                np.abs(coordinates.time_s - times[nearest])
                <= intervals[nearest] / 2
            )
            validity[:, antenna, :] = valid[nearest] & domain[:, None]
        jones = np.broadcast_to(
            output[:, None, :, :],
            (
                coordinates.time_s.size,
                coordinates.frequency_hz.size,
                antennas,
                receptors,
            ),
        ).astype(np.complex128)
        valid = np.broadcast_to(validity[:, None, :, :], jones.shape).copy()
        valid &= np.isfinite(jones) & (np.abs(jones) > 0)
        return jones, valid


@dataclass(frozen=True)
class CalibrationChain:
    """Ordered composable prior terms."""

    terms: tuple[PriorJonesTerm, ...] = ()
    antenna_position_m: np.ndarray | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.antenna_position_m is not None:
            position = np.asarray(self.antenna_position_m, dtype=np.float64)
            if position.ndim != 2 or position.shape[1] != 3:
                raise ValueError("chain antenna positions must have shape (antenna, 3)")
            object.__setattr__(self, "antenna_position_m", position)

    def evaluate(
        self, coordinates: CalibrationCoordinates
    ) -> tuple[np.ndarray, np.ndarray]:
        shape = (
            coordinates.time_s.size,
            coordinates.frequency_hz.size,
            coordinates.antenna_position_m.shape[0],
            coordinates.receptor_count,
        )
        value = np.ones(shape, dtype=np.complex128)
        valid = np.ones(shape, dtype=bool)
        for term in self.terms:
            selected, selected_valid = term.evaluate(coordinates)
            if selected.shape != shape or selected_valid.shape != shape:
                raise ValueError(f"{term.kind} returned incompatible Jones coordinates")
            value *= selected
            valid &= selected_valid
        return value, valid


@dataclass(frozen=True)
class PriorComparison:
    relative_rms: float
    maximum_relative_error: float
    compared_count: int
    flag_mismatch_count: int


def compare_jones(
    generated: np.ndarray,
    reference: np.ndarray,
    generated_valid: np.ndarray,
    reference_valid: np.ndarray,
) -> PriorComparison:
    """Compare generated and oracle Jones values on identical coordinates."""

    generated_array = np.asarray(generated)
    reference_array = np.asarray(reference)
    generated_mask = np.asarray(generated_valid, dtype=bool)
    reference_mask = np.asarray(reference_valid, dtype=bool)
    if (
        generated_array.shape != reference_array.shape
        or generated_mask.shape != generated_array.shape
        or reference_mask.shape != generated_array.shape
    ):
        raise ValueError("Jones comparison arrays must have identical shapes")
    selected = generated_mask & reference_mask
    finite = (
        selected
        & np.isfinite(generated_array)
        & np.isfinite(reference_array)
        & (np.abs(reference_array) > 0)
    )
    if not np.any(finite):
        raise ValueError("Jones comparison has no mutually valid samples")
    difference = generated_array[finite] - reference_array[finite]
    denominator = np.sum(np.abs(reference_array[finite]) ** 2)
    relative = np.abs(difference) / np.abs(reference_array[finite])
    return PriorComparison(
        relative_rms=float(np.sqrt(np.sum(np.abs(difference) ** 2) / denominator)),
        maximum_relative_error=float(np.max(relative)),
        compared_count=int(np.count_nonzero(finite)),
        flag_mismatch_count=int(np.count_nonzero(generated_mask != reference_mask)),
    )


def prior_baseline_jones(
    chain: CalibrationChain,
    coordinates: CalibrationCoordinates,
    antenna1: np.ndarray,
    antenna2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate `J_p conj(J_q)` for a prior chain."""

    antenna_jones, antenna_valid = chain.evaluate(coordinates)
    first = np.asarray(antenna1, dtype=np.int32)
    second = np.asarray(antenna2, dtype=np.int32)
    row = np.arange(coordinates.time_s.size)[:, None, None]
    channel = np.arange(coordinates.frequency_hz.size)[None, :, None]
    receptor = np.arange(coordinates.receptor_count)[None, None, :]
    j1 = antenna_jones[row, channel, first[:, None, None], receptor]
    j2 = antenna_jones[row, channel, second[:, None, None], receptor]
    v1 = antenna_valid[row, channel, first[:, None, None], receptor]
    v2 = antenna_valid[row, channel, second[:, None, None], receptor]
    return j1 * np.conj(j2), v1 & v2


def import_casa_prior_table(
    path: str | Path, *, row_stride: int = 1, chunk_rows: int = 65_536
) -> GainCurveTerm | OpacityTerm | RequantizerTerm:
    """Import one CASA prior table as a numerical comparison oracle."""

    if row_stride < 1 or chunk_rows < 1:
        raise ValueError("row_stride and chunk_rows must be positive")
    try:
        from casacore import tables
    except ImportError as exc:  # pragma: no cover - optional native dependency
        raise RuntimeError("CASA table import requires the `ms` extra") from exc
    source = Path(path)
    with tables.table(str(source), readonly=True, ack=False) as table:
        viscal = str(table.getkeyword("VisCal"))

        def read_column(name: str, dtype: Any = None) -> np.ndarray:
            chunks = []
            for start in range(0, table.nrows(), chunk_rows):
                count = min(chunk_rows, table.nrows() - start)
                offset = (-start) % row_stride
                chunk = np.asarray(
                    table.getcol(name, startrow=start, nrow=count), dtype=dtype
                )
                chunks.append(chunk[offset::row_stride])
            return np.concatenate(chunks)

        antenna = read_column("ANTENNA1", np.int32)
        spw = read_column("SPECTRAL_WINDOW_ID", np.int32)
        flag = read_column("FLAG", bool)
        values = read_column("FPARAM")
        time_s = read_column("TIME", np.float64)
        interval_s = read_column("INTERVAL", np.float64)
    spectral_windows = np.unique(spw)
    antennas = int(np.max(antenna)) + 1
    provenance = {"source": str(source.resolve()), "viscal": viscal, "role": "oracle"}
    if viscal == "EGainCurve":
        receptors = values.shape[-1] // 4
        coefficients = np.ones(
            (antennas, spectral_windows.size, receptors, 4), dtype=np.float64
        )
        valid = np.zeros(coefficients.shape[:-1], dtype=bool)
        lookup = {value: index for index, value in enumerate(spectral_windows)}
        reshaped = values[:, 0, :].reshape(values.shape[0], receptors, 4)
        row_valid = ~flag[:, 0, :].reshape(values.shape[0], receptors, 4)
        for row, selected_antenna in enumerate(antenna):
            index = lookup[int(spw[row])]
            coefficients[selected_antenna, index] = reshaped[row]
            valid[selected_antenna, index] = np.all(row_valid[row], axis=-1)
        return GainCurveTerm(coefficients, spectral_windows, valid, provenance)
    if viscal == "TOpac":
        tau = np.zeros((antennas, spectral_windows.size, 2), dtype=np.float64)
        valid = np.zeros(tau.shape, dtype=bool)
        lookup = {value: index for index, value in enumerate(spectral_windows)}
        for row, selected_antenna in enumerate(antenna):
            index = lookup[int(spw[row])]
            tau[selected_antenna, index, :] = float(values[row, 0, 0])
            valid[selected_antenna, index, :] = not bool(flag[row, 0, 0])
        return OpacityTerm(tau, spectral_windows, valid, provenance)
    if viscal == "G EVLASWPOW":
        gain = values[:, 0, 0::2]
        valid = ~flag[:, 0, 0::2]
        return RequantizerTerm(
            gain,
            time_s,
            interval_s,
            antenna,
            spw,
            valid,
            provenance,
        )
    raise ValueError(f"unsupported CASA prior VisCal {viscal!r}")


def generate_requantizer_gain(
    requantizer_gain: np.ndarray,
    *,
    time_s: np.ndarray,
    interval_s: np.ndarray,
    antenna_id: np.ndarray,
    spectral_window_id: np.ndarray,
    valid: np.ndarray | None = None,
    provenance: dict[str, Any] | None = None,
) -> RequantizerTerm:
    """Generate the RQ-only term from portable SYSPOWER requantizer samples."""

    gain = np.asarray(requantizer_gain, dtype=np.float64)
    selected_valid = (
        np.isfinite(gain) & (gain > 0)
        if valid is None
        else np.asarray(valid, dtype=bool) & np.isfinite(gain) & (gain > 0)
    )
    return RequantizerTerm(
        gain,
        time_s,
        interval_s,
        antenna_id,
        spectral_window_id,
        selected_valid,
        {
            "generator": "sl1mjax",
            "caltype": "rq",
            "tsys": 1.0,
            **({} if provenance is None else provenance),
        },
    )


def write_calibration_chain(chain: CalibrationChain, path: str | Path) -> None:
    """Write a portable prior chain as JSON metadata plus NumPy arrays."""

    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "provenance": chain.provenance,
        "terms": [],
    }
    arrays: dict[str, np.ndarray] = {}
    if chain.antenna_position_m is not None:
        arrays["antenna_position_m"] = chain.antenna_position_m
    for index, term in enumerate(chain.terms):
        prefix = f"term_{index}"
        record = {
            "kind": term.kind,
            "prefix": prefix,
            "provenance": term.provenance,
        }
        manifest["terms"].append(record)
        if isinstance(term, GainCurveTerm):
            arrays[f"{prefix}_coefficients"] = term.coefficients
            arrays[f"{prefix}_spectral_window_ids"] = term.spectral_window_ids
            arrays[f"{prefix}_valid"] = term.valid
        elif isinstance(term, OpacityTerm):
            arrays[f"{prefix}_zenith_opacity"] = term.zenith_opacity
            arrays[f"{prefix}_spectral_window_ids"] = term.spectral_window_ids
            arrays[f"{prefix}_valid"] = term.valid
        elif isinstance(term, RequantizerTerm):
            for name in (
                "gain",
                "time_s",
                "interval_s",
                "antenna_id",
                "spectral_window_id",
                "valid",
            ):
                arrays[f"{prefix}_{name}"] = getattr(term, name)
        else:  # pragma: no cover - protects future protocol implementations
            raise TypeError(f"cannot serialize prior term {type(term).__name__}")
    np.savez_compressed(destination / "arrays.npz", **arrays)  # type: ignore[arg-type]
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def read_calibration_chain(path: str | Path) -> CalibrationChain:
    """Read a chain written by :func:`write_calibration_chain`."""

    source = Path(path)
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "1.0":
        raise ValueError("unsupported prior-chain schema")
    terms: list[PriorJonesTerm] = []
    with np.load(source / "arrays.npz") as arrays:
        positions = (
            np.array(arrays["antenna_position_m"], copy=True)
            if "antenna_position_m" in arrays
            else None
        )
        for record in manifest["terms"]:
            prefix = record["prefix"]
            provenance = record.get("provenance", {})

            def array(name: str, selected_prefix: str = prefix) -> NDArray[Any]:
                return np.array(arrays[f"{selected_prefix}_{name}"], copy=True)

            if record["kind"] == "gain_curve":
                terms.append(
                    GainCurveTerm(
                        array("coefficients"),
                        array("spectral_window_ids"),
                        array("valid"),
                        provenance,
                    )
                )
            elif record["kind"] == "opacity":
                terms.append(
                    OpacityTerm(
                        array("zenith_opacity"),
                        array("spectral_window_ids"),
                        array("valid"),
                        provenance,
                    )
                )
            elif record["kind"] == "requantizer":
                terms.append(
                    RequantizerTerm(
                        array("gain"),
                        array("time_s"),
                        array("interval_s"),
                        array("antenna_id"),
                        array("spectral_window_id"),
                        array("valid"),
                        provenance,
                    )
                )
            else:
                raise ValueError(f"unsupported prior term {record['kind']!r}")
    return CalibrationChain(
        tuple(terms), positions, dict(manifest.get("provenance", {}))
    )
