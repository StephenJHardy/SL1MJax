"""Casacore Measurement Set inventory and POINTING ingest for holography.

Fixture-only readers must not be used on a real MS. This module opens
casacore tables, stream-hashes archives, and never loads MAIN visibilities
into memory for hashing.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from sl1mjax.holography import (
    OPTIONAL_HOLOGRAPHY_TABLES,
    POINTING_DIRECTION_COLUMNS,
    REQUIRED_HOLOGRAPHY_TABLES,
    THOL0001_HOLORASTER_FIELD,
    THOL0001_LOWER_C_NATIVE_HZ,
    AntennaPointingTable,
    DirectionMeasure,
    HolographyArchiveInventory,
    holography_execution_provenance,
    identities_from_holography_path,
)
from sl1mjax.holography_pointing_audit import (
    HolographyMainMetadata,
    PointingReconstructionAudit,
    reconstruct_holography_pointing,
    select_native_channels,
)

_CORRELATION_FROM_CODE = {
    5: "RR",
    6: "RL",
    7: "LR",
    8: "LL",
    9: "XX",
    10: "XY",
    11: "YX",
    12: "YY",
}
_METADATA_SUBTABLES = (
    "ANTENNA",
    "FIELD",
    "STATE",
    "POINTING",
    "POLARIZATION",
    "SPECTRAL_WINDOW",
    "OBSERVATION",
    "SOURCE",
    "FEED",
    "DATA_DESCRIPTION",
)
_HASH_SKIP_SUFFIXES = (".f0", ".f1", ".f2", ".f3", ".f4", ".f5", ".f6", ".f7", ".f8", ".f9")


def is_casacore_measurement_set(path: Path) -> bool:
    """True when ``path`` looks like a casacore Measurement Set, not a fixture."""

    root = Path(path)
    return (root / "table.dat").is_file() and (root / "ANTENNA").is_dir()


def stream_sha256(path: Path, *, chunk_bytes: int = 1 << 20) -> str:
    """Hash a file without reading it all into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(int(chunk_bytes))
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def hash_metadata_tree(path: Path, *, chunk_bytes: int = 1 << 20) -> str:
    """Hash MS metadata files. Skip MAIN visibility blobs and large ``table.f*``."""

    root = Path(path)
    digest = hashlib.sha256()
    for child in sorted(root.rglob("*")):
        if not child.is_file():
            continue
        relative = child.relative_to(root).as_posix()
        if relative.startswith("table.f") or "/table.f" in relative:
            continue
        if child.suffix in _HASH_SKIP_SUFFIXES:
            continue
        if child.stat().st_size > 64 * 1024 * 1024:
            digest.update(relative.encode("utf-8"))
            digest.update(str(child.stat().st_size).encode("utf-8"))
            continue
        digest.update(relative.encode("utf-8"))
        digest.update(stream_sha256(child, chunk_bytes=chunk_bytes).encode("ascii"))
    return digest.hexdigest()


def inventory_casacore_measurement_set(
    path: Path,
    *,
    archive_path: Path | None = None,
) -> HolographyArchiveInventory:
    """Inventory a real holography Measurement Set from casacore tables."""

    tables = _tables()
    measurement_set = Path(path)
    if not is_casacore_measurement_set(measurement_set):
        raise ValueError(f"not a casacore Measurement Set: {measurement_set}")
    presence = {
        name: (measurement_set / name).exists()
        for name in (*REQUIRED_HOLOGRAPHY_TABLES, *OPTIONAL_HOLOGRAPHY_TABLES)
    }
    notes: list[str] = []
    if not presence["POINTING"]:
        notes.append("POINTING table is absent; SDM-BDF inspection is required")
    with tables.table(str(measurement_set), readonly=True, ack=False) as main:
        main_columns = tuple(str(name) for name in main.colnames())
        main_row_count = int(main.nrows())
        time_s = (
            np.asarray(main.getcol("TIME"), dtype=np.float64)
            if "TIME" in main_columns
            else np.array([])
        )
        scan = (
            np.asarray(main.getcol("SCAN_NUMBER"), dtype=np.int32)
            if "SCAN_NUMBER" in main_columns
            else np.array([], dtype=np.int32)
        )
    observation = _read_observation(tables, measurement_set)
    observation_project = observation.get("project") or ""
    provenance = holography_execution_provenance(
        measurement_set,
        archive_path,
        observation_project=observation_project,
    )
    project = provenance["project"]
    scheduling_block = provenance["scheduling_block"]
    execution_block = provenance["execution_block"]
    if not scheduling_block or not execution_block:
        notes.append("scheduling or execution block was not parsed from the path")
    if observation_project and observation_project != project:
        notes.append(
            f"OBSERVATION.PROJECT {observation_project} is kept separate from "
            f"path-derived project {project}"
        )
    antennas = _read_antennas(tables, measurement_set)
    correlations, correlation_codes, corr_by_ddid = _read_correlations(tables, measurement_set)
    fields = _read_field_names(tables, measurement_set)
    states = _read_state_modes(tables, measurement_set)
    pointing_refs = _read_pointing_measure_refs(tables, measurement_set)
    freq_min, freq_max, n_spw, freq_mono = _read_frequency_span(tables, measurement_set)
    memo = "CHOLO-LO"
    if freq_min >= 5.9e9:
        memo = "CHOLO-HI"
    elif freq_max > 6.1e9 and freq_min < 5.0e9:
        notes.append("frequency span covers both lower and upper C; memo product is ambiguous")
    source_name = fields[0] if fields else ""
    if any("3C147" in name.upper() or "3C 147" in name.upper() for name in fields):
        source_name = "3C147"
    observation_utc = _observation_utc(observation, time_s)
    archive_hash = (
        stream_sha256(Path(archive_path))
        if archive_path is not None
        else hash_metadata_tree(measurement_set)
    )
    if archive_path is None:
        notes.append(
            "archive_sha256 is a metadata-tree hash; pass archive_path to hash the tarball"
        )
    if "DATA" not in main_columns:
        notes.append("MAIN has no DATA column")
    if "CORRECTED_DATA" not in main_columns:
        notes.append("MAIN has no CORRECTED_DATA column")
    if time_s.size and not bool(np.all(np.diff(time_s) >= 0.0)):
        notes.append("TIME axis is not non-decreasing")
    if corr_by_ddid:
        notes.append("correlations_by_ddid=" + ";".join(corr_by_ddid))
    return HolographyArchiveInventory(
        project=str(project),
        scheduling_block=str(scheduling_block),
        execution_block=str(execution_block),
        source_name=str(source_name),
        observation_utc=observation_utc,
        path=str(measurement_set),
        archive_sha256=archive_hash,
        table_presence=presence,
        correlations=correlations,
        antenna_count=max(len(antennas[0]), 1),
        memo_product=memo,
        notes=tuple(notes),
        correlation_codes=correlation_codes,
        antenna_ids=antennas[0],
        antenna_names=antennas[1],
        antenna_position_m=antennas[2],
        time_is_monotonic=bool(time_s.size == 0 or np.all(np.diff(time_s) >= 0.0)),
        frequency_is_monotonic=freq_mono,
        main_row_count=main_row_count,
        main_columns=main_columns,
        has_data="DATA" in main_columns,
        has_corrected_data="CORRECTED_DATA" in main_columns,
        field_names=fields,
        state_modes=states,
        pointing_measure_refs=pointing_refs,
        frequency_hz_min=float(freq_min),
        frequency_hz_max=float(freq_max),
        n_spw=int(n_spw),
        n_scan=int(np.unique(scan).size) if scan.size else 0,
        observation_project=observation_project,
    )


def read_pointing_table_from_ms(path: Path) -> AntennaPointingTable:
    """Read POINTING columns and measure metadata without selecting a raster."""

    tables = _tables()
    measurement_set = Path(path)
    pointing = measurement_set / "POINTING"
    if not pointing.exists():
        raise FileNotFoundError(f"POINTING table is absent: {pointing}")
    with tables.table(str(pointing), readonly=True, ack=False) as table:
        columns = set(table.colnames())
        required = {"TIME", "INTERVAL", "ANTENNA_ID"}
        missing = required - columns
        if missing:
            raise ValueError(f"POINTING is missing {sorted(missing)}")
        time_s = np.asarray(table.getcol("TIME"), dtype=np.float64).reshape(-1)
        interval_s = np.asarray(table.getcol("INTERVAL"), dtype=np.float64).reshape(-1)
        antenna_id = np.asarray(table.getcol("ANTENNA_ID"), dtype=np.int32).reshape(-1)
        measures = []
        for name in POINTING_DIRECTION_COLUMNS:
            if name not in columns:
                continue
            values = _direction_column(table, name)
            ref, units = _column_measure(table, name)
            measures.append(DirectionMeasure(name, values, ref, units))
        if not measures:
            raise ValueError("POINTING has no DIRECTION, TARGET, or POINTING_OFFSET")
        tracking = (
            np.asarray(table.getcol("TRACKING"), dtype=bool).reshape(-1)
            if "TRACKING" in columns
            else None
        )
        on_source = (
            np.asarray(table.getcol("ON_SOURCE"), dtype=bool).reshape(-1)
            if "ON_SOURCE" in columns
            else None
        )
        interpolation = (
            np.asarray(table.getcol("NAME"), dtype=str).reshape(-1) if "NAME" in columns else None
        )
    flag = np.zeros(time_s.size, dtype=bool)
    if interpolation is not None:
        flag = np.array(["interpol" in str(name).lower() for name in interpolation], dtype=bool)
    digest = hash_metadata_tree(pointing)
    return AntennaPointingTable(
        time_s=time_s,
        interval_s=interval_s,
        antenna_id=antenna_id,
        columns=tuple(measures),
        tracking=tracking,
        on_source=on_source,
        row_id=np.arange(time_s.size, dtype=np.int32),
        source_ms=str(measurement_set),
        table_sha256=digest,
        interpolation_flag=flag,
        time_is_midpoint=True,
    )


def audit_holography_measurement_set(
    path: Path,
    *,
    field_name: str = THOL0001_HOLORASTER_FIELD,
    target_frequency_hz: tuple[float, ...] = THOL0001_LOWER_C_NATIVE_HZ,
    selected_column: str = "POINTING_OFFSET",
    archive_path: Path | None = None,
) -> PointingReconstructionAudit:
    """Reconstruct HOLORASTER pointing from MAIN metadata and POINTING.

    Visibilities are not read. Calibration and Jones recovery are out of scope.
    """

    tables = _tables()
    measurement_set = Path(path)
    if not is_casacore_measurement_set(measurement_set):
        raise ValueError(f"not a casacore Measurement Set: {measurement_set}")
    observation = _read_observation(tables, measurement_set)
    provenance = holography_execution_provenance(
        measurement_set,
        archive_path,
        observation_project=observation.get("project") or "",
    )
    field_names = _read_field_names(tables, measurement_set)
    state_modes = _read_state_modes(tables, measurement_set)
    antennas = _read_antennas(tables, measurement_set)
    phase_centre = _read_field_phase_centre(tables, measurement_set, field_name)
    channel_frequency = _read_channel_frequencies(tables, measurement_set)
    data_descriptions = _read_data_descriptions(tables, measurement_set)
    native = select_native_channels(channel_frequency, data_descriptions, target_frequency_hz)
    field_ids = tuple(index for index, name in enumerate(field_names) if name == field_name)
    if not field_ids:
        raise ValueError(f"FIELD has no {field_name!r} entry")
    ddid_ids = tuple(sorted({channel.data_desc_id for channel in native}))
    main = _read_selected_main_metadata(
        tables,
        measurement_set,
        field_ids=field_ids,
        data_desc_ids=ddid_ids,
        field_names=field_names,
        state_modes=state_modes,
        antenna_names=antennas[1],
        phase_centre_rad=phase_centre,
    )
    pointing = read_pointing_table_from_ms(measurement_set)
    return reconstruct_holography_pointing(
        pointing,
        main,
        field_name=field_name,
        data_desc_ids=ddid_ids,
        selected_column=selected_column,
        provenance=provenance,
        native_channels=native,
    )


def _read_selected_main_metadata(
    tables: Any,
    measurement_set: Path,
    *,
    field_ids: tuple[int, ...],
    data_desc_ids: tuple[int, ...],
    field_names: tuple[str, ...],
    state_modes: tuple[str, ...],
    antenna_names: tuple[str, ...],
    phase_centre_rad: tuple[float, float],
) -> HolographyMainMetadata:
    with tables.table(str(measurement_set), readonly=True, ack=False) as main:
        main_row_count = int(main.nrows())
        columns = set(main.colnames())
        required = {
            "TIME",
            "ANTENNA1",
            "ANTENNA2",
            "FIELD_ID",
            "STATE_ID",
            "SCAN_NUMBER",
            "DATA_DESC_ID",
        }
        missing = required - columns
        if missing:
            raise ValueError(f"MAIN is missing {sorted(missing)}")
        has_interval = "INTERVAL" in columns
    interval_select = "INTERVAL," if has_interval else ""
    query = (
        f"SELECT TIME,{interval_select}ANTENNA1,ANTENNA2,FIELD_ID,STATE_ID,"
        f"SCAN_NUMBER,DATA_DESC_ID FROM '{measurement_set}' "
        f"WHERE FIELD_ID IN {_taql_int_list(field_ids)} "
        f"AND DATA_DESC_ID IN {_taql_int_list(data_desc_ids)}"
    )
    with tables.taql(query) as selected:
        if selected.nrows() == 0:
            raise ValueError("selected HOLORASTER/DDID MAIN metadata is empty")
        selected_columns = set(selected.colnames())
        time_s = np.asarray(selected.getcol("TIME"), dtype=np.float64).reshape(-1)
        interval_s = (
            np.asarray(selected.getcol("INTERVAL"), dtype=np.float64).reshape(-1)
            if "INTERVAL" in selected_columns
            else np.zeros(time_s.size, dtype=np.float64)
        )
        antenna1 = np.asarray(selected.getcol("ANTENNA1"), dtype=np.int32).reshape(-1)
        antenna2 = np.asarray(selected.getcol("ANTENNA2"), dtype=np.int32).reshape(-1)
        field_id = np.asarray(selected.getcol("FIELD_ID"), dtype=np.int32).reshape(-1)
        state_id = np.asarray(selected.getcol("STATE_ID"), dtype=np.int32).reshape(-1)
        scan_number = np.asarray(selected.getcol("SCAN_NUMBER"), dtype=np.int32).reshape(-1)
        data_desc_id = np.asarray(selected.getcol("DATA_DESC_ID"), dtype=np.int32).reshape(-1)
    return HolographyMainMetadata(
        time_s=time_s,
        interval_s=interval_s,
        antenna1=antenna1,
        antenna2=antenna2,
        field_id=field_id,
        state_id=state_id,
        scan_number=scan_number,
        data_desc_id=data_desc_id,
        field_names=field_names,
        state_modes=state_modes,
        antenna_names=antenna_names,
        phase_centre_rad=phase_centre_rad,
        main_row_count=main_row_count,
    )


def _read_channel_frequencies(tables: Any, measurement_set: Path) -> dict[int, NDArray[np.float64]]:
    window = measurement_set / "SPECTRAL_WINDOW"
    if not window.exists():
        raise ValueError("SPECTRAL_WINDOW table is required")
    frequencies: dict[int, NDArray[np.float64]] = {}
    with tables.table(str(window), readonly=True, ack=False) as table:
        for row in range(table.nrows()):
            frequencies[int(row)] = np.asarray(
                table.getcell("CHAN_FREQ", row), dtype=np.float64
            ).reshape(-1)
    return frequencies


def _read_data_descriptions(tables: Any, measurement_set: Path) -> tuple[tuple[int, int, int], ...]:
    description = measurement_set / "DATA_DESCRIPTION"
    if not description.exists():
        raise ValueError("DATA_DESCRIPTION table is required")
    rows: list[tuple[int, int, int]] = []
    with tables.table(str(description), readonly=True, ack=False) as table:
        columns = set(table.colnames())
        if "SPECTRAL_WINDOW_ID" not in columns or "POLARIZATION_ID" not in columns:
            raise ValueError("DATA_DESCRIPTION must provide SPECTRAL_WINDOW_ID and POLARIZATION_ID")
        for row in range(table.nrows()):
            rows.append(
                (
                    int(row),
                    int(table.getcell("SPECTRAL_WINDOW_ID", row)),
                    int(table.getcell("POLARIZATION_ID", row)),
                )
            )
    return tuple(rows)


def _read_field_phase_centre(
    tables: Any, measurement_set: Path, field_name: str
) -> tuple[float, float]:
    field = measurement_set / "FIELD"
    if not field.exists():
        raise ValueError("FIELD table is required")
    with tables.table(str(field), readonly=True, ack=False) as table:
        columns = set(table.colnames())
        if "NAME" not in columns or "PHASE_DIR" not in columns:
            raise ValueError("FIELD must provide NAME and PHASE_DIR")
        for row in range(table.nrows()):
            if str(table.getcell("NAME", row)) != field_name:
                continue
            direction = np.asarray(table.getcell("PHASE_DIR", row), dtype=np.float64).reshape(-1, 2)
            return float(direction[0, 0]), float(direction[0, 1])
    raise ValueError(f"FIELD has no {field_name!r} entry")


def _taql_int_list(values: tuple[int, ...]) -> str:
    return "[" + ",".join(str(int(value)) for value in values) + "]"


def _tables() -> Any:
    try:
        from casacore import tables
    except ImportError as exc:  # pragma: no cover - optional extra
        raise RuntimeError(
            "holography Measurement Set inventory requires `uv sync --extra ms`"
        ) from exc
    return tables


def _identities_from_path(*paths: Path | None) -> dict[str, str]:
    return identities_from_holography_path(*paths)


def _read_observation(tables: Any, measurement_set: Path) -> dict[str, str]:
    observation = measurement_set / "OBSERVATION"
    if not observation.exists():
        return {}
    with tables.table(str(observation), readonly=True, ack=False) as table:
        if table.nrows() == 0:
            return {}
        columns = set(table.colnames())
        payload = {}
        if "PROJECT" in columns:
            payload["project"] = str(table.getcell("PROJECT", 0))
        if "TELESCOPE_NAME" in columns:
            payload["telescope"] = str(table.getcell("TELESCOPE_NAME", 0))
        if "TIME_RANGE" in columns:
            span = np.asarray(table.getcell("TIME_RANGE", 0), dtype=np.float64).reshape(-1)
            if span.size >= 2:
                payload["time_start"] = str(float(span[0]))
                payload["time_stop"] = str(float(span[1]))
        return payload


def _read_antennas(
    tables: Any, measurement_set: Path
) -> tuple[tuple[int, ...], tuple[str, ...], tuple[tuple[float, float, float], ...]]:
    antenna = measurement_set / "ANTENNA"
    if not antenna.exists():
        return (), (), ()
    ids: list[int] = []
    names: list[str] = []
    positions: list[tuple[float, float, float]] = []
    with tables.table(str(antenna), readonly=True, ack=False) as table:
        columns = set(table.colnames())
        for row in range(table.nrows()):
            ids.append(row)
            names.append(str(table.getcell("NAME", row)) if "NAME" in columns else f"ANT{row}")
            if "POSITION" in columns:
                position = np.asarray(table.getcell("POSITION", row), dtype=np.float64).reshape(3)
                positions.append((float(position[0]), float(position[1]), float(position[2])))
            else:
                positions.append((float("nan"), float("nan"), float("nan")))
    return tuple(ids), tuple(names), tuple(positions)


def _read_correlations(
    tables: Any, measurement_set: Path
) -> tuple[tuple[str, ...], tuple[int, ...], tuple[str, ...]]:
    polarization = measurement_set / "POLARIZATION"
    if not polarization.exists():
        raise ValueError("POLARIZATION table is required; refusing to assume RR/RL/LR/LL")
    names: list[str] = []
    codes: list[int] = []
    by_ddid: list[str] = []
    with tables.table(str(polarization), readonly=True, ack=False) as table:
        for row in range(table.nrows()):
            row_codes = tuple(
                int(code) for code in np.asarray(table.getcell("CORR_TYPE", row)).ravel()
            )
            row_names = tuple(
                _CORRELATION_FROM_CODE.get(code, f"CODE_{code}") for code in row_codes
            )
            by_ddid.append(f"{row}:" + ",".join(row_names))
            for code, name in zip(row_codes, row_names, strict=True):
                if code not in codes:
                    codes.append(code)
                    names.append(name)
    if not names:
        raise ValueError("POLARIZATION.CORR_TYPE is empty")
    return tuple(names), tuple(codes), tuple(by_ddid)


def _read_field_names(tables: Any, measurement_set: Path) -> tuple[str, ...]:
    field = measurement_set / "FIELD"
    if not field.exists():
        return ()
    with tables.table(str(field), readonly=True, ack=False) as table:
        columns = set(table.colnames())
        if "NAME" not in columns:
            return ()
        return tuple(str(table.getcell("NAME", row)) for row in range(table.nrows()))


def _read_state_modes(tables: Any, measurement_set: Path) -> tuple[str, ...]:
    state = measurement_set / "STATE"
    if not state.exists():
        return ()
    with tables.table(str(state), readonly=True, ack=False) as table:
        columns = set(table.colnames())
        if "OBS_MODE" not in columns:
            return ()
        return tuple(str(table.getcell("OBS_MODE", row)) for row in range(table.nrows()))


def _read_pointing_measure_refs(tables: Any, measurement_set: Path) -> tuple[tuple[str, str], ...]:
    pointing = measurement_set / "POINTING"
    if not pointing.exists():
        return ()
    refs: list[tuple[str, str]] = []
    with tables.table(str(pointing), readonly=True, ack=False) as table:
        for name in POINTING_DIRECTION_COLUMNS:
            if name not in table.colnames():
                continue
            ref, _units = _column_measure(table, name)
            refs.append((name, ref))
    return tuple(refs)


def _read_frequency_span(tables: Any, measurement_set: Path) -> tuple[float, float, int, bool]:
    window = measurement_set / "SPECTRAL_WINDOW"
    if not window.exists():
        return 0.0, 0.0, 0, True
    frequencies: list[float] = []
    with tables.table(str(window), readonly=True, ack=False) as table:
        n_spw = int(table.nrows())
        for row in range(n_spw):
            channel = np.asarray(table.getcell("CHAN_FREQ", row), dtype=np.float64).reshape(-1)
            frequencies.extend(float(value) for value in channel)
    if not frequencies:
        return 0.0, 0.0, 0, True
    array = np.asarray(frequencies, dtype=np.float64)
    return float(np.min(array)), float(np.max(array)), n_spw, bool(np.all(np.diff(array) > 0.0))


def _observation_utc(observation: dict[str, str], time_s: np.ndarray) -> str:
    start = observation.get("time_start")
    stop = observation.get("time_stop")
    if start and stop:
        return f"{_mjd_seconds_to_utc(float(start))}/{_mjd_seconds_to_utc(float(stop))}"
    if time_s.size:
        return f"{_mjd_seconds_to_utc(float(time_s[0]))}/{_mjd_seconds_to_utc(float(time_s[-1]))}"
    return ""


def _mjd_seconds_to_utc(time_s: float) -> str:
    unix = float(time_s) - 3506716800.0
    return datetime.fromtimestamp(unix, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _direction_column(table: Any, name: str) -> NDArray[np.float64]:
    values = np.asarray(table.getcol(name), dtype=np.float64)
    if values.ndim == 3 and values.shape[-1] == 2:
        values = values[:, 0, :]
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError(f"{name} must reduce to shape (row, 2); got {values.shape}")
    return values


def _column_measure(table: Any, name: str) -> tuple[str, str]:
    try:
        info = table.getcolkeyword(name, "MEASINFO")
    except Exception as exc:
        raise ValueError(f"{name} measure reference cannot be omitted") from exc
    ref = ""
    if isinstance(info, dict):
        ref = str(info.get("Ref") or info.get("RefCode") or "")
    if not ref.strip():
        raise ValueError(f"{name} measure reference cannot be omitted")
    try:
        units = table.getcolkeyword(name, "QuantumUnits")
        unit = str(np.asarray(units).reshape(-1)[0])
    except Exception as exc:
        raise ValueError(f"{name} units cannot be omitted") from exc
    if not unit.strip():
        raise ValueError(f"{name} units cannot be omitted")
    return ref, unit
