"""Write a calibration-oriented, data-light MeasurementSet inventory as JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from casacore import tables

from sl1mjax.fullpol_prep import (
    FULLPOL_SCIENCE_FIELDS,
    attach_fullpol_contract,
    load_calibration_state,
    load_flag_versions,
)

CORRELATION_NAMES = {
    1: "I",
    2: "Q",
    3: "U",
    4: "V",
    5: "RR",
    6: "RL",
    7: "LR",
    8: "LL",
    9: "XX",
    10: "XY",
    11: "YX",
    12: "YY",
}


def _json_value(value: Any) -> Any:
    array = np.asarray(value)
    return array.item() if array.ndim == 0 else array.tolist()


def _column_if_present(table: Any, name: str) -> np.ndarray | None:
    if name not in table.colnames():
        return None
    return np.asarray(table.getcol(name))


def _rows(table: Any, columns: tuple[str, ...]) -> list[dict[str, Any]]:
    available = set(table.colnames())
    return [
        {
            name.lower(): _json_value(table.getcell(name, row))
            for name in columns
            if name in available
        }
        for row in range(table.nrows())
    ]


def inspect_measurement_set(path: Path, *, chunk_rows: int = 4096) -> dict[str, Any]:
    """Collect metadata and aggregate flags without loading visibility data."""

    with tables.table(str(path), readonly=True, ack=False) as main:
        columns = list(main.colnames())
        row_count = main.nrows()
        scalar_columns = {}
        for name in (
            "FIELD_ID",
            "DATA_DESC_ID",
            "SCAN_NUMBER",
            "STATE_ID",
            "OBSERVATION_ID",
            "ANTENNA1",
            "ANTENNA2",
            "TIME",
            "INTERVAL",
        ):
            values = _column_if_present(main, name)
            if values is not None:
                scalar_columns[name] = values

        flagged = 0
        samples = 0
        flagged_rows = 0
        field_ids_col = scalar_columns.get("FIELD_ID")
        active_by_field: dict[int, dict[str, int]] = {}
        active_by_channel: list[int] | None = None
        active_by_correlation: list[int] | None = None
        for start in range(0, row_count, chunk_rows):
            count = min(chunk_rows, row_count - start)
            flag = np.asarray(main.getcol("FLAG", startrow=start, nrow=count), dtype=bool)
            row_flag = np.asarray(
                main.getcol("FLAG_ROW", startrow=start, nrow=count), dtype=bool
            )
            effective = flag | row_flag[:, None, None]
            flagged += int(np.count_nonzero(effective))
            samples += effective.size
            flagged_rows += int(np.count_nonzero(row_flag))
            active = ~effective
            if active_by_channel is None:
                active_by_channel = [0] * int(active.shape[1])
                active_by_correlation = [0] * int(active.shape[2])
            for channel in range(active.shape[1]):
                active_by_channel[channel] += int(np.count_nonzero(active[:, channel, :]))
            for correlation in range(active.shape[2]):
                active_by_correlation[correlation] += int(
                    np.count_nonzero(active[:, :, correlation])
                )
            if field_ids_col is not None:
                chunk_fields = np.asarray(field_ids_col[start : start + count])
                for field_id in np.unique(chunk_fields):
                    selected = chunk_fields == field_id
                    payload = active_by_field.setdefault(
                        int(field_id), {name: 0 for name in ("RR", "RL", "LR", "LL")}
                    )
                    names = ("RR", "RL", "LR", "LL")
                    for index, name in enumerate(names[: active.shape[2]]):
                        payload[name] += int(
                            np.count_nonzero(active[selected][:, :, index])
                        )

    def unique(name: str) -> list[Any]:
        values = scalar_columns.get(name)
        return [] if values is None else np.unique(values).tolist()

    with tables.table(str(path / "ANTENNA"), readonly=True, ack=False) as table:
        antennas = _rows(
            table, ("NAME", "STATION", "POSITION", "DISH_DIAMETER", "MOUNT")
        )
    with tables.table(str(path / "FIELD"), readonly=True, ack=False) as table:
        fields = _rows(
            table,
            ("NAME", "SOURCE_ID", "PHASE_DIR", "DELAY_DIR", "REFERENCE_DIR"),
        )
    with tables.table(str(path / "STATE"), readonly=True, ack=False) as table:
        states = _rows(table, ("OBS_MODE", "SIG", "REF", "CAL", "LOAD"))
    with tables.table(
        str(path / "SPECTRAL_WINDOW"), readonly=True, ack=False
    ) as table:
        spectral_windows = []
        for row in range(table.nrows()):
            frequencies = np.asarray(table.getcell("CHAN_FREQ", row), dtype=np.float64)
            widths = np.asarray(table.getcell("CHAN_WIDTH", row), dtype=np.float64)
            spectral_windows.append(
                {
                    "name": str(table.getcell("NAME", row)),
                    "channel_count": int(frequencies.size),
                    "frequency_min_hz": float(np.min(frequencies)),
                    "frequency_max_hz": float(np.max(frequencies)),
                    "reference_frequency_hz": float(table.getcell("REF_FREQUENCY", row)),
                    "channel_width_hz": widths.tolist(),
                }
            )
    with tables.table(str(path / "POLARIZATION"), readonly=True, ack=False) as table:
        polarizations = []
        for row in range(table.nrows()):
            codes = np.asarray(table.getcell("CORR_TYPE", row), dtype=np.int32)
            polarizations.append(
                {
                    "correlation_codes": codes.tolist(),
                    "correlations": [
                        CORRELATION_NAMES.get(int(code), f"UNKNOWN_{code}")
                        for code in codes
                    ],
                }
            )
    with tables.table(
        str(path / "DATA_DESCRIPTION"), readonly=True, ack=False
    ) as table:
        data_descriptions = _rows(
            table, ("SPECTRAL_WINDOW_ID", "POLARIZATION_ID", "FLAG_ROW")
        )
    with tables.table(str(path / "OBSERVATION"), readonly=True, ack=False) as table:
        observations = _rows(
            table,
            (
                "TELESCOPE_NAME",
                "OBSERVER",
                "PROJECT",
                "TIME_RANGE",
                "RELEASE_DATE",
            ),
        )

    times = scalar_columns.get("TIME")
    intervals = scalar_columns.get("INTERVAL")
    correlation_names = (
        polarizations[0]["correlations"] if polarizations else []
    )
    flag_versions = load_flag_versions(path)
    calibration_state = load_calibration_state(path)
    inventory = {
        "schema_version": 2,
        "path": str(path.resolve()),
        "row_count": row_count,
        "columns": columns,
        "visibility_columns": [
            name for name in ("DATA", "MODEL_DATA", "CORRECTED_DATA") if name in columns
        ],
        "field_ids": unique("FIELD_ID"),
        "data_description_ids": unique("DATA_DESC_ID"),
        "scan_ids": unique("SCAN_NUMBER"),
        "state_ids": unique("STATE_ID"),
        "observation_ids": unique("OBSERVATION_ID"),
        "antenna_ids": sorted(
            set(unique("ANTENNA1")) | set(unique("ANTENNA2"))
        ),
        "time_range_s": (
            [] if times is None else [float(np.min(times)), float(np.max(times))]
        ),
        "integration_interval_s": (
            [] if intervals is None else np.unique(intervals).tolist()
        ),
        "flags": {
            "sample_count": samples,
            "flagged_sample_count": flagged,
            "flagged_fraction": (flagged / samples) if samples else 0.0,
            "flagged_row_count": flagged_rows,
        },
        "antennas": antennas,
        "fields": fields,
        "states": states,
        "spectral_windows": spectral_windows,
        "polarizations": polarizations,
        "data_descriptions": data_descriptions,
        "observations": observations,
        "flag_versions": flag_versions,
        "casa_version": (calibration_state or {}).get("casa_version") or _casa_version(path),
        "source_ms_hash": _source_ms_hashes(path, calibration_state),
        "calibration_table_hashes": _calibration_table_hashes(calibration_state),
        "active_samples": {
            "by_correlation": {
                name: int(active_by_correlation[index])
                if active_by_correlation is not None and index < len(active_by_correlation)
                else 0
                for index, name in enumerate(correlation_names)
            },
            "by_channel": active_by_channel or [],
            "by_field": {str(field): counts for field, counts in sorted(active_by_field.items())},
            "science_fields": list(FULLPOL_SCIENCE_FIELDS),
        },
    }
    return attach_fullpol_contract(inventory, calibration_state=calibration_state)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_ms_hashes(
    path: Path, state: dict[str, Any] | None
) -> dict[str, Any]:
    payload: dict[str, Any] = {"product": _path_fingerprint(path)}
    source = Path(str((state or {}).get("source_ms") or ""))
    if source.is_dir() and source.resolve() != path.resolve():
        payload["source"] = _path_fingerprint(source)
    return payload


def _path_fingerprint(path: Path) -> dict[str, Any]:
    info = path / "table.info"
    listing = Path(str(path) + ".flagversions") / "FLAG_VERSION_LIST"
    payload = {
        "path": str(path.resolve()),
        "table_info_sha256": _sha256_file(info) if info.is_file() else None,
        "flag_version_list_sha256": _sha256_file(listing) if listing.is_file() else None,
        "file_sizes": {
            item.name: item.stat().st_size
            for item in sorted(path.iterdir())
            if item.is_file() and not item.name.startswith(".")
        },
    }
    return payload


def _calibration_table_hashes(state: dict[str, Any] | None) -> dict[str, str]:
    hashes: dict[str, str] = {}
    if not state:
        return hashes
    tables = state.get("calibration_tables") or {}
    for name, value in tables.items():
        path = Path(str(value))
        candidate = path / "table.dat" if path.is_dir() else path
        if candidate.is_file():
            hashes[str(name)] = _sha256_file(candidate)
    return hashes


def _casa_version(path: Path) -> str | None:
    env = os.environ.get("SL1MJAX_CASA_VERSION")
    if env:
        return env
    history = path / "HISTORY"
    if not history.is_dir():
        return None
    try:
        with tables.table(str(history), readonly=True, ack=False) as table:
            columns = set(table.colnames())
            for row in range(min(table.nrows(), 200)):
                origin = (
                    str(table.getcell("ORIGIN", row)) if "ORIGIN" in columns else ""
                )
                message = (
                    str(table.getcell("MESSAGE", row)) if "MESSAGE" in columns else ""
                )
                for text in (origin, message):
                    if "CASA" in text and any(char.isdigit() for char in text):
                        return text.strip()
    except RuntimeError:
        return None
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("measurement_set", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--allow-contract-failure",
        action="store_true",
        help="Write the inventory even when RL/LR or science-field checks fail.",
    )
    arguments = parser.parse_args()
    inventory = inspect_measurement_set(arguments.measurement_set)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(inventory, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(arguments.output)
    failures = inventory.get("contract_failures") or []
    if failures and not arguments.allow_contract_failure:
        print("fullpol contract failed:", file=sys.stderr)
        for item in failures:
            print(f"  {item}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
