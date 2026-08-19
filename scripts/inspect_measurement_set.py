"""Write a calibration-oriented, data-light MeasurementSet inventory as JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from casacore import tables

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
    return {
        "schema_version": 1,
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
            "flagged_fraction": flagged / samples,
            "flagged_row_count": flagged_rows,
        },
        "antennas": antennas,
        "fields": fields,
        "states": states,
        "spectral_windows": spectral_windows,
        "polarizations": polarizations,
        "data_descriptions": data_descriptions,
        "observations": observations,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("measurement_set", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    inventory = inspect_measurement_set(arguments.measurement_set)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(inventory, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
