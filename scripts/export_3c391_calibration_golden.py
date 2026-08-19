"""Export compact 3C391 visibility and CASA calibration-table references."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from casacore import tables

MEASUREMENT_SET = Path(
    os.environ.get(
        "SL1MJAX_3C391_MS",
        "/Volumes/BagOfWinds/NRAO/3c391/work-v2/3c391_ctm_mosaic_10s_spw0.ms",
    )
)
REFERENCE = Path(
    os.environ.get(
        "SL1MJAX_3C391_REFERENCE",
        "/Volumes/BagOfWinds/NRAO/3c391/reference-v2",
    )
)
OUTPUT = Path(
    os.environ.get(
        "SL1MJAX_3C391_GOLDEN",
        "/Volumes/BagOfWinds/NRAO/3c391/golden/3c391_calibration_golden.npz",
    )
)

CALIBRATION_TABLES = {
    "antpos": "3c391.antpos",
    "phase": "3c391.G0",
    "delay": "3c391.K0",
    "bandpass": "3c391.B0",
    "gain": "3c391.G1",
    "flux_gain": "3c391.fluxscale1",
}
CALIBRATION_COLUMNS = (
    "TIME",
    "FIELD_ID",
    "SPECTRAL_WINDOW_ID",
    "ANTENNA1",
    "ANTENNA2",
    "INTERVAL",
    "SCAN_NUMBER",
    "OBSERVATION_ID",
    "CPARAM",
    "FPARAM",
    "PARAMERR",
    "FLAG",
    "SNR",
)


def _select_calibrator_times(
    field_id: np.ndarray,
    time_s: np.ndarray,
    flag_table: Any,
    field: int,
    count: int,
) -> np.ndarray:
    field_rows = np.flatnonzero(field_id == field)
    active_times = []
    for time in np.unique(time_s[field_rows]):
        rows = field_rows[time_s[field_rows] == time]
        if any(not np.all(flag_table.getcell("FLAG", int(row))) for row in rows):
            active_times.append(time)
    chosen = np.asarray(active_times)[
        np.linspace(0, len(active_times) - 1, count, dtype=np.int64)
    ]
    return field_rows[np.isin(time_s[field_rows], chosen)]


def _read_selected(table: Any, rows: np.ndarray, column: str) -> np.ndarray:
    selected = table.selectrows(rows)
    try:
        return np.asarray(selected.getcol(column))
    finally:
        selected.close()


payload: dict[str, Any] = {}
metadata: dict[str, Any] = {
    "schema_version": 1,
    "casa_version": "6.7.6.14",
    "measurement_set": MEASUREMENT_SET.name,
    "reference_directory": REFERENCE.name,
    "reference_antenna": "ea21",
    "correlations": ["RR", "LL"],
    "correlation_indices": [0, 3],
    "calibrators": {
        "flux_bandpass": {"field_id": 0, "name": "J1331+3030"},
        "time_gain": {"field_id": 1, "name": "J1822-0938"},
    },
    "flag_version": "sl1mjax_calibration_input",
    "effects": {
        "input": "raw DATA with deterministic tutorial flags",
        "model": "CASA MODEL_DATA after Perley-Butler 2017 setjy",
        "reference": "CASA CORRECTED_DATA after K/B/G calibration",
    },
    "visibility_cases": {},
    "calibration_tables": {},
}

flag_version_path = (
    Path(str(MEASUREMENT_SET) + ".flagversions")
    / "flags.sl1mjax_calibration_input"
)
with (
    tables.table(str(MEASUREMENT_SET), readonly=True, ack=False) as main,
    tables.table(str(flag_version_path), readonly=True, ack=False) as input_flags,
):
    field_id = np.asarray(main.getcol("FIELD_ID"), dtype=np.int32)
    time_s = np.asarray(main.getcol("TIME"), dtype=np.float64)
    for label, field, time_count in (
        ("flux_bandpass", 0, 4),
        ("time_gain", 1, 6),
    ):
        rows = _select_calibrator_times(
            field_id, time_s, input_flags, field, time_count
        )
        prefix = f"{label}_"
        for column, output_name in (
            ("UVW", "uvw_m"),
            ("TIME", "time_s"),
            ("ANTENNA1", "antenna1"),
            ("ANTENNA2", "antenna2"),
            ("SCAN_NUMBER", "scan_id"),
            ("FIELD_ID", "field_id"),
        ):
            payload[prefix + output_name] = _read_selected(main, rows, column)
        for column, output_name in (
            ("DATA", "data"),
            ("MODEL_DATA", "model_data"),
            ("CORRECTED_DATA", "corrected_data"),
        ):
            payload[prefix + output_name] = _read_selected(main, rows, column)[
                :, :, [0, 3]
            ]
        row_weight = _read_selected(main, rows, "WEIGHT")[:, [0, 3]]
        payload[prefix + "weight"] = np.broadcast_to(
            row_weight[:, None, :], payload[prefix + "data"].shape
        ).copy()
        payload[prefix + "flag"] = _read_selected(input_flags, rows, "FLAG")[
            :, :, [0, 3]
        ]
        payload[prefix + "post_apply_flag"] = _read_selected(main, rows, "FLAG")[
            :, :, [0, 3]
        ]
        payload[prefix + "row_index"] = rows
        metadata["visibility_cases"][label] = {
            "field_id": field,
            "row_count": int(rows.size),
            "time_count": int(np.unique(time_s[rows]).size),
            "scan_ids": np.unique(
                payload[prefix + "scan_id"]
            ).tolist(),
        }

with tables.table(
    str(MEASUREMENT_SET / "SPECTRAL_WINDOW"), readonly=True, ack=False
) as spectral_window:
    payload["frequency_hz"] = np.asarray(
        spectral_window.getcell("CHAN_FREQ", 0), dtype=np.float64
    )
with tables.table(
    str(MEASUREMENT_SET / "ANTENNA"), readonly=True, ack=False
) as antenna:
    payload["antenna_name"] = np.asarray(antenna.getcol("NAME"), dtype="U")
    payload["antenna_position_m"] = np.asarray(
        antenna.getcol("POSITION"), dtype=np.float64
    )
with tables.table(str(MEASUREMENT_SET / "FIELD"), readonly=True, ack=False) as field:
    payload["field_phase_direction_rad"] = np.asarray(
        field.getcol("PHASE_DIR"), dtype=np.float64
    ).reshape(field.nrows(), -1, 2)[:, 0, :]

for label, relative_path in CALIBRATION_TABLES.items():
    with tables.table(str(REFERENCE / relative_path), readonly=True, ack=False) as table:
        exported_columns = []
        for column in CALIBRATION_COLUMNS:
            if (
                column not in table.colnames()
                or not table.nrows()
                or not table.iscelldefined(column, 0)
            ):
                continue
            try:
                payload[f"{label}_{column.lower()}"] = np.asarray(
                    table.getcol(column)
                )
            except RuntimeError:
                continue
            exported_columns.append(column)
        metadata["calibration_tables"][label] = {
            "source": relative_path,
            "row_count": table.nrows(),
            "columns": exported_columns,
        }

result = json.loads((REFERENCE / "result.json").read_text(encoding="utf-8"))
metadata["flagged_fraction_calibration_input"] = (
    result["flag_summary_calibration_input"]["flagged"]
    / result["flag_summary_calibration_input"]["total"]
)
metadata["flagged_fraction_post_apply"] = (
    result["flag_summary_after"]["flagged"]
    / result["flag_summary_after"]["total"]
)
metadata["flux_scale"] = result["flux_scale"]

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
with OUTPUT.open("wb") as stream:
    np.savez_compressed(stream, **payload)
OUTPUT.with_suffix(".json").write_text(
    json.dumps(metadata, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(OUTPUT)
