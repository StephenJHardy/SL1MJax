"""Export a four-correlation 3C391 polarisation golden from the local MS."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from casacore import tables

REPO = Path(__file__).resolve().parents[1]
CORRELATIONS = ("RR", "RL", "LR", "LL")
CORRELATION_INDICES = (0, 1, 2, 3)
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
POL_TABLES = {
    "leakage_gain": "3c391.G84",
    "kcross": "3c391.Kcross",
    "dterms": "3c391.Df0",
    "angle": "3c391.Xf0",
}
VISIBILITY_CASES = (
    ("flux_angle", 0, 8),
    ("leakage_calibrator", 9, 8),
)


def default_measurement_set() -> Path:
    return Path(
        os.environ.get(
            "SL1MJAX_3C391_MS",
            REPO / "data/3c391_work_v2/3c391_ctm_mosaic_10s_spw0.ms",
        )
    )


def default_pol_reference() -> Path:
    return Path(
        os.environ.get(
            "SL1MJAX_3C391_POL_REFERENCE",
            REPO / "data/3c391/reference-pol",
        )
    )


def default_output() -> Path:
    return Path(
        os.environ.get(
            "SL1MJAX_3C391_POL_GOLDEN",
            REPO / "tests/fixtures/3c391_polarization_golden.npz",
        )
    )


def assert_disjoint_export_labels() -> None:
    visibility_labels = {label for label, *_ in VISIBILITY_CASES}
    overlap = visibility_labels & set(POL_TABLES)
    if overlap:
        raise ValueError(
            "visibility case labels collide with calibration table names: "
            + ", ".join(sorted(overlap))
        )


def select_calibrator_times(
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
    if not active_times:
        raise ValueError(f"field {field} has no unflagged times")
    chosen = np.asarray(active_times)[
        np.linspace(0, len(active_times) - 1, min(count, len(active_times)), dtype=np.int64)
    ]
    return field_rows[np.isin(time_s[field_rows], chosen)]


def _read_selected(table: Any, rows: np.ndarray, column: str) -> np.ndarray:
    selected = table.selectrows(rows)
    try:
        return np.asarray(selected.getcol(column))
    finally:
        selected.close()


def export_polarization_golden(
    *,
    measurement_set: Path | None = None,
    pol_reference: Path | None = None,
    output: Path | None = None,
) -> Path:
    measurement_set = (measurement_set or default_measurement_set()).resolve()
    pol_reference = (pol_reference or default_pol_reference()).resolve()
    output = (output or default_output()).resolve()
    flag_version_path = (
        Path(str(measurement_set) + ".flagversions") / "flags.sl1mjax_calibration_input"
    )
    payload: dict[str, Any] = {}
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "casa_version": "6.7.6.14",
        "measurement_set": measurement_set.name,
        "reference_directory": pol_reference.name,
        "reference_antenna": "ea21",
        "correlations": list(CORRELATIONS),
        "correlation_indices": list(CORRELATION_INDICES),
        "calibrators": {
            "flux_angle": {"field_id": 0, "name": "J1331+3030"},
            "leakage_calibrator": {"field_id": 9, "name": "J0319+4130"},
        },
        "flag_version": "sl1mjax_calibration_input",
        "effects": {
            "input": "raw DATA with deterministic tutorial flags",
            "model": (
                "CASA MODEL_DATA after setjy (3C286 Perley-Butler 2017 I "
                "replaced by casaguide constant 11.2% / 66° IQUV; "
                "3C84 manual unpolarised I)"
            ),
            "reference": "CASA CORRECTED_DATA after K/B/G plus Kcross/Df/Xf",
        },
        "visibility_cases": {},
        "calibration_tables": {},
    }
    assert_disjoint_export_labels()
    result_path = pol_reference / "result.json"
    if result_path.is_file():
        polarised = json.loads(result_path.read_text(encoding="utf-8"))
        if "flux_polarised_model" in polarised:
            metadata["flux_polarised_model"] = polarised["flux_polarised_model"]
    with (
        tables.table(str(measurement_set), readonly=True, ack=False) as main,
        tables.table(str(flag_version_path), readonly=True, ack=False) as input_flags,
    ):
        field_id = np.asarray(main.getcol("FIELD_ID"), dtype=np.int32)
        time_s = np.asarray(main.getcol("TIME"), dtype=np.float64)
        for label, field, time_count in VISIBILITY_CASES:
            rows = select_calibrator_times(
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
                payload[prefix + output_name] = _read_selected(main, rows, column)
            row_weight = _read_selected(main, rows, "WEIGHT")
            payload[prefix + "weight"] = np.broadcast_to(
                row_weight[:, None, :], payload[prefix + "data"].shape
            ).copy()
            payload[prefix + "flag"] = _read_selected(input_flags, rows, "FLAG")
            payload[prefix + "post_apply_flag"] = _read_selected(main, rows, "FLAG")
            payload[prefix + "row_index"] = rows
            metadata["visibility_cases"][label] = {
                "field_id": field,
                "row_count": int(rows.size),
                "time_count": int(np.unique(time_s[rows]).size),
                "scan_ids": np.unique(payload[prefix + "scan_id"]).tolist(),
            }

    with tables.table(str(measurement_set / "SPECTRAL_WINDOW"), readonly=True, ack=False) as spw:
        payload["frequency_hz"] = np.asarray(spw.getcell("CHAN_FREQ", 0), dtype=np.float64)
    with tables.table(str(measurement_set / "ANTENNA"), readonly=True, ack=False) as antenna:
        payload["antenna_name"] = np.asarray(antenna.getcol("NAME"), dtype="U")
        payload["antenna_position_m"] = np.asarray(antenna.getcol("POSITION"), dtype=np.float64)
    with tables.table(str(measurement_set / "FIELD"), readonly=True, ack=False) as field:
        payload["field_phase_direction_rad"] = np.asarray(
            field.getcol("PHASE_DIR"), dtype=np.float64
        ).reshape(field.nrows(), -1, 2)[:, 0, :]

    for label, relative_path in POL_TABLES.items():
        with tables.table(str(pol_reference / relative_path), readonly=True, ack=False) as table:
            exported_columns = []
            for column in CALIBRATION_COLUMNS:
                if (
                    column not in table.colnames()
                    or not table.nrows()
                    or not table.iscelldefined(column, 0)
                ):
                    continue
                try:
                    payload[f"{label}_{column.lower()}"] = np.asarray(table.getcol(column))
                except RuntimeError:
                    continue
                exported_columns.append(column)
            metadata["calibration_tables"][label] = {
                "source": relative_path,
                "viscal": table.getkeyword("VisCal") if "VisCal" in table.keywordnames() else None,
                "row_count": table.nrows(),
                "columns": exported_columns,
            }

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        np.savez_compressed(stream, **payload)
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


def main() -> int:
    path = export_polarization_golden()
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
