"""Summarize final K/B/G CASA tables from an extracted SRDP corpus."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from astropy.time import Time
from casacore import tables

ROLE_NAMES = {
    "finaldelay": "delay",
    "finalBPcal": "bandpass",
    "averagephasegain": "average_phase_gain",
    "finalampgaincal": "amplitude_gain",
    "finalphasegaincal": "phase_gain",
}
PRIOR_ROLE_NAMES = {
    "gc.tbl": "gain_curve",
    "opac.tbl": "opacity",
    "rq.tbl": "requantizer",
}
CSV_FIELDS = (
    "obs_id",
    "project_code",
    "obs_start",
    "configuration",
    "qa_class",
    "qa_status",
    "role",
    "viscal",
    "row_count",
    "antenna_count",
    "spectral_window_count",
    "field_count",
    "time_sample_count",
    "time_span_min",
    "flagged_fraction",
    "invalid_antenna_spw_fraction",
    "snr_p10",
    "snr_median",
    "snr_p90",
    "amplitude_p05",
    "amplitude_median",
    "amplitude_p95",
    "delay_abs_p50_ns",
    "delay_abs_p95_ns",
    "delay_abs_max_ns",
    "phase_circular_std_deg",
)


def _quantile(values: np.ndarray, percentile: float) -> float | None:
    finite = values[np.isfinite(values)]
    return None if finite.size == 0 else float(np.percentile(finite, percentile))


def _dataset_id(table_name: str) -> str:
    return table_name.split(".ms.", maxsplit=1)[0]


def _table_role(table_name: str) -> str | None:
    return next(
        (role for term, role in ROLE_NAMES.items() if term in table_name), None
    )


def _prior_table_role(table_name: str) -> str | None:
    return next(
        (
            role
            for suffix, role in PRIOR_ROLE_NAMES.items()
            if table_name.endswith(suffix)
        ),
        None,
    )


def _iso_time(casa_seconds: float) -> str:
    return str(Time(casa_seconds / 86400.0, format="mjd", scale="utc").isot)


def _invalid_pair_fraction(
    antenna: np.ndarray, spectral_window: np.ndarray, valid_rows: np.ndarray
) -> float:
    pairs = np.stack((antenna, spectral_window), axis=1)
    unique_pairs, inverse = np.unique(pairs, axis=0, return_inverse=True)
    valid_pairs = np.zeros(unique_pairs.shape[0], dtype=bool)
    np.logical_or.at(valid_pairs, inverse, valid_rows)
    return float(1.0 - np.mean(valid_pairs))


def _array_cells(table: Any, name: str) -> list[np.ndarray]:
    try:
        values = np.asarray(table.getcol(name))
        return [values[row] for row in range(values.shape[0])]
    except RuntimeError:
        return [np.asarray(table.getcell(name, row)) for row in range(table.nrows())]


def summarize_table(path: Path) -> dict[str, Any]:
    """Summarize one final calibration table."""

    role = _table_role(path.name)
    if role is None:
        raise ValueError(f"unrecognized final calibration table: {path}")
    with tables.table(str(path), readonly=True, ack=False) as table:
        columns = set(table.colnames())
        parameter_name = "CPARAM" if "CPARAM" in columns else "FPARAM"
        parameter = _array_cells(table, parameter_name)
        flag = _array_cells(table, "FLAG")
        snr = _array_cells(table, "SNR")
        antenna = np.asarray(table.getcol("ANTENNA1"), dtype=np.int32)
        spectral_window = np.asarray(
            table.getcol("SPECTRAL_WINDOW_ID"), dtype=np.int32
        )
        field = np.asarray(table.getcol("FIELD_ID"), dtype=np.int32)
        time_s = np.asarray(table.getcol("TIME"), dtype=np.float64)
        viscal = str(table.getkeyword("VisCal"))

    valid_parameter_rows = []
    valid_snr_rows = []
    valid_rows = np.zeros(len(parameter), dtype=bool)
    valid_count = 0
    sample_count = 0
    for row, (row_parameter, row_flag, row_snr) in enumerate(
        zip(parameter, flag, snr, strict=True)
    ):
        row_flag = np.asarray(row_flag, dtype=bool)
        row_snr = np.asarray(row_snr, dtype=np.float64)
        valid = ~row_flag & np.isfinite(row_parameter)
        valid_rows[row] = bool(np.any(valid))
        valid_count += int(np.count_nonzero(valid))
        sample_count += int(valid.size)
        valid_parameter_rows.append(row_parameter[valid])
        valid_snr_rows.append(row_snr[valid & np.isfinite(row_snr)])
    valid_parameter = np.concatenate(valid_parameter_rows)
    valid_snr = np.concatenate(valid_snr_rows)
    summary: dict[str, Any] = {
        "obs_id": _dataset_id(path.name),
        "role": role,
        "viscal": viscal,
        "path": str(path.resolve()),
        "row_count": len(parameter),
        "antenna_count": int(np.unique(antenna).size),
        "spectral_window_count": int(np.unique(spectral_window).size),
        "field_count": int(np.unique(field).size),
        "time_sample_count": int(np.unique(time_s).size),
        "time_start": _iso_time(float(np.min(time_s))),
        "time_stop": _iso_time(float(np.max(time_s))),
        "time_span_min": float(np.ptp(time_s) / 60.0),
        "flagged_fraction": float(1.0 - valid_count / sample_count),
        "invalid_antenna_spw_fraction": _invalid_pair_fraction(
            antenna, spectral_window, valid_rows
        ),
        "snr_p10": _quantile(valid_snr, 10),
        "snr_median": _quantile(valid_snr, 50),
        "snr_p90": _quantile(valid_snr, 90),
    }
    if np.iscomplexobj(valid_parameter):
        amplitude = np.abs(valid_parameter)
        phase = np.angle(valid_parameter)
        concentration = (
            float(np.abs(np.mean(np.exp(1j * phase)))) if phase.size else np.nan
        )
        summary.update(
            {
                "amplitude_p05": _quantile(amplitude, 5),
                "amplitude_median": _quantile(amplitude, 50),
                "amplitude_p95": _quantile(amplitude, 95),
                "phase_circular_std_deg": (
                    None
                    if not 0.0 < concentration <= 1.0
                    else float(np.degrees(np.sqrt(-2.0 * np.log(concentration))))
                ),
            }
        )
    elif role == "delay":
        delay_ns = np.abs(np.asarray(valid_parameter, dtype=np.float64))
        summary.update(
            {
                "delay_abs_p50_ns": _quantile(delay_ns, 50),
                "delay_abs_p95_ns": _quantile(delay_ns, 95),
                "delay_abs_max_ns": _quantile(delay_ns, 100),
            }
        )
    return summary


def discover_final_tables(root: Path) -> list[Path]:
    """Find top-level final calibration tables, excluding their subtables."""

    result = []
    for marker in root.rglob("table.dat"):
        path = marker.parent
        if _table_role(path.name) is None:
            continue
        try:
            with tables.table(str(path), readonly=True, ack=False) as table:
                if table.info().get("type") == "Calibration":
                    result.append(path)
        except RuntimeError:
            continue
    return sorted(result)


def summarize_prior_table(path: Path) -> dict[str, Any]:
    """Inventory a CASA EGainCurve, TOpac, or RQ comparison oracle."""

    role = _prior_table_role(path.name)
    if role is None:
        raise ValueError(f"unrecognized prior calibration table: {path}")
    with tables.table(str(path), readonly=True, ack=False) as table:
        values = np.asarray(table.getcol("FPARAM"))
        flag = np.asarray(table.getcol("FLAG"), dtype=bool)
        antenna = np.asarray(table.getcol("ANTENNA1"), dtype=np.int32)
        spw = np.asarray(table.getcol("SPECTRAL_WINDOW_ID"), dtype=np.int32)
        viscal = str(table.getkeyword("VisCal"))
    valid_values = values[~flag & np.isfinite(values)]
    return {
        "obs_id": _dataset_id(path.name),
        "role": role,
        "viscal": viscal,
        "path": str(path.resolve()),
        "row_count": int(values.shape[0]),
        "antenna_count": int(np.unique(antenna).size),
        "spectral_window_count": int(np.unique(spw).size),
        "flagged_fraction": float(np.mean(flag)),
        "value_p05": _quantile(valid_values, 5),
        "value_median": _quantile(valid_values, 50),
        "value_p95": _quantile(valid_values, 95),
    }


def discover_prior_tables(root: Path) -> list[Path]:
    """Find top-level SRDP prior tables without descending into subtables."""

    result = []
    for marker in root.rglob("table.dat"):
        path = marker.parent
        if ".hifv_priorcals." not in path.name or _prior_table_role(path.name) is None:
            continue
        try:
            with tables.table(str(path), readonly=True, ack=False) as table:
                if table.info().get("type") == "Calibration":
                    result.append(path)
        except RuntimeError:
            continue
    return sorted(result)


def summarize_flag_table(path: Path, *, chunk_rows: int = 8192) -> dict[str, Any]:
    """Summarize a restored Pipeline_Final flag version without the original MS."""

    flagversions = next(
        parent for parent in path.parents if parent.name.endswith(".ms.flagversions")
    )
    obs_id = flagversions.name.removesuffix(".ms.flagversions")
    flagged = 0
    samples = 0
    flagged_rows = 0
    with tables.table(str(path), readonly=True, ack=False) as table:
        row_count = int(table.nrows())
        sample_shape: tuple[int, ...] = ()
        for start in range(0, row_count, chunk_rows):
            count = min(chunk_rows, row_count - start)
            flag_row = np.asarray(
                table.getcol("FLAG_ROW", startrow=start, nrow=count), dtype=bool
            )
            try:
                flag_cells = np.asarray(
                    table.getcol("FLAG", startrow=start, nrow=count), dtype=bool
                )
                cells = [flag_cells[row] for row in range(count)]
            except RuntimeError:
                cells = [
                    np.asarray(value, dtype=bool)
                    for value in table.getvarcol(
                        "FLAG", startrow=start, nrow=count
                    ).values()
                ]
            for row, flag in enumerate(cells):
                sample_shape = tuple(int(value) for value in flag.shape)
                effective = flag | flag_row[row]
                flagged += int(np.count_nonzero(effective))
                samples += int(effective.size)
            flagged_rows += int(np.count_nonzero(flag_row))
    return {
        "obs_id": obs_id,
        "path": str(path.resolve()),
        "row_count": row_count,
        "sample_shape": sample_shape,
        "sample_count": samples,
        "flagged_sample_count": flagged,
        "flagged_fraction": flagged / samples,
        "flagged_row_count": flagged_rows,
        "flagged_row_fraction": flagged_rows / row_count,
    }


def discover_flag_tables(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("flags.Pipeline_Final")
        if (path / "table.dat").is_file()
    )


def _selected_metadata(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {record["obs_id"]: record for record in payload["selected"]}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "--selected",
        type=Path,
        default=Path("outputs/nrao_calibration_corpus/selected.json"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/srdp_corpus_analysis.json")
    )
    parser.add_argument(
        "--flag-root",
        type=Path,
        help="Optional root containing restored flags.Pipeline_Final tables.",
    )
    arguments = parser.parse_args()

    metadata = _selected_metadata(arguments.selected)
    rows = []
    for table_path in discover_final_tables(arguments.root):
        row = summarize_table(table_path)
        selected = metadata.get(row["obs_id"], {})
        row.update(
            {
                "project_code": selected.get("project_code"),
                "obs_start": selected.get("obs_start"),
                "configuration": selected.get("configuration"),
                "qa_class": selected.get("qa_class"),
                "qa_status": selected.get("qa_status"),
                "qa_notes": selected.get("qa_notes"),
            }
        )
        rows.append(row)
    prior_rows = [
        summarize_prior_table(table_path)
        for table_path in discover_prior_tables(arguments.root)
    ]

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["obs_id"]].append(row)
    flag_summaries = (
        {}
        if arguments.flag_root is None
        else {
            summary["obs_id"]: summary
            for summary in (
                summarize_flag_table(path)
                for path in discover_flag_tables(arguments.flag_root)
            )
        }
    )
    payload = {
        "schema_version": 1,
        "dataset_count": len(grouped),
        "table_count": len(rows),
        "prior_table_count": len(prior_rows),
        "prior_tables": prior_rows,
        "datasets": [
            {
                "obs_id": obs_id,
                "project_code": tables_for_dataset[0]["project_code"],
                "obs_start": tables_for_dataset[0]["obs_start"],
                "configuration": tables_for_dataset[0]["configuration"],
                "qa_class": tables_for_dataset[0]["qa_class"],
                "qa_status": tables_for_dataset[0]["qa_status"],
                "qa_notes": tables_for_dataset[0]["qa_notes"],
                "visibility_flags": flag_summaries.get(obs_id),
                "tables": tables_for_dataset,
            }
            for obs_id, tables_for_dataset in sorted(grouped.items())
        ],
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(arguments.output.with_suffix(".csv"), rows)
    print(
        f"{payload['dataset_count']} datasets, {payload['table_count']} final K/B/G tables, "
        f"{payload['prior_table_count']} prior tables"
    )
    print(arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
