"""Measure temporal cadence, drift, and jumps in CASA calibration tables."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from analyze_srdp_calibration_corpus import (
    _array_cells,
    _dataset_id,
    _table_role,
    discover_final_tables,
)
from casacore import tables


def _summary(chunks: list[np.ndarray]) -> dict[str, float | int] | None:
    if not chunks:
        return None
    values = np.concatenate([np.ravel(chunk) for chunk in chunks])
    values = values[np.isfinite(values)]
    if not values.size:
        return None
    q25, q50, q75, q90, q95, q99 = np.percentile(
        values, [25, 50, 75, 90, 95, 99]
    )
    threshold = q75 + 3.0 * (q75 - q25)
    return {
        "count": int(values.size),
        "p50": float(q50),
        "p90": float(q90),
        "p95": float(q95),
        "p99": float(q99),
        "max": float(np.max(values)),
        "abrupt_threshold": float(threshold),
        "abrupt_fraction": float(np.mean(values > threshold)),
    }


def _names(path: Path, subtable: str, column: str) -> list[str]:
    subtable_path = path / subtable
    if not subtable_path.is_dir():
        return []
    with tables.table(str(subtable_path), readonly=True, ack=False) as table:
        return [str(value) for value in table.getcol(column)]


def _spw_frequencies(path: Path) -> np.ndarray:
    subtable_path = path / "SPECTRAL_WINDOW"
    if not subtable_path.is_dir():
        return np.asarray([], dtype=np.float64)
    with tables.table(str(subtable_path), readonly=True, ack=False) as table:
        return np.asarray(table.getcol("REF_FREQUENCY"), dtype=np.float64)


def _append(
    target: dict[str, list[np.ndarray]], source: dict[str, list[np.ndarray]]
) -> None:
    for metric, chunks in source.items():
        target[metric].extend(chunks)


def _group_changes(
    indices: np.ndarray,
    parameter: list[np.ndarray],
    flag: list[np.ndarray],
    time_s: np.ndarray,
    *,
    complex_parameter: bool,
) -> dict[str, list[np.ndarray]]:
    result: dict[str, list[np.ndarray]] = defaultdict(list)
    ordered = indices[np.argsort(time_s[indices], kind="stable")]
    for previous, current in zip(ordered[:-1], ordered[1:], strict=True):
        elapsed_min = float((time_s[current] - time_s[previous]) / 60.0)
        if elapsed_min <= 0:
            continue
        before = np.asarray(parameter[previous])
        after = np.asarray(parameter[current])
        if before.shape != after.shape:
            continue
        valid = (
            ~np.asarray(flag[previous], dtype=bool)
            & ~np.asarray(flag[current], dtype=bool)
            & np.isfinite(before)
            & np.isfinite(after)
        )
        if not np.any(valid):
            continue
        result["cadence_min"].append(np.asarray([elapsed_min]))
        if complex_parameter:
            before_valid = before[valid]
            after_valid = after[valid]
            positive = (np.abs(before_valid) > 0) & (np.abs(after_valid) > 0)
            if np.any(positive):
                amplitude_step = (
                    100.0
                    * np.abs(
                        np.log(
                            np.abs(after_valid[positive])
                            / np.abs(before_valid[positive])
                        )
                    )
                )
                result["amplitude_step_percent"].append(amplitude_step)
                result["amplitude_rate_percent_per_min"].append(
                    amplitude_step / elapsed_min
                )
            phase_step = np.degrees(
                np.abs(np.angle(after_valid * np.conj(before_valid)))
            )
            result["phase_step_deg"].append(phase_step)
            result["phase_rate_deg_per_min"].append(phase_step / elapsed_min)
        else:
            delay_step = np.abs(after[valid] - before[valid])
            result["delay_step_ns"].append(delay_step)
            result["delay_rate_ns_per_min"].append(delay_step / elapsed_min)
    return result


def analyze_table(path: Path) -> dict[str, Any]:
    """Return transition distributions for one final calibration table."""

    role = _table_role(path.name)
    if role is None:
        raise ValueError(f"unrecognized calibration role: {path}")
    with tables.table(str(path), readonly=True, ack=False) as table:
        columns = set(table.colnames())
        parameter_name = "CPARAM" if "CPARAM" in columns else "FPARAM"
        parameter = _array_cells(table, parameter_name)
        flag = _array_cells(table, "FLAG")
        antenna = np.asarray(table.getcol("ANTENNA1"), dtype=np.int32)
        spw = np.asarray(table.getcol("SPECTRAL_WINDOW_ID"), dtype=np.int32)
        field = np.asarray(table.getcol("FIELD_ID"), dtype=np.int32)
        time_s = np.asarray(table.getcol("TIME"), dtype=np.float64)
    complex_parameter = any(np.iscomplexobj(value) for value in parameter)
    antenna_names = _names(path, "ANTENNA", "NAME")
    field_names = _names(path, "FIELD", "NAME")
    frequencies = _spw_frequencies(path)
    table_metrics: dict[str, list[np.ndarray]] = defaultdict(list)
    field_metrics: dict[int, dict[str, list[np.ndarray]]] = defaultdict(
        lambda: defaultdict(list)
    )
    antenna_metrics: dict[int, dict[str, list[np.ndarray]]] = defaultdict(
        lambda: defaultdict(list)
    )
    spw_metrics: dict[int, dict[str, list[np.ndarray]]] = defaultdict(
        lambda: defaultdict(list)
    )
    keys = np.stack((antenna, spw, field), axis=1)
    for key in np.unique(keys, axis=0):
        indices = np.flatnonzero(np.all(keys == key, axis=1))
        changes = _group_changes(
            indices,
            parameter,
            flag,
            time_s,
            complex_parameter=complex_parameter,
        )
        _append(table_metrics, changes)
        _append(field_metrics[int(key[2])], changes)
        _append(antenna_metrics[int(key[0])], changes)
        _append(spw_metrics[int(key[1])], changes)

    def summarized(metrics: dict[str, list[np.ndarray]]) -> dict[str, Any]:
        return {
            name: summary
            for name, chunks in sorted(metrics.items())
            if (summary := _summary(chunks)) is not None
        }

    return {
        "obs_id": _dataset_id(path.name),
        "role": role,
        "path": str(path.resolve()),
        "row_count": len(parameter),
        "time_sample_count": int(np.unique(time_s).size),
        "time_span_min": float(np.ptp(time_s) / 60.0),
        "metrics": summarized(table_metrics),
        "fields": [
            {
                "field_id": field_id,
                "field_name": (
                    field_names[field_id]
                    if 0 <= field_id < len(field_names)
                    else str(field_id)
                ),
                "metrics": summarized(metrics),
            }
            for field_id, metrics in sorted(field_metrics.items())
        ],
        "antennas": [
            {
                "antenna_id": antenna_id,
                "antenna_name": (
                    antenna_names[antenna_id]
                    if 0 <= antenna_id < len(antenna_names)
                    else str(antenna_id)
                ),
                "metrics": summarized(metrics),
            }
            for antenna_id, metrics in sorted(antenna_metrics.items())
        ],
        "spectral_windows": [
            {
                "spectral_window_id": spw_id,
                "reference_frequency_hz": (
                    float(frequencies[spw_id])
                    if 0 <= spw_id < frequencies.size
                    else None
                ),
                "metrics": summarized(metrics),
            }
            for spw_id, metrics in sorted(spw_metrics.items())
        ],
        "_raw_metrics": table_metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/calibration_weather.json")
    )
    arguments = parser.parse_args()
    role_metrics: dict[str, dict[str, list[np.ndarray]]] = defaultdict(
        lambda: defaultdict(list)
    )
    tables_output = []
    for path in discover_final_tables(arguments.root):
        result = analyze_table(path)
        _append(role_metrics[result["role"]], result.pop("_raw_metrics"))
        tables_output.append(result)
    payload = {
        "schema_version": 1,
        "dataset_count": len({row["obs_id"] for row in tables_output}),
        "table_count": len(tables_output),
        "roles": {
            role: {
                metric: summary
                for metric, chunks in sorted(metrics.items())
                if (summary := _summary(chunks)) is not None
            }
            for role, metrics in sorted(role_metrics.items())
        },
        "tables": tables_output,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"{payload['dataset_count']} datasets, {payload['table_count']} tables, "
        f"{len(role_metrics)} roles"
    )
    print(arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
