"""Build compact, deterministic CASA prior-table comparison fixtures."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from casacore import tables


def _dataset_id(name: str) -> str:
    return name.split(".ms.", maxsplit=1)[0]


def _role(name: str) -> str | None:
    for suffix, role in (
        ("gc.tbl", "gain_curve"),
        ("opac.tbl", "opacity"),
        ("rq.tbl", "requantizer"),
    ):
        if name.endswith(suffix):
            return role
    return None


def _sample_rows(row_count: int, limit: int) -> np.ndarray:
    if row_count <= limit:
        return np.arange(row_count)
    return np.unique(np.linspace(0, row_count - 1, limit).astype(np.int64))


def fixture_table(path: Path, *, sample_limit: int = 128) -> dict[str, Any]:
    """Extract coordinate-aligned samples from one CASA comparison table."""

    role = _role(path.name)
    if role is None:
        raise ValueError(f"unrecognized prior table {path}")
    with tables.table(str(path), readonly=True, ack=False) as table:
        selected = _sample_rows(table.nrows(), sample_limit)
        values = np.asarray(table.getcol("FPARAM"))[selected]
        flags = np.asarray(table.getcol("FLAG"), dtype=bool)[selected]
        result = {
            "role": role,
            "viscal": str(table.getkeyword("VisCal")),
            "source_table": path.name,
            "source_row_count": int(table.nrows()),
            "sample_row": selected.tolist(),
            "time_s": np.asarray(table.getcol("TIME"))[selected].tolist(),
            "interval_s": np.asarray(table.getcol("INTERVAL"))[selected].tolist(),
            "antenna_id": np.asarray(table.getcol("ANTENNA1"))[selected].tolist(),
            "spectral_window_id": np.asarray(
                table.getcol("SPECTRAL_WINDOW_ID")
            )[selected].tolist(),
            "fparam": values.tolist(),
            "flag": flags.tolist(),
        }
    spectral_window_table = path / "SPECTRAL_WINDOW"
    if spectral_window_table.is_dir():
        with tables.table(
            str(spectral_window_table), readonly=True, ack=False
        ) as table:
            result["spw_reference_frequency_hz"] = np.asarray(
                table.getcol("REF_FREQUENCY"), dtype=np.float64
            ).tolist()
    return result


def build_fixtures(root: Path, *, sample_limit: int = 128) -> dict[str, Any]:
    """Build fixtures for every complete SRDP prior-calibration case."""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for marker in root.rglob("table.dat"):
        path = marker.parent
        if ".hifv_priorcals." not in path.name or _role(path.name) is None:
            continue
        grouped[_dataset_id(path.name)].append(
            fixture_table(path, sample_limit=sample_limit)
        )
    return {
        "schema_version": 1,
        "purpose": "CASA comparison oracle; never an independent-generator input",
        "datasets": [
            {"obs_id": obs_id, "tables": sorted(rows, key=lambda row: row["role"])}
            for obs_id, rows in sorted(grouped.items())
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sample-limit", type=int, default=128)
    arguments = parser.parse_args()
    payload = build_fixtures(arguments.root, sample_limit=arguments.sample_limit)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"{len(payload['datasets'])} datasets -> {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
