#!/usr/bin/env python3
"""Re-export scored survey slots with row provenance. Does not overwrite archives."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sl1mjax.evla_c_diagonal_survey import (
    SCIENTIFIC_SPW45_MS,
    refuse_frozen_write,
    slot_product_stem,
    work_ms_path,
)

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from run_thol0001_evla_c_diagonal_survey import cmd_score


def _measurement_set(execution: str, spectral_window_id: int) -> Path:
    if execution == "lower_c" and spectral_window_id in {4, 5}:
        return SCIENTIFIC_SPW45_MS
    return work_ms_path(execution, spectral_window_id)


def _report_paths(report_dir: Path) -> list[Path]:
    patterns = (
        "*channel*_report.json",
        "channels/*channel*_report.json",
        "phase5_channels/*channel*_report.json",
        "phase5/*channel*_report.json",
    )
    found: list[Path] = []
    for pattern in patterns:
        found.extend(report_dir.glob(pattern))
    return [
        path
        for path in sorted(set(found))
        if "provenance_exports" not in path.parts
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--beam-root", type=Path, required=True)
    parser.add_argument("--scripts-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    arguments = parser.parse_args()
    refuse_frozen_write(arguments.export_dir)
    arguments.export_dir.mkdir(parents=True, exist_ok=True)
    seen: set[tuple[str, int, int]] = set()
    for path in _report_paths(arguments.report_dir):
        report = json.loads(path.read_text(encoding="utf-8"))
        execution = str(report.get("execution") or ("upper_c" if path.name.startswith("upper_c_") else "lower_c"))
        key = (execution, int(report["spectral_window_id"]), int(report["channel"]))
        if key in seen:
            continue
        seen.add(key)
        execution, spw, channel = key
        destination = slot_product_stem(
            arguments.export_dir,
            execution=execution,
            spectral_window_id=spw,
            channel=channel,
        ).with_name(
            slot_product_stem(
                arguments.export_dir,
                execution=execution,
                spectral_window_id=spw,
                channel=channel,
            ).name
            + "_export.npz"
        )
        if destination.is_file():
            print("HAVE", destination.name, flush=True)
            continue
        measurement_set = _measurement_set(execution, spw)
        print(f"REEXPORT {destination.name} {measurement_set}", flush=True)
        cmd_score(
            arguments.repo_root,
            arguments.export_dir,
            beam_root=arguments.beam_root,
            scripts_dir=arguments.scripts_dir,
            measurement_set=measurement_set,
            spectral_window_id=spw,
            channel=channel,
            rewrite_catalog_manifest=False,
            rewrite_status=False,
        )
    print("reexported", len(seen), "slots into", arguments.export_dir)


if __name__ == "__main__":
    main()
