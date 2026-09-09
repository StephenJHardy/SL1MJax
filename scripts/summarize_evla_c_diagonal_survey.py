"""Build a compact frequency-suitability table from scored channel reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sl1mjax.evla_c_diagonal_survey import build_slot_status, write_json_atomic
from sl1mjax.evla_c_survey_beam import survey_catalog_record
from sl1mjax.evla_c_survey_compare import score_fixed_geometry


def summarize(report_dir: Path) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for path in sorted(Path(report_dir).glob("*channel*_report.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        main = report.get("regions", {}).get("main_lobe", {})
        mid = report.get("regions", {}).get("mid", {})
        outer = report.get("regions", {}).get("outer_diagnostic", {})
        support = report.get("support") or {}
        rows.append(
            {
                "file": path.name,
                "execution": (
                    "upper_c" if path.name.startswith("upper_c_") else "lower_c"
                ),
                "spectral_window_id": report.get("spectral_window_id"),
                "channel": report.get("channel"),
                "frequency_hz": report.get("frequency_hz"),
                "status": report.get("status"),
                "main_lobe_both_hands_accepted": report.get("main_lobe_both_hands_accepted"),
                "residual_jones_policy": report.get("residual_jones_policy"),
                "n_development": report.get("n_development"),
                "main_lobe_rr": (main.get("RR") or {}).get("residual_power"),
                "main_lobe_ll": (main.get("LL") or {}).get("residual_power"),
                "mid_rr": (mid.get("RR") or {}).get("residual_power"),
                "mid_ll": (mid.get("LL") or {}).get("residual_power"),
                "outer_rr": (outer.get("RR") or {}).get("residual_power"),
                "outer_ll": (outer.get("LL") or {}).get("residual_power"),
                "main_lobe_class": (support.get("regions") or {}).get("main_lobe", {}).get(
                    "class"
                ),
                "mid_class": (support.get("regions") or {}).get("mid", {}).get("class"),
                "outer_class": (support.get("regions") or {}).get("outer_diagnostic", {}).get(
                    "class"
                ),
                "empirical_main_lobe_accepted": bool(
                    report.get("main_lobe_both_hands_accepted")
                ),
                "numerically_qualified": False,
                "archive_integrity": (
                    "unverified" if path.name.startswith("upper_c_") else "ok"
                ),
            }
        )
    return {
        "schema_version": 1,
        "n_scored": len(rows),
        "numerical_qualification": "incomplete_no_band_wide_convergence",
        "pass_b_scope": "lower_c_spw_4_6_only",
        "slots": rows,
    }


def remask_exports(channel_dir: Path) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for export_path in sorted(Path(channel_dir).glob("*channel*_export.npz")):
        report_path = export_path.with_name(
            export_path.name.replace("_export.npz", "_report.json")
        )
        report = (
            json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
        )
        with np.load(export_path, allow_pickle=False) as handle:
            measured = handle["measured"]
            predicted = handle["predicted"]
            weight = handle["weight"]
            offset = handle["source_lm_feed"]
        scored = score_fixed_geometry(
            measured=measured,
            predicted=predicted,
            weight=weight,
            offset_lm=offset,
            source_i_jy=float(report.get("source_i_jy") or 1.0),
        )
        rows.append(
            {
                "file": export_path.name,
                "execution": (
                    "upper_c" if export_path.name.startswith("upper_c_") else "lower_c"
                ),
                "frequency_hz": report.get("frequency_hz"),
                "spectral_window_id": report.get("spectral_window_id"),
                "channel": report.get("channel"),
                "historical_main_lobe_accepted": report.get("main_lobe_both_hands_accepted"),
                **scored,
            }
        )
    return {
        "schema_version": 1,
        "mask": "fixed_radius_arcmin",
        "matched_row_scores": "pending_export_provenance",
        "n_slots": len(rows),
        "slots": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frequency-table", type=Path, default=None)
    parser.add_argument("--status-output", type=Path, default=None)
    parser.add_argument("--beam-root", type=Path, default=None)
    parser.add_argument("--catalog-output", type=Path, default=None)
    parser.add_argument("--channel-dir", type=Path, default=None)
    parser.add_argument("--fixed-geometry-output", type=Path, default=None)
    arguments = parser.parse_args()
    table = summarize(arguments.report_dir)
    write_json_atomic(arguments.output, table)
    print(arguments.output)
    channel_dir = arguments.channel_dir or arguments.report_dir / "phase5_channels"
    if channel_dir.is_dir():
        remask = remask_exports(channel_dir)
        remask_path = arguments.fixed_geometry_output or arguments.output.with_name(
            "fixed_geometry_table.json"
        )
        write_json_atomic(remask_path, remask)
        print(remask_path, remask["n_slots"])
    frequency_table_path = arguments.frequency_table or arguments.report_dir / "frequency_table.json"
    if frequency_table_path.is_file():
        inventory_path = arguments.report_dir / "upper_c_scan_inventory.json"
        inventory = (
            json.loads(inventory_path.read_text(encoding="utf-8"))
            if inventory_path.is_file()
            else None
        )
        status = build_slot_status(
            json.loads(frequency_table_path.read_text(encoding="utf-8")),
            table["slots"],
            upper_c_inventory=inventory,
        )
        status_path = arguments.status_output or arguments.output.with_name("slot_status.json")
        write_json_atomic(status_path, status)
        print(status_path, status["counts"])
    if arguments.beam_root is not None:
        catalog = survey_catalog_record(
            root=arguments.beam_root,
            scored_slots=table["slots"],
        )
        catalog_path = arguments.catalog_output or arguments.output.with_name(
            "catalog_record.json"
        )
        write_json_atomic(catalog_path, catalog)
        print(catalog_path)


if __name__ == "__main__":
    main()
