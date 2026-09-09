#!/usr/bin/env python3
"""Matched-population and scan/antenna influence from provenance exports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sl1mjax.evla_c_diagonal_survey import refuse_frozen_write, write_json_atomic
from sl1mjax.evla_c_survey_compare import (
    fixed_radius_region_masks,
    outlier_influence_by_group,
    score_fixed_geometry,
)


def _row_key(export: dict[str, np.ndarray], index: int) -> tuple[object, ...]:
    return (
        float(export["time_s"][index]),
        int(export["antenna1"][index]),
        int(export["antenna2"][index]),
        int(export["scan_id"][index]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    refuse_frozen_write(arguments.output)
    exports: list[dict[str, object]] = []
    for path in sorted(arguments.export_dir.rglob("*channel*_export.npz")):
        with np.load(path, allow_pickle=False) as handle:
            if "time_s" not in handle.files:
                continue
            payload = {str(key): np.asarray(handle[key]) for key in handle.files}
        report_path = path.with_name(path.name.replace("_export.npz", "_report.json"))
        report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
        exports.append({"file": path.name, "arrays": payload, "report": report})
    if len(exports) < 2:
        raise SystemExit("need at least two provenance exports")
    keys_by_file = []
    for item in exports:
        arrays = item["arrays"]
        keys_by_file.append({_row_key(arrays, index) for index in range(arrays["time_s"].shape[0])})
    common = set.intersection(*keys_by_file)
    matched_rows = []
    for item, keys in zip(exports, keys_by_file, strict=True):
        arrays = item["arrays"]
        keep = np.array(
            [_row_key(arrays, index) in common for index in range(arrays["time_s"].shape[0])],
            dtype=bool,
        )
        scored = score_fixed_geometry(
            measured=arrays["measured"][keep],
            predicted=arrays["predicted"][keep],
            weight=arrays["weight"][keep],
            offset_lm=arrays["source_lm_feed"][keep],
            source_i_jy=float(item["report"].get("source_i_jy") or 1.0),
        )
        masks = fixed_radius_region_masks(arrays["source_lm_feed"])
        influence = {
            "scan_id": outlier_influence_by_group(
                measured=arrays["measured"],
                predicted=arrays["predicted"],
                weight=arrays["weight"],
                mask=masks["main_lobe"],
                group=arrays["scan_id"],
            )[:8],
            "antenna1": outlier_influence_by_group(
                measured=arrays["measured"],
                predicted=arrays["predicted"],
                weight=arrays["weight"],
                mask=masks["main_lobe"],
                group=arrays["antenna1"],
            )[:8],
        }
        matched_rows.append(
            {
                "file": item["file"],
                "frequency_hz": item["report"].get("frequency_hz"),
                "n_export": int(arrays["time_s"].shape[0]),
                "n_common": int(keep.sum()),
                "matched_fixed_geometry": scored,
                "outlier_influence": influence,
                "historical_main_lobe_accepted": item["report"].get("main_lobe_both_hands_accepted"),
            }
        )
    payload = {
        "schema_version": 1,
        "n_exports": len(exports),
        "n_common_rows": len(common),
        "note": "Common-row scores use time, antennas, and scan identity. Outliers are not dropped.",
        "slots": matched_rows,
    }
    print(write_json_atomic(arguments.output, payload), len(common))


if __name__ == "__main__":
    main()
