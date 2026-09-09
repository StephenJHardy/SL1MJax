#!/usr/bin/env python3
"""Verify the 64 native 3C391 planes against locked EVLA-C conventions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sl1mjax.cassbeam_evla_c import evla_c_feedtaper_db
from sl1mjax.cassbeam_highres import EXPECTED_RASTER, HighresCassbeamCatalog
from sl1mjax.evla_c_diagonal_survey import refuse_frozen_write, write_json_atomic
from sl1mjax.evla_c_survey_beam import (
    IMAGING_NODE_MHZ,
    NATIVE_3C391_MHZ,
    survey_catalog_digest,
)
from sl1mjax.evla_c_survey_compare import SURVEY_MODEL_ID


def _params(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.split("%", 1)[0].strip()
        if stripped and "=" in stripped:
            key, value = stripped.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beam-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--prior-manifest",
        type=Path,
        default=None,
        help="Optional previous manifest.json used to prove the three existing planes are unchanged.",
    )
    arguments = parser.parse_args()
    refuse_frozen_write(arguments.output)
    root = arguments.beam_root
    catalog = HighresCassbeamCatalog(root, expected_model_id=SURVEY_MODEL_ID)
    present = set(catalog.frequency_mhz_list())
    missing = [mhz for mhz in NATIVE_3C391_MHZ if mhz not in present]
    extra_native = [mhz for mhz in present if mhz in set(NATIVE_3C391_MHZ)]
    prior = {}
    if arguments.prior_manifest is not None and arguments.prior_manifest.is_file():
        prior = json.loads(arguments.prior_manifest.read_text(encoding="utf-8")).get("files_sha256") or {}
    checks: list[dict[str, object]] = []
    for mhz in NATIVE_3C391_MHZ:
        if mhz not in present:
            checks.append({"frequency_mhz": mhz, "present": False, "passed": False})
            continue
        plane = catalog.plane(mhz * 1.0e6)
        params_rel = catalog._planes_by_mhz[mhz]["params"]
        data_rel = catalog._planes_by_mhz[mhz]["data"]
        params = _params(root / params_rel)
        centre = plane.origin_native()
        normalized = np.linalg.inv(centre) @ centre
        _on_axis, support_ok = plane.lookup(np.array([0.0]), np.array([0.0]), off_diagonal=False)
        _far, far_ok = plane.lookup(np.array([1.0]), np.array([1.0]), off_diagonal=False)
        expected_taper = evla_c_feedtaper_db(mhz * 1.0e6)
        recorded_taper = float(params["feedtaper"]) if "feedtaper" in params else float("nan")
        prior_data = prior.get(str(data_rel))
        actual_data = catalog.files_sha256[str(data_rel)]
        row = {
            "frequency_mhz": mhz,
            "present": True,
            "params_freq_ghz": float(params["freq"]),
            "gridsize": int(float(params["gridsize"])),
            "pixelsperbeam": int(float(params["pixelsperbeam"])),
            "feed_x": params.get("feed_x"),
            "feedtaper_db": recorded_taper,
            "expected_feedtaper_db": expected_taper,
            "raster": list(plane.jones_norm.shape),
            "normalization_identity_abs": float(np.max(np.abs(normalized - np.eye(2)))),
            "on_axis_supported": bool(np.all(support_ok)),
            "far_field_supported": bool(np.any(far_ok)),
            "far_field_n_unsupported": int(np.sum(~far_ok)),
            "data_sha256": actual_data,
            "prior_data_sha256": prior_data,
            "prior_unchanged": prior_data is None or prior_data == actual_data,
            "checksum_ok": catalog._verify_file(str(data_rel)) == actual_data,
        }
        row["passed"] = bool(
            abs(float(params["freq"]) * 1000.0 - mhz) < 0.5
            and int(float(params["gridsize"])) == 1024
            and int(float(params["pixelsperbeam"])) == 32
            and abs(recorded_taper - expected_taper) < 5.0e-4
            and tuple(plane.jones_norm.shape) == EXPECTED_RASTER
            and row["normalization_identity_abs"] < 1.0e-12
            and row["on_axis_supported"]
            and not row["far_field_supported"]
            and row["prior_unchanged"]
            and row["checksum_ok"]
        )
        checks.append(row)
    existing_nodes = [row for row in checks if int(row["frequency_mhz"]) in IMAGING_NODE_MHZ]
    payload = {
        "schema_version": 1,
        "digest": survey_catalog_digest(root),
        "n_native_required": len(NATIVE_3C391_MHZ),
        "n_native_present": len(extra_native),
        "missing_mhz": missing,
        "existing_three_nodes_unchanged": all(bool(row.get("prior_unchanged", True)) for row in existing_nodes),
        "all_passed": not missing and all(bool(row.get("passed")) for row in checks),
        "planes": checks,
    }
    print(write_json_atomic(arguments.output, payload), payload["digest"], payload["all_passed"])
    if not payload["all_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
