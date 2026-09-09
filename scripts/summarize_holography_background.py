#!/usr/bin/env python3
"""Summarize a completed three-scenario background diagnostic, without new fits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def summarize(root: Path) -> dict:
    manifest = json.loads((root / "snapshot/snapshot.json").read_text())
    snapshot = root / "snapshot/snapshot.npz"
    if hashlib.sha256(snapshot.read_bytes()).hexdigest() != manifest["snapshot_sha256"]:
        raise ValueError("snapshot checksum mismatch")
    with np.load(snapshot) as f:
        data = {k: f[k] for k in f.files}
    radius = (
        np.maximum(
            np.linalg.norm(data["calibrator_lmn_p"][:, :2], axis=1),
            np.linalg.norm(data["calibrator_lmn_q"][:, :2], axis=1),
        )
        * 10800
        / np.pi
    )
    result = dict(
        frequency_hz=float(data["frequency_hz"]),
        n_snapshot_rows=len(radius),
        snapshot_radius_range_arcmin=[float(radius.min()), float(radius.max())],
        central_closure_max_jy=manifest["geometry_provenance"]["central_frozen_closure_max_jy"],
        scenarios={},
        production_accepted=False,
        limitations=[
            "catalogue-limited unresolved-source hypothesis",
            "native central-frequency, no exposure/channel integration",
            "incomplete support at wide raster offsets",
            "one opened channel, not a fresh scientific holdout",
            "beam outer response and source spectra remain uncertain",
        ],
    )
    for tag in ("alpha_m10", "alpha_m07", "alpha_0"):
        report = json.loads((root / tag / "report.json").read_text())
        with np.load(root / tag / "predictions.npz") as p:
            np.testing.assert_array_equal(p["row_id"], data["row_id"])
            b, support = p["background"], p["support"]
        hands = {}
        for i, hand in enumerate(("RR", "LL")):
            y, c, w = (data[k][:, i] for k in ("measured", "calibrator", "weight"))
            ok = (
                data["score_mask"]
                & support[:, i]
                & ~data["flag"][:, i]
                & np.isfinite(y)
                & np.isfinite(c)
                & np.isfinite(b[:, i])
                & np.isfinite(w)
                & (w > 0)
            )
            if not ok.any():
                hands[hand] = {"status": "no_supported_rows"}
                continue

            def rms(x, ok=ok, w=w):
                return float(np.sqrt(np.average(abs(x[ok]) ** 2, weights=w[ok])))

            h = dict(report["comparison"]["hands"][hand])
            h.update(
                background_rms_jy=rms(b[:, i]),
                residual_rms_jy=rms(y - c),
                max_background_jy=float(np.max(abs(b[ok, i]))),
                median_residual_jy=float(np.median(abs(y[ok] - c[ok]))),
                supported_radius_range_arcmin=[float(radius[ok].min()), float(radius[ok].max())],
                sum_source_peak_bound_jy=float(
                    sum(s["peak_apparent_jy_rr_ll"][i] for s in report["ranking"])
                ),
            )
            hands[hand] = h
        result["scenarios"][tag] = {
            "spectral_index": report["default_spectral_index"],
            "hands": hands,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing to overwrite summary")
    args.output.write_text(json.dumps(summarize(args.root), indent=2, allow_nan=False) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
