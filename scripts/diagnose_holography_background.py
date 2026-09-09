#!/usr/bin/env python3
"""Score fixed catalogue-source contamination from a frozen geometry snapshot.

No MS/network access. --dry-run validates inputs without opening beam planes
or writing outputs. See docs/holography-background-source-test.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from sl1mjax.catalog import read_radio_catalog
from sl1mjax.coordinates import radec_to_lmn
from sl1mjax.holography_background import (
    BackgroundGeometry,
    compare_background,
    predict_background,
    source_fluxes,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--catalogue", type=Path, required=True)
    parser.add_argument("--beam-root", type=Path)
    parser.add_argument("--beam-digest")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--default-spectral-index", type=float)
    parser.add_argument("--allow-unresolved-approximation", action="store_true")
    parser.add_argument("--row-chunk", type=int, default=2048)
    parser.add_argument("--bootstrap", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    required = {
        "schema_version",
        "snapshot_sha256",
        "catalogue_sha256",
        "phase_centre_rad",
        "source_names",
        "geometry_provenance",
        "calibration_provenance",
        "calibration_state",
        "uvw_convention",
        "beam_coordinate_frame",
        "split_provenance",
        "cluster_unit",
        "calibrator_exclusion_arcsec",
        "residual_jones_provenance",
    }
    if required - manifest.keys():
        raise ValueError(f"missing manifest fields: {sorted(required - manifest.keys())}")
    if manifest["schema_version"] != 1:
        raise ValueError("unsupported snapshot schema")
    if manifest["snapshot_sha256"] != sha256(args.snapshot):
        raise ValueError("snapshot checksum mismatch")
    if manifest["catalogue_sha256"] != sha256(args.catalogue):
        raise ValueError("catalogue checksum mismatch")
    if manifest["calibration_state"] != "casa_parang_true":
        raise ValueError("this first test requires sky-basis, parang-corrected data")
    if manifest["uvw_convention"] != "casa_positive_fringe_original_pq":
        raise ValueError("unknown UVW sign/order")
    if manifest["beam_coordinate_frame"] != "feed_source_direction_cosines":
        raise ValueError("unknown beam coordinate frame")
    if manifest["cluster_unit"] not in {"dwell", "scan", "visit"}:
        raise ValueError("individual rows/baselines are not independent bootstrap clusters")
    for key in (
        "geometry_provenance",
        "calibration_provenance",
        "split_provenance",
        "residual_jones_provenance",
    ):
        if not manifest[key]:
            raise ValueError(f"empty {key}")
    sources = read_radio_catalog(args.catalogue)
    if list(manifest["source_names"]) != [s.name for s in sources]:
        raise ValueError("catalogue order differs from geometry")
    if not args.allow_unresolved_approximation and any(
        (s.major_axis_arcsec or 0) > 0 for s in sources
    ):
        raise ValueError("catalogue sizes require --allow-unresolved-approximation")
    ra0, dec0 = manifest["phase_centre_rad"]
    lmns = np.array(
        [radec_to_lmn(ra0, dec0, np.deg2rad(s.ra_deg), np.deg2rad(s.dec_deg)) for s in sources],
        dtype=float,
    )
    separation = np.arccos(np.clip(lmns[:, 2], -1, 1)) * 180 / np.pi * 3600
    exclusion = float(manifest["calibrator_exclusion_arcsec"])
    if not np.isfinite(exclusion) or exclusion <= 0 or np.any(separation < exclusion):
        raise ValueError("exclude the calibrator and its catalogue components before prediction")
    with np.load(args.snapshot, allow_pickle=False) as handle:
        data = {key: handle[key] for key in handle.files}
    frequency = float(data["frequency_hz"])
    n = len(data["uvw_m"])
    geometry = BackgroundGeometry(
        data["uvw_m"],
        frequency,
        np.broadcast_to(lmns, (n, len(sources), 3)),
        data["beam_p_lmn"],
        data["beam_q_lmn"],
    )
    geometry.validate(len(sources))
    for key in ("residual_p", "residual_q"):
        if data[key].shape != (n, 2, 2) or not np.all(np.isfinite(data[key])):
            raise ValueError(f"{key} must be finite (row,2,2), in the calibrated sky basis")
    for key in ("measured", "calibrator", "weight", "flag"):
        if data[key].shape != (n, 2):
            raise ValueError(f"{key} must be (row,2) in RR/LL order")
    for key in ("score_mask", "cluster_id", "row_id"):
        if data[key].shape != (n,):
            raise ValueError(f"{key} must be (row,)")
    if len(np.unique(data["row_id"])) != n:
        raise ValueError("duplicate row identities")
    if data["flag"].dtype != bool or data["score_mask"].dtype != bool:
        raise ValueError("flag and score_mask must be boolean")
    flux = source_fluxes(sources, frequency, default_spectral_index=args.default_spectral_index)
    report = {
        "kind": "fixed_catalogue_background_diagnostic",
        "frequency_hz": frequency,
        "rows": n,
        "catalogue_sources": len(sources),
        "manifest": manifest,
        "default_spectral_index": args.default_spectral_index,
        "point_source_approximation": True,
        "time_bandwidth_smearing": "not_modelled",
        "catalogue_completeness": "not_established",
        "not_a_full_jones_test": True,
        "dry_run": args.dry_run,
        "source_flux_jy": dict(zip([s.name for s in sources], flux.tolist(), strict=True)),
    }
    if args.dry_run:
        print(json.dumps(report, indent=2, allow_nan=False))
        return
    if args.beam_root is None or args.beam_digest is None or args.output is None:
        parser.error("scoring requires --beam-root, --beam-digest, and --output")
    if args.output.exists():
        raise ValueError("output must be a new directory; existing products are never overwritten")
    from sl1mjax.evla_c_survey_beam import voltage_beam_for_survey_catalog

    beam = voltage_beam_for_survey_catalog(root=args.beam_root, digest=args.beam_digest)
    plane = beam.catalog.plane(frequency)

    def lookup(freq, l, m):
        if freq != frequency:
            raise ValueError("frequency mismatch")
        jones, valid = plane.lookup(l, m, off_diagonal=False)
        values = np.stack([jones[:, 0, 0], jones[:, 1, 1]], axis=-1)
        return values, np.broadcast_to(valid[:, None], values.shape)

    background, support, ranking = predict_background(
        geometry,
        flux,
        lookup,
        row_chunk=args.row_chunk,
        residual_p=data["residual_p"],
        residual_q=data["residual_q"],
    )
    for entry, source in zip(ranking, sources, strict=True):
        entry.update(name=source.name, catalog=source.catalog, reference_url=source.reference_url)
    report.update(
        beam_digest=args.beam_digest,
        ranking=ranking,
        comparison=compare_background(
            data["measured"],
            data["calibrator"],
            background,
            data["weight"],
            data["flag"],
            support,
            data["score_mask"],
            data["cluster_id"],
            n_boot=args.bootstrap,
            seed=args.seed,
        ),
    )
    args.output.mkdir(parents=True)
    np.savez_compressed(
        args.output / "predictions.npz",
        row_id=data["row_id"],
        background=background,
        support=support,
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    axes[0].scatter(lmns[:, 0] * 3437.75, lmns[:, 1] * 3437.75, s=20 + 40 * np.sqrt(flux))
    axes[0].scatter([0], [0], marker="*", color="red", label="calibrator")
    axes[0].set(
        xlabel="Phase-centre l (arcmin equivalent)",
        ylabel="Phase-centre m (arcmin equivalent)",
        title="Catalogue (point hypotheses)",
    )
    axes[0].legend()
    for i, hand in enumerate(("RR", "LL")):
        good = (
            data["score_mask"]
            & support[:, i]
            & ~data["flag"][:, i]
            & np.isfinite(data["weight"][:, i])
            & (data["weight"][:, i] > 0)
            & np.isfinite(data["measured"][:, i])
            & np.isfinite(data["calibrator"][:, i])
        )
        ids = np.flatnonzero(good)
        ids = ids[:: max(1, int(np.ceil(len(ids) / 4000)))]
        residual = data["measured"][ids, i] - data["calibrator"][ids, i]
        axes[i + 1].scatter(background[ids, i].real, residual.real, s=2, alpha=0.3, label="real")
        axes[i + 1].scatter(background[ids, i].imag, residual.imag, s=2, alpha=0.3, label="imag")
        axes[i + 1].set(
            xlabel="Predicted background (Jy)",
            ylabel="Calibrator residual (Jy)",
            title=f"{hand}: fixed prediction, no fitted scale",
        )
        axes[i + 1].legend()
    fig.savefig(args.output / "background_comparison.png", dpi=150)
    plt.close(fig)
    print(args.output / "report.json")


if __name__ == "__main__":
    main()
