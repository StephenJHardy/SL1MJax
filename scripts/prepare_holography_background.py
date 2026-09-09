#!/usr/bin/env python3
"""Prepare a bounded, read-only THOL0001 SPW-4 background diagnostic.

Commands: catalogue (public NVSS query) and snapshot (Bacchus casacore).
All output directories must be new. No calibration or beam artifacts are written.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np

from sl1mjax.catalog import RadioCatalogSource, read_radio_catalog, write_radio_catalog
from sl1mjax.coordinates import radec_to_lmn


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def repoint_sin(source_lmn, commanded):
    """Rotate the source into a pointing frame using the minimal SIN rotation.

    Mount +l/+m are fixed to the survey axes. This reproduces the locked
    central-source query (-dx,-dy) exactly, while retaining spherical geometry.
    The interpretation of the native offsets as SIN is recorded as a hypothesis.
    """
    d = np.asarray(commanded, float)
    if d.shape != (2,) or np.dot(d, d) >= 1:
        raise ValueError("invalid commanded offset")
    n = np.sqrt(1 - np.dot(d, d))
    # Rodrigues rotation z -> (dx,dy,n); its columns are the new basis.
    k = np.array([[0.0, 0.0, d[0]], [0.0, 0.0, d[1]], [-d[0], -d[1], 0.0]])
    rotation = np.eye(3) + k + k @ k / (1 + n)
    return np.asarray(source_lmn) @ rotation


def catalogue(args):
    import astropy.units as u
    from astropy.coordinates import SkyCoord

    params = {
        "-source": "VIII/65/nvss",
        "-c": f"{args.ra_deg} {args.dec_deg}",
        "-c.r": str(args.radius_deg),
        "-c.u": "deg",
        "-out.max": "unlimited",
        "S1.4": f">={args.min_flux_mjy}",
        "-out": "NVSS,RAJ2000,DEJ2000,S1.4,e_S1.4,MajAxis,MinAxis,PA",
    }
    url = "https://vizier.cds.unistra.fr/viz-bin/asu-tsv?" + urlencode(params)
    with urlopen(url, timeout=60) as response:  # noqa: S310
        raw = response.read().decode()
    lines = [line for line in raw.splitlines() if line and not line.startswith("#")]
    start = next(i for i, line in enumerate(lines) if line.startswith("NVSS\t"))
    rows = csv.DictReader(io.StringIO("\n".join(lines[start:])), delimiter="\t")
    selected, excluded = [], []
    centre = SkyCoord(args.ra_deg * u.deg, args.dec_deg * u.deg)
    for row in rows:
        try:
            flux = float(row["S1.4"]) / 1000
        except ValueError, TypeError:
            continue
        position = SkyCoord(row["RAJ2000"], row["DEJ2000"], unit=(u.hourangle, u.deg))
        separation = centre.separation(position).arcsec
        if separation < args.exclusion_arcsec:
            excluded.append(dict(row))
            continue

        def optional(key, row=row):
            try:
                return float(row[key])
            except ValueError, TypeError:
                return None

        selected.append(
            RadioCatalogSource(
                name="NVSS_" + row["NVSS"].strip(),
                ra_deg=position.ra.deg,
                dec_deg=position.dec.deg,
                reference_frequency_hz=1.4e9,
                integrated_flux_jy=flux,
                catalog="NVSS VIII/65",
                reference_url="https://cdsarc.cds.unistra.fr/viz-bin/cat/VIII/65",
                major_axis_arcsec=optional("MajAxis"),
                minor_axis_arcsec=optional("MinAxis"),
                position_angle_deg=optional("PA"),
                epoch="NVSS 1993-1996",
            )
        )
    if not selected:
        raise ValueError("empty background catalogue")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "nvss_response.tsv").write_text(raw)
    write_radio_catalog(args.output / "background.csv", tuple(selected))
    write_json(
        args.output / "catalogue_query.json",
        dict(
            url=url,
            centre_deg=[args.ra_deg, args.dec_deg],
            radius_deg=args.radius_deg,
            min_flux_mjy=args.min_flux_mjy,
            exclusion_arcsec=args.exclusion_arcsec,
            n_sources=len(selected),
            excluded_calibrator=excluded,
            total_1p4ghz_flux_jy=sum(s.integrated_flux_jy for s in selected),
            csv_sha256=digest(args.output / "background.csv"),
            source_selection="catalogue only, before visibility inspection",
            alpha_scenarios=[-1.0, -0.7, 0.0],
            unresolved_approximation=True,
        ),
    )
    print(args.output, len(selected), flush=True)


def snapshot(args):
    from casacore.measures import measures
    from casacore.quanta import quantity
    from casacore.tables import table

    from sl1mjax.evla_c_survey_beam import survey_catalog_digest, voltage_beam_for_survey_catalog
    from sl1mjax.evla_c_survey_compare import load_comparison_module, load_holoraster_slot

    if args.output.exists():
        raise ValueError("output must be new")
    sources = read_radio_catalog(args.catalogue)
    cmp = load_comparison_module(args.scripts_dir)
    # The scientific SPW4/5 reader has no survey row cache; source is read-only.
    print("extracting opened SPW4 channel32 development rows", flush=True)
    extract = load_holoraster_slot(
        cmp, measurement_set=args.measurement_set, spectral_window_id=4, channel=32
    )
    obs = extract["observation"]
    block = obs.block
    all_rows = np.asarray(extract["fields"]["rows"])
    times = block.time_s[all_rows]
    unique = np.unique(times)
    chosen = unique[np.linspace(0, len(unique) - 1, args.n_times, dtype=int)]
    ix = np.flatnonzero(np.isin(times, chosen))
    rows = all_rows[ix]
    print("snapshot rows", len(rows), "times", len(chosen), "sources", len(sources), flush=True)
    p, q = block.antenna1[rows], block.antenna2[rows]
    t = block.time_s[rows]
    offsets, inverse, _, _, _ = obs.pointing_state()
    phase = block.phase_centre_rad
    direction_list = [phase] + [(np.deg2rad(s.ra_deg), np.deg2rad(s.dec_deg)) for s in sources]
    shape = (len(rows), len(sources) + 1, 3)
    bp, bq = np.zeros(shape), np.zeros(shape)
    # An antenna-time lookup is evaluated once, then reused across baselines.
    for ti, timestamp in enumerate(chosen):
        where = np.flatnonzero(t == timestamp)
        for ant in np.union1d(p[where], q[where]):
            frame = measures()
            xyz = extract["positions"][ant]
            frame.doframe(frame.position("ITRF", *[quantity(float(x), "m") for x in xyz]))
            frame.doframe(frame.epoch("UTC", quantity(float(timestamp) / 86400, "d")))
            azel = []
            for ra, dec in direction_list:
                az = frame.measure(
                    frame.direction(
                        "J2000", quantity(float(ra), "rad"), quantity(float(dec), "rad")
                    ),
                    "AZELGEO",
                )
                azel.append((az["m0"]["value"], az["m1"]["value"]))
            azel = np.asarray(azel)
            horizon = np.stack(
                radec_to_lmn(azel[0, 0], azel[0, 1], azel[:, 0], azel[:, 1]), axis=-1
            )
            state_time = inverse[rows[where[0]]]
            query = repoint_sin(horizon, offsets[state_time, ant])
            np.testing.assert_allclose(query[0, :2], -offsets[state_time, ant], atol=1e-12)
            bp[where[p[where] == ant]] = query
            bq[where[q[where] == ant]] = query
        print("geometry", ti + 1, "/", len(chosen), flush=True)
    beam_digest = survey_catalog_digest(args.beam_root)
    plane = voltage_beam_for_survey_catalog(root=args.beam_root, digest=beam_digest).catalog.plane(
        float(extract["frequency_hz"])
    )
    ep, vp = plane.lookup(bp[:, 0, 0], bp[:, 0, 1], off_diagonal=False)
    eq, vq = plane.lookup(bq[:, 0, 0], bq[:, 0, 1], off_diagonal=False)
    src = np.asarray(extract["source"])
    if src.ndim == 4:
        src = src[:, 0]
    model = ep @ src[ix] @ np.swapaxes(eq.conj(), -1, -2)
    # Close against the existing serialized survey prediction, not a new fit.
    with np.load(args.survey_export) as frozen:
        np.testing.assert_allclose(frozen["measured"][ix], extract["measured"][ix], rtol=0, atol=0)
        comparison = frozen["predicted"][ix]
        closure = float(np.max(np.abs(model[:, [0, 1], [0, 1]] - comparison[:, [0, 1], [0, 1]])))
        if closure > 1e-6:
            raise ValueError(f"central-source frozen visibility closure failed: {closure}")
    # Recover original row ids, flags and exposure, avoiding reader omissions.
    with table(str(args.measurement_set), ack=False, readonly=True) as main:
        selected = main.query("FIELD_ID==10 && DATA_DESC_ID==4")
        with selected.selectrows(rows) as small:
            row_ids = small.rownumbers(main)
            flag = small.getcolslice("FLAG", [32, 0], [32, 3])[:, 0][:, [0, 3]]
            flag |= small.getcol("FLAG_ROW")[:, None]
            exposure = small.getcol("EXPOSURE")
        selected.close()
    weight = extract["weight"][ix][:, [0, 1], [0, 1]]
    flag |= ~vp[:, None] | ~vq[:, None]
    args.output.mkdir(parents=True, exist_ok=False)
    dest = args.output / "snapshot.npz"
    np.savez_compressed(
        dest,
        uvw_m=block.uvw_m[rows],
        frequency_hz=extract["frequency_hz"],
        beam_p_lmn=bp[:, 1:],
        beam_q_lmn=bq[:, 1:],
        residual_p=np.broadcast_to(np.eye(2), (len(rows), 2, 2)),
        residual_q=np.broadcast_to(np.eye(2), (len(rows), 2, 2)),
        measured=extract["measured"][ix][:, [0, 1], [0, 1]],
        calibrator=model[:, [0, 1], [0, 1]],
        weight=weight,
        flag=flag,
        row_id=row_ids,
        score_mask=np.ones(len(rows), bool),
        cluster_id=block.scan_id[rows],
        time_s=t,
        antenna1=p,
        antenna2=q,
        exposure_s=exposure,
        source_model=src[ix],
        calibrator_lmn_p=bp[:, 0],
        calibrator_lmn_q=bq[:, 0],
    )
    write_json(
        args.output / "snapshot.json",
        dict(
            schema_version=1,
            snapshot_sha256=digest(dest),
            catalogue_sha256=digest(args.catalogue),
            phase_centre_rad=list(phase),
            source_names=[s.name for s in sources],
            geometry_provenance={
                "method": "casacore J2000->AZELGEO per antenna/time + minimal SIN repoint",
                "source": "THOL0001 native offset tangent contract; angle/SIN uncertainty remains",
                "central_frozen_closure_max_jy": closure,
                "n_times": len(chosen),
            },
            calibration_provenance={
                "ms": str(args.measurement_set),
                "column": "CORRECTED_DATA",
                "survey_export": str(args.survey_export),
                "survey_export_sha256": digest(args.survey_export),
            },
            residual_jones_provenance="explicit identity, matching survey; no residual-Jones fit",
            calibration_state="casa_parang_true",
            uvw_convention="casa_positive_fringe_original_pq",
            beam_coordinate_frame="feed_source_direction_cosines",
            split_provenance="32 evenly spaced opened development times, no residual-based selection",
            cluster_unit="scan",
            calibrator_exclusion_arcsec=120,
            beam_digest=beam_digest,
            max_exposure_s=float(np.max(exposure)),
            native_channel_width_hz=2e6,
            source_is_unresolved_hypothesis=True,
        ),
    )
    print(args.output, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    cat = sub.add_parser("catalogue")
    cat.add_argument("--ra-deg", type=float, required=True)
    cat.add_argument("--dec-deg", type=float, required=True)
    cat.add_argument("--radius-deg", type=float, default=1.0)
    cat.add_argument("--min-flux-mjy", type=float, default=10.0)
    cat.add_argument("--exclusion-arcsec", type=float, default=120.0)
    cat.add_argument("--output", type=Path, required=True)
    snap = sub.add_parser("snapshot")
    for name in (
        "measurement-set",
        "scripts-dir",
        "beam-root",
        "catalogue",
        "survey-export",
        "output",
    ):
        snap.add_argument("--" + name, type=Path, required=True)
    snap.add_argument("--n-times", type=int, default=32)
    args = parser.parse_args()
    if args.command == "catalogue":
        catalogue(args)
    else:
        snapshot(args)


if __name__ == "__main__":
    main()
