#!/usr/bin/env python3
"""Build a pinned, beam-selected NVSS/VLASS guard catalogue for 3C391."""

from __future__ import annotations

import argparse
import csv
import io
import json
from dataclasses import asdict, replace
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord

from sl1mjax.beam import VLABeamCatalog, VLAPrimaryBeam
from sl1mjax.catalog import (
    RadioCatalogSource,
    select_catalog_guard_atoms,
    write_radio_catalog,
)
from sl1mjax.data.canonical import read_dataset

NVSS_TABLE = "VIII/65/nvss"
NVSS_REFERENCE_URL = "https://cdsarc.cds.unistra.fr/viz-bin/cat/VIII/65"
VLASS_TABLE = "J/ApJS/255/30/comp"
VLASS_REFERENCE_URL = "https://cdsarc.cds.unistra.fr/viz-bin/cat/J/ApJS/255/30"
VIZIER_TSV_URL = "https://vizier.cds.unistra.fr/viz-bin/asu-tsv"


def _optional_float(row: dict[str, str], name: str) -> float | None:
    value = row[name].strip()
    return None if not value else float(value)


def _query_url(
    *,
    ra_deg: float,
    dec_deg: float,
    radius_deg: float,
    minimum_flux_mjy: float,
) -> str:
    parameters = {
        "-source": NVSS_TABLE,
        "-c": f"{ra_deg:.10f} {dec_deg:.10f}",
        "-c.r": f"{radius_deg:g}",
        "-c.u": "deg",
        "-out.max": "unlimited",
        "S1.4": f">={minimum_flux_mjy:g}",
        "-out": "NVSS,RAJ2000,DEJ2000,S1.4,e_S1.4,MajAxis,MinAxis,PA",
    }
    return f"{VIZIER_TSV_URL}?{urlencode(parameters)}"


def _vlass_query_url(*, ra_deg: float, dec_deg: float, radius_arcsec: float) -> str:
    parameters = {
        "-source": VLASS_TABLE,
        "-c": f"{ra_deg:.10f} {dec_deg:.10f}",
        "-c.r": f"{radius_arcsec / 3600.0:.10g}",
        "-c.u": "deg",
        "-out.max": "unlimited",
        "-out": (
            "CompName,RAJ2000,DEJ2000,Ftot,Fpeak,SCode,DCMaj,DCMin,DCPA,"
            "NVSSdist,DupFlag,QualFlag,MainSample"
        ),
    }
    return f"{VIZIER_TSV_URL}?{urlencode(parameters)}"


def parse_nvss_tsv(text: str) -> tuple[RadioCatalogSource, ...]:
    """Parse the explicit column set requested by this script."""

    lines = [line for line in text.splitlines() if line and not line.startswith("#")]
    try:
        header_index = next(index for index, line in enumerate(lines) if line.startswith("NVSS\t"))
    except StopIteration as error:
        raise ValueError("VizieR response contains no NVSS table") from error
    data_lines = lines[header_index + 3 :]
    reader = csv.DictReader(
        io.StringIO("\n".join([lines[header_index], *data_lines])), delimiter="\t"
    )
    sources: list[RadioCatalogSource] = []
    for row in reader:
        coordinate = SkyCoord(
            row["RAJ2000"].strip(),
            row["DEJ2000"].strip(),
            unit=(u.hourangle, u.deg),
            frame="icrs",
        )

        sources.append(
            RadioCatalogSource(
                name=f"NVSS_{row['NVSS'].strip()}",
                ra_deg=float(coordinate.ra.deg),
                dec_deg=float(coordinate.dec.deg),
                reference_frequency_hz=1.4e9,
                integrated_flux_jy=float(row["S1.4"]) / 1_000.0,
                major_axis_arcsec=_optional_float(row, "MajAxis"),
                minor_axis_arcsec=_optional_float(row, "MinAxis"),
                position_angle_deg=_optional_float(row, "PA"),
                catalog="NVSS VIII/65",
                epoch="J2000 catalogue position; survey 1993-1996",
                reference_url=NVSS_REFERENCE_URL,
            )
        )
    if not sources:
        raise ValueError("VizieR response contains no NVSS sources")
    return tuple(sources)


def parse_vlass_tsv(text: str) -> tuple[RadioCatalogSource, ...]:
    """Parse reliable, non-duplicate VLASS main-sample components."""

    lines = [line for line in text.splitlines() if line and not line.startswith("#")]
    try:
        header_index = next(
            index for index, line in enumerate(lines) if line.startswith("CompName\t")
        )
    except StopIteration:
        return ()
    data_lines = lines[header_index + 1 :]
    reader = csv.DictReader(
        io.StringIO("\n".join([lines[header_index], *data_lines])), delimiter="\t"
    )
    sources: list[RadioCatalogSource] = []
    for row in reader:
        try:
            duplicate_flag = int(row["DupFlag"])
            quality_flag = int(row["QualFlag"])
            main_sample = int(row["MainSample"])
            total_flux_mjy = float(row["Ftot"])
        except TypeError, ValueError:
            # VizieR inserts units and separator rows below the header.
            continue
        if duplicate_flag != 0 or quality_flag != 0 or main_sample != 1 or total_flux_mjy <= 0:
            continue
        sources.append(
            RadioCatalogSource(
                name=row["CompName"].strip().replace("VLASS1QLCIR ", "VLASS_"),
                ra_deg=float(row["RAJ2000"]),
                dec_deg=float(row["DEJ2000"]),
                reference_frequency_hz=3.0e9,
                integrated_flux_jy=total_flux_mjy / 1_000.0,
                peak_flux_jy=float(row["Fpeak"]) / 1_000.0,
                major_axis_arcsec=_optional_float(row, "DCMaj"),
                minor_axis_arcsec=_optional_float(row, "DCMin"),
                position_angle_deg=_optional_float(row, "DCPA"),
                catalog="VLASS QL Epoch 1 J/ApJS/255/30",
                epoch="VLASS Epoch 1 Quick Look",
                reference_url=VLASS_REFERENCE_URL,
            )
        )
    return tuple(sources)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "fixture",
        type=Path,
        nargs="?",
        default=Path("outputs/3c391_mosaic_imaging_fixture.zarr"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("config/3c391_radio_guard_catalog.csv"),
    )
    parser.add_argument("--query-radius-deg", type=float, default=1.0)
    parser.add_argument("--minimum-catalog-flux-mjy", type=float, default=50.0)
    parser.add_argument("--minimum-apparent-flux-mjy", type=float, default=0.5)
    parser.add_argument("--maximum-major-axis-arcsec", type=float, default=45.0)
    parser.add_argument("--vlass-crossmatch-radius-arcsec", type=float, default=20.0)
    parser.add_argument("--central-root-size", type=int, default=104)
    parser.add_argument("--central-pixel-arcsec", type=float, default=16.0)
    parser.add_argument("--default-spectral-index", type=float, default=-0.7)
    parser.add_argument("--airy-max-radius-deg-at-1ghz", type=float, default=4.0)
    arguments = parser.parse_args()

    blocks = read_dataset(arguments.fixture).blocks
    mosaic_phase_centre = blocks[0].phase_centre_rad
    centre_deg = np.rad2deg(mosaic_phase_centre)
    query_url = _query_url(
        ra_deg=float(centre_deg[0]),
        dec_deg=float(centre_deg[1]),
        radius_deg=arguments.query_radius_deg,
        minimum_flux_mjy=arguments.minimum_catalog_flux_mjy,
    )
    with urlopen(query_url, timeout=120) as response:  # noqa: S310
        text = response.read().decode("utf-8")
    queried = parse_nvss_tsv(text)
    beam = VLAPrimaryBeam(
        kind="airy",
        catalog=replace(
            VLABeamCatalog(),
            airy_max_radius_rad_at_1ghz=np.deg2rad(arguments.airy_max_radius_deg_at_1ghz),
        ),
    )
    half_width = (
        arguments.central_root_size * np.deg2rad(arguments.central_pixel_arcsec / 3600.0) / 2.0
    )
    nvss_atoms = select_catalog_guard_atoms(
        queried,
        blocks,
        mosaic_phase_centre,
        primary_beam=beam,
        central_half_width_rad=(half_width, half_width),
        minimum_apparent_flux_jy=arguments.minimum_apparent_flux_mjy / 1_000.0,
        default_spectral_index=arguments.default_spectral_index,
        maximum_major_axis_arcsec=arguments.maximum_major_axis_arcsec,
    )
    vlass_sources: list[RadioCatalogSource] = []
    vlass_query_urls: list[str] = []
    for atom in nvss_atoms:
        vlass_query_url = _vlass_query_url(
            ra_deg=atom.source.ra_deg,
            dec_deg=atom.source.dec_deg,
            radius_arcsec=arguments.vlass_crossmatch_radius_arcsec,
        )
        vlass_query_urls.append(vlass_query_url)
        with urlopen(vlass_query_url, timeout=120) as response:  # noqa: S310
            vlass_sources.extend(parse_vlass_tsv(response.read().decode("utf-8")))
    atoms = (
        select_catalog_guard_atoms(
            tuple(vlass_sources),
            blocks,
            mosaic_phase_centre,
            primary_beam=beam,
            central_half_width_rad=(half_width, half_width),
            minimum_apparent_flux_jy=arguments.minimum_apparent_flux_mjy / 1_000.0,
            default_spectral_index=arguments.default_spectral_index,
            maximum_major_axis_arcsec=arguments.maximum_major_axis_arcsec,
        )
        if vlass_sources
        else nvss_atoms
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    write_radio_catalog(arguments.output, tuple(atom.source for atom in atoms))
    manifest = {
        "schema_version": 1,
        "query_url": query_url,
        "query_result_count": len(queried),
        "nvss_selected_count": len(nvss_atoms),
        "vlass_crossmatch_query_urls": vlass_query_urls,
        "vlass_reliable_component_count": len(vlass_sources),
        "selected_count": len(atoms),
        "selection": {
            "central_half_width_arcmin": float(np.rad2deg(half_width) * 60.0),
            "minimum_apparent_flux_mjy": arguments.minimum_apparent_flux_mjy,
            "maximum_major_axis_arcsec": arguments.maximum_major_axis_arcsec,
            "vlass_crossmatch_radius_arcsec": arguments.vlass_crossmatch_radius_arcsec,
            "default_spectral_index": arguments.default_spectral_index,
            "beam": "extended_airy",
            "airy_max_radius_deg_at_1ghz": arguments.airy_max_radius_deg_at_1ghz,
        },
        "atoms": [asdict(atom) for atom in atoms],
    }
    arguments.output.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
