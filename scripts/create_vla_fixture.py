"""Run with CASA to create a small known-truth VLA MeasurementSet."""

from __future__ import annotations

import json
import math
import os
import shutil
from pathlib import Path

from casatasks import simobserve
from casatools import componentlist

DESTINATION = Path(
    os.environ.get("SL1MJAX_CASA_FIXTURE", "outputs/casa_vla_fixture")
).resolve()
CASE = os.environ.get("SL1MJAX_CASA_CASE", "multi")
CONFIGURATION = os.environ.get("SL1MJAX_VLA_CONFIGURATION", "D").upper()
PROJECT = Path(DESTINATION.name)
COMPONENTS = PROJECT.with_suffix(".cl")
PHASE_CENTRE = (180.0, 45.0)
if CONFIGURATION not in {"A", "B", "C", "D"}:
    raise ValueError(f"unknown VLA configuration {CONFIGURATION!r}")

east_ten_arcsec_ra = 10 / (3600 * math.cos(math.radians(PHASE_CENTRE[1])))
cases = {
    "center": ((1.0, PHASE_CENTRE[0], PHASE_CENTRE[1]),),
    "east": ((1.0, PHASE_CENTRE[0] + east_ten_arcsec_ra, PHASE_CENTRE[1]),),
    "north": ((1.0, PHASE_CENTRE[0], PHASE_CENTRE[1] + 10 / 3600),),
    "diagonal": (
        (
            1.0,
            PHASE_CENTRE[0] + 7 / (3600 * math.cos(math.radians(PHASE_CENTRE[1]))),
            PHASE_CENTRE[1] + 13 / 3600,
        ),
    ),
    "multi": (
        (1.0, PHASE_CENTRE[0], PHASE_CENTRE[1]),
        (0.5, PHASE_CENTRE[0] + 6 / 3600, PHASE_CENTRE[1] + 20 / 3600),
        (0.25, PHASE_CENTRE[0] - 6 / 3600, PHASE_CENTRE[1] - 20 / 3600),
    ),
    "gaussian": (
        (
            1.0,
            PHASE_CENTRE[0]
            + 30 / (3600 * math.cos(math.radians(PHASE_CENTRE[1]))),
            PHASE_CENTRE[1] + 15 / 3600,
        ),
    ),
}
if CASE not in cases:
    raise ValueError(f"unknown SL1MJAX_CASA_CASE {CASE!r}")
sources = cases[CASE]
source_shape = (
    {
        "kind": "circular_gaussian",
        "major_axis": "20arcsec",
        "minor_axis": "20arcsec",
        "position_angle": "0deg",
        "fwhm_major_rad": math.radians(20 / 3600),
        "fwhm_minor_rad": math.radians(20 / 3600),
        "position_angle_rad": 0.0,
    }
    if CASE == "gaussian"
    else {"kind": "point"}
)

DESTINATION.parent.mkdir(parents=True, exist_ok=True)
if DESTINATION.exists():
    shutil.rmtree(DESTINATION)
absolute_components = DESTINATION.with_suffix(".cl")
if absolute_components.exists():
    shutil.rmtree(absolute_components)
os.chdir(DESTINATION.parent)

components = componentlist()
truth_sources = []
ra0, dec0 = map(math.radians, PHASE_CENTRE)
for flux, ra_deg, dec_deg in sources:
    ra, dec = math.radians(ra_deg), math.radians(dec_deg)
    delta_ra = ra - ra0
    l = math.cos(dec) * math.sin(delta_ra)
    m = (
        math.sin(dec) * math.cos(dec0)
        - math.cos(dec) * math.sin(dec0) * math.cos(delta_ra)
    )
    n = (
        math.sin(dec) * math.sin(dec0)
        + math.cos(dec) * math.cos(dec0) * math.cos(delta_ra)
    )
    component_shape = (
        {
            "shape": "Gaussian",
            "majoraxis": source_shape["major_axis"],
            "minoraxis": source_shape["minor_axis"],
            "positionangle": source_shape["position_angle"],
        }
        if source_shape["kind"] == "circular_gaussian"
        else {"shape": "point"}
    )
    components.addcomponent(
        flux=flux,
        fluxunit="Jy",
        dir=f"J2000 {ra_deg}deg {dec_deg}deg",
        spectrumtype="constant",
        freq="1.4GHz",
        **component_shape,
    )
    truth_sources.append(
        {
            "flux_jy": flux,
            "ra_rad": ra,
            "dec_rad": dec,
            "l": l,
            "m": m,
            "n": n,
            "shape": source_shape,
        }
    )
# Tiny symmetric anchors force simobserve's component-list image and MS phase
# centre to remain at PHASE_CENTRE for one-sided source cases.
anchor_offset = 120 / 3600
for ra_deg, dec_deg in (
    (
        PHASE_CENTRE[0]
        + anchor_offset / math.cos(math.radians(PHASE_CENTRE[1])),
        PHASE_CENTRE[1],
    ),
    (
        PHASE_CENTRE[0]
        - anchor_offset / math.cos(math.radians(PHASE_CENTRE[1])),
        PHASE_CENTRE[1],
    ),
    (PHASE_CENTRE[0], PHASE_CENTRE[1] + anchor_offset),
    (PHASE_CENTRE[0], PHASE_CENTRE[1] - anchor_offset),
):
    components.addcomponent(
        flux=1e-12,
        fluxunit="Jy",
        dir=f"J2000 {ra_deg}deg {dec_deg}deg",
        shape="point",
        spectrumtype="constant",
        freq="1.4GHz",
    )
components.rename(str(COMPONENTS))
components.close()

simobserve(
    project=str(PROJECT),
    complist=str(COMPONENTS),
    compwidth="1MHz",
    comp_nchan=2,
    setpointings=True,
    integration="10s",
    direction=f"J2000 {PHASE_CENTRE[0]}deg {PHASE_CENTRE[1]}deg",
    mapsize="4arcmin",
    obsmode="int",
    antennalist=f"vla.{CONFIGURATION.lower()}.cfg",
    hourangle="transit",
    totaltime="300s",
    thermalnoise="",
    graphics="none",
    overwrite=True,
    verbose=False,
)

DESTINATION.with_suffix(".truth.json").write_text(
    json.dumps(
        {
            "case": CASE,
            "phase_centre_ra_rad": ra0,
            "phase_centre_dec_rad": dec0,
            "sources": truth_sources,
            "array_configuration": f"vla.{CONFIGURATION.lower()}.cfg",
            "channels": 2,
            "component_width": "1MHz",
            "thermal_noise": False,
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
print(DESTINATION / f"{PROJECT.name}.vla.{CONFIGURATION.lower()}.ms")
