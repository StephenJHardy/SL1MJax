"""Run with CASA to create a small known-truth VLA MeasurementSet."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from casatasks import simobserve
from casatools import componentlist

DESTINATION = Path(
    os.environ.get("SL1MJAX_CASA_FIXTURE", "outputs/casa_vla_fixture")
).resolve()
PROJECT = Path(DESTINATION.name)
COMPONENTS = PROJECT.with_suffix(".cl")

if DESTINATION.exists():
    shutil.rmtree(DESTINATION)
absolute_components = DESTINATION.with_suffix(".cl")
if absolute_components.exists():
    shutil.rmtree(absolute_components)
os.chdir(DESTINATION.parent)

components = componentlist()
for flux, direction in (
    (1.0, "J2000 12h00m00.0s +45d00m00.0s"),
    (0.5, "J2000 12h00m00.4s +45d00m20.0s"),
    (0.25, "J2000 11h59m59.6s +44d59m40.0s"),
):
    components.addcomponent(
        flux=flux,
        fluxunit="Jy",
        dir=direction,
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
    direction="J2000 12h00m00.0s +45d00m00.0s",
    mapsize="4arcmin",
    obsmode="int",
    antennalist="vla.d.cfg",
    hourangle="transit",
    totaltime="300s",
    thermalnoise="",
    graphics="none",
    overwrite=True,
    verbose=False,
)

print(DESTINATION / f"{PROJECT.name}.vla.d.ms")
