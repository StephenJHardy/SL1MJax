"""Run with CASA to create staged 3C391 calibration reference products."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from casaplotms import plotms
from casatasks import (
    applycal,
    bandpass,
    clearstat,
    flagdata,
    flagmanager,
    fluxscale,
    gaincal,
    gencal,
    listobs,
    plotants,
    setjy,
)

MEASUREMENT_SET = Path(
    os.environ.get(
        "SL1MJAX_3C391_MS",
        Path(__file__).resolve().parents[1]
        / "data/3c391_work_v2/3c391_ctm_mosaic_10s_spw0.ms",
    )
).resolve()
OUTPUT = Path(
    os.environ.get(
        "SL1MJAX_3C391_REFERENCE",
        Path(__file__).resolve().parents[1] / "data/3c391/reference-v2",
    )
).resolve()

FLUX_CALIBRATOR = "J1331+3030"
GAIN_CALIBRATOR = "J1822-0938"
CALIBRATOR_FIELDS = "0,1,9"
SCIENCE_FIELDS = "2~8"
REFERENCE_ANTENNA = "ea21"


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    return str(value)


def _table(name: str) -> str:
    return str(OUTPUT / f"3c391.{name}")


def _plot(name: str) -> str:
    return str(OUTPUT / "plots" / f"{name}.png")


if not MEASUREMENT_SET.is_dir():
    raise FileNotFoundError(MEASUREMENT_SET)
OUTPUT.mkdir(parents=True, exist_ok=True)
(OUTPUT / "plots").mkdir(exist_ok=True)

listobs(
    vis=str(MEASUREMENT_SET),
    listfile=str(OUTPUT / "listobs.txt"),
    overwrite=True,
)
plotants(
    vis=str(MEASUREMENT_SET),
    figfile=_plot("antenna-layout"),
)
clearstat()

flag_before = flagdata(
    vis=str(MEASUREMENT_SET),
    mode="summary",
    action="calculate",
)
flagmanager(
    vis=str(MEASUREMENT_SET),
    mode="save",
    versionname="sl1mjax_pristine",
    comment="Before reproducible CASA tutorial flags",
)

# Deterministic edits from the CASA 6.7.2 3C391 continuum tutorial.
flagdata(
    vis=str(MEASUREMENT_SET),
    mode="manual",
    scan="1",
    flagbackup=False,
)
flagdata(
    vis=str(MEASUREMENT_SET),
    mode="manual",
    antenna="ea13,ea15",
    flagbackup=False,
)
flagdata(
    vis=str(MEASUREMENT_SET),
    mode="quack",
    quackinterval=10.0,
    quackmode="beg",
    flagbackup=False,
)

antpos = _table("antpos")
g0_all = _table("G0all")
g0 = _table("G0")
delay = _table("K0")
bandpass_table = _table("B0")
gain = _table("G1")
flux_gain = _table("fluxscale1")

gencal(
    vis=str(MEASUREMENT_SET),
    caltable=antpos,
    caltype="antpos",
)
setjy(
    vis=str(MEASUREMENT_SET),
    field=FLUX_CALIBRATOR,
    standard="Perley-Butler 2017",
    model="3C286_C.im",
    usescratch=True,
    scalebychan=True,
    spw="",
)

# Diagnostic solve across all calibrators, used by the tutorial to identify ea05.
gaincal(
    vis=str(MEASUREMENT_SET),
    caltable=g0_all,
    field=CALIBRATOR_FIELDS,
    refant=REFERENCE_ANTENNA,
    spw="0:27~36",
    gaintype="G",
    calmode="p",
    solint="int",
    minsnr=5,
    gaintable=[antpos],
)
plotms(
    vis=g0_all,
    xaxis="time",
    yaxis="phase",
    coloraxis="corr",
    iteraxis="antenna",
    plotrange=[-1, -1, -180, 180],
    plotfile=_plot("G0all-phase"),
    overwrite=True,
)
flagdata(
    vis=str(MEASUREMENT_SET),
    mode="manual",
    antenna="ea05",
    flagbackup=False,
)
flag_calibration_input = flagdata(
    vis=str(MEASUREMENT_SET),
    mode="summary",
    action="calculate",
)
flagmanager(
    vis=str(MEASUREMENT_SET),
    mode="save",
    versionname="sl1mjax_calibration_input",
    comment="Tutorial edits used as calibration-solver input",
)

gaincal(
    vis=str(MEASUREMENT_SET),
    caltable=g0,
    field=FLUX_CALIBRATOR,
    refant=REFERENCE_ANTENNA,
    spw="0:27~36",
    gaintype="G",
    calmode="p",
    solint="int",
    minsnr=5,
    gaintable=[antpos],
)
gaincal(
    vis=str(MEASUREMENT_SET),
    caltable=delay,
    field=FLUX_CALIBRATOR,
    refant=REFERENCE_ANTENNA,
    spw="0:5~58",
    gaintype="K",
    solint="inf",
    combine="scan",
    minsnr=5,
    gaintable=[antpos, g0],
)
plotms(
    vis=delay,
    xaxis="antenna1",
    yaxis="delay",
    coloraxis="baseline",
    plotfile=_plot("K0-delay"),
    overwrite=True,
)

bandpass(
    vis=str(MEASUREMENT_SET),
    caltable=bandpass_table,
    field=FLUX_CALIBRATOR,
    spw="",
    refant=REFERENCE_ANTENNA,
    combine="scan",
    solint="inf",
    bandtype="B",
    gaintable=[antpos, g0, delay],
)
for axis in ("amp", "phase"):
    plotms(
        vis=bandpass_table,
        field=FLUX_CALIBRATOR,
        xaxis="chan",
        yaxis=axis,
        coloraxis="corr",
        iteraxis="antenna",
        gridrows=2,
        gridcols=2,
        plotrange=[-1, -1, -180, 180] if axis == "phase" else [],
        plotfile=_plot(f"B0-{axis}"),
        overwrite=True,
    )

gaincal(
    vis=str(MEASUREMENT_SET),
    caltable=gain,
    field=FLUX_CALIBRATOR,
    spw="0:5~58",
    solint="inf",
    refant=REFERENCE_ANTENNA,
    gaintype="G",
    calmode="ap",
    solnorm=False,
    gaintable=[antpos, delay, bandpass_table],
    interp=["", "", "nearest"],
)
gaincal(
    vis=str(MEASUREMENT_SET),
    caltable=gain,
    field=GAIN_CALIBRATOR,
    spw="0:5~58",
    solint="inf",
    refant=REFERENCE_ANTENNA,
    gaintype="G",
    calmode="ap",
    gaintable=[antpos, delay, bandpass_table],
    append=True,
)
for axis in ("amp", "phase"):
    plotms(
        vis=gain,
        xaxis="time",
        yaxis=axis,
        iteraxis="corr",
        coloraxis="baseline",
        plotrange=[-1, -1, -180, 180] if axis == "phase" else [],
        plotfile=_plot(f"G1-{axis}"),
        overwrite=True,
    )

flux_scale = fluxscale(
    vis=str(MEASUREMENT_SET),
    caltable=gain,
    fluxtable=flux_gain,
    reference=FLUX_CALIBRATOR,
    transfer=[GAIN_CALIBRATOR],
    incremental=False,
)

common_tables = [antpos, flux_gain, delay, bandpass_table]
applycal(
    vis=str(MEASUREMENT_SET),
    field=FLUX_CALIBRATOR,
    gaintable=common_tables,
    gainfield=["", FLUX_CALIBRATOR, "", ""],
    interp=["", "nearest", "", ""],
    calwt=False,
)
applycal(
    vis=str(MEASUREMENT_SET),
    field=GAIN_CALIBRATOR,
    gaintable=common_tables,
    gainfield=["", GAIN_CALIBRATOR, "", ""],
    interp=["", "nearest", "", ""],
    calwt=False,
)
applycal(
    vis=str(MEASUREMENT_SET),
    field=SCIENCE_FIELDS,
    gaintable=common_tables,
    gainfield=["", GAIN_CALIBRATOR, "", ""],
    interp=["", "linear", "", ""],
    calwt=False,
)

for field, label in (
    (FLUX_CALIBRATOR, "flux-calibrator"),
    (GAIN_CALIBRATOR, "gain-calibrator"),
):
    for axis in ("amp", "phase"):
        plotms(
            vis=str(MEASUREMENT_SET),
            field=field,
            correlation="RR,LL",
            avgtime="60",
            xaxis="channel",
            yaxis=axis,
            ydatacolumn="corrected",
            coloraxis="corr",
            plotrange=[-1, -1, -180, 180] if axis == "phase" else [],
            plotfile=_plot(f"{label}-corrected-{axis}"),
            overwrite=True,
        )

flag_after = flagdata(
    vis=str(MEASUREMENT_SET),
    mode="summary",
    action="calculate",
)
flagmanager(
    vis=str(MEASUREMENT_SET),
    mode="save",
    versionname="sl1mjax_post_apply",
    comment="After CASA calibration application",
)
(OUTPUT / "result.json").write_text(
    json.dumps(
        {
            "schema_version": 1,
            "measurement_set": str(MEASUREMENT_SET),
            "reference_antenna": REFERENCE_ANTENNA,
            "flux_calibrator": FLUX_CALIBRATOR,
            "gain_calibrator": GAIN_CALIBRATOR,
            "science_fields": SCIENCE_FIELDS,
            "flag_summary_before": flag_before,
            "flag_summary_calibration_input": flag_calibration_input,
            "flag_summary_after": flag_after,
            "flux_scale": flux_scale,
            "calibration_tables": {
                "antenna_position": antpos,
                "diagnostic_phase": g0_all,
                "phase": g0,
                "delay": delay,
                "bandpass": bandpass_table,
                "gain": gain,
                "flux_scaled_gain": flux_gain,
            },
        },
        indent=2,
        sort_keys=True,
        default=_json_default,
    )
    + "\n",
    encoding="utf-8",
)
print(OUTPUT / "result.json")
