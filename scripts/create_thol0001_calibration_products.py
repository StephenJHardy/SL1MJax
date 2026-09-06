"""CASA 6.7.6 calibration of the THOL0001 lower-C commissioning work MS.

The immutable commissioning MS is never opened for write. Run this script
with the explicit Bacchus CASA executable after copying a writable work MS.

Diagonal product: antpos / K / B / G, parang=False, calwt=False.
Full-pol product: Kcross / Df / Xf applied once, parang=True, calwt=False.
D is solved from on-axis J0542+4951 only. C147-* is excluded.
G1_hold_scan51 is a G holdout only: scan 51 is excluded from G but used in B.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from casatasks import (
    applycal,
    bandpass,
    flagdata,
    flagmanager,
    gaincal,
    gencal,
    polcal,
    setjy,
)

WORK_MS = Path(
    os.environ.get(
        "SL1MJAX_THOL0001_WORK_MS",
        "/media/stephen/astro/vla/extracted/commissioning/work/THOL0001.lowerC.spw45.work.ms",
    )
)
PRODUCT_ROOT = Path(
    os.environ.get(
        "SL1MJAX_THOL0001_CAL_ROOT",
        "/media/stephen/astro/vla/extracted/commissioning/products",
    )
)
REFERENCE_ANTENNA = os.environ.get("SL1MJAX_THOL0001_REFANT", "ea02")
HELD_OUT_REFERENCE = os.environ.get("SL1MJAX_THOL0001_HOLD_REFANT", "ea26")
FLUX_SCANS = "2,51"
HELD_OUT_FLUX_SCAN = "51"
SOLVE_FLUX_SCAN = "2"
PHASE_SCANS = (
    "14,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49,53,56,"
    "58,60,62,64,66,68,70,72,74,76,78,80,82,84,86,88,90,92,94,96,98,100,102"
)
ON_AXIS_FIELDS = "0,9"
FLUX_FIELD = "0"
D_FIELD = "9"
THREE_C286_FIELD = "11"
SPW = "4,5"
EDGE_SPW = "4:5~58,5:5~58"
THREE_C286_FRACTIONAL_POLARISATION = 0.112
THREE_C286_EVPA_DEG = 66.0


def _table(product: str, name: str) -> str:
    destination = PRODUCT_ROOT / product
    destination.mkdir(parents=True, exist_ok=True)
    return str(destination / name)


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


if not WORK_MS.is_dir():
    raise FileNotFoundError(WORK_MS)

vis = str(WORK_MS)
flagmanager(vis=vis, mode="save", versionname="sl1mjax_pristine", comment="work MS before solves")

antpos = _table("diagonal", "antpos.cal")
g0 = _table("diagonal", "G0.cal")
delay = _table("diagonal", "K0.cal")
bandpass_table = _table("diagonal", "B0.cal")
gain = _table("diagonal", "G1.cal")
gain_hold = _table("diagonal", "G1_hold_scan51.cal")

gencal(vis=vis, caltable=antpos, caltype="antpos")
setjy(
    vis=vis,
    field=FLUX_FIELD,
    standard="Perley-Butler 2017",
    model="3C147_C.im",
    usescratch=True,
    scalebychan=True,
    spw=SPW,
)

gaincal(
    vis=vis,
    caltable=g0,
    field=FLUX_FIELD,
    scan=SOLVE_FLUX_SCAN,
    refant=REFERENCE_ANTENNA,
    spw="4:27~36,5:27~36",
    gaintype="G",
    calmode="p",
    solint="int",
    minsnr=5,
    gaintable=[antpos],
)
gaincal(
    vis=vis,
    caltable=delay,
    field=FLUX_FIELD,
    scan=SOLVE_FLUX_SCAN,
    refant=REFERENCE_ANTENNA,
    spw=EDGE_SPW,
    gaintype="K",
    solint="inf",
    combine="scan",
    minsnr=5,
    gaintable=[antpos, g0],
)
bandpass(
    vis=vis,
    caltable=bandpass_table,
    field=FLUX_FIELD,
    scan=FLUX_SCANS,
    spw=SPW,
    refant=REFERENCE_ANTENNA,
    combine="scan",
    solint="inf",
    bandtype="B",
    gaintable=[antpos, g0, delay],
)
gaincal(
    vis=vis,
    caltable=gain,
    field=ON_AXIS_FIELDS,
    scan=f"{FLUX_SCANS},{PHASE_SCANS}",
    spw=EDGE_SPW,
    solint="inf",
    refant=REFERENCE_ANTENNA,
    gaintype="G",
    calmode="ap",
    solnorm=False,
    minsnr=5,
    gaintable=[antpos, delay, bandpass_table],
    interp=["", "", "nearest"],
)
gaincal(
    vis=vis,
    caltable=gain_hold,
    field=ON_AXIS_FIELDS,
    scan=f"{SOLVE_FLUX_SCAN},{PHASE_SCANS}",
    spw=EDGE_SPW,
    solint="inf",
    refant=REFERENCE_ANTENNA,
    gaintype="G",
    calmode="ap",
    solnorm=False,
    minsnr=5,
    gaintable=[antpos, delay, bandpass_table],
    interp=["", "", "nearest"],
)

flagmanager(
    vis=vis,
    mode="save",
    versionname="sl1mjax_diagonal_input",
    comment="After 3C147 setjy; before applycal",
)
applycal(
    vis=vis,
    field=ON_AXIS_FIELDS,
    gaintable=[antpos, gain, delay, bandpass_table],
    gainfield=["", "", "", ""],
    interp=["", "linear", "", "nearest"],
    calwt=False,
    parang=False,
    applymode="calflag",
)
flagdata(vis=vis, mode="summary", action="calculate", name="diagonal_onaxis")
_write(
    PRODUCT_ROOT / "diagonal" / "product.json",
    {
        "product": "thol0001_lower_c_diagonal",
        "vis": vis,
        "refant": REFERENCE_ANTENNA,
        "held_out_reference": HELD_OUT_REFERENCE,
        "held_out_flux_scan": HELD_OUT_FLUX_SCAN,
        "held_out_flux_scan_kind": "g_holdout",
        "held_out_flux_scan_note": ("G1_hold_scan51 is a G holdout only; scan 51 was used in B0"),
        "parang": False,
        "calwt": False,
        "tables": {
            "antpos": antpos,
            "G0": g0,
            "K0": delay,
            "B0": bandpass_table,
            "G1": gain,
            "G1_hold_scan51": gain_hold,
        },
        "model": "CASA Perley-Butler 2017 3C147_C.im on field 0",
        "c147_offset_used_for_d": False,
        "holoraster_used_for_moving_gains": False,
        "flag_version": "sl1mjax_diagonal_input",
    },
)

kcross = _table("fullpol", "Kcross.cal")
dterms = _table("fullpol", "Df.cal")
angle = _table("fullpol", "Xf.cal")
setjy_flux = setjy(
    vis=vis,
    field=THREE_C286_FIELD,
    standard="Perley-Butler 2017",
    model="3C286_C.im",
    usescratch=True,
    scalebychan=True,
    spw=SPW,
)
stokes_i = 0.0
try:
    flux0 = next(iter(setjy_flux.values()))
    stokes_i = float(flux0["0"]["fluxd"][0])
except StopIteration, KeyError, TypeError:
    stokes_i = 7.5
polarised = THREE_C286_FRACTIONAL_POLARISATION * stokes_i
stokes_q = polarised * __import__("math").cos(__import__("math").radians(THREE_C286_EVPA_DEG))
stokes_u = polarised * __import__("math").sin(__import__("math").radians(THREE_C286_EVPA_DEG))
setjy(
    vis=vis,
    field=THREE_C286_FIELD,
    standard="manual",
    fluxdensity=[stokes_i, stokes_q, stokes_u, 0.0],
    usescratch=True,
    scalebychan=False,
    spw=SPW,
)
setjy(
    vis=vis,
    field=D_FIELD,
    standard="manual",
    fluxdensity=[1.0, 0.0, 0.0, 0.0],
    spix=[0.0],
    reffreq="4.6GHz",
    usescratch=True,
    scalebychan=True,
    spw=SPW,
)
gaincal(
    vis=vis,
    caltable=kcross,
    field=THREE_C286_FIELD,
    spw=EDGE_SPW,
    solint="inf",
    combine="scan",
    refant=REFERENCE_ANTENNA,
    gaintype="KCROSS",
    minsnr=3,
    gaintable=[antpos, gain, delay, bandpass_table],
    parang=True,
)
polcal(
    vis=vis,
    caltable=dterms,
    field=ON_AXIS_FIELDS,
    spw=EDGE_SPW,
    solint="inf",
    combine="scan",
    poltype="Df",
    refant=REFERENCE_ANTENNA,
    minsnr=3,
    gaintable=[antpos, gain, delay, bandpass_table, kcross],
)
polcal(
    vis=vis,
    caltable=angle,
    field=THREE_C286_FIELD,
    spw=EDGE_SPW,
    solint="inf",
    combine="scan",
    poltype="Xf",
    refant=REFERENCE_ANTENNA,
    minsnr=3,
    gaintable=[antpos, gain, delay, bandpass_table, kcross, dterms],
)
flagmanager(
    vis=vis,
    mode="save",
    versionname="sl1mjax_fullpol_input",
    comment="After Kcross/D/X solves",
)
applycal(
    vis=vis,
    field=f"{ON_AXIS_FIELDS},{THREE_C286_FIELD}",
    gaintable=[antpos, gain, delay, bandpass_table, kcross, dterms, angle],
    interp=["", "linear", "", "nearest", "", "", ""],
    calwt=False,
    parang=True,
    applymode="calflag",
)
_write(
    PRODUCT_ROOT / "fullpol" / "product.json",
    {
        "product": "thol0001_lower_c_fullpol",
        "vis": vis,
        "refant": REFERENCE_ANTENNA,
        "held_out_reference": HELD_OUT_REFERENCE,
        "parang": True,
        "calwt": False,
        "tables": {
            "antpos": antpos,
            "G1": gain,
            "K0": delay,
            "B0": bandpass_table,
            "Kcross": kcross,
            "Df": dterms,
            "Xf": angle,
        },
        "model": {
            "3C147_D": "manual unpolarized I=1 on field 9; Q=U=V=0 with explicit uncertainty",
            "3C286": {
                "I": stokes_i,
                "Q": stokes_q,
                "U": stokes_u,
                "V": 0.0,
                "fractional_polarisation": THREE_C286_FRACTIONAL_POLARISATION,
                "evpa_deg": THREE_C286_EVPA_DEG,
            },
        },
        "c147_offset_used_for_d": False,
        "flag_version": "sl1mjax_fullpol_input",
    },
)
print(PRODUCT_ROOT / "diagonal" / "product.json")
print(PRODUCT_ROOT / "fullpol" / "product.json")
