"""Scientific THOL0001 calibration with one 3C147 flux model on fields 0 and 9.

Compatibility products under products/diagonal and products/fullpol are not
overwritten. This script writes products/scientific/ on a dedicated work MS.

Field 10 receives the same 3C147 model for prediction only. G is solved from
fields 0 and 9. HOLORASTER moving baselines do not enter the G solve.
Standalone CASA 6.7.6 script; do not import sl1mjax.
"""

from __future__ import annotations

import json
import math
import os
import shutil
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
        "SL1MJAX_THOL0001_SCIENTIFIC_MS",
        "/media/stephen/astro/vla/extracted/commissioning/work/THOL0001.lowerC.spw45.scientific.ms",
    )
)
PRODUCT_ROOT = Path(
    os.environ.get(
        "SL1MJAX_THOL0001_SCIENTIFIC_CAL_ROOT",
        "/media/stephen/astro/vla/extracted/commissioning/products/scientific",
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
PREDICTION_FIELDS = "0,9,10"
APPLY_FIELDS = "0,9,10,11"
FLUX_FIELD = "0"
D_FIELD = "9"
HOLORASTER_FIELD = "10"
THREE_C286_FIELD = "11"
SPW = "4,5"
EDGE_SPW = "4:5~58,5:5~58"
THREE_C286_FRACTIONAL_POLARISATION = 0.112
THREE_C286_EVPA_DEG = 66.0
SETJY_STANDARD = "Perley-Butler 2017"
SETJY_3C147_MODEL = "3C147_C.im"


def _table(product: str, name: str) -> str:
    destination = PRODUCT_ROOT / product
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / name
    if path.exists():
        shutil.rmtree(path)
    return str(path)


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _write_setjy(path: Path, payload: dict) -> None:
    try:
        _write(path, payload)
    except (TypeError, ValueError) as error:
        print(f"warning: could not serialise {path}: {error}")


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    if hasattr(value, "tolist"):
        try:
            return _jsonable(value.tolist())
        except TypeError, ValueError:
            pass
    if hasattr(value, "item"):
        try:
            return _jsonable(value.item())
        except TypeError, ValueError:
            pass
    return str(value)


def _setjy_3c147(vis: str, field: str) -> object:
    return setjy(
        vis=vis,
        field=field,
        standard=SETJY_STANDARD,
        model=SETJY_3C147_MODEL,
        usescratch=True,
        scalebychan=True,
        spw=SPW,
    )


if not WORK_MS.is_dir():
    raise FileNotFoundError(
        f"{WORK_MS} is missing. Copy the immutable commissioning MS to this "
        "scientific work path before solving. Do not reuse the compatibility "
        "work MS."
    )
if (
    "products/scientific" not in str(PRODUCT_ROOT)
    and os.environ.get("SL1MJAX_ALLOW_SCIENTIFIC_CAL_ROOT", "") != "1"
):
    raise ValueError(
        f"refusing to write scientific tables to {PRODUCT_ROOT}; "
        "compatibility products/diagonal and products/fullpol must stay fixtures"
    )

vis = str(WORK_MS)
try:
    flagmanager(
        vis=vis,
        mode="save",
        versionname="sl1mjax_scientific_pristine",
        comment="scientific work MS before solves",
    )
except Exception as error:
    print(f"flagmanager save skipped: {error}")

antpos = _table("diagonal", "antpos.cal")
g0 = _table("diagonal", "G0.cal")
delay = _table("diagonal", "K0.cal")
bandpass_table = _table("diagonal", "B0.cal")
gain = _table("diagonal", "G1.cal")
gain_hold = _table("diagonal", "G1_hold_scan51.cal")

gencal(vis=vis, caltable=antpos, caltype="antpos")
setjy_field0 = _setjy_3c147(vis, FLUX_FIELD)
setjy_field9 = _setjy_3c147(vis, D_FIELD)
setjy_field10 = _setjy_3c147(vis, HOLORASTER_FIELD)
_write_setjy(
    PRODUCT_ROOT / "setjy_3c147.json",
    {
        "standard": SETJY_STANDARD,
        "model": SETJY_3C147_MODEL,
        "scalebychan": True,
        "usescratch": True,
        "fields": {
            "0": _jsonable(setjy_field0),
            "9": _jsonable(setjy_field9),
            "10": _jsonable(setjy_field10),
        },
        "same_intrinsic_model_on_fields_0_and_9": True,
        "field_10_prediction_only": True,
        "holoraster_used_for_moving_gains": False,
        "table5_jy_at_4p564ghz": 8.290006,
        "table5_coefficients": [1.4516, -0.6961, -0.2007, 0.0640, -0.0464, 0.0289],
        "notes": [
            "Scientific G uses this CASA setjy model, not a later I=1 overwrite",
            "Table 5 polynomial is 8.290006 Jy at 4.564 GHz; record CASA fluxd exactly",
        ],
    },
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
    versionname="sl1mjax_scientific_diagonal_input",
    comment="After consistent 3C147 setjy; before applycal",
)
applycal(
    vis=vis,
    field=PREDICTION_FIELDS,
    gaintable=[antpos, gain, delay, bandpass_table],
    gainfield=["", "", "", ""],
    interp=["", "linear", "", "nearest"],
    calwt=False,
    parang=False,
    applymode="calflag",
)
flagdata(vis=vis, mode="summary", action="calculate", name="scientific_diagonal")
_write(
    PRODUCT_ROOT / "diagonal" / "product.json",
    {
        "product": "thol0001_lower_c_scientific_diagonal",
        "compatibility_fixture": False,
        "vis": vis,
        "refant": REFERENCE_ANTENNA,
        "held_out_reference": HELD_OUT_REFERENCE,
        "held_out_flux_scan": HELD_OUT_FLUX_SCAN,
        "held_out_flux_scan_kind": "g_holdout",
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
        "model": {
            "standard": SETJY_STANDARD,
            "image": SETJY_3C147_MODEL,
            "fields": PREDICTION_FIELDS,
            "same_intrinsic_model_on_fields_0_and_9": True,
        },
        "c147_offset_used_for_d": False,
        "holoraster_used_for_moving_gains": False,
        "flag_version": "sl1mjax_scientific_diagonal_input",
        "setjy_record": str(PRODUCT_ROOT / "setjy_3c147.json"),
    },
)

kcross = _table("fullpol", "Kcross.cal")
dterms = _table("fullpol", "Df.cal")
angle = _table("fullpol", "Xf.cal")
dterms_qu = _table("fullpol", "Df_QU.cal")
setjy_flux = setjy(
    vis=vis,
    field=THREE_C286_FIELD,
    standard=SETJY_STANDARD,
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
stokes_q = polarised * math.cos(math.radians(THREE_C286_EVPA_DEG))
stokes_u = polarised * math.sin(math.radians(THREE_C286_EVPA_DEG))
setjy(
    vis=vis,
    field=THREE_C286_FIELD,
    standard="manual",
    fluxdensity=[stokes_i, stokes_q, stokes_u, 0.0],
    usescratch=True,
    scalebychan=False,
    spw=SPW,
)
# Re-assert the 3C147 model after the 3C286 block. Do not write I=1 on field 9.
setjy_field0 = _setjy_3c147(vis, FLUX_FIELD)
setjy_field9 = _setjy_3c147(vis, D_FIELD)
setjy_field10 = _setjy_3c147(vis, HOLORASTER_FIELD)
_write_setjy(
    PRODUCT_ROOT / "setjy_3c147.json",
    {
        "standard": SETJY_STANDARD,
        "model": SETJY_3C147_MODEL,
        "scalebychan": True,
        "usescratch": True,
        "fields": {
            "0": _jsonable(setjy_field0),
            "9": _jsonable(setjy_field9),
            "10": _jsonable(setjy_field10),
        },
        "3C286": _jsonable(setjy_flux),
        "same_intrinsic_model_on_fields_0_and_9": True,
        "field_10_prediction_only": True,
        "field_9_manual_i1_overwrite": False,
        "holoraster_used_for_moving_gains": False,
        "table5_jy_at_4p564ghz": 8.290006,
        "table5_coefficients": [1.4516, -0.6961, -0.2007, 0.0640, -0.0464, 0.0289],
    },
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
polcal(
    vis=vis,
    caltable=dterms_qu,
    field=ON_AXIS_FIELDS,
    spw=EDGE_SPW,
    solint="inf",
    combine="scan",
    poltype="Df+QU",
    refant=REFERENCE_ANTENNA,
    minsnr=3,
    gaintable=[antpos, gain, delay, bandpass_table, kcross],
)
flagmanager(
    vis=vis,
    mode="save",
    versionname="sl1mjax_scientific_fullpol_input",
    comment="After scientific Kcross/D/X solves",
)
applycal(
    vis=vis,
    field=APPLY_FIELDS,
    gaintable=[antpos, gain, delay, bandpass_table, kcross, dterms, angle],
    interp=["", "linear", "", "nearest", "", "", ""],
    calwt=False,
    parang=True,
    applymode="calflag",
)
_write(
    PRODUCT_ROOT / "fullpol" / "product.json",
    {
        "product": "thol0001_lower_c_scientific_fullpol",
        "compatibility_fixture": False,
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
            "Df_QU": dterms_qu,
        },
        "model": {
            "3C147": {
                "standard": SETJY_STANDARD,
                "image": SETJY_3C147_MODEL,
                "fields": PREDICTION_FIELDS,
                "same_intrinsic_model_on_fields_0_and_9": True,
                "manual_i1_on_field_9": False,
            },
            "3C286": {
                "I": stokes_i,
                "Q": stokes_q,
                "U": stokes_u,
                "V": 0.0,
                "fractional_polarisation": THREE_C286_FRACTIONAL_POLARISATION,
                "casaguide_two_chi_deg": THREE_C286_EVPA_DEG,
                "iau_evpa_deg": 0.5 * THREE_C286_EVPA_DEG,
                "evpa_deg": THREE_C286_EVPA_DEG,
                "evpa_label_is_casaguide_two_chi": True,
            },
        },
        "c147_offset_used_for_d": False,
        "holoraster_used_for_moving_gains": False,
        "apply_from": "DATA",
        "holoraster_applycal": True,
        "flag_version": "sl1mjax_scientific_fullpol_input",
    },
)
print(PRODUCT_ROOT / "setjy_3c147.json")
print(PRODUCT_ROOT / "diagonal" / "product.json")
print(PRODUCT_ROOT / "fullpol" / "product.json")
