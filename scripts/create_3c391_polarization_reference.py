"""Run with CASA to add Kcross / Df / Xf after the existing 3C391 K/B/G tables.

Uses the local Measurement Set under data/, not BagOfWinds.  3C84 has no G
in reference-v2, so this script solves G on J0319+4130 before leakage.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
FLUX_CALIBRATOR = "J1331+3030"
GAIN_CALIBRATOR = "J1822-0938"
LEAKAGE_CALIBRATOR = "J0319+4130"
SCIENCE_FIELDS = "2~8"
REFERENCE_ANTENNA = "ea21"
EDGE_SPW = "0:5~58"
INPUT_FLAG_VERSION = "sl1mjax_calibration_input"
POST_POLCAL_FLAG_VERSION = "sl1mjax_post_polcal"
THREE_C286_FRACTIONAL_POLARISATION = 0.112
# NRAO 3C391 casaguide uses this angle in Q=P cos θ, U=P sin θ (not 2χ).
THREE_C286_CASAGUIDE_ANGLE_DEG = 66.0


def default_measurement_set() -> Path:
    override = os.environ.get("SL1MJAX_3C391_MS")
    if override:
        return Path(override).expanduser().resolve()
    candidates = (
        REPO / "data/3c391_work_v2/3c391_ctm_mosaic_10s_spw0.ms",
        REPO / "data/3c391_ctm_mosaic_10s_spw0.ms",
        REPO / "data/3c391/work-v2/3c391_ctm_mosaic_10s_spw0.ms",
    )
    for path in candidates:
        if path.is_dir():
            return path.resolve()
    return candidates[0]


def default_kbg_reference() -> Path:
    return Path(
        os.environ.get(
            "SL1MJAX_3C391_REFERENCE",
            REPO / "data/3c391/reference-v2",
        )
    ).expanduser().resolve()


def default_pol_reference() -> Path:
    return Path(
        os.environ.get(
            "SL1MJAX_3C391_POL_REFERENCE",
            REPO / "data/3c391/reference-pol",
        )
    ).expanduser().resolve()


def kbg_tables(reference: Path) -> dict[str, Path]:
    return {
        "antpos": reference / "3c391.antpos",
        "delay": reference / "3c391.K0",
        "bandpass": reference / "3c391.B0",
        "flux_gain": reference / "3c391.fluxscale1",
    }


def pol_tables(output: Path) -> dict[str, Path]:
    return {
        "leakage_gain": output / "3c391.G84",
        "kcross": output / "3c391.Kcross",
        "dterms": output / "3c391.Df0",
        "angle": output / "3c391.Xf0",
    }


def require_kbg_tables(reference: Path) -> dict[str, Path]:
    tables = kbg_tables(reference)
    missing = [str(path) for path in tables.values() if not path.is_dir()]
    if missing:
        raise FileNotFoundError(
            "K/B/G reference tables are missing: " + ", ".join(missing)
        )
    return tables


def gaintable_plan(reference: Path, output: Path) -> dict[str, list[str]]:
    """Prior tables applied while solving each new polarisation term."""

    kbg = kbg_tables(reference)
    pol = pol_tables(output)
    antpos = str(kbg["antpos"])
    delay = str(kbg["delay"])
    bandpass = str(kbg["bandpass"])
    flux_gain = str(kbg["flux_gain"])
    leakage_gain = str(pol["leakage_gain"])
    kcross = str(pol["kcross"])
    dterms = str(pol["dterms"])
    return {
        "leakage_gain": [antpos, delay, bandpass],
        "kcross": [antpos, flux_gain, delay, bandpass],
        "dterms": [antpos, leakage_gain, delay, bandpass, kcross],
        "angle": [antpos, flux_gain, delay, bandpass, kcross, dterms],
        "apply_flux": [antpos, flux_gain, delay, bandpass, kcross, dterms, str(pol["angle"])],
        "apply_gain": [antpos, flux_gain, delay, bandpass, kcross, dterms, str(pol["angle"])],
        "apply_leakage": [
            antpos,
            leakage_gain,
            delay,
            bandpass,
            kcross,
            dterms,
            str(pol["angle"]),
        ],
        "apply_science": [antpos, flux_gain, delay, bandpass, kcross, dterms, str(pol["angle"])],
    }


def three_c286_casaguide_qu(stokes_i: float) -> tuple[float, float, float]:
    """Return P, Q, U for 3C286 using the NRAO 3C391 continuum recipe.

    That recipe uses 11.2% linear polarisation and
    ``Q = P cos 66°``, ``U = P sin 66°``, ``V = 0``.  The 66° is the
    angle in those trig functions, not the IAU EVPA ``χ = (1/2) atan2(U,Q)``.
    """

    polarised = THREE_C286_FRACTIONAL_POLARISATION * float(stokes_i)
    angle = THREE_C286_CASAGUIDE_ANGLE_DEG * 3.141592653589793 / 180.0
    return polarised, polarised * math.cos(angle), polarised * math.sin(angle)


def stokes_i_from_setjy(result: Any) -> float:
    """Extract Stokes I from a nested CASA ``setjy`` return value."""

    if isinstance(result, dict):
        if "fluxd" in result:
            flux = result["fluxd"]
            values = getattr(flux, "tolist", lambda: flux)()
            if isinstance(values, (list, tuple)):
                return float(values[0].real if isinstance(values[0], complex) else values[0])
            return float(values)
        for value in result.values():
            try:
                return stokes_i_from_setjy(value)
            except ValueError:
                continue
    raise ValueError("setjy did not return Stokes I")


def flag_version_names(listing: object) -> set[str]:
    names: set[str] = set()
    if isinstance(listing, dict):
        for value in listing.values():
            if isinstance(value, dict) and "name" in value:
                names.add(str(value["name"]))
            elif isinstance(value, str):
                names.add(value)
    return names


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Solve 3C391 cross-hand delay, 3C84 leakage, and 3C286 R-L phase "
            "from the local Measurement Set."
        )
    )
    parser.add_argument("--ms", type=Path, default=default_measurement_set())
    parser.add_argument("--kbg-reference", type=Path, default=default_kbg_reference())
    parser.add_argument("--output", type=Path, default=default_pol_reference())
    parser.add_argument(
        "--skip-apply",
        action="store_true",
        default=os.environ.get("SL1MJAX_3C391_SKIP_APPLY") == "1",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        default=os.environ.get("SL1MJAX_3C391_SKIP_PLOTS", "1") != "0",
    )
    if argv is None:
        argv = []
        skip_next = False
        script_name = Path(__file__).name
        for argument in sys.argv[1:]:
            if skip_next:
                skip_next = False
                continue
            if argument in {"-c", "--logfile"}:
                skip_next = True
                continue
            if argument in {"--nologger", "--nogui", "--agg"}:
                continue
            if argument.endswith(script_name):
                continue
            argv.append(argument)
    return parser.parse_args(argv)


def _json_default(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    return str(value)


def _remove_table(path: Path) -> None:
    if path.exists():
        import shutil

        shutil.rmtree(path)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_args(argv)
    measurement_set = arguments.ms.resolve()
    reference = arguments.kbg_reference.resolve()
    output = arguments.output.resolve()
    if not measurement_set.is_dir():
        raise FileNotFoundError(measurement_set)
    kbg = require_kbg_tables(reference)
    pol = pol_tables(output)
    plan = gaintable_plan(reference, output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "plots").mkdir(exist_ok=True)

    from casatasks import applycal, flagdata, flagmanager, gaincal, polcal, setjy

    vis = str(measurement_set)
    listing = flagmanager(vis=vis, mode="list")
    available = flag_version_names(listing)
    if INPUT_FLAG_VERSION not in available:
        raise FileNotFoundError(
            f"flag version {INPUT_FLAG_VERSION!r} is not on {measurement_set}"
        )
    flagmanager(vis=vis, mode="restore", versionname=INPUT_FLAG_VERSION)
    if POST_POLCAL_FLAG_VERSION in available:
        flagmanager(vis=vis, mode="delete", versionname=POST_POLCAL_FLAG_VERSION)

    setjy_flux = setjy(
        vis=vis,
        field=FLUX_CALIBRATOR,
        standard="Perley-Butler 2017",
        model="3C286_C.im",
        usescratch=True,
        scalebychan=True,
        spw="",
    )
    stokes_i = stokes_i_from_setjy(setjy_flux)
    polarised_flux, stokes_q, stokes_u = three_c286_casaguide_qu(stokes_i)
    setjy_polarised = setjy(
        vis=vis,
        field=FLUX_CALIBRATOR,
        standard="manual",
        fluxdensity=[stokes_i, stokes_q, stokes_u, 0.0],
        usescratch=True,
        scalebychan=False,
        spw="",
    )
    # 3C84 is variable and is not in Perley-Butler 2017 under J0319+4130.
    # Leakage only needs Q=U=V=0; G84 absorbs the Stokes-I scale.
    setjy_leakage = setjy(
        vis=vis,
        field=LEAKAGE_CALIBRATOR,
        standard="manual",
        fluxdensity=[1.0, 0.0, 0.0, 0.0],
        spix=[0.0],
        reffreq="4.6GHz",
        usescratch=True,
        scalebychan=True,
    )
    leakage_model = "manual unpolarized I=1 Jy (3C84 not in Perley-Butler 2017)"
    flux_polarised_model = {
        "standard": "NRAO 3C391 casaguide",
        "fractional_polarisation": THREE_C286_FRACTIONAL_POLARISATION,
        "casaguide_position_angle_deg": THREE_C286_CASAGUIDE_ANGLE_DEG,
        "stokes_i_jy": stokes_i,
        "stokes_q_jy": stokes_q,
        "stokes_u_jy": stokes_u,
        "stokes_v_jy": 0.0,
        "linear_flux_jy": polarised_flux,
    }

    for path in pol.values():
        _remove_table(path)

    gaincal(
        vis=vis,
        caltable=str(pol["leakage_gain"]),
        field=LEAKAGE_CALIBRATOR,
        spw=EDGE_SPW,
        solint="inf",
        combine="scan",
        refant=REFERENCE_ANTENNA,
        gaintype="G",
        calmode="ap",
        minsnr=5,
        gaintable=plan["leakage_gain"],
        parang=True,
    )
    gaincal(
        vis=vis,
        caltable=str(pol["kcross"]),
        field=FLUX_CALIBRATOR,
        spw=EDGE_SPW,
        solint="inf",
        combine="scan",
        refant=REFERENCE_ANTENNA,
        gaintype="KCROSS",
        minsnr=3,
        gaintable=plan["kcross"],
        parang=True,
    )
    polcal(
        vis=vis,
        caltable=str(pol["dterms"]),
        field=LEAKAGE_CALIBRATOR,
        spw=EDGE_SPW,
        solint="inf",
        combine="scan",
        poltype="Df",
        refant=REFERENCE_ANTENNA,
        minsnr=3,
        gaintable=plan["dterms"],
    )
    polcal(
        vis=vis,
        caltable=str(pol["angle"]),
        field=FLUX_CALIBRATOR,
        spw=EDGE_SPW,
        solint="inf",
        combine="scan",
        poltype="Xf",
        refant=REFERENCE_ANTENNA,
        minsnr=3,
        gaintable=plan["angle"],
    )

    apply_summaries: dict[str, Any] = {}
    if not arguments.skip_apply:
        applycal(
            vis=vis,
            field=FLUX_CALIBRATOR,
            gaintable=plan["apply_flux"],
            gainfield=["", FLUX_CALIBRATOR, "", "", "", "", ""],
            interp=["", "nearest", "", "", "", "", ""],
            calwt=False,
            parang=True,
        )
        applycal(
            vis=vis,
            field=GAIN_CALIBRATOR,
            gaintable=plan["apply_gain"],
            gainfield=["", GAIN_CALIBRATOR, "", "", "", "", ""],
            interp=["", "nearest", "", "", "", "", ""],
            calwt=False,
            parang=True,
        )
        applycal(
            vis=vis,
            field=LEAKAGE_CALIBRATOR,
            gaintable=plan["apply_leakage"],
            gainfield=["", LEAKAGE_CALIBRATOR, "", "", "", "", ""],
            interp=["", "nearest", "", "", "", "", ""],
            calwt=False,
            parang=True,
        )
        applycal(
            vis=vis,
            field=SCIENCE_FIELDS,
            gaintable=plan["apply_science"],
            gainfield=["", GAIN_CALIBRATOR, "", "", "", "", ""],
            interp=["", "linear", "", "", "", "", ""],
            calwt=False,
            parang=True,
        )
        apply_summaries = flagdata(vis=vis, mode="summary", action="calculate")
        listing = flagmanager(vis=vis, mode="list")
        if POST_POLCAL_FLAG_VERSION in flag_version_names(listing):
            flagmanager(vis=vis, mode="delete", versionname=POST_POLCAL_FLAG_VERSION)
        flagmanager(
            vis=vis,
            mode="save",
            versionname=POST_POLCAL_FLAG_VERSION,
            comment="After K/B/G plus Kcross/Df/Xf apply",
        )

    if not arguments.skip_plots:
        from casaplotms import plotms
        plotms(
            vis=str(pol["kcross"]),
            xaxis="antenna1",
            yaxis="delay",
            plotfile=str(output / "plots" / "Kcross-delay.png"),
            overwrite=True,
            showgui=False,
        )
        for table, name, axis in (
            (pol["dterms"], "Df0", "amp"),
            (pol["angle"], "Xf0", "phase"),
        ):
            plotms(
                vis=str(table),
                xaxis="chan",
                yaxis=axis,
                coloraxis="corr",
                iteraxis="antenna",
                plotfile=str(output / "plots" / f"{name}-{axis}.png"),
                overwrite=True,
                showgui=False,
            )

    payload = {
        "schema_version": 1,
        "measurement_set": str(measurement_set),
        "kbg_reference": str(reference),
        "reference_antenna": REFERENCE_ANTENNA,
        "flux_calibrator": FLUX_CALIBRATOR,
        "gain_calibrator": GAIN_CALIBRATOR,
        "leakage_calibrator": LEAKAGE_CALIBRATOR,
        "leakage_model": leakage_model,
        "flux_polarised_model": flux_polarised_model,
        "setjy_flux": setjy_flux,
        "setjy_polarised": setjy_polarised,
        "setjy_leakage": setjy_leakage,
        "tables": {name: str(path) for name, path in {**kbg, **pol}.items()},
        "gaintable_plan": plan,
        "skip_apply": arguments.skip_apply,
        "flag_summary_after_apply": apply_summaries,
    }
    (output / "result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    print(output / "result.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
