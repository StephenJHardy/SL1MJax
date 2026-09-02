"""Create the two immutable 3C391 four-correlation Measurement Sets.

``3c391_gkb_only_4corr.ms`` has CORRECTED_DATA = antenna position + K/B/G.
That is the JAX polarisation-apply input.

``3c391_casa_fullpol_4corr.ms`` has CORRECTED_DATA = K/B/G + Kcross/Df/Xf
with parang=True. That is the CASA comparison product. Do not apply the JAX
polarisation solution to it.

The current work_v2 MS already carries sl1mjax_post_polcal CORRECTED_DATA.
Restoring an older flag version does not undo that column, so this script
copies it for the fullpol product and re-applies G/K/B only onto a second copy.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
WORK_V2 = REPO / "data/3c391_work_v2/3c391_ctm_mosaic_10s_spw0.ms"
KBG_REFERENCE = REPO / "data/3c391/reference-v2"
POL_REFERENCE = REPO / "data/3c391/reference-pol"
OUTPUT = REPO / "data/3c391/fullpol_prep"
FLUX_CALIBRATOR = "J1331+3030"
GAIN_CALIBRATOR = "J1822-0938"
LEAKAGE_CALIBRATOR = "J0319+4130"
SCIENCE_FIELDS = "2~8"
GKB_ONLY_NAME = "3c391_gkb_only_4corr.ms"
CASA_FULLPOL_NAME = "3c391_casa_fullpol_4corr.ms"
CALIBRATION_INPUT_FLAG = "sl1mjax_calibration_input"
POST_POLCAL_FLAG = "sl1mjax_post_polcal"
GKB_ONLY_FLAG = "sl1mjax_gkb_only"


def _kbg_tables(reference: Path) -> dict[str, Path]:
    return {
        "antpos": reference / "3c391.antpos",
        "delay": reference / "3c391.K0",
        "bandpass": reference / "3c391.B0",
        "flux_gain": reference / "3c391.fluxscale1",
    }


def _pol_tables(reference: Path) -> dict[str, Path]:
    return {
        "leakage_gain": reference / "3c391.G84",
        "kcross": reference / "3c391.Kcross",
        "dterms": reference / "3c391.Df0",
        "angle": reference / "3c391.Xf0",
    }


def _copy_ms(source: Path, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    flag_src = Path(str(source) + ".flagversions")
    flag_dest = Path(str(dest) + ".flagversions")
    if flag_dest.exists():
        shutil.rmtree(flag_dest)
    if sys.platform == "darwin":
        subprocess.check_call(["cp", "-cR", str(source), str(dest)])
        if flag_src.is_dir():
            subprocess.check_call(["cp", "-cR", str(flag_src), str(flag_dest)])
        return
    subprocess.check_call(["rsync", "-a", str(source) + "/", str(dest) + "/"])
    if flag_src.is_dir():
        subprocess.check_call(["rsync", "-a", str(flag_src) + "/", str(flag_dest) + "/"])


def _flag_version_names(listing: object) -> set[str]:
    names: set[str] = set()
    if isinstance(listing, dict):
        for value in listing.values():
            if isinstance(value, dict) and "name" in value:
                names.add(str(value["name"]))
            elif isinstance(value, str):
                names.add(value)
    return names


def _write_state(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _casa_version() -> str:
    override = os.environ.get("SL1MJAX_CASA_VERSION")
    if override:
        return override
    casa = os.environ.get("SL1MJAX_CASA", "/Applications/CASA.app/Contents/MacOS/casa")
    try:
        result = subprocess.run(
            [casa, "--version"],
            check=False,
            capture_output=True,
            text=True,
        )
        text = (result.stdout or result.stderr or "").strip()
        return text.splitlines()[0] if text else "unknown"
    except OSError:
        return "unknown"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-ms", type=Path, default=WORK_V2)
    parser.add_argument("--kbg-reference", type=Path, default=KBG_REFERENCE)
    parser.add_argument("--pol-reference", type=Path, default=POL_REFERENCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--skip-gkb-apply", action="store_true")
    if argv is None:
        argv = [item for item in sys.argv[1:] if item not in {"--nologger", "--nogui", "--agg"}]
        cleaned: list[str] = []
        skip_next = False
        for item in argv:
            if skip_next:
                skip_next = False
                continue
            if item in {"-c", "--logfile"}:
                skip_next = True
                continue
            if item.endswith(Path(__file__).name):
                continue
            cleaned.append(item)
        argv = cleaned
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_args(argv)
    source = arguments.source_ms.resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    kbg = _kbg_tables(arguments.kbg_reference.resolve())
    pol = _pol_tables(arguments.pol_reference.resolve())
    missing = [str(path) for path in {**kbg, **pol}.values() if not path.is_dir()]
    if missing:
        raise FileNotFoundError("calibration tables missing: " + ", ".join(missing))
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    fullpol = output / CASA_FULLPOL_NAME
    gkb_only = output / GKB_ONLY_NAME
    casa_version = _casa_version()

    print(f"copying {source} -> {fullpol}", flush=True)
    _copy_ms(source, fullpol)
    print(f"copying {source} -> {gkb_only}", flush=True)
    _copy_ms(source, gkb_only)

    from casatasks import applycal, flagdata, flagmanager

    listing = flagmanager(vis=str(fullpol), mode="list")
    available = _flag_version_names(listing)
    if POST_POLCAL_FLAG not in available:
        raise FileNotFoundError(f"{source} is missing flag version {POST_POLCAL_FLAG}")
    flagmanager(vis=str(fullpol), mode="restore", versionname=POST_POLCAL_FLAG)
    _write_state(
        Path(str(fullpol) + ".calibration_state.json"),
        {
            "product": CASA_FULLPOL_NAME,
            "data_column": "CORRECTED_DATA",
            "calibration_state": "casa_fullpol",
            "applied": ["antpos", "K0", "B0", "fluxscale1", "Kcross", "Df", "Xf"],
            "parang": True,
            "calwt": False,
            "flag_version": POST_POLCAL_FLAG,
            "source_ms": str(source),
            "casa_version": casa_version,
            "do_not_apply_jax_polarisation": True,
            "calibration_tables": {name: str(path) for name, path in {**kbg, **pol}.items()},
        },
    )

    listing = flagmanager(vis=str(gkb_only), mode="list")
    available = _flag_version_names(listing)
    if CALIBRATION_INPUT_FLAG not in available:
        raise FileNotFoundError(f"{source} is missing flag version {CALIBRATION_INPUT_FLAG}")
    flagmanager(vis=str(gkb_only), mode="restore", versionname=CALIBRATION_INPUT_FLAG)
    if not arguments.skip_gkb_apply:
        common = [
            str(kbg["antpos"]),
            str(kbg["flux_gain"]),
            str(kbg["delay"]),
            str(kbg["bandpass"]),
        ]
        applycal(
            vis=str(gkb_only),
            field=FLUX_CALIBRATOR,
            gaintable=common,
            gainfield=["", FLUX_CALIBRATOR, "", ""],
            interp=["", "nearest", "", ""],
            calwt=False,
            parang=False,
        )
        applycal(
            vis=str(gkb_only),
            field=GAIN_CALIBRATOR,
            gaintable=common,
            gainfield=["", GAIN_CALIBRATOR, "", ""],
            interp=["", "nearest", "", ""],
            calwt=False,
            parang=False,
        )
        applycal(
            vis=str(gkb_only),
            field=SCIENCE_FIELDS,
            gaintable=common,
            gainfield=["", GAIN_CALIBRATOR, "", ""],
            interp=["", "linear", "", ""],
            calwt=False,
            parang=False,
        )
        applycal(
            vis=str(gkb_only),
            field=LEAKAGE_CALIBRATOR,
            gaintable=[
                str(kbg["antpos"]),
                str(pol["leakage_gain"]),
                str(kbg["delay"]),
                str(kbg["bandpass"]),
            ],
            gainfield=["", LEAKAGE_CALIBRATOR, "", ""],
            interp=["", "nearest", "", ""],
            calwt=False,
            parang=False,
        )
        if GKB_ONLY_FLAG in _flag_version_names(flagmanager(vis=str(gkb_only), mode="list")):
            flagmanager(vis=str(gkb_only), mode="delete", versionname=GKB_ONLY_FLAG)
        flagmanager(
            vis=str(gkb_only),
            mode="save",
            versionname=GKB_ONLY_FLAG,
            comment="After antenna-position + K/B/G only; no Kcross/D/X/parang",
        )
        flagdata(vis=str(gkb_only), mode="summary", action="calculate")
    _write_state(
        Path(str(gkb_only) + ".calibration_state.json"),
        {
            "product": GKB_ONLY_NAME,
            "data_column": "CORRECTED_DATA",
            "calibration_state": "gkb_only",
            "applied": ["antpos", "K0", "B0", "fluxscale1"],
            "applied_leakage_calibrator": ["antpos", "G84", "K0", "B0"],
            "not_applied": ["Kcross", "Df", "Xf", "parang"],
            "parang": False,
            "calwt": False,
            "flag_version": GKB_ONLY_FLAG,
            "source_ms": str(source),
            "casa_version": casa_version,
            "jax_polarisation_input": True,
            "calibration_tables": {
                **{name: str(path) for name, path in kbg.items()},
                "leakage_gain": str(pol["leakage_gain"]),
            },
        },
    )
    print(fullpol)
    print(gkb_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
