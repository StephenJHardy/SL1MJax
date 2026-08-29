"""Generate the Stage-1 CASA awp2 power-beam oracle on Bacchus.

Run this with CASA 6, not with ``uv run``. Runtime SL1MJax never invokes
CASA. Stage 1 asks whether the committed CASSBEAM evaluator reproduces
CASA's independent, default EVLA ray-traced A-term. ``awp2`` uses that
internal model and does not take a ``vpmanager`` / ``vptable`` beam.
Do not load our CASSBEAM tables into CASA here; that would test
ingestion, not beam-model accuracy.

Outputs land in ``outputs/casa_awp2_power_oracle/``. Copy that directory
into ``src/sl1mjax/data/casa_awp2_oracle/`` and set ``frozen: true`` only
after the comparison accepts coordinates, FWHM, and residual power.
``setpbimage`` remains a possible later plumbing experiment. It is not
this oracle and is not the ``awp2`` ingest path.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
from casatasks import exportfits, simobserve, tclean
from casatools import componentlist, table

OUTPUT = Path(
    os.environ.get("SL1MJAX_CASA_AWP2_ORACLE", "outputs/casa_awp2_power_oracle")
).resolve()
PHASE_CENTRE = (180.0, 45.0)
FREQUENCIES_MHZ = (4564, 4692)
HOURANGLES = {
    "ha0": "transit",
    "ha_plus2": "2:00:00",
    "ha_minus2": "-2:00:00",
}
STOKES = ("I", "RR", "LL")
IMAGE_SIZE = 512
CELL = "4arcsec"
PBLIMIT = 0.01
VOLTAGE_PATTERN = "casa_default_evla_raytraced"
_VLA_TELESCOPES = frozenset({"VLA", "EVLA"})
_WGS84_A_M = 6378137.0
_WGS84_E2 = 6.69437999014e-3


def _remove(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_ms(frequency_mhz: int, hourangle_key: str, hourangle: str) -> Path:
    project = OUTPUT / f"ms_{frequency_mhz}_{hourangle_key}"
    _remove(project)
    components = project.with_suffix(".cl")
    _remove(components)
    cl = componentlist()
    cl.addcomponent(
        flux=1.0,
        fluxunit="Jy",
        dir=f"J2000 {PHASE_CENTRE[0]}deg {PHASE_CENTRE[1]}deg",
        shape="point",
        spectrumtype="constant",
        freq=f"{frequency_mhz / 1000.0}GHz",
    )
    cl.rename(str(components))
    cl.close()
    cwd = Path.cwd()
    os.chdir(OUTPUT)
    try:
        simobserve(
            project=project.name,
            complist=str(components),
            compwidth="1MHz",
            comp_nchan=1,
            setpointings=True,
            integration="10s",
            direction=f"J2000 {PHASE_CENTRE[0]}deg {PHASE_CENTRE[1]}deg",
            mapsize="4arcmin",
            obsmode="int",
            antennalist="vla.d.cfg",
            hourangle=hourangle,
            totaltime="10s",
            thermalnoise="",
            graphics="none",
            overwrite=True,
            verbose=False,
        )
    finally:
        os.chdir(cwd)
    matches = list(project.glob("*.ms"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one MeasurementSet under {project}")
    return matches[0]


def _require_vla_telescope(vis: Path) -> str:
    tb = table()
    tb.open(str(vis) + "/OBSERVATION")
    try:
        names = [
            str(name).strip().upper()
            for name in np.ravel(tb.getcol("TELESCOPE_NAME"))
        ]
    finally:
        tb.close()
    if not names:
        raise RuntimeError(f"{vis} has no OBSERVATION.TELESCOPE_NAME")
    unknown = [name for name in names if name not in _VLA_TELESCOPES]
    if unknown:
        raise RuntimeError(
            f"awp2 default EVLA A-term requires VLA/EVLA; {vis} has {names}"
        )
    return names[0]


def _snapshot_parallactic_angle_rad(vis: Path) -> float:
    """Mean alt-az parallactic angle of the snapshot, radians.

    Uses the same GMST formula as ``calibration_terms.parallactic_angle_rad``.
    A 10 s D-config snapshot has one time and nearly identical antenna PA.
    """

    tb = table()
    tb.open(str(vis))
    try:
        times = np.asarray(tb.getcol("TIME"), dtype=np.float64)
    finally:
        tb.close()
    tb.open(str(vis) + "/ANTENNA")
    try:
        position = np.asarray(tb.getcol("POSITION"), dtype=np.float64)
    finally:
        tb.close()
    tb.open(str(vis) + "/FIELD")
    try:
        phase_dir = np.asarray(tb.getcol("PHASE_DIR"), dtype=np.float64)
    finally:
        tb.close()
    if position.ndim != 2 or 3 not in position.shape:
        raise RuntimeError(f"{vis} ANTENNA.POSITION has shape {position.shape}")
    if position.shape[0] == 3:
        position = position.T
    right_ascension = float(np.ravel(phase_dir[0])[0])
    declination = float(np.ravel(phase_dir[1])[0])
    time_mid = float(np.mean(times))
    equatorial = np.hypot(position[:, 0], position[:, 1])
    b_m = _WGS84_A_M * np.sqrt(1.0 - _WGS84_E2)
    e_prime2 = _WGS84_E2 / (1.0 - _WGS84_E2)
    theta = np.arctan2(position[:, 2] * _WGS84_A_M, equatorial * b_m)
    latitude = np.arctan2(
        position[:, 2] + e_prime2 * b_m * np.sin(theta) ** 3,
        equatorial - _WGS84_E2 * _WGS84_A_M * np.cos(theta) ** 3,
    )
    longitude = np.arctan2(position[:, 1], position[:, 0])
    julian_date = time_mid / 86400.0 + 2_400_000.5
    gmst = np.deg2rad(
        np.mod(280.46061837 + 360.98564736629 * (julian_date - 2_451_545.0), 360.0)
    )
    hour_angle = gmst + longitude - right_ascension
    numerator = np.cos(latitude) * np.sin(hour_angle)
    denominator = np.sin(latitude) * np.cos(declination) - np.cos(latitude) * np.sin(
        declination
    ) * np.cos(hour_angle)
    angles = np.arctan2(numerator, denominator)
    return float(np.mean(angles))


def _tclean_pb(vis: Path, imagename: Path, stokes: str) -> Path:
    for leftover in imagename.parent.glob(imagename.name + ".*"):
        _remove(leftover)
    tclean(
        vis=str(vis),
        imagename=str(imagename),
        gridder="awp2",
        vptable="",
        stokes=stokes,
        niter=0,
        aterm=True,
        psterm=False,
        wprojplanes=1,
        pblimit=PBLIMIT,
        imsize=IMAGE_SIZE,
        cell=CELL,
        phasecenter=f"J2000 {PHASE_CENTRE[0]}deg {PHASE_CENTRE[1]}deg",
        specmode="mfs",
        weighting="natural",
        savemodel="none",
        calcres=True,
        calcpsf=True,
        pbcor=False,
    )
    fits_path = imagename.with_name(imagename.name + ".pb.fits")
    exportfits(
        imagename=str(imagename) + ".pb",
        fitsimage=str(fits_path),
        overwrite=True,
        dropdeg=True,
    )
    return fits_path


def _casa_version() -> str:
    try:
        import casatools

        return ".".join(str(int(part)) for part in casatools.version())
    except Exception:
        return "unknown"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    planes = []
    telescope = None
    for frequency_mhz in FREQUENCIES_MHZ:
        for hourangle_key, hourangle in HOURANGLES.items():
            vis = _make_ms(frequency_mhz, hourangle_key, hourangle)
            telescope = _require_vla_telescope(vis)
            parallactic_angle_rad = _snapshot_parallactic_angle_rad(vis)
            for stokes in STOKES:
                name = f"{stokes}_{frequency_mhz}_{hourangle_key}"
                fits_path = _tclean_pb(vis, OUTPUT / name, stokes)
                stored = OUTPUT / f"{name}.pb.fits"
                if fits_path != stored:
                    shutil.copy2(fits_path, stored)
                planes.append(
                    {
                        "fits": stored.name,
                        "stokes": stokes,
                        "frequency_hz": frequency_mhz * 1.0e6,
                        "hourangle": hourangle,
                        "parallactic_angle_rad": parallactic_angle_rad,
                        "telescope": telescope,
                        "measurement_set": str(vis.relative_to(OUTPUT)),
                        "sha256": _sha256(stored),
                    }
                )

    manifest = {
        "schema_version": 1,
        "kind": "casa_awp2_power_oracle",
        "frozen": False,
        "casa_version": _casa_version(),
        "voltage_pattern": VOLTAGE_PATTERN,
        "telescope": telescope,
        "phase_centre_deg": list(PHASE_CENTRE),
        "imsize": IMAGE_SIZE,
        "cell": CELL,
        "pblimit": PBLIMIT,
        "tclean": {
            "gridder": "awp2",
            "vptable": "",
            "niter": 0,
            "aterm": True,
            "psterm": False,
            "wprojplanes": 1,
        },
        "planes": planes,
        "notes": (
            "Stage-1 CASA awp2 power beams from the default EVLA ray-traced "
            "A-term. Compare with the committed CASSBEAM evaluator. Not "
            "frozen until SL1MJax comparison accepts centre, FWHM, and "
            "residual power. Do not load CASSBEAM tables into CASA for "
            "this oracle."
        ),
    }
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(OUTPUT / "manifest.json")
    print(f"wrote {len(planes)} planes; frozen is false until comparison accepts")


if __name__ == "__main__":
    main()
