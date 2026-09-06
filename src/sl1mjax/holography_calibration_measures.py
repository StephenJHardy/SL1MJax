"""Casacore-measures parallactic angle for CASA applycal.

Uses the Measurement Set TIME epoch, FIELD.PHASE_DIR and its measure
reference, and per-antenna ITRF positions. Does not use POINTING_OFFSET.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from sl1mjax.calibration_terms import (
    geocentric_latitude_rad,
    parallactic_angle_from_hadec_rad,
)

MEASURE_DIRECTION_CODES = {
    0: "J2000",
    1: "JMEAN",
    2: "JTRUE",
    3: "APP",
    4: "B1950",
    21: "ICRS",
}


def _measures() -> Any:
    try:
        from casacore.measures import measures
        from casacore.quanta import quantity
    except ImportError as exc:  # pragma: no cover - optional extra
        raise RuntimeError(
            "casacore measures are required for CASA parallactic-angle apply"
        ) from exc
    return measures, quantity


def read_ms_time_reference(measurement_set: Path) -> str:
    from sl1mjax.holography_ms import _tables

    with _tables().table(str(measurement_set), readonly=True, ack=False) as main:
        info = main.getcolkeywords("TIME")
    meas = info.get("MEASINFO", {})
    return str(meas.get("Ref", "UTC"))


def read_field_direction_frame(measurement_set: Path, field_id: int) -> str:
    from sl1mjax.holography_ms import _tables

    field = Path(measurement_set) / "FIELD"
    with _tables().table(str(field), readonly=True, ack=False) as table:
        if "PhaseDir_Ref" in table.colnames():
            code = int(table.getcell("PhaseDir_Ref", int(field_id)))
            return MEASURE_DIRECTION_CODES.get(code, "J2000")
    return "J2000"


def read_receptor_angle_rad(measurement_set: Path) -> NDArray[np.float64]:
    """Per-antenna mean FEED.RECEPTOR_ANGLE. CASA P adds this to geometric χ."""

    from sl1mjax.holography_ms import _tables

    feed = Path(measurement_set) / "FEED"
    with _tables().table(str(feed), readonly=True, ack=False) as table:
        antenna = np.asarray(table.getcol("ANTENNA_ID"), dtype=np.int32)
        angle = np.asarray(table.getcol("RECEPTOR_ANGLE"), dtype=np.float64)
    n_ant = int(np.max(antenna)) + 1
    output = np.zeros(n_ant, dtype=np.float64)
    for index in range(n_ant):
        selected = angle[antenna == index]
        if selected.size:
            output[index] = float(np.mean(selected))
    return output


def parallactic_angle_casacore_rad(
    time_s: np.ndarray,
    phase_centre_rad: tuple[float, float],
    antenna_position_m: np.ndarray,
    *,
    time_reference: str = "UTC",
    direction_frame: str = "J2000",
    receptor_angle_rad: np.ndarray | None = None,
) -> np.ndarray:
    """CASA applycal χ: measures HADEC plus geocentric latitude.

    ``receptor_angle_rad`` is added per antenna when FEED is non-zero.
    POINTING_OFFSET is ignored; calibration uses the field direction.
    """

    measures, quantity = _measures()
    times = np.asarray(time_s, dtype=np.float64)
    position = np.asarray(antenna_position_m, dtype=np.float64)
    latitude = geocentric_latitude_rad(position)
    right_ascension, declination = phase_centre_rad
    output = np.empty((times.size, position.shape[0]), dtype=np.float64)
    for row, time in enumerate(times):
        for antenna, xyz in enumerate(position):
            frame = measures()
            frame.doframe(frame.epoch(time_reference, quantity(float(time) / 86400.0, "d")))
            frame.doframe(
                frame.position(
                    "ITRF",
                    quantity(float(xyz[0]), "m"),
                    quantity(float(xyz[1]), "m"),
                    quantity(float(xyz[2]), "m"),
                )
            )
            source = frame.direction(
                direction_frame,
                quantity(float(right_ascension), "rad"),
                quantity(float(declination), "rad"),
            )
            hadec = frame.measure(source, "HADEC")
            hour_angle = float(hadec["m0"]["value"])
            apparent_dec = float(hadec["m1"]["value"])
            output[row, antenna] = float(
                parallactic_angle_from_hadec_rad(hour_angle, apparent_dec, latitude[antenna])
            )
    if receptor_angle_rad is not None:
        output = output + np.asarray(receptor_angle_rad, dtype=np.float64)[None, :]
    return output


def itrf_direction_casacore(
    time_s: np.ndarray,
    phase_centre_rad: tuple[float, float],
    antenna_position_m: np.ndarray,
    *,
    time_reference: str = "UTC",
    direction_frame: str = "J2000",
) -> np.ndarray:
    """Apparent field direction in ITRF, matching CASA antpos application.

    Needs an epoch and an ITRF position so measures can convert J2000 to ITRF.
    Per-antenna positions change the vector at a negligible level for millimetre
    offsets; the first finite antenna is used as the frame.
    """

    measures, quantity = _measures()
    times = np.asarray(time_s, dtype=np.float64)
    position = np.asarray(antenna_position_m, dtype=np.float64)
    if position.ndim != 2 or position.shape[1] != 3 or position.shape[0] == 0:
        raise ValueError("antenna_position_m must have shape (antenna, 3)")
    finite = np.flatnonzero(np.all(np.isfinite(position), axis=1))
    if finite.size == 0:
        raise ValueError("antenna_position_m has no finite ITRF row")
    xyz = position[int(finite[0])]
    right_ascension, declination = phase_centre_rad
    output = np.empty((times.size, 3), dtype=np.float64)
    for row, time in enumerate(times):
        frame = measures()
        frame.doframe(frame.epoch(time_reference, quantity(float(time) / 86400.0, "d")))
        frame.doframe(
            frame.position(
                "ITRF",
                quantity(float(xyz[0]), "m"),
                quantity(float(xyz[1]), "m"),
                quantity(float(xyz[2]), "m"),
            )
        )
        source = frame.direction(
            direction_frame,
            quantity(float(right_ascension), "rad"),
            quantity(float(declination), "rad"),
        )
        itrf = frame.measure(source, "ITRF")
        longitude = float(itrf["m0"]["value"])
        latitude = float(itrf["m1"]["value"])
        output[row] = (
            np.cos(latitude) * np.cos(longitude),
            np.cos(latitude) * np.sin(longitude),
            np.sin(latitude),
        )
    return output
