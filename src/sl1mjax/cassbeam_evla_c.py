"""CASA EVLA C-band CASSBEAM parameters. The generic VLA template is an ablation.

CASA 6 ``EVLA_C`` stores ``feedpos = {+0.94300, -0.249152, 1.67640}`` and
negates ``feedpos[0]`` before the ray tracer. The matching CASSBEAM input
is therefore ``feed_x = -0.94300``. Taper is
``12.75 + 0.375 × (ν_GHz - 6.0)``. Do not overwrite the frozen generic
high-resolution artifact.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

CASA_EVLA_C_FEEDPOS_M = (0.94300, -0.249152, 1.67640)
CASA_EVLA_C_FEEDPOS_X_NEGATED = True
CASSBEAM_EVLA_C_FEED_M = (
    -CASA_EVLA_C_FEEDPOS_M[0],
    CASA_EVLA_C_FEEDPOS_M[1],
    CASA_EVLA_C_FEEDPOS_M[2],
)
GENERIC_VLA_FEED_M = (-0.6896837, -0.6896837, 1.67640)
CASA_EVLA_C_TAPER_REF_GHZ = 6.0
CASA_EVLA_C_TAPER_REF_DB = 12.75
CASA_EVLA_C_TAPER_SLOPE_DB_PER_GHZ = 0.375
GENERIC_VLA_TAPER_DB = 10.0


def evla_c_feedtaper_db(frequency_hz: float) -> float:
    ghz = float(frequency_hz) / 1.0e9
    return CASA_EVLA_C_TAPER_REF_DB + CASA_EVLA_C_TAPER_SLOPE_DB_PER_GHZ * (
        ghz - CASA_EVLA_C_TAPER_REF_GHZ
    )


def feed_ring_radius_m(feed_xyz_m: tuple[float, float, float]) -> float:
    return float(np.hypot(feed_xyz_m[0], feed_xyz_m[1]))


def feed_ring_angle_deg(feed_xyz_m: tuple[float, float, float]) -> float:
    return float(np.degrees(np.arctan2(feed_xyz_m[1], feed_xyz_m[0])))


def evla_c_versus_generic_feed() -> dict[str, float]:
    evla_r = feed_ring_radius_m(CASSBEAM_EVLA_C_FEED_M)
    generic_r = feed_ring_radius_m(GENERIC_VLA_FEED_M)
    return {
        "evla_radius_m": evla_r,
        "generic_radius_m": generic_r,
        "radius_difference_m": evla_r - generic_r,
        "evla_angle_deg": feed_ring_angle_deg(CASSBEAM_EVLA_C_FEED_M),
        "generic_angle_deg": feed_ring_angle_deg(GENERIC_VLA_FEED_M),
        "angle_difference_deg": feed_ring_angle_deg(CASSBEAM_EVLA_C_FEED_M)
        - feed_ring_angle_deg(GENERIC_VLA_FEED_M),
        "angle_separation_deg": abs(
            feed_ring_angle_deg(CASSBEAM_EVLA_C_FEED_M) - feed_ring_angle_deg(GENERIC_VLA_FEED_M)
        ),
        "taper_4564mhz_db": evla_c_feedtaper_db(4.564e9),
        "generic_taper_db": GENERIC_VLA_TAPER_DB,
    }


def evla_c_input_text(*, frequency_ghz: float, output_name: str) -> str:
    taper = evla_c_feedtaper_db(frequency_ghz * 1.0e9)
    feed_x, feed_y, feed_z = CASSBEAM_EVLA_C_FEED_M
    return "\n".join(
        [
            "% EVLA C-band CASSBEAM input from CASA 6 EVLA_C BeamCalc defaults.",
            "% CASA stores feedpos[0] > 0 and negates it before the ray tracer.",
            "% The generic packaged VLA template remains the ablation, not this file.",
            "% Do not overwrite the frozen generic high-resolution artifact.",
            "name = EVLA",
            "sub_h = 8.47852",
            f"feed_x = {feed_x:.5f}",
            f"feed_y = {feed_y:.6f}",
            f"feed_z = {feed_z:.5f}",
            "geom = vla_geom",
            f"feedtaper = {taper:.4f}",
            "feedthetamax = 9.26",
            "legwidth = 0.27",
            "legfoot = 7.55",
            "legapex = 10.93876",
            "hole_radius = 2.0",
            "roughness = 0.00035",
            f"freq = {frequency_ghz:.3f}",
            "gridsize = 64",
            "pixelsperbeam = 6",
            f"out = {output_name}",
            "compute = jp",
            "Trec = 10",
            "",
        ]
    )


def write_evla_c_input(path: Path, *, frequency_ghz: float, output_name: str) -> Path:
    path = Path(path)
    path.write_text(
        evla_c_input_text(frequency_ghz=frequency_ghz, output_name=output_name),
        encoding="utf-8",
    )
    return path
