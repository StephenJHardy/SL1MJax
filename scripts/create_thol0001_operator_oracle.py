"""Build a tiny writable MS and inject correlation-space basis visibilities.

Uses casacore, not CASA. Each selected row is copied, then DATA is replaced
by one of RR/RL/LR/LL = 1. CASA applycal is a separate step.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from sl1mjax.holography_calibration import REFERENCE_ANTENNA, write_json
from sl1mjax.holography_calibration_golden import copy_golden_measurement_set
from sl1mjax.holography_calibration_oracle import BASIS_NAMES, injected_basis_visibilities
from sl1mjax.holography_ms import _tables


def _pick_rows(measurement_set: Path) -> dict[str, dict[str, object]]:
    tables = _tables()
    with tables.table(str(measurement_set / "ANTENNA"), readonly=True, ack=False) as antenna:
        names = [str(name) for name in antenna.getcol("NAME")]
    with tables.table(str(measurement_set), readonly=True, ack=False) as main:
        antenna1 = np.asarray(main.getcol("ANTENNA1"), dtype=np.int32)
        antenna2 = np.asarray(main.getcol("ANTENNA2"), dtype=np.int32)
        time_s = np.asarray(main.getcol("TIME"), dtype=np.float64)
        ddid = np.asarray(main.getcol("DATA_DESC_ID"), dtype=np.int32)
    ref = names.index(REFERENCE_ANTENNA)
    held = names.index("ea26") if "ea26" in names else ref
    global_x = names.index("ea03") if "ea03" in names else ref
    moving = names.index("ea04") if "ea04" in names else (ref + 1) % len(names)
    times = np.unique(time_s[ddid == 4])
    first_time = times[0]
    last_time = times[-1]
    picked = {}
    for label, ant_a, ant_b, time in (
        ("ref_held_t0", ref, held, first_time),
        ("ref_held_t1", ref, held, last_time),
        ("moving_held_t0", moving, held, first_time),
        ("globalx_ref_t0", global_x, ref, first_time),
    ):
        forward = np.flatnonzero(
            (ddid == 4) & (time_s == time) & (antenna1 == ant_a) & (antenna2 == ant_b)
        )
        reverse = np.flatnonzero(
            (ddid == 4) & (time_s == time) & (antenna1 == ant_b) & (antenna2 == ant_a)
        )
        if forward.size:
            picked[label] = {
                "row": int(forward[0]),
                "antenna1": int(ant_a),
                "antenna2": int(ant_b),
                "contains_reference": REFERENCE_ANTENNA in (names[ant_a], names[ant_b]),
                "contains_global_x": "ea03" in (names[ant_a], names[ant_b]),
            }
        if reverse.size:
            picked[f"{label}_swap"] = {
                "row": int(reverse[0]),
                "antenna1": int(ant_b),
                "antenna2": int(ant_a),
                "contains_reference": REFERENCE_ANTENNA in (names[ant_a], names[ant_b]),
                "contains_global_x": "ea03" in (names[ant_a], names[ant_b]),
            }
    return picked


def _inject(path: Path, basis: np.ndarray, channels: tuple[int, ...]) -> None:
    tables = _tables()
    with tables.table(str(path), readonly=False, ack=False) as main:
        data = np.asarray(main.getcol("DATA"))
        injected = np.zeros_like(data)
        for channel in channels:
            injected[:, int(channel), :] = basis
        main.putcol("DATA", injected)
        if "FLAG" in main.colnames():
            flag = np.zeros(data.shape, dtype=bool)
            main.putcol("FLAG", flag)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--channels", default="8,56")
    arguments = parser.parse_args()
    channels = tuple(int(item) for item in arguments.channels.split(",") if item.strip())
    picked = _pick_rows(arguments.source)
    if not picked:
        raise ValueError("no oracle rows were found")
    arguments.destination.mkdir(parents=True, exist_ok=True)
    cases = []
    for label, meta in picked.items():
        for index, name in enumerate(BASIS_NAMES):
            dest = arguments.destination / f"{label}_{name}.ms"
            if dest.exists():
                shutil.rmtree(dest)
            copy_golden_measurement_set(
                arguments.source, dest, np.array([meta["row"]], dtype=np.int64)
            )
            _inject(dest, injected_basis_visibilities()[index], channels)
            cases.append(
                {
                    "label": label,
                    "basis": name,
                    "row": meta["row"],
                    "antenna1": meta["antenna1"],
                    "antenna2": meta["antenna2"],
                    "contains_reference": meta["contains_reference"],
                    "contains_global_x": meta["contains_global_x"],
                    "ms": str(dest),
                }
            )
    write_json(
        {"channels": list(channels), "cases": cases, "labels": sorted(picked)},
        arguments.destination / "oracle_cases.json",
    )
    print(
        json.dumps(
            {"n_cases": len(cases), "labels": sorted(picked), "channels": list(channels)}, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
