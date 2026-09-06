"""Estimate residual on-axis reference Jones from HOLORASTER ref--ref rows.

    V_rr' = E_r(0) S E_{r'}(0)^H

The seven reference antennas are not treated as exact identity. Corrections
are ridge-shrunk toward identity. ea02 is the gauge pin. ea26 is held out.
A later-time split is a second holdout. SPW 5 is not opened.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np

from sl1mjax.holography_calibration import (
    HELD_OUT_REFERENCE_ANTENNA,
    REFERENCE_ANTENNA,
    write_json,
)
from sl1mjax.holography_diagonal import THOL0001_REFERENCE_ANTENNA_NAMES
from sl1mjax.holography_ms import _read_antennas, _tables
from sl1mjax.holography_reference_jones import (
    IDENTITY_REFERENCE_JONES_BLOCKS_OFFDIAG_NOTE,
    estimate_residual_reference_jones,
    serialize_reference_jones,
)
from sl1mjax.polarization import Receptor, pack_coherency

SCIENTIFIC_MS = Path(
    "/media/stephen/astro/vla/extracted/commissioning/work/THOL0001.lowerC.spw45.scientific.ms"
)
UNBLOCK = Path(
    "/media/stephen/astro/vla/extracted/commissioning/validation/scientific/full_jones_unblock"
)


def _diag():
    path = Path(__file__).with_name("run_thol0001_diagonal_recovery.py")
    spec = importlib.util.spec_from_file_location("thol0001_diagonal_recovery", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--measurement-set", type=Path, default=SCIENTIFIC_MS)
    parser.add_argument("--output-dir", type=Path, default=UNBLOCK)
    parser.add_argument("--channel", type=int, default=32)
    parser.add_argument("--spectral-window", type=int, default=4)
    arguments = parser.parse_args()
    if int(arguments.spectral_window) != 4:
        raise ValueError("SPW 5 stays closed until the SPW-4 reference Jones is understood")
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    diag = _diag()
    tables = _tables()
    ids, names, _positions = _read_antennas(tables, arguments.measurement_set)
    name_to_id = {str(name): int(ant) for ant, name in zip(ids, names, strict=True)}
    ref_ids = np.array(
        [name_to_id[name] for name in THOL0001_REFERENCE_ANTENNA_NAMES],
        dtype=np.int32,
    )
    ddid = diag._ddid_for_spw(tables, arguments.measurement_set, arguments.spectral_window)
    block, source = diag._holoraster_channel_block(
        tables,
        arguments.measurement_set,
        data_desc_id=ddid,
        channel=arguments.channel,
        data_column="CORRECTED_DATA",
        spectral_window_id=arguments.spectral_window,
    )
    both_ref = np.isin(block.antenna1, ref_ids) & np.isin(block.antenna2, ref_ids)
    both_ref &= block.antenna1 != block.antenna2
    flag = np.asarray(block.flag[:, 0], dtype=bool)
    usable = both_ref & ~np.any(flag, axis=1)
    packed = pack_coherency(block.visibility[:, 0], block.correlations, (Receptor.R, Receptor.L))
    source_plane = np.asarray(source[:, 0], dtype=np.complex128)
    gauge = int(name_to_id[REFERENCE_ANTENNA])
    hold_ant = int(name_to_id[HELD_OUT_REFERENCE_ANTENNA])
    times = np.asarray(block.time_s, dtype=np.float64)
    later = (
        times >= float(np.median(times[usable]))
        if np.any(usable)
        else np.zeros(times.size, dtype=bool)
    )
    antenna_hold = estimate_residual_reference_jones(
        block.antenna1[usable],
        block.antenna2[usable],
        packed[usable],
        source=source_plane[usable],
        antenna_ids=ref_ids,
        gauge_antenna_id=gauge,
        holdout_antenna_id=hold_ant,
        ridge=1.0,
    )
    time_hold = estimate_residual_reference_jones(
        block.antenna1[usable],
        block.antenna2[usable],
        packed[usable],
        source=source_plane[usable],
        antenna_ids=ref_ids,
        gauge_antenna_id=gauge,
        row_mask=~later[usable],
        ridge=1.0,
    )
    payload = {
        "schema": "thol0001_residual_reference_jones_v1",
        "status": "warn",
        "measurement_set": str(arguments.measurement_set),
        "spectral_window_id": int(arguments.spectral_window),
        "channel": int(arguments.channel),
        "n_reference_reference_rows": int(np.sum(both_ref)),
        "n_usable": int(np.sum(usable)),
        "reference_antennas": list(THOL0001_REFERENCE_ANTENNA_NAMES),
        "gauge_antenna": REFERENCE_ANTENNA,
        "held_out_reference": HELD_OUT_REFERENCE_ANTENNA,
        "jones": serialize_reference_jones(antenna_hold["jones"]),
        "median_abs_epsilon": antenna_hold["median_abs_epsilon"],
        "max_abs_epsilon": antenna_hold["max_abs_epsilon"],
        "holdout": antenna_hold["holdout"],
        "later_time_holdout": {
            "n_train": time_hold["n_train"],
            "median_abs_epsilon": time_hold["median_abs_epsilon"],
        },
        "identity_forbidden_for_crosshand": True,
        "notes": [IDENTITY_REFERENCE_JONES_BLOCKS_OFFDIAG_NOTE],
    }
    path = arguments.output_dir / "residual_reference_jones.json"
    write_json(payload, path)
    print(path)
    print("n_usable", payload["n_usable"], "median_|ε|", payload["median_abs_epsilon"])
    print("holdout", payload["holdout"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
