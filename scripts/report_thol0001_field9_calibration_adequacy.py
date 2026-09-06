"""Record complex coherent leftovers for the complete-track field-9 solve.

Operator identifiability already passed. This scores calibration adequacy
on the sealed holdouts, by antenna and baseline. No new time freedom.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

from sl1mjax.calibration_terms import parallactic_angle_rad
from sl1mjax.holography_calibration import REFERENCE_ANTENNA, write_json
from sl1mjax.holography_ms import _read_antennas, _tables
from sl1mjax.holography_reference_jones import (
    CALIBRATION_ADEQUACY_NOT_OPERATOR_NOTE,
    calibration_adequacy_report,
    connected_baseline_holdout_mask,
    deserialize_reference_jones,
    field9_scan_clusters,
    held_out_scan_cluster_masks,
    sample_holdout_mask,
)
from sl1mjax.polarization import Correlation, Receptor, pack_coherency

SCIENTIFIC_MS = Path(
    "/media/stephen/astro/vla/extracted/commissioning/work/THOL0001.lowerC.spw45.scientific.ms"
)
UNBLOCK = Path(
    "/media/stephen/astro/vla/extracted/commissioning/validation/scientific/full_jones_unblock"
)
HOLORASTER_FIELD = 10
NATIVE_CHANNEL = 32


def _pol():
    path = Path(__file__).with_name("report_thol0001_scientific_pol_gates.py")
    spec = importlib.util.spec_from_file_location("thol0001_pol_gates", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _scan_predecessors(measurement_set: Path) -> dict[int, int | None]:
    tables = _tables()
    with tables.table(str(measurement_set), readonly=True, ack=False) as main:
        time = np.asarray(main.getcol("TIME"), dtype=np.float64)
        scan = np.asarray(main.getcol("SCAN_NUMBER"), dtype=np.int32)
        field = np.asarray(main.getcol("FIELD_ID"), dtype=np.int32)
    first: dict[int, tuple[float, int]] = {}
    for instant, scan_id, field_id in zip(time, scan, field, strict=True):
        current = first.get(int(scan_id))
        if current is None or float(instant) < current[0]:
            first[int(scan_id)] = (float(instant), int(field_id))
    ordered = sorted(first, key=lambda scan_id: first[scan_id][0])
    predecessors: dict[int, int | None] = {}
    for index, scan_id in enumerate(ordered):
        predecessors[scan_id] = None if index == 0 else int(first[ordered[index - 1]][1])
    return predecessors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--measurement-set", type=Path, default=SCIENTIFIC_MS)
    parser.add_argument("--output-dir", type=Path, default=UNBLOCK)
    parser.add_argument("--spectral-window", type=int, default=4)
    arguments = parser.parse_args()
    if int(arguments.spectral_window) != 4:
        raise ValueError("SPW 5 stays closed")
    product = json.loads(
        (arguments.output_dir / "field9_stabilized_residual_jones.json").read_text()
    )
    jones = deserialize_reference_jones(product["jones"])
    pol = _pol()
    tables = _tables()
    ids, names, positions = _read_antennas(tables, arguments.measurement_set)
    names = np.asarray(names)
    positions = np.asarray(positions, dtype=np.float64)
    with tables.table(
        str(arguments.measurement_set / "FIELD"), readonly=True, ack=False
    ) as field_table:
        direction = np.asarray(field_table.getcell("PHASE_DIR", 9), dtype=np.float64).reshape(-1, 2)
        phase = (float(direction[0, 0]), float(direction[0, 1]))
    block = pol._load_field(arguments.measurement_set, 9, 4)
    inner, _ = pol._channel_masks(block["vis"].shape[1])
    native = int(product.get("native_channel") or NATIVE_CHANNEL)
    packed = pack_coherency(
        np.asarray(block["vis"], dtype=np.complex128)[:, native],
        (Correlation.RR, Correlation.RL, Correlation.LR, Correlation.LL),
        (Receptor.R, Receptor.L),
    )
    intensity = np.real(0.5 * (packed[:, 0, 0] + packed[:, 1, 1]))
    chi_ant = parallactic_angle_rad(block["time"], phase, positions)
    chi1 = chi_ant[np.arange(block["time"].size), block["antenna1"]]
    chi2 = chi_ant[np.arange(block["time"].size), block["antenna2"]]
    usable = (
        (block["antenna1"] != block["antenna2"])
        & np.isfinite(intensity)
        & (np.abs(intensity) > 1.0)
        & np.isfinite(packed[:, 0, 1])
    )
    predecessors = _scan_predecessors(arguments.measurement_set)
    clusters = field9_scan_clusters(
        block["scan"],
        block["time"],
        usable,
        predecessor_field=predecessors,
        holoraster_field=HOLORASTER_FIELD,
    )
    cluster_train, cluster_hold, cluster_report = held_out_scan_cluster_masks(
        block["scan"], usable, clusters
    )
    baseline_train = cluster_train & connected_baseline_holdout_mask(
        block["antenna1"], block["antenna2"], holdout_fraction=0.2
    )
    baseline_hold = cluster_train & ~baseline_train
    sample_train = baseline_train & sample_holdout_mask(baseline_train, holdout_fraction=0.1)
    sample_hold = baseline_train & ~sample_train
    kwargs = dict(
        stokes_i=intensity,
        chi1=chi1,
        chi2=chi2,
        residual_jones=jones,
        q_over_i=float(product["q_over_i"]),
        u_over_i=float(product["u_over_i"]),
        antenna_names=names,
    )
    payload = {
        "schema": "thol0001_field9_calibration_adequacy_v1",
        "operator_identifiable": True,
        "calibration_adequate": False,
        "gauge_antenna": REFERENCE_ANTENNA,
        "held_out_scans": cluster_report.get("held_out_scans"),
        "scan_cluster": calibration_adequacy_report(
            block["antenna1"], block["antenna2"], packed, row_mask=cluster_hold, **kwargs
        ),
        "baseline": calibration_adequacy_report(
            block["antenna1"], block["antenna2"], packed, row_mask=baseline_hold, **kwargs
        ),
        "sample": calibration_adequacy_report(
            block["antenna1"], block["antenna2"], packed, row_mask=sample_hold, **kwargs
        ),
        "notes": [CALIBRATION_ADEQUACY_NOT_OPERATOR_NOTE],
    }
    payload["calibration_adequate"] = all(
        payload[name]["calibration_adequate"] for name in ("scan_cluster", "baseline", "sample")
    )
    path = arguments.output_dir / "field9_calibration_adequacy.json"
    write_json(payload, path)
    print(path)
    print(
        "adequate",
        payload["calibration_adequate"],
        "cluster",
        payload["scan_cluster"]["median_abs_rl_over_i"],
        "coherent",
        (payload["scan_cluster"]["coherent_rl"] or {}).get("coherent_mean_abs"),
        "noise",
        payload["scan_cluster"]["averages_as_noise"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
