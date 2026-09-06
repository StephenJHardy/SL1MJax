"""Complete-track static field-9 residual Jones with cluster holdouts.

The gate is visibility-operator identifiability, not unique ε_p.
HOLORASTER visibilities stay out of the fit. No scan offsets, state
term, temporal GP, or SPW 5.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np

from sl1mjax.calibration_terms import parallactic_angle_rad
from sl1mjax.holography_calibration import REFERENCE_ANTENNA, write_json
from sl1mjax.holography_ms import _read_antennas, _tables
from sl1mjax.holography_reference_jones import (
    COMPLETE_FIELD9_TRACK_NOTE,
    ESTIMATOR_IDENTIFIABILITY_NOTE,
    NO_SCAN_OFFSETS_OR_TIME_FREEDOM_NOTE,
    PREDICTION_EQUIVALENT_BEAM_NOTE,
    align_residual_jones_maps,
    classify_stabilized_residual_jones,
    connected_baseline_holdout_mask,
    estimate_all_antenna_residual_jones,
    field9_scan_clusters,
    held_out_scan_cluster_masks,
    jones_parameter_jump,
    predicted_operator_difference,
    residual_holdout_report,
    sample_holdout_mask,
    serialize_reference_jones,
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
        if index == 0:
            predecessors[scan_id] = None
            continue
        predecessors[scan_id] = int(first[ordered[index - 1]][1])
    return predecessors


def _fit(block, packed, intensity, chi1, chi2, mask, gauge, *, q=0.0, u=0.0, fit_qu=True, n_iter=2):
    return estimate_all_antenna_residual_jones(
        block["antenna1"],
        block["antenna2"],
        packed,
        stokes_i=intensity,
        chi1=chi1,
        chi2=chi2,
        gauge_antenna_id=gauge,
        row_mask=mask,
        q_over_i=q,
        u_over_i=u,
        fit_qu=fit_qu,
        ridge=1.0,
        qu_ridge=1.0e-4,
        n_iter=int(n_iter),
    )


def _named(jones: dict[int, np.ndarray], names: np.ndarray) -> dict[str, float]:
    named = {}
    for ant, plane in jones.items():
        if int(ant) >= len(names):
            continue
        named[str(names[int(ant)])] = float(np.linalg.norm(np.asarray(plane) - np.eye(2)))
    return named


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--measurement-set", type=Path, default=SCIENTIFIC_MS)
    parser.add_argument("--output-dir", type=Path, default=UNBLOCK)
    parser.add_argument("--spectral-window", type=int, default=4)
    arguments = parser.parse_args()
    if int(arguments.spectral_window) != 4:
        raise ValueError("SPW 5 stays closed until the SPW-4 estimator is stabilized")
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    pol = _pol()
    tables = _tables()
    ids, names, positions = _read_antennas(tables, arguments.measurement_set)
    names = np.asarray(names)
    positions = np.asarray(positions, dtype=np.float64)
    name_to_id = {str(name): int(ant) for ant, name in zip(ids, names, strict=True)}
    with tables.table(
        str(arguments.measurement_set / "FIELD"), readonly=True, ack=False
    ) as field_table:
        direction = np.asarray(field_table.getcell("PHASE_DIR", 9), dtype=np.float64).reshape(-1, 2)
        phase = (float(direction[0, 0]), float(direction[0, 1]))
    block = pol._load_field(arguments.measurement_set, 9, 4)
    inner, _ = pol._channel_masks(block["vis"].shape[1])
    native = (
        int(NATIVE_CHANNEL)
        if inner[NATIVE_CHANNEL]
        else int(np.flatnonzero(inner)[int(np.sum(inner) // 2)])
    )
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
        & np.isfinite(packed[:, 1, 0])
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
    if cluster_report["status"] != "ok":
        raise ValueError("complete-track solve needs at least two field-9 visit clusters")
    baseline_train = cluster_train & connected_baseline_holdout_mask(
        block["antenna1"], block["antenna2"], holdout_fraction=0.2
    )
    baseline_hold = cluster_train & ~baseline_train
    sample_train = baseline_train & sample_holdout_mask(baseline_train, holdout_fraction=0.1)
    sample_hold = baseline_train & ~sample_train
    gauge = int(name_to_id[REFERENCE_ANTENNA])
    print(
        "clusters",
        len(clusters),
        "held_out_scans",
        cluster_report["held_out_scans"],
        "n_train",
        int(np.sum(sample_train)),
        flush=True,
    )
    fit = _fit(block, packed, intensity, chi1, chi2, sample_train, gauge)
    cluster_resid = residual_holdout_report(
        block["antenna1"],
        block["antenna2"],
        packed,
        stokes_i=intensity,
        chi1=chi1,
        chi2=chi2,
        residual_jones=fit["jones"],
        q_over_i=fit["q_over_i"],
        u_over_i=fit["u_over_i"],
        row_mask=cluster_hold,
        cluster_ids=block["scan"],
    )
    baseline_resid = residual_holdout_report(
        block["antenna1"],
        block["antenna2"],
        packed,
        stokes_i=intensity,
        chi1=chi1,
        chi2=chi2,
        residual_jones=fit["jones"],
        q_over_i=fit["q_over_i"],
        u_over_i=fit["u_over_i"],
        row_mask=baseline_hold,
        cluster_ids=np.minimum(block["antenna1"], block["antenna2"]) * 100
        + np.maximum(block["antenna1"], block["antenna2"]),
    )
    sample_resid = residual_holdout_report(
        block["antenna1"],
        block["antenna2"],
        packed,
        stokes_i=intensity,
        chi1=chi1,
        chi2=chi2,
        residual_jones=fit["jones"],
        q_over_i=fit["q_over_i"],
        u_over_i=fit["u_over_i"],
        row_mask=sample_hold,
    )
    family = [
        ("nominal", fit),
        (
            "qu_zero",
            _fit(block, packed, intensity, chi1, chi2, sample_train, gauge, fit_qu=False, n_iter=1),
        ),
        (
            "qu_fixed_nominal",
            _fit(
                block,
                packed,
                intensity,
                chi1,
                chi2,
                sample_train,
                gauge,
                q=fit["q_over_i"],
                u=fit["u_over_i"],
                fit_qu=False,
                n_iter=1,
            ),
        ),
    ]
    family_ops = []
    family_params = []
    family_cross = []
    for _name, other in family[1:]:
        aligned, _ = align_residual_jones_maps(fit["jones"], other["jones"], gauge_antenna_id=gauge)
        family_params.append(jones_parameter_jump(fit["jones"], aligned)["max_abs_delta"])
        family_ops.append(
            predicted_operator_difference(
                block["antenna1"],
                block["antenna2"],
                stokes_i=intensity,
                chi1=chi1,
                chi2=chi2,
                first_jones=fit["jones"],
                second_jones=aligned,
                q_over_i=0.5 * (fit["q_over_i"] + other["q_over_i"]),
                u_over_i=0.5 * (fit["u_over_i"] + other["u_over_i"]),
                row_mask=sample_train,
            )["median_abs_rl_over_i"]
        )
        self_rl = residual_holdout_report(
            block["antenna1"],
            block["antenna2"],
            packed,
            stokes_i=intensity,
            chi1=chi1,
            chi2=chi2,
            residual_jones=fit["jones"],
            q_over_i=fit["q_over_i"],
            u_over_i=fit["u_over_i"],
            row_mask=cluster_hold,
        ).get("median_abs_rl_over_i")
        cross_rl = residual_holdout_report(
            block["antenna1"],
            block["antenna2"],
            packed,
            stokes_i=intensity,
            chi1=chi1,
            chi2=chi2,
            residual_jones=aligned,
            q_over_i=other["q_over_i"],
            u_over_i=other["u_over_i"],
            row_mask=cluster_hold,
        ).get("median_abs_rl_over_i")
        family_cross.append(
            float(cross_rl or np.nan) <= 2.0 * max(float(self_rl or np.nan), 1.0e-4)
        )
    cluster_rl = float(cluster_resid.get("median_abs_rl_over_i") or np.nan)
    baseline_rl = float(baseline_resid.get("median_abs_rl_over_i") or np.nan)
    sample_rl = float(sample_resid.get("median_abs_rl_over_i") or np.nan)
    gate = classify_stabilized_residual_jones(
        scan_cluster_residual=cluster_rl,
        baseline_residual=baseline_rl,
        sample_residual=sample_rl,
        family_operator_max_rl=float(np.max(family_ops)) if family_ops else float("nan"),
        family_cross_equivalent=all(family_cross) if family_cross else False,
        n_held_out_scans=len(cluster_report["held_out_scans"]),
    )
    payload = {
        "schema": "thol0001_field9_stabilized_residual_jones_v1",
        "status": gate["status"],
        "blocking": gate["blocking"],
        "spectral_window_id": 4,
        "native_channel": native,
        "gauge_antenna": REFERENCE_ANTENNA,
        "holoraster_in_fit": False,
        "time_freedom": False,
        "scan_offsets": False,
        "q_over_i": fit["q_over_i"],
        "u_over_i": fit["u_over_i"],
        "median_abs_epsilon": fit["median_abs_epsilon"],
        "max_abs_epsilon": fit["max_abs_epsilon"],
        "graph_condition": fit.get("graph_condition"),
        "clusters": clusters,
        "scan_cluster_holdout": {**cluster_report, "residual": cluster_resid},
        "baseline_holdout": baseline_resid,
        "sample_holdout": sample_resid,
        "family": {
            "members": [name for name, _item in family],
            "max_aligned_abs_delta": float(np.max(family_params))
            if family_params
            else float("nan"),
            "max_operator_abs_rl": float(np.max(family_ops)) if family_ops else float("nan"),
            "cross_apply_equivalent": all(family_cross) if family_cross else False,
        },
        "jones": serialize_reference_jones(fit["jones"]),
        "abs_epsilon_by_name": _named(fit["jones"], names),
        "gate": gate,
        "notes": [
            ESTIMATOR_IDENTIFIABILITY_NOTE,
            COMPLETE_FIELD9_TRACK_NOTE,
            PREDICTION_EQUIVALENT_BEAM_NOTE,
            NO_SCAN_OFFSETS_OR_TIME_FREEDOM_NOTE,
        ],
    }
    path = arguments.output_dir / "field9_stabilized_residual_jones.json"
    write_json(payload, path)
    print(path)
    print(
        "cluster RL/I",
        cluster_rl,
        "baseline",
        baseline_rl,
        "sample",
        sample_rl,
        "family op",
        payload["family"]["max_operator_abs_rl"],
        "gate",
        gate["status"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
