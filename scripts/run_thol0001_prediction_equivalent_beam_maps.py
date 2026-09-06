"""Propagate prediction-equivalent R_p through holography after the beam gauge.

Holdouts are sealed before any map, support, or normalization estimate.
The on-axis gauge is the complete refactorization R'=R A, E'=A^{-1} E.
Unpinned maps are diagnostic only. Q=U=0 is an ablation. SPW 5 stays closed.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

from sl1mjax.beam_operator import unique_visibility_times
from sl1mjax.calibration_terms import parallactic_angle_rad as parallactic_angle
from sl1mjax.holography import HolographyObservation
from sl1mjax.holography_calibration import (
    HELD_OUT_REFERENCE_ANTENNA,
    REFERENCE_ANTENNA,
    write_json,
)
from sl1mjax.holography_full_jones import (
    classify_prediction_equivalent_beam_maps,
    holography_visibility_holdout_masks,
    offdiag_beam_map_difference,
    predict_moving_reference_from_beam,
    recover_holography_full_jones,
    refactor_on_axis_gauge,
    visibility_correlation_holdout_report,
)
from sl1mjax.holography_ms import _read_antennas, _tables, audit_holography_measurement_set
from sl1mjax.holography_reference_jones import (
    CALIBRATION_ADEQUACY_NOT_OPERATOR_NOTE,
    NO_SCAN_OFFSETS_OR_TIME_FREEDOM_NOTE,
    PREDICTION_EQUIVALENT_BEAM_NOTE,
    ZERO_QU_IS_ABLATION_NOTE,
    deserialize_reference_jones,
    estimate_all_antenna_residual_jones,
    serialize_reference_jones,
)
from sl1mjax.polarization import Correlation, Receptor, pack_coherency

SCIENTIFIC_MS = Path(
    "/media/stephen/astro/vla/extracted/commissioning/work/THOL0001.lowerC.spw45.scientific.ms"
)
UNBLOCK = Path(
    "/media/stephen/astro/vla/extracted/commissioning/validation/scientific/full_jones_unblock"
)
HELD_MOVER = "ea04"
NATIVE_CHANNEL = 32


def _diag():
    path = Path(__file__).with_name("run_thol0001_diagonal_recovery.py")
    spec = importlib.util.spec_from_file_location("thol0001_diagonal_recovery", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pol():
    path = Path(__file__).with_name("report_thol0001_scientific_pol_gates.py")
    spec = importlib.util.spec_from_file_location("thol0001_pol_gates", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _qu_extremes(
    output_dir: Path, nominal_q: float, nominal_u: float
) -> list[tuple[str, float, float, bool]]:
    points = [("nominal", float(nominal_q), float(nominal_u), False)]
    qu_path = output_dir / "field9_qu.json"
    if qu_path.exists():
        hold = json.loads(qu_path.read_text()).get("holdout") or {}
        if hold.get("q_over_i_p16") is not None and hold.get("u_over_i_p16") is not None:
            points.append(
                ("qu_p16", float(hold["q_over_i_p16"]), float(hold["u_over_i_p16"]), False)
            )
        if hold.get("q_over_i_p84") is not None and hold.get("u_over_i_p84") is not None:
            points.append(
                ("qu_p84", float(hold["q_over_i_p84"]), float(hold["u_over_i_p84"]), False)
            )
    if not any(name.startswith("qu_p") for name, _q, _u, _ablation in points):
        points.extend(
            [
                ("qu_p16", float(nominal_q) - 0.003, float(nominal_u) - 0.003, False),
                ("qu_p84", float(nominal_q) + 0.003, float(nominal_u) + 0.003, False),
            ]
        )
    points.append(("qu_zero", 0.0, 0.0, True))
    return points


def _field9_family(measurement_set: Path, output_dir: Path, names, positions, name_to_id):
    product = json.loads((output_dir / "field9_stabilized_residual_jones.json").read_text())
    nominal = deserialize_reference_jones(product["jones"])
    family = {
        "nominal": {
            "jones": nominal,
            "q_over_i": float(product["q_over_i"]),
            "u_over_i": float(product["u_over_i"]),
            "ablation": False,
        }
    }
    pol = _pol()
    with _tables().table(str(measurement_set / "FIELD"), readonly=True, ack=False) as field_table:
        direction = np.asarray(field_table.getcell("PHASE_DIR", 9), dtype=np.float64).reshape(-1, 2)
        phase = (float(direction[0, 0]), float(direction[0, 1]))
    block = pol._load_field(measurement_set, 9, 4)
    native = int(product.get("native_channel") or NATIVE_CHANNEL)
    packed = pack_coherency(
        np.asarray(block["vis"], dtype=np.complex128)[:, native],
        (Correlation.RR, Correlation.RL, Correlation.LR, Correlation.LL),
        (Receptor.R, Receptor.L),
    )
    intensity = np.real(0.5 * (packed[:, 0, 0] + packed[:, 1, 1]))
    chi_ant = parallactic_angle(block["time"], phase, positions)
    chi1 = chi_ant[np.arange(block["time"].size), block["antenna1"]]
    chi2 = chi_ant[np.arange(block["time"].size), block["antenna2"]]
    usable = (
        (block["antenna1"] != block["antenna2"])
        & np.isfinite(intensity)
        & (np.abs(intensity) > 1.0)
        & np.isfinite(packed[:, 0, 1])
    )
    from sl1mjax.holography_reference_jones import (
        connected_baseline_holdout_mask,
        field9_scan_clusters,
        held_out_scan_cluster_masks,
        sample_holdout_mask,
    )

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
    predecessors = {
        scan_id: None if index == 0 else int(first[ordered[index - 1]][1])
        for index, scan_id in enumerate(ordered)
    }
    clusters = field9_scan_clusters(
        block["scan"], block["time"], usable, predecessor_field=predecessors, holoraster_field=10
    )
    cluster_train, _hold, _report = held_out_scan_cluster_masks(block["scan"], usable, clusters)
    baseline_train = cluster_train & connected_baseline_holdout_mask(
        block["antenna1"], block["antenna2"], holdout_fraction=0.2
    )
    sample_train = baseline_train & sample_holdout_mask(baseline_train, holdout_fraction=0.1)
    gauge = int(name_to_id[REFERENCE_ANTENNA])
    for name, q_frac, u_frac, ablation in _qu_extremes(
        output_dir, float(product["q_over_i"]), float(product["u_over_i"])
    ):
        if name == "nominal":
            continue
        print("refit family", name, q_frac, u_frac, flush=True)
        fit = estimate_all_antenna_residual_jones(
            block["antenna1"],
            block["antenna2"],
            packed,
            stokes_i=intensity,
            chi1=chi1,
            chi2=chi2,
            gauge_antenna_id=gauge,
            row_mask=sample_train,
            q_over_i=q_frac,
            u_over_i=u_frac,
            fit_qu=False,
            ridge=1.0,
            n_iter=1,
        )
        family[name] = {
            "jones": fit["jones"],
            "q_over_i": q_frac,
            "u_over_i": u_frac,
            "ablation": ablation,
        }
    return family


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--measurement-set", type=Path, default=SCIENTIFIC_MS)
    parser.add_argument("--output-dir", type=Path, default=UNBLOCK)
    parser.add_argument("--spectral-window", type=int, default=4)
    parser.add_argument("--channel", type=int, default=NATIVE_CHANNEL)
    arguments = parser.parse_args()
    if int(arguments.spectral_window) != 4:
        raise ValueError("SPW 5 stays closed")
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    tables = _tables()
    ids, names, positions = _read_antennas(tables, arguments.measurement_set)
    names = np.asarray(names)
    positions = np.asarray(positions, dtype=np.float64)
    name_to_id = {str(name): int(ant) for ant, name in zip(ids, names, strict=True)}
    family_cache = arguments.output_dir / "prediction_equivalent_family.json"
    if family_cache.exists():
        print("loading cached field-9 family", flush=True)
        raw = json.loads(family_cache.read_text())
        family = {
            name: {
                "jones": deserialize_reference_jones(item["jones"]),
                "q_over_i": item["q_over_i"],
                "u_over_i": item["u_over_i"],
                "ablation": item["ablation"],
            }
            for name, item in raw.items()
        }
    else:
        print("building field-9 family", flush=True)
        family = _field9_family(
            arguments.measurement_set, arguments.output_dir, names, positions, name_to_id
        )
        write_json(
            {
                name: {
                    "jones": serialize_reference_jones(item["jones"]),
                    "q_over_i": item["q_over_i"],
                    "u_over_i": item["u_over_i"],
                    "ablation": item["ablation"],
                }
                for name, item in family.items()
            },
            family_cache,
        )
    diag = _diag()
    audit = audit_holography_measurement_set(arguments.measurement_set)
    ddid = diag._ddid_for_spw(tables, arguments.measurement_set, 4)
    block, source = diag._holoraster_channel_block(
        tables,
        arguments.measurement_set,
        data_desc_id=ddid,
        channel=arguments.channel,
        data_column="CORRECTED_DATA",
        spectral_window_id=4,
    )
    observation = HolographyObservation(
        block=block,
        pointing=diag._subset_pointing(audit.resolved, block.time_s),
        antenna_position_m=positions,
        calibration_state="casa_parang_true",
        phase_centre_rad=block.phase_centre_rad,
        source_name="3C147",
        source_coherency_visibility=source,
        selected_spw_id=4,
    )
    held_ref = int(name_to_id[HELD_OUT_REFERENCE_ANTENNA])
    held_move = int(name_to_id[HELD_MOVER])
    masks = holography_visibility_holdout_masks(
        observation, held_reference_id=held_ref, held_moving_id=held_move
    )
    print(
        "holdouts train",
        int(np.sum(masks["train"])),
        "hold",
        int(np.sum(masks["holdout"])),
        flush=True,
    )
    unique_times, _ = unique_visibility_times(block.time_s)
    chi = parallactic_angle(unique_times, observation.phase_centre_rad, positions)
    packed = pack_coherency(block.visibility, block.correlations, (Receptor.R, Receptor.L))
    intensity = (
        np.real(0.5 * (packed[:, 0, 0, 0] + packed[:, 0, 1, 1]))
        if packed.ndim == 4
        else np.real(0.5 * (packed[:, 0, 0] + packed[:, 1, 1]))
    )
    if packed.ndim == 4:
        packed_plane = packed[:, 0]
    else:
        packed_plane = packed
    members = {}
    scientific = [name for name, item in family.items() if not item["ablation"]]
    for name, item in family.items():
        print("recover", name, flush=True)
        unpinned = recover_holography_full_jones(
            observation,
            residual_jones=item["jones"],
            parallactic_angle_rad=chi,
            row_mask=masks["train"],
        )
        pinned, residual_p, origin = refactor_on_axis_gauge(unpinned, item["jones"])
        members[name] = {
            "unpinned": unpinned,
            "pinned": pinned,
            "residual": residual_p,
            "origin": {
                str(ant): {
                    "n": report["n"],
                    "used_nearest": report["used_nearest"],
                    "abs_A": float(np.linalg.norm(np.asarray(report["jones"]) - np.eye(2))),
                    "sigma_rl": float(np.asarray(report["sigma"])[0, 1]),
                }
                for ant, report in origin.items()
            },
            "ablation": item["ablation"],
            "q_over_i": item["q_over_i"],
            "u_over_i": item["u_over_i"],
        }
    nominal = members["nominal"]
    map_deltas = {}
    for name in scientific:
        if name == "nominal":
            continue
        map_deltas[name] = {
            "unpinned": offdiag_beam_map_difference(nominal["unpinned"], members[name]["unpinned"]),
            "pinned": offdiag_beam_map_difference(nominal["pinned"], members[name]["pinned"]),
        }
    hold_names = ("later_visits", "spatial", "held_reference", "held_moving")
    predictions = {}
    rng = np.random.default_rng(0)
    for hold_name in hold_names:
        mask = np.array(masks[hold_name], copy=True)
        selected = np.flatnonzero(mask)
        if selected.size > 2000:
            keep = np.zeros(mask.size, dtype=bool)
            keep[rng.choice(selected, size=2000, replace=False)] = True
            mask = keep
        if not bool(np.any(mask)):
            continue
        cluster = np.minimum(block.antenna1, block.antenna2) * 100 + np.maximum(
            block.antenna1, block.antenna2
        )
        pred_full = predict_moving_reference_from_beam(
            observation,
            residual_jones=nominal["residual"],
            artifact=nominal["pinned"],
            row_mask=mask,
            parallactic_angle_rad=chi,
        )
        pred_diag = predict_moving_reference_from_beam(
            observation,
            residual_jones=nominal["residual"],
            artifact=nominal["pinned"],
            row_mask=mask,
            parallactic_angle_rad=chi,
            diagonal_only=True,
        )
        full_report = visibility_correlation_holdout_report(
            packed_plane,
            pred_full[:, 0] if pred_full.ndim == 4 else pred_full,
            stokes_i=intensity,
            row_mask=mask,
            cluster_ids=cluster,
            antenna_groups=block.antenna1,
            baseline_groups=cluster,
        )
        diag_report = visibility_correlation_holdout_report(
            packed_plane,
            pred_diag[:, 0] if pred_diag.ndim == 4 else pred_diag,
            stokes_i=intensity,
            row_mask=mask,
            cluster_ids=cluster,
        )
        family_hold = {}
        for name in scientific:
            if name == "nominal":
                continue
            other = predict_moving_reference_from_beam(
                observation,
                residual_jones=members[name]["residual"],
                artifact=members[name]["pinned"],
                row_mask=mask,
                parallactic_angle_rad=chi,
            )
            delta = other - pred_full
            scale = np.maximum(np.abs(intensity[mask]), 1.0e-3)
            if delta.ndim == 4:
                rl = delta[mask, 0, 0, 1] / scale
            else:
                rl = delta[mask, 0, 1] / scale
            finite = np.abs(rl[np.isfinite(rl)])
            family_hold[name] = float(np.median(finite)) if finite.size else float("nan")
        predictions[hold_name] = {
            "n": int(np.sum(mask)),
            "n_finite_rl": int(full_report.get("rl", {}).get("n") or 0),
            "full_jones": full_report,
            "diagonal": diag_report,
            "family_abs_rl_over_i": family_hold,
        }
    pinned_deltas = [item["pinned"]["median_abs_rl"] for item in map_deltas.values()]
    unpinned_deltas = [item["unpinned"]["median_abs_rl"] for item in map_deltas.values()]
    family_pred = [
        value
        for hold in predictions.values()
        for value in hold["family_abs_rl_over_i"].values()
        if value is not None and np.isfinite(value)
    ]

    def _median(values) -> float:
        packed = np.asarray(
            [value for value in values if value is not None and np.isfinite(value)],
            dtype=np.float64,
        )
        return float(np.median(packed)) if packed.size else float("nan")

    full_rl = _median(
        hold["full_jones"].get("median_abs_rl_over_i") for hold in predictions.values()
    )
    diag_rl = _median(hold["diagonal"].get("median_abs_rl_over_i") for hold in predictions.values())
    full_rr = _median(
        hold["full_jones"].get("median_abs_rr_over_i") for hold in predictions.values()
    )
    failing = []
    for hold in predictions.values():
        by_ant = (hold["full_jones"].get("rl_by_antenna") or {}).get("by_group") or {}
        for name, stats in by_ant.items():
            if float(stats.get("mean_abs") or 0.0) >= 0.02:
                failing.append(name)
    gate = classify_prediction_equivalent_beam_maps(
        visibility_equivalent=True,
        median_abs_rl_map_delta=float(np.nanmax(unpinned_deltas))
        if unpinned_deltas
        else float("nan"),
        median_abs_rl_map_delta_after_pin=float(np.nanmax(pinned_deltas))
        if pinned_deltas
        else float("nan"),
        full_jones_rl_residual=float(full_rl),
        diagonal_rl_residual=float(diag_rl),
        full_jones_rr_residual=float(full_rr),
        predictions_agree_after_pin=not family_pred or float(np.nanmax(family_pred)) < 0.01,
        n_failing_antennas=len(set(failing)),
        n_antennas=len(name_to_id),
    )
    payload = {
        "schema": "thol0001_prediction_equivalent_beam_maps_v1",
        "status": gate["status"],
        "blocking": gate["blocking"],
        "outcome": gate["outcome"],
        "spectral_window_id": 4,
        "native_channel": int(arguments.channel),
        "held_out_reference": HELD_OUT_REFERENCE_ANTENNA,
        "held_out_mover": HELD_MOVER,
        "holdout_counts": {name: int(np.sum(mask)) for name, mask in masks.items()},
        "family": {
            name: {
                "q_over_i": item["q_over_i"],
                "u_over_i": item["u_over_i"],
                "ablation": item["ablation"],
                "origin": members[name]["origin"],
                "jones": serialize_reference_jones(family[name]["jones"]),
            }
            for name, item in family.items()
        },
        "map_deltas": map_deltas,
        "unpinned_maps_are_diagnostic": True,
        "predictions": predictions,
        "gate": gate,
        "notes": [
            PREDICTION_EQUIVALENT_BEAM_NOTE,
            ZERO_QU_IS_ABLATION_NOTE,
            CALIBRATION_ADEQUACY_NOT_OPERATOR_NOTE,
            NO_SCAN_OFFSETS_OR_TIME_FREEDOM_NOTE,
        ],
    }
    path = arguments.output_dir / "prediction_equivalent_beam_maps.json"
    write_json(payload, path)
    print(path)
    print(
        "outcome",
        gate["outcome"],
        "status",
        gate["status"],
        "pinned ΔRL",
        gate.get("maps_agree_after_pin"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
