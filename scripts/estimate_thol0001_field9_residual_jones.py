"""Field-9 residual Jones controls, then one conservative time-smooth transfer.

    R_p(t) = P(χ_p)^H (I + ε_{p,0} + δε_p(t)) P(χ_p)
    E_p(0) = I

Controls run before time freedom. Drift ridge is chosen on inner blocked-time
folds of the earlier-scan training half only. The later-time holdout is
evaluated once. SPW 5 is not opened.
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
    ALL_ANTENNA_RESIDUAL_JONES_NOTE,
    CASA_FIELD9_NOT_END_TO_END_HOLDOUT_NOTE,
    CHANNEL_AVERAGING_CAN_FAKE_TIME_NOTE,
    CONNECTED_HOLDOUT_NOTE,
    ON_AXIS_BEAM_IDENTITY_NOTE,
    RESIDUAL_JONES_P_CONVENTION_NOTE,
    SCAN_STATE_MAY_NEED_OFFSETS_NOTE,
    THREE_C147_QU_IS_NUISANCE_NOTE,
    antenna_graph_is_connected,
    classify_field9_time_transfer,
    connected_baseline_holdout_mask,
    estimate_all_antenna_residual_jones,
    gauge_invariance_abs,
    identical_channel_support,
    injected_feed_leakage_rotation,
    residual_holdout_report,
    scan_state_jump_report,
    select_drift_ridge_by_inner_folds,
    serialize_reference_jones,
)
from sl1mjax.polarization import Correlation, Receptor, pack_coherency

SCIENTIFIC_MS = Path(
    "/media/stephen/astro/vla/extracted/commissioning/work/THOL0001.lowerC.spw45.scientific.ms"
)
UNBLOCK = Path(
    "/media/stephen/astro/vla/extracted/commissioning/validation/scientific/full_jones_unblock"
)
IDENTITY_PRIOR_ANTENNA = "ea11"
FOCUS_ANTENNAS = ("ea04", "ea11", "ea21")
HOLORASTER_FIELD = 10
NATIVE_CHANNEL = 32
PREVIOUS_LATER_COHERENT = 0.0099


def _pol():
    path = Path(__file__).with_name("report_thol0001_scientific_pol_gates.py")
    spec = importlib.util.spec_from_file_location("thol0001_pol_gates", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _scan_predecessors(measurement_set: Path) -> dict[int, dict[str, int] | None]:
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
    predecessors: dict[int, dict[str, int] | None] = {}
    for index, scan_id in enumerate(ordered):
        if index == 0:
            predecessors[scan_id] = None
            continue
        previous = ordered[index - 1]
        predecessors[scan_id] = {
            "scan": int(previous),
            "field": int(first[previous][1]),
        }
    return predecessors


def _pack_channel(vis: np.ndarray, channel: int):
    return pack_coherency(
        vis[:, int(channel)],
        (Correlation.RR, Correlation.RL, Correlation.LR, Correlation.LL),
        (Receptor.R, Receptor.L),
    )


def _named_epsilon(jones: dict[int, np.ndarray], names: np.ndarray) -> dict[str, dict]:
    named = {}
    for ant, plane in jones.items():
        if int(ant) >= len(names):
            continue
        named[str(names[int(ant)])] = {
            "real": np.asarray(plane).real.tolist(),
            "imag": np.asarray(plane).imag.tolist(),
            "abs_epsilon": float(np.linalg.norm(np.asarray(plane) - np.eye(2))),
            "abs_rl": float(np.abs(np.asarray(plane)[0, 1])),
            "abs_lr": float(np.abs(np.asarray(plane)[1, 0])),
        }
    return named


def _focus_coherent(block, packed, intensity, chi1, chi2, name_to_id, mask, fit):
    scores = {}
    for name in FOCUS_ANTENNAS:
        antenna = int(name_to_id[name])
        selected = mask & ((block["antenna1"] == antenna) | (block["antenna2"] == antenna))
        report = residual_holdout_report(
            block["antenna1"],
            block["antenna2"],
            packed,
            stokes_i=intensity,
            chi1=chi1,
            chi2=chi2,
            residual_jones=fit["jones"],
            q_over_i=fit["q_over_i"],
            u_over_i=fit["u_over_i"],
            row_mask=selected,
            time_s=block["time"],
            drift_offdiag=fit.get("drift_offdiag"),
            time_origin_s=fit.get("time_origin_s"),
            time_scale_s=fit.get("time_scale_s"),
        )
        coherent = report.get("coherent_rl") or {}
        scores[name] = float(coherent.get("coherent_mean_abs") or np.nan)
    return scores


def _plot_scan_state(path: Path, per_scan: dict, names: np.ndarray, name_to_id: dict) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 6.5), sharex=True)
    colors = {"ea04": "C0", "ea11": "C1", "ea21": "C2", "ea25": "C3"}
    for name, color in colors.items():
        if name not in name_to_id:
            continue
        antenna = int(name_to_id[name])
        xs, abs_rl, after_raster = [], [], []
        for scan_id, payload in sorted(per_scan.items(), key=lambda item: int(item[0])):
            plane = (payload.get("jones") or {}).get(antenna)
            if plane is None:
                continue
            xs.append(int(scan_id))
            abs_rl.append(float(np.abs(np.asarray(plane)[0, 1])))
            pred = payload.get("predecessor") or {}
            after_raster.append(int(pred.get("field") or -1) == HOLORASTER_FIELD)
        if not xs:
            continue
        axes[0].plot(xs, abs_rl, "o-", color=color, label=name)
        raster_x = [x for x, flag in zip(xs, after_raster, strict=True) if flag]
        raster_y = [y for y, flag in zip(abs_rl, after_raster, strict=True) if flag]
        if raster_x:
            axes[0].scatter(raster_x, raster_y, s=80, facecolors="none", edgecolors=color)
        axes[1].plot(xs, after_raster, "o-", color=color, label=name)
    axes[0].set_ylabel(r"$|\varepsilon_{RL}|$")
    axes[1].set_ylabel("preceded by HOLORASTER")
    axes[1].set_xlabel("field-9 scan")
    axes[0].legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--measurement-set", type=Path, default=SCIENTIFIC_MS)
    parser.add_argument("--output-dir", type=Path, default=UNBLOCK)
    parser.add_argument("--spectral-window", type=int, default=4)
    parser.add_argument("--skip-time-smooth", action="store_true")
    parser.add_argument("--n-iter", type=int, default=2)
    arguments = parser.parse_args()
    if int(arguments.spectral_window) != 4:
        raise ValueError("SPW 5 stays closed until the SPW-4 residual Jones transfers")
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    pol = _pol()
    tables = _tables()
    ids, names, positions = _read_antennas(tables, arguments.measurement_set)
    positions = np.asarray(positions, dtype=np.float64)
    name_to_id = {str(name): int(ant) for ant, name in zip(ids, names, strict=True)}
    with tables.table(
        str(arguments.measurement_set / "FIELD"), readonly=True, ack=False
    ) as field_table:
        direction = np.asarray(field_table.getcell("PHASE_DIR", 9), dtype=np.float64).reshape(-1, 2)
        phase = (float(direction[0, 0]), float(direction[0, 1]))
    p_convention = injected_feed_leakage_rotation(np.linspace(-0.5, 0.5, 11))
    print("p_convention", p_convention["status"], flush=True)
    block = pol._load_field(arguments.measurement_set, 9, 4)
    inner, _ = pol._channel_masks(block["vis"].shape[1])
    vis = np.asarray(block["vis"], dtype=np.complex128)
    finite_inner = np.isfinite(vis[:, inner]).all(axis=-1)
    _keep, channel_support = identical_channel_support(finite_inner)
    native_idx = int(NATIVE_CHANNEL)
    if native_idx >= vis.shape[1] or not bool(inner[native_idx]):
        native_idx = int(np.flatnonzero(inner)[int(np.sum(inner) // 2)])
    print(
        "native_channel",
        native_idx,
        "time_dependent_support",
        channel_support["time_dependent_support"],
        "n_kept",
        channel_support["n_kept"],
        flush=True,
    )
    packed = _pack_channel(vis, native_idx)
    packed_avg = pack_coherency(
        np.nanmean(vis[:, inner], axis=1),
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
        & np.isfinite(packed[:, 0, 0])
        & np.isfinite(packed[:, 1, 1])
        & np.isfinite(packed[:, 0, 1])
        & np.isfinite(packed[:, 1, 0])
    )
    scans = np.unique(block["scan"][usable])
    later = np.isin(block["scan"], scans[max(1, scans.size // 2) :])
    train_time = usable & ~later
    hold_time = usable & later
    if not antenna_graph_is_connected(block["antenna1"], block["antenna2"], row_mask=train_time):
        raise ValueError("later-scan holdout disconnects the antenna graph")
    gauge = int(name_to_id[REFERENCE_ANTENNA])
    predecessors = _scan_predecessors(arguments.measurement_set)
    static = estimate_all_antenna_residual_jones(
        block["antenna1"],
        block["antenna2"],
        packed,
        stokes_i=intensity,
        chi1=chi1,
        chi2=chi2,
        gauge_antenna_id=gauge,
        row_mask=train_time,
        ridge=1.0,
        qu_ridge=1.0e-4,
        n_iter=int(arguments.n_iter),
        time_s=block["time"],
    )
    time_hold = residual_holdout_report(
        block["antenna1"],
        block["antenna2"],
        packed,
        stokes_i=intensity,
        chi1=chi1,
        chi2=chi2,
        residual_jones=static["jones"],
        q_over_i=static["q_over_i"],
        u_over_i=static["u_over_i"],
        row_mask=hold_time,
        cluster_ids=block["scan"],
        time_s=block["time"],
    )
    static_focus = _focus_coherent(
        block, packed, intensity, chi1, chi2, name_to_id, hold_time, static
    )
    print(
        "static later coherent",
        (time_hold.get("coherent_rl") or {}).get("coherent_mean_abs"),
        "averages_as_noise",
        (time_hold.get("coherent_rl") or {}).get("averages_as_noise"),
        flush=True,
    )
    averaging_compare = None
    if not channel_support["time_dependent_support"]:
        intensity_avg = np.real(0.5 * (packed_avg[:, 0, 0] + packed_avg[:, 1, 1]))
        usable_avg = usable & np.isfinite(intensity_avg) & (np.abs(intensity_avg) > 1.0)
        avg_fit = estimate_all_antenna_residual_jones(
            block["antenna1"],
            block["antenna2"],
            packed_avg,
            stokes_i=intensity_avg,
            chi1=chi1,
            chi2=chi2,
            gauge_antenna_id=gauge,
            row_mask=usable_avg & ~later,
            ridge=1.0,
            n_iter=int(arguments.n_iter),
        )
        deltas = [
            float(
                np.linalg.norm(np.asarray(static["jones"][ant]) - np.asarray(avg_fit["jones"][ant]))
            )
            for ant in static["jones"]
            if ant in avg_fit["jones"]
        ]
        averaging_compare = {
            "max_abs_epsilon_delta": float(np.max(deltas)) if deltas else float("nan"),
            "median_abs_epsilon_delta": float(np.median(deltas)) if deltas else float("nan"),
        }
    per_scan = {}
    for scan_id in scans:
        scan_mask = usable & (block["scan"] == int(scan_id))
        if not antenna_graph_is_connected(block["antenna1"], block["antenna2"], row_mask=scan_mask):
            continue
        try:
            scan_fit = estimate_all_antenna_residual_jones(
                block["antenna1"],
                block["antenna2"],
                packed,
                stokes_i=intensity,
                chi1=chi1,
                chi2=chi2,
                gauge_antenna_id=gauge,
                row_mask=scan_mask,
                ridge=1.0,
                n_iter=1,
            )
        except ValueError:
            continue
        pred = predecessors.get(int(scan_id))
        per_scan[str(int(scan_id))] = {
            "n": int(np.sum(scan_mask)),
            "q_over_i": scan_fit["q_over_i"],
            "u_over_i": scan_fit["u_over_i"],
            "median_abs_epsilon": scan_fit["median_abs_epsilon"],
            "predecessor": pred,
            "preceded_by_holoraster": bool(
                pred is not None and int(pred["field"]) == HOLORASTER_FIELD
            ),
            "jones": scan_fit["jones"],
            "jones_by_name": _named_epsilon(scan_fit["jones"], names),
        }
    jump = scan_state_jump_report(
        {scan_id: payload["jones"] for scan_id, payload in per_scan.items()}
    )
    plot_path = arguments.output_dir / "field9_scan_state_epsilon.png"
    _plot_scan_state(plot_path, per_scan, names, name_to_id)
    print(
        "scan_state n",
        len(per_scan),
        "jumps",
        jump["n_jump"],
        "piecewise",
        jump["piecewise_state_term_indicated"],
        flush=True,
    )
    baseline_train = usable & connected_baseline_holdout_mask(
        block["antenna1"], block["antenna2"], holdout_fraction=0.2
    )
    baseline_hold = usable & ~baseline_train
    baseline_fit = estimate_all_antenna_residual_jones(
        block["antenna1"],
        block["antenna2"],
        packed,
        stokes_i=intensity,
        chi1=chi1,
        chi2=chi2,
        gauge_antenna_id=gauge,
        row_mask=baseline_train,
        ridge=1.0,
        n_iter=int(arguments.n_iter),
    )
    baseline_report = residual_holdout_report(
        block["antenna1"],
        block["antenna2"],
        packed,
        stokes_i=intensity,
        chi1=chi1,
        chi2=chi2,
        residual_jones=baseline_fit["jones"],
        q_over_i=baseline_fit["q_over_i"],
        u_over_i=baseline_fit["u_over_i"],
        row_mask=baseline_hold,
        cluster_ids=np.minimum(block["antenna1"], block["antenna2"]) * 100
        + np.maximum(block["antenna1"], block["antenna2"]),
    )
    prior_id = int(name_to_id[IDENTITY_PRIOR_ANTENNA])
    prior_train = usable & (block["antenna1"] != prior_id) & (block["antenna2"] != prior_id)
    prior_ids = np.array(
        [
            ant
            for ant in np.unique(np.concatenate([block["antenna1"], block["antenna2"]]))
            if int(ant) != prior_id
        ],
        dtype=np.int32,
    )
    prior_fit = estimate_all_antenna_residual_jones(
        block["antenna1"],
        block["antenna2"],
        packed,
        stokes_i=intensity,
        chi1=chi1,
        chi2=chi2,
        antenna_ids=prior_ids,
        gauge_antenna_id=gauge,
        row_mask=prior_train,
        ridge=1.0,
        n_iter=int(arguments.n_iter),
    )
    prior_jones = dict(prior_fit["jones"])
    prior_jones[prior_id] = np.eye(2, dtype=np.complex128)
    prior_report = residual_holdout_report(
        block["antenna1"],
        block["antenna2"],
        packed,
        stokes_i=intensity,
        chi1=chi1,
        chi2=chi2,
        residual_jones=prior_jones,
        q_over_i=prior_fit["q_over_i"],
        u_over_i=prior_fit["u_over_i"],
        row_mask=usable & ~prior_train,
    )
    gauge_abs = gauge_invariance_abs(
        block["antenna1"][hold_time],
        block["antenna2"][hold_time],
        stokes_i=intensity[hold_time],
        chi1=chi1[hold_time],
        chi2=chi2[hold_time],
        residual_jones=static["jones"],
        q_over_i=static["q_over_i"],
        u_over_i=static["u_over_i"],
    )
    controls = {
        "schema": "thol0001_field9_residual_jones_controls_v1",
        "spectral_window_id": 4,
        "native_channel": native_idx,
        "p_convention": p_convention,
        "channel_support": channel_support,
        "averaging_compare": averaging_compare,
        "scan_state": {
            "n_scan": len(per_scan),
            "jump": jump,
            "plot": str(plot_path) if plot_path.exists() else None,
            "preceded_by_holoraster": {
                scan_id: payload["preceded_by_holoraster"] for scan_id, payload in per_scan.items()
            },
            "jones_by_name": {
                scan_id: payload["jones_by_name"] for scan_id, payload in per_scan.items()
            },
        },
        "on_axis_beam_identity": ON_AXIS_BEAM_IDENTITY_NOTE,
        "notes": [
            RESIDUAL_JONES_P_CONVENTION_NOTE,
            CHANNEL_AVERAGING_CAN_FAKE_TIME_NOTE,
            SCAN_STATE_MAY_NEED_OFFSETS_NOTE,
            ON_AXIS_BEAM_IDENTITY_NOTE,
            CASA_FIELD9_NOT_END_TO_END_HOLDOUT_NOTE,
        ],
    }
    write_json(controls, arguments.output_dir / "field9_residual_jones_controls.json")
    static_payload = {
        "schema": "thol0001_field9_all_antenna_residual_jones_v2",
        "status": "warn",
        "measurement_set": str(arguments.measurement_set),
        "spectral_window_id": 4,
        "native_channel": native_idx,
        "n_usable": int(np.sum(usable)),
        "n_train_time": int(np.sum(train_time)),
        "gauge_antenna": REFERENCE_ANTENNA,
        "identity_prior_antenna": IDENTITY_PRIOR_ANTENNA,
        "q_over_i": static["q_over_i"],
        "u_over_i": static["u_over_i"],
        "frac_pol": static["frac_pol"],
        "median_abs_epsilon": static["median_abs_epsilon"],
        "max_abs_epsilon": static["max_abs_epsilon"],
        "residual_jones_sky_frame": static["residual_jones_sky_frame"],
        "jones": serialize_reference_jones(static["jones"]),
        "jones_by_name": _named_epsilon(static["jones"], names),
        "time_holdout": time_hold,
        "baseline_holdout": baseline_report,
        "identity_prior_holdout": {
            "antenna": IDENTITY_PRIOR_ANTENNA,
            **prior_report,
            "separate_gate": True,
        },
        "focus_later_coherent": static_focus,
        "qu_nuisance": {
            "fitted": {"q_over_i": static["q_over_i"], "u_over_i": static["u_over_i"]},
            "represented_as_exact_zero": False,
        },
        "gauge_invariance_abs": gauge_abs,
        "graph_connected_time": bool(time_hold.get("graph_connected")),
        "graph_connected_baseline": bool(baseline_report.get("graph_connected")),
        "notes": [
            ALL_ANTENNA_RESIDUAL_JONES_NOTE,
            THREE_C147_QU_IS_NUISANCE_NOTE,
            CONNECTED_HOLDOUT_NOTE,
            RESIDUAL_JONES_P_CONVENTION_NOTE,
            CASA_FIELD9_NOT_END_TO_END_HOLDOUT_NOTE,
        ],
    }
    write_json(static_payload, arguments.output_dir / "field9_all_antenna_residual_jones.json")
    transfer = {
        "schema": "thol0001_field9_time_smooth_residual_jones_v1",
        "status": "not_run",
        "spectral_window_id": 4,
        "native_channel": native_idx,
        "piecewise_state_term_indicated": jump["piecewise_state_term_indicated"],
    }
    if not arguments.skip_time_smooth:
        selection = select_drift_ridge_by_inner_folds(
            block["antenna1"],
            block["antenna2"],
            packed,
            stokes_i=intensity,
            chi1=chi1,
            chi2=chi2,
            time_s=block["time"],
            train_mask=train_time,
            gauge_antenna_id=gauge,
            n_iter=int(arguments.n_iter),
        )
        chosen = float(selection["chosen_drift_ridge"])
        print(
            "inner-fold chosen_drift_ridge",
            chosen,
            "qu_fold_spread",
            selection.get("qu_fold_spread"),
            flush=True,
        )
        smooth = estimate_all_antenna_residual_jones(
            block["antenna1"],
            block["antenna2"],
            packed,
            stokes_i=intensity,
            chi1=chi1,
            chi2=chi2,
            gauge_antenna_id=gauge,
            row_mask=train_time,
            ridge=1.0,
            qu_ridge=1.0e-4,
            n_iter=int(arguments.n_iter),
            time_s=block["time"],
            offdiag_drift=True,
            drift_ridge=chosen,
        )
        later_hold = residual_holdout_report(
            block["antenna1"],
            block["antenna2"],
            packed,
            stokes_i=intensity,
            chi1=chi1,
            chi2=chi2,
            residual_jones=smooth["jones"],
            q_over_i=smooth["q_over_i"],
            u_over_i=smooth["u_over_i"],
            row_mask=hold_time,
            cluster_ids=block["scan"],
            time_s=block["time"],
            drift_offdiag=smooth["drift_offdiag"],
            time_origin_s=smooth["time_origin_s"],
            time_scale_s=smooth["time_scale_s"],
        )
        later_focus = _focus_coherent(
            block, packed, intensity, chi1, chi2, name_to_id, hold_time, smooth
        )
        scale_coherent = {}
        for neighbor in (chosen * 0.1, chosen * 10.0):
            neighbor_fit = estimate_all_antenna_residual_jones(
                block["antenna1"],
                block["antenna2"],
                packed,
                stokes_i=intensity,
                chi1=chi1,
                chi2=chi2,
                gauge_antenna_id=gauge,
                row_mask=train_time,
                ridge=1.0,
                n_iter=int(arguments.n_iter),
                time_s=block["time"],
                offdiag_drift=True,
                drift_ridge=float(neighbor),
            )
            neighbor_hold = residual_holdout_report(
                block["antenna1"],
                block["antenna2"],
                packed,
                stokes_i=intensity,
                chi1=chi1,
                chi2=chi2,
                residual_jones=neighbor_fit["jones"],
                q_over_i=neighbor_fit["q_over_i"],
                u_over_i=neighbor_fit["u_over_i"],
                row_mask=hold_time,
                time_s=block["time"],
                drift_offdiag=neighbor_fit["drift_offdiag"],
                time_origin_s=neighbor_fit["time_origin_s"],
                time_scale_s=neighbor_fit["time_scale_s"],
            )
            scale_coherent[f"{neighbor:g}"] = float(
                (neighbor_hold.get("coherent_rl") or {}).get("coherent_mean_abs") or np.nan
            )
        later_coherent = later_hold.get("coherent_rl") or {}
        static_rr = float(time_hold.get("median_abs_rr_over_i") or 0.0)
        static_ll = float(time_hold.get("median_abs_ll_over_i") or 0.0)
        smooth_rr = float(later_hold.get("median_abs_rr_over_i") or 0.0)
        smooth_ll = float(later_hold.get("median_abs_ll_over_i") or 0.0)
        rr_ll_regression = (smooth_rr > static_rr + 0.002) or (smooth_ll > static_ll + 0.002)
        static_later = float(
            (time_hold.get("coherent_rl") or {}).get("coherent_mean_abs") or PREVIOUS_LATER_COHERENT
        )
        gate = classify_field9_time_transfer(
            later_coherent_abs=float(later_coherent.get("coherent_mean_abs") or np.nan),
            later_averages_as_noise=bool(later_coherent.get("averages_as_noise")),
            baseline_averages_as_noise=bool(
                (baseline_report.get("coherent_rl") or {}).get("averages_as_noise")
            ),
            qu_fold_spread=selection.get("qu_fold_spread"),
            rr_ll_regression=rr_ll_regression,
            antenna_later_coherent=later_focus,
            antenna_static_coherent=static_focus,
            scale_coherent=scale_coherent,
            previous_coherent_abs=static_later,
        )
        print(
            "later-time transfer",
            gate["status"],
            "coherent",
            gate["later_coherent_abs"],
            "noise-like",
            gate["later_averages_as_noise"],
            flush=True,
        )
        transfer = {
            "schema": "thol0001_field9_time_smooth_residual_jones_v1",
            "status": gate["status"],
            "blocking": gate["blocking"],
            "measurement_set": str(arguments.measurement_set),
            "spectral_window_id": 4,
            "native_channel": native_idx,
            "chosen_drift_ridge": chosen,
            "inner_fold_selection": selection,
            "q_over_i": smooth["q_over_i"],
            "u_over_i": smooth["u_over_i"],
            "frac_pol": smooth["frac_pol"],
            "median_abs_epsilon": smooth["median_abs_epsilon"],
            "jones": serialize_reference_jones(smooth["jones"]),
            "jones_by_name": _named_epsilon(smooth["jones"], names),
            "later_holdout": later_hold,
            "static_later_holdout": time_hold,
            "baseline_holdout": baseline_report,
            "focus_later_coherent": later_focus,
            "focus_static_coherent": static_focus,
            "neighbor_scale_later_coherent": scale_coherent,
            "rr_ll_regression": rr_ll_regression,
            "piecewise_state_term_indicated": jump["piecewise_state_term_indicated"],
            "gate": gate,
            "notes": [
                ALL_ANTENNA_RESIDUAL_JONES_NOTE,
                THREE_C147_QU_IS_NUISANCE_NOTE,
                CASA_FIELD9_NOT_END_TO_END_HOLDOUT_NOTE,
                SCAN_STATE_MAY_NEED_OFFSETS_NOTE,
            ],
        }
    write_json(transfer, arguments.output_dir / "field9_time_smooth_residual_jones.json")
    print(arguments.output_dir / "field9_residual_jones_controls.json")
    print(arguments.output_dir / "field9_all_antenna_residual_jones.json")
    print(arguments.output_dir / "field9_time_smooth_residual_jones.json")
    print(
        "P convention",
        p_convention["status"],
        "channel time-dependent",
        channel_support["time_dependent_support"],
    )
    print("static later coherent", (time_hold.get("coherent_rl") or {}).get("coherent_mean_abs"))
    print("time-smooth", transfer.get("status"), transfer.get("chosen_drift_ridge"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
