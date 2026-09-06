"""Diagnose identifiability of the all-antenna field-9 residual-Jones solve.

Scans 53 and 56 are placed on an identical antenna/baseline/time mask.
Jones maps are gauge-aligned before parameter comparison. Predicted
visibility operators, not raw ε, decide whether the jump is physical.
No scan offsets and no extra time freedom. SPW 5 stays closed.
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
    COMPLETE_FIELD9_TRACK_NOTE,
    ESTIMATOR_IDENTIFIABILITY_NOTE,
    PER_SCAN_EPSILON_NOT_OBSERVABLE_NOTE,
    align_residual_jones_maps,
    classify_estimator_identifiability,
    estimate_all_antenna_residual_jones,
    jones_parameter_jump,
    matched_scan_row_masks,
    predicted_operator_difference,
    residual_holdout_report,
    serialize_reference_jones,
)
from sl1mjax.polarization import Correlation, Receptor, pack_coherency

SCIENTIFIC_MS = Path(
    "/media/stephen/astro/vla/extracted/commissioning/work/THOL0001.lowerC.spw45.scientific.ms"
)
UNBLOCK = Path(
    "/media/stephen/astro/vla/extracted/commissioning/validation/scientific/full_jones_unblock"
)
FROM_SCAN = 53
TO_SCAN = 56
NATIVE_CHANNEL = 32


def _pol():
    path = Path(__file__).with_name("report_thol0001_scientific_pol_gates.py")
    spec = importlib.util.spec_from_file_location("thol0001_pol_gates", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _named(jones: dict[int, np.ndarray], names: np.ndarray) -> dict[str, float]:
    named = {}
    for ant, plane in jones.items():
        if int(ant) >= len(names):
            continue
        named[str(names[int(ant)])] = float(np.linalg.norm(np.asarray(plane) - np.eye(2)))
    return named


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--measurement-set", type=Path, default=SCIENTIFIC_MS)
    parser.add_argument("--output-dir", type=Path, default=UNBLOCK)
    parser.add_argument("--spectral-window", type=int, default=4)
    parser.add_argument("--n-bootstrap", type=int, default=6)
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
    vis = np.asarray(block["vis"], dtype=np.complex128)
    packed = pack_coherency(
        vis[:, native],
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
    mask_a, mask_b, match = matched_scan_row_masks(
        block["scan"], block["antenna1"], block["antenna2"], usable, FROM_SCAN, TO_SCAN
    )
    if not np.any(mask_a) or not np.any(mask_b):
        raise ValueError("matched 53/56 mask is empty")
    gauge = int(name_to_id[REFERENCE_ANTENNA])
    print("matched rows", match["n_matched_rows"], flush=True)
    fit_a = _fit(block, packed, intensity, chi1, chi2, mask_a, gauge)
    fit_b = _fit(block, packed, intensity, chi1, chi2, mask_b, gauge)
    aligned_b, phase = align_residual_jones_maps(
        fit_a["jones"], fit_b["jones"], gauge_antenna_id=gauge
    )
    raw_jump = jones_parameter_jump(fit_a["jones"], fit_b["jones"])
    aligned_jump = jones_parameter_jump(fit_a["jones"], aligned_b)
    named_jump = {
        str(names[ant]): aligned_jump["by_antenna"][ant]
        for ant in aligned_jump["by_antenna"]
        if ant < len(names)
    }
    operator = predicted_operator_difference(
        block["antenna1"],
        block["antenna2"],
        stokes_i=intensity,
        chi1=chi1,
        chi2=chi2,
        first_jones=fit_a["jones"],
        second_jones=aligned_b,
        q_over_i=0.5 * (fit_a["q_over_i"] + fit_b["q_over_i"]),
        u_over_i=0.5 * (fit_a["u_over_i"] + fit_b["u_over_i"]),
        row_mask=mask_a,
    )
    self_a = residual_holdout_report(
        block["antenna1"],
        block["antenna2"],
        packed,
        stokes_i=intensity,
        chi1=chi1,
        chi2=chi2,
        residual_jones=fit_a["jones"],
        q_over_i=fit_a["q_over_i"],
        u_over_i=fit_a["u_over_i"],
        row_mask=mask_a,
    )
    self_b = residual_holdout_report(
        block["antenna1"],
        block["antenna2"],
        packed,
        stokes_i=intensity,
        chi1=chi1,
        chi2=chi2,
        residual_jones=fit_b["jones"],
        q_over_i=fit_b["q_over_i"],
        u_over_i=fit_b["u_over_i"],
        row_mask=mask_b,
    )
    cross_ab = residual_holdout_report(
        block["antenna1"],
        block["antenna2"],
        packed,
        stokes_i=intensity,
        chi1=chi1,
        chi2=chi2,
        residual_jones=fit_a["jones"],
        q_over_i=fit_a["q_over_i"],
        u_over_i=fit_a["u_over_i"],
        row_mask=mask_b,
    )
    cross_ba = residual_holdout_report(
        block["antenna1"],
        block["antenna2"],
        packed,
        stokes_i=intensity,
        chi1=chi1,
        chi2=chi2,
        residual_jones=aligned_b,
        q_over_i=fit_b["q_over_i"],
        u_over_i=fit_b["u_over_i"],
        row_mask=mask_a,
    )
    self_rl = 0.5 * (
        float(self_a.get("median_abs_rl_over_i") or np.nan)
        + float(self_b.get("median_abs_rl_over_i") or np.nan)
    )
    cross_rl = 0.5 * (
        float(cross_ab.get("median_abs_rl_over_i") or np.nan)
        + float(cross_ba.get("median_abs_rl_over_i") or np.nan)
    )
    qu_points = [(0.0, 0.0), (fit_a["q_over_i"], fit_a["u_over_i"])]
    qu_path = arguments.output_dir / "field9_qu.json"
    if qu_path.exists():
        hold = json.loads(qu_path.read_text()).get("holdout") or {}
        for q_key, u_key in (("q_over_i_p16", "u_over_i_p16"), ("q_over_i_p84", "u_over_i_p84")):
            if hold.get(q_key) is not None and hold.get(u_key) is not None:
                qu_points.append((float(hold[q_key]), float(hold[u_key])))
    if len(qu_points) < 3:
        qu_points.extend([(0.003, -0.003), (-0.003, 0.003)])
    qu_offdiag = []
    qu_reports = []
    for q_frac, u_frac in qu_points:
        fixed = _fit(
            block,
            packed,
            intensity,
            chi1,
            chi2,
            mask_a,
            gauge,
            q=q_frac,
            u=u_frac,
            fit_qu=False,
            n_iter=1,
        )
        qu_offdiag.append(
            {
                name: float(np.abs(np.asarray(plane)[0, 1]))
                for name, plane in (
                    (str(names[ant]), plane)
                    for ant, plane in fixed["jones"].items()
                    if ant < len(names)
                )
            }
        )
        qu_reports.append(
            {
                "q_over_i": q_frac,
                "u_over_i": u_frac,
                "median_abs_epsilon": fixed["median_abs_epsilon"],
            }
        )
    spreads = []
    if qu_offdiag:
        for name in qu_offdiag[0]:
            values = [item[name] for item in qu_offdiag if name in item]
            if values:
                spreads.append(float(np.max(values) - np.min(values)))
    qu_spread = float(np.max(spreads)) if spreads else float("nan")
    init_points = [(0.0, 0.0), (0.005, -0.005), (-0.005, 0.005)]
    init_jones = []
    init_loss = []
    for q0, u0 in init_points:
        started = _fit(block, packed, intensity, chi1, chi2, mask_a, gauge, q=q0, u=u0, n_iter=2)
        hold = residual_holdout_report(
            block["antenna1"],
            block["antenna2"],
            packed,
            stokes_i=intensity,
            chi1=chi1,
            chi2=chi2,
            residual_jones=started["jones"],
            q_over_i=started["q_over_i"],
            u_over_i=started["u_over_i"],
            row_mask=mask_a,
        )
        init_jones.append(started["jones"])
        init_loss.append(float(hold.get("median_abs_rl_over_i") or np.nan))
    init_deltas = []
    for left, right in zip(init_jones[:-1], init_jones[1:], strict=True):
        aligned, _ = align_residual_jones_maps(left, right, gauge_antenna_id=gauge)
        init_deltas.append(jones_parameter_jump(left, aligned)["max_abs_delta"])
    rng = np.random.default_rng(0)
    rows_a = np.flatnonzero(mask_a)
    boot = []
    for _ in range(int(arguments.n_bootstrap)):
        chosen = np.zeros(mask_a.size, dtype=bool)
        chosen[rng.choice(rows_a, size=max(int(0.8 * rows_a.size), 8), replace=True)] = True
        try:
            sample = _fit(block, packed, intensity, chi1, chi2, chosen, gauge, n_iter=1)
        except ValueError:
            continue
        boot.append(
            {
                str(names[ant]): float(np.abs(np.asarray(plane)[0, 1]))
                for ant, plane in sample["jones"].items()
                if ant < len(names)
            }
        )
    boot_spread = {}
    if boot:
        for name in boot[0]:
            values = [item[name] for item in boot if name in item]
            if len(values) >= 3:
                boot_spread[name] = {
                    "p16": float(np.percentile(values, 16)),
                    "p84": float(np.percentile(values, 84)),
                    "spread": float(np.percentile(values, 84) - np.percentile(values, 16)),
                }
    gate = classify_estimator_identifiability(
        parameter_max_abs_delta=float(aligned_jump["max_abs_delta"]),
        operator_median_abs_rl=float(operator["median_abs_rl_over_i"]),
        self_residual=self_rl,
        cross_residual=cross_rl,
        gram_condition=(fit_a.get("graph_condition") or {}).get("condition"),
        qu_offdiag_spread=qu_spread,
        init_offdiag_spread=float(np.max(init_deltas)) if init_deltas else None,
    )
    payload = {
        "schema": "thol0001_field9_estimator_identifiability_v1",
        "status": gate["status"],
        "blocking": gate["blocking"],
        "spectral_window_id": 4,
        "native_channel": native,
        "from_scan": FROM_SCAN,
        "to_scan": TO_SCAN,
        "matched": match,
        "scan_53": {
            "q_over_i": fit_a["q_over_i"],
            "u_over_i": fit_a["u_over_i"],
            "median_abs_epsilon": fit_a["median_abs_epsilon"],
            "condition": (fit_a.get("graph_condition") or {}).get("condition"),
            "self_residual": self_a,
            "abs_epsilon_by_name": _named(fit_a["jones"], names),
        },
        "scan_56": {
            "q_over_i": fit_b["q_over_i"],
            "u_over_i": fit_b["u_over_i"],
            "median_abs_epsilon": fit_b["median_abs_epsilon"],
            "condition": (fit_b.get("graph_condition") or {}).get("condition"),
            "self_residual": self_b,
            "abs_epsilon_by_name": _named(fit_b["jones"], names),
        },
        "parameter_jump_raw": raw_jump,
        "parameter_jump_aligned": {**aligned_jump, "by_name": named_jump, "align_phase_rad": phase},
        "predicted_operator": operator,
        "cross_apply": {
            "53_on_56": cross_ab,
            "56_on_53": cross_ba,
            "self_median_abs_rl_over_i": self_rl,
            "cross_median_abs_rl_over_i": cross_rl,
        },
        "graph_condition": fit_a.get("graph_condition"),
        "bootstrap_offdiag": boot_spread,
        "qu_grid": qu_reports,
        "qu_offdiag_spread": qu_spread,
        "initializations": {
            "losses": init_loss,
            "max_aligned_abs_delta": float(np.max(init_deltas)) if init_deltas else float("nan"),
        },
        "jones_53": serialize_reference_jones(fit_a["jones"]),
        "jones_56_aligned": serialize_reference_jones(aligned_b),
        "gate": gate,
        "notes": [
            PER_SCAN_EPSILON_NOT_OBSERVABLE_NOTE,
            ESTIMATOR_IDENTIFIABILITY_NOTE,
            COMPLETE_FIELD9_TRACK_NOTE,
        ],
    }
    path = arguments.output_dir / "field9_estimator_identifiability.json"
    write_json(payload, path)
    print(path)
    print(
        "aligned max |Δε|",
        aligned_jump["max_abs_delta"],
        "operator RL/I",
        operator["median_abs_rl_over_i"],
    )
    print(
        "cross",
        cross_rl,
        "self",
        self_rl,
        "gate",
        gate["status"],
        "physical",
        gate["parameter_jump_is_physical"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
