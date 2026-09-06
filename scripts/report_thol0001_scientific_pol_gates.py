"""Field-9 Q/U and 3C286 apply-back on the scientific CORRECTED_DATA column.

Reads the scientific work MS. Does not apply calibration. Channel edges of
3C286 were excluded from Kcross/Xf (4:5~58). Later field-9 scans are a
time holdout for the on-axis Df assumption.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from casacore import tables

from sl1mjax.calibration_terms import parallactic_angle_rad
from sl1mjax.holography_calibration import (
    THREE_C286_CASAGUIDE_TWO_CHI_DEG,
    THREE_C286_IAU_EVPA_DEG,
    unwrapped_chi_span_rad,
    write_json,
)
from sl1mjax.holography_ms import _read_antennas, _tables
from sl1mjax.holography_reference_jones import (
    THREE_C147_QU_IS_NUISANCE_NOTE,
    THREE_C286_CIRCULAR_FLOOR_NOTE,
    classify_crosshand_floor,
    classify_three_c147_qu_model,
    coherent_residual_report,
)

SCIENTIFIC_MS = Path(
    "/media/stephen/astro/vla/extracted/commissioning/work/THOL0001.lowerC.spw45.scientific.ms"
)
EDGE = 5
GLOBAL_X = ("ea03", "ea11", "ea13", "ea19", "ea25")


def _stokes(vis: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rr, rl, lr, ll = vis[..., 0], vis[..., 1], vis[..., 2], vis[..., 3]
    stokes_i = 0.5 * (rr + ll)
    stokes_v = 0.5 * (rr - ll)
    stokes_q = 0.5 * (rl + lr)
    stokes_u = 0.5j * (lr - rl)
    return stokes_i, stokes_q, stokes_u, stokes_v


def _ratio(num: np.ndarray, den: np.ndarray) -> float:
    usable = np.isfinite(num) & np.isfinite(den) & (np.abs(den) > 1.0e-3)
    if not np.any(usable):
        return float("nan")
    return float(np.median(np.real(num[usable] / den[usable])))


def _median_complex(values: np.ndarray) -> complex:
    usable = values[np.isfinite(values)]
    if usable.size == 0:
        return complex("nan")
    return complex(np.median(usable.real) + 1j * np.median(usable.imag))


def _load_field(
    measurement_set: Path,
    field_id: int | None,
    data_desc_id: int | None,
) -> dict[str, np.ndarray]:
    clauses = []
    if field_id is not None:
        clauses.append(f"FIELD_ID={int(field_id)}")
    if data_desc_id is not None:
        clauses.append(f"DATA_DESC_ID={int(data_desc_id)}")
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    query = f"SELECT FROM '{measurement_set}'{where}"
    with tables.taql(query) as selected:
        if selected.nrows() == 0:
            raise ValueError(f"no rows for field {field_id} ddid {data_desc_id}")
        flag = np.asarray(selected.getcol("FLAG"), dtype=bool)
        vis = np.asarray(selected.getcol("CORRECTED_DATA"))
        vis = np.where(flag, np.nan, vis)
        return {
            "vis": vis,
            "flag": flag,
            "antenna1": np.asarray(selected.getcol("ANTENNA1"), dtype=np.int32),
            "antenna2": np.asarray(selected.getcol("ANTENNA2"), dtype=np.int32),
            "time": np.asarray(selected.getcol("TIME"), dtype=np.float64),
            "scan": np.asarray(selected.getcol("SCAN_NUMBER"), dtype=np.int32),
        }


def _channel_masks(n_chan: int) -> tuple[np.ndarray, np.ndarray]:
    hold = np.zeros(n_chan, dtype=bool)
    hold[:EDGE] = True
    hold[-EDGE:] = True
    return ~hold, hold


def _rl_delay_ns(vis: np.ndarray, frequency_hz: np.ndarray, inner: np.ndarray) -> dict[str, float]:
    rl = vis[:, inner, 1]
    lr = vis[:, inner, 2]
    freqs = frequency_hz[inner]
    if freqs.size < 4:
        return {"residual_rl_delay_ns": float("nan")}
    # Median crosshand phase vs frequency; delay from a linear fit.
    med_rl = np.array([_median_complex(rl[:, i]) for i in range(rl.shape[1])])
    usable = np.isfinite(med_rl)
    if int(np.sum(usable)) < 4:
        return {"residual_rl_delay_ns": float("nan"), "residual_crosshand_phase_deg": float("nan")}
    unwrap = np.unwrap(np.angle(med_rl[usable]))
    slope = float(np.polyfit(freqs[usable], unwrap, 1)[0])
    delay_s = slope / (2.0 * np.pi)
    med_lr = np.array([_median_complex(lr[:, i]) for i in range(lr.shape[1])])
    hand_diff = np.angle(med_lr[usable] * np.conj(med_rl[usable]))
    return {
        "residual_rl_delay_ns": delay_s * 1.0e9,
        "residual_crosshand_phase_deg": float(np.rad2deg(np.median(unwrap))),
        "residual_lr_minus_rl_phase_deg": float(np.rad2deg(np.median(hand_diff))),
    }


def report_3c286(measurement_set: Path) -> dict[str, object]:
    tables = _tables()
    _ids, names, _positions = _read_antennas(tables, measurement_set)
    by_spw = {}
    for ddid, freq_row in ((4, 4),):
        try:
            block = _load_field(measurement_set, 11, ddid)
        except ValueError:
            continue
        with tables.table(
            str(measurement_set / "SPECTRAL_WINDOW"), readonly=True, ack=False
        ) as window:
            freqs = np.asarray(window.getcell("CHAN_FREQ", freq_row), dtype=np.float64).reshape(-1)
        inner, edge = _channel_masks(block["vis"].shape[1])
        i, q, u, v = _stokes(block["vis"])
        two_chi = np.rad2deg(np.arctan2(np.real(u), np.real(q)))
        evpa = 0.5 * two_chi

        def pack(
            chan_mask: np.ndarray,
            i_plane: np.ndarray = i,
            q_plane: np.ndarray = q,
            u_plane: np.ndarray = u,
            v_plane: np.ndarray = v,
            two_chi_plane: np.ndarray = two_chi,
            evpa_plane: np.ndarray = evpa,
        ) -> dict[str, float]:
            return {
                "q_over_i": _ratio(q_plane[:, chan_mask], i_plane[:, chan_mask]),
                "u_over_i": _ratio(u_plane[:, chan_mask], i_plane[:, chan_mask]),
                "v_over_i": _ratio(v_plane[:, chan_mask], i_plane[:, chan_mask]),
                "casaguide_two_chi_deg": float(np.nanmedian(two_chi_plane[:, chan_mask])),
                "iau_evpa_deg": float(np.nanmedian(evpa_plane[:, chan_mask])),
                "n": int(np.sum(np.isfinite(i_plane[:, chan_mask]))),
            }

        absent = []
        present = set(
            int(a) for a in np.unique(np.concatenate([block["antenna1"], block["antenna2"]]))
        )
        for index, name in enumerate(names):
            if index not in present:
                absent.append(name)
        by_spw[str(ddid)] = {
            "inner_used_for_kcross_x": pack(inner),
            "edge_held_out_from_kcross_x": pack(edge),
            "delay": _rl_delay_ns(block["vis"], freqs, inner),
            "n_times": int(np.unique(block["time"]).size),
            "n_scans": int(np.unique(block["scan"]).size),
            "antennas_absent_from_3c286": absent,
            "global_x_required": [name for name in GLOBAL_X if name in absent or name in GLOBAL_X],
        }
    inner = by_spw.get("4", {}).get("edge_held_out_from_kcross_x", {})
    q_i = inner.get("q_over_i", float("nan"))
    u_i = inner.get("u_over_i", float("nan"))
    v_i = inner.get("v_over_i", float("nan"))
    two = inner.get("casaguide_two_chi_deg", float("nan"))
    evpa = inner.get("iau_evpa_deg", float("nan"))
    delay = by_spw.get("4", {}).get("delay", {})
    residual_x_deg = (
        float(two) - THREE_C286_CASAGUIDE_TWO_CHI_DEG if np.isfinite(two) else float("nan")
    )
    ok_qu = np.isfinite(q_i) and np.hypot(q_i, u_i) > 0.05
    ok_evpa = np.isfinite(evpa) and abs(evpa - THREE_C286_IAU_EVPA_DEG) < 8.0
    ok_two = np.isfinite(two) and abs(two - THREE_C286_CASAGUIDE_TWO_CHI_DEG) < 15.0
    ok_v = np.isfinite(v_i) and abs(v_i) < 0.03
    delay_ns = delay.get("residual_rl_delay_ns", float("nan"))
    ok_delay = np.isfinite(delay_ns) and abs(delay_ns) < 1.0
    ok_x = np.isfinite(residual_x_deg) and abs(residual_x_deg) < 8.0
    status = "pass" if (ok_qu and ok_v and (ok_evpa or ok_two) and ok_delay) else "fail"
    if not ok_qu:
        status = "fail"
    elif not (ok_evpa or ok_two):
        status = "warn"
    circular_floor = {
        "v_over_i": v_i,
        "status": "warn" if np.isfinite(v_i) and abs(v_i) > 0.001 else "pass",
        "acceptable_for_beam_experiment": bool(np.isfinite(v_i) and abs(v_i) < 0.03),
        "negligible_for_full_stokes": False,
        "notes": (THREE_C286_CIRCULAR_FLOOR_NOTE,),
    }
    return {
        "schema": "thol0001_scientific_3c286_applyback_v1",
        "measurement_set": str(measurement_set),
        "independence": "xf_kcross_excluded_edge_channels",
        "applied_from": "DATA_once_on_scientific_ms",
        "casaguide_two_chi_expected_deg": THREE_C286_CASAGUIDE_TWO_CHI_DEG,
        "iau_evpa_expected_deg": THREE_C286_IAU_EVPA_DEG,
        "by_spw": by_spw,
        "identity_inheritance_forbidden": True,
        "global_x_antennas": list(GLOBAL_X),
        "residual_x_deg": residual_x_deg,
        "circular_polarisation_floor": circular_floor,
        "gate_status": {
            "three_c286_qu_evpa_v": status,
            "residual_rl_delay_phase": "pass" if ok_delay and ok_x else "warn",
            "three_c286_circular_floor": circular_floor["status"],
        },
        "notes": [
            "iau_evpa_deg is (1/2) arg(Q+iU); casaguide_two_chi_deg is arg(Q+iU)",
            "Edge channels were excluded from Kcross/Xf; inner channels are in-sample",
            "Antennas absent from usable 3C286 data keep the justified global X, not identity",
            THREE_C286_CIRCULAR_FLOOR_NOTE,
        ],
    }


def report_field9(measurement_set: Path) -> dict[str, object]:
    tables = _tables()
    _ids, names, positions = _read_antennas(tables, measurement_set)
    positions = np.asarray(positions, dtype=np.float64)
    with tables.table(str(measurement_set / "FIELD"), readonly=True, ack=False) as field_table:
        direction = np.asarray(field_table.getcell("PHASE_DIR", 9), dtype=np.float64).reshape(-1, 2)
        phase = (float(direction[0, 0]), float(direction[0, 1]))
    block = _load_field(measurement_set, 9, 4)
    inner, _edge = _channel_masks(block["vis"].shape[1])
    i, q, u, v = _stokes(block["vis"][:, inner])
    chi_ant = parallactic_angle_rad(block["time"], phase, positions)
    chi = 0.5 * (
        chi_ant[np.arange(block["time"].size), block["antenna1"]]
        + chi_ant[np.arange(block["time"].size), block["antenna2"]]
    )
    scans = np.unique(block["scan"])
    train_scans = set(scans[: max(1, scans.size // 2)].tolist())
    hold_scans = set(scans[max(1, scans.size // 2) :].tolist())
    train = np.isin(block["scan"], list(train_scans))
    hold = np.isin(block["scan"], list(hold_scans))

    def qu_fit(mask: np.ndarray) -> dict[str, float]:
        usable = mask & np.isfinite(i[:, 0]) if i.ndim == 2 else mask
        # average over inner channels first
        ii = np.nanmedian(i, axis=1)
        qq = np.nanmedian(q, axis=1)
        uu = np.nanmedian(u, axis=1)
        vv = np.nanmedian(v, axis=1)
        good = usable & np.isfinite(ii) & (np.abs(ii) > 1.0)
        if int(np.sum(good)) < 16:
            return {"n": int(np.sum(good)), "q_over_i": float("nan"), "u_over_i": float("nan")}
        q_i = np.real(qq[good] / ii[good])
        u_i = np.real(uu[good] / ii[good])
        v_i = np.real(vv[good] / ii[good])
        angle = chi[good]
        # Q_app = Q cos 2χ + U sin 2χ; U_app = -Q sin 2χ + U cos 2χ after P
        # After parang=True applycal, residual Q/U should be sky-fixed.
        two = 2.0 * angle
        design = np.column_stack([np.cos(two), np.sin(two)])
        try:
            sky_q, _, _, _ = np.linalg.lstsq(design, q_i, rcond=None)
        except np.linalg.LinAlgError:
            sky_q = np.array([float("nan"), float("nan")])
        return {
            "n": int(np.sum(good)),
            "q_over_i": float(np.median(q_i)),
            "u_over_i": float(np.median(u_i)),
            "v_over_i": float(np.median(v_i)),
            "q_over_i_p16": float(np.quantile(q_i, 0.16)),
            "q_over_i_p84": float(np.quantile(q_i, 0.84)),
            "u_over_i_p16": float(np.quantile(u_i, 0.16)),
            "u_over_i_p84": float(np.quantile(u_i, 0.84)),
            "frac_pol": float(np.hypot(np.median(q_i), np.median(u_i))),
            "sky_q_from_qapp": float(sky_q[0]),
            "sky_u_from_qapp": float(sky_q[1]),
            "chi_deg_span": float(np.rad2deg(unwrapped_chi_span_rad(block["time"][good], angle))),
        }

    train_fit = qu_fit(train)
    hold_fit = qu_fit(hold)
    all_fit = qu_fit(np.ones(train.size, dtype=bool))
    frac = hold_fit.get("frac_pol", float("nan"))
    qu_gate = classify_three_c147_qu_model(
        q_over_i=float(hold_fit.get("q_over_i") or 0.0),
        u_over_i=float(hold_fit.get("u_over_i") or 0.0),
        frac_pol=frac if np.isfinite(frac) else None,
        represented_as_exact_zero=False,
    )
    qu_status = str(qu_gate["status"])
    model = {
        "q_over_i": hold_fit.get("q_over_i"),
        "u_over_i": hold_fit.get("u_over_i"),
        "q_over_i_p16": hold_fit.get("q_over_i_p16"),
        "q_over_i_p84": hold_fit.get("q_over_i_p84"),
        "u_over_i_p16": hold_fit.get("u_over_i_p16"),
        "u_over_i_p84": hold_fit.get("u_over_i_p84"),
        "frac_pol": frac,
        "exact_zero_forbidden": True,
        "represented_as_exact_zero": False,
        "relationship_to_model_data": qu_gate["relationship_to_model_data"],
        "note": THREE_C147_QU_IS_NUISANCE_NOTE,
        "compare_zero_and_fitted_on_holdout": True,
    }
    # Cross-hand floor on held-out times: |RL|/I, |LR|/I
    rl = block["vis"][:, inner, 1]
    lr = block["vis"][:, inner, 2]
    ii = np.nanmedian(i, axis=1)
    hold_rows = hold & np.isfinite(ii) & (np.abs(ii) > 1.0)
    rl_over_i = rl[hold_rows] / np.abs(ii[hold_rows, None])
    lr_over_i = lr[hold_rows] / np.abs(ii[hold_rows, None])
    coherence = coherent_residual_report(rl_over_i)
    floor_gate = classify_crosshand_floor(
        median_abs_rl_over_i=float(np.nanmedian(np.abs(rl_over_i))),
        coherent_mean_abs=float(coherence.get("coherent_mean_abs") or np.nan),
        averages_as_noise=bool(coherence.get("averages_as_noise")),
    )
    floor = {
        "n": int(np.sum(hold_rows)),
        "median_abs_rl_over_i": float(floor_gate["median_abs_rl_over_i"]),
        "median_abs_lr_over_i": float(np.nanmedian(np.abs(lr_over_i))),
        "p90_abs_rl_over_i": float(np.nanquantile(np.abs(rl_over_i), 0.9)),
        "metric": "absolute_resid_over_stokes_i",
        "max_rel_not_used": True,
        "coherence": coherence,
        "blocking": floor_gate["blocking"],
    }
    floor_status = str(floor_gate["status"])
    return {
        "schema": "thol0001_scientific_field9_qu_v1",
        "measurement_set": str(measurement_set),
        "d_product_on_corrected": "Df",
        "n_times": int(np.unique(block["time"]).size),
        "n_scans": int(scans.size),
        "train_scans": sorted(int(s) for s in train_scans),
        "holdout_scans": sorted(int(s) for s in hold_scans),
        "train": train_fit,
        "holdout": hold_fit,
        "all_scans": all_fit,
        "source_qu_model": model,
        "crosshand_residual_floor": floor,
        "antenna_names_n": len(names),
        "gate_status": {
            "three_c147_qu_source_model": qu_status,
            "crosshand_residual_floor": floor_status,
            "three_c147_df_vs_df_qu": "not_run",
        },
        "notes": [
            "CORRECTED_DATA already has scientific Df applied from DATA once",
            "Df+QU comparison requires a dedicated apply copy, not a second apply here",
            "Do not assume 3C147 is unpolarised; this report measures it",
            THREE_C147_QU_IS_NUISANCE_NOTE,
            "A few-percent |RL|/I floor is blocking until residuals average as noise",
        ],
    }


def compare_df_variants(df_ms: Path, df_qu_ms: Path) -> dict[str, object]:
    first = _load_field(
        df_ms,
        9 if "scientific.ms" in str(df_ms) else None,
        4 if "scientific.ms" in str(df_ms) else None,
    )
    second = _load_field(df_qu_ms, None, None)
    if first["vis"].shape != second["vis"].shape:
        raise ValueError("Df and Df+QU apply copies are not aligned")
    inner, _ = _channel_masks(first["vis"].shape[1])
    scans = np.unique(first["scan"])
    hold = np.isin(first["scan"], scans[max(1, scans.size // 2) :])
    delta = second["vis"][:, inner] - first["vis"][:, inner]
    i = 0.5 * (first["vis"][:, inner, 0] + first["vis"][:, inner, 3])
    by_corr = {}
    for slot, name in enumerate(("RR", "RL", "LR", "LL")):
        resid = delta[hold, :, slot]
        scale = np.abs(i[hold])
        usable = np.isfinite(resid) & np.isfinite(scale) & (scale > 1.0)
        if not np.any(usable):
            by_corr[name] = {"n": 0}
            continue
        abs_over_i = np.abs(resid[usable]) / scale[usable]
        by_corr[name] = {
            "n": int(np.sum(usable)),
            "median_abs_over_i": float(np.median(abs_over_i)),
            "p90_abs_over_i": float(np.quantile(abs_over_i, 0.9)),
        }
    rl = by_corr.get("RL", {}).get("median_abs_over_i", float("nan"))
    status = "warn"
    selected = "neither"
    if np.isfinite(rl) and rl < 0.005:
        status = "pass"
        selected = "Df"
    return {
        "schema": "thol0001_field9_df_vs_df_qu_v1",
        "holdout": "later_half_of_field9_scans",
        "by_correlation": by_corr,
        "selected_d_product": selected,
        "status": status,
        "note": "Visibility difference on held-out field-9 times; no empirical D scale applied",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--measurement-set", type=Path, default=SCIENTIFIC_MS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--df-ms", type=Path)
    parser.add_argument("--df-qu-ms", type=Path)
    arguments = parser.parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    c286 = report_3c286(arguments.measurement_set)
    write_json(c286, arguments.output_dir / "three_c286_applyback.json")
    field9 = report_field9(arguments.measurement_set)
    if arguments.df_ms and arguments.df_qu_ms and arguments.df_qu_ms.exists():
        variant = compare_df_variants(arguments.df_ms, arguments.df_qu_ms)
        write_json(variant, arguments.output_dir / "field9_df_vs_df_qu.json")
        field9["gate_status"]["three_c147_df_vs_df_qu"] = variant["status"]
        field9["df_vs_df_qu"] = variant
    write_json(field9, arguments.output_dir / "field9_qu.json")
    print(arguments.output_dir / "three_c286_applyback.json")
    print("3C286", c286["gate_status"])
    print(arguments.output_dir / "field9_qu.json")
    print("field9", field9["gate_status"])
    print("source_qu", field9["source_qu_model"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
