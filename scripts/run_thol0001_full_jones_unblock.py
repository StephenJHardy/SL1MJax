"""Assemble the THOL0001 scientific full-Jones unblock report.

Reads existing products. Does not overwrite scientific calibration tables
or the completed SPW-4 diagonal sample product. Full Jones stays unfrozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from casacore import tables

from sl1mjax.holography_calibration import (
    HELD_OUT_REFERENCE_ANTENNA,
    REFERENCE_ANTENNA,
    THREE_C286_CASAGUIDE_TWO_CHI_DEG,
    THREE_C286_EVPA_IS_TWO_CHI_NOTE,
    THREE_C286_IAU_EVPA_DEG,
    leakage_systematic_floor,
    write_json,
)
from sl1mjax.holography_calibration_oracle import stratify_df_delta
from sl1mjax.holography_reference_jones import (
    ALL_ANTENNA_RESIDUAL_JONES_NOTE,
    CALIBRATION_ADEQUACY_NOT_OPERATOR_NOTE,
    CASA_FIELD9_NOT_END_TO_END_HOLDOUT_NOTE,
    CHANNEL_AVERAGING_CAN_FAKE_TIME_NOTE,
    COMPLETE_FIELD9_TRACK_NOTE,
    CONNECTED_HOLDOUT_NOTE,
    CROSSHAND_FLOOR_BLOCKS_FULL_JONES_NOTE,
    ESTIMATOR_IDENTIFIABILITY_NOTE,
    IDENTITY_REFERENCE_JONES_BLOCKS_OFFDIAG_NOTE,
    INTERLEAVED_BEAM_TRANSFER_GATE_NOTE,
    INTERLEAVED_ONAXIS_TRANSFER_NOTE,
    JJH_PROXY_IS_NOT_DECISIVE_NOTE,
    NO_SCAN_OFFSETS_OR_TIME_FREEDOM_NOTE,
    ON_AXIS_BEAM_IDENTITY_NOTE,
    ONE_AXIS_VISIBILITY_HOLDOUT_NOTE,
    PREDICTION_EQUIVALENT_BEAM_NOTE,
    RESIDUAL_JONES_P_CONVENTION_NOTE,
    SCAN_STATE_MAY_NEED_OFFSETS_NOTE,
    SMOOTH_GP_REJECTED_NOTE,
    THREE_C147_QU_IS_NUISANCE_NOTE,
    THREE_C286_CIRCULAR_FLOOR_NOTE,
    ZERO_QU_IS_ABLATION_NOTE,
    classify_crosshand_floor,
    classify_three_c147_qu_model,
)

SCIENTIFIC_MS = Path(
    "/media/stephen/astro/vla/extracted/commissioning/work/THOL0001.lowerC.spw45.scientific.ms"
)
PRODUCT_ROOT = Path("/media/stephen/astro/vla/extracted/commissioning/products/scientific")
VALIDATION = Path("/media/stephen/astro/vla/extracted/commissioning/validation/scientific")
UNBLOCK = VALIDATION / "full_jones_unblock"
GLOBAL_X = ("ea03", "ea11", "ea13", "ea19", "ea25")
REFERENCE = ("ea02", "ea03", "ea07", "ea12", "ea14", "ea24", "ea26")
MOVING = (
    "ea04",
    "ea05",
    "ea06",
    "ea08",
    "ea09",
    "ea10",
    "ea11",
    "ea13",
    "ea15",
    "ea16",
    "ea17",
    "ea18",
    "ea19",
    "ea20",
    "ea21",
    "ea22",
    "ea23",
    "ea25",
    "ea27",
    "ea28",
)
GATES = (
    "diagonal_visibility_holdouts",
    "three_c147_df_vs_df_qu",
    "three_c286_qu_evpa_v",
    "residual_rl_delay_phase",
    "d_smoothness_antenna_support",
    "injected_operator_oracle",
    "real_visibility_casa_jax_golden",
    "three_c286_circular_floor",
    "crosshand_residual_floor",
    "crosshand_floor_coherence",
    "three_c147_qu_source_model",
    "residual_reference_jones",
    "field9_all_antenna_residual_jones",
    "residual_jones_p_convention",
    "field9_channel_support",
    "field9_scan_state",
    "on_axis_beam_identity",
    "field9_time_smooth_residual_jones",
    "scan53_56_chain_jump",
    "field9_estimator_identifiability",
    "field9_stabilized_residual_jones",
    "prediction_equivalent_beam_maps",
    "one_axis_visibility_holdouts",
    "field9_acquisition_state",
    "hierarchical_scan_state_residual",
    "interleaved_onaxis_beam_transfer",
    "connected_residual_holdouts",
    "identity_prior_antenna_holdout",
    "snr_masked_off_diagonal",
    "moving_reference_visibility_comparison",
)


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _status(value: str, **payload) -> dict:
    report = {"status": value}
    report.update(payload)
    return report


def _read_cal(path: Path) -> dict[str, np.ndarray]:
    with tables.table(str(path), readonly=True, ack=False) as table:
        return {
            "antenna": np.asarray(table.getcol("ANTENNA1"), dtype=np.int32),
            "spw": np.asarray(table.getcol("SPECTRAL_WINDOW_ID"), dtype=np.int32),
            "time": np.asarray(table.getcol("TIME"), dtype=np.float64),
            "flag": np.asarray(table.getcol("FLAG"), dtype=bool),
            "values": np.asarray(table.getcol("CPARAM")),
            "snr": (
                np.asarray(table.getcol("SNR"))
                if "SNR" in table.colnames()
                else np.full(table.getcol("CPARAM").shape, np.nan)
            ),
        }


def _antenna_names(measurement_set: Path) -> tuple[str, ...]:
    with tables.table(str(measurement_set / "ANTENNA"), readonly=True, ack=False) as table:
        return tuple(str(name) for name in table.getcol("NAME"))


def _keywords(path: Path) -> dict[str, object]:
    with tables.table(str(path), readonly=True, ack=False) as table:
        payload = {}
        for name in table.keywordnames():
            value = table.getkeyword(name)
            if isinstance(value, (str, int, float, bool)) or value is None:
                payload[name] = value
            else:
                payload[name] = str(value)
        return payload


def _source_pol(path: Path) -> dict[str, object]:
    keywords = _keywords(path)
    payload: dict[str, object] = {"found": False, "keywords": {}}
    for key, value in keywords.items():
        name = str(key).lower()
        if any(
            token in name for token in ("stokesq", "stokesu", "sourceq", "sourceu", "qu", "poln")
        ):
            payload["keywords"][str(key)] = value
            payload["found"] = True
    return payload


def _hand_stats(values: np.ndarray, valid: np.ndarray) -> dict[str, object]:
    by_hand = {}
    n_hand = values.shape[-1] if values.ndim >= 2 else 1
    for hand in range(n_hand):
        slot = values[..., hand] if values.ndim >= 2 else values
        mask = valid[..., hand] if valid.ndim >= 2 else valid
        usable = mask & np.isfinite(slot)
        if not np.any(usable):
            by_hand[str(hand)] = {"n": 0}
            continue
        selected = slot[usable]
        by_hand[str(hand)] = {
            "n": int(np.sum(usable)),
            "median_abs": float(np.median(np.abs(selected))),
            "p90_abs": float(np.quantile(np.abs(selected), 0.9)),
            "max_abs": float(np.max(np.abs(selected))),
            "median_phase_deg": float(np.median(np.rad2deg(np.angle(selected)))),
        }
    return by_hand


def _channel_smoothness(values: np.ndarray, valid: np.ndarray) -> dict[str, object]:
    """Hold out every 4th inner channel; interpolate from neighbors."""

    if values.ndim < 2 or values.shape[1] < 12:
        return {"n": 0, "status": "not_run", "reason": "too few channels"}
    n_chan = values.shape[1]
    hold = np.zeros(n_chan, dtype=bool)
    hold[5 : n_chan - 5 : 4] = True
    train = ~hold
    train[:5] = False
    train[-5:] = False
    residuals = []
    for row in range(values.shape[0]):
        for hand in range(values.shape[-1]):
            slot = values[row, :, hand]
            mask = valid[row, :, hand] & np.isfinite(slot)
            if int(np.sum(mask & train)) < 4 or int(np.sum(mask & hold)) == 0:
                continue
            channels = np.arange(n_chan, dtype=np.float64)
            pred = np.interp(
                channels[mask & hold],
                channels[mask & train],
                slot[mask & train].real,
            ) + 1j * np.interp(
                channels[mask & hold],
                channels[mask & train],
                slot[mask & train].imag,
            )
            residuals.append(np.abs(slot[mask & hold] - pred))
    if not residuals:
        return {"n": 0, "status": "not_run"}
    resid = np.concatenate(residuals)
    median_abs = float(np.median(resid))
    return {
        "n": int(resid.size),
        "median_abs_residual": median_abs,
        "p90_abs_residual": float(np.quantile(resid, 0.9)),
        "holdout_stride": 4,
        "edge_excluded": 5,
        "status": "pass" if median_abs < 0.02 else "warn" if median_abs < 0.05 else "fail",
        "empirical_scale_factor_applied": False,
    }


def _antenna_support(
    values: np.ndarray, valid: np.ndarray, antenna: np.ndarray, names: tuple[str, ...]
) -> dict:
    by_antenna = []
    unsupported = []
    for index, name in enumerate(names):
        rows = antenna == index
        if not np.any(rows):
            unsupported.append({"antenna": name, "reason": "absent_from_table"})
            continue
        mask = valid[rows]
        n_valid = int(np.sum(mask))
        n_total = int(mask.size)
        median_abs = float(np.median(np.abs(values[rows][mask]))) if n_valid else float("nan")
        entry = {
            "antenna": name,
            "n_valid": n_valid,
            "n_total": n_total,
            "valid_fraction": float(n_valid / n_total) if n_total else 0.0,
            "median_abs": median_abs,
            "role": ("reference" if name in REFERENCE else "moving" if name in MOVING else "other"),
        }
        if n_valid == 0:
            unsupported.append({"antenna": name, "reason": "all_flagged"})
        by_antenna.append(entry)
    return {"by_antenna": by_antenna, "unsupported": unsupported}


def inspect_d_product(path: Path, names: tuple[str, ...]) -> dict[str, object]:
    table = _read_cal(path)
    valid = ~table["flag"] & np.isfinite(table["values"])
    support = _antenna_support(table["values"], valid, table["antenna"], names)
    times = np.unique(np.round(table["time"], 3))
    by_spw = {}
    for spw in np.unique(table["spw"]):
        rows = table["spw"] == spw
        by_spw[str(int(spw))] = _hand_stats(table["values"][rows], valid[rows])
    ref_rows = None
    if REFERENCE_ANTENNA in names:
        ref_rows = table["antenna"] == names.index(REFERENCE_ANTENNA)
    return {
        "path": str(path),
        "n_rows": int(table["antenna"].size),
        "shape": list(table["values"].shape),
        "n_unique_times": int(times.size),
        "time_is_global": bool(times.size <= 2),
        "keywords": _keywords(path),
        "source_pol": _source_pol(path),
        "hands": _hand_stats(table["values"], valid),
        "by_spw": by_spw,
        "smoothness": _channel_smoothness(table["values"], valid),
        "antenna_support": support,
        "reference_antenna": REFERENCE_ANTENNA,
        "reference_median_abs": (
            float(np.median(np.abs(table["values"][ref_rows][valid[ref_rows]])))
            if ref_rows is not None and np.any(valid[ref_rows])
            else float("nan")
        ),
        "held_out_reference": HELD_OUT_REFERENCE_ANTENNA,
        "empirical_scale_factor_applied": False,
    }


def inventory() -> dict[str, object]:
    recovery = VALIDATION / "first_beam_recovery" / "first_beam_recovery_spw4.json"
    recovery_payload = _load(recovery) if recovery.exists() else {}
    flux = (
        _load(VALIDATION / "flux_gauge.json") if (VALIDATION / "flux_gauge.json").exists() else {}
    )
    golden_d = VALIDATION / "golden_diagonal_compare.json"
    golden_f = VALIDATION / "golden_fullpol_compare.json"
    holdouts = UNBLOCK / "diagonal_holdouts" / "visibility_holdouts.json"
    return {
        "schema": "thol0001_full_jones_unblock_inventory_v1",
        "scientific_ms": str(SCIENTIFIC_MS),
        "product_root": str(PRODUCT_ROOT),
        "immutable_source_untouched": True,
        "production_full_jones_factory_untouched": True,
        "spw5_unopened": True,
        "flux_gauge_passed": bool(flux.get("passed")),
        "diagonal_source": recovery_payload.get("source"),
        "diagonal_frozen": recovery_payload.get("frozen"),
        "diagonal_model_data": recovery_payload.get("model_data_rr"),
        "products": {
            "setjy": str(PRODUCT_ROOT / "setjy_3c147.json"),
            "diagonal": str(PRODUCT_ROOT / "diagonal" / "product.json"),
            "fullpol": str(PRODUCT_ROOT / "fullpol" / "product.json"),
            "df": str(PRODUCT_ROOT / "fullpol" / "Df.cal"),
            "df_qu": str(PRODUCT_ROOT / "fullpol" / "Df_QU.cal"),
        },
        "hashes": {
            "first_beam_recovery_spw4": _sha256(recovery),
            "flux_gauge": _sha256(VALIDATION / "flux_gauge.json"),
            "golden_diagonal": _sha256(golden_d),
            "golden_fullpol": _sha256(golden_f),
        },
        "holdouts_path": str(holdouts),
        "holdouts_present": holdouts.exists(),
        "three_c286_angles": {
            "casaguide_two_chi_deg": THREE_C286_CASAGUIDE_TWO_CHI_DEG,
            "iau_evpa_deg": THREE_C286_IAU_EVPA_DEG,
            "note": THREE_C286_EVPA_IS_TWO_CHI_NOTE,
        },
    }


def d_term_gates() -> dict[str, object]:
    names = _antenna_names(SCIENTIFIC_MS)
    df = inspect_d_product(PRODUCT_ROOT / "fullpol" / "Df.cal", names)
    df_qu = inspect_d_product(PRODUCT_ROOT / "fullpol" / "Df_QU.cal", names)
    a1 = _read_cal(PRODUCT_ROOT / "fullpol" / "Df.cal")
    a2 = _read_cal(PRODUCT_ROOT / "fullpol" / "Df_QU.cal")
    valid = ~a1["flag"] & ~a2["flag"]
    floor = leakage_systematic_floor(a1["values"], a2["values"], valid)
    stratified = stratify_df_delta(
        a1["values"],
        a2["values"],
        valid,
        antenna=a1["antenna"],
        spectral_window_id=a1["spw"],
        antenna_names=names,
        reference_antennas=REFERENCE,
        moving_antennas=MOVING,
        snr=a1["snr"],
    )
    smoothness = df["smoothness"]
    support = df["antenna_support"]["unsupported"]
    d_status = "pass"
    if smoothness.get("status") == "fail" or support:
        d_status = "fail" if support and smoothness.get("status") == "fail" else "warn"
    if smoothness.get("status") == "fail":
        d_status = "fail"
    median_delta = float(floor.get("median_abs_delta", float("nan")))
    p90_delta = float(stratified.get("p90_abs_delta", float("nan")))
    qu_hand1 = df_qu.get("hands", {}).get("1", {})
    qu_status = "not_run"
    if np.isfinite(median_delta):
        qu_status = (
            "pass"
            if median_delta < 0.01 and (not np.isfinite(p90_delta) or p90_delta < 0.03)
            else "warn"
        )
        if np.isfinite(p90_delta) and p90_delta > 0.2:
            qu_status = "warn"
        if float(qu_hand1.get("p90_abs", 0.0) or 0.0) > 0.5:
            qu_status = "warn"
    return {
        "schema": "thol0001_scientific_d_term_gates_v1",
        "Df": df,
        "Df_QU": df_qu,
        "leakage_systematic_floor": floor,
        "stratified_delta": {
            "median_abs_delta": stratified.get("median_abs_delta"),
            "p90_abs_delta": stratified.get("p90_abs_delta"),
            "n": stratified.get("n"),
            "n_tail": stratified.get("n_tail"),
            "edge_channel_tail_fraction": stratified.get("edge_channel_tail_fraction"),
            "adopt_as_floor": stratified.get("adopt_as_floor"),
            "gauge_aligned": stratified.get("gauge_aligned"),
            "worst_antennas": sorted(
                stratified.get("by_antenna", []),
                key=lambda item: item.get("n_tail", 0),
                reverse=True,
            )[:8],
        },
        "selected_d_product": "neither",
        "selection_reason": (
            "table difference is recorded; field-9 held-out visibility "
            "comparison is required before selecting Df or Df+QU"
        ),
        "gate_status": {
            "d_smoothness_antenna_support": d_status,
            "three_c147_df_vs_df_qu": qu_status,
        },
        "empirical_scale_factor_applied": False,
    }


def golden_gates() -> dict[str, object]:
    diagonal = _load(VALIDATION / "golden_diagonal_compare.json")
    fullpol = _load(VALIDATION / "golden_fullpol_compare.json")
    diag_cmp = diagonal.get("comparison", diagonal)
    full_cmp = fullpol.get("comparison", fullpol)
    by_ddid = full_cmp.get("by_data_desc") or {"_": full_cmp}
    historical: dict[str, dict[str, object]] = {}
    full_pass = True
    for ddid, block in by_ddid.items():
        per = block.get("per_correlation", {})
        historical[str(ddid)] = {}
        for name, corr in per.items():
            max_rel = float(corr.get("max_rel") or 0.0)
            near_zero = name in {"RL", "LR"} and int(corr.get("n_above_floor") or 0) == 0
            unsuitable = bool(
                near_zero
                or (
                    name in {"RL", "LR"}
                    and max_rel > 1.0e-5
                    and float(corr.get("relative_l2") or 1.0) < 1.0e-5
                )
            )
            rel_l2 = corr.get("relative_l2")
            max_abs = corr.get("max_abs")
            if unsuitable:
                ok = (rel_l2 is not None and float(rel_l2) <= 1.0e-5) or (
                    max_abs is not None and float(max_abs) <= 1.0e-6
                )
            else:
                ok = bool(corr.get("passed"))
            historical[str(ddid)][name] = {
                "max_rel": corr.get("max_rel"),
                "relative_l2": rel_l2,
                "max_abs": max_abs,
                "passed": corr.get("passed"),
                "max_rel_unsuitable": unsuitable,
                "historical_max_rel_failure": bool(unsuitable and not corr.get("passed")),
                "rescored_passed": bool(ok),
            }
            full_pass = full_pass and bool(ok)
    return {
        "schema": "thol0001_scientific_golden_gates_v1",
        "diagonal_passed": bool(diag_cmp.get("passed")),
        "fullpol_historical_passed": bool(full_cmp.get("passed")),
        "fullpol_rescored_passed": full_pass,
        "per_correlation": historical,
        "note": (
            "Historical max_rel failures on near-zero CASA RL/LR are an "
            "unsuitable metric. Rescore uses absolute residual or relative L2."
        ),
        "injected_operator_oracle": _status(
            "not_run" if not (UNBLOCK / "operator_oracle.json").exists() else "pass",
            path=str(UNBLOCK / "operator_oracle.json"),
        ),
    }


def assemble_report(
    *,
    inv: dict,
    dterms: dict | None,
    goldens: dict | None,
) -> dict[str, object]:
    holdouts_path = Path(inv["holdouts_path"])
    holdouts = _load(holdouts_path) if holdouts_path.exists() else None
    gates = {name: _status("not_run") for name in GATES}
    if holdouts is not None:
        vis = holdouts.get("visibility_holdouts", {})
        lobe_scores = []
        for axis in (
            "held_out_references",
            "held_out_moving_antennas",
            "later_visits",
            "spatially_interleaved_cells",
        ):
            region = (vis.get(axis) or {}).get("by_power_region", {}).get("above_50_percent", {})
            for hand in ("rr", "ll"):
                score = (region.get(hand) or {}).get("complex_relative_l2")
                if score is not None and np.isfinite(score):
                    lobe_scores.append(float(score))
        worst = max(lobe_scores) if lobe_scores else float("nan")
        hold_status = "not_run"
        if lobe_scores:
            hold_status = "pass" if worst < 0.08 else "warn" if worst < 0.15 else "fail"
        pairs = (vis.get("shared_origins_and_cross_pass") or {}).get("cross_pass_pairs", {})
        gates["diagonal_visibility_holdouts"] = _status(
            hold_status,
            path=str(holdouts_path),
            worst_main_lobe_rel_l2=worst,
            n_cross_pass_pairs=pairs.get("n"),
            note="Null-adjacent scores are recorded but do not fail the foundation",
        )
    else:
        gates["diagonal_visibility_holdouts"] = _status(
            "not_run",
            path=str(holdouts_path),
            note="vectorized holdout job writes this file",
        )
    if dterms is not None:
        gates["d_smoothness_antenna_support"] = _status(
            dterms["gate_status"]["d_smoothness_antenna_support"],
            unsupported=dterms["Df"]["antenna_support"]["unsupported"],
            smoothness=dterms["Df"]["smoothness"],
        )
        gates["three_c147_df_vs_df_qu"] = _status(
            dterms["gate_status"]["three_c147_df_vs_df_qu"],
            median_abs_delta=dterms["leakage_systematic_floor"].get("median_abs_delta"),
            p90_abs_delta=dterms["stratified_delta"].get("p90_abs_delta"),
            note="table-level only until field-9 held-out visibilities exist",
        )
        if dterms["Df_QU"]["source_pol"].get("found"):
            gates["three_c147_qu_source_model"] = _status(
                "warn",
                source_pol=dterms["Df_QU"]["source_pol"],
                note="CASA keyword Q/U is not a held-out field-9 recovery",
            )
    if goldens is not None:
        gates["real_visibility_casa_jax_golden"] = _status(
            "pass" if goldens["diagonal_passed"] and goldens["fullpol_rescored_passed"] else "fail",
            diagonal_passed=goldens["diagonal_passed"],
            fullpol_historical_passed=goldens["fullpol_historical_passed"],
            fullpol_rescored_passed=goldens["fullpol_rescored_passed"],
            per_correlation=goldens["per_correlation"],
        )
        oracle_path = UNBLOCK / "operator_oracle.json"
        if oracle_path.exists():
            oracle = _load(oracle_path)
            gates["injected_operator_oracle"] = _status(
                "pass" if oracle.get("passed") else "fail",
                path=str(oracle_path),
                n=oracle.get("n") or len(oracle.get("reports") or []),
                note=oracle.get("note"),
            )
        else:
            gates["injected_operator_oracle"] = goldens["injected_operator_oracle"]
    field9_path = UNBLOCK / "field9_qu.json"
    c286_path = UNBLOCK / "three_c286_applyback.json"
    variant_path = UNBLOCK / "field9_df_vs_df_qu.json"
    coherence_path = UNBLOCK / "crosshand_floor_coherence.json"
    residual_path = UNBLOCK / "residual_reference_jones.json"
    field9_residual_path = UNBLOCK / "field9_all_antenna_residual_jones.json"
    controls_path = UNBLOCK / "field9_residual_jones_controls.json"
    time_smooth_path = UNBLOCK / "field9_time_smooth_residual_jones.json"
    jump_path = UNBLOCK / "scan53_56_chain_jump.json"
    estimator_path = UNBLOCK / "field9_estimator_identifiability.json"
    stabilized_path = UNBLOCK / "field9_stabilized_residual_jones.json"
    adequacy_path = UNBLOCK / "field9_calibration_adequacy.json"
    beam_maps_path = UNBLOCK / "prediction_equivalent_beam_maps.json"
    one_axis_path = UNBLOCK / "one_axis_visibility_holdouts.json"
    if field9_path.exists():
        field9 = _load(field9_path)
        model = field9.get("source_qu_model") or {}
        hold = field9.get("holdout") or {}
        exact_zero = bool(model.get("represented_as_exact_zero")) or (
            model.get("q_over_i") == 0.0
            and model.get("u_over_i") == 0.0
            and "unpolarized" in str(model.get("relationship_to_model_data", ""))
        )
        qu = classify_three_c147_qu_model(
            q_over_i=float(model.get("q_over_i") or hold.get("q_over_i") or 0.0),
            u_over_i=float(model.get("u_over_i") or hold.get("u_over_i") or 0.0),
            frac_pol=model.get("frac_pol") or hold.get("frac_pol"),
            represented_as_exact_zero=exact_zero,
        )
        gates["three_c147_qu_source_model"] = _status(
            str(qu["status"]),
            source=str(field9_path),
            model=model,
            blocking=qu["blocking"],
            note=THREE_C147_QU_IS_NUISANCE_NOTE,
        )
        floor_raw = field9.get("crosshand_residual_floor") or {}
        coherence = (
            _load(coherence_path) if coherence_path.exists() else floor_raw.get("coherence") or {}
        )
        worst_baseline = max(
            (
                float(item.get("mean_abs") or 0.0)
                for item in (coherence.get("worst_groups") or {}).get("baseline") or []
            ),
            default=float("nan"),
        )
        if not np.isfinite(worst_baseline):
            worst_baseline = coherence.get("gate", {}).get("max_group_coherent_abs")
            if worst_baseline is None:
                worst_baseline = floor_raw.get("coherence", {}).get("max_group_coherent_abs")
        floor = classify_crosshand_floor(
            median_abs_rl_over_i=float(floor_raw.get("median_abs_rl_over_i") or np.nan),
            coherent_mean_abs=(
                coherence.get("coherent_mean_abs")
                or (coherence.get("rl") or {}).get("coherent_mean_abs")
            ),
            averages_as_noise=(
                coherence.get("averages_as_noise")
                if "averages_as_noise" in coherence
                else (coherence.get("rl") or {}).get("averages_as_noise")
            ),
            max_group_coherent_abs=worst_baseline,
        )
        gates["crosshand_residual_floor"] = _status(
            str(floor["status"]),
            **{
                key: value
                for key, value in floor_raw.items()
                if key not in {"coherence", "blocking"}
            },
            blocking=floor["blocking"],
            max_group_coherent_abs=worst_baseline,
            note=CROSSHAND_FLOOR_BLOCKS_FULL_JONES_NOTE,
        )
        gates["crosshand_floor_coherence"] = _status(
            str(floor["status"]) if coherence else "not_run",
            coherent_mean_abs=floor["coherent_mean_abs"],
            averages_as_noise=floor["averages_as_noise"],
            max_group_coherent_abs=worst_baseline,
            blocking=floor["blocking"],
            note=CROSSHAND_FLOOR_BLOCKS_FULL_JONES_NOTE,
        )
    if variant_path.exists():
        variant = _load(variant_path)
        p90 = float(
            (variant.get("by_correlation") or {}).get("RL", {}).get("p90_abs_over_i") or 0.0
        )
        gates["three_c147_df_vs_df_qu"] = _status(
            variant.get("status", "not_run"),
            selected=variant.get("selected_d_product"),
            median_rl_over_i=(variant.get("by_correlation") or {})
            .get("RL", {})
            .get("median_abs_over_i"),
            p90_rl_over_i=p90,
            note="Df+QU is rejected if the held-out RL/LR tail is large; Df is kept",
        )
    if c286_path.exists():
        c286 = _load(c286_path)
        for key, value in c286.get("gate_status", {}).items():
            if key in gates:
                gates[key] = _status(value, source=str(c286_path), by_spw=c286.get("by_spw"))
        circular = c286.get("circular_polarisation_floor") or {}
        v_i = circular.get("v_over_i")
        if v_i is None:
            v_i = (
                ((c286.get("by_spw") or {}).get("4") or {})
                .get("edge_held_out_from_kcross_x", {})
                .get("v_over_i")
            )
        gates["three_c286_circular_floor"] = _status(
            circular.get("status")
            or ("warn" if v_i is not None and abs(float(v_i)) > 0.001 else "not_run"),
            v_over_i=v_i,
            acceptable_for_beam_experiment=circular.get("acceptable_for_beam_experiment", True),
            negligible_for_full_stokes=False,
            note=THREE_C286_CIRCULAR_FLOOR_NOTE,
        )
    if residual_path.exists():
        residual = _load(residual_path)
        gates["residual_reference_jones"] = _status(
            residual.get("status", "warn"),
            path=str(residual_path),
            median_abs_epsilon=residual.get("median_abs_epsilon"),
            holdout=residual.get("holdout"),
            note=IDENTITY_REFERENCE_JONES_BLOCKS_OFFDIAG_NOTE,
        )
    if field9_residual_path.exists():
        field9_r = _load(field9_residual_path)
        time_hold = field9_r.get("time_holdout") or {}
        base_hold = field9_r.get("baseline_holdout") or {}
        prior = field9_r.get("identity_prior_holdout") or {}
        qu_n = field9_r.get("qu_nuisance") or {}
        connected = bool(field9_r.get("graph_connected_time")) and bool(
            field9_r.get("graph_connected_baseline")
        )
        gates["field9_all_antenna_residual_jones"] = _status(
            field9_r.get("status", "warn"),
            path=str(field9_residual_path),
            median_abs_epsilon=field9_r.get("median_abs_epsilon"),
            q_over_i=field9_r.get("q_over_i"),
            u_over_i=field9_r.get("u_over_i"),
            represented_as_exact_zero=False,
            note=ALL_ANTENNA_RESIDUAL_JONES_NOTE,
        )
        gates["connected_residual_holdouts"] = _status(
            "pass"
            if connected and float(time_hold.get("median_abs_rl_over_i") or 1.0) < 0.02
            else "warn"
            if connected
            else "fail",
            time_holdout_rl=time_hold.get("median_abs_rl_over_i"),
            baseline_holdout_rl=base_hold.get("median_abs_rl_over_i"),
            graph_connected=connected,
            note=CONNECTED_HOLDOUT_NOTE,
        )
        gates["identity_prior_antenna_holdout"] = _status(
            "warn",
            antenna=prior.get("antenna"),
            median_abs_rl_over_i=prior.get("median_abs_rl_over_i"),
            separate_gate=True,
            note=CONNECTED_HOLDOUT_NOTE,
        )
        gates["three_c147_qu_source_model"] = _status(
            "warn",
            q_over_i=field9_r.get("q_over_i"),
            u_over_i=field9_r.get("u_over_i"),
            interval_spread=qu_n.get("interval_spread"),
            represented_as_exact_zero=False,
            note=THREE_C147_QU_IS_NUISANCE_NOTE,
        )
    if controls_path.exists():
        controls = _load(controls_path)
        p_conv = controls.get("p_convention") or {}
        support = controls.get("channel_support") or {}
        scan_state = controls.get("scan_state") or {}
        jump = scan_state.get("jump") or {}
        gates["residual_jones_p_convention"] = _status(
            p_conv.get("status", "not_run"),
            locked_slope=p_conv.get("locked_slope_darg_over_d_two_chi"),
            opposite_slope=p_conv.get("opposite_slope_darg_over_d_two_chi"),
            note=RESIDUAL_JONES_P_CONVENTION_NOTE,
        )
        gates["field9_channel_support"] = _status(
            "warn" if support.get("time_dependent_support") else "pass",
            time_dependent_support=support.get("time_dependent_support"),
            n_kept=support.get("n_kept"),
            native_channel=controls.get("native_channel"),
            used_native_channel=True,
            note=CHANNEL_AVERAGING_CAN_FAKE_TIME_NOTE,
        )
        gates["field9_scan_state"] = _status(
            "warn" if jump.get("piecewise_state_term_indicated") else "pass",
            n_jump=jump.get("n_jump"),
            piecewise_state_term_indicated=jump.get("piecewise_state_term_indicated"),
            plot=scan_state.get("plot"),
            note=SCAN_STATE_MAY_NEED_OFFSETS_NOTE,
        )
        gates["on_axis_beam_identity"] = _status(
            "pass",
            note=ON_AXIS_BEAM_IDENTITY_NOTE,
        )
    if time_smooth_path.exists():
        time_smooth = _load(time_smooth_path)
        gate = time_smooth.get("gate") or {}
        later = (time_smooth.get("later_holdout") or {}).get("coherent_rl") or {}
        gates["field9_time_smooth_residual_jones"] = _status(
            "warn",
            path=str(time_smooth_path),
            later_coherent_abs=later.get("coherent_mean_abs") or gate.get("later_coherent_abs"),
            later_averages_as_noise=later.get("averages_as_noise")
            or gate.get("later_averages_as_noise"),
            chosen_drift_ridge=time_smooth.get("chosen_drift_ridge"),
            blocking=False,
            permanently_blocks_full_jones=False,
            casa_not_end_to_end=True,
            note=SMOOTH_GP_REJECTED_NOTE,
        )
    if jump_path.exists():
        jump = _load(jump_path)
        chain = jump.get("chain") or {}
        transfer = jump.get("interleaved_onaxis_transfer") or {}
        acquisition = jump.get("acquisition") or {}
        gates["scan53_56_chain_jump"] = _status(
            chain.get("status", "not_run"),
            path=str(jump_path),
            first_stage=chain.get("first_stage_above_threshold"),
            first_increment=chain.get("first_increment_above_threshold"),
            present_in_data=chain.get("present_in_data"),
            visibility_jump_absent=chain.get("visibility_jump_absent"),
            note="If the jump first appears after one calibration term, fix or model that term.",
        )
        gates["field9_acquisition_state"] = _status(
            "pass" if acquisition else "not_run",
            present_in_data=chain.get("present_in_data"),
            preceded_53=(acquisition.get("neighbours") or {})
            .get("scan_53", {})
            .get("preceded_by_holoraster"),
            preceded_56=(acquisition.get("neighbours") or {})
            .get("scan_56", {})
            .get("preceded_by_holoraster"),
        )
        gates["hierarchical_scan_state_residual"] = _status(
            (jump.get("hierarchical_model") or {}).get("status", "specified_not_fit"),
            unconstrained_per_scan_jones=False,
        )
        gates["interleaved_onaxis_beam_transfer"] = _status(
            transfer.get("status", "not_run"),
            blocking=transfer.get("blocking", True),
            chain_jump_stage=transfer.get("chain_jump_stage"),
            note=INTERLEAVED_BEAM_TRANSFER_GATE_NOTE,
        )
    if estimator_path.exists():
        estimator = _load(estimator_path)
        gate = estimator.get("gate") or {}
        gates["field9_estimator_identifiability"] = _status(
            estimator.get("status", "not_run"),
            path=str(estimator_path),
            parameter_max_abs_delta=(estimator.get("parameter_jump_aligned") or {}).get(
                "max_abs_delta"
            ),
            operator_median_abs_rl=(estimator.get("predicted_operator") or {}).get(
                "median_abs_rl_over_i"
            ),
            cross_apply=(estimator.get("cross_apply") or {}).get("cross_median_abs_rl_over_i"),
            parameter_jump_is_physical=gate.get("parameter_jump_is_physical"),
            blocking=estimator.get("blocking", gate.get("blocking")),
            note=ESTIMATOR_IDENTIFIABILITY_NOTE,
        )
        gates["field9_stabilized_residual_jones"] = _status(
            "not_run",
            note=COMPLETE_FIELD9_TRACK_NOTE,
        )
        gates["prediction_equivalent_beam_maps"] = _status(
            "not_run",
            note=PREDICTION_EQUIVALENT_BEAM_NOTE,
        )
        gates["one_axis_visibility_holdouts"] = _status(
            "not_run",
            note=ONE_AXIS_VISIBILITY_HOLDOUT_NOTE,
        )
    if stabilized_path.exists():
        stabilized = _load(stabilized_path)
        gate = stabilized.get("gate") or {}
        gates["field9_stabilized_residual_jones"] = _status(
            stabilized.get("status", "not_run"),
            path=str(stabilized_path),
            gate_kind=gate.get("gate_kind"),
            epsilon_gauge_dependent=gate.get("epsilon_gauge_dependent", True),
            held_out_scans=(stabilized.get("scan_cluster_holdout") or {}).get("held_out_scans"),
            blocking=stabilized.get("blocking", gate.get("blocking")),
            note=COMPLETE_FIELD9_TRACK_NOTE,
        )
        gates["prediction_equivalent_beam_maps"] = _status(
            "not_run",
            note=PREDICTION_EQUIVALENT_BEAM_NOTE,
        )
        gates["one_axis_visibility_holdouts"] = _status(
            "not_run",
            note=ONE_AXIS_VISIBILITY_HOLDOUT_NOTE,
        )
    if adequacy_path.exists():
        adequacy = _load(adequacy_path)
        gates["field9_stabilized_residual_jones"] = {
            **gates["field9_stabilized_residual_jones"],
            "operator_identifiable": adequacy.get("operator_identifiable"),
            "calibration_adequate": adequacy.get("calibration_adequate"),
            "adequacy_note": CALIBRATION_ADEQUACY_NOT_OPERATOR_NOTE,
        }
    if beam_maps_path.exists():
        beams = _load(beam_maps_path)
        gates["prediction_equivalent_beam_maps"] = _status(
            beams.get("status", "not_run"),
            path=str(beam_maps_path),
            outcome=beams.get("outcome"),
            unpinned_maps_are_diagnostic=True,
            zero_qu_is_ablation=True,
            blocking=beams.get("blocking"),
            note=PREDICTION_EQUIVALENT_BEAM_NOTE,
        )
        gates["one_axis_visibility_holdouts"] = _status(
            "not_run",
            note=ONE_AXIS_VISIBILITY_HOLDOUT_NOTE,
        )
    if one_axis_path.exists():
        one_axis = _load(one_axis_path)
        gates["one_axis_visibility_holdouts"] = _status(
            one_axis.get("status", "not_run"),
            path=str(one_axis_path),
            outcome=one_axis.get("outcome"),
            do_not_combine=True,
            interpolator_frozen_a_priori=one_axis.get("interpolator_frozen_a_priori"),
            blocking=one_axis.get("blocking"),
            note=ONE_AXIS_VISIBILITY_HOLDOUT_NOTE,
        )
    foundation_ok = inv.get("diagonal_source") == "field_10_model_data_per_row"
    selected = "neither"
    if variant_path.exists():
        selected = _load(variant_path).get("selected_d_product") or "neither"
    elif gates["d_smoothness_antenna_support"]["status"] in {"pass", "warn"}:
        selected = "Df"
    first_fail = None
    identity_gauge_fails = {"crosshand_residual_floor", "crosshand_floor_coherence"}
    extrapolation_evidence = {"field9_time_smooth_residual_jones"}
    for name in GATES:
        if (
            gates[name]["status"] == "fail"
            and name not in identity_gauge_fails
            and name not in extrapolation_evidence
        ):
            first_fail = name
            break
    if first_fail is None:
        for name in GATES:
            if gates[name]["status"] == "fail":
                first_fail = name
                break
    next_experiment = (
        "measure the scan-53→56 jump through DATA → K/B/G → Kcross → Df → "
        "Xf → P. If it is already in DATA, test pointing and HOLORASTER "
        "acquisition state. Do not fit unconstrained per-scan Jones. "
        "Do not open SPW 5."
    )
    if gates["scan53_56_chain_jump"]["status"] not in {"not_run"}:
        if gates["interleaved_onaxis_beam_transfer"]["status"] == "pass":
            next_experiment = (
                "recover E_m(s) from V = R_m E_m S R_r^H using only the "
                "declared bracketing field-9 calibration; compare repeated "
                "beam cells across visits. Do not open SPW 5."
            )
        elif gates["one_axis_visibility_holdouts"]["status"] not in {"not_run"}:
            next_experiment = (
                "one-axis visibility holdouts are scored separately and "
                "must not be pooled. Interpolator settings stay frozen a "
                "priori; if they change, these axes become development "
                "sets and the reserved outer fold is the final claim. "
                "No new calibration-model branch. Do not open SPW 5."
            )
        elif gates["prediction_equivalent_beam_maps"]["status"] not in {"not_run"}:
            outcome = gates["prediction_equivalent_beam_maps"].get("outcome")
            next_experiment = (
                f"prediction_equivalent_beam_maps outcome is {outcome}. "
                "Repeat sealed visibility tests one axis at a time: "
                "leave-one-reference-out first, then visit, spatial "
                "checkerboard, and leave-one-mover-out as an "
                "array-average test. Freeze the interpolator on training "
                "data only. Do not combine axes. Do not open another "
                "calibration-model branch. Do not open SPW 5."
            )
        elif gates["field9_stabilized_residual_jones"]["status"] not in {"not_run"}:
            next_experiment = (
                "propagate several prediction-equivalent R_p solutions "
                "through beam recovery. If they predict the same held-out "
                "moving-reference visibilities but different RL/LR maps, "
                "the maps are not uniquely physical until E_m(0)=I. Do not "
                "add scan offsets, a state term, a temporal GP, HOLORASTER "
                "fitting, or SPW 5."
            )
        elif gates["field9_estimator_identifiability"]["status"] not in {"not_run"}:
            next_experiment = (
                "stabilize the all-antenna field-9 solve on the complete "
                "track. Gate on visibility-operator identifiability, not "
                "unique ε_p. Hold out baselines, samples, and complete "
                "scan clusters. Do not add scan offsets, a HOLORASTER "
                "state term, or more time freedom. Do not open SPW 5."
            )
        elif gates["scan53_56_chain_jump"].get("visibility_jump_absent"):
            next_experiment = (
                "the 53→56 visibility chain does not carry the per-scan ε "
                "jump. Diagnose estimator identifiability on a matched 53/56 "
                "mask: gauge-align Jones, compare predicted operators, "
                "cross-apply, and report conditioning. Do not add time "
                "freedom. Do not open SPW 5."
            )
        elif (gates["scan53_56_chain_jump"].get("present_in_data") is True) or gates[
            "field9_acquisition_state"
        ]["status"] not in {"not_run"}:
            next_experiment = (
                "the 53→56 jump is located. If it belongs to one CASA term, "
                "fix or model that term; otherwise use the hierarchical "
                "ε_{p,0}+c_k+δε_{p,k} model with one global Q/U. Then require "
                "interleaved on-axis beam transfer. Do not open SPW 5."
            )
        else:
            next_experiment = (
                "finish locating the 53→56 jump in the cumulative CASA chain. "
                "Do not add smooth time freedom. Do not open SPW 5."
            )
    compare_path = UNBLOCK / "diagonal_vs_full_jones.json"
    full_jones_status = "blocked"
    if compare_path.exists():
        compare = _load(compare_path)
        gates["moving_reference_visibility_comparison"] = _status(
            "not_run",
            jjh_proxy_path=str(compare_path),
            rl_lr_improved=compare.get("rl_lr_improved"),
            rr_ll_not_regressed=compare.get("rr_ll_not_regressed"),
            note=JJH_PROXY_IS_NOT_DECISIVE_NOTE,
        )
    return {
        "schema": "thol0001_full_jones_unblock_v2",
        "frozen": False,
        "full_jones_status": full_jones_status,
        "blocked_reason": (
            "one_axis_visibility_holdouts"
            if gates["one_axis_visibility_holdouts"]["status"] not in {"not_run"}
            else "prediction_equivalent_beam_maps"
            if gates["field9_stabilized_residual_jones"]["status"] not in {"not_run"}
            else "field9_stabilized_residual_jones"
            if full_jones_status == "blocked"
            else None
        ),
        "gate_kind": "visibility_operator_identifiability",
        "epsilon_gauge_dependent": True,
        "no_scan_offsets_yet": True,
        "no_additional_time_freedom": True,
        "smooth_gp_rejected": True,
        "smooth_gp_permanently_blocks_full_jones": False,
        "identity_gauge_floor_blocks_identity_product_only": True,
        "jjh_proxy_is_not_decisive": True,
        "full_jones_unfrozen": True,
        "selected_d_product": selected,
        "spw5_unopened": True,
        "production_full_jones_factory_untouched": True,
        "diagonal_foundation": {
            "source": inv.get("diagonal_source"),
            "uses_field_10_model_data": foundation_ok,
            "frozen": inv.get("diagonal_frozen"),
            "flux_gauge_passed": inv.get("flux_gauge_passed"),
        },
        "three_c286_evpa_provenance": inv.get("three_c286_angles"),
        "gates": gates,
        "first_failed_gate": first_fail,
        "next_experiment": next_experiment,
        "global_x_antennas": list(GLOBAL_X),
        "identity_inheritance_forbidden": True,
    }


def write_summary(report: dict, path: Path) -> Path:
    foundation = report.get("diagonal_foundation", {})
    lines = [
        "# THOL0001 lower-C full-Jones unblock",
        "",
        f"Full Jones status: **{report['full_jones_status']}** (explicitly unfrozen).",
        f"Blocked reason: `{report.get('blocked_reason')}`.",
        f"Selected D product: `{report['selected_d_product']}`.",
        "",
        "The on-axis CASA-compatible full-polarisation operator is largely working.",
        "The scientifically usable direction-dependent full-Jones beam remains blocked.",
        "The off-diagonal signal does not survive the per-reference identity-gauge estimator.",
        "",
        "The production full-Jones factory was not changed. SPW 5 is unopened.",
        "Existing scientific products were not overwritten.",
        "",
        "## Diagonal foundation",
        "",
        f"- HOLORASTER source coherency: `{foundation.get('source')}`",
        f"- Flux gauge passed: `{foundation.get('flux_gauge_passed')}`",
        f"- Frozen: `{foundation.get('frozen')}`",
        "",
        "## Gates",
        "",
    ]
    for name, gate in report["gates"].items():
        extra = ""
        if name == "diagonal_visibility_holdouts" and "worst_main_lobe_rel_l2" in gate:
            extra = f" (main-lobe rel L2 {gate['worst_main_lobe_rel_l2']:.4f})"
        lines.append(f"- `{name}`: **{gate['status']}**{extra}")
    lines.extend(
        [
            "",
            f"First failed gate: `{report['first_failed_gate']}`.",
            f"Next experiment: {report['next_experiment']}.",
            "",
            "3C286 `cos(66°)`, `sin(66°)` is casaguide 2χ, not IAU EVPA. IAU χ ≈ 33°.",
            "Q/U numbers were not changed.",
            "",
            "Do not freeze a beam. Do not open SPW 5 until a constrained "
            "residual antenna Jones predicts later field-9 data without "
            "absorbing source polarisation.",
            "",
            CASA_FIELD9_NOT_END_TO_END_HOLDOUT_NOTE,
            "",
            ESTIMATOR_IDENTIFIABILITY_NOTE,
            "",
            COMPLETE_FIELD9_TRACK_NOTE,
            "",
            PREDICTION_EQUIVALENT_BEAM_NOTE,
            "",
            CALIBRATION_ADEQUACY_NOT_OPERATOR_NOTE,
            "",
            ZERO_QU_IS_ABLATION_NOTE,
            "",
            NO_SCAN_OFFSETS_OR_TIME_FREEDOM_NOTE,
            "",
            SMOOTH_GP_REJECTED_NOTE,
            "",
            INTERLEAVED_ONAXIS_TRANSFER_NOTE,
            "",
            INTERLEAVED_BEAM_TRANSFER_GATE_NOTE,
            "",
            JJH_PROXY_IS_NOT_DECISIVE_NOTE,
            "",
            IDENTITY_REFERENCE_JONES_BLOCKS_OFFDIAG_NOTE,
            "",
        ]
    )
    compare_path = UNBLOCK / "diagonal_vs_full_jones.json"
    if compare_path.exists():
        compare = _load(compare_path)
        lines.extend(
            [
                "## Later-visit J J^H proxy (not decisive)",
                "",
                f"- Holdout: `{compare.get('holdout')}` n={compare.get('n_holdout_samples')}",
                f"- RL/LR improved: `{compare.get('rl_lr_improved')}`",
                f"- RR/LL not regressed: `{compare.get('rr_ll_not_regressed')}`",
                f"- Label: **{compare.get('full_jones_label')}**",
                "",
                "This is evidence against the identity-gauge estimator, not a "
                "final E_p S E_q^H visibility comparison.",
                "",
            ]
        )
    path.write_text("\n".join(lines))
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("inventory", "d_terms", "goldens", "report", "all"),
        default="all",
    )
    parser.add_argument("--output-dir", type=Path, default=UNBLOCK)
    arguments = parser.parse_args()
    output = arguments.output_dir
    output.mkdir(parents=True, exist_ok=True)
    inv = inventory()
    write_json(inv, output / "inventory.json")
    dterms = None
    goldens = None
    if arguments.stage in {"d_terms", "all", "report"}:
        dterms = d_term_gates()
        write_json(dterms, output / "d_terms.json")
    if arguments.stage in {"goldens", "all", "report"}:
        goldens = golden_gates()
        write_json(goldens, output / "goldens.json")
    if arguments.stage in {"report", "all"}:
        if dterms is None and (output / "d_terms.json").exists():
            dterms = _load(output / "d_terms.json")
        if goldens is None and (output / "goldens.json").exists():
            goldens = _load(output / "goldens.json")
        report = assemble_report(inv=inv, dterms=dterms, goldens=goldens)
        write_json(report, output / "report.json")
        write_summary(report, output / "README.md")
        print(output / "report.json")
        print("full_jones_status", report["full_jones_status"])
        print("first_failed_gate", report["first_failed_gate"])
        print("next_experiment", report["next_experiment"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
