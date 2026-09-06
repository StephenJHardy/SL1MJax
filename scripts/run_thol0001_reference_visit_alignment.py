"""Run the SPW-4 reference/visit aligned copolar-transfer gate.

Fits constant complex 2x2 A_mv and B_rv from origin moving-reference
dwells and HOLORASTER reference-reference rows. Non-origin moving-
reference rows stay out of those fits. Exact-cell LORO, uncalibrated
visit transfer, and beam-repeatability holdouts are scored after
alignment. Earlier products are not overwritten. Full Jones stays
blocked until this gate passes.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import traceback
from pathlib import Path

import numpy as np

from sl1mjax.beam_operator import unique_visibility_times
from sl1mjax.calibration_terms import parallactic_angle_rad as parallactic_angle
from sl1mjax.holography import HolographyObservation
from sl1mjax.holography_alignment import (
    ALIGNMENT_NOTE,
    BEAM_REPEATABILITY,
    BEAM_REPEATABILITY_NOTE,
    COPOLAR_FLOOR,
    FULL_JONES_BLOCKED_UNTIL_ALIGNMENT_NOTE,
    REFERENCE_VISIT_ALIGNED_COPOLAR_TRANSFER,
    UNCALIBRATED_VISIT_NOTE,
    UNCALIBRATED_VISIT_TRANSFER,
    VOLTAGE_MAINLOBE,
    VOLTAGE_MID,
    _relative_cell_residual,
    align_recovered_jones,
    alignment_fit_mask,
    apparent_voltage_response,
    artifact_from_recovered_rows,
    classify_reference_visit_aligned_copolar_transfer,
    combine_alignment_transfer_gates,
    copy_visit_factors,
    direction_residual_after_alignment,
    estimate_mover_visit_A,
    estimate_reference_visit_B,
    factor_tables,
    holoraster_pair_masks,
    identical_cell_disagreement,
    predict_aligned_moving_reference,
    recover_rows,
    score_copolar_residuals,
    visit_holdout_masks,
    visit_id_per_time,
)
from sl1mjax.holography_calibration import (
    HELD_OUT_REFERENCE_ANTENNA,
    REFERENCE_ANTENNA,
    write_json,
)
from sl1mjax.holography_diagonal import (
    copolar_hand_active_rows,
    holography_row_exclusion_counts,
    source_model_stokes_i,
)
from sl1mjax.holography_full_jones import (
    DEVELOPMENT_SET_NOTE,
    INTERPOLATOR_NEIGHBOR_K,
    INTERPOLATOR_RADIUS_SCALE,
    classify_holdout_interpolation_support,
    freeze_interpolation_support_from_rows,
    moving_reference_row_geometry,
    predict_moving_reference_from_beam,
    thin_training_rows,
)
from sl1mjax.holography_ms import _read_antennas, _tables, audit_holography_measurement_set
from sl1mjax.holography_reference_jones import (
    NO_SCAN_OFFSETS_OR_TIME_FREEDOM_NOTE,
    deserialize_reference_jones,
)
from sl1mjax.polarization import Receptor, pack_coherency

SCIENTIFIC_MS = Path(
    "/media/stephen/astro/vla/extracted/commissioning/work/THOL0001.lowerC.spw45.scientific.ms"
)
UNBLOCK = Path(
    "/media/stephen/astro/vla/extracted/commissioning/validation/scientific/full_jones_unblock"
)
NATIVE_CHANNEL = 32
HELD_MOVER = "ea04"


def _diag():
    path = Path(__file__).with_name("run_thol0001_diagonal_recovery.py")
    spec = importlib.util.spec_from_file_location("thol0001_diagonal_recovery", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.complexfloating, complex)):
        number = complex(value)
        if not (np.isfinite(number.real) and np.isfinite(number.imag)):
            return None
        return [number.real, number.imag]
    return value


def _subsample(mask: np.ndarray, max_rows: int, *, seed: int) -> np.ndarray:
    selected = np.flatnonzero(mask)
    if selected.size <= int(max_rows):
        return np.asarray(mask, dtype=bool)
    keep = np.zeros(mask.shape, dtype=bool)
    keep[np.random.default_rng(seed).choice(selected, size=int(max_rows), replace=False)] = True
    return keep


def _region_score(scored: dict, name: str) -> dict:
    return scored.get(name) or {}


def _gate_from_scores(unaligned: dict, aligned: dict, direction: dict, *, kind: str) -> dict:
    main_un = _region_score(unaligned, "main_lobe")
    main_al = _region_score(aligned, "main_lobe")
    availability = {
        UNCALIBRATED_VISIT_TRANSFER: "insufficient_evidence",
        BEAM_REPEATABILITY: "not_testable",
    }.get(kind, "available")
    return classify_reference_visit_aligned_copolar_transfer(
        unaligned_rr=float(main_un.get("median_abs_rr_over_i") or np.nan),
        unaligned_ll=float(main_un.get("median_abs_ll_over_i") or np.nan),
        aligned_rr=float(main_al.get("median_abs_rr_over_i") or np.nan),
        aligned_ll=float(main_al.get("median_abs_ll_over_i") or np.nan),
        direction_dependent=direction.get("direction_dependent"),
        n_finite=int(main_al.get("n_finite") or 0),
        availability=availability,
    )


def _factor_summary(factors: dict) -> dict[str, object]:
    if not factors:
        return {"n": 0, "median_abs_diag": None, "median_abs_offdiag": None}
    planes = np.stack([np.asarray(plane) for plane in factors.values()], axis=0)
    diag = np.abs(np.stack([planes[:, 0, 0], planes[:, 1, 1]], axis=0))
    off = np.abs(np.stack([planes[:, 0, 1], planes[:, 1, 0]], axis=0))
    return {
        "n": int(planes.shape[0]),
        "median_abs_diag": float(np.median(diag)),
        "median_abs_offdiag": float(np.median(off)),
    }


def _fit_factors(observation, *, residual, chi, visits, refant, row_mask):
    allowed = alignment_fit_mask(observation, exclude=~np.asarray(row_mask, dtype=bool))
    b_fit = estimate_reference_visit_B(
        observation,
        residual_jones=residual,
        parallactic_angle_rad=chi,
        reference_antenna_id=int(refant),
        visit_id=visits,
        row_mask=allowed,
    )
    a_fit = estimate_mover_visit_A(
        observation,
        residual_jones=residual,
        parallactic_angle_rad=chi,
        visit_id=visits,
        reference_factors=b_fit,
        row_mask=allowed,
    )
    return allowed, a_fit, b_fit


def _n_ant(observation) -> int:
    return int(np.max(observation.pointing.antenna_id)) + 1


def _run_holdout(
    observation,
    *,
    kind: str,
    residual,
    chi,
    visits,
    packed_plane,
    intensity,
    rr_ok,
    ll_ok,
    voltage,
    geometry,
    refant: int,
    held_visit: int,
    held_reference_id: int | None,
    max_train_per_cell: int,
    max_score_rows: int,
    transfer_from_visit: int | None,
):
    print("holdout", kind, flush=True)
    masks = visit_holdout_masks(
        observation,
        kind=kind,
        visit_id=visits,
        held_visit=held_visit,
        held_reference_id=held_reference_id,
        reserve_outer_fold=True,
    )
    allowed, a_fit, b_fit = _fit_factors(
        observation,
        residual=residual,
        chi=chi,
        visits=visits,
        refant=refant,
        row_mask=~masks["alignment_exclude"],
    )
    if transfer_from_visit is not None:
        a_fit = copy_visit_factors(a_fit, source_visit=transfer_from_visit, dest_visit=held_visit)
        b_fit = copy_visit_factors(b_fit, source_visit=transfer_from_visit, dest_visit=held_visit)
    n_visit = int(np.max(visits)) + 1
    n_ant = _n_ant(observation)
    a_table, b_table = factor_tables(n_visit, n_ant, a_fit, b_fit)
    identity_a, identity_b = factor_tables(n_visit, n_ant)
    train = thin_training_rows(observation, masks["train"], max_per_cell=max_train_per_cell)
    support = freeze_interpolation_support_from_rows(observation, train)
    recovered, meta = recover_rows(
        observation,
        residual_jones=residual,
        parallactic_angle_rad=chi,
        row_mask=train,
        mover_factors=a_table,
        reference_factors=b_table,
        visit_id=visits,
    )
    artifact = artifact_from_recovered_rows(observation, recovered, meta)
    unaligned_recovered, unaligned_meta = recover_rows(
        observation,
        residual_jones=residual,
        parallactic_angle_rad=chi,
        row_mask=train,
    )
    unaligned_artifact = artifact_from_recovered_rows(
        observation, unaligned_recovered, unaligned_meta
    )
    labeled = classify_holdout_interpolation_support(support, geometry, masks["scored"])
    score_mask = _subsample(masks["scored"], max_score_rows, seed=32)
    exact = score_mask & labeled["exact"]
    aligned_pred = predict_aligned_moving_reference(
        observation,
        residual_jones=residual,
        artifact=artifact,
        row_mask=score_mask,
        parallactic_angle_rad=chi,
        visit_id=visits,
        mover_factors=a_table,
        reference_factors=b_table,
    )
    unaligned_pred = predict_moving_reference_from_beam(
        observation,
        residual_jones=residual,
        artifact=unaligned_artifact,
        row_mask=score_mask,
        parallactic_angle_rad=chi,
    )
    unaligned = score_copolar_residuals(
        packed_plane, unaligned_pred, intensity, rr_ok, ll_ok, exact, voltage
    )
    aligned = score_copolar_residuals(
        packed_plane, aligned_pred, intensity, rr_ok, ll_ok, exact, voltage
    )
    supported = score_mask & labeled["supported"]
    unaligned_supported = score_copolar_residuals(
        packed_plane, unaligned_pred, intensity, rr_ok, ll_ok, supported, voltage
    )
    aligned_supported = score_copolar_residuals(
        packed_plane, aligned_pred, intensity, rr_ok, ll_ok, supported, voltage
    )
    radius = np.asarray(geometry["radius_rad"], dtype=np.float64)
    scientific = exact & (np.asarray(voltage) >= VOLTAGE_MID)
    rr_over_i = np.full(exact.shape, np.nan, dtype=np.float64)
    ll_over_i = np.full(exact.shape, np.nan, dtype=np.float64)
    pred = aligned_pred[:, 0] if aligned_pred.ndim == 4 else aligned_pred
    scale = np.maximum(np.abs(intensity), 1.0e-3)
    rr_sel = exact & rr_ok & np.isfinite(pred[:, 0, 0])
    ll_sel = exact & ll_ok & np.isfinite(pred[:, 1, 1])
    rr_over_i[rr_sel] = np.abs(packed_plane[rr_sel, 0, 0] - pred[rr_sel, 0, 0]) / scale[rr_sel]
    ll_over_i[ll_sel] = np.abs(packed_plane[ll_sel, 1, 1] - pred[ll_sel, 1, 1]) / scale[ll_sel]
    direction = direction_residual_after_alignment(
        radius[scientific],
        rr_over_i[scientific],
        ll_over_i[scientific],
    )
    gate = _gate_from_scores(unaligned, aligned, direction, kind=kind)
    main = aligned.get("main_lobe") or {}
    print(
        " ",
        "train",
        int(np.sum(train)),
        "exact",
        int(np.sum(exact)),
        "main_lobe_rr",
        main.get("median_abs_rr_over_i"),
        "main_lobe_rr_jy",
        main.get("median_abs_rr_jy"),
        gate["outcome"],
        flush=True,
    )
    return {
        "kind": kind,
        "counts": {
            "usable": int(np.sum(masks["usable"])),
            "train": int(np.sum(masks["train"])),
            "train_thinned": int(np.sum(train)),
            "holdout": int(np.sum(masks["holdout"])),
            "reserved": int(np.sum(masks["reserved"])),
            "scored": int(np.sum(masks["scored"])),
            "alignment_rows": int(np.sum(allowed)),
            "n_A": len(a_fit),
            "n_B": len(b_fit),
        },
        "holdout_support": {
            key: labeled[key]
            for key in (
                "n_holdout",
                "n_supported",
                "support_fraction",
                "n_exact",
                "n_interpolation",
                "n_extrapolation",
                "n_unsupported",
            )
        },
        "unaligned_exact_cell": unaligned,
        "aligned_exact_cell": aligned,
        "unaligned_supported": unaligned_supported,
        "aligned_supported": aligned_supported,
        "direction_residual": direction,
        "gate": gate,
        "transferred_visit_factors": transfer_from_visit is not None,
        "factors": {"A": _factor_summary(a_fit), "B": _factor_summary(b_fit)},
        "_scored_rows": {
            "rows": np.flatnonzero(score_mask).astype(np.int32),
            "exact": exact[score_mask],
            "voltage": np.asarray(voltage, dtype=np.float64)[score_mask],
            "intensity": np.asarray(intensity, dtype=np.float64)[score_mask],
            "radius_rad": radius[score_mask],
            "measured_rr": packed_plane[score_mask, 0, 0],
            "measured_ll": packed_plane[score_mask, 1, 1],
            "aligned_rr": pred[score_mask, 0, 0],
            "aligned_ll": pred[score_mask, 1, 1],
            "rr_ok": rr_ok[score_mask],
            "ll_ok": ll_ok[score_mask],
        },
    }


def _map_disagreement(observation, *, residual, chi, visits, geometry, refant: int, max_rows: int):
    usable = _subsample(geometry["usable"], max_rows, seed=7)
    recovered, meta = recover_rows(
        observation,
        residual_jones=residual,
        parallactic_angle_rad=chi,
        row_mask=usable,
    )
    n_visit = int(np.max(visits)) + 1
    allowed, a_fit, b_fit = _fit_factors(
        observation,
        residual=residual,
        chi=chi,
        visits=visits,
        refant=int(refant),
        row_mask=np.ones(observation.block.time_s.size, dtype=bool),
    )
    a_table, b_table = factor_tables(n_visit, _n_ant(observation), a_fit, b_fit)
    visit = visits[meta["time_index"]]
    a_m = a_table[visit, meta["moving_id"]]
    b_r = b_table[visit, meta["reference_id"]]
    aligned = align_recovered_jones(recovered, a_m, b_r)
    cell_l = np.rint(meta["offset_lm_rad"][:, 0] * 1.0e6).astype(np.int64)
    cell_m = np.rint(meta["offset_lm_rad"][:, 1] * 1.0e6).astype(np.int64)
    keys = np.stack(
        [meta["moving_id"], cell_l, cell_m, meta["reference_id"], visit],
        axis=1,
    )
    report = identical_cell_disagreement(recovered, keys, aligned=aligned)
    leftover = _relative_cell_residual(aligned, keys[:, :3])
    direction = direction_residual_after_alignment(
        meta["radius_rad"],
        leftover[:, 0, 0],
        leftover[:, 1, 1],
    )
    return {
        "n_recovered": int(recovered.shape[0]),
        "alignment_rows": int(np.sum(allowed)),
        "n_A": len(a_fit),
        "n_B": len(b_fit),
        "disagreement": report,
        "aligned_direction_residual": direction,
    }


def _write_plots(output_dir: Path, payload: dict) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written = []
    tests = payload.get("tests") or {}
    names = []
    un_rr = []
    al_rr = []
    un_ll = []
    al_ll = []
    for name, report in tests.items():
        names.append(name)
        unaligned = (report.get("unaligned_exact_cell") or {}).get("main_lobe") or {}
        aligned = (report.get("aligned_exact_cell") or {}).get("main_lobe") or {}
        un_rr.append(unaligned.get("median_abs_rr_over_i"))
        al_rr.append(aligned.get("median_abs_rr_over_i"))
        un_ll.append(unaligned.get("median_abs_ll_over_i"))
        al_ll.append(aligned.get("median_abs_ll_over_i"))
    if names:
        fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
        x = np.arange(len(names))
        axes[0].bar(x - 0.15, un_rr, width=0.3, label="unaligned RR")
        axes[0].bar(x + 0.15, al_rr, width=0.3, label="aligned RR")
        axes[0].axhline(COPOLAR_FLOOR, color="0.4", ls="--", lw=0.8)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(names, rotation=20, ha="right", fontsize=8)
        axes[0].set_ylabel("|RR residual| / I_model (voltage>=0.5)")
        axes[0].legend(fontsize=8)
        axes[1].bar(x - 0.15, un_ll, width=0.3, label="unaligned LL")
        axes[1].bar(x + 0.15, al_ll, width=0.3, label="aligned LL")
        axes[1].axhline(COPOLAR_FLOOR, color="0.4", ls="--", lw=0.8)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(names, rotation=20, ha="right", fontsize=8)
        axes[1].set_ylabel("|LL residual| / I_model (voltage>=0.5)")
        axes[1].legend(fontsize=8)
        fig.tight_layout()
        path = output_dir / "aligned_vs_unaligned_copolar.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        written.append(str(path.name))
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), sharey=True)
    regions = ("main_lobe", "mid", "outer_diagnostic")
    for axis, region in zip(axes, regions, strict=True):
        labels = []
        values = []
        for name, report in tests.items():
            labels.append(name)
            values.append(
                ((report.get("aligned_exact_cell") or {}).get(region) or {}).get(
                    "median_abs_rr_over_i"
                )
            )
        axis.bar(np.arange(len(labels)), values)
        axis.axhline(COPOLAR_FLOOR, color="0.4", ls="--", lw=0.8)
        axis.set_xticks(np.arange(len(labels)))
        axis.set_xticklabels(labels, rotation=20, ha="right", fontsize=7)
        axis.set_title(region.replace("_", " "))
    axes[0].set_ylabel("|RR residual| / I_model")
    fig.tight_layout()
    path = output_dir / "residuals_by_voltage_region.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    written.append(str(path.name))
    return written


def _readme(payload: dict) -> str:
    gate = payload.get("gate") or {}
    tests = payload.get("tests") or {}
    lines = [
        "# THOL0001 SPW-4 reference/visit aligned copolar transfer",
        "",
        f"Status: **{gate.get('status', 'unknown')}**. "
        f"Outcome: `{gate.get('outcome', 'unknown')}`.",
        "",
        ALIGNMENT_NOTE,
        "",
        UNCALIBRATED_VISIT_NOTE,
        "",
        BEAM_REPEATABILITY_NOTE,
        "",
        FULL_JONES_BLOCKED_UNTIL_ALIGNMENT_NOTE,
        "",
        "Alignment fits use settled four-hand origin moving-reference "
        "dwells for A_mv and reference-reference rows for B_rv, with "
        "B_rv inside the RIME inversion. Residuals are "
        "|V-pred| / I_model and |V-pred| in Jy. I_model is "
        "|(S_RR + S_LL)/2| from source_coherency_visibility.",
        "",
        f"Voltage-response bins: main lobe >= {VOLTAGE_MAINLOBE}, "
        f"mid {VOLTAGE_MID}-{VOLTAGE_MAINLOBE}, outer < {VOLTAGE_MID} "
        "(diagnostic only). The gate uses the main lobe.",
        "",
    ]
    exclusion = payload.get("row_exclusion") or {}
    if exclusion:
        exclusive = exclusion.get("exclusive") or {}
        lines.extend(
            [
                "## Row selection",
                "",
                f"- unsettled: {exclusion.get('n_unsettled')}",
                f"- any flag: {exclusion.get('n_any_flag')}",
                f"- invalid weight: {exclusion.get('n_invalid_weight')}",
                f"- incomplete coherency: {exclusion.get('n_incomplete_coherency')}",
                f"- exclusive kept: {exclusive.get('kept')}",
                "",
            ]
        )
    lines.extend(["## Exact-cell copolar residuals", ""])
    for name, report in tests.items():
        aligned = report.get("aligned_exact_cell") or {}
        unaligned = report.get("unaligned_exact_cell") or {}
        item = report.get("gate") or {}
        main_al = aligned.get("main_lobe") or {}
        main_un = unaligned.get("main_lobe") or {}
        lines.append(
            f"- `{name}` main lobe: unaligned RR/LL "
            f"{main_un.get('median_abs_rr_over_i')} / "
            f"{main_un.get('median_abs_ll_over_i')}; aligned "
            f"{main_al.get('median_abs_rr_over_i')} / "
            f"{main_al.get('median_abs_ll_over_i')} "
            f"({main_al.get('median_abs_rr_jy')} / "
            f"{main_al.get('median_abs_ll_jy')} Jy); "
            f"{item.get('outcome')} ({item.get('status')})."
        )
        for region in ("mid", "outer_diagnostic"):
            bin_al = aligned.get(region) or {}
            lines.append(
                f"  - {region}: RR/LL "
                f"{bin_al.get('median_abs_rr_over_i')} / "
                f"{bin_al.get('median_abs_ll_over_i')} "
                f"({bin_al.get('median_abs_rr_jy')} / "
                f"{bin_al.get('median_abs_ll_jy')} Jy)."
            )
    lines.extend(
        [
            "",
            "Full Jones remains blocked. This is not an interpolation, "
            "temporal-GP, or CASSBEAM experiment.",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--measurement-set", type=Path, default=SCIENTIFIC_MS)
    parser.add_argument("--product-dir", type=Path, default=UNBLOCK)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=UNBLOCK / "reference_visit_alignment",
    )
    parser.add_argument("--spectral-window", type=int, default=4)
    parser.add_argument("--channel", type=int, default=NATIVE_CHANNEL)
    parser.add_argument("--max-train-per-cell", type=int, default=4)
    parser.add_argument("--max-score-rows", type=int, default=15000)
    parser.add_argument("--max-map-rows", type=int, default=20000)
    arguments = parser.parse_args()
    if int(arguments.spectral_window) != 4:
        raise ValueError("SPW 5 stays closed")
    if arguments.output_dir.resolve() in {
        arguments.product_dir.resolve(),
        (arguments.product_dir / "forward_closure").resolve(),
    }:
        raise ValueError("refusing to overwrite earlier scientific products")
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "thol0001_reference_visit_aligned_copolar_transfer_v2",
        "gate_name": REFERENCE_VISIT_ALIGNED_COPOLAR_TRANSFER,
        "status": "fail",
        "blocking": True,
        "spectral_window_id": 4,
        "native_channel": int(arguments.channel),
        "held_out_reference": HELD_OUT_REFERENCE_ANTENNA,
        "held_out_mover": HELD_MOVER,
        "reference_antenna": REFERENCE_ANTENNA,
        "copolar_floor": COPOLAR_FLOOR,
        "interpolator_neighbor_k": INTERPOLATOR_NEIGHBOR_K,
        "interpolator_radius_scale": INTERPOLATOR_RADIUS_SCALE,
        "reserved_outer_fold": True,
        "spw5_closed": True,
        "full_jones_blocked": True,
        "residual_denominator": "abs((S_RR + S_LL) / 2)",
        "sample_mask": {
            "fit": "settled_four_hand_finite_positive_weight",
            "score": "per_hand_rr_ll",
        },
        "visit_labels": "raster_pass_scan_groups",
        "voltage_regions": {
            "main_lobe": VOLTAGE_MAINLOBE,
            "mid": VOLTAGE_MID,
            "outer_diagnostic": 0.0,
        },
        "cheap_rescore_possible": False,
        "cheap_rescore_note": (
            "The previous run retained only summary JSON, so a source-I "
            "rescore of the reported 40% was not possible. This run writes "
            "scored_rows_*.npz for later rescoring."
        ),
        "do_not_overwrite": [
            "one_axis_visibility_holdouts.json",
            "one_axis_visibility_holdouts_v2.json",
            "forward_closure/",
        ],
        "tests": {},
        "notes": (
            ALIGNMENT_NOTE,
            UNCALIBRATED_VISIT_NOTE,
            BEAM_REPEATABILITY_NOTE,
            FULL_JONES_BLOCKED_UNTIL_ALIGNMENT_NOTE,
            DEVELOPMENT_SET_NOTE,
            NO_SCAN_OFFSETS_OR_TIME_FREEDOM_NOTE,
        ),
    }
    try:
        tables = _tables()
        ids, names, positions = _read_antennas(tables, arguments.measurement_set)
        names = np.asarray(names)
        positions = np.asarray(positions, dtype=np.float64)
        name_to_id = {str(name): int(ant) for ant, name in zip(ids, names, strict=True)}
        family_cache = arguments.product_dir / "prediction_equivalent_family.json"
        stabilized = arguments.product_dir / "field9_stabilized_residual_jones.json"
        if family_cache.exists():
            residual = deserialize_reference_jones(
                json.loads(family_cache.read_text())["nominal"]["jones"]
            )
        elif stabilized.exists():
            residual = deserialize_reference_jones(json.loads(stabilized.read_text())["jones"])
        else:
            raise FileNotFoundError(
                "need field9_stabilized_residual_jones.json or the family cache"
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
        unique_times, _ = unique_visibility_times(block.time_s)
        chi = parallactic_angle(unique_times, observation.phase_centre_rad, positions)
        packed = pack_coherency(block.visibility, block.correlations, (Receptor.R, Receptor.L))
        packed_plane = packed[:, 0] if packed.ndim == 4 else packed
        intensity = source_model_stokes_i(observation)
        rr_ok, ll_ok = copolar_hand_active_rows(observation)
        voltage = apparent_voltage_response(packed_plane, intensity, rr_ok, ll_ok)
        payload["row_exclusion"] = holography_row_exclusion_counts(observation)
        payload["source_i"] = {
            "median": float(np.median(intensity)),
            "min": float(np.min(intensity)),
            "max": float(np.max(intensity)),
        }
        visits = visit_id_per_time(observation)
        geometry = moving_reference_row_geometry(observation)
        pair = holoraster_pair_masks(observation)
        print(
            "loaded",
            "rows",
            int(observation.block.time_s.size),
            "visits",
            {int(visit): int(np.sum(visits == visit)) for visit in np.unique(visits)},
            "origin",
            int(np.sum(pair["origin"])),
            "ref_ref",
            int(np.sum(pair["reference_reference"])),
            "exclusion",
            payload["row_exclusion"]["exclusive"],
            flush=True,
        )
        held_ref = int(name_to_id[HELD_OUT_REFERENCE_ANTENNA])
        refant = int(name_to_id[REFERENCE_ANTENNA])
        held_visit = int(np.max(visits)) if int(np.unique(visits).size) > 1 else 0
        source_visit = 0 if held_visit != 0 else int(np.min(visits))
        payload["visit_counts"] = {
            str(int(visit)): int(np.sum(visits == visit)) for visit in np.unique(visits)
        }
        payload["pair_counts"] = {
            "moving_reference": int(np.sum(pair["moving_reference"])),
            "reference_reference": int(np.sum(pair["reference_reference"])),
            "origin": int(np.sum(pair["origin"])),
        }
        payload["held_visit"] = held_visit
        payload["map_disagreement"] = _map_disagreement(
            observation,
            residual=residual,
            chi=chi,
            visits=visits,
            geometry=geometry,
            refant=refant,
            max_rows=int(arguments.max_map_rows),
        )
        payload["tests"]["leave_one_reference_out"] = _run_holdout(
            observation,
            kind="leave_one_reference_out",
            residual=residual,
            chi=chi,
            visits=visits,
            packed_plane=packed_plane,
            intensity=intensity,
            rr_ok=rr_ok,
            ll_ok=ll_ok,
            voltage=voltage,
            geometry=geometry,
            refant=refant,
            held_visit=held_visit,
            held_reference_id=held_ref,
            max_train_per_cell=int(arguments.max_train_per_cell),
            max_score_rows=int(arguments.max_score_rows),
            transfer_from_visit=None,
        )
        payload["tests"][UNCALIBRATED_VISIT_TRANSFER] = _run_holdout(
            observation,
            kind=UNCALIBRATED_VISIT_TRANSFER,
            residual=residual,
            chi=chi,
            visits=visits,
            packed_plane=packed_plane,
            intensity=intensity,
            rr_ok=rr_ok,
            ll_ok=ll_ok,
            voltage=voltage,
            geometry=geometry,
            refant=refant,
            held_visit=held_visit,
            held_reference_id=None,
            max_train_per_cell=int(arguments.max_train_per_cell),
            max_score_rows=int(arguments.max_score_rows),
            transfer_from_visit=source_visit,
        )
        payload["tests"][BEAM_REPEATABILITY] = _run_holdout(
            observation,
            kind=BEAM_REPEATABILITY,
            residual=residual,
            chi=chi,
            visits=visits,
            packed_plane=packed_plane,
            intensity=intensity,
            rr_ok=rr_ok,
            ll_ok=ll_ok,
            voltage=voltage,
            geometry=geometry,
            refant=refant,
            held_visit=held_visit,
            held_reference_id=None,
            max_train_per_cell=int(arguments.max_train_per_cell),
            max_score_rows=int(arguments.max_score_rows),
            transfer_from_visit=None,
        )
        gate = combine_alignment_transfer_gates(
            leave_one_reference_out=payload["tests"]["leave_one_reference_out"]["gate"],
            beam_repeatability=payload["tests"][BEAM_REPEATABILITY]["gate"],
            uncalibrated_visit_transfer=payload["tests"][UNCALIBRATED_VISIT_TRANSFER]["gate"],
        )
        payload["gate"] = gate
        payload["status"] = gate["status"]
        payload["blocking"] = gate["blocking"]
        payload["comparison_fair"] = gate["comparison_fair"]
        payload["scored_row_files"] = []
        for name, report in payload["tests"].items():
            scored = report.pop("_scored_rows", None)
            if scored is None:
                continue
            path = arguments.output_dir / f"scored_rows_{name}.npz"
            np.savez_compressed(path, **scored)
            payload["scored_row_files"].append(str(path.name))
    except Exception as error:
        print("error", error, flush=True)
        traceback.print_exc()
        payload["error"] = f"{type(error).__name__}: {error}"
        payload["gate"] = {
            "gate": REFERENCE_VISIT_ALIGNED_COPOLAR_TRANSFER,
            "status": "fail",
            "blocking": True,
            "outcome": "runner_error",
            "full_jones_blocked": True,
            "comparison_fair": False,
        }
    try:
        payload["plots"] = _write_plots(arguments.output_dir, payload)
    except Exception as error:
        print("plot_error", error, flush=True)
        payload["plots"] = []
        payload["plot_error"] = f"{type(error).__name__}: {error}"
    (arguments.output_dir / "README.md").write_text(_readme(payload))
    try:
        write_json(
            _jsonable(payload), arguments.output_dir / "reference_visit_alignment_report.json"
        )
    except Exception as error:
        print("json_error", error, flush=True)
        payload["json_error"] = f"{type(error).__name__}: {error}"
        compact = {key: payload[key] for key in payload if key not in {"tests", "map_disagreement"}}
        compact["tests"] = {
            name: {key: report[key] for key in report if key != "factors"}
            if isinstance(report, dict)
            else report
            for name, report in (payload.get("tests") or {}).items()
        }
        write_json(
            _jsonable(compact), arguments.output_dir / "reference_visit_alignment_report.json"
        )
    print(arguments.output_dir / "reference_visit_alignment_report.json")
    print("status", payload.get("status"), (payload.get("gate") or {}).get("outcome"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
