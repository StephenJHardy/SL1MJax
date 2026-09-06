"""SPW-4 LORO full-Jones versus its diagonal projection.

Uses source-normalized, settled four-hand exact cells. Recovers with a
training-only off-diagonal SNR mask, zeros unsupported leaks, and keeps
those rows in the total score. Repeats leave-one-reference-out over every
usable holography reference. SPW 5 stays closed until this experiment is
technically sound. Full Jones stays unfrozen.
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
    FULL_JONES_BLOCKED_UNTIL_ALIGNMENT_NOTE,
    LORO_FULL_VERSUS_DIAGONAL,
    apparent_voltage_response,
    holoraster_pair_masks,
)
from sl1mjax.holography_calibration import write_json
from sl1mjax.holography_diagonal import (
    copolar_hand_active_rows,
    holography_row_exclusion_counts,
    source_model_stokes_i,
)
from sl1mjax.holography_full_jones import (
    DEVELOPMENT_SET_NOTE,
    LORO_HOLDOUT_NOTE,
    classify_holdout_interpolation_support,
    classify_loro_full_versus_diagonal,
    combine_loro_full_versus_diagonal,
    freeze_interpolation_support_from_rows,
    holography_one_axis_holdout_masks,
    moving_reference_row_geometry,
    predict_moving_reference_from_beam,
    recover_holography_full_jones,
    refactor_on_axis_gauge,
    score_full_versus_diagonal_regions,
    thin_training_rows,
    zero_unsupported_off_diagonals,
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
MIN_REFERENCE_ROWS = 1000


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


def _cell_ids(geometry: dict[str, np.ndarray]) -> np.ndarray:
    moving = np.asarray(geometry["moving_id"], dtype=np.int64)
    cell_l = np.asarray(geometry["cell_l"], dtype=np.int64)
    cell_m = np.asarray(geometry["cell_m"], dtype=np.int64)
    packed = moving.astype(np.int64) * 2_000_000_003 + cell_l * 10_007 + cell_m
    return packed


def _usable_references(observation, id_to_name: dict[int, str]) -> list[tuple[str, int]]:
    pair = holoraster_pair_masks(observation)
    usable = pair["four_hand_moving_reference"]
    references = np.asarray(pair["reference_id"], dtype=np.int32)
    found = []
    for antenna in np.unique(references[usable]):
        if int(np.sum(usable & (references == int(antenna)))) < MIN_REFERENCE_ROWS:
            continue
        name = id_to_name.get(int(antenna), f"antenna_{int(antenna)}")
        found.append((str(name), int(antenna)))
    return found


def _run_reference(
    observation,
    *,
    name: str,
    held_ref: int,
    residual_jones,
    chi,
    packed_plane,
    intensity,
    voltage,
    geometry,
    max_train_per_cell: int,
):
    print("loro", name, flush=True)
    masks = holography_one_axis_holdout_masks(
        observation,
        axis="leave_one_reference_out",
        held_reference_id=int(held_ref),
        reserve_outer_fold=True,
    )
    train = thin_training_rows(observation, masks["train"], max_per_cell=max_train_per_cell)
    support = freeze_interpolation_support_from_rows(observation, train)
    labeled = classify_holdout_interpolation_support(support, geometry, masks["scored"])
    exact = np.asarray(masks["scored"], dtype=bool) & labeled["exact"]
    unpinned = recover_holography_full_jones(
        observation,
        residual_jones=residual_jones,
        parallactic_angle_rad=chi,
        row_mask=train,
    )
    pinned, residual, _origin = refactor_on_axis_gauge(unpinned, residual_jones)
    artifact = zero_unsupported_off_diagonals(pinned)
    pred_full = predict_moving_reference_from_beam(
        observation,
        residual_jones=residual,
        artifact=artifact,
        row_mask=exact,
        parallactic_angle_rad=chi,
    )
    pred_diag = predict_moving_reference_from_beam(
        observation,
        residual_jones=residual,
        artifact=artifact,
        row_mask=exact,
        parallactic_angle_rad=chi,
        diagonal_only=True,
    )
    scores = score_full_versus_diagonal_regions(
        packed_plane,
        pred_full[:, 0] if pred_full.ndim == 4 else pred_full,
        pred_diag[:, 0] if pred_diag.ndim == 4 else pred_diag,
        stokes_i=intensity,
        row_mask=exact,
        voltage=voltage,
        dwell_ids=geometry["time_index"],
        mover_ids=geometry["moving_id"],
        reference_ids=geometry["reference_id"],
        cell_ids=_cell_ids(geometry),
    )
    main = scores.get("main_lobe") or scores["all"]
    gate = classify_loro_full_versus_diagonal(
        rl_improved=main.get("rl_improved"),
        lr_improved=main.get("lr_improved"),
        rr_ll_regression=main.get("rr_ll_regression"),
        n_finite=int(main.get("n_finite") or 0),
    )
    print(
        " ",
        name,
        "exact",
        int(np.sum(exact)),
        "main_n",
        main.get("n_finite"),
        "rl",
        main.get("full_median_abs_rl_over_i"),
        "vs",
        main.get("diagonal_median_abs_rl_over_i"),
        gate["outcome"],
        flush=True,
    )
    return {
        "held_reference": name,
        "held_reference_id": int(held_ref),
        "counts": {
            "train": int(np.sum(masks["train"])),
            "train_thinned": int(np.sum(train)),
            "holdout": int(np.sum(masks["holdout"])),
            "scored": int(np.sum(masks["scored"])),
            "exact": int(np.sum(exact)),
            "off_diagonal_supported": int(np.sum(artifact.off_diagonal_valid)),
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
        "scores": scores,
        "gate": gate,
    }


def _readme(payload: dict) -> str:
    gate = payload.get("gate") or {}
    lines = [
        "# THOL0001 SPW-4 LORO full Jones versus diagonal",
        "",
        f"Status: **{gate.get('status', 'unknown')}**. "
        f"Outcome: `{gate.get('outcome', 'unknown')}`.",
        "",
        LORO_HOLDOUT_NOTE,
        "",
        FULL_JONES_BLOCKED_UNTIL_ALIGNMENT_NOTE,
        "",
        "Residuals are source-normalized and in Jy on settled four-hand "
        "exact cells. Unsupported off-diagonals are zeroed and remain in "
        "the total score. References are not pooled.",
        "",
    ]
    for name, report in (payload.get("per_reference") or {}).items():
        item = report.get("gate") or {}
        main = (report.get("scores") or {}).get("main_lobe") or {}
        lines.append(
            f"- `{name}`: {item.get('outcome')} ({item.get('status')}); "
            f"RL {main.get('full_median_abs_rl_over_i')} vs "
            f"{main.get('diagonal_median_abs_rl_over_i')}; "
            f"LR {main.get('full_median_abs_lr_over_i')} vs "
            f"{main.get('diagonal_median_abs_lr_over_i')}."
        )
    lines.extend(["", "Full Jones remains unfrozen. SPW 5 stays closed.", ""])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--measurement-set", type=Path, default=SCIENTIFIC_MS)
    parser.add_argument("--product-dir", type=Path, default=UNBLOCK)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=UNBLOCK / "loro_full_versus_diagonal",
    )
    parser.add_argument("--spectral-window", type=int, default=4)
    parser.add_argument("--channel", type=int, default=NATIVE_CHANNEL)
    parser.add_argument("--max-train-per-cell", type=int, default=4)
    arguments = parser.parse_args()
    if int(arguments.spectral_window) != 4:
        raise ValueError("SPW 5 stays closed until this SPW-4 experiment is sound")
    if arguments.output_dir.resolve() in {
        arguments.product_dir.resolve(),
        (arguments.product_dir / "forward_closure").resolve(),
        (arguments.product_dir / "reference_visit_alignment").resolve(),
    }:
        raise ValueError("refusing to overwrite earlier scientific products")
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "thol0001_loro_full_versus_diagonal_v1",
        "gate_name": LORO_FULL_VERSUS_DIAGONAL,
        "status": "fail",
        "blocking": True,
        "spectral_window_id": 4,
        "native_channel": int(arguments.channel),
        "residual_denominator": "abs((S_RR + S_LL) / 2)",
        "sample_mask": "settled_four_hand_exact_cells",
        "off_diagonal_policy": "training_snr_then_zero",
        "spw5_closed": True,
        "full_jones_blocked": True,
        "full_jones_frozen": False,
        "do_not_overwrite": [
            "one_axis_visibility_holdouts.json",
            "one_axis_visibility_holdouts_v2.json",
            "forward_closure/",
            "reference_visit_alignment/",
        ],
        "per_reference": {},
        "notes": (
            LORO_HOLDOUT_NOTE,
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
        id_to_name = {int(ant): str(name) for ant, name in zip(ids, names, strict=True)}
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
        geometry = moving_reference_row_geometry(observation)
        references = _usable_references(observation, id_to_name)
        payload["usable_references"] = [name for name, _antenna in references]
        print("references", payload["usable_references"], flush=True)
        for name, antenna in references:
            payload["per_reference"][name] = _run_reference(
                observation,
                name=name,
                held_ref=antenna,
                residual_jones=residual,
                chi=chi,
                packed_plane=packed_plane,
                intensity=intensity,
                voltage=voltage,
                geometry=geometry,
                max_train_per_cell=int(arguments.max_train_per_cell),
            )
        gate = combine_loro_full_versus_diagonal(
            {name: report["gate"] for name, report in payload["per_reference"].items()}
        )
        payload["gate"] = gate
        payload["status"] = gate["status"]
        payload["blocking"] = gate["blocking"]
        payload["spw5_closed"] = gate["spw5_closed"]
    except Exception as error:
        print("error", error, flush=True)
        traceback.print_exc()
        payload["error"] = f"{type(error).__name__}: {error}"
        payload["gate"] = {
            "gate": LORO_FULL_VERSUS_DIAGONAL,
            "status": "fail",
            "blocking": True,
            "outcome": "runner_error",
            "full_jones_blocked": True,
        }
    (arguments.output_dir / "README.md").write_text(_readme(payload))
    write_json(_jsonable(payload), arguments.output_dir / "loro_full_versus_diagonal_report.json")
    print(arguments.output_dir / "loro_full_versus_diagonal_report.json")
    print("status", payload.get("status"), (payload.get("gate") or {}).get("outcome"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
