"""Score THOL0001 full-Jones visibility holdouts one axis at a time.

Axes are never unioned or pooled. The interpolator neighbor rule stays the
a-priori 1.5× median-spacing, k=4 freeze. A reserved outer fold is recorded
and not scored. Leave-one-mover-out is an array-average test; per-antenna
zeros for the held mover are expected. SPW 5 stays closed.
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
    write_json,
)
from sl1mjax.holography_full_jones import (
    ARRAY_AVERAGE_ANTENNA_ID,
    DEVELOPMENT_SET_NOTE,
    INTERPOLATOR_FROZEN_ON_TRAIN_NOTE,
    INTERPOLATOR_NEIGHBOR_K,
    INTERPOLATOR_RADIUS_SCALE,
    LOMO_HOLDOUT_NOTE,
    LORO_HOLDOUT_NOTE,
    LOVO_HOLDOUT_NOTE,
    ONE_AXIS_HOLDOUT_AXES,
    ONE_AXIS_HOLDOUT_NOTE,
    SPATIAL_CHECKERBOARD_NOTE,
    average_holography_full_jones,
    classify_holdout_interpolation_support,
    classify_one_axis_visibility_holdout,
    combine_one_axis_results,
    freeze_interpolation_support,
    freeze_interpolation_support_from_rows,
    holography_one_axis_holdout_masks,
    moving_reference_row_geometry,
    paired_full_versus_diagonal_scores,
    predict_moving_reference_from_beam,
    recover_holography_full_jones,
    refactor_on_axis_gauge,
    thin_training_rows,
)
from sl1mjax.holography_ms import _read_antennas, _tables, audit_holography_measurement_set
from sl1mjax.holography_reference_jones import (
    NO_SCAN_OFFSETS_OR_TIME_FREEDOM_NOTE,
    ONE_AXIS_VISIBILITY_HOLDOUT_NOTE,
    deserialize_reference_jones,
)
from sl1mjax.polarization import Receptor, pack_coherency

SCIENTIFIC_MS = Path(
    "/media/stephen/astro/vla/extracted/commissioning/work/THOL0001.lowerC.spw45.scientific.ms"
)
UNBLOCK = Path(
    "/media/stephen/astro/vla/extracted/commissioning/validation/scientific/full_jones_unblock"
)
HELD_MOVER = "ea04"
NATIVE_CHANNEL = 32
AXIS_NOTES = {
    "leave_one_reference_out": LORO_HOLDOUT_NOTE,
    "leave_one_visit_out": LOVO_HOLDOUT_NOTE,
    "spatial_checkerboard": SPATIAL_CHECKERBOARD_NOTE,
    "leave_one_mover_out": LOMO_HOLDOUT_NOTE,
}


def _diag():
    path = Path(__file__).with_name("run_thol0001_diagonal_recovery.py")
    spec = importlib.util.spec_from_file_location("thol0001_diagonal_recovery", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _subsample(mask: np.ndarray, max_rows: int, *, seed: int = 0) -> np.ndarray:
    selected = np.flatnonzero(mask)
    if selected.size <= int(max_rows):
        return np.asarray(mask, dtype=bool)
    keep = np.zeros(mask.shape, dtype=bool)
    keep[np.random.default_rng(seed).choice(selected, size=int(max_rows), replace=False)] = True
    return keep


def _cell_ids(geometry: dict[str, np.ndarray]) -> np.ndarray:
    moving = np.asarray(geometry["moving_id"], dtype=np.int64)
    cell_l = np.asarray(geometry["cell_l"], dtype=np.int64)
    cell_m = np.asarray(geometry["cell_m"], dtype=np.int64)
    keys = list(zip(moving.tolist(), cell_l.tolist(), cell_m.tolist(), strict=True))
    index = {key: i for i, key in enumerate(sorted(set(keys)))}
    return np.asarray([index[key] for key in keys], dtype=np.int64)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _score_predictions(
    observation,
    *,
    residual,
    artifact,
    chi,
    packed_plane,
    intensity,
    geometry,
    row_mask,
):
    pred_full = predict_moving_reference_from_beam(
        observation,
        residual_jones=residual,
        artifact=artifact,
        row_mask=row_mask,
        parallactic_angle_rad=chi,
    )
    pred_diag = predict_moving_reference_from_beam(
        observation,
        residual_jones=residual,
        artifact=artifact,
        row_mask=row_mask,
        parallactic_angle_rad=chi,
        diagonal_only=True,
    )
    pred_f = pred_full[:, 0] if pred_full.ndim == 4 else pred_full
    pred_d = pred_diag[:, 0] if pred_diag.ndim == 4 else pred_diag
    return paired_full_versus_diagonal_scores(
        packed_plane,
        pred_f,
        pred_d,
        stokes_i=intensity,
        row_mask=row_mask,
        dwell_ids=geometry["time_index"],
        mover_ids=geometry["moving_id"],
        reference_ids=geometry["reference_id"],
        cell_ids=_cell_ids(geometry),
    )


def _run_axis(
    observation,
    *,
    axis: str,
    residual_jones,
    chi,
    packed_plane,
    intensity,
    held_ref: int,
    held_move: int,
    max_train_per_cell: int,
    max_score_rows: int,
):
    print("axis", axis, flush=True)
    masks = holography_one_axis_holdout_masks(
        observation,
        axis=axis,
        held_reference_id=held_ref,
        held_moving_id=held_move,
    )
    train = thin_training_rows(observation, masks["train"], max_per_cell=max_train_per_cell)
    support = freeze_interpolation_support_from_rows(observation, train)
    geometry = moving_reference_row_geometry(observation)
    print(
        "  train",
        int(np.sum(masks["train"])),
        "thinned",
        int(np.sum(train)),
        "holdout",
        int(np.sum(masks["holdout"])),
        "reserved",
        int(np.sum(masks["reserved"])),
        flush=True,
    )
    unpinned = recover_holography_full_jones(
        observation,
        residual_jones=residual_jones,
        parallactic_angle_rad=chi,
        row_mask=train,
    )
    pinned, residual, _origin = refactor_on_axis_gauge(unpinned, residual_jones)
    representations = [("per_antenna", pinned, support)]
    if axis == "leave_one_mover_out":
        averaged = average_holography_full_jones(pinned)
        union = (
            np.concatenate(list(support.offsets_by_antenna.values()), axis=0)
            if support.offsets_by_antenna
            else np.zeros((0, 2), dtype=np.float64)
        )
        avg_support = freeze_interpolation_support({ARRAY_AVERAGE_ANTENNA_ID: union})
        representations.append(("array_average", averaged, avg_support))
    reports = {}
    for representation, artifact, frozen in representations:
        labeled_hold = classify_holdout_interpolation_support(
            frozen, geometry, masks["holdout"], representation=representation
        )
        labeled_scored = classify_holdout_interpolation_support(
            frozen, geometry, masks["scored"], representation=representation
        )
        score_mask = _subsample(masks["scored"], max_score_rows)
        supported_mask = score_mask & labeled_scored["supported"]
        complete = _score_predictions(
            observation,
            residual=residual,
            artifact=artifact,
            chi=chi,
            packed_plane=packed_plane,
            intensity=intensity,
            geometry=geometry,
            row_mask=score_mask,
        )
        supported = _score_predictions(
            observation,
            residual=residual,
            artifact=artifact,
            chi=chi,
            packed_plane=packed_plane,
            intensity=intensity,
            geometry=geometry,
            row_mask=supported_mask,
        )
        extras = {}
        if axis == "leave_one_visit_out":
            extras["exact_repeated_cells"] = _score_predictions(
                observation,
                residual=residual,
                artifact=artifact,
                chi=chi,
                packed_plane=packed_plane,
                intensity=intensity,
                geometry=geometry,
                row_mask=score_mask & labeled_scored["exact"],
            )
            extras["spatially_interpolated_cells"] = _score_predictions(
                observation,
                residual=residual,
                artifact=artifact,
                chi=chi,
                packed_plane=packed_plane,
                intensity=intensity,
                geometry=geometry,
                row_mask=score_mask & labeled_scored["interpolation"],
            )
        gate = classify_one_axis_visibility_holdout(
            axis=axis,
            representation=representation,
            support_fraction=float(labeled_hold["support_fraction"]),
            n_holdout=int(labeled_hold["n_holdout"]),
            n_supported=int(labeled_hold["n_supported"]),
            n_finite=int(complete["n_finite"]),
            rl_improved=complete.get("rl_improved"),
            lr_improved=complete.get("lr_improved"),
            rr_ll_regression=complete.get("rr_ll_regression"),
            supported_rl_improved=supported.get("rl_improved"),
            supported_lr_improved=supported.get("lr_improved"),
            supported_rr_ll_regression=supported.get("rr_ll_regression"),
            supported_n_finite=int(supported["n_finite"]),
        )
        print(
            " ",
            representation,
            "support",
            labeled_hold["support_fraction"],
            "outcome",
            gate["outcome"],
            flush=True,
        )
        reports[representation] = {
            "gate": gate,
            "holdout_support": {
                key: labeled_hold[key]
                for key in (
                    "n_holdout",
                    "n_supported",
                    "support_fraction",
                    "n_exact",
                    "n_interpolation",
                    "n_extrapolation",
                    "n_unsupported",
                    "median_distance_rad",
                )
            },
            "scored_support": {
                key: labeled_scored[key]
                for key in (
                    "n_holdout",
                    "n_supported",
                    "support_fraction",
                    "n_exact",
                    "n_interpolation",
                    "n_extrapolation",
                    "n_unsupported",
                    "median_distance_rad",
                )
            },
            "complete_holdout": complete,
            "supported_rows": supported,
            **extras,
        }
    primary = reports["array_average"] if axis == "leave_one_mover_out" else reports["per_antenna"]
    return {
        "axis": axis,
        "note": AXIS_NOTES[axis],
        "counts": {
            "usable": int(np.sum(masks["usable"])),
            "train": int(np.sum(masks["train"])),
            "train_thinned": int(np.sum(train)),
            "holdout": int(np.sum(masks["holdout"])),
            "reserved": int(np.sum(masks["reserved"])),
            "scored": int(np.sum(masks["scored"])),
        },
        "interpolator": {
            "neighbor_k": INTERPOLATOR_NEIGHBOR_K,
            "radius_scale": INTERPOLATOR_RADIUS_SCALE,
            "frozen_a_priori": True,
            "frozen_from": "thinned_training_rows",
        },
        "representations": reports,
        "gate": primary["gate"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--measurement-set", type=Path, default=SCIENTIFIC_MS)
    parser.add_argument("--output-dir", type=Path, default=UNBLOCK)
    parser.add_argument("--spectral-window", type=int, default=4)
    parser.add_argument("--channel", type=int, default=NATIVE_CHANNEL)
    parser.add_argument(
        "--axis",
        action="append",
        choices=["all", *ONE_AXIS_HOLDOUT_AXES],
        help="Repeat to run selected axes. Default is all four, separately.",
    )
    parser.add_argument("--max-train-per-cell", type=int, default=4)
    parser.add_argument("--max-score-rows", type=int, default=15000)
    parser.add_argument(
        "--output-name",
        default="one_axis_visibility_holdouts.json",
        help="Filename under --output-dir. Do not overwrite earlier products.",
    )
    arguments = parser.parse_args()
    if int(arguments.spectral_window) != 4:
        raise ValueError("SPW 5 stays closed")
    requested = []
    for item in arguments.axis or ["all"]:
        if item == "all":
            requested.extend(ONE_AXIS_HOLDOUT_AXES)
        else:
            requested.append(item)
    axes = tuple(dict.fromkeys(requested))
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    tables = _tables()
    ids, names, positions = _read_antennas(tables, arguments.measurement_set)
    names = np.asarray(names)
    positions = np.asarray(positions, dtype=np.float64)
    name_to_id = {str(name): int(ant) for ant, name in zip(ids, names, strict=True)}
    family_cache = arguments.output_dir / "prediction_equivalent_family.json"
    stabilized = arguments.output_dir / "field9_stabilized_residual_jones.json"
    if family_cache.exists():
        nominal = deserialize_reference_jones(
            json.loads(family_cache.read_text())["nominal"]["jones"]
        )
    elif stabilized.exists():
        nominal = deserialize_reference_jones(json.loads(stabilized.read_text())["jones"])
    else:
        raise FileNotFoundError("need field9_stabilized_residual_jones.json or the family cache")
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
    from sl1mjax.holography_diagonal import source_model_stokes_i

    intensity = source_model_stokes_i(observation)
    held_ref = int(name_to_id[HELD_OUT_REFERENCE_ANTENNA])
    held_move = int(name_to_id[HELD_MOVER])
    axis_reports = {}
    for axis in axes:
        axis_reports[axis] = _run_axis(
            observation,
            axis=axis,
            residual_jones=nominal,
            chi=chi,
            packed_plane=packed_plane,
            intensity=intensity,
            held_ref=held_ref,
            held_move=held_move,
            max_train_per_cell=int(arguments.max_train_per_cell),
            max_score_rows=int(arguments.max_score_rows),
        )
    combined = combine_one_axis_results({name: item["gate"] for name, item in axis_reports.items()})
    blocking = False
    for name, item in axis_reports.items():
        gate = item["gate"]
        if name == "leave_one_mover_out":
            continue
        if gate.get("blocking"):
            blocking = True
    payload = {
        "schema": "thol0001_one_axis_visibility_holdouts_v1",
        "status": "fail" if blocking else "warn",
        "blocking": blocking,
        "outcome": "axes_reported_separately",
        "spectral_window_id": 4,
        "native_channel": int(arguments.channel),
        "held_out_reference": HELD_OUT_REFERENCE_ANTENNA,
        "held_out_mover": HELD_MOVER,
        "interpolator_frozen_a_priori": True,
        "interpolator_neighbor_k": INTERPOLATOR_NEIGHBOR_K,
        "interpolator_radius_scale": INTERPOLATOR_RADIUS_SCALE,
        "reserved_outer_fold": True,
        "do_not_combine": True,
        "pooled_score": None,
        "axes": axis_reports,
        "combined": combined,
        "notes": [
            ONE_AXIS_HOLDOUT_NOTE,
            ONE_AXIS_VISIBILITY_HOLDOUT_NOTE,
            INTERPOLATOR_FROZEN_ON_TRAIN_NOTE,
            DEVELOPMENT_SET_NOTE,
            NO_SCAN_OFFSETS_OR_TIME_FREEDOM_NOTE,
        ],
    }
    path = arguments.output_dir / str(arguments.output_name)
    write_json(_jsonable(payload), path)
    print(path)
    for name, item in axis_reports.items():
        print(name, item["gate"]["outcome"], item["gate"]["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
