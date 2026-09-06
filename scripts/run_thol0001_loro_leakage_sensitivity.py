"""SPW-4 LORO leakage sensitivity diagnosis.

Injects known feed-frame leakage into the real training coordinates and
flag/noise pattern, inspects unmasked estimates, aggregates before the
SNR mask, compares CASSBEAM full Jones to CASSBEAM diagonal, and can
combine inner SPW-4 channels after delay removal. Full Jones stays
unfrozen. SPW 5 stays closed.
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
from sl1mjax.holography import HolographyObservation, circular_visibility_to_source_coherency
from sl1mjax.holography_alignment import (
    FULL_JONES_BLOCKED_UNTIL_ALIGNMENT_NOTE,
    LORO_FULL_VERSUS_DIAGONAL,
    LORO_LEAKAGE_SENSITIVITY,
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
    freeze_interpolation_support_from_rows,
    holography_one_axis_holdout_masks,
    moving_reference_row_geometry,
    predict_moving_reference_from_beam,
    recover_holography_full_jones,
    recover_holography_full_jones_rows,
    refactor_on_axis_gauge,
    score_full_versus_diagonal_regions,
    thin_training_rows,
    zero_unsupported_off_diagonals,
)
from sl1mjax.holography_leakage_sensitivity import (
    AGGREGATION_BEFORE_SNR_NOTE,
    CASSBEAM_DIRECT_NOTE,
    FEED_FRAME_INJECTION_NOTE,
    INJECTED_LEAKAGE_AMPLITUDES,
    LEAKAGE_SENSITIVITY_NOTE,
    aggregate_feed_frame_leakage,
    cassbeam_predicted_leakage_amplitude,
    classify_leakage_sensitivity,
    combine_channel_leakage_after_delay,
    inject_feed_frame_leakage,
    injection_recovery_point,
    predict_moving_reference_from_cassbeam,
    recovered_leakage_from_artifact,
    unmasked_leakage_diagnostics,
)
from sl1mjax.holography_ms import _read_antennas, _tables, audit_holography_measurement_set
from sl1mjax.holography_reference_jones import (
    NO_SCAN_OFFSETS_OR_TIME_FREEDOM_NOTE,
    deserialize_reference_jones,
)
from sl1mjax.polarization import Correlation, Receptor, ReceptorBasis, pack_coherency

SCIENTIFIC_MS = Path(
    "/media/stephen/astro/vla/extracted/commissioning/work/THOL0001.lowerC.spw45.scientific.ms"
)
UNBLOCK = Path(
    "/media/stephen/astro/vla/extracted/commissioning/validation/scientific/full_jones_unblock"
)
NATIVE_CHANNEL = 32
MIN_REFERENCE_ROWS = 1000
HOLORASTER_FIELD = 10


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
    return moving.astype(np.int64) * 2_000_000_003 + cell_l * 10_007 + cell_m


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


def _exact_mask(observation, held_ref: int, max_train_per_cell: int):
    masks = holography_one_axis_holdout_masks(
        observation,
        axis="leave_one_reference_out",
        held_reference_id=int(held_ref),
        reserve_outer_fold=True,
    )
    train = thin_training_rows(observation, masks["train"], max_per_cell=max_train_per_cell)
    geometry = moving_reference_row_geometry(observation)
    support = freeze_interpolation_support_from_rows(observation, train)
    labeled = classify_holdout_interpolation_support(support, geometry, masks["scored"])
    exact = np.asarray(masks["scored"], dtype=bool) & labeled["exact"]
    return masks, train, exact, geometry, labeled


def _score_pair(
    packed_plane,
    pred_full,
    pred_diag,
    intensity,
    exact,
    voltage,
    geometry,
):
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
    return scores, main, gate


def _recover_and_score(
    observation,
    *,
    visibility,
    residual_jones,
    chi,
    train,
    exact,
    intensity,
    voltage,
    geometry,
):
    measured = (
        np.asarray(observation.block.visibility, dtype=np.complex128)
        if visibility is None
        else np.asarray(visibility, dtype=np.complex128)
    )
    packed = pack_coherency(measured, observation.block.correlations, (Receptor.R, Receptor.L))
    packed_plane = packed[:, 0] if packed.ndim == 4 else packed
    rows = recover_holography_full_jones_rows(
        observation,
        visibility,
        residual_jones=residual_jones,
        parallactic_angle_rad=chi,
        row_mask=train,
    )
    unpinned = recover_holography_full_jones(
        observation,
        visibility,
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
    scores, main, gate = _score_pair(
        packed_plane, pred_full, pred_diag, intensity, exact, voltage, geometry
    )
    aggregated = aggregate_feed_frame_leakage(rows, observation=observation)
    pred_agg = predict_moving_reference_from_beam(
        observation,
        residual_jones=residual_jones,
        artifact=aggregated,
        row_mask=exact,
        parallactic_angle_rad=chi,
    )
    pred_agg_diag = predict_moving_reference_from_beam(
        observation,
        residual_jones=residual_jones,
        artifact=aggregated,
        row_mask=exact,
        parallactic_angle_rad=chi,
        diagonal_only=True,
    )
    agg_scores, agg_main, agg_gate = _score_pair(
        packed_plane, pred_agg, pred_agg_diag, intensity, exact, voltage, geometry
    )
    return {
        "unmasked": unmasked_leakage_diagnostics(rows, voltage=voltage),
        "unmasked_recovered": recovered_leakage_from_artifact(pinned, off_axis_only=True),
        "unpinned_recovered": recovered_leakage_from_artifact(unpinned, off_axis_only=True),
        "masked_recovered": recovered_leakage_from_artifact(artifact, off_axis_only=True),
        "aggregated_recovered": recovered_leakage_from_artifact(aggregated, off_axis_only=True),
        "counts": {
            "train_thinned": int(np.sum(train)),
            "exact": int(np.sum(exact)),
            "off_diagonal_supported": int(np.sum(artifact.off_diagonal_valid)),
            "aggregated_off_diagonal_supported": int(np.sum(aggregated.off_diagonal_valid)),
        },
        "scores": scores,
        "gate": gate,
        "aggregated_scores": agg_scores,
        "aggregated_gate": agg_gate,
        "main": main,
        "aggregated_main": agg_main,
    }


def _holoraster_channel_range_block(
    tables,
    measurement_set: Path,
    *,
    data_desc_id: int,
    channel_start: int,
    channel_stop: int,
    data_column: str,
    spectral_window_id: int,
):
    from sl1mjax.holography_ms import _read_field_phase_centre

    with tables.table(str(measurement_set / "SPECTRAL_WINDOW"), readonly=True, ack=False) as window:
        frequencies = np.asarray(
            window.getcell("CHAN_FREQ", spectral_window_id), dtype=np.float64
        ).reshape(-1)
    last = int(channel_stop) - 1
    query = (
        f"SELECT FROM '{measurement_set}' "
        f"WHERE FIELD_ID={HOLORASTER_FIELD} AND DATA_DESC_ID={int(data_desc_id)}"
    )
    with tables.taql(query) as selected:
        visibility = np.asarray(
            selected.getcolslice(data_column, [int(channel_start), 0], [last, -1])
        )
        model = np.asarray(selected.getcolslice("MODEL_DATA", [int(channel_start), 0], [last, -1]))
        flag = np.asarray(
            selected.getcolslice("FLAG", [int(channel_start), 0], [last, -1]), dtype=bool
        )
        n_chan = int(channel_stop) - int(channel_start)
        weight = np.repeat(np.asarray(selected.getcol("WEIGHT"))[:, None, :], n_chan, axis=1)
        correlations = (Correlation.RR, Correlation.RL, Correlation.LR, Correlation.LL)
        source = circular_visibility_to_source_coherency(model, correlations)
        from sl1mjax.data.canonical import VisibilityBlock

        block = VisibilityBlock(
            uvw_m=np.asarray(selected.getcol("UVW"), dtype=np.float64),
            frequency_hz=frequencies[int(channel_start) : int(channel_stop)],
            visibility=visibility,
            weight=weight,
            flag=flag,
            time_s=np.asarray(selected.getcol("TIME"), dtype=np.float64),
            antenna1=np.asarray(selected.getcol("ANTENNA1"), dtype=np.int32),
            antenna2=np.asarray(selected.getcol("ANTENNA2"), dtype=np.int32),
            field_id=np.asarray(selected.getcol("FIELD_ID"), dtype=np.int32),
            scan_id=np.asarray(selected.getcol("SCAN_NUMBER"), dtype=np.int32),
            correlations=correlations,
            receptor_basis=ReceptorBasis.CIRCULAR,
            phase_centre_rad=_read_field_phase_centre(tables, measurement_set, "HOLORASTER"),
            data_description_id=int(data_desc_id),
            spectral_window_id=int(spectral_window_id),
            provenance={
                "source": str(measurement_set),
                "column": data_column,
                "model_column": "MODEL_DATA",
                "channel_start": int(channel_start),
                "channel_stop": int(channel_stop),
            },
        )
    return block, source


def _readme(payload: dict) -> str:
    gate = payload.get("gate") or {}
    lines = [
        "# THOL0001 SPW-4 LORO leakage sensitivity",
        "",
        f"Status: **{gate.get('status', 'unknown')}**. "
        f"Outcome: `{gate.get('outcome', 'unknown')}`.",
        "",
        LEAKAGE_SENSITIVITY_NOTE,
        "",
        gate.get("interpretation", ""),
        "",
        FEED_FRAME_INJECTION_NOTE,
        "",
        AGGREGATION_BEFORE_SNR_NOTE,
        "",
        CASSBEAM_DIRECT_NOTE,
        "",
        "Full Jones remains unfrozen. SPW 5 stays closed.",
        "",
    ]
    cassbeam = payload.get("cassbeam_predicted_leakage") or {}
    lines.append(
        f"CASSBEAM median |E_RL/E_RR| at measured cells: {cassbeam.get('median_abs_rl_over_rr')}."
    )
    lines.append("")
    for name, report in (payload.get("per_reference") or {}).items():
        real = (report.get("real") or {}).get("masked_recovered") or {}
        lines.append(
            f"- `{name}` real masked leaks {real.get('n_off_diagonal_valid')}; "
            f"coherent |d_RL| {real.get('coherent_abs_rl')}."
        )
        for amp, point in (report.get("injection_curve") or {}).items():
            lines.append(
                f"  - inject {amp}: recovered={point.get('recovered')} "
                f"|d_RL|={point.get('recovered_coherent_abs_rl')} "
                f"RL improved={point.get('rl_improved')} "
                f"LR improved={point.get('lr_improved')}."
            )
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--measurement-set", type=Path, default=SCIENTIFIC_MS)
    parser.add_argument("--product-dir", type=Path, default=UNBLOCK)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=UNBLOCK / "loro_leakage_sensitivity",
    )
    parser.add_argument("--spectral-window", type=int, default=4)
    parser.add_argument("--channel", type=int, default=NATIVE_CHANNEL)
    parser.add_argument("--max-train-per-cell", type=int, default=4)
    parser.add_argument(
        "--held-references",
        default="ea26",
        help="Comma-separated names, or 'all'. Default ea26 matches the sealed LORO.",
    )
    parser.add_argument(
        "--frequency-start",
        type=int,
        default=24,
        help="Inclusive inner SPW-4 channel for the frequency diagnosis",
    )
    parser.add_argument(
        "--frequency-stop",
        type=int,
        default=40,
        help="Exclusive inner SPW-4 channel for the frequency diagnosis",
    )
    parser.add_argument("--skip-frequency", action="store_true")
    arguments = parser.parse_args()
    if int(arguments.spectral_window) != 4:
        raise ValueError("SPW 5 stays closed during this SPW-4 diagnosis")
    if arguments.output_dir.resolve() in {
        arguments.product_dir.resolve(),
        (arguments.product_dir / "forward_closure").resolve(),
        (arguments.product_dir / "reference_visit_alignment").resolve(),
        (arguments.product_dir / "loro_full_versus_diagonal").resolve(),
    }:
        raise ValueError("refusing to overwrite earlier scientific products")
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "thol0001_loro_leakage_sensitivity_v1",
        "gate_name": LORO_LEAKAGE_SENSITIVITY,
        "status": "fail",
        "blocking": True,
        "spectral_window_id": 4,
        "native_channel": int(arguments.channel),
        "injected_amplitudes": list(INJECTED_LEAKAGE_AMPLITUDES),
        "residual_denominator": "abs((S_RR + S_LL) / 2)",
        "sample_mask": "settled_four_hand_exact_cells",
        "spw5_closed": True,
        "full_jones_blocked": True,
        "full_jones_frozen": False,
        "most_important_next_artifact": LORO_LEAKAGE_SENSITIVITY,
        "do_not_overwrite": [
            "one_axis_visibility_holdouts.json",
            "one_axis_visibility_holdouts_v2.json",
            "forward_closure/",
            "reference_visit_alignment/",
            "loro_full_versus_diagonal/",
        ],
        "per_reference": {},
        "notes": (
            LEAKAGE_SENSITIVITY_NOTE,
            FEED_FRAME_INJECTION_NOTE,
            AGGREGATION_BEFORE_SNR_NOTE,
            CASSBEAM_DIRECT_NOTE,
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
        if arguments.held_references != "all":
            wanted = {item.strip() for item in arguments.held_references.split(",") if item.strip()}
            references = [item for item in references if item[0] in wanted]
        payload["usable_references"] = [name for name, _antenna in references]
        print("references", payload["usable_references"], flush=True)
        usable = np.asarray(geometry["usable"], dtype=bool)
        payload["cassbeam_predicted_leakage"] = cassbeam_predicted_leakage_amplitude(
            geometry["offset_lm_rad"][usable],
            np.full(int(np.sum(usable)), float(observation.block.frequency_hz[0])),
            chi[geometry["time_index"][usable], geometry["moving_id"][usable]],
        )
        print("cassbeam_leak", payload["cassbeam_predicted_leakage"], flush=True)
        cassbeam_rl = []
        cassbeam_lr = []
        cassbeam_rr = []
        for name, held_ref in references:
            print("loro", name, flush=True)
            masks, train, exact, geometry, labeled = _exact_mask(
                observation, held_ref, int(arguments.max_train_per_cell)
            )
            pred_full = predict_moving_reference_from_cassbeam(
                observation,
                residual_jones=residual,
                row_mask=exact,
                parallactic_angle_rad=chi,
                off_diagonal=True,
            )
            pred_diag = predict_moving_reference_from_cassbeam(
                observation,
                residual_jones=residual,
                row_mask=exact,
                parallactic_angle_rad=chi,
                off_diagonal=False,
            )
            cassbeam_scores, cassbeam_main, cassbeam_gate = _score_pair(
                packed_plane, pred_full, pred_diag, intensity, exact, voltage, geometry
            )
            cassbeam_rl.append(cassbeam_main.get("rl_improved"))
            cassbeam_lr.append(cassbeam_main.get("lr_improved"))
            cassbeam_rr.append(cassbeam_main.get("rr_ll_regression"))
            real = _recover_and_score(
                observation,
                visibility=None,
                residual_jones=residual,
                chi=chi,
                train=train,
                exact=exact,
                intensity=intensity,
                voltage=voltage,
                geometry=geometry,
            )
            print(
                " ",
                name,
                "real",
                "offdiag",
                real["counts"]["off_diagonal_supported"],
                "agg",
                real["counts"]["aggregated_off_diagonal_supported"],
                "coherent",
                real["unmasked_recovered"]["coherent_abs_rl"],
                flush=True,
            )
            baseline_abs = float(real["unmasked_recovered"].get("coherent_abs_rl") or 0.0)
            curve = {}
            for amplitude in INJECTED_LEAKAGE_AMPLITUDES:
                print(" ", name, "inject", amplitude, flush=True)
                injected = inject_feed_frame_leakage(
                    observation,
                    epsilon=float(amplitude),
                    residual_jones=residual,
                    parallactic_angle_rad=chi,
                    row_mask=train | exact,
                )
                result = _recover_and_score(
                    observation,
                    visibility=injected,
                    residual_jones=residual,
                    chi=chi,
                    train=train,
                    exact=exact,
                    intensity=intensity,
                    voltage=voltage,
                    geometry=geometry,
                )
                point = injection_recovery_point(
                    injected_amplitude=float(amplitude),
                    recovered=result["unmasked_recovered"],
                    heldout_scores=result["main"],
                    baseline_abs=baseline_abs,
                )
                point["aggregated"] = injection_recovery_point(
                    injected_amplitude=float(amplitude),
                    recovered=result["aggregated_recovered"],
                    heldout_scores=result["aggregated_main"],
                    baseline_abs=float(real["aggregated_recovered"].get("coherent_abs_rl") or 0.0),
                )
                point["masked"] = result["masked_recovered"]
                curve[f"{amplitude:.6g}"] = point
                print(
                    " ",
                    name,
                    amplitude,
                    "recovered",
                    point["recovered"],
                    "abs",
                    point["recovered_coherent_abs_rl"],
                    "rl",
                    point["rl_improved"],
                    flush=True,
                )
            payload["per_reference"][name] = {
                "held_reference": name,
                "held_reference_id": int(held_ref),
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
                "cassbeam_direct": {
                    "scores": cassbeam_scores,
                    "gate": cassbeam_gate,
                },
                "real": real,
                "injection_curve": curve,
            }
        first_curve = {}
        first_real = float("nan")
        if payload["per_reference"]:
            first = next(iter(payload["per_reference"].values()))
            first_curve = first.get("injection_curve") or {}
            first_real = float(
                (first.get("real") or {}).get("unmasked_recovered", {}).get("coherent_abs_rl")
                or np.nan
            )
        cassbeam_amp = float(
            (payload.get("cassbeam_predicted_leakage") or {}).get("median_abs_rl_over_rr") or np.nan
        )
        gate = classify_leakage_sensitivity(
            cassbeam_amplitude=cassbeam_amp,
            injection_curve=first_curve,
            cassbeam_rl_improved=bool(cassbeam_rl and all(cassbeam_rl)),
            cassbeam_lr_improved=bool(cassbeam_lr and all(cassbeam_lr)),
            cassbeam_rr_ll_regression=bool(any(cassbeam_rr)),
            real_coherent_abs=first_real,
        )
        if not arguments.skip_frequency:
            try:
                print(
                    "frequency",
                    arguments.frequency_start,
                    arguments.frequency_stop,
                    flush=True,
                )
                freq_block, freq_source = _holoraster_channel_range_block(
                    tables,
                    arguments.measurement_set,
                    data_desc_id=ddid,
                    channel_start=int(arguments.frequency_start),
                    channel_stop=int(arguments.frequency_stop),
                    data_column="CORRECTED_DATA",
                    spectral_window_id=4,
                )
                freq_obs = HolographyObservation(
                    block=freq_block,
                    pointing=diag._subset_pointing(audit.resolved, freq_block.time_s),
                    antenna_position_m=positions,
                    calibration_state="casa_parang_true",
                    phase_centre_rad=freq_block.phase_centre_rad,
                    source_name="3C147",
                    source_coherency_visibility=freq_source,
                    selected_spw_id=4,
                )
                name, held_ref = references[0]
                masks, train, exact, _geometry, _labeled = _exact_mask(
                    freq_obs, held_ref, int(arguments.max_train_per_cell)
                )
                freq_rows = recover_holography_full_jones_rows(
                    freq_obs,
                    residual_jones=residual,
                    parallactic_angle_rad=parallactic_angle(
                        unique_visibility_times(freq_block.time_s)[0],
                        freq_obs.phase_centre_rad,
                        positions,
                    ),
                    row_mask=train,
                )
                combined = combine_channel_leakage_after_delay(freq_rows)
                payload["frequency_sensitivity"] = {
                    "held_reference": name,
                    "channel_start": int(arguments.frequency_start),
                    "channel_stop": int(arguments.frequency_stop),
                    "n_channel": int(arguments.frequency_stop) - int(arguments.frequency_start),
                    "n_cell": int(combined["moving_id"].size),
                    "median_coherent_abs_rl": float(np.nanmedian(combined["coherent_abs_rl"])),
                    "median_n_channel_combined": float(np.median(combined["n_channel_combined"])),
                    "spw5_closed": True,
                }
                print("frequency", payload["frequency_sensitivity"], flush=True)
            except Exception as error:
                print("frequency_error", error, flush=True)
                traceback.print_exc()
                payload["frequency_sensitivity"] = {
                    "error": f"{type(error).__name__}: {error}",
                    "spw5_closed": True,
                }
        payload["gate"] = gate
        payload["status"] = gate["status"]
        payload["blocking"] = gate["blocking"]
        payload["previous_gate"] = LORO_FULL_VERSUS_DIAGONAL
    except Exception as error:
        print("error", error, flush=True)
        traceback.print_exc()
        payload["error"] = f"{type(error).__name__}: {error}"
        payload["gate"] = {
            "gate": LORO_LEAKAGE_SENSITIVITY,
            "status": "fail",
            "blocking": True,
            "outcome": "runner_error",
            "full_jones_blocked": True,
            "full_jones_frozen": False,
            "spw5_closed": True,
        }
    (arguments.output_dir / "README.md").write_text(_readme(payload))
    write_json(_jsonable(payload), arguments.output_dir / "loro_leakage_sensitivity_report.json")
    print(arguments.output_dir / "loro_leakage_sensitivity_report.json")
    print("status", payload.get("status"), (payload.get("gate") or {}).get("outcome"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
