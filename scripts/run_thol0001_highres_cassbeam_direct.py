"""SPW-4 high-resolution CASSBEAM direct visibility-domain template test.

Loads the external 513×513 artifact, locks conventions on training data
only, and scores the matched template. Does not recover an empirical
beam, apply an off-diagonal SNR mask, open SPW 5, or freeze full Jones.
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
from sl1mjax.cassbeam_highres import DEFAULT_HIGHRES_ROOT, HighresCassbeamCatalog
from sl1mjax.holography import HolographyObservation
from sl1mjax.holography_alignment import (
    FULL_JONES_BLOCKED_UNTIL_ALIGNMENT_NOTE,
    HIGHRES_CASSBEAM_DIRECT,
    apparent_voltage_response,
    holoraster_pair_masks,
)
from sl1mjax.holography_calibration import write_json
from sl1mjax.holography_diagonal import (
    copolar_hand_active_rows,
    holography_row_exclusion_counts,
    source_model_stokes_i,
)
from sl1mjax.holography_full_jones import DEVELOPMENT_SET_NOTE, LORO_HOLDOUT_NOTE
from sl1mjax.holography_highres_cassbeam import (
    CHANNEL_COVARIANCE_NOTE,
    HIGHRES_CASSBEAM_NOTE,
    NO_GLOBAL_FIVE_PERCENT_NOTE,
    TEMPLATE_ALPHA_NOTE,
    TRAINING_ONLY_CONVENTION_NOTE,
    artifact_checksum_report,
    classify_highres_cassbeam_direct,
    convention_ladder,
    lock_cassbeam_convention,
    loro_masks,
    predict_full_and_diagonal,
    qu_nuisance_robustness,
    run_software_gates,
    score_template_predictions,
    template_injection_curve,
    write_template_plots,
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
HELD_REFERENCE_NAME = "ea26"
SPW4_MHZ = tuple(range(4500, 4628, 2))


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
    if isinstance(value, list):
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


def _load_observation(diag, tables, measurement_set, *, channel: int, positions):
    audit = audit_holography_measurement_set(measurement_set)
    ddid = diag._ddid_for_spw(tables, measurement_set, 4)
    block, source = diag._holoraster_channel_block(
        tables,
        measurement_set,
        data_desc_id=ddid,
        channel=channel,
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
    return observation, chi, packed_plane, intensity, voltage


def _score_reference(
    observation,
    *,
    name: str,
    held_ref: int,
    residual_jones,
    catalog,
    convention,
    chi,
    intensity,
    voltage,
    max_train_per_cell: int,
    fit_alpha_on_training: bool,
):
    split = loro_masks(
        observation,
        held_reference_id=int(held_ref),
        max_train_per_cell=max_train_per_cell,
    )
    pred_full, pred_diag = predict_full_and_diagonal(
        observation,
        residual_jones=residual_jones,
        catalog=catalog,
        convention=convention,
        row_mask=split["train"] | split["exact"],
        parallactic_angle_rad=chi,
    )
    train_score = score_template_predictions(
        observation,
        pred_full,
        pred_diag,
        row_mask=split["train"],
        stokes_i=intensity,
        geometry=split["geometry"],
        voltage=voltage,
    )
    alpha_hat = complex(train_score["alpha"]["real"], train_score["alpha"]["imag"])
    hold_score = score_template_predictions(
        observation,
        pred_full,
        pred_diag,
        row_mask=split["exact"],
        stokes_i=intensity,
        geometry=split["geometry"],
        voltage=voltage,
        fitted_alpha=alpha_hat if fit_alpha_on_training else 1.0 + 0.0j,
    )
    hold_score["training_correlation"] = train_score["holdout_correlation"]
    hold_score["training_alpha"] = train_score["alpha"]
    main = (hold_score.get("paired_scores") or {}).get("main_lobe") or {}
    return {
        "held_reference": name,
        "held_reference_id": int(held_ref),
        "counts": {
            "train": int(np.sum(split["masks"]["train"])),
            "train_thinned": int(np.sum(split["train"])),
            "holdout": int(np.sum(split["masks"]["holdout"])),
            "scored": int(np.sum(split["masks"]["scored"])),
            "exact": int(np.sum(split["exact"])),
        },
        "training": train_score,
        "holdout": hold_score,
        "rr_ll_regression": main.get("rr_ll_regression"),
        "pred_full": pred_full,
        "pred_diag": pred_diag,
        "split": split,
    }


def _readme(payload: dict) -> str:
    gate = payload.get("gate") or {}
    lines = [
        "# THOL0001 SPW-4 high-resolution CASSBEAM direct template",
        "",
        f"Status: **{gate.get('status', 'unknown')}**. "
        f"Outcome: `{gate.get('outcome', 'unknown')}`.",
        "",
        HIGHRES_CASSBEAM_NOTE,
        "",
        TRAINING_ONLY_CONVENTION_NOTE,
        "",
        TEMPLATE_ALPHA_NOTE,
        "",
        NO_GLOBAL_FIVE_PERCENT_NOTE,
        "",
        CHANNEL_COVARIANCE_NOTE,
        "",
        FULL_JONES_BLOCKED_UNTIL_ALIGNMENT_NOTE,
        "",
    ]
    ch32 = payload.get("channel32_ea26") or {}
    hold = ch32.get("holdout") or {}
    alpha = hold.get("alpha") or {}
    lines.append(
        f"Channel 32 / ea26 fitted α = {alpha.get('real')} + {alpha.get('imag')}j "
        f"(consistent with 0: {alpha.get('consistent_with_zero')}, "
        f"consistent with 1: {alpha.get('consistent_with_one')})."
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
        default=UNBLOCK / "highres_cassbeam_direct",
    )
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_HIGHRES_ROOT)
    parser.add_argument("--spectral-window", type=int, default=4)
    parser.add_argument("--channel", type=int, default=NATIVE_CHANNEL)
    parser.add_argument("--held-reference", default=HELD_REFERENCE_NAME)
    parser.add_argument("--max-train-per-cell", type=int, default=4)
    parser.add_argument(
        "--stage",
        choices=("channel32", "all_references", "all_channels", "all"),
        default="channel32",
    )
    parser.add_argument("--expand-on-close", action="store_true")
    arguments = parser.parse_args()
    if int(arguments.spectral_window) != 4:
        raise ValueError("SPW 5 stays closed")
    forbidden = {
        arguments.product_dir.resolve(),
        (arguments.product_dir / "forward_closure").resolve(),
        (arguments.product_dir / "reference_visit_alignment").resolve(),
        (arguments.product_dir / "loro_full_versus_diagonal").resolve(),
        (arguments.product_dir / "loro_leakage_sensitivity").resolve(),
        (arguments.product_dir / "highres_cassbeam_direct" / "previous_mixed_scorer").resolve(),
    }
    if arguments.output_dir.resolve() in forbidden:
        raise ValueError("refusing to overwrite earlier scientific products")
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "thol0001_highres_cassbeam_direct_v1",
        "gate_name": HIGHRES_CASSBEAM_DIRECT,
        "status": "fail",
        "blocking": True,
        "spectral_window_id": 4,
        "native_channel": int(arguments.channel),
        "held_reference": str(arguments.held_reference),
        "sample_mask": "settled_four_hand_exact_cells",
        "residual_denominator": "abs((S_RR + S_LL) / 2)",
        "off_diagonal_policy": "direct_highres_cassbeam_template",
        "empirical_recovery": False,
        "snr_mask": False,
        "committed_33x33_loaded": False,
        "nearest_frequency_substitution": False,
        "spw5_closed": True,
        "full_jones_blocked": True,
        "full_jones_frozen": False,
        "do_not_overwrite": [
            "one_axis_visibility_holdouts.json",
            "one_axis_visibility_holdouts_v2.json",
            "forward_closure/",
            "reference_visit_alignment/",
            "loro_full_versus_diagonal/",
            "loro_leakage_sensitivity/",
        ],
        "notes": (
            HIGHRES_CASSBEAM_NOTE,
            TRAINING_ONLY_CONVENTION_NOTE,
            TEMPLATE_ALPHA_NOTE,
            NO_GLOBAL_FIVE_PERCENT_NOTE,
            CHANNEL_COVARIANCE_NOTE,
            LORO_HOLDOUT_NOTE,
            FULL_JONES_BLOCKED_UNTIL_ALIGNMENT_NOTE,
            DEVELOPMENT_SET_NOTE,
            NO_SCAN_OFFSETS_OR_TIME_FREEDOM_NOTE,
        ),
    }
    try:
        catalog = HighresCassbeamCatalog(arguments.artifact_root)
        tables = _tables()
        ids, names, positions = _read_antennas(tables, arguments.measurement_set)
        names = np.asarray(names)
        positions = np.asarray(positions, dtype=np.float64)
        id_to_name = {int(ant): str(name) for ant, name in zip(ids, names, strict=True)}
        name_to_id = {str(name): int(ant) for ant, name in id_to_name.items()}
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
        observation, chi, packed_plane, intensity, voltage = _load_observation(
            diag,
            tables,
            arguments.measurement_set,
            channel=int(arguments.channel),
            positions=positions,
        )
        frequency = float(observation.block.frequency_hz[0])
        payload["frequency_hz"] = frequency
        payload["artifact"] = artifact_checksum_report(catalog, frequency)
        plane = catalog.plane(frequency)
        pair = holoraster_pair_masks(observation)
        measured_lm = pair["offset_lm_rad"][pair["four_hand_moving_reference"]]
        measured_lm = measured_lm[np.isfinite(measured_lm).all(axis=1)]
        payload["software_gates"] = run_software_gates(plane, measured_lm_rad=measured_lm)
        payload["row_exclusion"] = holography_row_exclusion_counts(observation)
        payload["source_i"] = {
            "median": float(np.median(intensity)),
            "min": float(np.min(intensity)),
            "max": float(np.max(intensity)),
        }
        references = _usable_references(observation, id_to_name)
        payload["usable_references"] = [name for name, _antenna in references]
        held_name = str(arguments.held_reference)
        if held_name not in name_to_id:
            raise KeyError(f"held reference {held_name} is not in the antenna table")
        held_id = int(name_to_id[held_name])
        print("software_gates", payload["software_gates"]["passed"], flush=True)
        split = loro_masks(
            observation,
            held_reference_id=held_id,
            max_train_per_cell=int(arguments.max_train_per_cell),
        )
        payload["convention_ladder"] = lock_cassbeam_convention(
            observation,
            residual_jones=residual,
            catalog=catalog,
            train_mask=split["train"],
            parallactic_angle_rad=chi,
            geometry=split["geometry"],
            candidates=convention_ladder(),
        )
        print(
            "convention",
            payload["convention_ladder"]["status"],
            payload["convention_ladder"].get("selected"),
            flush=True,
        )
        convention = payload["convention_ladder"].get("selected_convention")
        if convention is not None and payload["software_gates"]["passed"]:
            scored = _score_reference(
                observation,
                name=held_name,
                held_ref=held_id,
                residual_jones=residual,
                catalog=catalog,
                convention=convention,
                chi=chi,
                intensity=intensity,
                voltage=voltage,
                max_train_per_cell=int(arguments.max_train_per_cell),
                fit_alpha_on_training=True,
            )
            pred_full = scored.pop("pred_full")
            pred_diag = scored.pop("pred_diag")
            split = scored.pop("split")
            payload["channel32_ea26"] = {
                key: value
                for key, value in scored.items()
                if key not in {"pred_full", "pred_diag", "split"}
            }
            payload["template_injection"] = template_injection_curve(
                observation,
                pred_full,
                pred_diag,
                train_mask=split["train"],
                holdout_mask=split["exact"],
                geometry=split["geometry"],
            )
            payload["qu_nuisance"] = qu_nuisance_robustness(
                observation,
                residual_jones=residual,
                catalog=catalog,
                convention=convention,
                row_mask=split["train"],
                parallactic_angle_rad=chi,
                geometry=split["geometry"],
            )
            plot_dir = arguments.output_dir / "plots"
            payload["plots"] = write_template_plots(
                plot_dir,
                observation=observation,
                pred_full=pred_full,
                pred_diag=pred_diag,
                row_mask=split["exact"],
                geometry=split["geometry"],
                voltage=voltage,
            )
            print(
                "channel32",
                held_name,
                "alpha",
                (scored["holdout"].get("alpha") or {}).get("real"),
                flush=True,
            )
            expand_refs = arguments.stage in {"all_references", "all_channels", "all"} or (
                arguments.expand_on_close
            )
            expand_channels = arguments.stage in {"all_channels", "all"} or (
                arguments.expand_on_close and arguments.stage != "all_references"
            )
            if expand_refs:
                per_reference = {held_name: payload["channel32_ea26"]}
                for name, antenna in references:
                    if name == held_name:
                        continue
                    print("loro", name, flush=True)
                    report = _score_reference(
                        observation,
                        name=name,
                        held_ref=antenna,
                        residual_jones=residual,
                        catalog=catalog,
                        convention=convention,
                        chi=chi,
                        intensity=intensity,
                        voltage=voltage,
                        max_train_per_cell=int(arguments.max_train_per_cell),
                        fit_alpha_on_training=True,
                    )
                    report.pop("pred_full", None)
                    report.pop("pred_diag", None)
                    report.pop("split", None)
                    per_reference[name] = report
                payload["per_reference"] = per_reference
            if expand_channels:
                per_channel = {}
                for channel in range(64):
                    print("channel", channel, flush=True)
                    try:
                        ch_obs, ch_chi, _plane, ch_i, ch_v = _load_observation(
                            diag,
                            tables,
                            arguments.measurement_set,
                            channel=channel,
                            positions=positions,
                        )
                        ch_freq = float(ch_obs.block.frequency_hz[0])
                        catalog.require_exact_mhz(ch_freq)
                    except Exception as error:
                        per_channel[str(channel)] = {
                            "error": f"{type(error).__name__}: {error}",
                        }
                        continue
                    report = _score_reference(
                        ch_obs,
                        name=held_name,
                        held_ref=held_id,
                        residual_jones=residual,
                        catalog=catalog,
                        convention=convention,
                        chi=ch_chi,
                        intensity=ch_i,
                        voltage=ch_v,
                        max_train_per_cell=int(arguments.max_train_per_cell),
                        fit_alpha_on_training=True,
                    )
                    report.pop("pred_full", None)
                    report.pop("pred_diag", None)
                    report.pop("split", None)
                    report["frequency_hz"] = ch_freq
                    per_channel[str(channel)] = report
                payload["per_channel"] = per_channel
                payload["joint_channel"] = {
                    "status": "not_a_channel_block",
                    "reason": (
                        "Per-channel JSON stores scores, not row-level "
                        "templates. A channel-block bootstrap needs the "
                        "in-memory row payload."
                    ),
                    "sqrt_n_not_assumed": True,
                }
        hold = (payload.get("channel32_ea26") or {}).get("holdout") or {}
        paired = hold.get("paired_scores") or {}
        fitted = hold.get("fitted_paired_scores") or {}
        payload["gate"] = classify_highres_cassbeam_direct(
            software_gates_passed=bool(payload["software_gates"]["passed"]),
            convention_status=str(payload["convention_ladder"]["status"]),
            injection=payload.get("template_injection"),
            holdout_alpha=hold.get("alpha"),
            holdout_paired=paired.get("main_lobe") or paired.get("all"),
            holdout_fitted_paired=fitted.get("main_lobe") or fitted.get("all"),
        )
        payload["status"] = payload["gate"]["status"]
        payload["blocking"] = payload["gate"]["blocking"]
        payload["spw5_closed"] = True
        payload["full_jones_frozen"] = False
    except Exception as error:
        print("error", error, flush=True)
        traceback.print_exc()
        payload["error"] = f"{type(error).__name__}: {error}"
        payload["gate"] = {
            "gate": HIGHRES_CASSBEAM_DIRECT,
            "status": "fail",
            "blocking": True,
            "outcome": "runner_error",
            "full_jones_blocked": True,
            "full_jones_frozen": False,
            "spw5_closed": True,
        }
    if "selected_convention" in (payload.get("convention_ladder") or {}):
        payload["convention_ladder"] = {
            key: value
            for key, value in payload["convention_ladder"].items()
            if key != "selected_convention"
        }
    (arguments.output_dir / "README.md").write_text(_readme(payload))
    write_json(_jsonable(payload), arguments.output_dir / "highres_cassbeam_direct_report.json")
    print(arguments.output_dir / "highres_cassbeam_direct_report.json")
    print("status", payload.get("status"), (payload.get("gate") or {}).get("outcome"))
    outcome = str((payload.get("gate") or {}).get("outcome") or "")
    if outcome in {"runner_error", "software_gate_failed"} or payload.get("error"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
