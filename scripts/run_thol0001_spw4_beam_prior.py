"""SPW-4 multi-channel visibility-domain holography beam prior.

Uses all 64 native channels jointly. Does not average them into one
visibility, open SPW 5, freeze full Jones, or modify the production factory.
"""

from __future__ import annotations

import argparse
import hashlib
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
    apparent_voltage_response,
    holoraster_pair_masks,
)
from sl1mjax.holography_beam_prior import (
    CHANNEL_COVARIANCE_NOTE,
    INJECTION_NOTE,
    INJECTION_VOLTAGE,
    NO_HOLORASTER_IN_DI_NOTE,
    PRIOR_NOTE,
    SPW4_MULTICHANNEL_BEAM_PRIOR,
    all_conventions,
    assert_origin_uninjected,
    assert_split_isolation,
    beams_from_native_unique,
    bootstrap_complex_from_moments,
    bootstrap_increment_from_moments,
    classify_beam_prior_decision,
    contiguous_channel_block_masks,
    convention_equivalence_classes,
    fit_frequency_smooth_alpha,
    fit_per_antenna_if_supported,
    fit_spatial_shrinkage,
    fit_unit_and_scalar,
    hand_weight_cube,
    inject_voltage_leakage,
    moments_from_template,
    paired_power_improves,
    predict_from_moving_beams,
    prior_to_dict,
    residual_power_at,
    select_convention_class,
    spatial_convention_key,
    supported_prior_from_decision,
    unique_native_jones,
    vis_planes,
    write_beam_prior_plots,
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
    holography_one_axis_holdout_masks,
    moving_reference_row_geometry,
    thin_training_rows,
)
from sl1mjax.holography_highres_cassbeam import (
    DEFAULT_CONVENTION,
    artifact_checksum_report,
    run_software_gates,
)
from sl1mjax.holography_ms import _read_antennas, _tables, audit_holography_measurement_set
from sl1mjax.holography_reference_jones import (
    NO_SCAN_OFFSETS_OR_TIME_FREEDOM_NOTE,
    deserialize_reference_jones,
)
from sl1mjax.polarization import Receptor, circular_stokes_to_coherency, pack_coherency

SCIENTIFIC_MS = Path(
    "/media/stephen/astro/vla/extracted/commissioning/work/THOL0001.lowerC.spw45.scientific.ms"
)
UNBLOCK = Path(
    "/media/stephen/astro/vla/extracted/commissioning/validation/scientific/full_jones_unblock"
)
HELD_REFERENCE_NAME = "ea26"
MIN_AXIS_ROWS = 400
CONVENTION_TRAIN_ROWS = 4000
NATIVE_CHANNEL_COUNT = 64


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


def _source_revision() -> dict[str, object]:
    isolated = Path("/tmp/sl1mjax-pointing-audit/source_revision.json")
    if isolated.is_file():
        return json.loads(isolated.read_text())
    root = Path("/tmp/sl1mjax-pointing-audit/sl1mjax")
    if not root.is_dir():
        return {"status": "not_on_bacchus"}
    files = sorted(root.rglob("*.py"))
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return {
        "tree": str(root),
        "n_py": len(files),
        "tree_sha256": digest.hexdigest(),
    }


def _load_spw4(diag, tables, measurement_set, positions):
    audit = audit_holography_measurement_set(measurement_set)
    ddid = diag._ddid_for_spw(tables, measurement_set, 4)
    block, source = diag._holoraster_channel_block(
        tables,
        measurement_set,
        data_desc_id=ddid,
        channel=0,
        channel_stop=NATIVE_CHANNEL_COUNT,
        data_column="CORRECTED_DATA",
        spectral_window_id=4,
    )
    if block.frequency_hz.size != NATIVE_CHANNEL_COUNT:
        raise ValueError(
            f"expected {NATIVE_CHANNEL_COUNT} native SPW-4 channels, got {block.frequency_hz.size}"
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
    intensity = source_model_stokes_i(observation)
    rr_ok, ll_ok = copolar_hand_active_rows(observation)
    voltage = apparent_voltage_response(vis_planes(packed)[:, 0], intensity, rr_ok, ll_ok)
    return observation, chi, packed, intensity, voltage


def _row_fields(geometry, pair, chi, rows):
    rows = np.asarray(rows, dtype=np.int64).reshape(-1)
    time_index = np.asarray(geometry["time_index"], dtype=np.int32)[rows]
    moving = np.asarray(geometry["moving_id"], dtype=np.int32)[rows]
    reference = np.asarray(geometry["reference_id"], dtype=np.int32)[rows]
    return {
        "rows": rows,
        "offset": np.asarray(geometry["offset_lm_rad"], dtype=np.float64)[rows],
        "moving": moving,
        "reference": reference,
        "moving_is_p": np.asarray(pair["moving_is_p"], dtype=bool)[rows],
        "chi_m": chi[time_index, moving],
        "chi_r": chi[time_index, reference],
    }


def _predict_pair(
    native, valid, inverse, convention, catalog, frequencies, fields, residual, source
):
    full, ok = beams_from_native_unique(
        native,
        valid,
        inverse,
        convention,
        catalog=catalog,
        frequencies_hz=frequencies,
        chi=fields["chi_m"],
        off_diagonal=True,
        calibration_state="casa_parang_true",
    )
    diag, _ok_d = beams_from_native_unique(
        native,
        valid,
        inverse,
        convention,
        catalog=catalog,
        frequencies_hz=frequencies,
        chi=fields["chi_m"],
        off_diagonal=False,
        calibration_state="casa_parang_true",
    )
    pred_full = predict_from_moving_beams(
        residual,
        fields["moving"],
        fields["reference"],
        fields["moving_is_p"],
        fields["chi_m"],
        fields["chi_r"],
        full,
        source,
    )
    pred_diag = predict_from_moving_beams(
        residual,
        fields["moving"],
        fields["reference"],
        fields["moving_is_p"],
        fields["chi_m"],
        fields["chi_r"],
        diag,
        source,
    )
    pred_full = np.where(ok[..., None, None], pred_full, np.nan)
    pred_diag = np.where(ok[..., None, None], pred_diag, np.nan)
    return pred_full, pred_diag


def _moments(observation, geometry, rows, template, residual, n_channel, *, spatial_groups=True):
    weight = hand_weight_cube(observation, _mask_from_rows(observation, rows))
    return moments_from_template(
        template,
        residual,
        weight,
        geometry,
        rows,
        n_channel=n_channel,
        spatial_groups=spatial_groups,
    )


def _mask_from_rows(observation, rows) -> np.ndarray:
    mask = np.zeros(observation.block.time_s.size, dtype=bool)
    mask[np.asarray(rows, dtype=np.int64)] = True
    return mask


def _source_slice_with_qu(source_rows, q_over_i: float, u_over_i: float):
    intensity = 0.5 * (np.asarray(source_rows)[..., 0, 0] + np.asarray(source_rows)[..., 1, 1])
    return circular_stokes_to_coherency(
        intensity,
        float(q_over_i) * intensity,
        float(u_over_i) * intensity,
        0.0 * intensity,
    )


def _axis_masks(observation, *, held_ref, held_mover):
    axes = {
        "leave_one_reference_out": holography_one_axis_holdout_masks(
            observation,
            axis="leave_one_reference_out",
            held_reference_id=int(held_ref),
            reserve_outer_fold=True,
            require_all_channels=False,
        ),
        "leave_one_visit_out": holography_one_axis_holdout_masks(
            observation,
            axis="leave_one_visit_out",
            reserve_outer_fold=True,
            require_all_channels=False,
        ),
        "spatial_checkerboard": holography_one_axis_holdout_masks(
            observation,
            axis="spatial_checkerboard",
            reserve_outer_fold=True,
            require_all_channels=False,
        ),
    }
    if held_mover is not None:
        axes["leave_one_mover_out"] = holography_one_axis_holdout_masks(
            observation,
            axis="leave_one_mover_out",
            held_moving_id=int(held_mover),
            reserve_outer_fold=True,
            require_all_channels=False,
        )
    for name, split in axes.items():
        assert_split_isolation(split["train"], split["holdout"])
        if bool(np.any(split["train"] & split["scored"])):
            raise ValueError(f"contaminated split on {name}: train overlaps scored holdout")
    return axes


def _readme(payload: dict) -> str:
    gate = payload.get("gate") or {}
    prior = payload.get("holography_beam_prior") or {}
    lines = [
        "# THOL0001 SPW-4 multi-channel holography beam prior",
        "",
        f"Status: **{gate.get('status', 'unknown')}**. "
        f"Decision: `{gate.get('decision', 'unknown')}`.",
        "",
        PRIOR_NOTE,
        "",
        INJECTION_NOTE,
        "",
        CHANNEL_COVARIANCE_NOTE,
        "",
        NO_HOLORASTER_IN_DI_NOTE,
        "",
        FULL_JONES_BLOCKED_UNTIL_ALIGNMENT_NOTE,
        "",
        f"Supported quantities: {prior.get('supported_quantities')}.",
        "",
        "Full Jones remains unfrozen. The production factory is unmodified. SPW 5 stays closed.",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--measurement-set", type=Path, default=SCIENTIFIC_MS)
    parser.add_argument("--product-dir", type=Path, default=UNBLOCK)
    parser.add_argument("--output-dir", type=Path, default=UNBLOCK / "spw4_beam_prior")
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_HIGHRES_ROOT)
    parser.add_argument("--spectral-window", type=int, default=4)
    parser.add_argument("--held-reference", default=HELD_REFERENCE_NAME)
    parser.add_argument("--max-train-per-cell", type=int, default=4)
    parser.add_argument("--convention-train-rows", type=int, default=CONVENTION_TRAIN_ROWS)
    parser.add_argument(
        "--resume-convention-checkpoint",
        type=Path,
        default=None,
        help="Reuse a prior convention lock instead of rescoring 128 candidates",
    )
    arguments = parser.parse_args()
    if int(arguments.spectral_window) != 4:
        raise ValueError("SPW 5 stays closed")
    forbidden = {
        arguments.product_dir.resolve(),
        (arguments.product_dir / "forward_closure").resolve(),
        (arguments.product_dir / "reference_visit_alignment").resolve(),
        (arguments.product_dir / "loro_full_versus_diagonal").resolve(),
        (arguments.product_dir / "loro_leakage_sensitivity").resolve(),
        (arguments.product_dir / "highres_cassbeam_direct").resolve(),
        (arguments.product_dir / "highres_cassbeam_direct" / "previous_mixed_scorer").resolve(),
        (
            arguments.product_dir / "highres_cassbeam_direct" / "previous_convention_unresolved"
        ).resolve(),
    }
    if arguments.output_dir.resolve() in forbidden:
        raise ValueError("refusing to overwrite earlier scientific products")
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "thol0001_spw4_beam_prior_v1",
        "gate_name": SPW4_MULTICHANNEL_BEAM_PRIOR,
        "status": "fail",
        "blocking": True,
        "spectral_window_id": 4,
        "n_native_channels": NATIVE_CHANNEL_COUNT,
        "channels_averaged": False,
        "held_reference": str(arguments.held_reference),
        "holoraster_in_di_fit": False,
        "committed_33x33_loaded": False,
        "nearest_frequency_substitution": False,
        "spw5_closed": True,
        "full_jones_blocked": True,
        "full_jones_frozen": False,
        "production_factory_modified": False,
        "source_revision": _source_revision(),
        "preserved_convention_unresolved": str(
            arguments.product_dir / "highres_cassbeam_direct" / "previous_convention_unresolved"
        ),
        "do_not_overwrite": [
            "one_axis_visibility_holdouts.json",
            "one_axis_visibility_holdouts_v2.json",
            "forward_closure/",
            "reference_visit_alignment/",
            "loro_full_versus_diagonal/",
            "loro_leakage_sensitivity/",
            "highres_cassbeam_direct/",
            "highres_cassbeam_direct/previous_convention_unresolved/",
        ],
        "notes": (
            PRIOR_NOTE,
            INJECTION_NOTE,
            CHANNEL_COVARIANCE_NOTE,
            NO_HOLORASTER_IN_DI_NOTE,
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
        print("loading 64 native SPW-4 channels", flush=True)
        observation, chi, packed, intensity, voltage = _load_spw4(
            diag, tables, arguments.measurement_set, positions
        )
        frequencies = np.asarray(observation.block.frequency_hz, dtype=np.float64)
        payload["frequency_hz"] = frequencies.tolist()
        payload["artifact"] = artifact_checksum_report(catalog, float(frequencies[32]))
        print("preloading CASSBEAM planes", flush=True)
        for freq in frequencies:
            catalog.require_exact_mhz(float(freq))
            catalog.plane(float(freq))
        pair = holoraster_pair_masks(observation)
        measured_lm = pair["offset_lm_rad"][pair["moving_reference"]]
        measured_lm = measured_lm[np.isfinite(measured_lm).all(axis=1)]
        plane = catalog.plane(float(frequencies[32]))
        payload["software_gates"] = run_software_gates(plane, measured_lm_rad=measured_lm)
        payload["row_exclusion"] = holography_row_exclusion_counts(observation)
        payload["source_i"] = {
            "median": float(np.median(intensity)),
            "min": float(np.min(intensity)),
            "max": float(np.max(intensity)),
        }
        if not payload["software_gates"]["passed"]:
            raise RuntimeError("software_gate_failed")
        held_name = str(arguments.held_reference)
        held_id = int(name_to_id[held_name])
        geometry = moving_reference_row_geometry(observation, require_all_channels=False)
        usable = np.asarray(geometry["usable"], dtype=bool)
        movers, mover_counts = np.unique(geometry["moving_id"][usable], return_counts=True)
        held_mover = None
        if movers.size >= 2 and int(np.max(mover_counts)) >= MIN_AXIS_ROWS:
            held_mover = int(movers[int(np.argmax(mover_counts))])
        axes = _axis_masks(observation, held_ref=held_id, held_mover=held_mover)
        loro = axes["leave_one_reference_out"]
        train = thin_training_rows(
            observation,
            loro["train"],
            max_per_cell=int(arguments.max_train_per_cell),
            require_all_channels=False,
        )
        payload["row_accounting"] = {
            name: {
                "train": int(np.sum(split["train"])),
                "holdout": int(np.sum(split["holdout"])),
                "scored": int(np.sum(split["scored"])),
                "train_holdout_overlap": int(np.sum(split["train"] & split["holdout"])),
            }
            for name, split in axes.items()
        }
        payload["row_accounting"]["loro_train_thinned"] = int(np.sum(train))
        payload["held_mover_id"] = held_mover
        payload["holdout_axes_unioned"] = False
        channel_hold = contiguous_channel_block_masks(frequencies.size)
        payload["channel_block"] = {
            "hold_start": int(channel_hold["hold_start"]),
            "hold_stop": int(channel_hold["hold_stop"]),
            "n_train": int(np.sum(channel_hold["train"])),
            "n_holdout": int(np.sum(channel_hold["holdout"])),
        }
        source = np.asarray(observation.source_coherency_visibility, dtype=np.complex128)
        measured = vis_planes(packed)
        origin = np.asarray(geometry["radius_rad"]) <= 1.0e-6

        lock_mask = np.asarray(train, dtype=bool)
        lock_index = np.flatnonzero(lock_mask)
        if lock_index.size > int(arguments.convention_train_rows):
            lock_index = np.random.default_rng(0).choice(
                lock_index, size=int(arguments.convention_train_rows), replace=False
            )
            lock_mask[:] = False
            lock_mask[lock_index] = True
        lock_fields = _row_fields(geometry, pair, chi, lock_index)
        checkpoint_path = arguments.resume_convention_checkpoint
        if checkpoint_path is None:
            default_ckpt = arguments.output_dir / "convention_checkpoint.json"
            if default_ckpt.is_file():
                checkpoint_path = default_ckpt
        if checkpoint_path is not None and checkpoint_path.is_file():
            saved = json.loads(checkpoint_path.read_text())
            payload["convention_equivalence"] = saved.get("convention_equivalence")
            payload["selected_convention"] = (
                saved.get("selected_convention") or DEFAULT_CONVENTION.name
            )
            payload["software_gates"] = saved.get("software_gates") or payload["software_gates"]
            print("resumed convention", payload["selected_convention"], flush=True)
            selected = next(
                item for item in all_conventions() if item.name == payload["selected_convention"]
            )
            convention_report = (payload.get("convention_equivalence") or {}).get("selection") or {
                "status": "convention_unresolved"
            }
            spatial_cache = {}
            templates = {}
        else:
            spatial_cache = {}
            templates = {}
            class_scores = {}
            print(
                "scoring",
                len(all_conventions()),
                "conventions on",
                lock_index.size,
                "rows",
                flush=True,
            )
            for convention in all_conventions():
                key = spatial_convention_key(convention)
                if key not in spatial_cache:
                    spatial_cache[key] = unique_native_jones(
                        lock_fields["offset"],
                        lock_fields["chi_m"],
                        frequencies,
                        catalog,
                        convention,
                    )
                    print(
                        "spatial cache", key, "unique", spatial_cache[key][0].shape[0], flush=True
                    )
                native, valid, inverse = spatial_cache[key]
                pred_full, pred_diag = _predict_pair(
                    native,
                    valid,
                    inverse,
                    convention,
                    catalog,
                    frequencies,
                    lock_fields,
                    residual,
                    source[lock_index],
                )
                template = pred_full - pred_diag
                residual_vis = measured[lock_index] - pred_diag
                moments = _moments(
                    observation, geometry, lock_index, template, residual_vis, frequencies.size
                )
                smooth = fit_frequency_smooth_alpha(moments)
                templates[convention.name] = template[:, 32]
                class_scores[convention.name] = float(smooth["power"])
            weight_lock = hand_weight_cube(observation, lock_mask)[:, 32]
            classes = convention_equivalence_classes(templates, weight_lock)
            class_power = {}
            for group in classes:
                class_power[group[0]] = min(class_scores[name] for name in group)
            convention_report = select_convention_class(class_power, classes)
            payload["convention_equivalence"] = {
                "n_candidates": len(all_conventions()),
                "n_classes": len(classes),
                "classes": classes[:16],
                "selection": convention_report,
            }
            selected_name = convention_report.get("selected") or DEFAULT_CONVENTION.name
            selected = next(item for item in all_conventions() if item.name == selected_name)
            payload["selected_convention"] = selected.name
            print("convention", convention_report["status"], selected.name, flush=True)
            write_json(_jsonable(payload), arguments.output_dir / "convention_checkpoint.json")
            del spatial_cache
            del templates
        qu_scores = []
        for q_over_i, u_over_i in (
            (0.0, 0.0),
            (-0.001, 0.0),
            (0.001, 0.0),
            (0.0, -0.001),
            (0.0, 0.001),
        ):
            native, valid, inverse = unique_native_jones(
                lock_fields["offset"],
                lock_fields["chi_m"],
                frequencies,
                catalog,
                selected,
            )
            pred_full, pred_diag = _predict_pair(
                native,
                valid,
                inverse,
                selected,
                catalog,
                frequencies,
                lock_fields,
                residual,
                _source_slice_with_qu(source[lock_index], q_over_i, u_over_i),
            )
            moments = _moments(
                observation,
                geometry,
                lock_index,
                pred_full - pred_diag,
                measured[lock_index] - pred_diag,
                frequencies.size,
            )
            qu_scores.append(
                {
                    "q_over_i": q_over_i,
                    "u_over_i": u_over_i,
                    "power": fit_frequency_smooth_alpha(moments)["power"],
                }
            )
        qu_best = min(qu_scores, key=lambda item: item["power"])
        payload["qu_nuisance"] = {
            "independent_q_and_u": True,
            "evaluated_on": "training_rows",
            "points": qu_scores,
            "selected": qu_best,
        }
        print("qu", qu_best, flush=True)

        def _predict_mask(mask, *, chunk=4000, keep_templates=True):
            rows = np.flatnonzero(np.asarray(mask, dtype=bool))
            templates = []
            residuals = []
            kept = []
            moments = []
            for start in range(0, max(rows.size, 1), chunk):
                if rows.size == 0:
                    break
                part = rows[start : start + chunk]
                fields = _row_fields(geometry, pair, chi, part)
                native, valid, inverse = unique_native_jones(
                    fields["offset"],
                    fields["chi_m"],
                    frequencies,
                    catalog,
                    selected,
                )
                pred_full, pred_diag = _predict_pair(
                    native,
                    valid,
                    inverse,
                    selected,
                    catalog,
                    frequencies,
                    fields,
                    residual,
                    _source_slice_with_qu(
                        source[part],
                        float(qu_best["q_over_i"]),
                        float(qu_best["u_over_i"]),
                    ),
                )
                template = pred_full - pred_diag
                resid = measured[part] - pred_diag
                moments.extend(
                    _moments(
                        observation,
                        geometry,
                        part,
                        template,
                        resid,
                        frequencies.size,
                        spatial_groups=keep_templates,
                    )
                )
                if keep_templates:
                    templates.append(template)
                    residuals.append(resid)
                    kept.append(part)
                print("predicted", min(start + part.size, rows.size), "/", rows.size, flush=True)
            if keep_templates:
                if not kept:
                    empty_t = np.zeros((0, frequencies.size, 2, 2), dtype=np.complex128)
                    return np.zeros(0, dtype=np.int64), empty_t, empty_t.copy(), moments
                return (
                    np.concatenate(kept),
                    np.concatenate(templates),
                    np.concatenate(residuals),
                    moments,
                )
            return rows, None, None, moments

        print("fitting nested models on", lock_index.size, "training rows", flush=True)
        train_rows, train_t, train_r, train_moments = _predict_mask(lock_mask, keep_templates=True)
        nested = {
            "diagonal": fit_unit_and_scalar(train_moments),
            "cassbeam_unit": {"alpha": 1.0 + 0.0j, "power": residual_power_at(train_moments, 1.0)},
            "frequency_smooth": fit_frequency_smooth_alpha(train_moments),
            "spatial_shrinkage": fit_spatial_shrinkage(train_moments),
            "per_antenna": fit_per_antenna_if_supported(train_moments),
        }
        payload["nested_models_training"] = {
            "n": nested["diagonal"]["n"],
            "alpha_hat": nested["diagonal"]["alpha_hat"],
            "power_0": nested["diagonal"]["power_0"],
            "power_1": nested["diagonal"]["power_1"],
            "power_hat": nested["diagonal"]["power_hat"],
            "power_smooth": nested["frequency_smooth"]["power"],
            "power_spatial": nested["spatial_shrinkage"]["power"],
            "power_antenna": nested["per_antenna"]["power"],
        }

        holdouts = {}
        for name, split in axes.items():
            print("predicting holdout", name, int(np.sum(split["scored"])), flush=True)
            hold_rows, _hold_t, _hold_r, hold_moments = _predict_mask(
                split["scored"], keep_templates=False
            )
            if hold_rows.size < MIN_AXIS_ROWS:
                holdouts[name] = {"status": "insufficient_rows", "n": int(hold_rows.size)}
                continue
            alpha_hat = complex(nested["diagonal"]["alpha_hat"])
            print("scoring holdout", name, "n_moments", len(hold_moments), flush=True)
            holdouts[name] = {
                "n": int(hold_rows.size),
                "n_moments": len(hold_moments),
                "alpha": bootstrap_complex_from_moments(hold_moments),
                "power_0": residual_power_at(hold_moments, 0.0),
                "power_1": residual_power_at(hold_moments, 1.0),
                "power_hat": residual_power_at(hold_moments, alpha_hat),
                "a1_improves": paired_power_improves(hold_moments, 1.0 + 0.0j),
                "hat_improves": paired_power_improves(hold_moments, alpha_hat),
            }
        train_chan = np.flatnonzero(channel_hold["train"])
        hold_chan = np.flatnonzero(channel_hold["holdout"])
        chan_train = [item for item in train_moments if item.channel in set(train_chan.tolist())]
        chan_fit = fit_unit_and_scalar(chan_train) if chan_train else {"alpha_hat": 0.0 + 0.0j}
        chan_hold_moments = moments_from_template(
            train_t[:, hold_chan],
            train_r[:, hold_chan],
            hand_weight_cube(observation, _mask_from_rows(observation, train_rows))[:, hold_chan],
            geometry,
            train_rows,
            n_channel=int(hold_chan.size),
        )
        # Re-index channel ids to native numbers for reporting.
        holdouts["contiguous_channel_block"] = {
            "n": int(train_rows.size * hold_chan.size),
            "n_channels": int(hold_chan.size),
            "alpha": bootstrap_complex_from_moments(chan_hold_moments),
            "alpha_from_train_channels": chan_fit["alpha_hat"],
            "power_0": residual_power_at(chan_hold_moments, 0.0),
            "power_1": residual_power_at(chan_hold_moments, 1.0),
            "power_hat": residual_power_at(chan_hold_moments, complex(chan_fit["alpha_hat"])),
            "trained_on_channels_only": True,
            "train_channels": int(train_chan.size),
        }
        payload["holdouts"] = holdouts
        loro_hold = holdouts.get("leave_one_reference_out") or {}
        antenna_hold = holdouts.get("leave_one_mover_out") or {}
        antenna_supported = bool((antenna_hold.get("hat_improves") or {}).get("improves"))

        print("injection curve", flush=True)
        inj_train_rows, inj_t, inj_r = train_rows, train_t, train_r
        uninjected = train_moments
        origin_sel = origin[inj_train_rows]
        injection = {}
        for amplitude in INJECTION_VOLTAGE:
            copies = inject_voltage_leakage(
                measured[inj_train_rows],
                inj_t,
                amplitude,
                origin_mask=origin_sel,
            )
            assert_origin_uninjected(origin_sel, copies, measured[inj_train_rows])
            injected_resid = copies - (measured[inj_train_rows] - inj_r)
            injected_moments = _moments(
                observation, geometry, inj_train_rows, inj_t, injected_resid, frequencies.size
            )
            increment = bootstrap_increment_from_moments(uninjected, injected_moments)
            injection[f"{amplitude:g}"] = {
                "voltage_amplitude": amplitude,
                "increment": increment,
                "detectable": bool(increment["inconsistent_with_zero"]),
            }
        payload["injection"] = injection
        detectable = {key: bool(item["detectable"]) for key, item in injection.items()}
        if not any(detectable.get(key) for key in ("0.003", "0.01", "0.03")):
            payload["stop_reason"] = (
                "CASSBEAM-scale injection remains undetectable. "
                "Not adding temporal freedom, scan offsets, or unconstrained pixels."
            )

        a1 = bool((loro_hold.get("a1_improves") or {}).get("improves"))
        hat = bool((loro_hold.get("hat_improves") or {}).get("improves"))
        spatial_hold_power = residual_power_at(train_moments, 1.0)
        spatial_improves = nested["spatial_shrinkage"]["power"] < 0.98 * spatial_hold_power and hat
        rr_ll = False
        payload["gate"] = classify_beam_prior_decision(
            software_ok=True,
            split_clean=True,
            injection_detectable=detectable,
            holdout_alpha=loro_hold.get("alpha"),
            a1_improves=a1,
            hat_improves=hat,
            spatial_improves=spatial_improves,
            rr_ll_regression=rr_ll,
            convention_resolved=convention_report["status"] == "locked",
        )
        prior = supported_prior_from_decision(
            decision=str(payload["gate"]["decision"]),
            scalar=nested["diagonal"],
            smooth=nested["frequency_smooth"],
            spatial=nested["spatial_shrinkage"],
            antenna=nested["per_antenna"],
            holdout_alpha=loro_hold.get("alpha"),
            provenance={
                "measurement_set": str(arguments.measurement_set),
                "spectral_window_id": 4,
                "n_channels": NATIVE_CHANNEL_COUNT,
                "convention": selected.name,
                "source_revision": payload["source_revision"],
                "gauge": "on_axis_identity",
                "holoraster_in_di_fit": False,
            },
            antenna_supported=antenna_supported,
        )
        payload["holography_beam_prior"] = prior_to_dict(prior)
        plot_mask = np.asarray(loro["scored"], dtype=bool)
        plot_index = np.flatnonzero(plot_mask)
        if plot_index.size > 4000:
            plot_index = np.random.default_rng(1).choice(plot_index, size=4000, replace=False)
            plot_mask[:] = False
            plot_mask[plot_index] = True
        plot_rows, plot_t, plot_r, _plot_m = _predict_mask(plot_mask, keep_templates=True)
        pred_plot = (measured[plot_rows] - plot_r) + complex(
            nested["diagonal"]["alpha_hat"]
        ) * plot_t
        payload["plots"] = write_beam_prior_plots(
            arguments.output_dir / "plots",
            frequencies_hz=frequencies,
            per_channel_alpha=np.array(
                [stack_if_present(train_moments, channel) for channel in range(frequencies.size)],
                dtype=np.complex128,
            ),
            smooth_alpha=nested["frequency_smooth"]["alpha_channel"],
            measured=measured[plot_rows],
            predicted=pred_plot,
            row_mask=np.ones(plot_rows.size, dtype=bool),
            geometry={
                key: np.asarray(geometry[key])[plot_rows]
                for key in (
                    "moving_id",
                    "reference_id",
                    "offset_lm_rad",
                    "radius_rad",
                    "time_index",
                )
            },
        )
        payload["status"] = payload["gate"]["status"]
        payload["blocking"] = payload["gate"]["blocking"]
        payload["decision"] = payload["gate"]["decision"]
        print("decision", payload["decision"], flush=True)
    except Exception as error:
        print("error", error, flush=True)
        traceback.print_exc()
        payload["error"] = f"{type(error).__name__}: {error}"
        payload["gate"] = classify_beam_prior_decision(
            software_ok=False,
            split_clean=True,
            injection_detectable={},
            holdout_alpha=None,
            a1_improves=False,
            hat_improves=False,
            spatial_improves=False,
            rr_ll_regression=False,
        )
        payload["gate"]["decision"] = "runner_error"
        payload["status"] = "fail"
        payload["blocking"] = True
    (arguments.output_dir / "README.md").write_text(_readme(payload))
    write_json(_jsonable(payload), arguments.output_dir / "spw4_beam_prior_report.json")
    write_json(
        _jsonable(payload.get("holography_beam_prior") or {}),
        arguments.output_dir / "holography_beam_prior.json",
    )
    write_json(
        _jsonable(payload.get("convention_equivalence") or {}),
        arguments.output_dir / "convention_equivalence.json",
    )
    print(arguments.output_dir / "spw4_beam_prior_report.json")
    print("status", payload.get("status"), (payload.get("gate") or {}).get("decision"))
    decision = str((payload.get("gate") or {}).get("decision") or "")
    if decision in {"runner_error", "software_gate_failed", "contaminated_split"} or payload.get(
        "error"
    ):
        return 1
    return 0


def stack_if_present(moments, channel: int) -> complex:
    from sl1mjax.holography_beam_prior import stack_moments

    group = [item for item in moments if int(item.channel) == int(channel)]
    return stack_moments(group).alpha


if __name__ == "__main__":
    raise SystemExit(main())
