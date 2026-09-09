"""Direct HOLORASTER versus CASSBEAM comparison report.

Uses already-applied HOLORASTER CORRECTED_DATA. No calibration fit, no
convention search, and no model selection. Full-Jones panels are labelled
experimental prediction. SPW 5 stays sealed.
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
from sl1mjax.holography import HolographyObservation, THOL0001_SPW4_CHANNEL_32_HZ
from sl1mjax.holography_alignment import apparent_voltage_response, holoraster_pair_masks
from sl1mjax.holography_beam_prior import (
    evaluate_holoraster_cassbeam,
    unique_native_jones,
    vis_planes,
)
from sl1mjax.holography_calibration import write_json
from sl1mjax.holography_cassbeam_holoraster_report import (
    COMPARISON_NOTE,
    DIAGONAL_NOTE,
    FULL_JONES_NOTE,
    HOLORASTER_CASSBEAM_COMPARISON,
    SPW5_CLOSED_NOTE,
    classify_diagonal_region_support,
    classify_holoraster_comparison_report,
    locked_convention,
    quadrant_crosshand_summaries,
    refuse_convention_search,
    region_copolar_summaries,
    residual_geometry_summaries,
    rescore_squint_from_comparison_arrays,
    squint_from_voltage_maps,
    visibility_hand_summaries,
    write_holoraster_comparison_plots,
    write_squint_publication_plot,
)
from sl1mjax.holography_diagonal import copolar_hand_active_rows, source_model_stokes_i
from sl1mjax.holography_full_jones import moving_reference_row_geometry
from sl1mjax.holography_highres_cassbeam import artifact_checksum_report, run_software_gates
from sl1mjax.holography_ms import _read_antennas, _tables, audit_holography_measurement_set
from sl1mjax.holography_reference_jones import deserialize_reference_jones
from sl1mjax.polarization import Receptor, pack_coherency

SCIENTIFIC_MS = Path(
    "/media/stephen/astro/vla/extracted/commissioning/work/THOL0001.lowerC.spw45.scientific.ms"
)
UNBLOCK = Path(
    "/media/stephen/astro/vla/extracted/commissioning/validation/scientific/full_jones_unblock"
)
PRODUCT_DIR = Path(
    "/media/stephen/astro/vla/extracted/commissioning/validation/scientific/"
    "holoraster_cassbeam_comparison"
)
FROZEN_NAMES = (
    "one_axis_visibility_holdouts.json",
    "one_axis_visibility_holdouts_v2.json",
    "forward_closure",
    "reference_visit_alignment",
    "loro_full_versus_diagonal",
    "loro_leakage_sensitivity",
    "highres_cassbeam_direct",
    "spw4_beam_prior",
    "c147_offset_ring",
)
SMOKE_CHANNEL = 32
NATIVE_CHANNEL_COUNT = 64
FREQUENCY_CHANNELS = (0, 8, 16, 24, 32, 40, 48, 56, 63)


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


def _residual_jones(product_dir: Path) -> dict[int, np.ndarray]:
    family = product_dir / "prediction_equivalent_family.json"
    stabilized = product_dir / "field9_stabilized_residual_jones.json"
    if family.is_file():
        return deserialize_reference_jones(json.loads(family.read_text())["nominal"]["jones"])
    if stabilized.is_file():
        return deserialize_reference_jones(json.loads(stabilized.read_text())["jones"])
    raise FileNotFoundError("need field9_stabilized_residual_jones.json or the family cache")


def _load_holoraster_channel(diag, tables, measurement_set, positions, *, channel: int):
    audit = audit_holography_measurement_set(measurement_set)
    ddid = diag._ddid_for_spw(tables, measurement_set, 4)
    block, source = diag._holoraster_channel_block(
        tables,
        measurement_set,
        data_desc_id=ddid,
        channel=int(channel),
        channel_stop=int(channel) + 1,
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
    intensity = source_model_stokes_i(observation)
    rr_ok, ll_ok = copolar_hand_active_rows(observation)
    voltage = apparent_voltage_response(vis_planes(packed)[:, 0], intensity, rr_ok, ll_ok)
    return observation, chi, packed, intensity, voltage, rr_ok, ll_ok


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


def _predict_pair(catalog, frequencies, fields, residual, source):
    convention = locked_convention()
    refuse_convention_search([convention])
    unique = unique_native_jones(
        fields["offset"],
        fields["chi_m"],
        frequencies,
        catalog,
        convention,
    )
    kwargs = {
        "catalog": catalog,
        "frequencies_hz": frequencies,
        "convention": convention,
        "offset_lm_rad": fields["offset"],
        "chi_moving": fields["chi_m"],
        "chi_reference": fields["chi_r"],
        "moving_id": fields["moving"],
        "reference_id": fields["reference"],
        "moving_is_p": fields["moving_is_p"],
        "residual_jones": residual,
        "source": source,
        "unique_lookup": unique,
    }
    full = evaluate_holoraster_cassbeam(off_diagonal=True, **kwargs)
    diag = evaluate_holoraster_cassbeam(off_diagonal=False, **kwargs)
    return full.visibility, diag.visibility


def _hand_weight(observation, row_mask):
    from sl1mjax.holography_beam_prior import hand_weight_cube

    return hand_weight_cube(observation, row_mask)


def _compare_channel(catalog, observation, chi, packed, intensity, voltage, rr_ok, ll_ok, residual):
    geometry = moving_reference_row_geometry(observation, require_all_channels=False)
    pair = holoraster_pair_masks(observation)
    usable = np.asarray(geometry["usable"], dtype=bool) & np.asarray(
        pair["moving_reference"], dtype=bool
    )
    fields = _row_fields(geometry, pair, chi, np.flatnonzero(usable))
    source = np.asarray(observation.source_coherency_visibility, dtype=np.complex128)
    if source.ndim == 3:
        source = source[:, None, :, :]
    pred_full, pred_diag = _predict_pair(
        catalog,
        observation.block.frequency_hz,
        fields,
        residual,
        source[fields["rows"]],
    )
    measured = vis_planes(packed)[fields["rows"]]
    weight = _hand_weight(observation, usable)
    row_mask = np.ones(fields["rows"].size, dtype=bool)
    voltage_rows = np.asarray(voltage, dtype=np.float64)[fields["rows"]]
    rr_rows = np.asarray(rr_ok, dtype=bool)[fields["rows"]]
    ll_rows = np.asarray(ll_ok, dtype=bool)[fields["rows"]]
    intensity_rows = np.asarray(intensity, dtype=np.float64).reshape(-1)[fields["rows"]]
    summaries = {
        "diagonal": visibility_hand_summaries(measured, pred_diag, weight, row_mask),
        "experimental_full_jones": visibility_hand_summaries(measured, pred_full, weight, row_mask),
        "regions": region_copolar_summaries(
            measured,
            pred_diag,
            weight,
            intensity_rows,
            rr_rows,
            ll_rows,
            row_mask,
            voltage=voltage_rows,
        ),
        "quadrants": quadrant_crosshand_summaries(
            measured,
            pred_full,
            pred_diag,
            weight,
            fields["offset"],
            row_mask,
        ),
        "residual_geometry": residual_geometry_summaries(
            measured - pred_diag,
            weight,
            fields["offset"],
            np.arange(measured.shape[1], dtype=np.int64),
            row_mask,
        ),
        "squint_measured": squint_from_voltage_maps(
            fields["offset"],
            np.abs(measured[:, 0, 0, 0]) ** 2,
            np.abs(measured[:, 0, 1, 1]) ** 2,
            weight[:, 0, 0, 0],
            weight[:, 0, 1, 1],
            frequency_hz=float(observation.block.frequency_hz[0]),
            series="measured",
        ),
        "squint_cassbeam": squint_from_voltage_maps(
            fields["offset"],
            np.abs(pred_diag[:, 0, 0, 0]) ** 2,
            np.abs(pred_diag[:, 0, 1, 1]) ** 2,
            weight[:, 0, 0, 0],
            weight[:, 0, 1, 1],
            frequency_hz=float(observation.block.frequency_hz[0]),
            series="cassbeam",
        ),
        "n_rows": int(fields["rows"].size),
        "frequency_hz": float(observation.block.frequency_hz[0]),
    }
    arrays = {
        "measured": measured,
        "predicted_diag": pred_diag,
        "predicted_full": pred_full,
        "weight": weight,
        "offset": fields["offset"],
        "moving": fields["moving"],
        "reference": fields["reference"],
        "mask": row_mask,
    }
    finite = bool(np.any(np.isfinite(pred_diag)) and np.any(np.isfinite(pred_full)))
    return summaries, arrays, finite


def _readme(payload: dict) -> str:
    gate = payload.get("decision_gate")
    if not isinstance(gate, dict):
        gate = payload
    lines = [
        "# THOL0001 HOLORASTER versus CASSBEAM comparison",
        "",
        f"Status: **{gate.get('status', 'unknown')}**. "
        f"Decision: `{gate.get('decision', 'unknown')}`.",
        "",
        COMPARISON_NOTE,
        "",
        DIAGONAL_NOTE,
        "",
        FULL_JONES_NOTE,
        "",
        SPW5_CLOSED_NOTE,
        "",
        "No model was selected. Full Jones remains unfrozen. The production "
        "factory is unmodified.",
        "",
    ]
    return "\n".join(lines) + "\n"


def _write_payload(output_dir: Path, payload: dict) -> None:
    write_json(_jsonable(payload), output_dir / "holoraster_cassbeam_comparison.json")
    (output_dir / "README.md").write_text(_readme(payload))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--measurement-set", type=Path, default=SCIENTIFIC_MS)
    parser.add_argument("--product-dir", type=Path, default=UNBLOCK)
    parser.add_argument("--output-dir", type=Path, default=PRODUCT_DIR)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_HIGHRES_ROOT)
    parser.add_argument("--spectral-window", type=int, default=4)
    parser.add_argument(
        "--stage",
        choices=("channel32", "frequency", "plots", "squint", "all"),
        default="all",
    )
    arguments = parser.parse_args()
    if int(arguments.spectral_window) != 4:
        raise ValueError("HOLORASTER comparison is defined for SPW 4; SPW 5 stays sealed")
    output_dir = arguments.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frozen = {name: (arguments.product_dir / name).resolve() for name in FROZEN_NAMES}
    if output_dir in set(frozen.values()) or output_dir.name in FROZEN_NAMES:
        raise RuntimeError(f"refusing to write into a frozen product path: {output_dir}")
    payload = {
        "gate": HOLORASTER_CASSBEAM_COMPARISON,
        "stage": arguments.stage,
        "source_revision": _source_revision(),
        "measurement_set": str(arguments.measurement_set),
        "full_jones_label": "experimental prediction",
        "model_selected": False,
        "spw5_closed": True,
    }
    try:
        refuse_convention_search(None)
        catalog = HighresCassbeamCatalog(arguments.artifact_root)
        plane = catalog.plane(THOL0001_SPW4_CHANNEL_32_HZ)
        software = run_software_gates(plane)
        payload["software_gates"] = software
        payload["artifact_checksum"] = artifact_checksum_report(
            catalog, THOL0001_SPW4_CHANNEL_32_HZ
        )
        software_ok = bool(software.get("passed", False))
        residual = _residual_jones(arguments.product_dir)
        diag = _diag()
        tables = _tables()
        _ids, _names, positions = _read_antennas(tables, arguments.measurement_set)
        positions = np.asarray(positions, dtype=np.float64)
        plots: list[str] = []
        finite = False
        saved_npz = output_dir / "channel32_comparison.npz"
        if arguments.stage in {"frequency", "plots", "squint"} and saved_npz.is_file():
            print("resuming from", saved_npz, flush=True)
            prior = output_dir / "channel32_summaries.json"
            if prior.is_file():
                payload["channel32"] = json.loads(prior.read_text())
                finite = True
        elif arguments.stage in {"channel32", "all", "frequency"}:
            observation, chi, packed, intensity, voltage, rr_ok, ll_ok = _load_holoraster_channel(
                diag,
                tables,
                arguments.measurement_set,
                positions,
                channel=SMOKE_CHANNEL,
            )
            summaries, arrays, finite = _compare_channel(
                catalog,
                observation,
                chi,
                packed,
                intensity,
                voltage,
                rr_ok,
                ll_ok,
                residual,
            )
            payload["channel32"] = summaries
            np.savez_compressed(output_dir / "channel32_comparison.npz", **arrays)
            write_json(_jsonable(summaries), output_dir / "channel32_summaries.json")
            plots.extend(
                write_holoraster_comparison_plots(
                    output_dir / "plots",
                    measured=arrays["measured"],
                    predicted_diag=arrays["predicted_diag"],
                    predicted_full=arrays["predicted_full"],
                    weight=arrays["weight"],
                    offset_lm_rad=arrays["offset"],
                    moving_id=arrays["moving"],
                    reference_id=arrays["reference"],
                    row_mask=arrays["mask"],
                    channel_id=np.array([SMOKE_CHANNEL]),
                    squint_records=[summaries["squint_measured"], summaries["squint_cassbeam"]],
                )
            )
            print(
                "channel32",
                summaries["n_rows"],
                "rr residual",
                summaries["diagonal"]["rr"]["residual_power"],
                flush=True,
            )
        if arguments.stage in {"frequency", "all"}:
            records_meas = []
            records_pred = []
            by_channel = []
            for channel in FREQUENCY_CHANNELS:
                print("frequency channel", channel, flush=True)
                observation, chi, packed, intensity, voltage, rr_ok, ll_ok = _load_holoraster_channel(
                    diag,
                    tables,
                    arguments.measurement_set,
                    positions,
                    channel=channel,
                )
                summaries, _arrays, chan_finite = _compare_channel(
                    catalog,
                    observation,
                    chi,
                    packed,
                    intensity,
                    voltage,
                    rr_ok,
                    ll_ok,
                    residual,
                )
                finite = finite or chan_finite
                records_meas.append(summaries["squint_measured"])
                records_pred.append(summaries["squint_cassbeam"])
                by_channel.append(
                    {
                        "channel": int(channel),
                        "frequency_hz": summaries["frequency_hz"],
                        "diagonal": summaries["diagonal"],
                        "experimental_full_jones": summaries["experimental_full_jones"],
                        "squint_measured": summaries["squint_measured"],
                        "squint_cassbeam": summaries["squint_cassbeam"],
                    }
                )
            payload["frequency"] = {
                "channels": by_channel,
                "squint_measured": records_meas,
                "squint_cassbeam": records_pred,
            }
            write_json(_jsonable(payload["frequency"]), output_dir / "frequency_summaries.json")
            saved = output_dir / "channel32_comparison.npz"
            if saved.is_file():
                loaded = np.load(saved)
                plots.extend(
                    write_holoraster_comparison_plots(
                        output_dir / "plots",
                        measured=loaded["measured"],
                        predicted_diag=loaded["predicted_diag"],
                        predicted_full=loaded["predicted_full"],
                        weight=loaded["weight"],
                        offset_lm_rad=loaded["offset"],
                        moving_id=loaded["moving"],
                        reference_id=loaded["reference"],
                        row_mask=loaded["mask"],
                        channel_id=np.array([SMOKE_CHANNEL]),
                        squint_records=records_meas,
                    )
                )
        if arguments.stage == "plots" and saved_npz.is_file():
            loaded = np.load(saved_npz)
            freq_path = output_dir / "frequency_summaries.json"
            records = []
            if freq_path.is_file():
                freq = json.loads(freq_path.read_text())
                payload["frequency"] = freq
                records = list(freq.get("squint_measured") or [])
            plots.extend(
                write_holoraster_comparison_plots(
                    output_dir / "plots",
                    measured=loaded["measured"],
                    predicted_diag=loaded["predicted_diag"],
                    predicted_full=loaded["predicted_full"],
                    weight=loaded["weight"],
                    offset_lm_rad=loaded["offset"],
                    moving_id=loaded["moving"],
                    reference_id=loaded["reference"],
                    row_mask=loaded["mask"],
                    channel_id=np.array([SMOKE_CHANNEL]),
                    squint_records=records,
                )
            )
            finite = True
        if saved_npz.is_file() and arguments.stage in {"squint", "all", "plots", "channel32"}:
            loaded = np.load(saved_npz)
            freq = float(
                ((payload.get("channel32") or {}).get("frequency_hz"))
                or THOL0001_SPW4_CHANNEL_32_HZ
            )
            rescored = rescore_squint_from_comparison_arrays(
                loaded["measured"],
                loaded["predicted_diag"],
                loaded["weight"],
                loaded["offset"],
                frequency_hz=freq,
            )
            payload["squint_publication"] = rescored
            write_json(_jsonable(rescored), output_dir / "squint_mainlobe_20pct.json")
            plots.extend(
                write_squint_publication_plot(
                    output_dir / "plots",
                    measured=rescored["measured"],
                    cassbeam=rescored["cassbeam"],
                )
            )
            print(
                "squint measured",
                rescored["measured"]["separation_arcmin"],
                "cassbeam",
                rescored["cassbeam"]["separation_arcmin"],
                "memo",
                rescored["measured"].get("memo195_separation_arcmin"),
                flush=True,
            )
        channel32 = payload.get("channel32") or {}
        regions = (channel32.get("regions") or {}).get("hand_residual_power")
        support = classify_diagonal_region_support(regions) if regions else {}
        payload["diagonal_support"] = support
        payload["plots"] = plots
        gate = classify_holoraster_comparison_report(
            software_ok=software_ok,
            predictions_finite=finite or bool(payload.get("squint_publication")),
            plots_written=bool(plots) or arguments.stage == "squint",
            diagonal_support=support,
        )
        payload["decision_gate"] = gate
        payload.update(gate)
        _write_payload(output_dir, payload)
        print(json.dumps(_jsonable({"decision": gate["decision"], "status": gate["status"]})), flush=True)
        return 0 if not gate["process_failure"] else 1
    except Exception as error:
        payload["error"] = f"{type(error).__name__}: {error}"
        payload["traceback"] = traceback.format_exc()
        payload["decision_gate"] = classify_holoraster_comparison_report(
            software_ok=False,
            predictions_finite=False,
            plots_written=False,
        )
        _write_payload(output_dir, payload)
        print(payload["traceback"], flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
