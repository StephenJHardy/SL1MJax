"""First SPW-4 CASSBEAM diagonal-correction ladder on HOLORASTER field 10.

Fits on training rows, ranks with paired spatial and mover ΔL, stops at the
first rejection, then refits the accepted prefix on all development rows.
SPW 5 stays sealed. C147-* fields are unused.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import traceback
from pathlib import Path

import numpy as np

from sl1mjax.beam_validation_statistics import classify_diagonal_region_support
from sl1mjax.cassbeam_highres import DEFAULT_HIGHRES_ROOT, HighresCassbeamCatalog
from sl1mjax.holography import THOL0001_SPW4_CHANNEL_32_HZ
from sl1mjax.holography_alignment import holoraster_pair_masks
from sl1mjax.holography_calibration import hash_path, write_json
from sl1mjax.holography_beam_prior import (
    compare_holoraster_stages,
    evaluate_holoraster_cassbeam,
    squeeze_sample_jones,
)
from sl1mjax.holography_cassbeam_correction import (
    IDENTITY_CORRECTION,
    CorrectionSamples,
    catalog_diagonal_feed_lookup,
    correction_path_stages,
    identity_predictions,
    ladder_result_to_dict,
    predict_from_state,
    refuse_spw5,
    region_masks_from_voltage,
    require_aligned_frozen_prediction,
    require_identity_matches_baseline,
    require_positive_envelope,
    run_first_ladder,
)
from sl1mjax.holography_cassbeam_holoraster_report import (
    CASSBEAM_DIAGONAL_CORRECTION,
    region_copolar_summaries,
)
from sl1mjax.holography_diagonal_correction import (
    HOLORASTER_FIELD_ID,
    PAIRED_DELTA_BOOTSTRAP,
    frozen_protocol_payload,
    protocol_as_mapping,
    refuse_c147_training,
    spw4_correction_holdouts,
)
from sl1mjax.holography_full_jones import moving_reference_row_geometry
from sl1mjax.holography_highres_cassbeam import artifact_checksum_report, run_software_gates

COMPARISON_DIR = Path(
    "/media/stephen/astro/vla/extracted/commissioning/validation/scientific/"
    "holoraster_cassbeam_comparison"
)
PRODUCT_DIR = Path(
    "/media/stephen/astro/vla/extracted/commissioning/validation/scientific/"
    "holoraster_cassbeam_correction"
)
UNBLOCK = Path(
    "/media/stephen/astro/vla/extracted/commissioning/validation/scientific/full_jones_unblock"
)
SCIENTIFIC_MS = Path(
    "/media/stephen/astro/vla/extracted/commissioning/work/THOL0001.lowerC.spw45.scientific.ms"
)
FROZEN_NAMES = (
    "holoraster_cassbeam_comparison",
    "one_axis_visibility_holdouts.json",
    "c147_offset_ring",
)
BOOTSTRAP_SEED = 0
CHANNEL = 32


def _comparison():
    path = Path(__file__).with_name("run_thol0001_holoraster_cassbeam_comparison.py")
    spec = importlib.util.spec_from_file_location("thol0001_holoraster_comparison", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    return _sha256_bytes(array.tobytes() + np.asarray(array.shape).tobytes() + array.dtype.str.encode())


def _mask_record(name: str, mask: np.ndarray) -> dict[str, object]:
    flags = np.asarray(mask, dtype=bool).reshape(-1)
    return {"name": name, "n": int(np.sum(flags)), "sha256": _sha256_array(flags)}


def _readme(payload: dict) -> str:
    accepted = payload.get("accepted") or {}
    stage = payload.get("stage") or "identity"
    lines = [
        "# THOL0001 CASSBEAM diagonal correction",
        "",
        f"Artifact: `{CASSBEAM_DIAGONAL_CORRECTION}`.",
        f"Stage: `{stage}`.",
        f"Identity matches baseline: **{payload.get('identity_matches_baseline')}**.",
        f"First divergent stage: `{payload.get('first_divergent_stage')}`.",
        f"Accepted terms: `{accepted.get('accepted_terms')}`.",
        f"Stopped at: `{payload.get('stopped_at')}`.",
        "",
    ]
    if stage == "identity":
        lines.append(
            "Identity smoke only. The ladder was not started. SPW 5 stayed sealed."
        )
    else:
        lines.append(
            "The first ladder was fit on HOLORASTER field 10, SPW 4, channel 32. "
            "Selection used paired spatial and mover ΔL. SPW 5 stayed sealed."
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--measurement-set", type=Path, default=SCIENTIFIC_MS)
    parser.add_argument("--product-dir", type=Path, default=UNBLOCK)
    parser.add_argument("--comparison-dir", type=Path, default=COMPARISON_DIR)
    parser.add_argument("--output-dir", type=Path, default=PRODUCT_DIR)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_HIGHRES_ROOT)
    parser.add_argument("--spectral-window", type=int, default=4)
    parser.add_argument("--channel", type=int, default=CHANNEL)
    parser.add_argument("--n-boot", type=int, default=PAIRED_DELTA_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument(
        "--stage",
        choices=("identity", "ladder", "all"),
        default="identity",
        help="identity smoke only, or continue to the first ladder after identity agrees",
    )
    arguments = parser.parse_args()
    if int(arguments.spectral_window) != 4 or int(arguments.channel) != CHANNEL:
        raise ValueError("correction runner is defined for SPW 4 channel 32; SPW 5 stays sealed")
    refuse_spw5(spectral_window_id=int(arguments.spectral_window), opened=False)
    output_dir = arguments.output_dir.resolve()
    if output_dir.name in FROZEN_NAMES or output_dir == arguments.comparison_dir.resolve():
        raise RuntimeError(f"refusing to write into a frozen product path: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    cmp = _comparison()
    payload: dict[str, object] = {
        "artifact": CASSBEAM_DIAGONAL_CORRECTION,
        "source_revision": cmp._source_revision(),
        "measurement_set": str(arguments.measurement_set),
        "spw5_closed": True,
        "model_selected": False,
        "full_jones_frozen": False,
        "production_factory_modified": False,
        "protocol": protocol_as_mapping(frozen_protocol_payload()),
        "grids": {},
        "bootstrap": {"n_boot": int(arguments.n_boot), "seed": int(arguments.seed)},
        "stage": arguments.stage,
        "ladder_started": False,
        "identity_matches_baseline": False,
        "first_divergent_stage": None,
    }
    try:
        catalog = HighresCassbeamCatalog(arguments.artifact_root)
        plane = catalog.plane(THOL0001_SPW4_CHANNEL_32_HZ)
        software = run_software_gates(plane)
        payload["software_gates"] = software
        payload["artifact_checksum"] = artifact_checksum_report(
            catalog, THOL0001_SPW4_CHANNEL_32_HZ
        )
        if not bool(software.get("passed", False)):
            raise RuntimeError("high-resolution CASSBEAM software gates failed")
        residual = cmp._residual_jones(arguments.product_dir)
        residual_payload = np.stack(
            [np.asarray(residual[key], dtype=np.complex128) for key in sorted(residual)]
        )
        payload["hashes"] = {
            "measurement_set": hash_path(arguments.measurement_set),
            "residual_jones": _sha256_array(residual_payload),
            "residual_antenna_ids": [int(key) for key in sorted(residual)],
        }
        diag = cmp._diag()
        from sl1mjax.holography_ms import _read_antennas, _tables

        tables = _tables()
        _ids, names, positions = _read_antennas(tables, arguments.measurement_set)
        names = tuple(names)
        positions = np.asarray(positions, dtype=np.float64)
        observation, chi, packed, intensity, voltage, rr_ok, ll_ok = cmp._load_holoraster_channel(
            diag,
            tables,
            arguments.measurement_set,
            positions,
            channel=int(arguments.channel),
        )
        frequencies = np.asarray(observation.block.frequency_hz, dtype=np.float64).reshape(-1)
        if frequencies.size != 1:
            raise ValueError("correction samples must contain exactly one channel")
        if not np.isclose(frequencies[0], THOL0001_SPW4_CHANNEL_32_HZ, rtol=0.0, atol=0.5e6):
            raise ValueError("correction samples must be THOL0001 SPW 4 channel 32")
        refuse_c147_training(observation.block.field_id)
        if np.any(np.asarray(observation.block.field_id) != HOLORASTER_FIELD_ID):
            raise ValueError("correction samples must be HOLORASTER field 10")
        geometry = moving_reference_row_geometry(observation, require_all_channels=False)
        pair = holoraster_pair_masks(observation)
        usable = np.asarray(geometry["usable"], dtype=bool) & np.asarray(
            pair["moving_reference"], dtype=bool
        )
        holdouts = spw4_correction_holdouts(observation, names)
        development = holdouts.train | holdouts.spatial_holdout | holdouts.mover_holdout
        keep = usable & development
        if not (
            bool(np.any(holdouts.train & keep))
            and bool(np.any(holdouts.spatial_holdout & keep))
            and bool(np.any(holdouts.mover_holdout & keep))
        ):
            raise ValueError("train, spatial holdout, and mover holdout must be non-empty")
        # Same usable moving–reference rows and order as the frozen comparison.
        fields = cmp._row_fields(geometry, pair, chi, np.flatnonzero(usable))
        print("usable_rows", int(np.sum(usable)), "development_rows", int(np.sum(keep)), flush=True)
        source = np.asarray(observation.source_coherency_visibility, dtype=np.complex128)
        if source.shape == (2, 2):
            source_rows = source
        elif source.ndim == 3:
            source_rows = source[fields["rows"], None, :, :]
        else:
            source_rows = source[fields["rows"]]
        print("predicting frozen HOLORASTER baseline", flush=True)
        convention = cmp.locked_convention()
        cmp.refuse_convention_search([convention])
        comparison_stages = evaluate_holoraster_cassbeam(
            catalog=catalog,
            frequencies_hz=frequencies,
            convention=convention,
            offset_lm_rad=fields["offset"],
            chi_moving=fields["chi_m"],
            chi_reference=fields["chi_r"],
            moving_id=fields["moving"],
            reference_id=fields["reference"],
            moving_is_p=fields["moving_is_p"],
            residual_jones=residual,
            source=source_rows,
            off_diagonal=False,
        )
        pred_diag = squeeze_sample_jones(comparison_stages.visibility)
        measured = np.asarray(packed, dtype=np.complex128)
        if measured.ndim == 4:
            measured = measured[fields["rows"], 0]
        else:
            measured = measured[fields["rows"]]
        weight = cmp._hand_weight(observation, usable)
        if weight.ndim == 4:
            weight = weight[:, 0]
        comparison_npz = arguments.comparison_dir / "channel32_comparison.npz"
        existing_match = None
        if comparison_npz.is_file():
            print("comparing to", comparison_npz, flush=True)
            loaded = np.load(comparison_npz)
            payload["hashes"]["channel32_comparison_npz"] = hash_path(comparison_npz)
            frozen_diag = np.asarray(loaded["predicted_diag"], dtype=np.complex128)
            if frozen_diag.ndim == 4:
                frozen_diag = frozen_diag[:, 0]
            require_aligned_frozen_prediction(
                fields["offset"],
                fields["moving"],
                fields["reference"],
                pred_diag,
                loaded["offset"],
                loaded["moving"],
                loaded["reference"],
                frozen_diag,
            )
            existing_match = True
        lookup = catalog_diagonal_feed_lookup(
            catalog, float(frequencies[0]), convention=convention
        )
        print("identity smoke on usable rows", int(fields["offset"].shape[0]), flush=True)
        correction_stages = correction_path_stages(
            fields["offset"],
            lookup,
            IDENTITY_CORRECTION,
            residual_jones=residual,
            moving_id=fields["moving"],
            reference_id=fields["reference"],
            moving_is_p=fields["moving_is_p"],
            chi_moving=fields["chi_m"],
            chi_reference=fields["chi_r"],
            source=source_rows,
        )
        stage_report = compare_holoraster_stages(comparison_stages, correction_stages)
        payload["stage_compare"] = stage_report
        payload["first_divergent_stage"] = stage_report["first_divergent_stage"]
        payload["identity_matches_existing_holoraster"] = existing_match
        print(json.dumps(cmp._jsonable(stage_report)), flush=True)
        require_identity_matches_baseline(correction_stages["visibility"], pred_diag)
        payload["identity_matches_baseline"] = True
        on_usable = keep[usable]
        fields = cmp._row_fields(geometry, pair, chi, np.flatnonzero(keep))
        pred_diag = pred_diag[on_usable]
        measured = measured[on_usable]
        weight = weight[on_usable]
        if source_rows.shape == (2, 2):
            source = source_rows
        else:
            source = source_rows[on_usable]
        voltage_rows = np.asarray(voltage, dtype=np.float64)[fields["rows"]]
        regions = region_masks_from_voltage(voltage_rows)
        samples = CorrectionSamples(
            offset_lm_rad=fields["offset"],
            measured=measured,
            baseline=pred_diag,
            weight=weight,
            source=source,
            moving_is_p=fields["moving_is_p"],
            moving_id=fields["moving"],
            reference_id=fields["reference"],
            residual_jones=residual,
            parallactic_angle_rad=fields["chi_m"],
            reference_parallactic_angle_rad=fields["chi_r"],
            antenna_names=names,
            train=holdouts.train[keep],
            spatial_holdout=holdouts.spatial_holdout[keep],
            mover_holdout=holdouts.mover_holdout[keep],
            main_lobe=regions["main_lobe"],
            mid=regions["mid"],
            outer=regions["outer_diagnostic"],
            field_id=np.asarray(observation.block.field_id, dtype=np.int32)[keep],
            frequency_hz=float(frequencies[0]),
            spectral_window_id=4,
        )
        require_positive_envelope(samples.offset_lm_rad, IDENTITY_CORRECTION)
        n_train = int(np.sum(samples.train))
        print("identity gate on development rows", n_train, flush=True)
        identity = predict_from_state(samples, lookup, IDENTITY_CORRECTION)
        require_identity_matches_baseline(identity_predictions(samples.baseline), samples.baseline)
        require_identity_matches_baseline(identity, samples.baseline)
        payload["masks"] = {
            "train": _mask_record("train", samples.train),
            "spatial_holdout": _mask_record("spatial_holdout", samples.spatial_holdout),
            "mover_holdout": _mask_record("mover_holdout", samples.mover_holdout),
            "main_lobe": _mask_record("main_lobe", samples.main_lobe),
            "development": _mask_record("development", samples.development_mask()),
        }
        payload["hashes"]["baseline"] = _sha256_array(samples.baseline)
        payload["hashes"]["measured"] = _sha256_array(samples.measured)
        payload["n_samples"] = int(samples.train.size)
        payload["n_train"] = n_train
        payload["n_spatial_holdout"] = int(np.sum(samples.spatial_holdout))
        payload["n_mover_holdout"] = int(np.sum(samples.mover_holdout))
        np.savez_compressed(
            output_dir / "correction_masks.npz",
            train=samples.train,
            spatial_holdout=samples.spatial_holdout,
            mover_holdout=samples.mover_holdout,
            main_lobe=samples.main_lobe,
            mid=samples.mid,
            outer=samples.outer,
            offset=samples.offset_lm_rad,
            moving=samples.moving_id,
            reference=samples.reference_id,
        )
        write_json(cmp._jsonable(payload), output_dir / "identity_smoke.json")
        if arguments.stage == "identity":
            write_json(cmp._jsonable(payload), output_dir / f"{CASSBEAM_DIAGONAL_CORRECTION}.json")
            (output_dir / "README.md").write_text(_readme(payload))
            print(
                json.dumps(
                    cmp._jsonable(
                        {
                            "stage": "identity",
                            "identity_matches_baseline": True,
                            "first_divergent_stage": payload["first_divergent_stage"],
                            "n_train": n_train,
                            "n_usable": int(np.sum(usable)),
                            "ladder_started": False,
                        }
                    )
                ),
                flush=True,
            )
            return 0
        print("running first ladder", flush=True)
        payload["ladder_started"] = True
        result = run_first_ladder(
            samples,
            lookup,
            n_boot=int(arguments.n_boot),
            seed=int(arguments.seed),
            refit_development=True,
        )
        recorded = ladder_result_to_dict(result)
        accepted_vis = predict_from_state(samples, lookup, result.accepted)
        region_scores = region_copolar_summaries(
            samples.measured,
            accepted_vis,
            samples.weight,
            np.asarray(intensity, dtype=np.float64).reshape(-1)[fields["rows"]],
            np.asarray(rr_ok, dtype=bool)[fields["rows"]],
            np.asarray(ll_ok, dtype=bool)[fields["rows"]],
            np.ones(samples.train.size, dtype=bool),
            voltage=voltage_rows,
        )
        support = classify_diagonal_region_support(region_scores["hand_residual_power"])
        main = support["regions"]["main_lobe"]
        payload.update(recorded)
        payload["grids"] = {
            key: recorded["hyperparameters"][key]
            for key in (
                "squint_scale_grid",
                "width_scale_grid",
                "pointing_arcmin_grid",
                "sidelobe_radius_grid",
                "sidelobe_amplitude_grid",
            )
        }
        payload["diagonal_support"] = support
        payload["main_lobe_matches_class"] = bool(main["matches_class"])
        payload["main_lobe_class"] = str(support["main_lobe"])
        write_json(cmp._jsonable(payload), output_dir / f"{CASSBEAM_DIAGONAL_CORRECTION}.json")
        (output_dir / "README.md").write_text(_readme(payload))
        print(
            json.dumps(
                cmp._jsonable(
                    {
                        "accepted_terms": recorded["accepted"]["accepted_terms"],
                        "stopped_at": recorded["stopped_at"],
                        "identity_matches_baseline": True,
                        "main_lobe_matches_class": payload["main_lobe_matches_class"],
                    }
                )
            ),
            flush=True,
        )
        return 0
    except Exception as error:
        payload["error"] = f"{type(error).__name__}: {error}"
        payload["traceback"] = traceback.format_exc()
        write_json(cmp._jsonable(payload), output_dir / f"{CASSBEAM_DIAGONAL_CORRECTION}.json")
        (output_dir / "README.md").write_text(_readme(payload))
        print(payload["traceback"], flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
