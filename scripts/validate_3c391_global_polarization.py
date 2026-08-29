#!/usr/bin/env python3
"""Validation-grade global q,u,v test on frozen Stokes I.

Predicts the accepted mosaic M_I at the exact target rows, fits null versus
global q,u,v by complex visibility regression, reports deterministic
baseline/time/channel partitions, processes all seven 3C391 pointings,
aligns dirty Q/U/V on one sky grid, holds out one pointing, and splits
3C286 so Kcross/Xf are solved and evaluated on different cohorts.

This is not RM, self-cal, or per-pixel polarisation. Same-data 3C286 is
labelled apply-back; the scan/time split is the held-out calibrator floor.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from sl1mjax.beam import VLABeamCatalog, VLAPrimaryBeam
from sl1mjax.calibration import (
    apply_calibration,
    identity_solution,
    import_casa_polarization_solution,
    load_casa_calibration_golden,
)
from sl1mjax.composite import MosaicSkyComponent, predict_mosaic_composite
from sl1mjax.data.averaging import average_frequency_bins, average_time_bins
from sl1mjax.data.canonical import VisibilityBlock
from sl1mjax.data.ms import extract_measurement_set
from sl1mjax.direct_operator import DirectDFTConfig
from sl1mjax.polarization_diagnostics import (
    calibrator_polarization_floor,
    deterministic_calibrator_cohort_split,
    dirty_mosaic_stokes_images,
    dirty_stokes_images,
    evaluate_global_fractional_polarization,
    fit_global_fractional_polarization,
    fit_global_fractional_polarization_blocks,
    fit_partitioned_global_polarization,
    global_fractional_polarization_as_dict,
    map_image_agreement,
    outer_image_robust_scale,
    polarization_floor_as_dict,
)
from sl1mjax.polarization_inference import solve_cross_hand_delay, solve_rl_phase
from sl1mjax.sky import RegularGrid

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POL = ROOT / "tests" / "fixtures" / "3c391_polarization_golden.npz"
DEFAULT_KBG = ROOT / "tests" / "fixtures" / "3c391_calibration_golden.npz"
DEFAULT_MS = Path(
    "/home/stephen/checkouts/SL1MJax-frozen-20260824/data/3c391_ctm_mosaic_10s_spw0.ms"
)
DEFAULT_SKY_ROOT = Path("/home/stephen/checkouts/SL1MJax")
DEFAULT_PROTOCOL = DEFAULT_SKY_ROOT / "outputs/3c391_composite_catalogue_stage3/protocol.json"
DEFAULT_CHECKPOINT = (
    DEFAULT_SKY_ROOT / "outputs/3c391_recovery_policy_fit_zero/sealed_active_only.npz"
)
DEFAULT_OUTPUT = ROOT / "outputs" / "3c391_global_polarization_validation"
SCIENCE_FIELDS = (2, 3, 4, 5, 6, 7, 8)


def _corrected_from_casa(case):
    return replace(
        case.block,
        visibility=case.corrected_visibility,
        flag=case.block.flag | case.post_apply_flag,
    )


def _polarization_terms_only(solution):
    """Keep Kcross/D/X/P. Identity G/K/B for already-corrected target DATA."""

    identity = identity_solution(
        antenna_count=solution.antenna_count,
        correlations=solution.correlations,
        frequency_hz=solution.bandpass_frequency_hz,
        time_s=solution.gain_time_s,
        reference_antenna=solution.reference_antenna,
    )
    return replace(
        identity,
        receptors=solution.receptors,
        reference_frequency_hz=solution.reference_frequency_hz,
        antenna_position_m=solution.antenna_position_m,
        cross_hand_delay_s=solution.cross_hand_delay_s,
        cross_hand_delay_valid=solution.cross_hand_delay_valid,
        leakage=solution.leakage,
        leakage_frequency_hz=solution.leakage_frequency_hz,
        leakage_valid=solution.leakage_valid,
        leakage_application=solution.leakage_application,
        rl_phase=solution.rl_phase,
        rl_phase_frequency_hz=solution.rl_phase_frequency_hz,
        rl_phase_valid=solution.rl_phase_valid,
        apply_parallactic_angle=True,
        provenance={
            **solution.provenance,
            "gkb": "identity_on_corrected_data",
            "evidence_grade": False,
        },
    )


def _field_label(field_id: int) -> str:
    if 2 <= field_id <= 8:
        return f"C{field_id - 1}"
    return f"field_{field_id}"


def _resolve_protocol_paths(protocol: dict[str, Any], *roots: Path) -> dict[str, Any]:
    resolved = dict(protocol)
    frozen = resolved.get("frozen_directory")
    if not frozen:
        return resolved
    path = Path(frozen)
    candidates = (path, *(root / path for root in roots if root is not None))
    for candidate in candidates:
        if (candidate / "summary.json").exists():
            resolved["frozen_directory"] = str(candidate)
            return resolved
    return resolved


def _load_sky_components(
    protocol_path: Path,
    checkpoint: Path,
    mosaic_phase_centre_rad: tuple[float, float],
    sky_roots: tuple[Path, ...],
) -> tuple[MosaicSkyComponent, ...]:
    scripts_directory = str(Path(__file__).resolve().parent)
    if scripts_directory not in sys.path:
        sys.path.insert(0, scripts_directory)
    from compare_3c391_composite_existing_flags import _components_from_checkpoint

    protocol = _resolve_protocol_paths(
        json.loads(protocol_path.read_text(encoding="utf-8")),
        *sky_roots,
        protocol_path.resolve().parent.parent.parent,
        checkpoint.resolve().parent.parent.parent,
    )
    frozen = Path(protocol["frozen_directory"])
    if not (frozen / "summary.json").exists():
        raise FileNotFoundError(
            f"frozen Stokes-I directory is missing summary.json: {frozen}"
        )
    return _components_from_checkpoint(checkpoint, protocol, mosaic_phase_centre_rad)


def _average_target(
    block: VisibilityBlock,
    *,
    frequency_bins: int,
    time_bin_seconds: float,
) -> VisibilityBlock:
    averaged = block
    if frequency_bins < averaged.frequency_hz.size:
        averaged = average_frequency_bins(averaged, bin_count=frequency_bins)
    if time_bin_seconds > 0:
        averaged = average_time_bins(averaged, bin_seconds=time_bin_seconds)
    return averaged


def _fit_payload(fitted) -> dict[str, Any]:
    return global_fractional_polarization_as_dict(fitted)


def _image_peaks(images) -> dict[str, float]:
    return {
        "peak_i": images.peak_i,
        "peak_q": images.peak_q,
        "peak_u": images.peak_u,
        "peak_v": images.peak_v,
        "abs_peak_q": float(np.max(np.abs(images.stokes_q))),
        "abs_peak_u": float(np.max(np.abs(images.stokes_u))),
        "abs_peak_v": float(np.max(np.abs(images.stokes_v))),
        "outer_robust_q": outer_image_robust_scale(images.stokes_q),
        "outer_robust_u": outer_image_robust_scale(images.stokes_u),
        "outer_robust_v": outer_image_robust_scale(images.stokes_v),
    }


def _three_c286_holdout(flux, imported):
    split = deterministic_calibrator_cohort_split(flux.block)
    kcross = solve_cross_hand_delay(flux.block, imported, split=split)
    with_leakage = replace(
        kcross.solution,
        leakage=imported.leakage,
        leakage_frequency_hz=imported.leakage_frequency_hz,
        leakage_valid=imported.leakage_valid,
        leakage_application=imported.leakage_application,
    )
    angle = solve_rl_phase(flux.block, with_leakage, split=split)
    corrected = apply_calibration(flux.block, angle.solution, extrapolate=True)
    train_floor = calibrator_polarization_floor(
        corrected,
        independence="apply_back",
        label="flux_angle_train",
        sample_mask=split.train,
    )
    holdout_floor = calibrator_polarization_floor(
        corrected,
        independence="held_out_calibrator",
        label="flux_angle_holdout",
        sample_mask=split.holdout,
    )
    holdout_fit = fit_global_fractional_polarization(
        corrected, sample_mask=split.holdout
    )
    return {
        "split": {
            "strategy": split.strategy,
            "train_samples": int(np.count_nonzero(split.train)),
            "holdout_samples": int(np.count_nonzero(split.holdout)),
        },
        "train_floor": polarization_floor_as_dict(train_floor),
        "holdout_floor": polarization_floor_as_dict(holdout_floor),
        "holdout_global_quv": _fit_payload(holdout_fit),
        "kcross_holdout_rms": float(kcross.holdout_rms),
        "xf_holdout_rms": float(angle.holdout_rms),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--polarization-golden", type=Path, default=DEFAULT_POL)
    parser.add_argument("--calibration-golden", type=Path, default=DEFAULT_KBG)
    parser.add_argument("--measurement-set", type=Path, default=None)
    parser.add_argument(
        "--no-measurement-set",
        action="store_true",
        help="Skip the science mosaic even if the default MeasurementSet exists",
    )
    parser.add_argument("--sky-protocol", type=Path, default=None)
    parser.add_argument("--sky-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--sky-root",
        type=Path,
        action="append",
        default=None,
        help="Directory that contains outputs/ for relative frozen_directory paths",
    )
    parser.add_argument("--science-fields", default="2,3,4,5,6,7,8")
    parser.add_argument("--holdout-field", type=int, default=8)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--frequency-bins", type=int, default=8)
    parser.add_argument("--time-bin-seconds", type=float, default=60.0)
    parser.add_argument("--row-stride", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=17)
    parser.add_argument("--pixel-arcsec", type=float, default=2.0)
    parser.add_argument("--mosaic-image-size", type=int, default=97)
    parser.add_argument("--mosaic-pixel-arcsec", type=float, default=10.0)
    parser.add_argument("--target-image-size", type=int, default=65)
    parser.add_argument("--target-pixel-arcsec", type=float, default=8.0)
    parser.add_argument("--skip-images", action="store_true")
    parser.add_argument("--precision", choices=("float32", "float64"), default="float32")
    arguments = parser.parse_args(argv)

    flux = load_casa_calibration_golden(arguments.polarization_golden, label="flux_angle")
    leakage = load_casa_calibration_golden(
        arguments.polarization_golden, label="leakage_calibrator"
    )
    casa_flux = _corrected_from_casa(flux)
    casa_leakage = _corrected_from_casa(leakage)
    imported = import_casa_polarization_solution(
        arguments.polarization_golden,
        arguments.calibration_golden,
        label="flux_angle",
    )
    jax_flux = apply_calibration(flux.block, imported, extrapolate=True)

    report: dict[str, Any] = {
        "evidence_grade": False,
        "milestone": "complex_global_regression_and_cross_pointing_validation",
        "apply": "casa_parallel_preserving",
        "next_step_not_started": [
            "spatial_qu_activation",
            "self_cal",
            "rm",
            "per_pixel_polarization",
        ],
        "floors": {
            "casa_3c286_apply_back": polarization_floor_as_dict(
                calibrator_polarization_floor(
                    casa_flux, independence="apply_back", label="flux_angle"
                )
            ),
            "jax_3c286_apply_back": polarization_floor_as_dict(
                calibrator_polarization_floor(
                    jax_flux, independence="apply_back", label="flux_angle_jax"
                )
            ),
            "casa_3c84_in_sample": polarization_floor_as_dict(
                calibrator_polarization_floor(
                    casa_leakage,
                    independence="in_sample",
                    label="leakage_calibrator",
                )
            ),
        },
        "global_quv": {
            "casa_3c286_apply_back": _fit_payload(
                fit_global_fractional_polarization(casa_flux)
            ),
            "jax_3c286_apply_back": _fit_payload(
                fit_global_fractional_polarization(jax_flux)
            ),
        },
        "three_c286_holdout": _three_c286_holdout(flux, imported),
    }

    arguments.output.mkdir(parents=True, exist_ok=True)
    if not arguments.skip_images:
        grid = RegularGrid(
            arguments.image_size, np.deg2rad(arguments.pixel_arcsec / 3600)
        )
        images = dirty_stokes_images(jax_flux, grid)
        np.savez(
            arguments.output / "3c286_dirty_stokes.npz",
            stokes_i=images.stokes_i,
            stokes_q=images.stokes_q,
            stokes_u=images.stokes_u,
            stokes_v=images.stokes_v,
            psf=images.psf,
        )
        report["dirty_stokes_3c286"] = _image_peaks(images)

    measurement_set = None
    if not arguments.no_measurement_set:
        measurement_set = arguments.measurement_set
        if measurement_set is None and DEFAULT_MS.is_dir():
            measurement_set = DEFAULT_MS
        if measurement_set is not None and not measurement_set.exists():
            raise FileNotFoundError(f"measurement set does not exist: {measurement_set}")
    protocol_path = arguments.sky_protocol
    checkpoint_path = arguments.sky_checkpoint
    if protocol_path is None and DEFAULT_PROTOCOL.is_file():
        protocol_path = DEFAULT_PROTOCOL
    if checkpoint_path is None and DEFAULT_CHECKPOINT.is_file():
        checkpoint_path = DEFAULT_CHECKPOINT

    if measurement_set is not None:
        science_fields = tuple(
            int(value) for value in arguments.science_fields.split(",") if value
        )
        if arguments.holdout_field not in science_fields:
            raise ValueError("holdout field must be one of the science fields")
        print(f"extracting fields {science_fields} from {measurement_set}", flush=True)
        dataset = extract_measurement_set(
            measurement_set,
            data_column="CORRECTED_DATA",
            fields=science_fields,
            row_stride=arguments.row_stride,
        )
        if len(dataset.blocks) != len(science_fields):
            raise ValueError(
                f"expected {len(science_fields)} science blocks, got {len(dataset.blocks)}"
            )
        pol_only = _polarization_terms_only(imported)
        mosaic_phase_centre = dataset.blocks[0].phase_centre_rad
        sky_roots = tuple(arguments.sky_root or ())
        if DEFAULT_SKY_ROOT.is_dir():
            sky_roots = (*sky_roots, DEFAULT_SKY_ROOT)
        sky_roots = (*sky_roots, ROOT, Path.cwd())
        components = None
        if protocol_path is not None and checkpoint_path is not None:
            print("loading frozen Stokes-I mosaic", flush=True)
            components = _load_sky_components(
                protocol_path,
                checkpoint_path,
                mosaic_phase_centre,
                sky_roots,
            )
        protocol_payload = (
            json.loads(protocol_path.read_text(encoding="utf-8"))
            if protocol_path is not None
            else {}
        )
        beam = VLAPrimaryBeam(
            kind="airy",
            catalog=replace(
                VLABeamCatalog(),
                airy_max_radius_rad_at_1ghz=np.deg2rad(
                    float(protocol_payload.get("airy_max_radius_deg_at_1ghz", 4.0))
                ),
            ),
        )
        direct = DirectDFTConfig(precision=arguments.precision)
        labels: list[str] = []
        field_ids: list[int] = []
        averaged_blocks: list[VisibilityBlock] = []
        for block in dataset.blocks:
            field_id = int(np.unique(block.field_id)[0]) if block.field_id is not None else -1
            label = _field_label(field_id)
            print(f"averaging {label}", flush=True)
            averaged_blocks.append(
                _average_target(
                    block,
                    frequency_bins=arguments.frequency_bins,
                    time_bin_seconds=arguments.time_bin_seconds,
                )
            )
            labels.append(label)
            field_ids.append(field_id)
        if components is not None:
            print(
                f"predicting frozen M_I at {len(averaged_blocks)} pointing frames",
                flush=True,
            )
            predictions = predict_mosaic_composite(
                tuple(averaged_blocks),
                components,
                mosaic_phase_centre,
                primary_beam=beam,
                config=direct,
            )
            averaged_blocks = [
                replace(block, model_visibility=prediction)
                for block, prediction in zip(averaged_blocks, predictions, strict=True)
            ]
        print("applying polarisation-only Jones to CORRECTED_DATA", flush=True)
        prepared = []
        for block in averaged_blocks:
            corrected = apply_calibration(block, pol_only, extrapolate=True)
            if block.model_visibility is not None:
                corrected = replace(corrected, model_visibility=block.model_visibility)
            prepared.append(corrected)

        report["corrected_data_state"] = {
            "data_column": "CORRECTED_DATA",
            "measurement_set": str(measurement_set),
            "pre_polarization_state": (
                "CASA G/K/B already present in CORRECTED_DATA; DATA was not reread"
            ),
            "polarization_apply": (
                "identity G/K/B plus imported Kcross/D/X/P, "
                "leakage_application=casa_parallel_preserving"
            ),
            "frequency_bins": arguments.frequency_bins,
            "time_bin_seconds": arguments.time_bin_seconds,
            "sky_protocol": None if protocol_path is None else str(protocol_path),
            "sky_checkpoint": None if checkpoint_path is None else str(checkpoint_path),
            "mosaic_phase_centre_rad": list(mosaic_phase_centre),
        }

        per_pointing: dict[str, Any] = {}
        local_grid = RegularGrid(
            arguments.target_image_size,
            np.deg2rad(arguments.target_pixel_arcsec / 3600),
        )
        for label, field_id, block in zip(labels, field_ids, prepared, strict=True):
            entry: dict[str, Any] = {
                "field_id": field_id,
                "rows": int(block.visibility.shape[0]),
                "channels": int(block.frequency_hz.size),
                "has_stokes_i_model": block.model_visibility is not None,
            }
            if block.model_visibility is not None:
                fitted = fit_global_fractional_polarization(block)
                entry["global_quv"] = _fit_payload(fitted)
                entry["partitions"] = {
                    name: _fit_payload(result)
                    for name, result in fit_partitioned_global_polarization((block,)).items()
                }
            if not arguments.skip_images:
                images = dirty_stokes_images(block, local_grid)
                np.savez(
                    arguments.output / f"{label.lower()}_dirty_stokes.npz",
                    stokes_i=images.stokes_i,
                    stokes_q=images.stokes_q,
                    stokes_u=images.stokes_u,
                    stokes_v=images.stokes_v,
                    psf=images.psf,
                )
                entry["dirty_stokes"] = _image_peaks(images)
            per_pointing[label] = entry
        report["pointings"] = per_pointing

        if all(block.model_visibility is not None for block in prepared):
            holdout_index = field_ids.index(arguments.holdout_field)
            selection = tuple(
                block for index, block in enumerate(prepared) if index != holdout_index
            )
            holdout = prepared[holdout_index]
            selected = fit_global_fractional_polarization_blocks(selection)
            holdout_own = fit_global_fractional_polarization(holdout)
            transferred = evaluate_global_fractional_polarization(
                holdout, selected.q, selected.u, selected.v
            )
            report["pointing_holdout"] = {
                "selection_fields": [
                    field_ids[index]
                    for index in range(len(field_ids))
                    if index != holdout_index
                ],
                "holdout_field": arguments.holdout_field,
                "holdout_label": labels[holdout_index],
                "selection_global_quv": _fit_payload(selected),
                "selection_partitions": {
                    name: _fit_payload(result)
                    for name, result in fit_partitioned_global_polarization(selection).items()
                },
                "holdout_own_global_quv": _fit_payload(holdout_own),
                "holdout_transferred_from_selection": transferred,
            }

        if not arguments.skip_images:
            mosaic_grid = RegularGrid(
                arguments.mosaic_image_size,
                np.deg2rad(arguments.mosaic_pixel_arcsec / 3600),
            )
            print("forming common-sky dirty Q/U/V", flush=True)
            mosaic = dirty_mosaic_stokes_images(
                tuple(prepared),
                mosaic_grid,
                mosaic_phase_centre,
                labels=tuple(labels),
                primary_beam=beam,
                config=direct,
            )
            payload = {
                "stokes_i": mosaic.stokes_i,
                "stokes_q": mosaic.stokes_q,
                "stokes_u": mosaic.stokes_u,
                "stokes_v": mosaic.stokes_v,
                "sensitivity": mosaic.sensitivity,
                "sensitivity_fraction": mosaic.sensitivity_fraction,
            }
            for label, images in mosaic.per_pointing.items():
                for name, array in images.items():
                    payload[f"{label}_{name}"] = array
            np.savez(arguments.output / "mosaic_dirty_stokes.npz", **payload)
            report["mosaic_dirty_stokes"] = {
                "abs_peak_q": float(np.nanmax(np.abs(mosaic.stokes_q))),
                "abs_peak_u": float(np.nanmax(np.abs(mosaic.stokes_u))),
                "abs_peak_v": float(np.nanmax(np.abs(mosaic.stokes_v))),
                "outer_robust_q": outer_image_robust_scale(mosaic.stokes_q),
                "outer_robust_u": outer_image_robust_scale(mosaic.stokes_u),
                "outer_robust_v": outer_image_robust_scale(mosaic.stokes_v),
                "recurrence": mosaic.recurrence,
                "image_size": arguments.mosaic_image_size,
                "pixel_arcsec": arguments.mosaic_pixel_arcsec,
            }
            if arguments.holdout_field in field_ids:
                holdout_index = field_ids.index(arguments.holdout_field)
                holdout_label = labels[holdout_index]
                selection_labels = [
                    label for index, label in enumerate(labels) if index != holdout_index
                ]
                selection_q = np.mean(
                    [mosaic.per_pointing[label]["stokes_q"] for label in selection_labels],
                    axis=0,
                )
                selection_u = np.mean(
                    [mosaic.per_pointing[label]["stokes_u"] for label in selection_labels],
                    axis=0,
                )
                valid = mosaic.sensitivity_fraction >= 0.1
                report["mosaic_dirty_stokes"]["holdout_recurrence"] = {
                    "q": map_image_agreement(
                        selection_q,
                        mosaic.per_pointing[holdout_label]["stokes_q"],
                        valid,
                    ),
                    "u": map_image_agreement(
                        selection_u,
                        mosaic.per_pointing[holdout_label]["stokes_u"],
                        valid,
                    ),
                }

    destination = arguments.output / "report.json"
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
