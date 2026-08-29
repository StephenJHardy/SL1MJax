#!/usr/bin/env python3
"""Coarse joint Q/U activation on the frozen 3C391 Stokes-I topology.

Freezes calibration, I topology, frequency, and the existing averaging.
Declares I-only regions (no Q/U peaks). Compares unpolarised, one global
q+iu, and joint regional q_r+iu_r, all with v=0. Scores leave-one-pointing-out
across all seven pointings plus deterministic baseline/time/channel halves,
and reports primary-beam-radius and parallactic-angle consistency.

C7 is not a sealed partition for choosing regions. This is not RM, self-cal,
or polarisation-driven pixel splits.
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
from sl1mjax.calibration import apply_calibration, import_casa_polarization_solution
from sl1mjax.composite import predict_mosaic_composite
from sl1mjax.data.ms import extract_measurement_set
from sl1mjax.direct_operator import DirectDFTConfig
from sl1mjax.polarization_activation import (
    beam_radius_cohort_masks,
    compare_linear_polarization_models,
    declare_coarse_stokes_i_regions,
    evaluate_linear_polarization,
    leave_one_block_out_scores,
    linear_polarization_as_dict,
    observing_geometry_report,
    parallactic_cohort_masks,
    partitioned_linear_polarization_scores,
    stokes_i_from_visibility,
)
from sl1mjax.polarization_diagnostics import (
    dirty_stokes_images,
    outer_image_robust_scale,
    stokes_visibility_planes,
)
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
DEFAULT_OUTPUT = ROOT / "outputs" / "3c391_coarse_polarization_validation"


def _load_global_helpers():
    scripts_directory = str(Path(__file__).resolve().parent)
    if scripts_directory not in sys.path:
        sys.path.insert(0, scripts_directory)
    import validate_3c391_global_polarization as global_validation

    return global_validation


def _region_payload(regions) -> list[dict[str, Any]]:
    return [
        {
            "name": region.name,
            "stokes_i_jy": region.stokes_i_jy,
            "component_names": [component.name for component in region.components],
            "provenance": region.provenance,
        }
        for region in regions
    ]


def _predict_regions(blocks, regions, mosaic_phase_centre, beam, direct):
    predicted: dict[str, tuple[np.ndarray, ...]] = {}
    for region in regions:
        print(f"predicting frozen M_I for {region.name}", flush=True)
        visibilities = predict_mosaic_composite(
            blocks,
            region.components,
            mosaic_phase_centre,
            primary_beam=beam,
            config=direct,
        )
        predicted[region.name] = tuple(
            stokes_i_from_visibility(visibility, block.correlations)
            for visibility, block in zip(visibilities, blocks, strict=True)
        )
    return predicted


def _diagnostic_v(blocks) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for index, block in enumerate(blocks, start=1):
        planes = stokes_visibility_planes(block)
        selected = planes.weight_v > 0
        if not np.any(selected):
            continue
        residual = planes.stokes_v[selected]
        weight = planes.weight_v[selected]
        power = float(np.sum(weight * np.abs(residual) ** 2))
        report[f"C{index}"] = {
            "weighted_v_power": power,
            "n_samples": int(np.count_nonzero(selected)),
            "note": "sky model holds v=0; this is observed residual V",
        }
    return report


def _cohort_comparison(blocks, region_stokes_i, first_masks, second_masks, median, unit):
    first = compare_linear_polarization_models(
        blocks, region_stokes_i, sample_masks=first_masks
    )
    second = compare_linear_polarization_models(
        blocks, region_stokes_i, sample_masks=second_masks
    )
    return {
        "median": median,
        "unit": unit,
        "low_or_inner": {
            kind: linear_polarization_as_dict(model) for kind, model in first.items()
        },
        "high_or_outer": {
            kind: linear_polarization_as_dict(model) for kind, model in second.items()
        },
        "transfer_low_to_high": {
            kind: evaluate_linear_polarization(
                blocks, region_stokes_i, model.q, model.u, sample_masks=second_masks
            )
            for kind, model in first.items()
        },
        "transfer_high_to_low": {
            kind: evaluate_linear_polarization(
                blocks, region_stokes_i, model.q, model.u, sample_masks=first_masks
            )
            for kind, model in second.items()
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--polarization-golden", type=Path, default=DEFAULT_POL)
    parser.add_argument("--calibration-golden", type=Path, default=DEFAULT_KBG)
    parser.add_argument("--measurement-set", type=Path, default=None)
    parser.add_argument("--no-measurement-set", action="store_true")
    parser.add_argument("--sky-protocol", type=Path, default=None)
    parser.add_argument("--sky-checkpoint", type=Path, default=None)
    parser.add_argument("--sky-root", type=Path, action="append", default=None)
    parser.add_argument("--science-fields", default="2,3,4,5,6,7,8")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--frequency-bins", type=int, default=8)
    parser.add_argument("--time-bin-seconds", type=float, default=60.0)
    parser.add_argument("--row-stride", type=int, default=1)
    parser.add_argument(
        "--region-scheme",
        choices=(
            "central_radial_widefield",
            "component_dictionaries",
            "central_ancestor_cells",
        ),
        default="central_radial_widefield",
    )
    parser.add_argument("--ancestor-arcsec", type=float, default=64.0)
    parser.add_argument("--minimum-region-i-jy", type=float, default=0.2)
    parser.add_argument(
        "--no-widefield",
        action="store_true",
        help="Omit coarse/catalogue from the polarised sky (recommended off-axis)",
    )
    parser.add_argument("--target-image-size", type=int, default=65)
    parser.add_argument("--target-pixel-arcsec", type=float, default=8.0)
    parser.add_argument("--skip-images", action="store_true")
    parser.add_argument("--precision", choices=("float32", "float64"), default="float32")
    arguments = parser.parse_args(argv)

    helper = _load_global_helpers()
    imported = import_casa_polarization_solution(
        arguments.polarization_golden,
        arguments.calibration_golden,
        label="flux_angle",
    )
    measurement_set = None
    if not arguments.no_measurement_set:
        measurement_set = arguments.measurement_set
        if measurement_set is None and DEFAULT_MS.is_dir():
            measurement_set = DEFAULT_MS
        if measurement_set is None:
            raise FileNotFoundError("measurement set is required for the coarse Q/U test")
        if not measurement_set.exists():
            raise FileNotFoundError(f"measurement set does not exist: {measurement_set}")

    arguments.output.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "evidence_grade": False,
        "milestone": "coarse_joint_qu_activation",
        "claim": "test_spatial_polarization_not_an_astrophysical_image",
        "v_in_sky_model": 0.0,
        "next_step_not_started": [
            "rm",
            "frequency_dependent_polarization",
            "self_cal",
            "polarization_driven_pixel_splits",
            "spatial_v_activation",
        ],
        "c7_status": (
            "examined_in_global_test_not_sealed_for_region_choice;"
            "regions_declared_from_stokes_i_only;"
            "scoring_is_leave_one_pointing_out_across_all_seven"
        ),
    }
    if measurement_set is None:
        destination = arguments.output / "report.json"
        destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(destination)
        return 0

    protocol_path = arguments.sky_protocol
    checkpoint_path = arguments.sky_checkpoint
    if protocol_path is None and DEFAULT_PROTOCOL.is_file():
        protocol_path = DEFAULT_PROTOCOL
    if checkpoint_path is None and DEFAULT_CHECKPOINT.is_file():
        checkpoint_path = DEFAULT_CHECKPOINT
    if protocol_path is None or checkpoint_path is None:
        raise FileNotFoundError("frozen Stokes-I protocol and checkpoint are required")

    science_fields = tuple(int(value) for value in arguments.science_fields.split(",") if value)
    print(f"extracting fields {science_fields} from {measurement_set}", flush=True)
    dataset = extract_measurement_set(
        measurement_set,
        data_column="CORRECTED_DATA",
        fields=science_fields,
        row_stride=arguments.row_stride,
    )
    mosaic_phase_centre = dataset.blocks[0].phase_centre_rad
    sky_roots = tuple(arguments.sky_root or ())
    if DEFAULT_SKY_ROOT.is_dir():
        sky_roots = (*sky_roots, DEFAULT_SKY_ROOT)
    sky_roots = (*sky_roots, ROOT, Path.cwd())
    components = helper._load_sky_components(
        protocol_path, checkpoint_path, mosaic_phase_centre, sky_roots
    )
    regions = declare_coarse_stokes_i_regions(
        components,
        scheme=arguments.region_scheme,
        ancestor_arcsec=arguments.ancestor_arcsec,
        minimum_region_i_jy=arguments.minimum_region_i_jy,
        include_widefield=not arguments.no_widefield,
    )
    print(
        f"declared {len(regions)} I-only regions: "
        + ", ".join(f"{region.name}={region.stokes_i_jy:.3f}Jy" for region in regions),
        flush=True,
    )
    report["regions"] = _region_payload(regions)

    protocol_payload = json.loads(protocol_path.read_text(encoding="utf-8"))
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
    averaged_blocks = []
    for block in dataset.blocks:
        field_id = int(np.unique(block.field_id)[0]) if block.field_id is not None else -1
        labels.append(helper._field_label(field_id))
        print(f"averaging {labels[-1]}", flush=True)
        averaged_blocks.append(
            helper._average_target(
                block,
                frequency_bins=arguments.frequency_bins,
                time_bin_seconds=arguments.time_bin_seconds,
            )
        )

    region_stokes_i = _predict_regions(
        tuple(averaged_blocks),
        regions,
        mosaic_phase_centre,
        beam,
        direct,
    )
    total_i = []
    for block_index, block in enumerate(averaged_blocks):
        total = np.zeros(block.visibility.shape[:2], dtype=np.complex128)
        for planes in region_stokes_i.values():
            total = total + planes[block_index]
        total_i.append(total)

    print("applying polarisation-only Jones to CORRECTED_DATA", flush=True)
    pol_only = helper._polarization_terms_only(imported)
    prepared = []
    for block, model_i in zip(averaged_blocks, total_i, strict=True):
        corrected = apply_calibration(block, pol_only, extrapolate=True)
        model_visibility = np.zeros(block.visibility.shape, dtype=np.complex128)
        model_visibility[..., 0] = model_i
        model_visibility[..., 3] = model_i
        prepared.append(replace(corrected, model_visibility=model_visibility))
    prepared_tuple = tuple(prepared)
    label_tuple = tuple(labels)

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
        "sky_protocol": str(protocol_path),
        "sky_checkpoint": str(checkpoint_path),
        "region_scheme": arguments.region_scheme,
        "ancestor_arcsec": arguments.ancestor_arcsec,
        "minimum_region_i_jy": arguments.minimum_region_i_jy,
        "include_widefield": not arguments.no_widefield,
        "mosaic_phase_centre_rad": list(mosaic_phase_centre),
    }

    print("fitting unpolarised / global / regional q+iu with v=0", flush=True)
    in_sample = compare_linear_polarization_models(prepared_tuple, region_stokes_i)
    report["in_sample"] = {
        kind: linear_polarization_as_dict(model) for kind, model in in_sample.items()
    }
    print("leave-one-pointing-out scores", flush=True)
    report["leave_one_pointing_out"] = leave_one_block_out_scores(
        prepared_tuple, region_stokes_i, labels=label_tuple
    )
    report["partitions"] = partitioned_linear_polarization_scores(
        prepared_tuple, region_stokes_i
    )
    report["observing_geometry"] = observing_geometry_report(
        regions,
        prepared_tuple,
        mosaic_phase_centre,
        imported.antenna_position_m,
        labels=label_tuple,
    )
    low_pa, high_pa, pa_median = parallactic_cohort_masks(
        prepared_tuple, imported.antenna_position_m
    )
    report["parallactic_consistency"] = _cohort_comparison(
        prepared_tuple,
        region_stokes_i,
        low_pa,
        high_pa,
        float(np.rad2deg(pa_median)),
        "deg",
    )
    radius_region = next(
        (region.name for region in regions if region.name == "central_inner"),
        max(regions, key=lambda region: region.stokes_i_jy).name,
    )
    inner_beam, outer_beam, radius_median = beam_radius_cohort_masks(
        regions,
        prepared_tuple,
        mosaic_phase_centre,
        region_name=radius_region,
    )
    report["beam_radius_consistency"] = _cohort_comparison(
        prepared_tuple,
        region_stokes_i,
        inner_beam,
        outer_beam,
        float(np.rad2deg(radius_median)),
        "deg",
    )
    report["beam_radius_consistency"]["reference_region"] = radius_region
    report["diagnostic_v"] = _diagnostic_v(prepared_tuple)

    if not arguments.skip_images:
        grid = RegularGrid(
            arguments.target_image_size,
            np.deg2rad(arguments.target_pixel_arcsec / 3600),
        )
        dirty_v = {}
        for label, block in zip(labels, prepared, strict=True):
            images = dirty_stokes_images(block, grid)
            np.savez(
                arguments.output / f"{label.lower()}_dirty_stokes.npz",
                stokes_v=images.stokes_v,
                stokes_q=images.stokes_q,
                stokes_u=images.stokes_u,
                stokes_i=images.stokes_i,
            )
            dirty_v[label] = {
                "abs_peak_v": float(np.max(np.abs(images.stokes_v))),
                "outer_robust_v": outer_image_robust_scale(images.stokes_v),
                "peak_v": images.peak_v,
            }
        report["dirty_v"] = dirty_v

    destination = arguments.output / "report.json"
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
