#!/usr/bin/env python3
"""Ablate CASA and JAX G/K/B calibration terms against a frozen 3C391 sky."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, replace
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

from sl1mjax.calibration import (
    CalibrationSolution,
    baseline_jones,
    import_casa_golden_solution,
    read_calibration,
)
from sl1mjax.calibration_diagnostics import evaluate_fixed_sky_calibration
from sl1mjax.data.canonical import VisibilityBlock, read_dataset
from sl1mjax.split import interleaved_time_fold_masks


def _assert_compatible(first: CalibrationSolution, second: CalibrationSolution) -> None:
    """Require compatible coordinates for term substitution."""

    if first.antenna_count != second.antenna_count:
        raise ValueError("calibration solutions have different antenna counts")
    if first.correlations != second.correlations:
        raise ValueError("calibration solutions have different correlations")
    if not np.array_equal(first.bandpass_frequency_hz, second.bandpass_frequency_hz):
        raise ValueError("calibration solutions have different bandpass frequencies")
    if not np.isclose(first.reference_frequency_hz, second.reference_frequency_hz):
        raise ValueError("calibration solutions have different reference frequencies")
    if first.reference_antenna != second.reference_antenna:
        raise ValueError("calibration solutions have different reference antennas")


def _rebase_reference_frequency(
    solution: CalibrationSolution,
    reference_frequency_hz: float,
) -> CalibrationSolution:
    """Change the delay reference while preserving every antenna Jones term."""

    if not np.isfinite(reference_frequency_hz) or reference_frequency_hz <= 0:
        raise ValueError("reference frequency must be finite and positive")
    offset_hz = reference_frequency_hz - solution.reference_frequency_hz
    gain_phase = np.exp(-2j * np.pi * solution.delays_s * offset_hz)
    return replace(
        solution,
        gains=solution.gains * gain_phase[None, :, :],
        reference_frequency_hz=reference_frequency_hz,
        provenance={
            **solution.provenance,
            "ablation_original_reference_frequency_hz": (solution.reference_frequency_hz),
        },
    )


def _canonicalize_gain_bandpass_gauge(
    solution: CalibrationSolution,
) -> CalibrationSolution:
    """Move the bandpass reference-channel value into the time gains.

    A frequency-independent complex factor can live in either G or B without
    changing the combined Jones term. CASA and SL1MJax use different choices
    for that factor, so individual G/B substitutions are meaningful only after
    both solutions use the same convention.
    """

    bandpass = solution.bandpass.copy()
    gains = solution.gains.copy()
    reference_index = int(
        np.argmin(np.abs(solution.bandpass_frequency_hz - solution.reference_frequency_hz))
    )
    factors = np.empty(
        (solution.antenna_count, solution.receptor_count),
        dtype=np.complex128,
    )
    for antenna in range(solution.antenna_count):
        for receptor in range(solution.receptor_count):
            valid = (
                solution.bandpass_valid[antenna, :, receptor]
                & np.isfinite(bandpass[antenna, :, receptor])
                & (np.abs(bandpass[antenna, :, receptor]) > 0)
            )
            if not np.any(valid):
                factors[antenna, receptor] = 1.0
                continue
            valid_indices = np.flatnonzero(valid)
            selected = valid_indices[np.argmin(np.abs(valid_indices - reference_index))]
            factors[antenna, receptor] = bandpass[antenna, selected, receptor]
    gains *= factors[None, :, :]
    bandpass /= factors[:, None, :]
    return replace(
        solution,
        gains=gains,
        bandpass=bandpass,
        provenance={
            **solution.provenance,
            "ablation_g_b_gauge": "bandpass unity at nearest valid reference channel",
        },
    )


def _hybrid_solution(
    jax_solution: CalibrationSolution,
    casa_solution: CalibrationSolution,
    *,
    casa_gain: bool,
    casa_delay: bool,
    casa_bandpass: bool,
) -> CalibrationSolution:
    """Construct one auditable G/K/B factorial candidate."""

    _assert_compatible(jax_solution, casa_solution)
    changes: dict[str, Any] = {}
    if casa_gain:
        changes.update(
            gains=casa_solution.gains,
            gain_time_s=casa_solution.gain_time_s,
            gain_valid=casa_solution.gain_valid,
            gain_interval_s=casa_solution.gain_interval_s,
        )
    if casa_delay:
        changes.update(
            delays_s=casa_solution.delays_s,
            delay_valid=casa_solution.delay_valid,
        )
    if casa_bandpass:
        changes.update(
            bandpass=casa_solution.bandpass,
            bandpass_valid=casa_solution.bandpass_valid,
        )
    sources = {
        "G": "casa" if casa_gain else "jax",
        "K": "casa" if casa_delay else "jax",
        "B": "casa" if casa_bandpass else "jax",
    }
    return replace(
        jax_solution,
        **changes,
        interpolation="linear",
        provenance={
            **jax_solution.provenance,
            "ablation_terms": sources,
        },
    )


def _factorial_candidates(
    jax_solution: CalibrationSolution,
    casa_solution: CalibrationSolution,
) -> list[tuple[str, CalibrationSolution, dict[str, Any]]]:
    candidates = []
    for casa_gain, casa_delay, casa_bandpass in product((False, True), repeat=3):
        terms = {
            "G": "casa" if casa_gain else "jax",
            "K": "casa" if casa_delay else "jax",
            "B": "casa" if casa_bandpass else "jax",
        }
        label = "_".join(f"{source}_{term}" for term, source in terms.items())
        candidates.append(
            (
                label,
                _hybrid_solution(
                    jax_solution,
                    casa_solution,
                    casa_gain=casa_gain,
                    casa_delay=casa_delay,
                    casa_bandpass=casa_bandpass,
                ),
                {
                    "kind": "factorial",
                    "gain_source": terms["G"],
                    "delay_source": terms["K"],
                    "bandpass_source": terms["B"],
                    "flux_scaled": True,
                    "propagate_weights": False,
                },
            )
        )
    return candidates


def _load_predictions(
    path: Path,
    blocks: tuple[VisibilityBlock, ...],
) -> tuple[np.ndarray, ...]:
    with np.load(path) as stored:
        predictions = tuple(
            np.asarray(stored[f"prediction_C{index + 1}"]) for index in range(len(blocks))
        )
    for index, (block, prediction) in enumerate(zip(blocks, predictions, strict=True), start=1):
        if prediction.shape != block.shape or np.any(~np.isfinite(prediction)):
            raise ValueError(f"prediction_C{index} is incompatible with its block")
    return predictions


def _assert_aligned(
    reference: VisibilityBlock,
    candidate: VisibilityBlock,
    label: str,
) -> None:
    if reference.shape != candidate.shape:
        raise ValueError(f"{label} shape does not match the reference")
    for name in ("antenna1", "antenna2", "scan_id", "field_id"):
        if not np.array_equal(getattr(reference, name), getattr(candidate, name)):
            raise ValueError(f"{label} does not align on {name}")
    if not np.allclose(reference.time_s, candidate.time_s, rtol=0.0, atol=1e-6):
        raise ValueError(f"{label} does not align on time")
    if not np.allclose(reference.uvw_m, candidate.uvw_m, rtol=1e-12, atol=1e-9):
        raise ValueError(f"{label} does not align on UVW coordinates")


def _write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _relative(before: float, after: float) -> float:
    return after / before - 1.0


def _propagate_aligned_weights(
    blocks: tuple[VisibilityBlock, ...],
    solution: CalibrationSolution,
) -> tuple[VisibilityBlock, ...]:
    """Propagate gain weights without changing frozen averaging coordinates."""

    output = []
    for block in blocks:
        baseline, valid = baseline_jones(
            solution,
            block.time_s,
            block.frequency_hz,
            block.antenna1,
            block.antenna2,
            extrapolate=True,
            phase_centre_rad=block.phase_centre_rad,
            spectral_window_id=block.spectral_window_id,
        )
        baseline_array = np.asarray(baseline)
        valid_array = np.asarray(valid) & np.isfinite(baseline_array)
        output.append(
            replace(
                block,
                weight=block.weight
                * np.where(
                    valid_array,
                    np.abs(baseline_array) ** 2,
                    0.0,
                ),
                flag=block.flag | ~valid_array,
                provenance={
                    **block.provenance,
                    "ablation_weight_policy": "post-average gain propagation",
                },
            )
        )
    return tuple(output)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "measurement_set",
        type=Path,
        nargs="?",
        default=Path("data/3c391_work_v2/3c391_ctm_mosaic_10s_spw0.ms"),
    )
    parser.add_argument(
        "--golden",
        type=Path,
        default=Path("tests/fixtures/3c391_calibration_golden.npz"),
    )
    parser.add_argument(
        "--jax-calibration",
        type=Path,
        default=Path(
            "outputs/3c391_calibration_composite_time_complexity/selected_calibration.npz"
        ),
    )
    parser.add_argument(
        "--calibration-sweep-summary",
        type=Path,
        default=Path("outputs/3c391_calibration_composite_time_complexity/summary.json"),
    )
    parser.add_argument(
        "--reference-fixture",
        type=Path,
        default=Path("outputs/3c391_mosaic_imaging_fixture.zarr"),
    )
    parser.add_argument(
        "--composite-checkpoint",
        type=Path,
        default=Path("outputs/3c391_composite_catalogue_stage3/full_lambda_0.0003.npz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/3c391_calibration_term_ablation"),
    )
    parser.add_argument("--frequency-bins", type=int, default=4)
    parser.add_argument("--time-bin-s", type=float, default=60.0)
    parser.add_argument("--time-fold-seconds", type=float, default=60.0)
    parser.add_argument("--chunk-rows", type=int, default=2048)
    arguments = parser.parse_args()

    from image_3c391_target import _extract_target

    reference_blocks = read_dataset(arguments.reference_fixture).blocks
    predictions = _load_predictions(arguments.composite_checkpoint, reference_blocks)
    jax_solution = read_calibration(arguments.jax_calibration)
    casa_solution = import_casa_golden_solution(
        arguments.golden,
        field_id=1,
        interpolation="linear",
        gain_table="flux_gain",
    )
    jax_solution = _rebase_reference_frequency(
        jax_solution,
        casa_solution.reference_frequency_hz,
    )
    jax_solution = _canonicalize_gain_bandpass_gauge(jax_solution)
    casa_solution = _canonicalize_gain_bandpass_gauge(casa_solution)
    casa_unscaled = import_casa_golden_solution(
        arguments.golden,
        field_id=1,
        interpolation="linear",
        gain_table="gain",
    )
    casa_unscaled = _canonicalize_gain_bandpass_gauge(casa_unscaled)
    sweep = json.loads(arguments.calibration_sweep_summary.read_text(encoding="utf-8"))
    jax_flux_jy = float(sweep["calibrator_metrics"]["secondary_flux_jy"])
    jax_unscaled = replace(
        jax_solution,
        gains=jax_solution.gains * np.sqrt(jax_flux_jy),
        provenance={
            **jax_solution.provenance,
            "flux_scale_removed_for_ablation": jax_flux_jy,
        },
    )

    specifications = _factorial_candidates(jax_solution, casa_solution)
    specifications.extend(
        (
            (
                "jax_unscaled_G_jax_K_jax_B",
                jax_unscaled,
                {
                    "kind": "flux_control",
                    "gain_source": "jax",
                    "delay_source": "jax",
                    "bandpass_source": "jax",
                    "flux_scaled": False,
                    "propagate_weights": False,
                },
            ),
            (
                "casa_unscaled_G_casa_K_casa_B",
                casa_unscaled,
                {
                    "kind": "flux_control",
                    "gain_source": "casa",
                    "delay_source": "casa",
                    "bandpass_source": "casa",
                    "flux_scaled": False,
                    "propagate_weights": False,
                },
            ),
        )
    )

    candidate_blocks: dict[str, tuple[VisibilityBlock, ...]] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for label, solution, candidate_metadata in specifications:
        print(f"extracting {label}", flush=True)
        blocks = []
        for index, reference in enumerate(reference_blocks, start=1):
            if reference.field_id is None:
                raise ValueError(f"reference C{index} has no field ID")
            field_ids = np.unique(reference.field_id)
            if field_ids.size != 1:
                raise ValueError(f"reference C{index} mixes field IDs")
            _, calibrated = _extract_target(
                arguments.measurement_set,
                solution,
                field_id=int(field_ids[0]),
                frequency_bins=arguments.frequency_bins,
                time_bin_s=arguments.time_bin_s,
                chunk_rows=arguments.chunk_rows,
                raw_flag_source="post_application",
                propagate_weights=False,
            )
            _assert_aligned(reference, calibrated, f"{label} C{index}")
            blocks.append(calibrated)
        candidate_blocks[label] = tuple(blocks)
        metadata[label] = candidate_metadata

    for label, source_label, solution, source in (
        (
            "jax_G_jax_K_jax_B_propagated_weights",
            "jax_G_jax_K_jax_B",
            jax_solution,
            "jax",
        ),
        (
            "casa_G_casa_K_casa_B_propagated_weights",
            "casa_G_casa_K_casa_B",
            casa_solution,
            "casa",
        ),
    ):
        candidate_blocks[label] = _propagate_aligned_weights(
            candidate_blocks[source_label],
            solution,
        )
        metadata[label] = {
            "kind": "weight_control",
            "gain_source": source,
            "delay_source": source,
            "bandpass_source": source,
            "flux_scaled": True,
            "propagate_weights": True,
        }

    common_active = tuple(
        reference.active
        & np.logical_and.reduce([blocks[index].active for blocks in candidate_blocks.values()])
        for index, reference in enumerate(reference_blocks)
    )
    development, validation, sealed_test = interleaved_time_fold_masks(
        reference_blocks,
        bin_seconds=arguments.time_fold_seconds,
    )
    development_masks = tuple(
        mask & active for mask, active in zip(development, common_active, strict=True)
    )
    validation_masks = tuple(
        mask & active for mask, active in zip(validation, common_active, strict=True)
    )
    test_masks = tuple(
        mask & active for mask, active in zip(sealed_test, common_active, strict=True)
    )

    casa_reference_selection = evaluate_fixed_sky_calibration(
        "casa_corrected",
        reference_blocks,
        predictions,
        development_masks,
        validation_masks,
    )
    casa_reference_test = evaluate_fixed_sky_calibration(
        "casa_corrected",
        reference_blocks,
        predictions,
        development_masks,
        test_masks,
    )
    evaluations = []
    rows = []
    reference_visibilities = tuple(block.visibility for block in reference_blocks)
    for label, blocks in candidate_blocks.items():
        selection = evaluate_fixed_sky_calibration(
            label,
            blocks,
            predictions,
            development_masks,
            validation_masks,
        )
        test = evaluate_fixed_sky_calibration(
            label,
            blocks,
            predictions,
            development_masks,
            test_masks,
        )
        casa_distance = evaluate_fixed_sky_calibration(
            label,
            blocks,
            reference_visibilities,
            development_masks,
            test_masks,
        )
        evaluations.append(
            {
                "label": label,
                **metadata[label],
                "selection": asdict(selection),
                "sealed_test": asdict(test),
                "sealed_distance_from_casa_corrected": asdict(casa_distance),
            }
        )
        rows.append(
            {
                "label": label,
                **metadata[label],
                "validation_power": selection.holdout["normalized_residual_power"],
                "sealed_test_power": test.holdout["normalized_residual_power"],
                "sealed_distance_from_casa": casa_distance.holdout["normalized_residual_power"],
            }
        )

    by_label = {row["label"]: row for row in rows}
    base_label = "jax_G_jax_K_jax_B"
    casa_label = "casa_G_casa_K_casa_B"
    effects = {}
    for term, label in (
        ("G", "casa_G_jax_K_jax_B"),
        ("K", "jax_G_casa_K_jax_B"),
        ("B", "jax_G_jax_K_casa_B"),
    ):
        effects[term] = {
            "validation_relative_to_all_jax": _relative(
                float(by_label[base_label]["validation_power"]),
                float(by_label[label]["validation_power"]),
            ),
            "sealed_test_relative_to_all_jax": _relative(
                float(by_label[base_label]["sealed_test_power"]),
                float(by_label[label]["sealed_test_power"]),
            ),
        }
    effects["all_casa"] = {
        "validation_relative_to_all_jax": _relative(
            float(by_label[base_label]["validation_power"]),
            float(by_label[casa_label]["validation_power"]),
        ),
        "sealed_test_relative_to_all_jax": _relative(
            float(by_label[base_label]["sealed_test_power"]),
            float(by_label[casa_label]["sealed_test_power"]),
        ),
    }

    rows.sort(key=lambda row: float(row["validation_power"]))
    summary = {
        "schema_version": 1,
        "measurement_set": arguments.measurement_set.name,
        "reference_fixture": str(arguments.reference_fixture),
        "composite_checkpoint": str(arguments.composite_checkpoint),
        "protocol": {
            "development_folds": [0, 1, 2],
            "validation_fold": 3,
            "sealed_test_fold": 4,
            "time_bin_seconds": arguments.time_fold_seconds,
            "target_flags": "post-application flags",
            "default_weight_policy": "fixed MS WEIGHT; CASA calwt=False",
        },
        "gain_epoch_counts": {
            "jax": int(jax_solution.gain_time_s.size),
            "casa": int(casa_solution.gain_time_s.size),
        },
        "jax_flux_jy": jax_flux_jy,
        "casa_flux_jy": 2.296007665849216,
        "casa_corrected_selection": asdict(casa_reference_selection),
        "casa_corrected_sealed_test": asdict(casa_reference_test),
        "candidates": evaluations,
        "validation_ranking": [row["label"] for row in rows],
        "one_term_effects": effects,
    }
    arguments.output.mkdir(parents=True, exist_ok=True)
    (arguments.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_table(arguments.output / "ranking.csv", rows)
    print(
        json.dumps(
            {
                "gain_epoch_counts": summary["gain_epoch_counts"],
                "one_term_effects": effects,
                "rows": rows,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
