#!/usr/bin/env python3
"""Global q,u (v=0) under diagonal versus full Jones on calibrated 4-pol data.

Holds the frozen polarisation-test ancestor Stokes I fixed. Fits one global
q,u. Scores fold 3 on every science pointing. RL/LR improvement is the
evidence for leakage modelling; an RR/LL-only change is not.

Does not change default imaging. Full Jones is ``allow_unfrozen`` here
only. This is not RM, self-cal, per-pixel Q/U, or spatial V.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from sl1mjax.beam_operator import BeamOperatorConfig
from sl1mjax.calibration import import_casa_polarization_solution
from sl1mjax.data.canonical import VisibilityBlock
from sl1mjax.data.ms import extract_measurement_set
from sl1mjax.fullpol_prep import (
    SCALAR_WEIGHT_LIMITATION,
    apply_polarization_before_averaging,
    beam_comparison_mask,
    diagnostic_run_metadata,
    fold3_mosaic_score,
    fold_mask_digest,
    fullpol_phase6_folds,
    leakage_modelling_evidence,
    paired_fold3_delta,
    require_calibrator_gate_report,
    require_fullpol_diagnostic_ok,
    require_polarisation_ancestor,
    stamp_diagnostic_interpretation,
)
from sl1mjax.phase6_protocol import sky_and_plan_from_product
from sl1mjax.voltage_flux_refit import FlattenedSky, mosaic_local_directions
from sl1mjax.voltage_polarization import (
    compare_unpolarised_and_global_qu,
    fit_global_qu_voltage,
    require_circular_coherency,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POL = ROOT / "tests" / "fixtures" / "3c391_polarization_golden.npz"
DEFAULT_KBG = ROOT / "tests" / "fixtures" / "3c391_calibration_golden.npz"
DEFAULT_MS = ROOT / "data/3c391/fullpol_prep/3c391_gkb_only_4corr.ms"
DEFAULT_PROTOCOL = Path("outputs/3c391_composite_catalogue_stage3/protocol.json")
DEFAULT_ANCESTOR = ROOT / "outputs/3c391_fullpol_prep/frozen_diagonal_ancestor"
DEFAULT_GATE = ROOT / "outputs/3c391_fullpol_prep/calibrator_gate/report.json"
DEFAULT_OUTPUT = Path("outputs/3c391_voltage_beam_polarization")
BEAM_NAMES = ("diagonal_copolar", "full_jones_unfrozen")
SCIENCE_FIELDS = (2, 3, 4, 5, 6, 7, 8)


def _to_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _field_label(field_id: int) -> str:
    if 2 <= field_id <= 8:
        return f"C{field_id - 1}"
    return f"field_{field_id}"


def _prepare_target_folds(blocks: tuple[VisibilityBlock, ...]):
    folds, prepared = fullpol_phase6_folds(blocks, poison_sealed=True)
    return prepared, folds


def _leave_one_pointing_masks(
    blocks: tuple[VisibilityBlock, ...], holdout_index: int
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    train = []
    holdout = []
    for index, block in enumerate(blocks):
        if index == holdout_index:
            train.append(np.zeros(block.shape, dtype=bool))
            holdout.append(np.asarray(block.active, dtype=bool))
        else:
            train.append(np.asarray(block.active, dtype=bool))
            holdout.append(np.zeros(block.shape, dtype=bool))
    return tuple(train), tuple(holdout)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--polarization-golden", type=Path, default=DEFAULT_POL)
    parser.add_argument("--calibration-golden", type=Path, default=DEFAULT_KBG)
    parser.add_argument("--measurement-set", type=Path, default=DEFAULT_MS)
    parser.add_argument("--sky-protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--frozen-diagonal-product",
        type=Path,
        default=DEFAULT_ANCESTOR,
        help="Frozen polarisation-test ancestor, not the raw baseline product.",
    )
    parser.add_argument("--calibrator-gate-report", type=Path, default=DEFAULT_GATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--science-fields", default="2,3,4,5,6,7,8")
    parser.add_argument("--holdout-field", type=int, default=8)
    parser.add_argument("--beams", default=",".join(BEAM_NAMES))
    parser.add_argument("--qu-steps", type=int, default=8)
    parser.add_argument("--row-stride", type=int, default=1)
    parser.add_argument(
        "--pointing-transfer-diagnostic",
        action="store_true",
        help="Also run leave-one-pointing-out. Does not replace the time-fold protocol.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_args(argv)
    selected_beams = tuple(item.strip() for item in arguments.beams.split(",") if item.strip())
    unknown = [name for name in selected_beams if name not in BEAM_NAMES]
    if unknown:
        raise SystemExit(f"unknown beams {unknown}; choose from {BEAM_NAMES}")
    ancestor, freeze = require_polarisation_ancestor(arguments.frozen_diagonal_product)
    gate = require_calibrator_gate_report(arguments.calibrator_gate_report)
    if not arguments.measurement_set.is_dir():
        raise FileNotFoundError(
            "polarisation experiment needs the G/K/B-only four-correlation MeasurementSet"
        )
    require_fullpol_diagnostic_ok(measurement_set=arguments.measurement_set)

    scripts_directory = str(Path(__file__).resolve().parent)
    if scripts_directory not in sys.path:
        sys.path.insert(0, scripts_directory)
    from diagnose_3c391_voltage_beam_transfer import load_antenna_positions
    from refit_3c391_voltage_beam_fluxes import construct_beams

    science_fields = tuple(int(value) for value in arguments.science_fields.split(",") if value)
    if arguments.holdout_field not in science_fields:
        raise ValueError("holdout field must be one of the science fields")
    dataset = extract_measurement_set(
        arguments.measurement_set,
        data_column="CORRECTED_DATA",
        fields=science_fields,
        row_stride=arguments.row_stride,
    )
    imported = import_casa_polarization_solution(
        arguments.polarization_golden,
        arguments.calibration_golden,
        label="flux_angle",
    )
    blocks = []
    labels = []
    for block in dataset.blocks:
        require_circular_coherency(block)
        field_id = int(np.unique(block.field_id)[0]) if block.field_id is not None else -1
        blocks.append(
            apply_polarization_before_averaging(
                block,
                imported,
                frequency_bins=8,
                time_bin_seconds=60.0,
            )
        )
        labels.append(_field_label(field_id))
    blocks_t, folds = _prepare_target_folds(tuple(blocks))
    train_masks = tuple(
        folds.train[index] & beam_comparison_mask(block)
        for index, block in enumerate(blocks_t)
    )
    holdout_masks = tuple(
        folds.holdout[index] & beam_comparison_mask(block)
        for index, block in enumerate(blocks_t)
    )
    require_fullpol_diagnostic_ok(
        measurement_set=arguments.measurement_set,
        blocks=blocks_t,
        train_masks=train_masks,
        holdout_masks=holdout_masks,
        sealed_masks=folds.sealed,
        regional_started=False,
        fold4_opened=False,
    )
    if arguments.sky_protocol.is_file():
        protocol = json.loads(arguments.sky_protocol.read_text(encoding="utf-8"))
        airy_radius = np.deg2rad(float(protocol["airy_max_radius_deg_at_1ghz"]))
    else:
        airy_radius = np.deg2rad(4.0)
    table, flux, _plan = sky_and_plan_from_product(ancestor)
    active = table.active()
    sky = FlattenedSky(
        l_rad=np.asarray([component.l_rad for component in active], dtype=np.float64),
        m_rad=np.asarray([component.m_rad for component in active], dtype=np.float64),
        flux=np.asarray(flux, dtype=np.float64),
        names=("diagonal_copolar",),
        sizes=(len(active),),
    )
    if sky.flux.size != sky.l_rad.size:
        raise ValueError("frozen diagonal flux does not match active sky atoms")
    local_directions = tuple(
        mosaic_local_directions(
            sky.l_rad, sky.m_rad, table.mosaic_phase_centre_rad, block.phase_centre_rad
        )
        for block in blocks_t
    )
    antenna_position_m = load_antenna_positions(
        polarization_golden=arguments.polarization_golden,
        measurement_set=arguments.measurement_set,
        antenna_count=blocks_t[0].antenna_count,
    )
    beams = construct_beams(airy_radius)
    operator = BeamOperatorConfig(visibility_chunk_size=256, pixel_chunk_size=512)
    arguments.output.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "diagnostic": "3c391_voltage_beam_polarization",
        **diagnostic_run_metadata(),
        "role": "polarisation_test_ancestor",
        "not_final_cband_stokes_i": True,
        "intensity_frozen": True,
        "v": 0.0,
        "regional_polarization": "not_started",
        "next_step_not_started": [
            "regional_qu",
            "self_cal",
            "rm",
            "per_pixel_polarization",
            "spatial_v",
        ],
        "full_jones_factory_unchanged": True,
        "frozen_diagonal_product": str(ancestor.resolve()),
        "ancestor": freeze,
        "calibrator_gate": {
            "path": str(Path(arguments.calibrator_gate_report).resolve()),
            "passed": gate["passed"],
            "polarisation_applies": gate.get("polarisation_applies"),
        },
        "stokes_i_beam": "diagonal_copolar",
        "topology_ladder": "not_required",
        "weight_limitation": SCALAR_WEIGHT_LIMITATION,
        "leakage_evidence_rule": (
            "Improvement in RL/LR is the main evidence for leakage modelling. "
            "A change only in RR/LL is not sufficient evidence for the "
            "off-diagonal Jones terms."
        ),
        "masks": {
            "beam_comparison": "jax_prepared_active_only",
            "casa_comparison": "intersection_of_jax_and_casa_actives",
        },
        "folds": {
            "train": [0, 1, 2],
            "holdout": 3,
            "sealed": 4,
            "poisoned": True,
            "bin_seconds": folds.bin_seconds,
            "train_digest": fold_mask_digest(train_masks),
            "holdout_digest": fold_mask_digest(holdout_masks),
            "sealed_digest": fold_mask_digest(folds.sealed),
            "train_samples": [int(np.count_nonzero(mask)) for mask in train_masks],
            "holdout_samples": [int(np.count_nonzero(mask)) for mask in holdout_masks],
            "sealed_samples": [int(np.count_nonzero(mask)) for mask in folds.sealed],
        },
        "pointing_ids": labels,
        "pointing_transfer_diagnostic": bool(arguments.pointing_transfer_diagnostic),
        "convention_acceptance": "separate_gate",
        "scientific_beam_acceptance": "separate_gate",
        "limited_parallactic_angle": True,
        "beams": {},
    }
    for beam_name in selected_beams:
        print(f"fitting global q,u under {beam_name}", flush=True)
        fitted = fit_global_qu_voltage(
            blocks_t,
            sky.flux,
            local_directions,
            beams[beam_name],
            antenna_position_m=antenna_position_m,
            train_masks=train_masks,
            holdout_masks=holdout_masks,
            config=operator,
            steps=arguments.qu_steps,
        )
        compared = compare_unpolarised_and_global_qu(
            blocks_t,
            sky.flux,
            local_directions,
            beams[beam_name],
            antenna_position_m=antenna_position_m,
            sample_masks=holdout_masks,
            config=operator,
            q=fitted.q,
            u=fitted.u,
        )
        fold3 = fold3_mosaic_score(
            blocks_t,
            compared["global_qu_predictions"],
            holdout_masks,
            antenna_position_m=antenna_position_m,
            pointing_ids=labels,
        )
        summary["beams"][beam_name] = {
            "q": fitted.q,
            "u": fitted.u,
            "v": fitted.v,
            "train_loss": fitted.train_loss,
            "holdout_loss": fitted.holdout_loss,
            "regional_polarization": fitted.regional_polarization,
            "fold3": fold3,
            "global_qu_improves_unpolarised": bool(
                compared["global_qu_mse"] < compared["unpolarised_mse"]
            ),
        }
        print(
            beam_name,
            f"q={fitted.q:.5f}",
            f"u={fitted.u:.5f}",
            "fold3",
            fold3["total"]["mse"],
            "RR",
            fold3["RR"]["mse"],
            "LL",
            fold3["LL"]["mse"],
            "RL",
            fold3["RL"]["mse"],
            "LR",
            fold3["LR"]["mse"],
            flush=True,
        )
        if arguments.pointing_transfer_diagnostic:
            holdout_index = science_fields.index(arguments.holdout_field)
            loo_train, loo_holdout = _leave_one_pointing_masks(blocks_t, holdout_index)
            loo_train = tuple(
                mask & beam_comparison_mask(block)
                for mask, block in zip(loo_train, blocks_t, strict=True)
            )
            loo_holdout = tuple(
                mask & beam_comparison_mask(block)
                for mask, block in zip(loo_holdout, blocks_t, strict=True)
            )
            loo = fit_global_qu_voltage(
                blocks_t,
                sky.flux,
                local_directions,
                beams[beam_name],
                antenna_position_m=antenna_position_m,
                train_masks=loo_train,
                holdout_masks=loo_holdout,
                config=operator,
                steps=arguments.qu_steps,
            )
            summary["beams"][beam_name]["pointing_transfer"] = {
                "holdout_label": labels[holdout_index],
                "q": loo.q,
                "u": loo.u,
                "holdout_loss": loo.holdout_loss,
                "replaces_time_fold": False,
            }
        require_fullpol_diagnostic_ok(summary=summary, fold4_opened=False, regional_started=False)
        (arguments.output / "summary.json").write_text(
            json.dumps(_to_json(stamp_diagnostic_interpretation(summary)), indent=2, sort_keys=True)
            + "\n"
        )
    if "diagonal_copolar" in summary["beams"] and "full_jones_unfrozen" in summary["beams"]:
        delta = paired_fold3_delta(
            summary["beams"]["diagonal_copolar"]["fold3"],
            summary["beams"]["full_jones_unfrozen"]["fold3"],
        )
        evidence = leakage_modelling_evidence(delta)
        summary["comparison"] = {
            "same_active_samples": True,
            "mask": "beam_comparison_mask",
            "delta_full_jones_minus_diagonal": delta,
            "leakage_evidence": evidence,
        }
        print(
            "full_jones_minus_diagonal",
            "total",
            delta["total"]["mse"],
            "RR_LL",
            delta["RR_LL"]["mse"],
            "RL_LR",
            delta["RL_LR"]["mse"],
            "off_diagonal_evidence",
            evidence["sufficient_for_off_diagonal_jones"],
            flush=True,
        )
        stamped = stamp_diagnostic_interpretation(summary)
        require_fullpol_diagnostic_ok(summary=stamped, fold4_opened=False, regional_started=False)
        (arguments.output / "summary.json").write_text(
            json.dumps(_to_json(stamped), indent=2, sort_keys=True) + "\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
