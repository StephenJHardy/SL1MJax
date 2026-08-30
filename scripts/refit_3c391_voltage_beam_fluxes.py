#!/usr/bin/env python3
"""Controlled Stokes-I flux refit under streamed voltage beams.

Freezes pixel geometry, source positions, the sealed train/holdout split,
and the Airy L1 sensitivity weights. Refits only leaf fluxes. Compares
held-out loss by pointing, channel, time, RR and LL, and reports paired
deltas plus off-axis catalogue flux movement.

Does not change default imaging. Full Jones is ``allow_unfrozen`` here
only. Polarisation is a separate experiment.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from sl1mjax.beam import VLABeamCatalog, VLAPrimaryBeam
from sl1mjax.beam_operator import BeamOperatorConfig
from sl1mjax.cassbeam_beam import CassbeamCBandVoltageBeam, voltage_beam_for_mode
from sl1mjax.composite import mosaic_beam_sensitivity_weights
from sl1mjax.data.canonical import VisibilityBlock, read_dataset
from sl1mjax.inference import InferenceConfig
from sl1mjax.split import interleaved_time_folds
from sl1mjax.voltage_beam import AnalyticAiryVoltageBeam
from sl1mjax.voltage_flux_refit import (
    flatten_sky_atoms,
    mosaic_local_directions,
    off_axis_atom_report,
    paired_score_delta,
    predict_voltage_mosaic,
    refit_stokes_i_fluxes,
    replace_component_fluxes,
    score_visibility_prediction,
    transfer_diagonal_is_consistent,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = Path("outputs/3c391_recovery_policies")
DEFAULT_REFERENCE = Path("outputs/3c391_gain_time_model_sweep/selected_native_fixture.zarr")
DEFAULT_PROTOCOL = Path("outputs/3c391_composite_catalogue_stage3/protocol.json")
DEFAULT_CHECKPOINT = Path("outputs/3c391_recovery_policy_fit_zero/sealed_active_only.npz")
DEFAULT_TRANSFER = Path("outputs/3c391_voltage_beam_transfer_jax/summary.json")
DEFAULT_OUTPUT = Path("outputs/3c391_voltage_beam_flux_refit")
DEFAULT_POL_GOLDEN = ROOT / "tests" / "fixtures" / "3c391_polarization_golden.npz"
BEAM_NAMES = (
    "static_scalar",
    "streamed_scalar",
    "diagonal_copolar",
    "full_jones_unfrozen",
)


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


def _load_folds(
    path: Path, blocks: tuple[VisibilityBlock, ...], *, fold_count: int = 5
) -> tuple[tuple[np.ndarray, ...], ...]:
    with np.load(path) as stored:
        folds = tuple(
            tuple(
                np.asarray(stored[f"fold{fold}_C{pointing + 1}"], dtype=bool)
                for pointing in range(len(blocks))
            )
            for fold in range(fold_count)
        )
    return folds


def _or_folds(
    folds: tuple[tuple[np.ndarray, ...], ...], selected: tuple[int, ...]
) -> tuple[np.ndarray, ...]:
    return tuple(
        np.logical_or.reduce([folds[fold][pointing] for fold in selected])
        for pointing in range(len(folds[0]))
    )


def _common_active_masks(
    policy_blocks: tuple[VisibilityBlock, ...],
    reference_blocks: tuple[VisibilityBlock, ...],
    reference_masks: tuple[np.ndarray, ...],
) -> tuple[np.ndarray, ...]:
    output = []
    for policy, reference, mask in zip(
        policy_blocks, reference_blocks, reference_masks, strict=True
    ):
        embedded = np.zeros(policy.shape, dtype=bool)
        embedded[: reference.shape[0]] = mask
        output.append(embedded)
    return tuple(output)


def _load_components(protocol_path: Path, checkpoint: Path, phase_centre):
    scripts_directory = str(Path(__file__).resolve().parent)
    if scripts_directory not in sys.path:
        sys.path.insert(0, scripts_directory)
    from compare_3c391_composite_existing_flags import _components_from_checkpoint
    from diagnose_3c391_voltage_beam_transfer import _resolve_protocol_paths

    protocol = _resolve_protocol_paths(
        json.loads(protocol_path.read_text(encoding="utf-8")),
        Path.cwd(),
        protocol_path.resolve().parent.parent.parent,
        checkpoint.resolve().parent.parent.parent,
        ROOT,
    )
    return _components_from_checkpoint(checkpoint, protocol, phase_centre)


def construct_beams(airy_max_radius_rad_at_1ghz: float) -> dict[str, Any]:
    catalog = VLABeamCatalog(airy_max_radius_rad_at_1ghz=airy_max_radius_rad_at_1ghz)
    diagonal = voltage_beam_for_mode("diagonal_copolar")
    return {
        "static_scalar": AnalyticAiryVoltageBeam(catalog=catalog),
        "streamed_scalar": voltage_beam_for_mode("streamed_scalar"),
        "diagonal_copolar": diagonal,
        "full_jones_unfrozen": CassbeamCBandVoltageBeam(
            diagonal.artifact,
            off_diagonal=True,
            allow_unfrozen=True,
            outer=diagonal.outer,
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-directory", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--reference-fixture", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--sky-protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--sky-checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--transfer-summary", type=Path, default=DEFAULT_TRANSFER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--polarization-golden", type=Path, default=DEFAULT_POL_GOLDEN)
    parser.add_argument("--policy", default="active_only")
    parser.add_argument("--beams", default=",".join(BEAM_NAMES))
    parser.add_argument("--lambda-l1", type=float, default=3e-4)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--validation-interval", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--off-axis-arcmin", type=float, default=8.0)
    parser.add_argument(
        "--require-consistent-diagonal",
        action="store_true",
        help="Refuse to run unless the transfer summary has diagonal < Airy on every pointing",
    )
    arguments = parser.parse_args()
    selected_beams = tuple(item.strip() for item in arguments.beams.split(",") if item.strip())
    unknown = [name for name in selected_beams if name not in BEAM_NAMES]
    if unknown:
        parser.error(f"unknown beams {unknown}; choose from {BEAM_NAMES}")

    transfer_gate = None
    if arguments.transfer_summary.is_file():
        transfer_gate = transfer_diagonal_is_consistent(
            json.loads(arguments.transfer_summary.read_text(encoding="utf-8"))
        )
        print("transfer gate", json.dumps(_to_json(transfer_gate)), flush=True)
        if arguments.require_consistent_diagonal and not transfer_gate["consistent"]:
            print("diagonal improvement is not consistent across pointings; stop")
            return 2

    reference = read_dataset(arguments.reference_fixture).blocks
    blocks = read_dataset(arguments.policy_directory / f"{arguments.policy}.zarr").blocks
    folds = _load_folds(arguments.policy_directory / f"{arguments.policy}_folds.npz", blocks)
    train = _or_folds(folds, (0, 1, 2, 3))
    reference_test = interleaved_time_folds(reference, bin_seconds=60.0)[4]
    holdout = _common_active_masks(blocks, reference, reference_test)
    mosaic_phase_centre = reference[0].phase_centre_rad
    protocol = json.loads(arguments.sky_protocol.read_text(encoding="utf-8"))
    airy_radius = np.deg2rad(float(protocol["airy_max_radius_deg_at_1ghz"]))
    components = _load_components(
        arguments.sky_protocol, arguments.sky_checkpoint, mosaic_phase_centre
    )
    sky = flatten_sky_atoms(components)
    beam_i = VLAPrimaryBeam(
        kind="airy",
        catalog=VLABeamCatalog(airy_max_radius_rad_at_1ghz=airy_radius),
    )
    sparsity = np.concatenate(
        mosaic_beam_sensitivity_weights(
            blocks, components, train, mosaic_phase_centre, primary_beam=beam_i
        )
    )
    from diagnose_3c391_voltage_beam_transfer import load_antenna_positions

    antenna_position_m = load_antenna_positions(
        polarization_golden=arguments.polarization_golden,
        measurement_set=None,
        antenna_count=blocks[0].antenna_count,
    )
    local_directions = tuple(
        mosaic_local_directions(
            sky.l_rad, sky.m_rad, mosaic_phase_centre, block.phase_centre_rad
        )
        for block in blocks
    )
    beams = construct_beams(airy_radius)
    operator = BeamOperatorConfig(visibility_chunk_size=256, pixel_chunk_size=512)
    inference = InferenceConfig(
        solver="proximal_sgd",
        batch_grouping="times",
        steps=arguments.steps,
        learning_rate=arguments.learning_rate,
        sparsity_weight=arguments.lambda_l1,
        validation_interval=arguments.validation_interval,
        operator_mode="autodiff",
    )
    arguments.output.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "diagnostic": "3c391_voltage_beam_flux_refit",
        "frozen": {
            "geometry": True,
            "source_positions": True,
            "training_folds": [0, 1, 2, 3],
            "holdout_fold": 4,
            "lambda_l1": arguments.lambda_l1,
            "sparsity_weights": "airy_mosaic_beam_sensitivity",
        },
        "full_jones_factory_unchanged": True,
        "default_imaging_unchanged": True,
        "transfer_gate": transfer_gate,
        "beams": {},
        "notes": (
            "Refit is I-only. Native fixtures stay unused here so the sealed "
            "split is the held-out comparison. Polarisation is not started."
        ),
    }
    reference_scores: dict[str, dict[str, Any]] | None = None
    for beam_name in selected_beams:
        print(f"refitting {beam_name}", flush=True)
        result = refit_stokes_i_fluxes(
            blocks,
            train,
            local_directions,
            sky.flux,
            beams[beam_name],
            antenna_position_m=antenna_position_m,
            config=inference,
            operator_config=operator,
            holdout_masks=holdout,
            sparsity_weights=sparsity,
        )
        fitted = replace_component_fluxes(components, result.flux)
        predictions = predict_voltage_mosaic(
            result.flux,
            blocks,
            local_directions,
            beams[beam_name],
            antenna_position_m=antenna_position_m,
            calibration_state="casa_parang_true",
            config=operator,
        )
        pointing_scores = {}
        for index, (block, prediction, mask) in enumerate(
            zip(blocks, predictions, holdout, strict=True), start=1
        ):
            pointing_scores[f"C{index}"] = score_visibility_prediction(
                block, prediction, mask=mask
            )
        if reference_scores is None:
            reference_scores = pointing_scores
        deltas = {
            name: paired_score_delta(reference_scores[name], pointing_scores[name])
            for name in pointing_scores
        }
        off_axis = off_axis_atom_report(
            sky, sky.flux, result.flux, radius_arcmin_cut=arguments.off_axis_arcmin
        )
        np.savez(
            arguments.output / f"flux_{beam_name}.npz",
            **{f"flux_{component.name}": component.flux for component in fitted},
        )
        payload = {
            "best_step": result.best_step,
            "steps": result.steps,
            "holdout_history": list(result.holdout_history),
            "pointings": pointing_scores,
            "paired_delta_vs_first_beam": deltas,
            "off_axis": off_axis,
        }
        summary["beams"][beam_name] = payload
        print(
            beam_name,
            "holdout",
            {name: scores["total"] for name, scores in pointing_scores.items()},
            flush=True,
        )
        (arguments.output / "summary.json").write_text(
            json.dumps(_to_json(summary), indent=2, sort_keys=True) + "\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
