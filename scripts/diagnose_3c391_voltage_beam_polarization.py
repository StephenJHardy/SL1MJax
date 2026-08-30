#!/usr/bin/env python3
"""Global q,u (v=0) under diagonal versus full Jones on calibrated 4-pol data.

Keeps the refitted Stokes-I geometry frozen. Starts with one global q,u.
Compares held-out RR/LL and RL/LR. Regional polarisation is recorded as
not started.

Does not change default imaging. Full Jones is ``allow_unfrozen`` here
only. This is not RM, self-cal, per-pixel Q/U, or spatial V.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from sl1mjax.beam_operator import BeamOperatorConfig
from sl1mjax.calibration import (
    apply_calibration,
    identity_solution,
    import_casa_polarization_solution,
)
from sl1mjax.data.canonical import VisibilityBlock
from sl1mjax.data.ms import extract_measurement_set
from sl1mjax.voltage_flux_refit import (
    flatten_sky_atoms,
    mosaic_local_directions,
    paired_score_delta,
    score_visibility_prediction,
)
from sl1mjax.voltage_polarization import (
    compare_unpolarised_and_global_qu,
    fit_global_qu_voltage,
    require_circular_coherency,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POL = ROOT / "tests" / "fixtures" / "3c391_polarization_golden.npz"
DEFAULT_KBG = ROOT / "tests" / "fixtures" / "3c391_calibration_golden.npz"
DEFAULT_MS = Path(
    "/home/stephen/checkouts/SL1MJax-frozen-20260824/data/3c391_ctm_mosaic_10s_spw0.ms"
)
DEFAULT_PROTOCOL = Path("outputs/3c391_composite_catalogue_stage3/protocol.json")
DEFAULT_CHECKPOINT = Path("outputs/3c391_voltage_beam_flux_refit/flux_diagonal_copolar.npz")
DEFAULT_FALLBACK_CHECKPOINT = Path(
    "outputs/3c391_recovery_policy_fit_zero/sealed_active_only.npz"
)
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


def _polarization_terms_only(solution):
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--polarization-golden", type=Path, default=DEFAULT_POL)
    parser.add_argument("--calibration-golden", type=Path, default=DEFAULT_KBG)
    parser.add_argument("--measurement-set", type=Path, default=DEFAULT_MS)
    parser.add_argument("--sky-protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--sky-checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--fallback-checkpoint", type=Path, default=DEFAULT_FALLBACK_CHECKPOINT
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--science-fields", default="2,3,4,5,6,7,8")
    parser.add_argument("--holdout-field", type=int, default=8)
    parser.add_argument("--beams", default=",".join(BEAM_NAMES))
    parser.add_argument("--qu-steps", type=int, default=8)
    parser.add_argument("--row-stride", type=int, default=1)
    arguments = parser.parse_args()
    selected_beams = tuple(item.strip() for item in arguments.beams.split(",") if item.strip())
    unknown = [name for name in selected_beams if name not in BEAM_NAMES]
    if unknown:
        parser.error(f"unknown beams {unknown}; choose from {BEAM_NAMES}")
    checkpoint = arguments.sky_checkpoint
    if not checkpoint.is_file():
        checkpoint = arguments.fallback_checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError("need a refitted or sealed Stokes-I checkpoint")
    if not arguments.measurement_set.is_dir():
        raise FileNotFoundError(
            "polarisation experiment needs calibrated RR/RL/LR/LL from the MeasurementSet"
        )

    scripts_directory = str(Path(__file__).resolve().parent)
    if scripts_directory not in sys.path:
        sys.path.insert(0, scripts_directory)
    from diagnose_3c391_voltage_beam_transfer import load_antenna_positions
    from refit_3c391_voltage_beam_fluxes import construct_beams
    from validate_3c391_global_polarization import _average_target

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
    pol_only = _polarization_terms_only(imported)
    blocks = []
    labels = []
    for block in dataset.blocks:
        require_circular_coherency(block)
        field_id = int(np.unique(block.field_id)[0]) if block.field_id is not None else -1
        averaged = _average_target(block, frequency_bins=8, time_bin_seconds=60.0)
        blocks.append(apply_calibration(averaged, pol_only, extrapolate=True))
        labels.append(_field_label(field_id))
    blocks_t = tuple(blocks)
    holdout_index = science_fields.index(arguments.holdout_field)
    train_masks, holdout_masks = _leave_one_pointing_masks(blocks_t, holdout_index)
    mosaic_phase_centre = blocks_t[0].phase_centre_rad
    protocol = json.loads(arguments.sky_protocol.read_text(encoding="utf-8"))
    airy_radius = np.deg2rad(float(protocol["airy_max_radius_deg_at_1ghz"]))
    from compare_3c391_composite_existing_flags import _components_from_checkpoint
    from diagnose_3c391_voltage_beam_transfer import _resolve_protocol_paths

    protocol_resolved = _resolve_protocol_paths(
        protocol,
        Path.cwd(),
        arguments.sky_protocol.resolve().parent.parent.parent,
        checkpoint.resolve().parent.parent.parent,
        ROOT,
    )
    components = _components_from_checkpoint(checkpoint, protocol_resolved, mosaic_phase_centre)
    sky = flatten_sky_atoms(components)
    local_directions = tuple(
        mosaic_local_directions(
            sky.l_rad, sky.m_rad, mosaic_phase_centre, block.phase_centre_rad
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
        "sky_checkpoint": str(checkpoint),
        "holdout_label": labels[holdout_index],
        "beams": {},
    }
    reference_holdout = None
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
        holdout_block = blocks_t[holdout_index]
        unpolarised = score_visibility_prediction(
            holdout_block,
            compared["unpolarised_predictions"][holdout_index],
            mask=holdout_masks[holdout_index],
        )
        polarised = score_visibility_prediction(
            holdout_block,
            compared["global_qu_predictions"][holdout_index],
            mask=holdout_masks[holdout_index],
        )
        if reference_holdout is None:
            reference_holdout = polarised
        summary["beams"][beam_name] = {
            "q": fitted.q,
            "u": fitted.u,
            "v": fitted.v,
            "train_loss": fitted.train_loss,
            "holdout_loss": fitted.holdout_loss,
            "regional_polarization": fitted.regional_polarization,
            "unpolarised_holdout": unpolarised,
            "global_qu_holdout": polarised,
            "paired_delta_vs_first_beam": paired_score_delta(reference_holdout, polarised),
            "global_qu_improves_unpolarised": bool(
                compared["global_qu_mse"] < compared["unpolarised_mse"]
            ),
        }
        print(
            beam_name,
            f"q={fitted.q:.5f}",
            f"u={fitted.u:.5f}",
            "holdout",
            polarised["total"],
            "RL",
            polarised["correlations"]["RL"]["held_out_loss"],
            "LR",
            polarised["correlations"]["LR"]["held_out_loss"],
            flush=True,
        )
        (arguments.output / "summary.json").write_text(
            json.dumps(_to_json(summary), indent=2, sort_keys=True) + "\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
