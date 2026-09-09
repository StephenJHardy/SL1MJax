#!/usr/bin/env python3
"""Seven-pointing opt-in EVLA-C survey smoke test.

Uses the existing 3C391 FOV, guard, and bright-source catalog. Fits a few
flux steps on a small deterministic training subset, writes a checkpoint,
reloads it, and renders products. Does not demand KKT 1e-5.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from sl1mjax.beam_aware_imaging import sky_table_from_records
from sl1mjax.data.canonical import VisibilityBlock
from sl1mjax.evla_c_imaging_convergence import unsupported_direction_account
from sl1mjax.evla_c_survey_beam import IMAGING_NODE_MHZ, survey_catalog_digest
from sl1mjax.inference import InferenceConfig
from sl1mjax.phase6_protocol import compare_operator_modes, write_smoke_reconstruction_products
from sl1mjax.voltage_beam import BeamCoordinates
from sl1mjax.voltage_reconstruction import (
    OPT_IN_SURVEY_BEAM,
    VoltageReconstructionConfig,
    merge_hysteresis_from_records,
    reconstruct_voltage_stokes_i,
    stokes_i_beam,
)
from sl1mjax.wide_field_sky import catalogue_components_from_pinned_json, phase5_starting_table

import sys

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from run_3c391_phase6_bacchus import (
    POINTINGS,
    ROOT,
    _write_checkpoint,
    load_antenna_positions,
    load_pointing_blocks,
    protocol_config,
)

DEFAULT_CATALOGUE = ROOT / "config" / "3c391_radio_guard_catalog.json"
DEFAULT_POL_GOLDEN = ROOT / "tests" / "fixtures" / "3c391_polarization_golden.npz"


def _select_exact_channels(block: VisibilityBlock, allowed_mhz: set[int]) -> VisibilityBlock:
    keep = np.array(
        [int(round(float(frequency) / 1.0e6)) in allowed_mhz for frequency in block.frequency_hz],
        dtype=bool,
    )
    if not np.any(keep):
        native = [int(round(float(frequency) / 1.0e6)) for frequency in block.frequency_hz]
        raise ValueError(
            "no exact survey plane matches the native 3C391 frequencies "
            f"{native}; generate those nodes or refuse interpolation"
        )
    model = None if block.model_visibility is None else block.model_visibility[:, keep]
    return VisibilityBlock(
        uvw_m=block.uvw_m,
        frequency_hz=block.frequency_hz[keep],
        visibility=block.visibility[:, keep],
        weight=block.weight[:, keep],
        flag=block.flag[:, keep],
        time_s=block.time_s,
        antenna1=block.antenna1,
        antenna2=block.antenna2,
        correlations=block.correlations,
        receptor_basis=block.receptor_basis,
        model_visibility=model,
        field_id=block.field_id,
        scan_id=block.scan_id,
        state_id=block.state_id,
        observation_id=block.observation_id,
        feed1=block.feed1,
        feed2=block.feed2,
        interval_s=block.interval_s,
        phase_centre_rad=block.phase_centre_rad,
        data_description_id=block.data_description_id,
        spectral_window_id=block.spectral_window_id,
        polarization_id=block.polarization_id,
        provenance=dict(block.provenance),
    )


def _training_subset(block: VisibilityBlock, max_rows: int) -> VisibilityBlock:
    active_rows = np.flatnonzero(np.any(block.active, axis=(1, 2)))
    if active_rows.size == 0:
        raise ValueError("no active training rows")
    chosen = active_rows[: int(max_rows)]
    return block.select_rows(chosen)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-root", type=Path, required=True)
    parser.add_argument("--survey-catalog-root", type=Path, required=True)
    parser.add_argument("--survey-catalog-digest", default=None)
    parser.add_argument("--catalogue", type=Path, default=DEFAULT_CATALOGUE)
    parser.add_argument("--polarization-golden", type=Path, default=DEFAULT_POL_GOLDEN)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-rows-per-pointing", type=int, default=32)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--skip-operator-gate", action="store_true")
    parser.add_argument("--peak-rss-limit-bytes", type=int, default=16 * 1024**3)
    arguments = parser.parse_args()
    digest = arguments.survey_catalog_digest or survey_catalog_digest(arguments.survey_catalog_root)
    beam = stokes_i_beam(
        OPT_IN_SURVEY_BEAM,
        survey_catalog_root=arguments.survey_catalog_root,
        survey_catalog_digest=digest,
    )
    if beam.model_id.startswith("cassbeam_nominal"):
        raise RuntimeError("survey selector loaded the packaged generic CASSBEAM tables")
    blocks = load_pointing_blocks(arguments.native_root, POINTINGS)
    allowed = set(IMAGING_NODE_MHZ)
    subset = tuple(
        _training_subset(_select_exact_channels(block, allowed), arguments.max_rows_per_pointing)
        for block in blocks
    )
    antenna = load_antenna_positions(arguments.polarization_golden, subset[0].antenna_count)
    table = phase5_starting_table(
        mosaic_phase_centre_rad=subset[0].phase_centre_rad,
        catalogue=catalogue_components_from_pinned_json(arguments.catalogue),
    )
    on_axis = beam.evaluate(
        BeamCoordinates(
            l_rad=np.array([0.0]),
            m_rad=np.array([0.0]),
            frequency_hz=np.array([float(mhz) * 1.0e6 for mhz in IMAGING_NODE_MHZ]),
            parallactic_angle_rad=np.array([0.0]),
            antenna_id=np.array([0], dtype=np.int32),
        ),
        calibration_state="casa_parang_true",
    )
    far = beam.evaluate(
        BeamCoordinates(
            l_rad=np.array([0.5]),
            m_rad=np.array([0.5]),
            frequency_hz=np.array([float(mhz) * 1.0e6 for mhz in IMAGING_NODE_MHZ]),
            parallactic_angle_rad=np.array([0.0]),
            antenna_id=np.array([0], dtype=np.int32),
        ),
        calibration_state="casa_parang_true",
    )
    support = {
        "on_axis": unsupported_direction_account(on_axis.valid),
        "far_field": unsupported_direction_account(far.valid),
    }
    if int(support["on_axis"]["n_unsupported"]):
        raise RuntimeError("survey smoke on-axis directions must be supported")
    if int(support["far_field"]["n_supported"]):
        raise RuntimeError("survey smoke must not silently support 0.5 rad offsets")
    print(
        "smoke preflight "
        f"pointings={len(subset)} rows={[int(block.uvw_m.shape[0]) for block in subset]} "
        f"n_freq={[int(block.frequency_hz.size) for block in subset]} "
        f"mhz={sorted(allowed)} n_sky={len(table.components)} "
        f"digest={digest[:12]} unsupported_far={support['far_field']['n_unsupported']}",
        flush=True,
    )
    config = protocol_config(
        steps=int(arguments.steps),
        max_rounds=0,
        max_splits_per_round=0,
        max_split_fraction=0.0,
        patience=8,
        sparsity_weight=0.0,
        strict_audit=False,
        operator_mode="explicit_jax",
        integration_max_depth=0,
    )
    config = replace(
        config,
        inference=replace(
            config.inference,
            batch_size_rows=min(32, arguments.max_rows_per_pointing),
            kkt_tolerance=1.0,
        ),
        kkt_max_batches=1,
    )
    directory = arguments.output / "smoke" / OPT_IN_SURVEY_BEAM
    directory.mkdir(parents=True, exist_ok=True)
    checkpoints: list[Path] = []

    def on_checkpoint(fit, state) -> None:
        _write_checkpoint(directory, fit.table, fit, state, topology_callbacks=len(checkpoints) + 1)
        checkpoints.append(directory / "checkpoint.json")

    result = reconstruct_voltage_stokes_i(
        table,
        subset,
        beam,
        antenna_position_m=antenna,
        calibration_state="casa_parang_true",
        config=config,
        beam_mode=OPT_IN_SURVEY_BEAM,
        pointing_ids=POINTINGS,
        on_checkpoint=on_checkpoint,
    )
    if not np.isfinite(result.fit.train_loss):
        raise RuntimeError("survey smoke produced a non-finite train loss")
    payload = json.loads((directory / "checkpoint.json").read_text(encoding="utf-8"))
    reloaded = sky_table_from_records(
        payload["components"],
        mosaic_phase_centre_rad=tuple(payload["mosaic_phase_centre_rad"]),
        source=payload.get("source", "phase6_checkpoint"),
    )
    merge_hysteresis_from_records(payload.get("hysteresis", {}))
    second = reconstruct_voltage_stokes_i(
        reloaded,
        subset,
        beam,
        antenna_position_m=antenna,
        calibration_state="casa_parang_true",
        config=replace(config, inference=replace(config.inference, steps=2)),
        beam_mode=OPT_IN_SURVEY_BEAM,
        pointing_ids=POINTINGS,
        skip_flux_optimize=True,
    )
    write_smoke_reconstruction_products(
        directory,
        result,
        subset,
        pointing_ids=POINTINGS,
        config={"steps": arguments.steps, "catalog_digest": digest},
        manifest={"beam_mode": OPT_IN_SURVEY_BEAM, "model_id": beam.model_id},
    )
    gradient = result.fit.gradient
    finite_gradient = bool(gradient is not None and np.all(np.isfinite(gradient)))
    operator_gate = None
    if not arguments.skip_operator_gate:
        operator_gate = compare_operator_modes(
            result.fit.flux,
            subset[0],
            result.fit.plan,
            beam,
            antenna_position_m=antenna,
            calibration_state="casa_parang_true",
            product="evla_c_imaging_smoke_gate",
        )
        if not bool(operator_gate.get("passed")):
            raise RuntimeError("survey smoke forward/adjoint gate failed")
        if int(operator_gate.get("peak_rss_bytes") or 0) > int(arguments.peak_rss_limit_bytes):
            raise RuntimeError("survey smoke exceeded the peak RSS bound")
    summary = {
        "beam_mode": OPT_IN_SURVEY_BEAM,
        "model_id": beam.model_id,
        "catalog_digest": digest,
        "experimental_diagonal_beam_imaging_run": True,
        "not_a_continuously_validated_c_band_beam": True,
        "pointings": list(POINTINGS),
        "n_rows": [int(block.uvw_m.shape[0]) for block in subset],
        "frequencies_hz": [block.frequency_hz.tolist() for block in subset],
        "train_loss": result.fit.train_loss,
        "reload_train_loss": second.fit.train_loss,
        "finite_train_loss": bool(np.isfinite(result.fit.train_loss)),
        "finite_gradient": finite_gradient,
        "unsupported_directions": support,
        "operator_gate": operator_gate,
        "checkpoint": str(directory / "checkpoint.json"),
    }
    (directory / "smoke_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
