#!/usr/bin/env python3
"""Phase 6 Stokes-I reconstructions on 3C391 with the sealed fold closed.

Stages are an explicit ladder with review points, not one monolithic job:

1. ``commissioning`` — C1, all production beams, one topology round.
2. ``commissioning-c4`` — C4 fixed-topology comparison (no splits or guard).
3. Review C1/C4 products before seven pointings.
4. ``baseline`` — seven pointings, fixed topology, all production beams.
5. ``full`` — one topology round for the beams passed in ``--beams``.
   Further rounds are started only after a previous round accepts a change.

``--stage all`` runs step 1 only, then stops. Experimental full Jones is
excluded unless ``--allow-diagnostic-full-jones`` is set. That path is a
diagnostic reconstruction, not a production freeze. Fold 4 is never evaluated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

import numpy as np

from sl1mjax.beam_aware_imaging import sky_table_from_records, sky_table_to_records
from sl1mjax.beam_operator import BeamOperatorConfig
from sl1mjax.data.canonical import VisibilityBlock, read_dataset
from sl1mjax.inference import InferenceConfig
from sl1mjax.integration_planner import IntegrationTolerance
from sl1mjax.phase6_protocol import (
    FIELD_EXPANSION,
    accept_inactive_guard,
    phase6_folds,
    write_reconstruction_products,
)
from sl1mjax.voltage_reconstruction import (
    DIAGNOSTIC_STOKES_I_BEAMS,
    PRODUCTION_STOKES_I_BEAMS,
    VoltageReconstructionConfig,
    merge_hysteresis_from_records,
    merge_hysteresis_to_records,
    reconstruct_voltage_stokes_i,
    stokes_i_beam,
)
from sl1mjax.wide_field_sky import (
    CENTRAL_ROOT_SIZE,
    catalogue_components_from_pinned_json,
    central_pixel_size_rad,
    phase5_starting_table,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NATIVE = Path("outputs/3c391_native_averaging_ablation")
DEFAULT_CATALOGUE = ROOT / "config" / "3c391_radio_guard_catalog.json"
DEFAULT_POL_GOLDEN = ROOT / "tests" / "fixtures" / "3c391_polarization_golden.npz"
DEFAULT_OUTPUT = Path("outputs/3c391_phase6_20260830")
POINTINGS = ("C1", "C2", "C3", "C4", "C5", "C6", "C7")


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


def _source_manifest(path: Path | None = None) -> dict[str, Any]:
    if path is not None and path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    env_path = os.environ.get("SL1MJAX_SOURCE_MANIFEST")
    if env_path and Path(env_path).is_file():
        return json.loads(Path(env_path).read_text(encoding="utf-8"))
    git_roots = [ROOT]
    source_root = os.environ.get("SL1MJAX_PHASE6_SRC")
    if source_root:
        git_roots.append(Path(source_root))
    commit = "unknown"
    diff = ""
    for git_root in git_roots:
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=git_root, text=True
            ).strip()
            diff = subprocess.check_output(["git", "diff", "HEAD"], cwd=git_root, text=True)
            break
        except subprocess.CalledProcessError, FileNotFoundError:
            continue
    lock = ROOT / "uv.lock"
    if not lock.is_file() and source_root:
        lock = Path(source_root) / "uv.lock"
    return {
        "commit": commit,
        "uncommitted_diff_sha256": hashlib.sha256(diff.encode()).hexdigest(),
        "lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest() if lock.is_file() else None,
    }


def load_pointing_blocks(
    native_root: Path, pointings: tuple[str, ...]
) -> tuple[VisibilityBlock, ...]:
    blocks = []
    for pointing in pointings:
        path = native_root / f"native_{pointing}.zarr"
        if not path.exists():
            raise FileNotFoundError(f"missing native fixture {path}")
        loaded = read_dataset(path).blocks
        if len(loaded) != 1:
            raise ValueError(f"{path} must contain exactly one visibility block")
        blocks.append(loaded[0])
    return tuple(blocks)


def load_antenna_positions(path: Path, antenna_count: int) -> np.ndarray:
    scripts_directory = str(ROOT / "scripts")
    if scripts_directory not in sys.path:
        sys.path.insert(0, scripts_directory)
    from diagnose_3c391_voltage_beam_transfer import load_antenna_positions as load_positions

    return load_positions(
        polarization_golden=path,
        measurement_set=None,
        antenna_count=antenna_count,
    )


def protocol_config(
    *,
    steps: int,
    max_rounds: int,
    max_splits_per_round: int,
    max_split_fraction: float,
    patience: int,
    sparsity_weight: float,
    strict_audit: bool,
    operator_mode: Literal["vjp", "explicit_jax"] = "vjp",
    integration_max_depth: int = 3,
    learning_rate: float = 0.05,
    validation_interval: int = 5,
) -> VoltageReconstructionConfig:
    if operator_mode not in {"vjp", "explicit_jax"}:
        raise ValueError("operator_mode must be 'vjp' or 'explicit_jax'")
    return VoltageReconstructionConfig(
        root_size=CENTRAL_ROOT_SIZE,
        root_pixel_size_rad=central_pixel_size_rad(),
        inference=InferenceConfig(
            solver="proximal_sgd",
            batch_grouping="times",
            steps=steps,
            learning_rate=learning_rate,
            patience=patience,
            validation_interval=validation_interval,
            sparsity_weight=sparsity_weight,
            kkt_tolerance=1e-5,
            batch_size_rows=64,
        ),
        kkt_max_batches=32,
        operator_mode=operator_mode,
        predict_batch_size_rows=256,
        tolerance=IntegrationTolerance(max_depth=int(integration_max_depth)),
        operator=BeamOperatorConfig(
            max_timestep_jones_bytes=256 * 1024**2,
            visibility_chunk_size=64,
            pixel_chunk_size=128,
        ),
        leaf_penalty=0.0,
        max_rounds=max_rounds,
        max_depth=2,
        max_splits_per_round=max_splits_per_round,
        max_split_fraction=max_split_fraction,
        strict_audit=strict_audit,
        screen_parent_limit=max(64, 4 * max_splits_per_round),
    )


def checkpoint_resume(
    topology_callbacks: int, max_rounds: int
) -> tuple[bool, int, str]:
    """How to continue from a written checkpoint.

    Callback 1 is the initial flux fit. Callback 2+ is after a topology
    round. Keep ``max_rounds`` so a pre-topology checkpoint still splits.
    """

    callbacks = int(topology_callbacks)
    if callbacks <= 0:
        return False, int(max_rounds), "warm-start"
    if callbacks > 1 and max_rounds > 0:
        return True, 0, "post-topology"
    return True, int(max_rounds), "post-SGD"


def _write_checkpoint(
    directory: Path, table, fit, hysteresis, *, topology_callbacks: int = 0
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "mosaic_phase_centre_rad": list(table.mosaic_phase_centre_rad),
        "source": table.source,
        "components": sky_table_to_records(table),
        "converged": fit.converged,
        "kkt_residual": fit.kkt_residual,
        "stationarity_history": list(fit.stationarity_history),
        "train_loss": fit.train_loss,
        "holdout_loss": fit.holdout_loss,
        "optimization_curve": {
            "steps": list(getattr(fit, "curve_steps", ())),
            "train_loss": list(fit.objective_history),
            "holdout_loss": list(fit.holdout_history),
        },
        "hysteresis": merge_hysteresis_to_records(hysteresis),
        "topology_callbacks": int(topology_callbacks),
    }
    (directory / "checkpoint.json").write_text(
        json.dumps(_to_json(checkpoint), indent=2), encoding="utf-8"
    )


def run_one(
    *,
    stage: str,
    beam_mode: str,
    table,
    blocks,
    folds,
    pointing_ids: tuple[str, ...],
    antenna_position_m: np.ndarray,
    config: VoltageReconstructionConfig,
    output: Path,
    source: dict[str, Any],
    hysteresis=None,
    allow_diagnostic_full_jones: bool = False,
) -> dict[str, Any]:
    if beam_mode == "full_jones":
        if not allow_diagnostic_full_jones:
            raise ValueError(f"{beam_mode} is not a production Stokes-I candidate")
    elif beam_mode not in PRODUCTION_STOKES_I_BEAMS:
        raise ValueError(f"{beam_mode} is not a production Stokes-I candidate")
    directory = output / stage / beam_mode
    checkpoint_path = directory / "checkpoint.json"
    if (directory / "summary.json").is_file() and checkpoint_path.is_file():
        print(f"=== skip completed {stage} {beam_mode} ===", flush=True)
        return json.loads(checkpoint_path.read_text(encoding="utf-8"))
    start_table = table
    start_hysteresis = hysteresis
    start_config = _config_for_beam(config, beam_mode)
    skip_flux_optimize = False
    if checkpoint_path.is_file() and not (directory / "summary.json").is_file():
        payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        start_table = sky_table_from_records(
            payload["components"],
            mosaic_phase_centre_rad=tuple(payload["mosaic_phase_centre_rad"]),
            source=payload.get("source", "phase6_checkpoint"),
        )
        start_hysteresis = merge_hysteresis_from_records(payload.get("hysteresis", {}))
        callbacks = int(payload.get("topology_callbacks", 0))
        skip_flux_optimize, resumed_rounds, resume_kind = checkpoint_resume(
            callbacks, config.max_rounds
        )
        if resumed_rounds != start_config.max_rounds:
            start_config = replace_rounds(start_config, resumed_rounds)
        print(f"=== resume {resume_kind} {stage} {beam_mode} ===", flush=True)
    beam = stokes_i_beam(
        beam_mode, allow_unfrozen_full_jones=allow_diagnostic_full_jones
    )
    print(
        f"=== {stage} {beam_mode} {pointing_ids} "
        f"batch_rows={start_config.inference.batch_size_rows} "
        f"pixel_chunk={start_config.operator.pixel_chunk_size} ===",
        flush=True,
    )
    checkpoint_callbacks = 0

    def on_checkpoint(fit, state) -> None:
        nonlocal checkpoint_callbacks
        checkpoint_callbacks += 1
        _write_checkpoint(
            directory,
            fit.table,
            fit,
            state,
            topology_callbacks=checkpoint_callbacks,
        )

    result = reconstruct_voltage_stokes_i(
        start_table,
        blocks,
        beam,
        antenna_position_m=antenna_position_m,
        calibration_state="casa_parang_true",
        train_masks=folds.train,
        holdout_masks=folds.holdout,
        config=start_config,
        beam_mode=beam_mode,
        pointing_ids=pointing_ids,
        hysteresis=start_hysteresis,
        on_checkpoint=on_checkpoint,
        skip_flux_optimize=skip_flux_optimize,
    )
    if not np.isfinite(result.fit.train_loss):
        raise RuntimeError(f"{stage} {beam_mode} train loss is not finite")
    guard = None
    if config.max_rounds > 0:

        def _refit(proposed):
            return reconstruct_voltage_stokes_i(
                proposed,
                blocks,
                beam,
                antenna_position_m=antenna_position_m,
                calibration_state="casa_parang_true",
                train_masks=folds.train,
                holdout_masks=folds.holdout,
                config=replace_rounds(start_config, 0),
                beam_mode=beam_mode,
                pointing_ids=pointing_ids,
            ).fit

        accepted_fit, guard = accept_inactive_guard(
            result.fit,
            blocks,
            beam,
            antenna_position_m=antenna_position_m,
            calibration_state="casa_parang_true",
            train_masks=folds.train,
            holdout_masks=folds.holdout,
            refit=_refit,
        )
        if guard.status == FIELD_EXPANSION:
            raise RuntimeError("outer guard requests a larger refinable field")
        result = replace(result, table=accepted_fit.table, fit=accepted_fit)
    write_reconstruction_products(
        directory,
        result,
        blocks,
        pointing_ids=pointing_ids,
        antenna_position_m=antenna_position_m,
        folds=folds,
        config={
            "stage": stage,
            "beam_mode": beam_mode,
            "steps": config.inference.steps,
            "max_rounds": config.max_rounds,
            "max_splits_per_round": config.max_splits_per_round,
            "max_split_fraction": config.max_split_fraction,
            "sky_max_depth": config.max_depth,
            "integration_max_depth": config.tolerance.max_depth,
            "sparsity_weight": config.inference.sparsity_weight,
            "leaf_penalty": config.leaf_penalty,
            "strict_audit": config.strict_audit,
            "kkt_max_batches": config.kkt_max_batches,
            "operator_mode": config.operator_mode,
            "predict_batch_size_rows": config.predict_batch_size_rows,
            "command": sys.argv,
            "diagnostic_full_jones": bool(allow_diagnostic_full_jones),
            "do_not_freeze_full_jones": beam_mode == "full_jones",
        },
        manifest={
            **source,
            "pointing_ids": list(pointing_ids),
            "fold_bin_seconds": folds.bin_seconds,
            "train_folds": list(folds.train_folds),
            "holdout_fold": folds.holdout_fold,
            "sealed_fold": folds.sealed_fold,
        },
        guard=guard,
    )
    _write_checkpoint(
        directory,
        result.table,
        result.fit,
        result.hysteresis,
        topology_callbacks=checkpoint_callbacks,
    )
    print(
        json.dumps(
            {
                "stage": stage,
                "beam": beam_mode,
                "train_loss": result.fit.train_loss,
                "holdout_loss": result.fit.holdout_loss,
                "kkt": result.fit.kkt_residual,
                "converged": result.fit.converged,
                "steps": result.fit.steps,
                "audit": bool(result.audit.under_resolved),
            }
        ),
        flush=True,
    )
    return json.loads((directory / "checkpoint.json").read_text(encoding="utf-8"))


def replace_rounds(
    config: VoltageReconstructionConfig, max_rounds: int
) -> VoltageReconstructionConfig:
    return replace(config, max_rounds=max_rounds)


def _config_for_beam(
    config: VoltageReconstructionConfig, beam_mode: str
) -> VoltageReconstructionConfig:
    """CASSBEAM copolar VJPs rematerialize more than the scalar Airy path."""

    if beam_mode not in {"diagonal_copolar", "full_jones"}:
        return config
    # VJP tiles must stay small; the explicit adjoint streams parents.
    sgd_rows = 64 if config.operator_mode == "explicit_jax" else 32
    return replace(
        config,
        inference=replace(config.inference, batch_size_rows=sgd_rows),
        operator=replace(
            config.operator,
            pixel_chunk_size=512,
            visibility_chunk_size=32,
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-root", type=Path, default=DEFAULT_NATIVE)
    parser.add_argument("--catalogue", type=Path, default=DEFAULT_CATALOGUE)
    parser.add_argument("--polarization-golden", type=Path, default=DEFAULT_POL_GOLDEN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-manifest", type=Path, default=None)
    parser.add_argument(
        "--stage",
        choices=("commissioning", "commissioning-c4", "baseline", "full", "all"),
        default="commissioning",
        help="Ladder step. 'all' is C1 commissioning only; later steps are explicit.",
    )
    parser.add_argument("--beams", default=",".join(PRODUCTION_STOKES_I_BEAMS))
    parser.add_argument(
        "--allow-diagnostic-full-jones",
        action="store_true",
        help="Permit experimental full Jones as a labelled diagnostic, not a freeze.",
    )
    parser.add_argument(
        "--operator-mode",
        choices=("vjp", "explicit_jax"),
        default="vjp",
        help="Use explicit_jax after C1/C4 review; vjp remains the live-job default.",
    )
    parser.add_argument(
        "--integration-max-depth",
        type=int,
        default=3,
        help="Planner integration cap. Raise after a depth ablation, not sky max_depth.",
    )
    arguments = parser.parse_args()
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    beams = tuple(item.strip() for item in arguments.beams.split(",") if item.strip())
    allowed_beams = PRODUCTION_STOKES_I_BEAMS
    if arguments.allow_diagnostic_full_jones:
        allowed_beams = PRODUCTION_STOKES_I_BEAMS + DIAGNOSTIC_STOKES_I_BEAMS
    if any(item not in allowed_beams for item in beams):
        parser.error(f"beams must be a subset of {allowed_beams}")
    if "full_jones" in beams and arguments.stage == "full":
        parser.error("diagnostic full Jones must not start a topology round")
    source = _source_manifest(arguments.source_manifest)
    arguments.output.mkdir(parents=True, exist_ok=True)
    (arguments.output / "source.json").write_text(json.dumps(source, indent=2), encoding="utf-8")

    def starting_table(blocks: tuple[VisibilityBlock, ...]):
        centre = blocks[0].phase_centre_rad
        catalogue = catalogue_components_from_pinned_json(arguments.catalogue)
        return phase5_starting_table(mosaic_phase_centre_rad=centre, catalogue=catalogue)

    if arguments.stage == "all":
        print(
            "=== --stage all runs C1 commissioning only; later ladder steps are explicit ===",
            flush=True,
        )
    stages = (
        ("commissioning",)
        if arguments.stage == "all"
        else (arguments.stage,)
    )
    for stage in stages:
        if stage == "commissioning":
            blocks = load_pointing_blocks(arguments.native_root, ("C1",))
            antenna = load_antenna_positions(arguments.polarization_golden, blocks[0].antenna_count)
            folds = phase6_folds(blocks)
            table = starting_table(blocks)
            config = protocol_config(
                steps=80,
                max_rounds=1,
                max_splits_per_round=8,
                max_split_fraction=0.05,
                patience=20,
                sparsity_weight=3e-4,
                strict_audit=False,
                operator_mode=arguments.operator_mode,
                integration_max_depth=arguments.integration_max_depth,
            )
            for beam in beams:
                run_one(
                    stage=stage,
                    beam_mode=beam,
                    table=table,
                    blocks=blocks,
                    folds=folds,
                    pointing_ids=("C1",),
                    antenna_position_m=antenna,
                    config=config,
                    output=arguments.output,
                    source=source,
                    allow_diagnostic_full_jones=arguments.allow_diagnostic_full_jones,
                )
        elif stage == "commissioning-c4":
            blocks = load_pointing_blocks(arguments.native_root, ("C4",))
            antenna = load_antenna_positions(arguments.polarization_golden, blocks[0].antenna_count)
            folds = phase6_folds(blocks)
            table = starting_table(blocks)
            config = protocol_config(
                steps=80,
                max_rounds=0,
                max_splits_per_round=8,
                max_split_fraction=0.05,
                patience=20,
                sparsity_weight=3e-4,
                strict_audit=False,
                operator_mode=arguments.operator_mode,
                integration_max_depth=arguments.integration_max_depth,
            )
            for beam in beams:
                run_one(
                    stage=stage,
                    beam_mode=beam,
                    table=table,
                    blocks=blocks,
                    folds=folds,
                    pointing_ids=("C4",),
                    antenna_position_m=antenna,
                    config=config,
                    output=arguments.output,
                    source=source,
                    allow_diagnostic_full_jones=arguments.allow_diagnostic_full_jones,
                )
        elif stage == "baseline":
            blocks = load_pointing_blocks(arguments.native_root, POINTINGS)
            antenna = load_antenna_positions(arguments.polarization_golden, blocks[0].antenna_count)
            folds = phase6_folds(blocks)
            table = starting_table(blocks)
            config = protocol_config(
                steps=200,
                max_rounds=0,
                max_splits_per_round=32,
                max_split_fraction=0.05,
                patience=20,
                sparsity_weight=3e-4,
                strict_audit=False,
                operator_mode=arguments.operator_mode,
                integration_max_depth=arguments.integration_max_depth,
            )
            for beam in beams:
                run_one(
                    stage=stage,
                    beam_mode=beam,
                    table=table,
                    blocks=blocks,
                    folds=folds,
                    pointing_ids=POINTINGS,
                    antenna_position_m=antenna,
                    config=config,
                    output=arguments.output,
                    source=source,
                    allow_diagnostic_full_jones=arguments.allow_diagnostic_full_jones,
                )
        else:
            blocks = load_pointing_blocks(arguments.native_root, POINTINGS)
            antenna = load_antenna_positions(arguments.polarization_golden, blocks[0].antenna_count)
            folds = phase6_folds(blocks)
            shared = starting_table(blocks)
            config = protocol_config(
                steps=200,
                max_rounds=1,
                max_splits_per_round=32,
                max_split_fraction=0.05,
                patience=20,
                sparsity_weight=3e-4,
                strict_audit=False,
                operator_mode=arguments.operator_mode,
                integration_max_depth=arguments.integration_max_depth,
            )
            for beam in beams:
                run_one(
                    stage="full_round1",
                    beam_mode=beam,
                    table=shared,
                    blocks=blocks,
                    folds=folds,
                    pointing_ids=POINTINGS,
                    antenna_position_m=antenna,
                    config=config,
                    output=arguments.output,
                    source=source,
                    allow_diagnostic_full_jones=arguments.allow_diagnostic_full_jones,
                )
            print(
                "=== one topology round written; "
                "start another only if a beam accepted a change ===",
                flush=True,
            )
    if arguments.stage in {"all", "commissioning"}:
        print(
            "=== C1 commissioning step finished. Next: review, then "
            "--stage commissioning-c4 (fixed topology). Then --stage baseline, "
            "then --stage full --beams <control>,<selected> ===",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
