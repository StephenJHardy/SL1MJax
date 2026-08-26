#!/usr/bin/env python3
"""Select and seal a 3C391 flagged-visibility recovery policy."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from compare_3c391_composite_existing_flags import _components_from_checkpoint
from fit_3c391_composite import _metrics

from sl1mjax.beam import VLABeamCatalog, VLAPrimaryBeam
from sl1mjax.composite import (
    MosaicSkyComponent,
    infer_mosaic_composite,
    mosaic_beam_sensitivity_weights,
)
from sl1mjax.data.canonical import VisibilityBlock, read_dataset
from sl1mjax.direct_operator import DirectDFTConfig
from sl1mjax.inference import InferenceConfig
from sl1mjax.split import interleaved_time_folds


def _load_folds(
    path: Path,
    blocks: tuple[VisibilityBlock, ...],
    *,
    fold_count: int = 5,
) -> tuple[tuple[np.ndarray, ...], ...]:
    with np.load(path) as stored:
        folds = tuple(
            tuple(
                np.asarray(stored[f"fold{fold}_C{pointing + 1}"], dtype=bool)
                for pointing in range(len(blocks))
            )
            for fold in range(fold_count)
        )
    for fold, masks in enumerate(folds):
        for pointing, (block, mask) in enumerate(zip(blocks, masks, strict=True), start=1):
            if mask.shape != block.shape or np.any(mask & ~block.active):
                raise ValueError(f"fold {fold} C{pointing} is incompatible with its fixture")
    for pointing, block in enumerate(blocks):
        combined = np.logical_or.reduce([fold[pointing] for fold in folds])
        if np.any(combined != block.active):
            raise ValueError(f"folds do not partition every active C{pointing + 1} sample")
    return folds


def _or_folds(
    folds: tuple[tuple[np.ndarray, ...], ...],
    selected: tuple[int, ...],
) -> tuple[np.ndarray, ...]:
    if not selected:
        raise ValueError("at least one fold must be selected")
    return tuple(
        np.logical_or.reduce([folds[fold][pointing] for fold in selected])
        for pointing in range(len(folds[0]))
    )


def _common_active_masks(
    policy_blocks: tuple[VisibilityBlock, ...],
    reference_blocks: tuple[VisibilityBlock, ...],
    reference_masks: tuple[np.ndarray, ...],
) -> tuple[np.ndarray, ...]:
    """Embed untouched active-reference masks at the head of policy blocks."""

    if not (len(policy_blocks) == len(reference_blocks) == len(reference_masks)):
        raise ValueError("policy, reference, and masks must have equal block counts")
    output = []
    for policy, reference, mask in zip(
        policy_blocks, reference_blocks, reference_masks, strict=True
    ):
        if policy.shape[0] < reference.shape[0] or mask.shape != reference.shape:
            raise ValueError("policy blocks must begin with their complete reference block")
        if not np.allclose(policy.time_s[: reference.shape[0]], reference.time_s):
            raise ValueError("policy block does not preserve reference row order")
        embedded = np.zeros(policy.shape, dtype=bool)
        embedded[: reference.shape[0]] = mask
        output.append(embedded)
    return tuple(output)


def _components_with_sensitivity(
    components: tuple[MosaicSkyComponent, ...],
    blocks: tuple[VisibilityBlock, ...],
    train_masks: tuple[np.ndarray, ...],
    mosaic_phase_centre: tuple[float, float],
    beam: VLAPrimaryBeam,
) -> tuple[MosaicSkyComponent, ...]:
    weights = mosaic_beam_sensitivity_weights(
        blocks,
        components,
        train_masks,
        mosaic_phase_centre,
        primary_beam=beam,
    )
    return tuple(
        replace(component, sparsity_weights=value)
        for component, value in zip(components, weights, strict=True)
    )


def _initial_components(
    components: tuple[MosaicSkyComponent, ...],
    initialization: str,
) -> tuple[MosaicSkyComponent, ...]:
    if initialization == "checkpoint":
        return components
    if initialization == "zero":
        return tuple(
            replace(component, flux=np.zeros_like(component.flux))
            for component in components
        )
    raise ValueError(f"unsupported initialization {initialization!r}")


def _save_result(path: Path, result: Any) -> None:
    np.savez(
        path,
        **{
            **{
                f"flux_{component.name}": np.asarray(component.flux)
                for component in result.components
            },
            **{
                f"prediction_C{index + 1}": prediction
                for index, prediction in enumerate(result.predictions)
            },
        },
    )


def _selection(arguments: argparse.Namespace) -> dict[str, Any]:
    build = json.loads((arguments.policy_directory / "summary.json").read_text())
    policies = tuple(build["policies"])
    reference = read_dataset(arguments.reference_fixture).blocks
    reference_folds = interleaved_time_folds(
        reference, bin_seconds=arguments.time_bin_seconds
    )
    reference_validation = reference_folds[3]
    protocol = json.loads(arguments.sky_protocol.read_text())
    phase_centre = reference[0].phase_centre_rad
    beam = VLAPrimaryBeam(
        kind="airy",
        catalog=replace(
            VLABeamCatalog(),
            airy_max_radius_rad_at_1ghz=np.deg2rad(
                float(protocol["airy_max_radius_deg_at_1ghz"])
            ),
        ),
    )
    base_components = _initial_components(
        _components_from_checkpoint(arguments.initial_checkpoint, protocol, phase_centre),
        arguments.initialization,
    )
    direct = DirectDFTConfig(
        visibility_chunk_size=arguments.visibility_tile_size,
        pixel_chunk_size=arguments.pixel_tile_size,
        precision=arguments.precision,
    )
    candidates = []
    for policy in policies:
        fixture = arguments.policy_directory / f"{policy}.zarr"
        blocks = read_dataset(fixture).blocks
        folds = _load_folds(arguments.policy_directory / f"{policy}_folds.npz", blocks)
        train = _or_folds(folds, (0, 1, 2))
        validation = _common_active_masks(blocks, reference, reference_validation)
        result_path = arguments.output / f"selection_{policy}.npz"
        payload_path = arguments.output / f"selection_{policy}.json"
        if result_path.exists() and payload_path.exists() and not arguments.no_resume:
            payload = json.loads(payload_path.read_text())
            print(f"{policy}: resumed", flush=True)
        else:
            print(f"{policy}: fitting fixed composite sky", flush=True)
            components = _components_with_sensitivity(
                base_components, blocks, train, phase_centre, beam
            )
            result = infer_mosaic_composite(
                blocks,
                components,
                train,
                phase_centre,
                InferenceConfig(
                    solver="fista",
                    steps=arguments.steps,
                    sparsity_weight=arguments.lambda_l1,
                    validation_interval=arguments.validation_interval,
                    kkt_tolerance=arguments.kkt_tolerance,
                    operator_mode="explicit",
                    direct_dft=direct,
                ),
                holdout_masks=validation,
                primary_beam=beam,
            )
            payload = {
                "policy": policy,
                "steps": result.steps,
                "best_step": result.best_step,
                "converged": result.converged,
                "kkt_residual": result.kkt_residual,
                "train": _metrics(blocks, result.predictions, train),
                "validation": _metrics(blocks, result.predictions, validation),
                "component_flux_jy": {
                    component.name: float(np.sum(component.flux))
                    for component in result.components
                },
            }
            _save_result(result_path, result)
            payload_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        candidates.append(payload)
        print(
            f"{policy}: validation MSE={payload['validation']['weighted_complex_mse']:.8g}",
            flush=True,
        )
    winner = min(candidates, key=lambda item: item["validation"]["weighted_complex_mse"])
    return {
        "schema_version": 1,
        "protocol": {
            "policy_directory": str(arguments.policy_directory),
            "reference_fixture": str(arguments.reference_fixture),
            "sky_protocol": str(arguments.sky_protocol),
            "initial_checkpoint": str(arguments.initial_checkpoint),
            "initialization": arguments.initialization,
            "training_folds": [0, 1, 2],
            "selection_fold": 3,
            "sealed_fold": 4,
            "selection_evaluation_samples": "originally active samples only",
            "lambda_l1": arguments.lambda_l1,
            "steps": arguments.steps,
            "validation_interval": arguments.validation_interval,
            "sealed_opened": False,
        },
        "candidates": candidates,
        "selected_policy": winner["policy"],
    }


def _sealed(arguments: argparse.Namespace, summary: dict[str, Any]) -> dict[str, Any]:
    if summary["protocol"].get("sealed_opened"):
        raise ValueError("sealed fold has already been opened")
    selected = str(summary["selected_policy"])
    controls = tuple(dict.fromkeys(("active_only", selected)))
    reference = read_dataset(arguments.reference_fixture).blocks
    reference_folds = interleaved_time_folds(
        reference, bin_seconds=arguments.time_bin_seconds
    )
    reference_test = reference_folds[4]
    protocol = json.loads(arguments.sky_protocol.read_text())
    phase_centre = reference[0].phase_centre_rad
    beam = VLAPrimaryBeam(
        kind="airy",
        catalog=replace(
            VLABeamCatalog(),
            airy_max_radius_rad_at_1ghz=np.deg2rad(
                float(protocol["airy_max_radius_deg_at_1ghz"])
            ),
        ),
    )
    initial = _initial_components(
        _components_from_checkpoint(arguments.initial_checkpoint, protocol, phase_centre),
        str(summary["protocol"]["initialization"]),
    )
    direct = DirectDFTConfig(
        visibility_chunk_size=arguments.visibility_tile_size,
        pixel_chunk_size=arguments.pixel_tile_size,
        precision=arguments.precision,
    )
    selection_by_policy = {item["policy"]: item for item in summary["candidates"]}
    sealed_results = {}
    for policy in controls:
        blocks = read_dataset(arguments.policy_directory / f"{policy}.zarr").blocks
        folds = _load_folds(arguments.policy_directory / f"{policy}_folds.npz", blocks)
        train = _or_folds(folds, (0, 1, 2, 3))
        test = _common_active_masks(blocks, reference, reference_test)
        selected_steps = max(
            arguments.validation_interval,
            int(selection_by_policy[policy]["best_step"]) + 1,
        )
        components = _components_with_sensitivity(initial, blocks, train, phase_centre, beam)
        print(f"sealed {policy}: refitting for {selected_steps} fixed steps", flush=True)
        result = infer_mosaic_composite(
            blocks,
            components,
            train,
            phase_centre,
            InferenceConfig(
                solver="fista",
                steps=selected_steps,
                sparsity_weight=arguments.lambda_l1,
                validation_interval=arguments.validation_interval,
                kkt_tolerance=arguments.kkt_tolerance,
                operator_mode="explicit",
                direct_dft=direct,
            ),
            primary_beam=beam,
        )
        _save_result(arguments.output / f"sealed_{policy}.npz", result)
        sealed_results[policy] = {
            "steps": result.steps,
            "kkt_residual": result.kkt_residual,
            "train": _metrics(blocks, result.predictions, train),
            "test": _metrics(blocks, result.predictions, test),
            "component_flux_jy": {
                component.name: float(np.sum(component.flux))
                for component in result.components
            },
        }
    summary["sealed"] = sealed_results
    summary["protocol"]["sealed_opened"] = True
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy-directory",
        type=Path,
        default=Path("outputs/3c391_recovery_policies"),
    )
    parser.add_argument(
        "--reference-fixture",
        type=Path,
        default=Path("outputs/3c391_gain_time_model_sweep/selected_native_fixture.zarr"),
    )
    parser.add_argument(
        "--sky-protocol",
        type=Path,
        default=Path("outputs/3c391_composite_catalogue_stage3/protocol.json"),
    )
    parser.add_argument(
        "--initial-checkpoint",
        type=Path,
        default=Path("outputs/3c391_composite_catalogue_stage3/full_lambda_0.0003.npz"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/3c391_recovery_policy_fit")
    )
    parser.add_argument("--time-bin-seconds", type=float, default=60.0)
    parser.add_argument("--lambda-l1", type=float, default=3e-4)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--validation-interval", type=int, default=10)
    parser.add_argument("--kkt-tolerance", type=float, default=3e-5)
    parser.add_argument("--visibility-tile-size", type=int, default=256)
    parser.add_argument("--pixel-tile-size", type=int, default=1024)
    parser.add_argument("--precision", choices=("float32", "float64"), default="float32")
    parser.add_argument(
        "--initialization",
        choices=("zero", "checkpoint"),
        default="zero",
        help="Zero is fold-3 blind; checkpoint is diagnostic only because it used fold 3.",
    )
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--open-sealed", action="store_true")
    arguments = parser.parse_args()

    arguments.output.mkdir(parents=True, exist_ok=True)
    summary_path = arguments.output / "summary.json"
    if arguments.open_sealed:
        if not summary_path.exists():
            raise ValueError("selection must finish before opening the sealed fold")
        summary = _sealed(arguments, json.loads(summary_path.read_text()))
    else:
        summary = _selection(arguments)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "selected_policy": summary["selected_policy"],
                "sealed_opened": summary["protocol"]["sealed_opened"],
                "sealed": summary.get("sealed"),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
