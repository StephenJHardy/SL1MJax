#!/usr/bin/env python3
"""Fit a shared-support circular contrast on a frozen 3C391 Stokes-I sky.

The sky remains one non-negative I.  Parallel hands are I_RR = I(1+v) and
I_LL = I(1-v) with |v| <= 1.  This diagnostic keeps I frozen, fits a global
v on discovery baselines, and scores that frozen v on held-out baselines.

A balanced visibility-scale split RR→(1+δ)M, LL→(1-δ)M is the same model as
sky v=δ, including on held-out baselines.  Separate RR and LL scales are
reported only to flag *unbalanced* errors (one hand low, both low, …).
When the two implied contrasts agree, the data do not distinguish sky V
from a common-mode receptor-gain ratio.  An external unpolarised calibrator
is required for that distinction.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from sl1mjax.beam import VLABeamCatalog, VLAPrimaryBeam
from sl1mjax.circular_contrast import (
    apply_global_circular_contrast,
    correlation_residual_power,
    fit_global_circular_contrast,
)
from sl1mjax.composite import MosaicSkyComponent, infer_mosaic_composite
from sl1mjax.data.canonical import VisibilityBlock, read_dataset
from sl1mjax.direct_operator import DirectDFTConfig
from sl1mjax.inference import InferenceConfig
from sl1mjax.polarization import Correlation
from sl1mjax.residual_models import (
    fit_real_linear_statistics,
    real_linear_statistics,
    residual_power_for_coefficients,
)
from sl1mjax.sky_recovery import split_search_baselines


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


def _weighted_power(
    residual: np.ndarray, weight: np.ndarray, mask: np.ndarray
) -> float:
    selected = mask & np.isfinite(weight) & (weight > 0)
    if not np.any(selected):
        return float("nan")
    values = residual[selected]
    finite = np.isfinite(values.real) & np.isfinite(values.imag)
    if not np.any(finite):
        return float("nan")
    return float(np.sum(weight[selected][finite] * np.abs(values[finite]) ** 2))


def _hand_index(correlations: tuple[Correlation, ...], hand: Correlation) -> int:
    try:
        return correlations.index(hand)
    except ValueError as error:
        raise ValueError(f"block is missing {hand.value}") from error


def _hand_mask(
    block: VisibilityBlock, base: np.ndarray, hand: Correlation
) -> np.ndarray:
    mask = np.asarray(base, dtype=bool).copy()
    for index, correlation in enumerate(block.correlations):
        if correlation is not hand:
            mask[..., index] = False
    return mask & block.active


def _hand_scale_response(
    model: np.ndarray, correlations: tuple[Correlation, ...], hand: Correlation
) -> np.ndarray:
    index = _hand_index(correlations, hand)
    response = np.zeros(model.shape, dtype=model.dtype)
    response[..., index] = model[..., index]
    return response


def parallel_hand_scale_residual_power(
    block: VisibilityBlock,
    model: np.ndarray,
    mask: np.ndarray,
    hand: Correlation,
    scale: float,
) -> float:
    """Hand-only residual power of vis ≈ scale * model (scale is not refit)."""

    residual = block.visibility - model
    statistics = real_linear_statistics(
        residual,
        block.weight,
        _hand_mask(block, mask, hand),
        _hand_scale_response(model, block.correlations, hand)[..., None],
    )
    return residual_power_for_coefficients(statistics, np.asarray([scale - 1.0]))


def fit_parallel_hand_scale(
    block: VisibilityBlock,
    model: np.ndarray,
    mask: np.ndarray,
    hand: Correlation,
) -> tuple[float, float]:
    """Return (scale, residual_power) for vis ≈ scale * model on one hand."""

    residual = block.visibility - model
    statistics = real_linear_statistics(
        residual,
        block.weight,
        _hand_mask(block, mask, hand),
        _hand_scale_response(model, block.correlations, hand)[..., None],
    )
    # vis = s * model = model + (s-1) * model, so the fitted coefficient is s-1.
    fit = fit_real_linear_statistics(statistics, ridge_fraction=0.0)
    delta = float(fit.coefficients[0])
    scale = 1.0 + delta
    power = residual_power_for_coefficients(statistics, np.asarray([delta]))
    return scale, power


def diagnose_circular_contrast(
    block: VisibilityBlock,
    model: np.ndarray,
    *,
    seed: int,
    selection_fraction: float,
    evaluation_fraction: float,
) -> dict[str, Any]:
    """Fit global v and per-hand scales on disjoint baseline cohorts."""

    if model.shape != block.shape:
        raise ValueError("model visibility must match the visibility block")
    if Correlation.RR not in block.correlations or Correlation.LL not in block.correlations:
        raise ValueError("circular-contrast diagnosis requires RR and LL")
    split = split_search_baselines(
        block,
        selection_fraction=selection_fraction,
        evaluation_fraction=evaluation_fraction,
        seed=seed,
    )
    residual = block.visibility - model
    contrast, fit, _statistics = fit_global_circular_contrast(
        residual,
        block.weight,
        split.discovery_mask,
        model,
        block.correlations,
    )
    corrected = apply_global_circular_contrast(model, block.correlations, contrast)
    corrected_residual = block.visibility - corrected
    rr_scale, rr_discovery_power = fit_parallel_hand_scale(
        block, model, split.discovery_mask, Correlation.RR
    )
    ll_scale, ll_discovery_power = fit_parallel_hand_scale(
        block, model, split.discovery_mask, Correlation.LL
    )
    rr_implied = rr_scale - 1.0
    ll_implied = 1.0 - ll_scale
    rr_eval = parallel_hand_scale_residual_power(
        block, model, split.evaluation_mask, Correlation.RR, rr_scale
    )
    ll_eval = parallel_hand_scale_residual_power(
        block, model, split.evaluation_mask, Correlation.LL, ll_scale
    )
    null_eval = _weighted_power(residual, block.weight, split.evaluation_mask)
    sky_eval = _weighted_power(corrected_residual, block.weight, split.evaluation_mask)
    null_disc = _weighted_power(residual, block.weight, split.discovery_mask)
    sky_disc = _weighted_power(corrected_residual, block.weight, split.discovery_mask)
    payload = {
        "global_contrast": contrast,
        "global_fit_rank": fit.rank,
        "discovery_null_residual_power": null_disc,
        "discovery_contrast_residual_power": sky_disc,
        "evaluation_null_residual_power": null_eval,
        "evaluation_contrast_residual_power": sky_eval,
        "evaluation_relative_improvement": (
            float("nan")
            if not np.isfinite(null_eval) or null_eval <= 0
            else 1.0 - sky_eval / null_eval
        ),
        "rr_scale": rr_scale,
        "ll_scale": ll_scale,
        "rr_implied_contrast": rr_implied,
        "ll_implied_contrast": ll_implied,
        "implied_contrast_difference": rr_implied - ll_implied,
        "rr_discovery_residual_power": rr_discovery_power,
        "ll_discovery_residual_power": ll_discovery_power,
        "rr_evaluation_residual_power": rr_eval,
        "ll_evaluation_residual_power": ll_eval,
        "null_correlation_power": correlation_residual_power(
            residual, block.weight, block.active, block.correlations
        ),
        "contrast_correlation_power": correlation_residual_power(
            corrected_residual, block.weight, block.active, block.correlations
        ),
        "n_discovery_baselines": len(split.discovery_baselines),
        "n_selection_baselines": len(split.selection_baselines),
        "n_evaluation_baselines": len(split.evaluation_baselines),
        "seed": seed,
    }
    return payload


def _channel_power(
    residual: np.ndarray, weight: np.ndarray, mask: np.ndarray, index: int
) -> np.ndarray:
    selected = mask & np.isfinite(weight) & (weight > 0)
    powers = np.full(residual.shape[1], np.nan, dtype=np.float64)
    for channel in range(residual.shape[1]):
        sample = selected[:, channel, index]
        if not np.any(sample):
            continue
        values = residual[:, channel, index][sample]
        finite = np.isfinite(values.real) & np.isfinite(values.imag)
        if not np.any(finite):
            continue
        selected_weight = weight[:, channel, index][sample][finite]
        powers[channel] = float(np.sum(selected_weight * np.abs(values[finite]) ** 2))
    return powers


def _write_plots(
    output: Path,
    block: VisibilityBlock,
    model: np.ndarray,
    contrast: float,
) -> None:
    residual = block.visibility - model
    corrected = apply_global_circular_contrast(model, block.correlations, contrast)
    rr = _hand_index(block.correlations, Correlation.RR)
    ll = _hand_index(block.correlations, Correlation.LL)
    frequency_mhz = np.asarray(block.frequency_hz) / 1e6
    figure, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), sharey=True)
    for axis, title, vis in (
        (axes[0], "frozen Stokes I", residual),
        (axes[1], f"I(1±v), v={contrast:.4f}", block.visibility - corrected),
    ):
        axis.plot(
            frequency_mhz,
            _channel_power(vis, block.weight, block.active, rr),
            label="RR",
        )
        axis.plot(
            frequency_mhz,
            _channel_power(vis, block.weight, block.active, ll),
            label="LL",
        )
        axis.set_title(title)
        axis.set_xlabel("Frequency (MHz)")
        axis.legend(frameon=False)
    axes[0].set_ylabel("Weighted residual power")
    figure.tight_layout()
    figure.savefig(output / "rr_ll_channel_residual_power.png", dpi=140)
    plt.close(figure)


def _concatenated_flux(components: tuple[MosaicSkyComponent, ...]) -> np.ndarray:
    return np.concatenate(
        [np.asarray(component.flux, dtype=np.float64) for component in components]
    )


def _resolve_protocol_paths(
    protocol: dict[str, Any], *roots: Path
) -> dict[str, Any]:
    """Resolve repo-relative frozen_directory against known data roots."""

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


def _component_names(components: tuple[MosaicSkyComponent, ...]) -> np.ndarray:
    names: list[str] = []
    for component in components:
        names.extend([component.name] * np.asarray(component.flux).size)
    return np.asarray(names)


def fit_independent_hand_skies(
    block: VisibilityBlock,
    components: tuple[MosaicSkyComponent, ...],
    train_mask: np.ndarray,
    holdout_mask: np.ndarray,
    mosaic_phase_centre_rad: tuple[float, float],
    *,
    beam: VLAPrimaryBeam,
    inference: InferenceConfig,
) -> dict[str, Any]:
    """Refit the frozen topology on RR-only and LL-only visibilities."""

    fits = {}
    for hand in (Correlation.RR, Correlation.LL):
        print(f"Fitting independent {hand.value} sky", flush=True)
        result = infer_mosaic_composite(
            (block,),
            components,
            (_hand_mask(block, train_mask, hand),),
            mosaic_phase_centre_rad,
            inference,
            holdout_masks=(_hand_mask(block, holdout_mask, hand),),
            primary_beam=beam,
        )
        fits[hand.value] = result
    flux_rr = _concatenated_flux(fits["RR"].components)
    flux_ll = _concatenated_flux(fits["LL"].components)
    total = flux_rr + flux_ll
    contrast = np.divide(
        flux_rr - flux_ll,
        total,
        out=np.zeros_like(total),
        where=total > 0,
    )
    return {
        "rr": fits["RR"],
        "ll": fits["LL"],
        "flux_rr": flux_rr,
        "flux_ll": flux_ll,
        "contrast": contrast,
        "component": _component_names(fits["RR"].components),
    }


def _write_independent_hand_plot(
    output: Path,
    flux_i: np.ndarray,
    flux_rr: np.ndarray,
    flux_ll: np.ndarray,
    component: np.ndarray,
) -> None:
    total = flux_rr + flux_ll
    contrast = np.divide(
        flux_rr - flux_ll,
        total,
        out=np.zeros_like(total),
        where=total > 0,
    )
    figure, axis = plt.subplots(figsize=(5.5, 4.4))
    for name in sorted(set(component.tolist())):
        selected = component == name
        axis.scatter(
            flux_i[selected],
            contrast[selected],
            s=12,
            alpha=0.7,
            label=name,
        )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xscale("symlog", linthresh=1e-4)
    axis.set_xlabel("Frozen Stokes I (Jy)")
    axis.set_ylabel(r"$(I_{RR}-I_{LL})/(I_{RR}+I_{LL})$")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output / "independent_hand_contrast_vs_flux.png", dpi=140)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--native-fixture",
        type=Path,
        default=Path("outputs/3c391_native_averaging_ablation/native_C1.zarr"),
    )
    parser.add_argument(
        "--sky-protocol",
        type=Path,
        default=Path("outputs/3c391_composite_catalogue_stage3/protocol.json"),
    )
    parser.add_argument(
        "--sky-checkpoint",
        type=Path,
        default=Path("outputs/3c391_recovery_policy_fit_zero/sealed_active_only.npz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/3c391_circular_contrast"),
    )
    parser.add_argument("--seed", type=int, default=391)
    parser.add_argument("--selection-baseline-fraction", type=float, default=0.2)
    parser.add_argument("--evaluation-baseline-fraction", type=float, default=0.2)
    parser.add_argument(
        "--fit-independent-hands",
        action="store_true",
        help="Refit the frozen topology separately on RR and on LL.",
    )
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--sparsity-weight", type=float, default=1e-4)
    parser.add_argument("--visibility-tile-size", type=int, default=128)
    parser.add_argument("--pixel-tile-size", type=int, default=256)
    parser.add_argument("--precision", choices=("float32", "float64"), default="float32")
    arguments = parser.parse_args()
    blocks = read_dataset(arguments.native_fixture).blocks
    if len(blocks) != 1:
        raise ValueError("native fixture must contain exactly one visibility block")
    block = blocks[0]
    model = block.model_visibility
    if model is None:
        raise ValueError("native fixture must store a frozen model_visibility")
    payload = diagnose_circular_contrast(
        block,
        model,
        seed=arguments.seed,
        selection_fraction=arguments.selection_baseline_fraction,
        evaluation_fraction=arguments.evaluation_baseline_fraction,
    )
    arguments.output.mkdir(parents=True, exist_ok=True)
    _write_plots(arguments.output, block, model, float(payload["global_contrast"]))
    if arguments.fit_independent_hands:
        scripts_directory = str(Path(__file__).resolve().parent)
        if scripts_directory not in sys.path:
            sys.path.insert(0, scripts_directory)
        from compare_3c391_composite_existing_flags import _components_from_checkpoint

        protocol = _resolve_protocol_paths(
            json.loads(arguments.sky_protocol.read_text(encoding="utf-8")),
            Path.cwd(),
            arguments.native_fixture.resolve().parent.parent.parent,
            arguments.sky_protocol.resolve().parent.parent.parent,
            arguments.sky_checkpoint.resolve().parent.parent.parent,
        )
        components = _components_from_checkpoint(
            arguments.sky_checkpoint,
            protocol,
            block.phase_centre_rad,
        )
        split = split_search_baselines(
            block,
            selection_fraction=arguments.selection_baseline_fraction,
            evaluation_fraction=arguments.evaluation_baseline_fraction,
            seed=arguments.seed,
        )
        train_mask = split.discovery_mask | split.selection_mask
        beam = VLAPrimaryBeam(
            kind="airy",
            catalog=VLABeamCatalog(
                airy_max_radius_rad_at_1ghz=np.deg2rad(
                    float(
                        protocol["airy_max_radius_deg_at_1ghz"]
                    )
                    if "airy_max_radius_deg_at_1ghz" in protocol
                    else np.rad2deg(VLABeamCatalog().airy_max_radius_rad_at_1ghz)
                )
            ),
        )
        inference = InferenceConfig(
            solver="hybrid",
            operator_mode="explicit",
            steps=arguments.steps,
            sparsity_weight=arguments.sparsity_weight,
            learning_rate=0.03,
            validation_interval=5,
            kkt_tolerance=1e-5,
            direct_dft=DirectDFTConfig(
                visibility_chunk_size=arguments.visibility_tile_size,
                pixel_chunk_size=arguments.pixel_tile_size,
                precision=arguments.precision,
            ),
        )
        independent = fit_independent_hand_skies(
            block,
            components,
            train_mask,
            split.evaluation_mask,
            block.phase_centre_rad,
            beam=beam,
            inference=inference,
        )
        flux_i = _concatenated_flux(components)
        np.savez(
            arguments.output / "independent_hand_flux.npz",
            flux_i=flux_i,
            flux_rr=independent["flux_rr"],
            flux_ll=independent["flux_ll"],
            contrast=independent["contrast"],
            component=independent["component"],
        )
        _write_independent_hand_plot(
            arguments.output,
            flux_i,
            independent["flux_rr"],
            independent["flux_ll"],
            independent["component"],
        )
        bright = flux_i >= max(1e-3, 0.01 * float(np.max(flux_i)))
        payload["independent_hands"] = {
            "rr_steps": independent["rr"].steps,
            "ll_steps": independent["ll"].steps,
            "rr_kkt": independent["rr"].kkt_residual,
            "ll_kkt": independent["ll"].kkt_residual,
            "rr_holdout": (
                independent["rr"].holdout_history[-1]
                if independent["rr"].holdout_history
                else None
            ),
            "ll_holdout": (
                independent["ll"].holdout_history[-1]
                if independent["ll"].holdout_history
                else None
            ),
            "bright_leaf_count": int(np.count_nonzero(bright)),
            "bright_median_contrast": (
                float(np.median(independent["contrast"][bright])) if np.any(bright) else None
            ),
            "bright_rms_contrast": (
                float(np.sqrt(np.mean(independent["contrast"][bright] ** 2)))
                if np.any(bright)
                else None
            ),
        }
    (arguments.output / "summary.json").write_text(
        json.dumps(_to_json(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(_to_json(payload), indent=2, sort_keys=True))
    print(arguments.output / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
