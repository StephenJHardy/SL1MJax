#!/usr/bin/env python3
"""Describe model--data changes in the currently flagged 3C391 visibilities."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from compare_3c391_composite_existing_flags import _components_from_checkpoint
from matplotlib.colors import LogNorm

from sl1mjax.beam import VLABeamCatalog, VLAPrimaryBeam
from sl1mjax.composite import MosaicSkyComponent, predict_mosaic_composite
from sl1mjax.data.canonical import VisibilityBlock, read_dataset
from sl1mjax.direct_operator import DirectDFTConfig
from sl1mjax.residual_audit import audit_visibility_residuals, robust_residual_scores

SPEED_OF_LIGHT_M_S = 299_792_458.0
QUANTILES = (0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999, 1.0)


def _load_active_predictions(
    checkpoint: Path,
    blocks: tuple[VisibilityBlock, ...],
) -> tuple[np.ndarray, ...]:
    """Read the leading, originally-active predictions from a policy fit."""

    with np.load(checkpoint) as stored:
        predictions = tuple(
            np.asarray(stored[f"prediction_C{index + 1}"])[: block.shape[0]]
            for index, block in enumerate(blocks)
        )
    for index, (block, prediction) in enumerate(zip(blocks, predictions, strict=True)):
        if prediction.shape != block.shape or np.any(~np.isfinite(prediction)):
            raise ValueError(f"prediction_C{index + 1} does not cover the active fixture")
    return predictions


def _components_with_fitted_flux(
    checkpoint: Path,
    protocol: dict[str, Any],
    phase_centre_rad: tuple[float, float],
) -> tuple[MosaicSkyComponent, ...]:
    """Rebuild fixed component geometry and replace it with fitted flux."""

    components = _components_from_checkpoint(checkpoint, protocol, phase_centre_rad)
    with np.load(checkpoint) as stored:
        fitted = []
        for component in components:
            key = f"flux_{component.name}"
            if key not in stored:
                raise ValueError(f"checkpoint is missing {key}")
            flux = np.asarray(stored[key], dtype=np.float64)
            if flux.shape != component.flux.shape:
                raise ValueError(f"{key} has shape {flux.shape}; expected {component.flux.shape}")
            fitted.append(replace(component, flux=flux))
    return tuple(fitted)


def _flatten_samples(
    blocks: tuple[VisibilityBlock, ...],
    predictions: tuple[np.ndarray, ...],
    scores: tuple[np.ndarray, ...],
    *,
    start_time_s: float,
    ratio_snr_threshold: float,
) -> dict[str, np.ndarray]:
    """Flatten usable visibilities into physical residual coordinates."""

    result: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "pointing",
            "time_hour",
            "time_bin",
            "scan",
            "antenna1",
            "antenna2",
            "channel",
            "correlation",
            "frequency_ghz",
            "uv_klambda",
            "observed_real_jy",
            "observed_imag_jy",
            "predicted_real_jy",
            "predicted_imag_jy",
            "weight",
            "residual_amplitude_jy",
            "weighted_residual_amplitude",
            "score",
            "model_snr",
            "observed_snr",
            "log_amplitude_ratio",
            "phase_error_deg",
            "radial_residual_sigma",
            "tangential_residual_sigma",
        )
    }
    for pointing, (block, prediction, score) in enumerate(
        zip(blocks, predictions, scores, strict=True)
    ):
        selected = block.active & np.isfinite(score)
        row, channel, correlation = np.nonzero(selected)
        observed = block.visibility[selected]
        model = prediction[selected]
        weight = block.weight[selected]
        sqrt_weight = np.sqrt(weight)
        residual = observed - model
        model_amplitude = np.abs(model)
        observed_amplitude = np.abs(observed)
        safe_model_amplitude = np.maximum(model_amplitude, np.finfo(np.float64).tiny)
        aligned_residual = residual * np.conj(model) / safe_model_amplitude
        model_snr = model_amplitude * sqrt_weight
        observed_snr = observed_amplitude * sqrt_weight
        ratio_valid = (model_snr >= ratio_snr_threshold) & (observed_snr >= ratio_snr_threshold)
        log_ratio = np.full(observed.shape, np.nan, dtype=np.float64)
        phase_error = np.full(observed.shape, np.nan, dtype=np.float64)
        log_ratio[ratio_valid] = np.log10(
            observed_amplitude[ratio_valid] / model_amplitude[ratio_valid]
        )
        phase_error[ratio_valid] = np.rad2deg(
            np.angle(observed[ratio_valid] * np.conj(model[ratio_valid]))
        )
        frequency = block.frequency_hz[channel]
        uv_distance_m = np.linalg.norm(block.uvw_m[row, :2], axis=1)
        values: dict[str, np.ndarray] = {
            "pointing": np.full(row.size, pointing, dtype=np.int16),
            "time_hour": (block.time_s[row] - start_time_s) / 3600.0,
            "time_bin": np.floor(block.time_s[row] / 60.0).astype(np.int64),
            "scan": block.scan_id[row].astype(np.int32),
            "antenna1": block.antenna1[row].astype(np.int16),
            "antenna2": block.antenna2[row].astype(np.int16),
            "channel": channel.astype(np.int8),
            "correlation": correlation.astype(np.int8),
            "frequency_ghz": frequency / 1e9,
            "uv_klambda": uv_distance_m * frequency / SPEED_OF_LIGHT_M_S / 1000.0,
            "observed_real_jy": observed.real,
            "observed_imag_jy": observed.imag,
            "predicted_real_jy": model.real,
            "predicted_imag_jy": model.imag,
            "weight": weight,
            "residual_amplitude_jy": np.abs(residual),
            "weighted_residual_amplitude": np.abs(residual) * sqrt_weight,
            "score": score[selected],
            "model_snr": model_snr,
            "observed_snr": observed_snr,
            "log_amplitude_ratio": log_ratio,
            "phase_error_deg": phase_error,
            "radial_residual_sigma": aligned_residual.real * sqrt_weight,
            "tangential_residual_sigma": aligned_residual.imag * sqrt_weight,
        }
        for name, value in values.items():
            result[name].append(np.asarray(value))
    return {name: np.concatenate(parts) for name, parts in result.items()}


def _quantiles(values: np.ndarray) -> dict[str, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {}
    result = np.quantile(finite, QUANTILES)
    return {f"p{100 * q:g}": float(value) for q, value in zip(QUANTILES, result, strict=True)}


def _coherence_fraction(features: dict[str, np.ndarray], tail: np.ndarray) -> np.ndarray:
    """Fraction of flagged samples in the same time/pointing/chan/corr cell in the tail."""

    keys = np.column_stack(
        (
            features["pointing"],
            features["time_bin"],
            features["channel"],
            features["correlation"],
        )
    )
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    total = np.bincount(inverse)
    tail_count = np.bincount(inverse, weights=tail.astype(np.float64))
    return tail_count[inverse] / total[inverse]


def _cohort_summary(
    features: dict[str, np.ndarray],
    *,
    score_threshold: float,
) -> dict[str, Any]:
    score = features["score"]
    tail = score > score_threshold
    power = features["weight"] * np.square(features["residual_amplitude_jy"])
    ratio_valid = np.isfinite(features["log_amplitude_ratio"])
    radial = features["radial_residual_sigma"]
    tangential = features["tangential_residual_sigma"]
    coherent_fraction = _coherence_fraction(features, tail)
    tail_count = int(np.count_nonzero(tail))
    tiny = np.finfo(np.float64).tiny
    return {
        "sample_count": int(score.size),
        "tail_count": tail_count,
        "tail_fraction": float(np.mean(tail)),
        "tail_residual_power_fraction": float(np.sum(power[tail]) / max(np.sum(power), tiny)),
        "score": _quantiles(score),
        "residual_amplitude_jy": _quantiles(features["residual_amplitude_jy"]),
        "weighted_residual_amplitude": _quantiles(features["weighted_residual_amplitude"]),
        "model_snr": _quantiles(features["model_snr"]),
        "high_snr_ratio_sample_count": int(np.count_nonzero(ratio_valid)),
        "log10_observed_to_model_amplitude": _quantiles(features["log_amplitude_ratio"]),
        "absolute_phase_error_deg": _quantiles(np.abs(features["phase_error_deg"])),
        "tail_morphology": {
            "amplitude_dominated_fraction": float(
                np.mean(np.abs(radial[tail]) >= np.abs(tangential[tail]))
            )
            if tail_count
            else float("nan"),
            "phase_dominated_fraction": float(
                np.mean(np.abs(tangential[tail]) > np.abs(radial[tail]))
            )
            if tail_count
            else float("nan"),
            "positive_radial_fraction": float(np.mean(radial[tail] > 0))
            if tail_count
            else float("nan"),
            "negative_radial_fraction": float(np.mean(radial[tail] < 0))
            if tail_count
            else float("nan"),
            "cross_baseline_coherent_fraction": float(np.mean(coherent_fraction[tail] >= 0.25))
            if tail_count
            else float("nan"),
        },
    }


def _group_rows(
    features: dict[str, np.ndarray],
    *,
    score_threshold: float,
) -> list[dict[str, Any]]:
    definitions = {
        "pointing": (features["pointing"],),
        "channel": (features["channel"],),
        "correlation": (features["correlation"],),
        "scan": (features["pointing"], features["scan"]),
        "baseline": (
            np.minimum(features["antenna1"], features["antenna2"]),
            np.maximum(features["antenna1"], features["antenna2"]),
        ),
    }
    total_power = np.sum(features["weight"] * np.square(features["residual_amplitude_jy"]))
    rows: list[dict[str, Any]] = []
    for kind, columns in definitions.items():
        keys = np.column_stack(columns)
        unique, inverse = np.unique(keys, axis=0, return_inverse=True)
        for index, key in enumerate(unique):
            selected = inverse == index
            power = features["weight"][selected] * np.square(
                features["residual_amplitude_jy"][selected]
            )
            score = features["score"][selected]
            rows.append(
                {
                    "kind": kind,
                    "key": ":".join(map(str, np.atleast_1d(key))),
                    "sample_count": int(np.count_nonzero(selected)),
                    "tail_count": int(np.count_nonzero(score > score_threshold)),
                    "tail_fraction": float(np.mean(score > score_threshold)),
                    "median_score": float(np.median(score)),
                    "score_p99": float(np.quantile(score, 0.99)),
                    "residual_power_fraction": float(
                        np.sum(power) / max(total_power, np.finfo(np.float64).tiny)
                    ),
                }
            )
    return rows


def _write_group_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _survival(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite = np.sort(np.asarray(values)[np.isfinite(values)])
    return finite, 1.0 - np.arange(finite.size, dtype=np.float64) / finite.size


def _plot_diagnostics(
    path: Path,
    active: dict[str, np.ndarray],
    flagged: dict[str, np.ndarray],
    groups: list[dict[str, Any]],
    *,
    score_threshold: float,
) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    for features, label in ((active, "originally active"), (flagged, "pre-existing flag")):
        x, y = _survival(features["score"])
        axes[0, 0].semilogy(x, y, label=label)
    axes[0, 0].axvline(score_threshold, color="black", linestyle="--", linewidth=1)
    axes[0, 0].set(xlabel="robust residual score", ylabel="fraction above score", xlim=(-3, 50))
    axes[0, 0].legend()

    model_amplitude = np.hypot(flagged["predicted_real_jy"], flagged["predicted_imag_jy"])
    observed_amplitude = np.hypot(flagged["observed_real_jy"], flagged["observed_imag_jy"])
    positive = (model_amplitude > 0) & (observed_amplitude > 0)
    axes[0, 1].hexbin(
        model_amplitude[positive],
        observed_amplitude[positive],
        gridsize=70,
        bins="log",
        mincnt=1,
        xscale="log",
        yscale="log",
        cmap="viridis",
    )
    limits = np.quantile(
        np.concatenate((model_amplitude[positive], observed_amplitude[positive])), (0.001, 0.999)
    )
    axes[0, 1].plot(limits, limits, color="white", linewidth=1)
    axes[0, 1].set(xlabel="predicted amplitude (Jy)", ylabel="observed amplitude (Jy)")

    valid = np.isfinite(flagged["log_amplitude_ratio"])
    axes[0, 2].hist(flagged["log_amplitude_ratio"][valid], bins=120, range=(-1, 1), density=True)
    axes[0, 2].axvline(0, color="black", linewidth=1)
    axes[0, 2].set(xlabel="log10(observed / predicted amplitude)", ylabel="density")

    phase = flagged["phase_error_deg"]
    axes[1, 0].hist(phase[np.isfinite(phase)], bins=120, range=(-180, 180), density=True)
    axes[1, 0].set(xlabel="observed - predicted phase (deg)", ylabel="density")

    limit = float(
        np.quantile(
            np.abs(
                np.concatenate(
                    (
                        flagged["radial_residual_sigma"],
                        flagged["tangential_residual_sigma"],
                    )
                )
            ),
            0.995,
        )
    )
    axes[1, 1].hexbin(
        flagged["radial_residual_sigma"],
        flagged["tangential_residual_sigma"],
        gridsize=70,
        bins="log",
        mincnt=1,
        extent=(-limit, limit, -limit, limit),
        cmap="magma",
    )
    axes[1, 1].axhline(0, color="white", linewidth=0.7)
    axes[1, 1].axvline(0, color="white", linewidth=0.7)
    axes[1, 1].set(
        xlabel="amplitude-like residual (sigma)",
        ylabel="phase-like residual (sigma)",
    )

    baselines = [row for row in groups if row["kind"] == "baseline"]
    baselines.sort(key=lambda row: row["tail_count"], reverse=True)
    selected = baselines[:20][::-1]
    axes[1, 2].barh([row["key"] for row in selected], [row["tail_count"] for row in selected])
    axes[1, 2].set(xlabel="samples above threshold", ylabel="antenna baseline")
    figure.suptitle("3C391 frozen-sky audit of pre-existing flagged visibilities")
    figure.savefig(path, dpi=170)
    plt.close(figure)


def _plot_amplitude_comparison(
    path: Path,
    active: dict[str, np.ndarray],
    flagged: dict[str, np.ndarray],
) -> None:
    """Plot matched model--data amplitude densities for both flag cohorts."""

    cohorts = (("Originally active", active), ("Pre-existing flags", flagged))
    amplitudes = []
    for label, features in cohorts:
        predicted = np.hypot(features["predicted_real_jy"], features["predicted_imag_jy"])
        observed = np.hypot(features["observed_real_jy"], features["observed_imag_jy"])
        valid = (predicted > 0) & (observed > 0)
        amplitudes.append((label, predicted[valid], observed[valid]))
    combined = np.concatenate(
        [values for _, predicted, observed in amplitudes for values in (predicted, observed)]
    )
    lower, upper = np.quantile(combined, (0.0005, 0.9995))
    limits = (float(lower), float(upper))

    # Measure each hex density before drawing so both panels share one cohort-
    # normalized colour scale despite their different sample counts.
    probe, probe_axes = plt.subplots(1, 2)
    maximum_density = 0.0
    for axis, (_, predicted, observed) in zip(probe_axes, amplitudes, strict=True):
        density = axis.hexbin(
            predicted,
            observed,
            C=np.full(predicted.size, 1.0 / predicted.size),
            reduce_C_function=np.sum,
            gridsize=75,
            mincnt=1,
            xscale="log",
            yscale="log",
            extent=(np.log10(limits[0]), np.log10(limits[1])) * 2,
        )
        maximum_density = max(maximum_density, float(np.max(density.get_array())))
    plt.close(probe)

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12.5, 5.6),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    shown = None
    for axis, (label, predicted, observed) in zip(axes, amplitudes, strict=True):
        shown = axis.hexbin(
            predicted,
            observed,
            C=np.full(predicted.size, 1.0 / predicted.size),
            reduce_C_function=np.sum,
            gridsize=75,
            mincnt=1,
            xscale="log",
            yscale="log",
            extent=(np.log10(limits[0]), np.log10(limits[1])) * 2,
            norm=LogNorm(vmin=max(maximum_density * 1e-4, 1e-8), vmax=maximum_density),
            cmap="viridis",
        )
        axis.plot(limits, limits, color="white", linewidth=1.2)
        axis.set_xlim(limits)
        axis.set_ylim(limits)
        axis.set_title(f"{label} (n={predicted.size:,})")
        axis.set_xlabel("predicted amplitude (Jy)")
        axis.set_aspect("equal", adjustable="box")
    axes[0].set_ylabel("observed amplitude (Jy)")
    assert shown is not None
    figure.colorbar(
        shown,
        ax=axes,
        fraction=0.035,
        pad=0.025,
        label="fraction of cohort per hexagon",
    )
    figure.suptitle("3C391 predicted and observed visibility amplitudes")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--active-fixture",
        type=Path,
        default=Path("outputs/3c391_gain_time_model_sweep/selected_native_fixture.zarr"),
    )
    parser.add_argument(
        "--flagged-fixture",
        type=Path,
        default=Path("outputs/3c391_matched_existing_flag_audit/flagged_fixture.zarr"),
    )
    parser.add_argument(
        "--fit-directory",
        type=Path,
        default=Path("outputs/3c391_recovery_policy_fit_zero"),
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("outputs/3c391_composite_catalogue_stage3/protocol.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/3c391_flagged_visibility_distribution"),
    )
    parser.add_argument("--score-threshold", type=float, default=6.0)
    parser.add_argument("--ratio-snr-threshold", type=float, default=3.0)
    parser.add_argument("--visibility-tile-size", type=int, default=256)
    parser.add_argument("--pixel-tile-size", type=int, default=1024)
    parser.add_argument("--precision", choices=("float32", "float64"), default="float32")
    arguments = parser.parse_args()

    fit_summary = json.loads((arguments.fit_directory / "summary.json").read_text())
    if not fit_summary["protocol"].get("sealed_opened"):
        raise ValueError("the recovery-policy sealed fold has not been opened")
    selected_policy = str(fit_summary["selected_policy"])
    checkpoint = arguments.fit_directory / f"sealed_{selected_policy}.npz"
    active_blocks = read_dataset(arguments.active_fixture).blocks
    flagged_blocks = read_dataset(arguments.flagged_fixture).blocks
    if len(active_blocks) != len(flagged_blocks):
        raise ValueError("active and flagged fixtures must contain the same pointings")
    protocol = json.loads(arguments.protocol.read_text())
    phase_centre = active_blocks[0].phase_centre_rad
    components = _components_with_fitted_flux(checkpoint, protocol, phase_centre)
    active_predictions = _load_active_predictions(checkpoint, active_blocks)
    beam = VLAPrimaryBeam(
        kind="airy",
        catalog=replace(
            VLABeamCatalog(),
            airy_max_radius_rad_at_1ghz=np.deg2rad(float(protocol["airy_max_radius_deg_at_1ghz"])),
        ),
    )
    direct = DirectDFTConfig(
        visibility_chunk_size=arguments.visibility_tile_size,
        pixel_chunk_size=arguments.pixel_tile_size,
        precision=arguments.precision,
    )
    arguments.output.mkdir(parents=True, exist_ok=True)
    prediction_path = arguments.output / "flagged_predictions.npz"
    if prediction_path.exists():
        with np.load(prediction_path) as stored:
            flagged_predictions = tuple(
                np.asarray(stored[f"prediction_C{index + 1}"])
                for index in range(len(flagged_blocks))
            )
    else:
        print("predicting the frozen best sky at flagged coordinates", flush=True)
        flagged_predictions = predict_mosaic_composite(
            flagged_blocks,
            components,
            phase_centre,
            primary_beam=beam,
            config=direct,
        )
        np.savez_compressed(
            prediction_path,
            **{
                f"prediction_C{index + 1}": prediction
                for index, prediction in enumerate(flagged_predictions)
            },
        )
    active_masks = tuple(block.active for block in active_blocks)
    active_scores, scales = robust_residual_scores(active_blocks, active_predictions, active_masks)
    combined_blocks = tuple(
        replace(
            active,
            visibility=np.concatenate((active.visibility, flagged.visibility)),
            weight=np.concatenate((active.weight, flagged.weight)),
            flag=np.concatenate((active.flag, flagged.flag)),
            uvw_m=np.concatenate((active.uvw_m, flagged.uvw_m)),
            time_s=np.concatenate((active.time_s, flagged.time_s)),
            antenna1=np.concatenate((active.antenna1, flagged.antenna1)),
            antenna2=np.concatenate((active.antenna2, flagged.antenna2)),
            scan_id=np.concatenate((active.scan_id, flagged.scan_id)),
            field_id=np.concatenate((active.field_id, flagged.field_id)),
            state_id=np.concatenate((active.state_id, flagged.state_id)),
            observation_id=np.concatenate((active.observation_id, flagged.observation_id)),
            feed1=np.concatenate((active.feed1, flagged.feed1)),
            feed2=np.concatenate((active.feed2, flagged.feed2)),
            interval_s=np.concatenate((active.interval_s, flagged.interval_s)),
        )
        for active, flagged in zip(active_blocks, flagged_blocks, strict=True)
    )
    combined_predictions = tuple(
        np.concatenate((active, flagged), axis=0)
        for active, flagged in zip(active_predictions, flagged_predictions, strict=True)
    )
    combined_discovery = tuple(
        np.concatenate((mask, np.zeros(flagged.shape, dtype=bool)), axis=0)
        for mask, flagged in zip(active_masks, flagged_blocks, strict=True)
    )
    combined_evaluation = tuple(
        np.concatenate((np.zeros(active.shape, dtype=bool), flagged.active), axis=0)
        for active, flagged in zip(active_blocks, flagged_blocks, strict=True)
    )
    audit = audit_visibility_residuals(
        combined_blocks,
        combined_predictions,
        combined_discovery,
        combined_evaluation,
        score_threshold=arguments.score_threshold,
        group_kinds=("pointing", "baseline", "antenna", "channel", "correlation", "scan"),
        fixed_scales=scales,
    )
    combined_scores, _ = robust_residual_scores(
        combined_blocks, combined_predictions, combined_discovery
    )
    flagged_scores = tuple(
        score[active.shape[0] :]
        for score, active in zip(combined_scores, active_blocks, strict=True)
    )
    start_time = min(float(np.min(block.time_s)) for block in active_blocks)
    active_features = _flatten_samples(
        active_blocks,
        active_predictions,
        active_scores,
        start_time_s=start_time,
        ratio_snr_threshold=arguments.ratio_snr_threshold,
    )
    flagged_features = _flatten_samples(
        flagged_blocks,
        flagged_predictions,
        flagged_scores,
        start_time_s=start_time,
        ratio_snr_threshold=arguments.ratio_snr_threshold,
    )
    groups = _group_rows(flagged_features, score_threshold=arguments.score_threshold)
    _write_group_csv(arguments.output / "flagged_groups.csv", groups)
    np.savez_compressed(arguments.output / "flagged_sample_diagnostics.npz", **flagged_features)
    _plot_diagnostics(
        arguments.output / "flagged_visibility_diagnostics.jpg",
        active_features,
        flagged_features,
        groups,
        score_threshold=arguments.score_threshold,
    )
    _plot_amplitude_comparison(
        arguments.output / "flagged_unflagged_amplitude_comparison.jpg",
        active_features,
        flagged_features,
    )
    summary = {
        "schema_version": 1,
        "selected_policy": selected_policy,
        "checkpoint": str(checkpoint),
        "score_threshold": arguments.score_threshold,
        "ratio_snr_threshold": arguments.ratio_snr_threshold,
        "interpretation": {
            "score_scale": "median and MAD fitted on originally active samples only",
            "amplitude_like": "residual projected parallel to the predicted complex visibility",
            "phase_like": "residual projected perpendicular to the predicted complex visibility",
            "coherent": (
                "at least 25% of flagged samples in one pointing/time/channel/"
                "correlation cell exceed the score threshold"
            ),
            "limitation": (
                "the averaged fixture identifies pre-existing flags but not the "
                "exact CASA flag command responsible for each sample"
            ),
        },
        "active": _cohort_summary(active_features, score_threshold=arguments.score_threshold),
        "flagged": _cohort_summary(flagged_features, score_threshold=arguments.score_threshold),
        "residual_audit": asdict(audit),
    }
    (arguments.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"active": summary["active"], "flagged": summary["flagged"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
