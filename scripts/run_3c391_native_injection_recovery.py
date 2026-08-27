#!/usr/bin/env python3
"""Inject and recover temporal and spectral sky atoms on native 3C391 holdouts."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

from sl1mjax.beam import VLABeamCatalog, VLAPrimaryBeam
from sl1mjax.composite import MosaicPointComponent, predict_mosaic_composite
from sl1mjax.data.canonical import VisibilityBlock, read_dataset
from sl1mjax.direct_operator import DirectDFTConfig
from sl1mjax.sky_recovery import (
    ComponentRecoveryFit,
    RecoveryScore,
    fit_real_sky_component,
    inject_sky_component,
    score_recovery_residual,
    spectral_support_mask,
    split_native_baselines,
    temporal_support_mask,
)


def _parse_positive_values(text: str, *, integer: bool) -> tuple[float, ...] | tuple[int, ...]:
    try:
        values = tuple(int(value) if integer else float(value) for value in text.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated numbers") from error
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("values must be positive")
    if len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("values must not contain duplicates")
    return values


def _parse_durations(text: str) -> tuple[float, ...]:
    return _parse_positive_values(text, integer=False)  # type: ignore[return-value]


def _parse_channel_widths(text: str) -> tuple[int, ...]:
    return _parse_positive_values(text, integer=True)  # type: ignore[return-value]


def _parse_snrs(text: str) -> tuple[float, ...]:
    try:
        values = tuple(float(value) for value in text.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated SNR values") from error
    if not values or any(not np.isfinite(value) or value < 0 for value in values):
        raise argparse.ArgumentTypeError("SNR values must be finite and non-negative")
    if len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("SNR values must not contain duplicates")
    return values


def _central_time_interval(
    block: VisibilityBlock, duration_s: float
) -> tuple[float, float, tuple[float, ...]]:
    """Choose a central contiguous run containing the requested native samples."""

    cadence = float(np.median(block.interval_s))
    count = max(1, int(round(duration_s / cadence)))
    if not np.isclose(count * cadence, duration_s, rtol=0.0, atol=1e-6):
        raise ValueError("transient duration must be an integer multiple of native cadence")
    times = np.unique(block.time_s)
    breaks = np.flatnonzero(np.diff(times) > 1.5 * cadence) + 1
    runs = np.split(times, breaks)
    eligible = [run for run in runs if run.size >= count]
    if not eligible:
        raise ValueError(f"no contiguous native run can support {duration_s:g} seconds")
    centre = float(np.median(times))
    run = min(eligible, key=lambda item: abs(float(np.median(item)) - centre))
    start_index = (run.size - count) // 2
    selected = run[start_index : start_index + count]
    return float(selected[0]), count * cadence, tuple(float(value) for value in selected)


def _central_channels(block: VisibilityBlock, channel_count: int) -> tuple[int, tuple[float, ...]]:
    if channel_count > block.shape[1]:
        raise ValueError("spectral width exceeds the native channel count")
    first = (block.shape[1] - channel_count) // 2
    frequencies = tuple(float(value) for value in block.frequency_hz[first : first + channel_count])
    return first, frequencies


def _score_payload(score: RecoveryScore) -> dict[str, Any]:
    return asdict(score)


def _fit_payload(
    fit: ComponentRecoveryFit,
    *,
    event_support: np.ndarray,
    block: VisibilityBlock,
    evaluation_mask: np.ndarray,
) -> dict[str, Any]:
    assert block.model_visibility is not None
    residual = block.visibility - block.model_visibility - fit.prediction
    null_residual = block.visibility - block.model_visibility
    event_mask = evaluation_mask & event_support & block.active
    null_event_score = score_recovery_residual(
        null_residual,
        block.weight,
        event_mask,
    )
    event_score = score_recovery_residual(
        residual,
        block.weight,
        event_mask,
    )
    return {
        "coefficient_jy": fit.coefficient,
        "information_scale_jy": fit.information_scale,
        "discovery_sample_count": fit.discovery_sample_count,
        "event_evaluation_sample_count": fit.evaluation_sample_count,
        "evaluation_relative_improvement": fit.evaluation_relative_improvement,
        "event_evaluation_relative_improvement": (
            1.0 - event_score.residual_power / null_event_score.residual_power
            if null_event_score.residual_power > 0
            else float("nan")
        ),
        "evaluation": _score_payload(fit.component_evaluation),
        "event_evaluation": _score_payload(event_score),
        "null_event_evaluation": _score_payload(null_event_score),
    }


def _run_support_trials(
    block: VisibilityBlock,
    unit_response: np.ndarray,
    event_support: np.ndarray,
    *,
    injection_snrs: tuple[float, ...],
    evaluation_fraction: float,
    seed: int,
    split_count: int = 1,
) -> dict[str, Any]:
    if split_count < 1:
        raise ValueError("split_count must be positive")
    static_support = np.ones(block.shape, dtype=bool)
    split_controls = []
    for split_index in range(split_count):
        split_seed = seed + split_index
        holdout = split_native_baselines(
            block,
            evaluation_fraction=evaluation_fraction,
            seed=split_seed,
        )
        control_event = fit_real_sky_component(
            block,
            unit_response,
            event_support,
            holdout.discovery_mask,
            holdout.evaluation_mask,
        )
        control_static = fit_real_sky_component(
            block,
            unit_response,
            static_support,
            holdout.discovery_mask,
            holdout.evaluation_mask,
        )
        split_controls.append((split_seed, holdout, control_event, control_static))
    amplitude_scale = float(
        np.mean([control[2].information_scale for control in split_controls])
    )
    trials = []
    for snr in injection_snrs:
        amplitude = float(snr * amplitude_scale)
        injected = inject_sky_component(block, unit_response, event_support, amplitude)
        folds = []
        for split_seed, holdout, control_event, control_static in split_controls:
            event_fit = fit_real_sky_component(
                injected,
                unit_response,
                event_support,
                holdout.discovery_mask,
                holdout.evaluation_mask,
            )
            static_fit = fit_real_sky_component(
                injected,
                unit_response,
                static_support,
                holdout.discovery_mask,
                holdout.evaluation_mask,
            )
            event_payload = _fit_payload(
                event_fit,
                event_support=event_support,
                block=injected,
                evaluation_mask=holdout.evaluation_mask,
            )
            static_payload = _fit_payload(
                static_fit,
                event_support=event_support,
                block=injected,
                evaluation_mask=holdout.evaluation_mask,
            )
            candidate_losses = {
                "null": event_fit.null_evaluation.weighted_complex_mse,
                "static": static_fit.component_evaluation.weighted_complex_mse,
                "supported": event_fit.component_evaluation.weighted_complex_mse,
            }
            control_supported_gain = (
                control_event.null_evaluation.weighted_complex_mse
                - control_event.component_evaluation.weighted_complex_mse
            )
            control_static_gain = (
                control_static.null_evaluation.weighted_complex_mse
                - control_static.component_evaluation.weighted_complex_mse
            )
            supported_gain = candidate_losses["null"] - candidate_losses["supported"]
            static_gain = candidate_losses["null"] - candidate_losses["static"]
            folds.append(
                {
                    "seed": split_seed,
                    "selected_raw_model": min(candidate_losses, key=candidate_losses.get),
                    "full_evaluation_weighted_mse": candidate_losses,
                    "paired_supported_mse_gain": supported_gain - control_supported_gain,
                    "paired_static_mse_gain": static_gain - control_static_gain,
                    "recovered_paired_amplitude_jy": (
                        event_fit.coefficient - control_event.coefficient
                    ),
                    "protected_at_25_percent_event_improvement": (
                        event_payload["event_evaluation_relative_improvement"] >= 0.25
                    ),
                    "supported_fit": event_payload,
                    "static_fit": static_payload,
                }
            )
        mean_losses = {
            name: float(np.mean([fold["full_evaluation_weighted_mse"][name] for fold in folds]))
            for name in ("null", "static", "supported")
        }
        mean_paired_gain = {
            "static": float(np.mean([fold["paired_static_mse_gain"] for fold in folds])),
            "supported": float(
                np.mean([fold["paired_supported_mse_gain"] for fold in folds])
            ),
        }
        paired_candidates = {"null": 0.0, **mean_paired_gain}
        recovered = np.asarray(
            [fold["recovered_paired_amplitude_jy"] for fold in folds], dtype=np.float64
        )
        trials.append(
            {
                "injected_nominal_information_snr": snr,
                "injected_amplitude_jy": amplitude,
                "recovered_paired_amplitude_jy": float(np.mean(recovered)),
                "recovered_paired_amplitude_std_jy": float(np.std(recovered)),
                "recovery_fraction": (
                    float(np.mean(recovered)) / amplitude if amplitude != 0 else float("nan")
                ),
                "recovery_error_jy": float(np.mean(recovered)) - amplitude,
                "selected_full_evaluation_model": min(mean_losses, key=mean_losses.get),
                "paired_consensus_model": max(paired_candidates, key=paired_candidates.get),
                "raw_supported_selection_fraction": float(
                    np.mean([fold["selected_raw_model"] == "supported" for fold in folds])
                ),
                "paired_supported_selection_fraction": float(
                    np.mean(
                        [
                            fold["paired_supported_mse_gain"]
                            > max(0.0, fold["paired_static_mse_gain"])
                            for fold in folds
                        ]
                    )
                ),
                "mean_full_evaluation_weighted_mse": mean_losses,
                "mean_paired_mse_gain": mean_paired_gain,
                "protected_at_25_percent_event_improvement": (
                    bool(
                        np.mean(
                            [
                                fold["protected_at_25_percent_event_improvement"]
                                for fold in folds
                            ]
                        )
                        >= 0.8
                    )
                ),
                "protection_fraction": float(
                    np.mean(
                        [
                            fold["protected_at_25_percent_event_improvement"]
                            for fold in folds
                        ]
                    )
                ),
                "folds": folds,
            }
        )
    return {
        "baseline_splits": [
            {
                "evaluation_fraction": evaluation_fraction,
                "seed": split_seed,
                "discovery_baselines": [list(value) for value in holdout.discovery_baselines],
                "evaluation_baselines": [list(value) for value in holdout.evaluation_baselines],
            }
            for split_seed, holdout, _, _ in split_controls
        ],
        "control": {
            "supported_coherent_offset_jy": float(
                np.mean([control[2].coefficient for control in split_controls])
            ),
            "static_coherent_offset_jy": float(
                np.mean([control[3].coefficient for control in split_controls])
            ),
            "supported_information_scale_jy": amplitude_scale,
            "supported_event_evaluation_relative_improvement": float(
                np.mean(
                    [
                        control[2].supported_evaluation_relative_improvement
                        for control in split_controls
                    ]
                )
            ),
        },
        "trials": trials,
    }


def _unit_response(
    block: VisibilityBlock,
    *,
    mosaic_phase_centre_rad: tuple[float, float],
    l_rad: float,
    m_rad: float,
    primary_beam: VLAPrimaryBeam,
    direct: DirectDFTConfig,
    cache: Path,
) -> np.ndarray:
    metadata_path = cache.with_suffix(".json")
    metadata = {
        "shape": list(block.shape),
        "phase_centre_rad": list(block.phase_centre_rad),
        "mosaic_phase_centre_rad": list(mosaic_phase_centre_rad),
        "time_min_s": float(np.min(block.time_s)),
        "time_max_s": float(np.max(block.time_s)),
        "frequency_hz": [float(value) for value in block.frequency_hz],
        "l_rad": l_rad,
        "m_rad": m_rad,
        "beam_kind": primary_beam.kind,
        "beam_apply_squint": primary_beam.apply_squint,
        "airy_max_radius_rad_at_1ghz": (
            primary_beam.catalog.airy_max_radius_rad_at_1ghz
        ),
        "precision": direct.precision,
    }
    if cache.exists():
        if not metadata_path.exists():
            raise ValueError(f"cached response {cache} has no provenance sidecar")
        stored_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if stored_metadata != metadata:
            raise ValueError(f"cached response {cache} does not match this protocol")
        response = np.load(cache)
        if response.shape != block.shape:
            raise ValueError(f"cached response {cache} does not match the native block")
        return np.asarray(response, dtype=np.complex128)
    component = MosaicPointComponent(
        name="unit_injection_atom",
        l_rad=np.asarray([l_rad]),
        m_rad=np.asarray([m_rad]),
        flux=np.asarray([1.0]),
    )
    response = predict_mosaic_composite(
        (block,),
        (component,),
        mosaic_phase_centre_rad,
        primary_beam=primary_beam,
        config=direct,
    )[0]
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, response)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return response


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--native-fixture",
        type=Path,
        default=Path("outputs/3c391_native_averaging_ablation/native_C1.zarr"),
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
        "--output",
        type=Path,
        default=Path("outputs/3c391_native_injection_recovery"),
    )
    parser.add_argument(
        "--transient-durations-seconds",
        type=_parse_durations,
        default=(10.0, 30.0, 60.0),
    )
    parser.add_argument(
        "--spectral-channel-widths",
        type=_parse_channel_widths,
        default=(1, 2, 4, 8),
    )
    parser.add_argument(
        "--injection-snrs",
        type=_parse_snrs,
        default=(0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0),
    )
    parser.add_argument("--evaluation-baseline-fraction", type=float, default=0.25)
    parser.add_argument("--baseline-split-count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=391)
    parser.add_argument("--l-offset-arcmin", type=float, default=0.0)
    parser.add_argument("--m-offset-arcmin", type=float, default=0.0)
    parser.add_argument("--visibility-tile-size", type=int, default=256)
    parser.add_argument("--precision", choices=("float32", "float64"), default="float32")
    arguments = parser.parse_args()
    if not 0 < arguments.evaluation_baseline_fraction < 1:
        parser.error("--evaluation-baseline-fraction must be between zero and one")
    if arguments.seed < 0:
        parser.error("--seed must be non-negative")
    if arguments.baseline_split_count < 1:
        parser.error("--baseline-split-count must be positive")

    stored = read_dataset(arguments.native_fixture).blocks
    if len(stored) != 1 or stored[0].model_visibility is None:
        raise ValueError("native fixture must contain one block with a frozen prediction")
    block = stored[0]
    reference = read_dataset(arguments.reference_fixture).blocks
    if not reference:
        raise ValueError("reference fixture has no visibility blocks")
    mosaic_phase_centre_rad = reference[0].phase_centre_rad
    protocol = json.loads(arguments.sky_protocol.read_text(encoding="utf-8"))
    beam = VLAPrimaryBeam(
        kind="airy",
        catalog=replace(
            VLABeamCatalog(),
            airy_max_radius_rad_at_1ghz=np.deg2rad(
                float(protocol["airy_max_radius_deg_at_1ghz"])
            ),
        ),
    )
    arcmin_to_rad = np.deg2rad(1.0 / 60.0)
    l_rad = float(arguments.l_offset_arcmin * arcmin_to_rad)
    m_rad = float(arguments.m_offset_arcmin * arcmin_to_rad)
    direct = DirectDFTConfig(
        visibility_chunk_size=arguments.visibility_tile_size,
        pixel_chunk_size=1,
        precision=arguments.precision,
    )
    cache = arguments.output / "unit_response.npy"
    print(f"Predicting or loading one-Jy native response for {block.shape}", flush=True)
    response = _unit_response(
        block,
        mosaic_phase_centre_rad=mosaic_phase_centre_rad,
        l_rad=l_rad,
        m_rad=m_rad,
        primary_beam=beam,
        direct=direct,
        cache=cache,
    )

    transient = []
    for duration in arguments.transient_durations_seconds:
        start, actual_duration, selected_times = _central_time_interval(block, duration)
        print(f"Transient support: {actual_duration:g} s at {start:.1f}", flush=True)
        support = temporal_support_mask(block, start_s=start, duration_s=actual_duration)
        result = _run_support_trials(
            block,
            response,
            support,
            injection_snrs=arguments.injection_snrs,
            evaluation_fraction=arguments.evaluation_baseline_fraction,
            seed=arguments.seed,
            split_count=arguments.baseline_split_count,
        )
        result["duration_s"] = actual_duration
        result["selected_time_centres_s"] = list(selected_times)
        transient.append(result)

    spectral = []
    for width in arguments.spectral_channel_widths:
        first, frequencies = _central_channels(block, width)
        print(f"Spectral support: channels {first}:{first + width}", flush=True)
        support = spectral_support_mask(block, first_channel=first, channel_count=width)
        result = _run_support_trials(
            block,
            response,
            support,
            injection_snrs=arguments.injection_snrs,
            evaluation_fraction=arguments.evaluation_baseline_fraction,
            seed=arguments.seed,
            split_count=arguments.baseline_split_count,
        )
        result["first_channel"] = first
        result["channel_count"] = width
        result["frequency_hz"] = list(frequencies)
        spectral.append(result)

    summary = {
        "protocol": {
            "native_fixture": str(arguments.native_fixture),
            "frozen_static_model": "model_visibility in native fixture",
            "pointing_phase_centre_rad": list(block.phase_centre_rad),
            "mosaic_phase_centre_rad": list(mosaic_phase_centre_rad),
            "injection_l_rad": l_rad,
            "injection_m_rad": m_rad,
            "primary_beam": "extended VLA Airy from frozen sky protocol",
            "frequency_model": "exact per-channel RIME response",
            "injection_snrs": list(arguments.injection_snrs),
            "paired_zero_injection_control": True,
            "selection_data": "disjoint whole-baseline native holdout",
            "baseline_split_count": arguments.baseline_split_count,
        },
        "native_block": {
            "shape": list(block.shape),
            "active_sample_count": int(np.count_nonzero(block.active)),
            "native_integration_s": float(np.median(block.interval_s)),
            "native_channel_width_hz": float(np.median(np.diff(block.frequency_hz))),
        },
        "transient": transient,
        "spectral": spectral,
    }
    arguments.output.mkdir(parents=True, exist_ok=True)
    destination = arguments.output / "summary.json"
    destination.write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    print(f"Wrote {destination}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
