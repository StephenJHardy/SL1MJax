#!/usr/bin/env python3
"""Blindly search native 3C391 residuals for temporal and spectral refinements."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

from sl1mjax.data.canonical import VisibilityBlock, read_dataset
from sl1mjax.sky_recovery import (
    BlindSkyVariationSearchResult,
    SkyVariationCandidate,
    blind_search_sky_variation,
    inject_sky_component,
    native_variation_candidates,
    spectral_support_mask,
    split_search_baselines,
    temporal_support_mask,
)


def _parse_positive_values(text: str) -> tuple[float, ...]:
    try:
        values = tuple(float(value) for value in text.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated numbers") from error
    if not values or any(not np.isfinite(value) or value <= 0 for value in values):
        raise argparse.ArgumentTypeError("values must be finite and positive")
    if len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("values must not contain duplicates")
    return values


def _candidate_payload(candidate: SkyVariationCandidate | None) -> dict[str, Any] | None:
    return None if candidate is None else asdict(candidate)


def _score_payload(result: BlindSkyVariationSearchResult) -> dict[str, Any]:
    return {
        "candidate_count": result.candidate_count,
        "shortlist_size": result.shortlist_size,
        "selected_candidate": _candidate_payload(result.selected_candidate),
        "refit_static_coefficient_jy": result.refit_static_coefficient,
        "refit_variation_coefficient_jy": result.refit_variation_coefficient,
        "selection_incremental_weighted_mse": (
            result.selection_incremental_weighted_mse
        ),
        "evaluation_static_weighted_mse": result.evaluation_static_weighted_mse,
        "evaluation_candidate_weighted_mse": result.evaluation_candidate_weighted_mse,
        "evaluation_incremental_weighted_mse": (
            result.evaluation_incremental_weighted_mse
        ),
        "evaluation_relative_improvement": result.evaluation_relative_improvement,
        "accepted": result.accepted,
        "top_discovery": [asdict(score) for score in result.discovery_ranking[:8]],
        "shortlist": [asdict(score) for score in result.shortlist],
    }


def _central_candidate(
    candidates: tuple[SkyVariationCandidate, ...],
    block: VisibilityBlock,
    *,
    kind: str,
    width: int,
) -> SkyVariationCandidate:
    selected = [
        candidate
        for candidate in candidates
        if candidate.kind == kind and candidate.bin_count == width
    ]
    if not selected:
        raise ValueError(f"candidate bank contains no {kind} width {width}")
    if kind == "temporal_interval":
        target = float(np.median(np.unique(block.time_s)))
    else:
        target = float(np.median(block.frequency_hz))

    def distance(candidate: SkyVariationCandidate) -> float:
        assert candidate.coordinate_start is not None
        assert candidate.coordinate_stop is not None
        return abs(0.5 * (candidate.coordinate_start + candidate.coordinate_stop) - target)

    return min(selected, key=lambda candidate: (distance(candidate), candidate.name))


def _candidate_support(
    block: VisibilityBlock,
    candidate: SkyVariationCandidate,
) -> np.ndarray:
    if candidate.kind == "temporal_interval":
        assert candidate.coordinate_start is not None
        assert candidate.coordinate_stop is not None
        return temporal_support_mask(
            block,
            start_s=candidate.coordinate_start,
            duration_s=candidate.coordinate_stop - candidate.coordinate_start,
        )
    if candidate.kind == "spectral_interval":
        assert candidate.start_index is not None and candidate.bin_count is not None
        return spectral_support_mask(
            block,
            first_channel=candidate.start_index,
            channel_count=candidate.bin_count,
        )
    raise ValueError("only interval candidates have Boolean support")


def _repeatable_base_amplitudes(
    summary: dict[str, Any],
    *,
    native_integration_s: float,
) -> dict[tuple[str, int], float]:
    amplitudes: dict[tuple[str, int], float] = {}
    for case in summary["transient"]:
        width = int(round(float(case["duration_s"]) / native_integration_s))
        amplitudes[("temporal_interval", width)] = _first_repeatable_amplitude(case)
    for case in summary["spectral"]:
        width = int(case["channel_count"])
        amplitudes[("spectral_interval", width)] = _first_repeatable_amplitude(case)
    return amplitudes


def _first_repeatable_amplitude(case: dict[str, Any]) -> float:
    for trial in case["trials"]:
        if (
            trial["selected_full_evaluation_model"] == "supported"
            and trial["paired_consensus_model"] == "supported"
            and float(trial["paired_supported_selection_fraction"]) >= 0.8
        ):
            return float(trial["injected_amplitude_jy"])
    raise ValueError("matched-support summary contains no repeatable amplitude")


def _load_unit_response(
    block: VisibilityBlock,
    response_path: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    metadata_path = response_path.with_suffix(".json")
    if not metadata_path.exists():
        raise ValueError("unit response has no provenance sidecar")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata["shape"] != list(block.shape):
        raise ValueError("unit response shape metadata does not match native block")
    if float(metadata["l_rad"]) != 0.0 or float(metadata["m_rad"]) != 0.0:
        raise ValueError("the initial blind search requires the beam-centre response")
    response = np.asarray(np.load(response_path), dtype=np.complex128)
    if response.shape != block.shape:
        raise ValueError("unit response array does not match native block")
    return response, metadata


def _run_repeated_search(
    block: VisibilityBlock,
    response: np.ndarray,
    candidates: tuple[SkyVariationCandidate, ...],
    *,
    true_candidate: SkyVariationCandidate | None,
    split_count: int,
    seed: int,
    selection_fraction: float,
    evaluation_fraction: float,
    shortlist_size: int,
) -> dict[str, Any]:
    folds = []
    for split_index in range(split_count):
        split_seed = seed + split_index
        split = split_search_baselines(
            block,
            selection_fraction=selection_fraction,
            evaluation_fraction=evaluation_fraction,
            seed=split_seed,
        )
        result = blind_search_sky_variation(
            block,
            response,
            candidates,
            split,
            shortlist_size=shortlist_size,
        )
        exact = true_candidate is not None and result.selected_candidate == true_candidate
        family = (
            true_candidate is not None
            and result.selected_candidate is not None
            and result.selected_candidate.kind == true_candidate.kind
        )
        folds.append(
            {
                "seed": split_seed,
                "baseline_counts": {
                    "discovery": len(split.discovery_baselines),
                    "selection": len(split.selection_baselines),
                    "evaluation": len(split.evaluation_baselines),
                },
                "exact_candidate_selected": exact,
                "correct_family_selected": family,
                "exact_candidate_accepted": exact and result.accepted,
                "result": _score_payload(result),
            }
        )
    exact_selection = float(np.mean([fold["exact_candidate_selected"] for fold in folds]))
    exact_acceptance = float(np.mean([fold["exact_candidate_accepted"] for fold in folds]))
    return {
        "true_candidate": _candidate_payload(true_candidate),
        "split_count": split_count,
        "any_acceptance_fraction": float(
            np.mean([fold["result"]["accepted"] for fold in folds])
        ),
        "correct_family_selection_fraction": float(
            np.mean([fold["correct_family_selected"] for fold in folds])
        ),
        "exact_candidate_selection_fraction": exact_selection,
        "exact_candidate_acceptance_fraction": exact_acceptance,
        "repeatably_recovered": bool(exact_acceptance >= 0.8),
        "median_evaluation_relative_improvement": float(
            np.median(
                [fold["result"]["evaluation_relative_improvement"] for fold in folds]
            )
        ),
        "folds": folds,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--native-fixture",
        type=Path,
        default=Path("outputs/3c391_native_averaging_ablation/native_C1.zarr"),
    )
    parser.add_argument(
        "--unit-response",
        type=Path,
        default=Path("outputs/3c391_native_injection_recovery/unit_response.npy"),
    )
    parser.add_argument(
        "--matched-summary",
        type=Path,
        default=Path("outputs/3c391_native_injection_recovery/summary.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/3c391_native_variability_search"),
    )
    parser.add_argument(
        "--amplitude-multipliers",
        type=_parse_positive_values,
        default=(1.0, 2.0, 4.0, 8.0, 16.0),
    )
    parser.add_argument(
        "--slope-edge-difference-mjy",
        type=_parse_positive_values,
        default=(2.0, 5.0, 10.0),
    )
    parser.add_argument("--split-count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=391)
    parser.add_argument("--selection-baseline-fraction", type=float, default=0.2)
    parser.add_argument("--evaluation-baseline-fraction", type=float, default=0.2)
    parser.add_argument("--shortlist-size", type=int, default=16)
    arguments = parser.parse_args()
    if arguments.split_count < 1:
        parser.error("--split-count must be positive")
    if arguments.shortlist_size < 1:
        parser.error("--shortlist-size must be positive")

    stored = read_dataset(arguments.native_fixture).blocks
    if len(stored) != 1 or stored[0].model_visibility is None:
        raise ValueError("native fixture must contain one block with a frozen prediction")
    block = stored[0]
    response, response_metadata = _load_unit_response(block, arguments.unit_response)
    candidates = native_variation_candidates(block)
    native_integration_s = float(np.median(block.interval_s))
    matched_summary = json.loads(arguments.matched_summary.read_text(encoding="utf-8"))
    base_amplitudes = _repeatable_base_amplitudes(
        matched_summary,
        native_integration_s=native_integration_s,
    )
    common = {
        "split_count": arguments.split_count,
        "seed": arguments.seed,
        "selection_fraction": arguments.selection_baseline_fraction,
        "evaluation_fraction": arguments.evaluation_baseline_fraction,
        "shortlist_size": arguments.shortlist_size,
    }

    print(f"Null search over {len(candidates)} candidates", flush=True)
    null = _run_repeated_search(
        block,
        response,
        candidates,
        true_candidate=None,
        **common,
    )
    interval_cases = []
    for kind, widths in (
        ("temporal_interval", (1, 3, 6)),
        ("spectral_interval", (1, 2, 4, 8)),
    ):
        for width in widths:
            true_candidate = _central_candidate(
                candidates,
                block,
                kind=kind,
                width=width,
            )
            support = _candidate_support(block, true_candidate)
            base_amplitude = base_amplitudes[(kind, width)]
            for multiplier in arguments.amplitude_multipliers:
                amplitude = float(base_amplitude * multiplier)
                print(
                    f"{true_candidate.name}: {amplitude * 1e3:.3f} mJy",
                    flush=True,
                )
                injected = inject_sky_component(
                    block,
                    response,
                    support,
                    amplitude,
                )
                result = _run_repeated_search(
                    injected,
                    response,
                    candidates,
                    true_candidate=true_candidate,
                    **common,
                )
                result.update(
                    {
                        "kind": kind,
                        "width": width,
                        "matched_support_base_amplitude_jy": base_amplitude,
                        "amplitude_multiplier": multiplier,
                        "injected_amplitude_jy": amplitude,
                    }
                )
                interval_cases.append(result)

    slope_candidate = next(
        candidate for candidate in candidates if candidate.kind == "spectral_slope"
    )
    frequency_ratio_log = float(
        np.log(np.max(block.frequency_hz) / np.min(block.frequency_hz))
    )
    reference_frequency = float(np.exp(np.mean(np.log(block.frequency_hz))))
    slope_multiplier = np.log(block.frequency_hz / reference_frequency)[None, :, None]
    slope_cases = []
    for edge_difference_mjy in arguments.slope_edge_difference_mjy:
        coefficient = float(edge_difference_mjy * 1e-3 / frequency_ratio_log)
        print(f"spectral slope: {edge_difference_mjy:g} mJy edge difference", flush=True)
        injected = replace(
            block,
            visibility=block.visibility + coefficient * response * slope_multiplier,
        )
        result = _run_repeated_search(
            injected,
            response,
            candidates,
            true_candidate=slope_candidate,
            **common,
        )
        result.update(
            {
                "injected_coefficient_jy_per_log_frequency": coefficient,
                "injected_edge_to_edge_difference_mjy": edge_difference_mjy,
            }
        )
        slope_cases.append(result)

    summary = {
        "protocol": {
            "native_fixture": str(arguments.native_fixture),
            "unit_response": str(arguments.unit_response),
            "matched_summary": str(arguments.matched_summary),
            "response_metadata": response_metadata,
            "candidate_generation": "coordinates only; no visibility values",
            "selection_protocol": (
                "discovery ranking, selection shortlist, sealed evaluation; whole baselines"
            ),
            "split_count": arguments.split_count,
            "seed": arguments.seed,
            "selection_baseline_fraction": arguments.selection_baseline_fraction,
            "evaluation_baseline_fraction": arguments.evaluation_baseline_fraction,
            "shortlist_size": arguments.shortlist_size,
            "amplitude_multipliers": list(arguments.amplitude_multipliers),
            "slope_edge_difference_mjy": list(arguments.slope_edge_difference_mjy),
            "temporal_width_integrations": [1, 3, 6],
            "spectral_width_channels": [1, 2, 4, 8],
        },
        "candidate_bank": {
            "count": len(candidates),
            "temporal_interval_count": sum(
                candidate.kind == "temporal_interval" for candidate in candidates
            ),
            "spectral_interval_count": sum(
                candidate.kind == "spectral_interval" for candidate in candidates
            ),
            "spectral_slope_count": sum(
                candidate.kind == "spectral_slope" for candidate in candidates
            ),
        },
        "null": null,
        "interval_injections": interval_cases,
        "slope_injections": slope_cases,
    }
    arguments.output.mkdir(parents=True, exist_ok=True)
    destination = arguments.output / "summary.json"
    destination.write_text(
        json.dumps(summary, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {destination}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
