#!/usr/bin/env python3
"""Compare structured explanations for the native 3C391 scan-29 residual.

The spatially blind search found a one-minute decrement near the end of C1
scan 29.  This diagnostic extracts the complete five-minute native scan and
keeps the frozen sky fixed.  It gives every explanation the same static flux
scale and local-leaf nuisance terms, then compares:

* a local sky change over the discovered minute;
* a common multiplicative amplitude change over that minute;
* a common primary-beam pointing shift over that minute;
* per-antenna complex gain changes over that minute.

Models are fitted on discovery baselines and selected on disjoint baselines.
Only the selected family is refitted and exposed to the sealed evaluation
baselines.  This tests whether a coherent sky/calibration explanation predicts
antenna pairs that did not contribute to its fit.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from ablate_3c391_native_averaging import _selected_column
from casacore import tables
from compare_3c391_composite_existing_flags import _components_from_checkpoint
from image_3c391_target import _block, _concatenate, _select_rows
from search_3c391_native_spatial_variability import (
    _load_topology,
    _unit_leaf_response,
)

from sl1mjax.beam import VLABeamCatalog, VLAPrimaryBeam
from sl1mjax.calibration import apply_calibration, read_calibration
from sl1mjax.composite import MosaicSkyComponent, predict_mosaic_composite
from sl1mjax.data.canonical import (
    VisibilityBlock,
    VisibilityDataset,
    read_dataset,
    write_dataset,
)
from sl1mjax.direct_operator import DirectDFTConfig
from sl1mjax.quadtree import QuadtreeLeaf, QuadtreeTopology
from sl1mjax.residual_models import (
    RealLinearSufficientStatistics,
    add_real_linear_statistics,
    empty_real_linear_statistics,
    fit_real_linear_statistics,
    real_linear_statistics,
    scan_residual_response_matrix,
    score_real_linear_fit,
)
from sl1mjax.sky_recovery import split_search_baselines

FAMILIES = (
    "static_nuisance",
    "local_sky_event",
    "common_amplitude_event",
    "common_pointing_event",
    "antenna_gain_event",
)


@dataclass(frozen=True)
class CandidateDefinition:
    """The spatial leaf and time interval discovered before this diagnostic."""

    leaf: QuadtreeLeaf
    event_start_s: float
    event_stop_s: float
    source: str


def _candidate_definition(path: Path) -> CandidateDefinition:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = payload["result"]
    leaf = result["selected_leaf"]
    variation = result["selected_variation"]
    if leaf is None or variation is None:
        raise ValueError("candidate result did not select a spatial variation")
    if variation["kind"] != "temporal_interval":
        raise ValueError("scan diagnostic requires a temporal candidate")
    return CandidateDefinition(
        leaf=QuadtreeLeaf(
            int(leaf["level"]),
            int(leaf["iy"]),
            int(leaf["ix"]),
        ),
        event_start_s=float(variation["coordinate_start"]),
        event_stop_s=float(variation["coordinate_stop"]),
        source=str(path),
    )


def _extract_native_scan(
    measurement_set: Path,
    solution: Any,
    *,
    field_id: int,
    scan_id: int,
    chunk_rows: int,
) -> VisibilityBlock:
    """Read, calibrate, and retain all active native samples in one scan."""

    chunks: list[VisibilityBlock] = []
    with (
        tables.table(str(measurement_set), readonly=True, ack=False) as main,
        tables.table(
            str(measurement_set / "SPECTRAL_WINDOW"), readonly=True, ack=False
        ) as spectral_window,
        tables.table(str(measurement_set / "FIELD"), readonly=True, ack=False) as field,
    ):
        selected = main.query(f"FIELD_ID=={field_id} && SCAN_NUMBER=={scan_id}")
        try:
            if selected.nrows() == 0:
                raise ValueError(f"field {field_id}, scan {scan_id} has no rows")
            data_description_ids = np.unique(selected.getcol("DATA_DESC_ID"))
            if not np.array_equal(data_description_ids, [0]):
                raise ValueError(
                    "scan diagnostic currently requires DATA_DESC_ID 0 only"
                )
            frequency_hz = np.asarray(
                spectral_window.getcell("CHAN_FREQ", 0), dtype=np.float64
            )
            direction = np.asarray(
                field.getcell("PHASE_DIR", field_id), dtype=np.float64
            ).reshape(-1, 2)[0]
            phase_centre = (float(direction[0]), float(direction[1]))
            for start in range(0, selected.nrows(), chunk_rows):
                count = min(chunk_rows, selected.nrows() - start)
                keep = np.ones(count, dtype=bool)
                time_s = np.asarray(
                    selected.getcol("TIME", startrow=start, nrow=count),
                    dtype=np.float64,
                )
                raw = _block(
                    visibility=_selected_column(
                        selected, "DATA", start=start, count=count, keep=keep
                    ),
                    flag=_selected_column(
                        selected,
                        "FLAG",
                        start=start,
                        count=count,
                        keep=keep,
                        dtype=bool,
                    ),
                    flag_row=_selected_column(
                        selected,
                        "FLAG_ROW",
                        start=start,
                        count=count,
                        keep=keep,
                        dtype=bool,
                    ),
                    weight=_selected_column(
                        selected,
                        "WEIGHT",
                        start=start,
                        count=count,
                        keep=keep,
                        dtype=np.float64,
                    ),
                    uvw_m=_selected_column(
                        selected,
                        "UVW",
                        start=start,
                        count=count,
                        keep=keep,
                        dtype=np.float64,
                    ),
                    frequency_hz=frequency_hz,
                    time_s=time_s,
                    antenna1=_selected_column(
                        selected,
                        "ANTENNA1",
                        start=start,
                        count=count,
                        keep=keep,
                        dtype=np.int32,
                    ),
                    antenna2=_selected_column(
                        selected,
                        "ANTENNA2",
                        start=start,
                        count=count,
                        keep=keep,
                        dtype=np.int32,
                    ),
                    scan_id=_selected_column(
                        selected,
                        "SCAN_NUMBER",
                        start=start,
                        count=count,
                        keep=keep,
                        dtype=np.int32,
                    ),
                    field_id=field_id,
                    interval_s=_selected_column(
                        selected,
                        "INTERVAL",
                        start=start,
                        count=count,
                        keep=keep,
                        dtype=np.float64,
                    ),
                    phase_centre_rad=phase_centre,
                    column="DATA",
                )
                chunks.append(
                    apply_calibration(
                        raw,
                        solution,
                        extrapolate=True,
                        propagate_weights=False,
                    )
                )
        finally:
            selected.close()
    block = _concatenate(
        chunks,
        label=f"native_field_{field_id}_scan_{scan_id}",
    )
    block = _select_rows(block, np.any(block.active, axis=(1, 2)))
    if np.any(block.field_id != field_id) or np.any(block.scan_id != scan_id):
        raise RuntimeError("native scan extraction escaped its requested field or scan")
    return block


def _load_or_extract_scan(
    arguments: argparse.Namespace,
    solution: Any,
) -> VisibilityBlock:
    fixture = arguments.output / "native_scan.zarr"
    if fixture.exists():
        stored = read_dataset(fixture).blocks
        if len(stored) != 1:
            raise ValueError("native scan fixture must contain one block")
        print(f"Resumed {fixture}", flush=True)
        return stored[0]
    print(
        f"Extracting field {arguments.field_id}, scan {arguments.scan_id}",
        flush=True,
    )
    block = _extract_native_scan(
        arguments.measurement_set,
        solution,
        field_id=arguments.field_id,
        scan_id=arguments.scan_id,
        chunk_rows=arguments.chunk_rows,
    )
    write_dataset(
        VisibilityDataset(
            (block,),
            provenance={
                "experiment": "3c391_scan_residual_diagnostic",
                "field_id": arguments.field_id,
                "scan_id": arguments.scan_id,
                "state": "calibrated_native_scan",
            },
        ),
        fixture,
    )
    return block


def _load_topology_from_protocol(
    protocol: dict[str, Any],
) -> QuadtreeTopology:
    frozen_directory = Path(protocol["frozen_directory"])
    frozen_summary = json.loads(
        (frozen_directory / "summary.json").read_text(encoding="utf-8")
    )
    return _load_topology(
        frozen_directory / "consensus_topology.csv",
        root_size=int(frozen_summary["root_size"]),
        root_pixel_size_rad=np.deg2rad(
            float(frozen_summary["root_pixel_arcsec"]) / 3600.0
        ),
    )


def _beam(
    protocol: dict[str, Any], *, pointing_lm: tuple[float, float]
) -> VLAPrimaryBeam:
    return VLAPrimaryBeam(
        kind="airy",
        pointing_lm=pointing_lm,
        catalog=replace(
            VLABeamCatalog(),
            airy_max_radius_rad_at_1ghz=np.deg2rad(
                float(protocol["airy_max_radius_deg_at_1ghz"])
            ),
        ),
    )


def _load_or_predict_responses(
    arguments: argparse.Namespace,
    block: VisibilityBlock,
    candidate: CandidateDefinition,
    protocol: dict[str, Any],
    topology: QuadtreeTopology,
    components: tuple[MosaicSkyComponent, ...],
    direct: DirectDFTConfig,
) -> tuple[VisibilityBlock, np.ndarray, np.ndarray, np.ndarray]:
    fixture = arguments.output / "native_scan.zarr"
    legacy_cache = arguments.output / (
        f"structured_responses_l{candidate.leaf.level}_"
        f"y{candidate.leaf.iy}_x{candidate.leaf.ix}.npz"
    )
    local_cache = arguments.output / (
        f"local_response_l{candidate.leaf.level}_"
        f"y{candidate.leaf.iy}_x{candidate.leaf.ix}.npz"
    )
    pointing_cache = arguments.output / "pointing_responses.npz"
    if block.model_visibility is None:
        print(
            f"Predicting frozen sky for {np.count_nonzero(block.active):,} active samples",
            flush=True,
        )
        prediction = predict_mosaic_composite(
            (block,),
            components,
            block.phase_centre_rad,
            primary_beam=_beam(protocol, pointing_lm=(0.0, 0.0)),
            config=direct,
        )[0]
        block = replace(block, model_visibility=prediction)
        write_dataset(
            VisibilityDataset(
                (block,),
                provenance={
                    "experiment": "3c391_scan_residual_diagnostic",
                    "field_id": arguments.field_id,
                    "scan_id": arguments.scan_id,
                    "state": "calibrated_native_scan_with_frozen_prediction",
                },
            ),
            fixture,
        )
    assert block.model_visibility is not None
    if legacy_cache.exists() and not (local_cache.exists() and pointing_cache.exists()):
        with np.load(legacy_cache) as stored:
            cached_leaf = tuple(int(value) for value in stored["leaf"])
            cached_step = float(stored["pointing_step_arcsec"])
            if cached_leaf != (
                candidate.leaf.level,
                candidate.leaf.iy,
                candidate.leaf.ix,
            ):
                raise ValueError("cached response uses a different spatial leaf")
            if not np.isclose(cached_step, arguments.pointing_step_arcsec):
                raise ValueError("cached response uses a different pointing step")
            np.savez(
                local_cache,
                leaf=stored["leaf"],
                local_sky_response=stored["local_sky_response"],
            )
            np.savez(
                pointing_cache,
                pointing_step_arcsec=cached_step,
                pointing_l_response_per_arcsec=stored["pointing_l_response_per_arcsec"],
                pointing_m_response_per_arcsec=stored["pointing_m_response_per_arcsec"],
            )
        print(f"Migrated {legacy_cache} into shared response caches", flush=True)
    base_beam = _beam(protocol, pointing_lm=(0.0, 0.0))
    if local_cache.exists():
        with np.load(local_cache) as stored:
            cached_leaf = tuple(int(value) for value in stored["leaf"])
            if cached_leaf != (
                candidate.leaf.level,
                candidate.leaf.iy,
                candidate.leaf.ix,
            ):
                raise ValueError("cached local response uses a different spatial leaf")
            local_response = np.asarray(stored["local_sky_response"])
        print(f"Resumed {local_cache}", flush=True)
    else:
        print("Predicting exact unit response of selected sky leaf", flush=True)
        local_response = _unit_leaf_response(
            block,
            topology,
            candidate.leaf,
            base_beam,
            block.phase_centre_rad,
            direct,
        )
        np.savez(
            local_cache,
            leaf=np.asarray(
                [candidate.leaf.level, candidate.leaf.iy, candidate.leaf.ix],
                dtype=np.int32,
            ),
            local_sky_response=local_response,
        )
    if pointing_cache.exists():
        with np.load(pointing_cache) as stored:
            cached_step = float(stored["pointing_step_arcsec"])
            if not np.isclose(cached_step, arguments.pointing_step_arcsec):
                raise ValueError("cached pointing response uses a different step")
            pointing_responses = [
                np.asarray(stored["pointing_l_response_per_arcsec"]),
                np.asarray(stored["pointing_m_response_per_arcsec"]),
            ]
        print(f"Resumed {pointing_cache}", flush=True)
    else:
        step_rad = np.deg2rad(arguments.pointing_step_arcsec / 3600.0)
        pointing_responses = []
        for label, pointing_lm in (
            ("l", (step_rad, 0.0)),
            ("m", (0.0, step_rad)),
        ):
            print(
                f"Predicting frozen sky with +{arguments.pointing_step_arcsec:g} arcsec "
                f"pointing-{label} shift",
                flush=True,
            )
            shifted = predict_mosaic_composite(
                (replace(block, model_visibility=None),),
                components,
                block.phase_centre_rad,
                primary_beam=_beam(protocol, pointing_lm=pointing_lm),
                config=direct,
            )[0]
            pointing_responses.append(
                (shifted - block.model_visibility) / arguments.pointing_step_arcsec
            )
        np.savez(
            pointing_cache,
            pointing_step_arcsec=arguments.pointing_step_arcsec,
            pointing_l_response_per_arcsec=pointing_responses[0],
            pointing_m_response_per_arcsec=pointing_responses[1],
        )
    return block, local_response, pointing_responses[0], pointing_responses[1]


def _ridge_grid(family: str, requested: tuple[float, ...]) -> tuple[float, ...]:
    return requested if family == "antenna_gain_event" else (0.0,)


def _accumulate_family_statistics(
    block: VisibilityBlock,
    local_response: np.ndarray,
    pointing_l_response: np.ndarray,
    pointing_m_response: np.ndarray,
    event_rows: np.ndarray,
    split: Any,
    *,
    antenna_ids: tuple[int, ...],
    reference_antenna: int,
    row_batch_size: int,
) -> tuple[
    dict[str, tuple[str, ...]],
    dict[str, np.ndarray],
    dict[str, dict[str, RealLinearSufficientStatistics]],
]:
    assert block.model_visibility is not None
    parameter_names: dict[str, tuple[str, ...]] = {}
    penalties: dict[str, np.ndarray] = {}
    statistics: dict[str, dict[str, RealLinearSufficientStatistics]] = {}
    residual = block.visibility - block.model_visibility
    masks = {
        "discovery": split.discovery_mask,
        "selection": split.selection_mask,
        "evaluation": split.evaluation_mask,
    }
    for family in FAMILIES:
        for start in range(0, block.shape[0], row_batch_size):
            stop = min(block.shape[0], start + row_batch_size)
            names, responses, penalty = scan_residual_response_matrix(
                family,
                block.model_visibility[start:stop],
                local_response[start:stop],
                pointing_l_response[start:stop],
                pointing_m_response[start:stop],
                block.antenna1[start:stop],
                block.antenna2[start:stop],
                event_rows[start:stop],
                antenna_ids=antenna_ids,
                reference_antenna=reference_antenna,
            )
            if family not in parameter_names:
                parameter_names[family] = names
                penalties[family] = penalty
                statistics[family] = {
                    name: empty_real_linear_statistics(len(names)) for name in masks
                }
            for cohort, mask in masks.items():
                tile = real_linear_statistics(
                    residual[start:stop],
                    block.weight[start:stop],
                    mask[start:stop],
                    responses,
                )
                statistics[family][cohort] = add_real_linear_statistics(
                    statistics[family][cohort], tile
                )
        print(f"Accumulated {family}", flush=True)
    return parameter_names, penalties, statistics


def _fit_and_select(
    parameter_names: dict[str, tuple[str, ...]],
    penalties: dict[str, np.ndarray],
    statistics: dict[str, dict[str, RealLinearSufficientStatistics]],
    ridge_fractions: tuple[float, ...],
) -> tuple[dict[str, Any], str, float, np.ndarray, dict[str, Any]]:
    candidates: dict[str, Any] = {}
    best_key: tuple[float, str, float] | None = None
    best: tuple[str, float, np.ndarray] | None = None
    baseline_selection_mse: float | None = None
    for family in FAMILIES:
        discovery = statistics[family]["discovery"]
        selection = statistics[family]["selection"]
        for ridge in _ridge_grid(family, ridge_fractions):
            fit = fit_real_linear_statistics(
                discovery,
                ridge_fraction=ridge,
                penalty=penalties[family],
            )
            selection_power, selection_mse = score_real_linear_fit(
                selection, fit.coefficients
            )
            label = f"{family}:ridge={ridge:g}"
            candidates[label] = {
                "family": family,
                "ridge_fraction": ridge,
                "parameter_names": parameter_names[family],
                "discovery_coefficients": fit.coefficients.tolist(),
                "discovery_weighted_complex_mse": fit.weighted_complex_mse,
                "selection_residual_power": selection_power,
                "selection_weighted_complex_mse": selection_mse,
                "selection_relative_improvement_from_frozen": (
                    1.0 - selection_power / selection.residual_power
                ),
                "normal_rank": fit.rank,
            }
            if family == "static_nuisance":
                baseline_selection_mse = selection_mse
            key = (selection_mse, family, ridge)
            if best_key is None or key < best_key:
                best_key = key
                best = (family, ridge, fit.coefficients)
    assert best is not None and baseline_selection_mse is not None
    family, ridge, _ = best
    combined = add_real_linear_statistics(
        statistics[family]["discovery"],
        statistics[family]["selection"],
    )
    refit = fit_real_linear_statistics(
        combined,
        ridge_fraction=ridge,
        penalty=penalties[family],
    )
    evaluation = statistics[family]["evaluation"]
    evaluation_power, evaluation_mse = score_real_linear_fit(
        evaluation, refit.coefficients
    )
    baseline_combined = add_real_linear_statistics(
        statistics["static_nuisance"]["discovery"],
        statistics["static_nuisance"]["selection"],
    )
    baseline_refit = fit_real_linear_statistics(
        baseline_combined,
        ridge_fraction=0.0,
        penalty=penalties["static_nuisance"],
    )
    baseline_evaluation = statistics["static_nuisance"]["evaluation"]
    baseline_evaluation_power, baseline_evaluation_mse = score_real_linear_fit(
        baseline_evaluation,
        baseline_refit.coefficients,
    )
    sealed = {
        "family": family,
        "ridge_fraction": ridge,
        "parameter_names": parameter_names[family],
        "refit_coefficients": refit.coefficients.tolist(),
        "frozen_residual_power": evaluation.residual_power,
        "frozen_weighted_complex_mse": (
            evaluation.residual_power / evaluation.weight_sum
        ),
        "static_nuisance_parameter_names": parameter_names["static_nuisance"],
        "static_nuisance_refit_coefficients": (baseline_refit.coefficients.tolist()),
        "static_nuisance_residual_power": baseline_evaluation_power,
        "static_nuisance_weighted_complex_mse": baseline_evaluation_mse,
        "selected_residual_power": evaluation_power,
        "selected_weighted_complex_mse": evaluation_mse,
        "relative_improvement_from_frozen": (
            1.0 - evaluation_power / evaluation.residual_power
        ),
        "relative_improvement_from_static_nuisance": (
            1.0 - evaluation_power / baseline_evaluation_power
        ),
        "sample_count": evaluation.sample_count,
        "weight_sum": evaluation.weight_sum,
    }
    selection = {
        "selected_family": family,
        "selected_ridge_fraction": ridge,
        "selected_improves_on_static_nuisance": (best_key[0] < baseline_selection_mse),
        "static_nuisance_selection_weighted_complex_mse": baseline_selection_mse,
        "selected_selection_weighted_complex_mse": best_key[0],
    }
    return (
        candidates,
        family,
        ridge,
        refit.coefficients,
        {"selection": selection, "sealed": sealed},
    )


def _sealed_time_series(
    block: VisibilityBlock,
    family: str,
    coefficients: np.ndarray,
    local_response: np.ndarray,
    pointing_l_response: np.ndarray,
    pointing_m_response: np.ndarray,
    event_rows: np.ndarray,
    evaluation_mask: np.ndarray,
    *,
    antenna_ids: tuple[int, ...],
    reference_antenna: int,
    row_batch_size: int,
) -> list[dict[str, Any]]:
    assert block.model_visibility is not None
    residual = block.visibility - block.model_visibility
    corrected = np.empty_like(residual)
    for start in range(0, block.shape[0], row_batch_size):
        stop = min(block.shape[0], start + row_batch_size)
        _, responses, _ = scan_residual_response_matrix(
            family,
            block.model_visibility[start:stop],
            local_response[start:stop],
            pointing_l_response[start:stop],
            pointing_m_response[start:stop],
            block.antenna1[start:stop],
            block.antenna2[start:stop],
            event_rows[start:stop],
            antenna_ids=antenna_ids,
            reference_antenna=reference_antenna,
        )
        corrected[start:stop] = residual[start:stop] - responses @ coefficients
    rows: list[dict[str, Any]] = []
    for time_s in np.unique(block.time_s):
        selected = evaluation_mask & (block.time_s == time_s)[:, None, None]
        weight = block.weight[selected]
        if weight.size == 0:
            continue
        frozen_power = float(np.sum(weight * np.abs(residual[selected]) ** 2))
        selected_power = float(np.sum(weight * np.abs(corrected[selected]) ** 2))
        weight_sum = float(np.sum(weight))
        rows.append(
            {
                "time_s": float(time_s),
                "offset_from_scan_start_s": float(time_s - np.min(block.time_s)),
                "in_event": bool(np.any(event_rows & (block.time_s == time_s))),
                "sample_count": int(np.count_nonzero(selected)),
                "weight_sum": weight_sum,
                "frozen_residual_power": frozen_power,
                "selected_residual_power": selected_power,
                "frozen_weighted_complex_mse": frozen_power / weight_sum,
                "selected_weighted_complex_mse": selected_power / weight_sum,
                "relative_improvement": (
                    1.0 - selected_power / frozen_power
                    if frozen_power > 0
                    else float("nan")
                ),
            }
        )
    return rows


def _plot_summary(payload: dict[str, Any], path: Path) -> None:
    candidates = payload["model_candidates"]
    best_by_family: dict[str, float] = {}
    for candidate in candidates.values():
        family = candidate["family"]
        mse = candidate["selection_weighted_complex_mse"]
        best_by_family[family] = min(mse, best_by_family.get(family, np.inf))
    time_series = payload["sealed_time_series"]
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.5), constrained_layout=True)
    labels = list(best_by_family)
    axes[0].barh(labels, [best_by_family[label] for label in labels])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Selection weighted complex MSE")
    axes[0].set_title("Model-family comparison")
    axes[0].grid(axis="x", alpha=0.25)
    offset = [row["offset_from_scan_start_s"] for row in time_series]
    axes[1].plot(
        offset,
        [row["frozen_weighted_complex_mse"] for row in time_series],
        marker="o",
        label="Frozen sky",
    )
    axes[1].plot(
        offset,
        [row["selected_weighted_complex_mse"] for row in time_series],
        marker="o",
        label="Selected model",
    )
    event_offsets = [
        row["offset_from_scan_start_s"] for row in time_series if row["in_event"]
    ]
    if event_offsets:
        axes[1].axvspan(min(event_offsets) - 5, max(event_offsets) + 5, alpha=0.15)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Seconds from scan start")
    axes[1].set_ylabel("Sealed-baseline weighted complex MSE")
    axes[1].set_title("Residual through the five-minute scan")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _parse_float_tuple(value: str) -> tuple[float, ...]:
    try:
        selected = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "values must be comma-separated floats"
        ) from error
    if not selected or any(not np.isfinite(item) or item < 0 for item in selected):
        raise argparse.ArgumentTypeError("values must be finite and non-negative")
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--measurement-set",
        type=Path,
        default=Path("data/3c391_work_v2/3c391_ctm_mosaic_10s_spw0.ms"),
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path("outputs/3c391_full_scan_gain_baseline/full_scan_calibration.npz"),
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
        "--candidate-result",
        type=Path,
        default=Path(
            "outputs/3c391_native_spatial_variability_search/null_seed391.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/3c391_scan29_residual_diagnostic"),
    )
    parser.add_argument("--field-id", type=int, default=2)
    parser.add_argument("--scan-id", type=int, default=29)
    parser.add_argument("--seed", type=int, default=391)
    parser.add_argument("--selection-baseline-fraction", type=float, default=0.2)
    parser.add_argument("--evaluation-baseline-fraction", type=float, default=0.2)
    parser.add_argument("--pointing-step-arcsec", type=float, default=10.0)
    parser.add_argument(
        "--gain-ridge-fractions",
        type=_parse_float_tuple,
        default=_parse_float_tuple("0,1e-8,1e-6,1e-4,1e-2,1"),
    )
    parser.add_argument("--chunk-rows", type=int, default=2048)
    parser.add_argument("--row-batch-size", type=int, default=256)
    parser.add_argument("--visibility-tile-size", type=int, default=256)
    parser.add_argument("--pixel-tile-size", type=int, default=512)
    parser.add_argument(
        "--precision", choices=("float32", "float64"), default="float32"
    )
    arguments = parser.parse_args()
    if arguments.chunk_rows < 1 or arguments.row_batch_size < 1:
        parser.error("row chunk and batch sizes must be positive")
    if arguments.pointing_step_arcsec <= 0:
        parser.error("--pointing-step-arcsec must be positive")
    arguments.output.mkdir(parents=True, exist_ok=True)
    destination = arguments.output / f"seed{arguments.seed}.json"
    if destination.exists():
        print(f"Already complete: {destination}", flush=True)
        return 0

    candidate = _candidate_definition(arguments.candidate_result)
    protocol = json.loads(arguments.sky_protocol.read_text(encoding="utf-8"))
    solution = read_calibration(arguments.calibration)
    block = _load_or_extract_scan(arguments, solution)
    if candidate.leaf not in _load_topology_from_protocol(protocol).leaves:
        raise ValueError("selected candidate leaf is not in the frozen topology")
    event_rows = (block.time_s >= candidate.event_start_s) & (
        block.time_s < candidate.event_stop_s
    )
    if not np.any(event_rows) or np.all(event_rows):
        raise ValueError("candidate event must cover some, but not all, scan rows")
    topology = _load_topology_from_protocol(protocol)
    components = _components_from_checkpoint(
        arguments.sky_checkpoint,
        protocol,
        block.phase_centre_rad,
    )
    direct = DirectDFTConfig(
        visibility_chunk_size=arguments.visibility_tile_size,
        pixel_chunk_size=arguments.pixel_tile_size,
        precision=arguments.precision,
    )
    block, local_response, pointing_l_response, pointing_m_response = (
        _load_or_predict_responses(
            arguments,
            block,
            candidate,
            protocol,
            topology,
            components,
            direct,
        )
    )
    split = split_search_baselines(
        block,
        selection_fraction=arguments.selection_baseline_fraction,
        evaluation_fraction=arguments.evaluation_baseline_fraction,
        seed=arguments.seed,
    )
    antenna_ids = tuple(
        int(value) for value in np.unique(np.r_[block.antenna1, block.antenna2])
    )
    reference_antenna = int(solution.reference_antenna)
    parameter_names, penalties, statistics = _accumulate_family_statistics(
        block,
        local_response,
        pointing_l_response,
        pointing_m_response,
        event_rows,
        split,
        antenna_ids=antenna_ids,
        reference_antenna=reference_antenna,
        row_batch_size=arguments.row_batch_size,
    )
    candidates, selected_family, _, coefficients, decision = _fit_and_select(
        parameter_names,
        penalties,
        statistics,
        arguments.gain_ridge_fractions,
    )
    time_series = _sealed_time_series(
        block,
        selected_family,
        coefficients,
        local_response,
        pointing_l_response,
        pointing_m_response,
        event_rows,
        split.evaluation_mask,
        antenna_ids=antenna_ids,
        reference_antenna=reference_antenna,
        row_batch_size=arguments.row_batch_size,
    )
    payload = {
        "protocol": {
            "measurement_set": str(arguments.measurement_set),
            "calibration": str(arguments.calibration),
            "sky_protocol": str(arguments.sky_protocol),
            "sky_checkpoint": str(arguments.sky_checkpoint),
            "candidate": asdict(candidate),
            "field_id": arguments.field_id,
            "scan_id": arguments.scan_id,
            "seed": arguments.seed,
            "selection_baseline_fraction": arguments.selection_baseline_fraction,
            "evaluation_baseline_fraction": arguments.evaluation_baseline_fraction,
            "pointing_step_arcsec": arguments.pointing_step_arcsec,
            "gain_ridge_fractions": arguments.gain_ridge_fractions,
            "reference_antenna": reference_antenna,
            "antenna_ids": antenna_ids,
            "precision": arguments.precision,
        },
        "scan": {
            "row_count": block.shape[0],
            "channel_count": block.shape[1],
            "correlation_count": block.shape[2],
            "active_sample_count": int(np.count_nonzero(block.active)),
            "integration_count": int(np.unique(block.time_s).size),
            "start_s": float(np.min(block.time_s)),
            "stop_s": float(np.max(block.time_s)),
            "event_integration_count": int(np.unique(block.time_s[event_rows]).size),
            "event_active_sample_count": int(
                np.count_nonzero(block.active & event_rows[:, None, None])
            ),
        },
        "baseline_cohorts": {
            "discovery": split.discovery_baselines,
            "selection": split.selection_baselines,
            "evaluation": split.evaluation_baselines,
        },
        "model_candidates": candidates,
        **decision,
        "sealed_time_series": time_series,
    }
    destination.write_text(
        json.dumps(payload, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    _plot_summary(payload, arguments.output / f"seed{arguments.seed}.png")
    print(
        f"Selected {selected_family}; sealed relative improvement "
        f"{decision['sealed']['relative_improvement_from_frozen']:.4%}",
        flush=True,
    )
    print(f"Wrote {destination}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
