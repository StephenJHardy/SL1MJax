#!/usr/bin/env python3
"""Audit 3C391 model-residual tails with a frozen train/test protocol."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from sl1mjax.beam import primary_beam_from_name
from sl1mjax.data.canonical import VisibilityBlock, read_dataset
from sl1mjax.direct_operator import DirectDFTConfig
from sl1mjax.inference import InferenceConfig, infer_mosaic_quadtree
from sl1mjax.quadtree import QuadtreeGrid, QuadtreeLeaf, QuadtreeSky, QuadtreeTopology
from sl1mjax.residual_audit import (
    ResidualGroupSummary,
    audit_visibility_residuals,
    masks_excluding_groups,
)
from sl1mjax.sky import GaussianApproximation


def _load_sky(
    path: Path, *, root_size: int, root_pixel_size_rad: float
) -> QuadtreeSky:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = tuple(csv.DictReader(stream))
    if not rows:
        raise ValueError("topology CSV must contain at least one leaf")
    flux_by_leaf = {
        QuadtreeLeaf(int(row["level"]), int(row["iy"]), int(row["ix"])): float(
            row["flux_jy"]
        )
        for row in rows
    }
    if len(flux_by_leaf) != len(rows):
        raise ValueError("topology CSV contains duplicate leaves")
    topology = QuadtreeTopology(
        QuadtreeGrid(root_size, root_pixel_size_rad), tuple(flux_by_leaf)
    )
    flux = np.asarray([flux_by_leaf[leaf] for leaf in topology.leaves])
    if np.any(~np.isfinite(flux)) or np.any(flux < 0):
        raise ValueError("topology flux must be finite and non-negative")
    return QuadtreeSky(topology.grid, topology.leaves, flux)


def _residual_metrics(
    blocks: tuple[VisibilityBlock, ...],
    predictions: tuple[np.ndarray, ...],
    masks: tuple[np.ndarray, ...],
) -> dict[str, float | int]:
    residual_power = 0.0
    signal_power = 0.0
    weight_sum = 0.0
    count = 0
    for block, prediction, mask in zip(blocks, predictions, masks, strict=True):
        selected = mask & block.active
        residual_power += float(
            np.sum(
                block.weight[selected]
                * np.abs(block.visibility[selected] - prediction[selected]) ** 2
            )
        )
        signal_power += float(
            np.sum(block.weight[selected] * np.abs(block.visibility[selected]) ** 2)
        )
        weight_sum += float(np.sum(block.weight[selected]))
        count += int(np.count_nonzero(selected))
    if count == 0 or weight_sum <= 0 or signal_power <= 0:
        raise ValueError("residual metric mask contains no usable samples")
    return {
        "sample_count": count,
        "weighted_complex_mse": residual_power / weight_sum,
        "normalized_residual_power": residual_power / signal_power,
    }


def _prediction_comparison(
    blocks: tuple[VisibilityBlock, ...],
    original: tuple[np.ndarray, ...],
    candidate: tuple[np.ndarray, ...],
    discovery_masks: tuple[np.ndarray, ...],
    evaluation_masks: tuple[np.ndarray, ...],
    retained_discovery_masks: tuple[np.ndarray, ...],
    retained_evaluation_masks: tuple[np.ndarray, ...],
) -> dict[str, Any]:
    excluded_discovery = tuple(
        mask & ~retained
        for mask, retained in zip(discovery_masks, retained_discovery_masks, strict=True)
    )
    excluded_evaluation = tuple(
        mask & ~retained
        for mask, retained in zip(evaluation_masks, retained_evaluation_masks, strict=True)
    )
    partitions = {
        "all_discovery": discovery_masks,
        "retained_discovery": retained_discovery_masks,
        "excluded_discovery": excluded_discovery,
        "all_evaluation": evaluation_masks,
        "retained_evaluation": retained_evaluation_masks,
        "excluded_evaluation": excluded_evaluation,
    }
    result: dict[str, Any] = {}
    for label, masks in partitions.items():
        if not any(np.any(mask) for mask in masks):
            continue
        before = _residual_metrics(blocks, original, masks)
        after = _residual_metrics(blocks, candidate, masks)
        result[label] = {
            "original": before,
            "candidate": after,
            "relative_normalized_residual_power_change": (
                after["normalized_residual_power"]
                / before["normalized_residual_power"]
                - 1.0
            ),
        }
    return result


def _antenna_names(path: Path | None) -> dict[int, str]:
    if path is None:
        return {}
    try:
        from casacore.tables import table
    except ImportError as error:  # pragma: no cover - optional local dependency
        raise RuntimeError("antenna names require the optional MeasurementSet extra") from error
    with table(str(path / "ANTENNA"), ack=False, readonly=True) as antennas:
        return {
            index: str(name)
            for index, name in enumerate(np.asarray(antennas.getcol("NAME")))
        }


def _display_label(group: ResidualGroupSummary, names: dict[int, str]) -> str:
    if group.kind == "baseline" and len(group.key) == 2:
        first = names.get(group.key[0], str(group.key[0]))
        second = names.get(group.key[1], str(group.key[1]))
        return f"{first}–{second}"
    if group.kind == "antenna" and len(group.key) == 1:
        return names.get(group.key[0], str(group.key[0]))
    return group.label


def _write_group_csv(
    path: Path,
    groups: tuple[ResidualGroupSummary, ...],
    names: dict[int, str],
    baseline_uv_klambda: dict[tuple[int, int], float],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "kind",
                "key",
                "label",
                "candidate",
                "validated",
                "median_uv_klambda",
                "discovery_samples",
                "discovery_outlier_fraction",
                "discovery_normalized_residual_power",
                "evaluation_samples",
                "evaluation_outlier_fraction",
                "evaluation_normalized_residual_power",
            )
        )
        for group in groups:
            writer.writerow(
                (
                    group.kind,
                    ":".join(map(str, group.key)),
                    _display_label(group, names),
                    int(group.candidate),
                    int(group.validated),
                    baseline_uv_klambda.get(group.key, "")
                    if group.kind == "baseline"
                    else "",
                    group.discovery.sample_count,
                    group.discovery.outlier_fraction,
                    group.discovery.normalized_residual_power,
                    group.evaluation.sample_count,
                    group.evaluation.outlier_fraction,
                    group.evaluation.normalized_residual_power,
                )
            )


def _plot_baselines(
    path: Path,
    groups: tuple[ResidualGroupSummary, ...],
    names: dict[int, str],
    *,
    maximum: int = 30,
) -> None:
    baselines = [
        group
        for group in groups
        if group.kind == "baseline"
        and group.discovery.sample_count > 0
        and group.evaluation.sample_count > 0
    ]
    baselines.sort(key=lambda group: group.discovery.outlier_fraction, reverse=True)
    selected = baselines[:maximum]
    positions = np.arange(len(selected))
    figure, axis = plt.subplots(figsize=(12, 6), constrained_layout=True)
    axis.bar(
        positions - 0.2,
        [group.discovery.outlier_fraction for group in selected],
        width=0.4,
        label="discovery scans",
    )
    axis.bar(
        positions + 0.2,
        [group.evaluation.outlier_fraction for group in selected],
        width=0.4,
        label="held-out scans",
    )
    axis.set_xticks(positions, [_display_label(group, names) for group in selected])
    axis.tick_params(axis="x", rotation=65)
    axis.set_ylabel("fraction above robust residual threshold")
    axis.set_title("3C391 baseline residual-tail transfer to held-out scans")
    axis.legend()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _baseline_uv_distances(
    blocks: tuple[VisibilityBlock, ...], groups: tuple[ResidualGroupSummary, ...]
) -> dict[tuple[int, int], float]:
    speed_of_light_m_s = 299_792_458.0
    result: dict[tuple[int, int], float] = {}
    for group in groups:
        if group.kind != "baseline" or len(group.key) != 2:
            continue
        distances: list[np.ndarray] = []
        for block in blocks:
            first = np.minimum(block.antenna1, block.antenna2)
            second = np.maximum(block.antenna1, block.antenna2)
            rows = (
                (first == group.key[0])
                & (second == group.key[1])
                & np.any(block.active, axis=(1, 2))
            )
            if np.any(rows):
                distances.append(
                    np.linalg.norm(block.uvw_m[rows, :2], axis=1)
                    * float(np.mean(block.frequency_hz))
                    / speed_of_light_m_s
                    / 1000.0
                )
        if distances:
            result[(group.key[0], group.key[1])] = float(
                np.median(np.concatenate(distances))
            )
    return result


def _uv_dependence(
    groups: tuple[ResidualGroupSummary, ...],
    baseline_uv_klambda: dict[tuple[int, int], float],
) -> dict[str, Any]:
    baselines = [
        group
        for group in groups
        if group.kind == "baseline" and group.key in baseline_uv_klambda
    ]
    distance = np.asarray([baseline_uv_klambda[group.key] for group in baselines])
    outlier_fraction = np.asarray(
        [group.discovery.outlier_fraction for group in baselines]
    )
    bins = (0.0, 0.75, 1.5, 3.0, 6.0, 12.0, np.inf)
    summaries = []
    for lower, upper in zip(bins[:-1], bins[1:], strict=True):
        selected = (distance >= lower) & (distance < upper)
        if not np.any(selected):
            continue
        summaries.append(
            {
                "lower_klambda": lower,
                "upper_klambda": None if np.isinf(upper) else upper,
                "baseline_count": int(np.count_nonzero(selected)),
                "median_discovery_outlier_fraction": float(
                    np.median(outlier_fraction[selected])
                ),
                "mean_discovery_outlier_fraction": float(
                    np.mean(outlier_fraction[selected])
                ),
            }
        )
    return {
        "baseline_count": len(baselines),
        "pearson_log_uv_vs_discovery_outlier_fraction": float(
            np.corrcoef(np.log(distance), outlier_fraction)[0, 1]
        ),
        "bins": summaries,
    }


def _plot_uv_dependence(
    path: Path,
    groups: tuple[ResidualGroupSummary, ...],
    names: dict[int, str],
    baseline_uv_klambda: dict[tuple[int, int], float],
) -> None:
    baselines = [
        group
        for group in groups
        if group.kind == "baseline" and group.key in baseline_uv_klambda
    ]
    distance = np.asarray([baseline_uv_klambda[group.key] for group in baselines])
    discovery = np.asarray([group.discovery.outlier_fraction for group in baselines])
    evaluation = np.asarray([group.evaluation.outlier_fraction for group in baselines])
    figure, axis = plt.subplots(figsize=(9, 6), constrained_layout=True)
    axis.scatter(distance, discovery, s=18, alpha=0.7, label="discovery scans")
    axis.scatter(distance, evaluation, s=18, alpha=0.7, label="held-out scans")
    for group, x, y in zip(baselines, distance, discovery, strict=True):
        if y >= 0.5:
            axis.annotate(_display_label(group, names), (x, y), fontsize=8)
    axis.set_xscale("log")
    axis.set_xlabel("median projected baseline distance (kλ)")
    axis.set_ylabel("fraction above robust residual threshold")
    axis.set_title("3C391 residual tail is concentrated on short baselines")
    axis.legend()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "fixture",
        type=Path,
        nargs="?",
        default=Path("outputs/3c391_mosaic_imaging_fixture.zarr"),
    )
    parser.add_argument(
        "--frozen-directory",
        type=Path,
        default=Path("outputs/3c391_mosaic_hierarchical_frozen_104"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/3c391_residual_flag_audit"),
    )
    parser.add_argument("--antenna-names-ms", type=Path)
    parser.add_argument("--score-threshold", type=float, default=6.0)
    parser.add_argument("--minimum-baseline-samples", type=int, default=128)
    parser.add_argument("--minimum-baseline-outlier-fraction", type=float, default=0.2)
    parser.add_argument("--maximum-baseline-candidates", type=int, default=8)
    parser.add_argument("--refit", action="store_true")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--visibility-tile-size", type=int, default=256)
    parser.add_argument("--pixel-tile-size", type=int, default=1024)
    parser.add_argument("--precision", choices=("float32", "float64"), default="float32")
    arguments = parser.parse_args()

    blocks = read_dataset(arguments.fixture).blocks
    with np.load(arguments.frozen_directory / "predictions.npz") as stored:
        predictions = tuple(
            stored[f"consensus_C{index + 1}"] for index in range(len(blocks))
        )
        discovery_masks = tuple(
            stored[f"outer_train_C{index + 1}"].astype(bool)
            for index in range(len(blocks))
        )
        evaluation_masks = tuple(
            stored[f"outer_test_C{index + 1}"].astype(bool)
            for index in range(len(blocks))
        )
    audit = audit_visibility_residuals(
        blocks,
        predictions,
        discovery_masks,
        evaluation_masks,
        score_threshold=arguments.score_threshold,
        minimum_group_samples=arguments.minimum_baseline_samples,
        minimum_group_outlier_fraction=arguments.minimum_baseline_outlier_fraction,
    )
    names = _antenna_names(arguments.antenna_names_ms)
    if arguments.maximum_baseline_candidates < 1:
        raise ValueError("maximum-baseline-candidates must be positive")
    baseline_candidates = tuple(
        group
        for group in audit.groups
        if group.kind == "baseline" and group.candidate
    )[: arguments.maximum_baseline_candidates]
    retained_discovery = masks_excluding_groups(
        blocks, discovery_masks, baseline_candidates
    )
    retained_evaluation = masks_excluding_groups(
        blocks, evaluation_masks, baseline_candidates
    )
    arguments.output.mkdir(parents=True, exist_ok=True)
    baseline_uv_klambda = _baseline_uv_distances(blocks, audit.groups)
    uv_dependence = _uv_dependence(audit.groups, baseline_uv_klambda)
    _write_group_csv(
        arguments.output / "groups.csv", audit.groups, names, baseline_uv_klambda
    )
    _plot_baselines(arguments.output / "baseline_outlier_transfer.jpg", audit.groups, names)
    _plot_uv_dependence(
        arguments.output / "baseline_outlier_vs_uv_distance.jpg",
        audit.groups,
        names,
        baseline_uv_klambda,
    )
    summary: dict[str, Any] = {
        "schema_version": 1,
        "fixture": str(arguments.fixture),
        "frozen_directory": str(arguments.frozen_directory),
        "score_definition": (
            "(sqrt(weight)*abs(observed-predicted)-discovery_median)"
            "/(1.4826*discovery_MAD)"
        ),
        "audit": asdict(audit),
        "baseline_uv_dependence": uv_dependence,
        "baseline_candidates_selected_from_discovery_only": [
            {
                "key": list(group.key),
                "label": _display_label(group, names),
                "validated_on_held_out_scans": group.validated,
                "discovery_outlier_fraction": group.discovery.outlier_fraction,
                "evaluation_outlier_fraction": group.evaluation.outlier_fraction,
            }
            for group in baseline_candidates
        ],
        "fixed_prediction_partitions": _prediction_comparison(
            blocks,
            predictions,
            predictions,
            discovery_masks,
            evaluation_masks,
            retained_discovery,
            retained_evaluation,
        ),
    }
    if arguments.refit:
        frozen_summary = json.loads(
            (arguments.frozen_directory / "summary.json").read_text(encoding="utf-8")
        )
        sky = _load_sky(
            arguments.frozen_directory / "consensus_topology.csv",
            root_size=int(frozen_summary["root_size"]),
            root_pixel_size_rad=np.deg2rad(
                float(frozen_summary["root_pixel_arcsec"]) / 3600.0
            ),
        )
        config = InferenceConfig(
            solver="fista",
            steps=arguments.steps,
            sparsity_weight=float(frozen_summary["lambda_l1"]),
            kkt_tolerance=float(frozen_summary["kkt_tolerance"]),
            validation_interval=25,
            operator_mode="explicit",
            direct_dft=DirectDFTConfig(
                visibility_chunk_size=arguments.visibility_tile_size,
                pixel_chunk_size=arguments.pixel_tile_size,
                precision=arguments.precision,
            ),
        )
        print(
            "fixed-topology refit: excluding "
            f"{len(baseline_candidates)} discovery-selected baselines",
            flush=True,
        )
        fit = infer_mosaic_quadtree(
            blocks,
            sky.topology,
            retained_discovery,
            blocks[0].phase_centre_rad,
            config,
            primary_beam=primary_beam_from_name("airy"),
            approximation=GaussianApproximation.WIDE_FIELD,
            initial_flux=sky.flux,
        )
        summary["refit"] = {
            "steps": fit.steps,
            "best_step": fit.best_step,
            "converged": fit.converged,
            "kkt_residual": fit.kkt_residual,
            "total_flux_jy": float(np.sum(fit.flux)),
            "comparison": _prediction_comparison(
                blocks,
                predictions,
                fit.predictions,
                discovery_masks,
                evaluation_masks,
                retained_discovery,
                retained_evaluation,
            ),
        }
        np.savez(
            arguments.output / "refit_predictions.npz",
            **{
                f"prediction_C{index + 1}": prediction
                for index, prediction in enumerate(fit.predictions)
            },
            **{
                f"retained_discovery_C{index + 1}": mask
                for index, mask in enumerate(retained_discovery)
            },
            **{
                f"retained_evaluation_C{index + 1}": mask
                for index, mask in enumerate(retained_evaluation)
            },
        )
    (arguments.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    concise = {
        "discovery": asdict(audit.discovery),
        "evaluation": asdict(audit.evaluation),
        "baseline_candidate_count": len(baseline_candidates),
        "validated_baseline_candidate_count": sum(
            group.validated for group in baseline_candidates
        ),
        "candidates": summary["baseline_candidates_selected_from_discovery_only"],
        "refit": summary.get("refit"),
    }
    print(json.dumps(concise, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
