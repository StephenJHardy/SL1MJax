#!/usr/bin/env python3
"""Measure time/frequency averaging loss against a frozen 3C391 sky.

The experiment starts from calibrated 10-second, 2 MHz target samples in a
sealed time fold.  For each requested averaging case it compares two forward
models against the same averaged observations:

* ``exact`` averages predictions made at every native coordinate;
* ``centroid`` evaluates the sky once at the averaged UVW and frequency.

The first measures how the frozen sky generalises at each data resolution.  The
second isolates the approximation made by the current coarse imaging fixture.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from casacore import tables
from compare_3c391_composite_existing_flags import _components_from_checkpoint
from image_3c391_target import _block, _concatenate, _select_rows

from sl1mjax.beam import VLABeamCatalog, VLAPrimaryBeam
from sl1mjax.calibration import apply_calibration, read_calibration
from sl1mjax.composite import MosaicSkyComponent, predict_mosaic_composite
from sl1mjax.data.averaging import average_frequency_bins, average_time_bins
from sl1mjax.data.canonical import (
    VisibilityBlock,
    VisibilityDataset,
    read_dataset,
    write_dataset,
)
from sl1mjax.direct_operator import DirectDFTConfig
from sl1mjax.split import interleaved_time_folds

SPEED_OF_LIGHT_M_S = 299_792_458.0
BASELINE_EDGES_KLAMBDA = (0.0, 2.5, 5.0, 7.5, 10.0, np.inf)


@dataclass(frozen=True)
class AveragingCase:
    """One rectangular time/frequency averaging case."""

    time_seconds: float
    channel_width_mhz: float

    @property
    def label(self) -> str:
        return f"{self.time_seconds:g}s_{self.channel_width_mhz:g}MHz"


@dataclass
class MetricMoments:
    """Additive moments used to combine fields without averaging ratios."""

    residual_power: float = 0.0
    signal_power: float = 0.0
    weight_sum: float = 0.0
    sample_count: int = 0

    def add(self, other: MetricMoments) -> None:
        self.residual_power += other.residual_power
        self.signal_power += other.signal_power
        self.weight_sum += other.weight_sum
        self.sample_count += other.sample_count


def _parse_cases(value: str) -> tuple[AveragingCase, ...]:
    """Parse comma-separated ``time_seconds:channel_width_mhz`` cases."""

    cases: list[AveragingCase] = []
    try:
        for item in value.split(","):
            time_s, width_mhz = (float(part.strip()) for part in item.split(":"))
            cases.append(AveragingCase(time_s, width_mhz))
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "averaging cases must use time_seconds:channel_width_mhz"
        ) from error
    if not cases or any(
        not np.isfinite(case.time_seconds)
        or case.time_seconds <= 0
        or not np.isfinite(case.channel_width_mhz)
        or case.channel_width_mhz <= 0
        for case in cases
    ):
        raise argparse.ArgumentTypeError("averaging cases must be finite and positive")
    if len({case.label for case in cases}) != len(cases):
        raise argparse.ArgumentTypeError("averaging cases must be unique")
    return tuple(cases)


def _holdout_time_bins(
    blocks: tuple[VisibilityBlock, ...],
    *,
    fold_bin_seconds: float,
    fold_count: int,
    holdout_fold: int,
) -> tuple[np.ndarray, ...]:
    """Return the exact fixed-width time bins assigned to one frozen fold."""

    if not 0 <= holdout_fold < fold_count:
        raise ValueError("holdout_fold must select one of the requested folds")
    folds = interleaved_time_folds(
        blocks,
        bin_seconds=fold_bin_seconds,
        fold_count=fold_count,
    )
    result = []
    for block, mask in zip(blocks, folds[holdout_fold], strict=True):
        rows = np.any(mask, axis=(1, 2))
        result.append(
            np.unique(np.floor(block.time_s[rows] / fold_bin_seconds).astype(np.int64))
        )
    return tuple(result)


def _selected_column(
    table: Any,
    name: str,
    *,
    start: int,
    count: int,
    keep: np.ndarray,
    dtype: Any | None = None,
) -> np.ndarray:
    """Read one row slice and retain the selected rows."""

    return np.asarray(
        table.getcol(name, startrow=start, nrow=count), dtype=dtype
    )[keep]


def _extract_native_holdout(
    measurement_set: Path,
    solution: Any,
    *,
    field_id: int,
    selected_time_bins: np.ndarray,
    fold_bin_seconds: float,
    chunk_rows: int,
) -> VisibilityBlock:
    """Read and calibrate only native rows belonging to selected time bins."""

    chunks: list[VisibilityBlock] = []
    with (
        tables.table(str(measurement_set), readonly=True, ack=False) as main,
        tables.table(
            str(measurement_set / "SPECTRAL_WINDOW"), readonly=True, ack=False
        ) as spectral_window,
        tables.table(str(measurement_set / "FIELD"), readonly=True, ack=False) as field,
    ):
        selected = main.query(f"FIELD_ID=={field_id}")
        try:
            if selected.nrows() == 0:
                raise ValueError(f"field {field_id} has no rows")
            frequency_hz = np.asarray(
                spectral_window.getcell("CHAN_FREQ", 0), dtype=np.float64
            )
            direction = np.asarray(
                field.getcell("PHASE_DIR", field_id), dtype=np.float64
            ).reshape(-1, 2)[0]
            phase_centre = (float(direction[0]), float(direction[1]))
            for start in range(0, selected.nrows(), chunk_rows):
                count = min(chunk_rows, selected.nrows() - start)
                time_s = np.asarray(
                    selected.getcol("TIME", startrow=start, nrow=count), dtype=np.float64
                )
                time_bin = np.floor(time_s / fold_bin_seconds).astype(np.int64)
                keep = np.isin(time_bin, selected_time_bins)
                if not np.any(keep):
                    continue

                raw = _block(
                    visibility=_selected_column(
                        selected, "DATA", start=start, count=count, keep=keep
                    ),
                    flag=_selected_column(
                        selected, "FLAG", start=start, count=count, keep=keep, dtype=bool
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
                    time_s=time_s[keep],
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
    if not chunks:
        raise ValueError(f"field {field_id} has no rows in the selected time fold")
    block = _concatenate(chunks, label=f"native_holdout_field_{field_id}")
    block = _select_rows(block, np.any(block.active, axis=(1, 2)))
    if not np.all(
        np.isin(
            np.floor(block.time_s / fold_bin_seconds).astype(np.int64),
            selected_time_bins,
        )
    ):
        raise RuntimeError("native extraction escaped the selected holdout time bins")
    return block


def _frequency_bin_count(block: VisibilityBlock, case: AveragingCase) -> int:
    if block.frequency_hz.size < 2:
        native_width_mhz = case.channel_width_mhz
    else:
        native_width_mhz = float(np.median(np.diff(block.frequency_hz)) / 1e6)
    channels_per_bin = case.channel_width_mhz / native_width_mhz
    rounded = int(round(channels_per_bin))
    if not np.isclose(channels_per_bin, rounded, rtol=0.0, atol=1e-8):
        raise ValueError(f"{case.label} is not an integer multiple of native channel width")
    if rounded < 1 or block.frequency_hz.size % rounded:
        raise ValueError(f"{case.label} does not divide {block.frequency_hz.size} channels")
    return block.frequency_hz.size // rounded


def _average_case(block: VisibilityBlock, case: AveragingCase) -> VisibilityBlock:
    """Average observations and attached native predictions identically."""

    frequency_averaged = average_frequency_bins(
        block,
        bin_count=_frequency_bin_count(block, case),
    )
    return average_time_bins(frequency_averaged, bin_seconds=case.time_seconds)


def _is_identity_averaging(
    native: VisibilityBlock,
    averaged: VisibilityBlock,
) -> bool:
    """Return whether averaging retained one native coordinate per output sample."""

    if averaged.shape != native.shape or not np.array_equal(
        averaged.frequency_hz, native.frequency_hz
    ):
        return False

    def order(block: VisibilityBlock) -> np.ndarray:
        return np.lexsort(
            (
                block.time_s,
                block.antenna2,
                block.antenna1,
                block.scan_id,
                block.field_id,
            )
        )

    native_order = order(native)
    averaged_order = order(averaged)
    for name in ("field_id", "scan_id", "antenna1", "antenna2"):
        if not np.array_equal(
            getattr(native, name)[native_order],
            getattr(averaged, name)[averaged_order],
        ):
            return False
    return np.allclose(
        native.time_s[native_order],
        averaged.time_s[averaged_order],
        rtol=0.0,
        atol=1e-6,
    ) and np.allclose(
        native.uvw_m[native_order],
        averaged.uvw_m[averaged_order],
        rtol=1e-12,
        atol=1e-9,
    )


def _metric_moments(
    block: VisibilityBlock,
    prediction: np.ndarray,
    mask: np.ndarray | None = None,
) -> MetricMoments:
    if prediction.shape != block.shape:
        raise ValueError("prediction must match its visibility block")
    selected = block.active if mask is None else block.active & mask
    if not np.any(selected):
        return MetricMoments()
    weight = block.weight[selected]
    observed = block.visibility[selected]
    residual = observed - prediction[selected]
    return MetricMoments(
        residual_power=float(np.sum(weight * np.abs(residual) ** 2)),
        signal_power=float(np.sum(weight * np.abs(observed) ** 2)),
        weight_sum=float(np.sum(weight)),
        sample_count=int(np.count_nonzero(selected)),
    )


def _metric_payload(moments: MetricMoments) -> dict[str, float | int | dict[str, Any]]:
    if moments.sample_count == 0 or moments.weight_sum <= 0 or moments.signal_power <= 0:
        return {
            "sample_count": 0,
            "weight_sum": 0.0,
            "weighted_complex_mse": float("nan"),
            "normalized_residual_power": float("nan"),
            "moments": asdict(moments),
        }
    return {
        "sample_count": moments.sample_count,
        "weight_sum": moments.weight_sum,
        "weighted_complex_mse": moments.residual_power / moments.weight_sum,
        "mean_weighted_squared_residual": (
            moments.residual_power / moments.sample_count
        ),
        "normalized_residual_power": moments.residual_power / moments.signal_power,
        "moments": asdict(moments),
    }


def _metric_groups(
    block: VisibilityBlock,
    prediction: np.ndarray,
    *,
    native_frequency_hz: np.ndarray,
) -> dict[str, dict[str, dict[str, float | int | dict[str, Any]]]]:
    """Resolve metrics by UV distance, broad frequency band, and time quarter."""

    uv_klambda = (
        np.linalg.norm(block.uvw_m[:, :2], axis=1)[:, None]
        * block.frequency_hz[None, :]
        / SPEED_OF_LIGHT_M_S
        / 1_000.0
    )
    baseline: dict[str, Any] = {}
    for lower, upper in zip(
        BASELINE_EDGES_KLAMBDA[:-1], BASELINE_EDGES_KLAMBDA[1:], strict=True
    ):
        label = f"{lower:g}-{upper:g}" if np.isfinite(upper) else f">={lower:g}"
        mask = (uv_klambda >= lower) & (uv_klambda < upper)
        baseline[label] = _metric_payload(
            _metric_moments(block, prediction, mask[:, :, None])
        )

    native_groups = np.array_split(native_frequency_hz, 4)
    frequency: dict[str, Any] = {}
    for index, group in enumerate(native_groups, start=1):
        lower = float(np.min(group) - 0.5 * np.median(np.diff(native_frequency_hz)))
        upper = float(np.max(group) + 0.5 * np.median(np.diff(native_frequency_hz)))
        mask = (block.frequency_hz >= lower) & (block.frequency_hz < upper)
        frequency[f"band_{index}"] = {
            **_metric_payload(
                _metric_moments(block, prediction, mask[None, :, None])
            ),
            "frequency_min_hz": lower,
            "frequency_max_hz": upper,
        }

    time_edges = np.linspace(float(np.min(block.time_s)), float(np.max(block.time_s)) + 1, 5)
    time: dict[str, Any] = {}
    for index, (lower, upper) in enumerate(
        zip(time_edges[:-1], time_edges[1:], strict=True), start=1
    ):
        mask = (block.time_s >= lower) & (block.time_s < upper)
        time[f"quarter_{index}"] = {
            **_metric_payload(
                _metric_moments(block, prediction, mask[:, None, None])
            ),
            "time_min_s": lower,
            "time_max_s": upper,
        }
    return {"baseline": baseline, "frequency": frequency, "time": time}


def _angular_separation_arcmin(
    first: tuple[float, float], second: tuple[float, float]
) -> float:
    delta_ra = first[0] - second[0]
    cosine = (
        np.sin(first[1]) * np.sin(second[1])
        + np.cos(first[1]) * np.cos(second[1]) * np.cos(delta_ra)
    )
    return float(np.rad2deg(np.arccos(np.clip(cosine, -1.0, 1.0))) * 60.0)


def _load_or_extract_native_field(
    arguments: argparse.Namespace,
    *,
    label: str,
    field_id: int,
    selected_time_bins: np.ndarray,
    solution: Any,
) -> VisibilityBlock:
    fixture = arguments.output / f"native_{label}.zarr"
    if fixture.exists() and not arguments.no_resume:
        stored = read_dataset(fixture).blocks
        if len(stored) != 1:
            raise ValueError(f"{fixture} must contain exactly one native block")
        state = "prediction" if stored[0].model_visibility is not None else "extraction"
        print(f"{label}: resumed native {state}", flush=True)
        return stored[0]
    print(f"{label}: extracting sealed native holdout", flush=True)
    block = _extract_native_holdout(
        arguments.measurement_set,
        solution,
        field_id=field_id,
        selected_time_bins=selected_time_bins,
        fold_bin_seconds=arguments.fold_bin_seconds,
        chunk_rows=arguments.chunk_rows,
    )
    write_dataset(
        VisibilityDataset(
            (block,),
            provenance={
                "experiment": "3c391_native_averaging_ablation",
                "pointing": label,
                "field_id": field_id,
                "state": "calibrated_native_holdout",
            },
        ),
        fixture,
    )
    return block


def _load_or_predict_native_field(
    arguments: argparse.Namespace,
    *,
    label: str,
    field_id: int,
    selected_time_bins: np.ndarray,
    solution: Any,
    components: tuple[MosaicSkyComponent, ...],
    mosaic_phase_centre_rad: tuple[float, float],
    beam: VLAPrimaryBeam,
    direct: DirectDFTConfig,
) -> VisibilityBlock:
    fixture = arguments.output / f"native_{label}.zarr"
    block = _load_or_extract_native_field(
        arguments,
        label=label,
        field_id=field_id,
        selected_time_bins=selected_time_bins,
        solution=solution,
    )
    if block.model_visibility is not None:
        return block
    print(
        f"{label}: predicting {np.count_nonzero(block.active):,} active native samples",
        flush=True,
    )
    prediction = predict_mosaic_composite(
        (block,),
        components,
        mosaic_phase_centre_rad,
        primary_beam=beam,
        config=direct,
    )[0]
    block = replace(block, model_visibility=prediction)
    write_dataset(
        VisibilityDataset(
            (block,),
            provenance={
                "experiment": "3c391_native_averaging_ablation",
                "pointing": label,
                "field_id": field_id,
                "state": "calibrated_native_holdout_with_frozen_prediction",
            },
        ),
        fixture,
    )
    return block


def _analyse_field(
    arguments: argparse.Namespace,
    *,
    label: str,
    field_id: int,
    selected_time_bins: np.ndarray,
    solution: Any,
    components: tuple[MosaicSkyComponent, ...],
    mosaic_phase_centre_rad: tuple[float, float],
    beam: VLAPrimaryBeam,
    direct: DirectDFTConfig,
) -> dict[str, Any]:
    path = arguments.output / f"{label}.json"
    if path.exists() and not arguments.no_resume:
        print(f"{label}: resumed completed averaging cases", flush=True)
        return json.loads(path.read_text(encoding="utf-8"))
    native = _load_or_predict_native_field(
        arguments,
        label=label,
        field_id=field_id,
        selected_time_bins=selected_time_bins,
        solution=solution,
        components=components,
        mosaic_phase_centre_rad=mosaic_phase_centre_rad,
        beam=beam,
        direct=direct,
    )
    assert native.model_visibility is not None
    cases: dict[str, Any] = {}
    for case in arguments.averaging_cases:
        print(f"{label}: evaluating {case.label}", flush=True)
        averaged = _average_case(native, case)
        if averaged.model_visibility is None:
            raise RuntimeError("averaging discarded the native model prediction")
        exact = np.asarray(averaged.model_visibility)
        centroid_block = replace(averaged, model_visibility=None)
        reused_exact = _is_identity_averaging(native, averaged)
        if reused_exact:
            centroid = exact.copy()
        else:
            centroid = predict_mosaic_composite(
                (centroid_block,),
                components,
                mosaic_phase_centre_rad,
                primary_beam=beam,
                config=direct,
            )[0]
        exact_metric = _metric_payload(_metric_moments(averaged, exact))
        centroid_metric = _metric_payload(_metric_moments(averaged, centroid))
        disagreement_block = replace(averaged, visibility=exact)
        disagreement = _metric_payload(_metric_moments(disagreement_block, centroid))
        exact_groups = _metric_groups(
            averaged,
            exact,
            native_frequency_hz=native.frequency_hz,
        )
        centroid_groups = (
            exact_groups
            if reused_exact
            else _metric_groups(
                averaged,
                centroid,
                native_frequency_hz=native.frequency_hz,
            )
        )
        cases[case.label] = {
            "averaging": asdict(case),
            "row_count": averaged.shape[0],
            "channel_count": averaged.shape[1],
            "exact": exact_metric,
            "centroid": centroid_metric,
            "centroid_to_exact_prediction": disagreement,
            "centroid_prediction_reused_exact": reused_exact,
            "centroid_mse_change_from_exact": (
                float(centroid_metric["weighted_complex_mse"])
                / float(exact_metric["weighted_complex_mse"])
                - 1.0
            ),
            "exact_groups": exact_groups,
            "centroid_groups": centroid_groups,
        }
    payload = {
        "label": label,
        "field_id": field_id,
        "pointing_offset_arcmin": _angular_separation_arcmin(
            native.phase_centre_rad, mosaic_phase_centre_rad
        ),
        "native_row_count": native.shape[0],
        "native_active_sample_count": int(np.count_nonzero(native.active)),
        "selected_fold_time_bin_count": int(selected_time_bins.size),
        "cases": cases,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _moments_from_payload(payload: dict[str, Any]) -> MetricMoments:
    return MetricMoments(**payload["moments"])


def _aggregate_metric(fields: list[dict[str, Any]], case: str, model: str) -> dict[str, Any]:
    total = MetricMoments()
    for field in fields:
        total.add(_moments_from_payload(field["cases"][case][model]))
    return _metric_payload(total)


def _aggregate_group_metrics(
    fields: list[dict[str, Any]],
    case: str,
    model: str,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Combine resolved metric moments across fields."""

    group_name = f"{model}_groups"
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for kind in ("baseline", "frequency", "time"):
        result[kind] = {}
        labels = fields[0]["cases"][case][group_name][kind]
        for label in labels:
            total = MetricMoments()
            for field in fields:
                total.add(
                    _moments_from_payload(
                        field["cases"][case][group_name][kind][label]
                    )
                )
            result[kind][label] = _metric_payload(total)
    return result


def _aggregate_fields(
    fields: list[dict[str, Any]], cases: tuple[AveragingCase, ...]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for case in cases:
        exact = _aggregate_metric(fields, case.label, "exact")
        centroid = _aggregate_metric(fields, case.label, "centroid")
        disagreement = _aggregate_metric(
            fields, case.label, "centroid_to_exact_prediction"
        )
        exact_groups = _aggregate_group_metrics(fields, case.label, "exact")
        centroid_groups = _aggregate_group_metrics(fields, case.label, "centroid")
        group_changes = {
            kind: {
                label: (
                    float(centroid_groups[kind][label]["weighted_complex_mse"])
                    / float(exact_groups[kind][label]["weighted_complex_mse"])
                    - 1.0
                )
                for label in exact_groups[kind]
                if int(exact_groups[kind][label]["sample_count"]) > 0
            }
            for kind in exact_groups
        }
        result[case.label] = {
            "averaging": asdict(case),
            "exact": exact,
            "centroid": centroid,
            "centroid_to_exact_prediction": disagreement,
            "exact_groups": exact_groups,
            "centroid_groups": centroid_groups,
            "centroid_group_mse_change_from_exact": group_changes,
            "centroid_mse_change_from_exact": (
                float(centroid["weighted_complex_mse"])
                / float(exact["weighted_complex_mse"])
                - 1.0
            ),
        }
    finest = result[cases[0].label]["exact"]
    finest_samples = int(finest["sample_count"])
    finest_weighted_residual = float(finest["mean_weighted_squared_residual"])
    for value in result.values():
        exact = value["exact"]
        value["retained_complex_sample_fraction"] = (
            int(exact["sample_count"]) / finest_samples
        )
        value["exact_weighted_residual_ratio_to_finest"] = (
            float(exact["mean_weighted_squared_residual"])
            / finest_weighted_residual
        )
    return result


def _plot_summary(summary: dict[str, Any], path: Path) -> None:
    labels = list(summary["cases"])
    display_labels = [
        (
            f"{summary['cases'][label]['averaging']['time_seconds']:g} s / "
            f"{summary['cases'][label]['averaging']['channel_width_mhz']:g} MHz"
        )
        for label in labels
    ]
    x = np.arange(len(labels))
    exact = np.asarray(
        [summary["cases"][label]["exact"]["normalized_residual_power"] for label in labels]
    )
    centroid = np.asarray(
        [
            summary["cases"][label]["centroid"]["normalized_residual_power"]
            for label in labels
        ]
    )
    penalty = np.asarray(
        [100.0 * summary["cases"][label]["centroid_mse_change_from_exact"] for label in labels]
    )
    exact_weighted = np.asarray(
        [summary["cases"][label]["exact"]["mean_weighted_squared_residual"] for label in labels]
    )
    centroid_weighted = np.asarray(
        [
            summary["cases"][label]["centroid"]["mean_weighted_squared_residual"]
            for label in labels
        ]
    )
    plt.style.use("default")
    figure, axes = plt.subplots(2, 2, figsize=(13.0, 9.0), constrained_layout=True)
    top_left, top_right, lower_left, lower_right = axes.ravel()
    top_left.plot(x, exact, marker="o", linewidth=2, label="Native predictions averaged")
    top_left.plot(x, centroid, marker="s", linewidth=2, label="One centroid prediction")
    top_left.set_xticks(x, display_labels, rotation=20, ha="right")
    top_left.set_ylabel("Normalized residual power")
    top_left.set_xlabel("Time and channel averaging")
    top_left.grid(alpha=0.25)
    top_left.margins(y=0.1)
    top_left.legend(frameon=False)
    for index, value in enumerate(penalty):
        top_left.annotate(
            f"{value:+.3f}%",
            (index, centroid[index]),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    top_right.plot(
        x, exact_weighted, marker="o", linewidth=2, label="Native predictions averaged"
    )
    top_right.plot(
        x, centroid_weighted, marker="s", linewidth=2, label="One centroid prediction"
    )
    top_right.set_xticks(x, display_labels, rotation=20, ha="right")
    top_right.set_yscale("log")
    top_right.set_ylabel("Mean weighted squared residual per complex sample")
    top_right.set_xlabel("Time and channel averaging")
    top_right.grid(alpha=0.25)
    top_right.legend(frameon=False)

    baseline_labels = list(
        summary["cases"][labels[0]]["centroid_group_mse_change_from_exact"]["baseline"]
    )
    baseline_x = np.arange(len(baseline_labels))
    colors = plt.get_cmap("viridis")(np.linspace(0.15, 0.85, len(labels)))
    for case_label, display_label, color in zip(
        labels, display_labels, colors, strict=True
    ):
        values = 100.0 * np.asarray(
            [
                summary["cases"][case_label]["centroid_group_mse_change_from_exact"][
                    "baseline"
                ][baseline]
                for baseline in baseline_labels
            ]
        )
        lower_left.plot(
            baseline_x,
            values,
            marker="o",
            linewidth=1.8,
            label=display_label,
            color=color,
        )
    lower_left.axhline(0.0, color="0.25", linewidth=1)
    lower_left.set_xticks(baseline_x, baseline_labels)
    lower_left.set_ylabel("Centroid residual-MSE change (%)")
    lower_left.set_xlabel("Projected baseline length (kλ)")
    lower_left.grid(alpha=0.25)
    lower_left.legend(frameon=False, ncols=2)

    pointing_labels = [
        f"{field['label']}\n{field['pointing_offset_arcmin']:.1f}′"
        for field in summary["fields"]
    ]
    pointing_x = np.arange(len(pointing_labels))
    first_case = labels[0]
    last_case = labels[-1]
    lower_right.plot(
        pointing_x,
        [field["cases"][first_case]["exact_weighted"] for field in summary["fields"]],
        marker="o",
        linewidth=1.8,
        label=f"{display_labels[0]}, exact",
    )
    lower_right.plot(
        pointing_x,
        [field["cases"][last_case]["exact_weighted"] for field in summary["fields"]],
        marker="s",
        linewidth=1.8,
        label=f"{display_labels[-1]}, exact",
    )
    lower_right.plot(
        pointing_x,
        [field["cases"][last_case]["centroid_weighted"] for field in summary["fields"]],
        marker="^",
        linewidth=1.8,
        label=f"{display_labels[-1]}, centroid",
    )
    lower_right.set_xticks(pointing_x, pointing_labels)
    lower_right.set_yscale("log")
    lower_right.set_ylabel("Mean weighted squared residual per complex sample")
    lower_right.set_xlabel("Pointing and offset from C1 phase centre")
    lower_right.grid(alpha=0.25)
    lower_right.legend(frameon=False)
    figure.savefig(path, dpi=180)
    plt.close(figure)


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
        "--sky-checkpoint",
        type=Path,
        default=Path("outputs/3c391_recovery_policy_fit_zero/sealed_active_only.npz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/3c391_native_averaging_ablation"),
    )
    parser.add_argument(
        "--averaging-cases",
        type=_parse_cases,
        default=_parse_cases("10:2,20:4,30:8,60:32"),
    )
    parser.add_argument("--fold-bin-seconds", type=float, default=60.0)
    parser.add_argument("--fold-count", type=int, default=5)
    parser.add_argument("--holdout-fold", type=int, default=4)
    parser.add_argument("--chunk-rows", type=int, default=2048)
    parser.add_argument("--visibility-tile-size", type=int, default=256)
    parser.add_argument("--pixel-tile-size", type=int, default=512)
    parser.add_argument("--precision", choices=("float32", "float64"), default="float32")
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="write calibrated native holdout fixtures without predicting the sky",
    )
    parser.add_argument("--no-resume", action="store_true")
    arguments = parser.parse_args()
    if arguments.chunk_rows <= 0:
        parser.error("--chunk-rows must be positive")
    if arguments.fold_bin_seconds <= 0:
        parser.error("--fold-bin-seconds must be positive")

    arguments.output.mkdir(parents=True, exist_ok=True)
    reference = read_dataset(arguments.reference_fixture).blocks
    selected_bins = _holdout_time_bins(
        reference,
        fold_bin_seconds=arguments.fold_bin_seconds,
        fold_count=arguments.fold_count,
        holdout_fold=arguments.holdout_fold,
    )
    protocol = json.loads(arguments.sky_protocol.read_text(encoding="utf-8"))
    mosaic_phase_centre_rad = reference[0].phase_centre_rad
    components = _components_from_checkpoint(
        arguments.sky_checkpoint,
        protocol,
        mosaic_phase_centre_rad,
    )
    solution = read_calibration(arguments.calibration)
    beam = VLAPrimaryBeam(
        kind="airy",
        catalog=replace(
            VLABeamCatalog(),
            airy_max_radius_rad_at_1ghz=np.deg2rad(
                float(protocol["airy_max_radius_deg_at_1ghz"])
            ),
        ),
    )
    direct = DirectDFTConfig(
        visibility_chunk_size=arguments.visibility_tile_size,
        pixel_chunk_size=arguments.pixel_tile_size,
        precision=arguments.precision,
    )

    if arguments.extract_only:
        for index, (reference_block, bins) in enumerate(
            zip(reference, selected_bins, strict=True), start=1
        ):
            field_ids = np.unique(reference_block.field_id)
            if field_ids.size != 1:
                raise ValueError(f"reference block C{index} mixes field IDs")
            block = _load_or_extract_native_field(
                arguments,
                label=f"C{index}",
                field_id=int(field_ids[0]),
                selected_time_bins=bins,
                solution=solution,
            )
            print(
                f"C{index}: stored {np.count_nonzero(block.active):,} active native samples",
                flush=True,
            )
        return 0

    fields = []
    for index, (reference_block, bins) in enumerate(
        zip(reference, selected_bins, strict=True), start=1
    ):
        field_ids = np.unique(reference_block.field_id)
        if field_ids.size != 1:
            raise ValueError(f"reference block C{index} mixes field IDs")
        fields.append(
            _analyse_field(
                arguments,
                label=f"C{index}",
                field_id=int(field_ids[0]),
                selected_time_bins=bins,
                solution=solution,
                components=components,
                mosaic_phase_centre_rad=mosaic_phase_centre_rad,
                beam=beam,
                direct=direct,
            )
        )

    summary = {
        "schema_version": 1,
        "measurement_set": str(arguments.measurement_set),
        "calibration": str(arguments.calibration),
        "reference_fixture": str(arguments.reference_fixture),
        "sky_protocol": str(arguments.sky_protocol),
        "sky_checkpoint": str(arguments.sky_checkpoint),
        "protocol": {
            "holdout_kind": "sealed interleaved complete 60-second time bins",
            "fold_bin_seconds": arguments.fold_bin_seconds,
            "fold_count": arguments.fold_count,
            "holdout_fold": arguments.holdout_fold,
            "raw_resolution": "10 seconds by 2 MHz",
            "flag_source": "post-application Measurement Set flags",
            "calibration_weights_propagated": False,
            "exact_prediction": "native model predictions averaged with observation weights",
            "centroid_prediction": "model evaluated at averaged UVW and frequency",
            "primary_beam": "extended Airy",
            "precision": arguments.precision,
        },
        "cases": _aggregate_fields(fields, arguments.averaging_cases),
        "fields": [
            {
                "label": field["label"],
                "field_id": field["field_id"],
                "pointing_offset_arcmin": field["pointing_offset_arcmin"],
                "native_active_sample_count": field["native_active_sample_count"],
                "case_file": f"{field['label']}.json",
                "cases": {
                    label: {
                        "exact": field["cases"][label]["exact"][
                            "normalized_residual_power"
                        ],
                        "centroid": field["cases"][label]["centroid"][
                            "normalized_residual_power"
                        ],
                        "centroid_mse_change_from_exact": field["cases"][label][
                            "centroid_mse_change_from_exact"
                        ],
                        "exact_weighted": (
                            field["cases"][label]["exact"]["moments"][
                                "residual_power"
                            ]
                            / field["cases"][label]["exact"]["sample_count"]
                        ),
                        "centroid_weighted": (
                            field["cases"][label]["centroid"]["moments"][
                                "residual_power"
                            ]
                            / field["cases"][label]["centroid"]["sample_count"]
                        ),
                    }
                    for label in field["cases"]
                },
            }
            for field in fields
        ],
    }
    (arguments.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _plot_summary(summary, arguments.output / "native_averaging_ablation.jpg")
    print(json.dumps(summary["cases"], indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
