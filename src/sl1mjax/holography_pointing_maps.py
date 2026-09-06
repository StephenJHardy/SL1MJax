"""Per-pass occupancy, INTERVAL-normalized residuals, and transition plots."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from sl1mjax.holography import (
    AntennaPointingRole,
    HolographyObservation,
    HolographyRowReason,
    Memo195LowerCRaster,
    ResolvedAntennaPointing,
    cluster_antenna_dwells,
)
from sl1mjax.holography_pointing_audit import (
    PointingReconstructionAudit,
    RasterCellOccupancy,
    TimeAlignmentResiduals,
    _merge_raster_cells,
    _relabel_cells_by_nearest_spacing,
    pointing_reconstruction_audit_as_dict,
)

HOLOGRAPHY_THOL0001_LOWER_C_ROOT = (
    Path(__file__).resolve().parent / "data" / "holography_thol0001_lower_c"
)
POINTING_AUDIT_SCHEMA_VERSION = 2
GRID_MATCH_TOLERANCE_ARCMIN = 0.35
PASS2_MEASURED_RASTER_NOTE = (
    "pass-2 is a complete measured raster, not a Memo-lattice raster; "
    "do not snap recovered cells onto the nominal 23×23 lattice"
)


@dataclass(frozen=True)
class OverlapProof:
    """POINTING intervals may overlap; visibility times must not hit two."""

    n_visibility_times_in_multiple_intervals: int
    n_pointing_interval_overlap_pairs: int
    max_pointing_interval_s: float
    passed: bool


@dataclass(frozen=True)
class RasterPassOccupancy:
    """Recovered cells for one HOLORASTER scan block."""

    name: str
    scan_numbers: tuple[int, ...]
    n_unique_times: int
    n_settled_moving_samples: int
    n_clustered_cells: int
    n_dense_family: int
    n_sparse_family: int
    square_17_matched: int
    square_23_matched: int
    circular_17_matched: int
    circular_23_matched: int
    occupancy_min: int
    occupancy_max: int


@dataclass(frozen=True)
class RasterOccupancyReport:
    """Why clustered counts are not exactly 17² and 23²."""

    passes: tuple[RasterPassOccupancy, ...]
    combined: RasterPassOccupancy
    square_17_expected: int
    square_23_expected: int
    circular_17_expected: int
    circular_23_expected: int
    explanation: str


def memo195_expected_offsets(
    n: int,
    spacing_arcmin: float,
    *,
    circular: bool,
) -> NDArray[np.float64]:
    half = (int(n) - 1) // 2
    cells: list[tuple[float, float]] = []
    for iy in range(-half, half + 1):
        for ix in range(-half, half + 1):
            if circular and float(np.hypot(ix, iy)) > half + 1e-12:
                continue
            cells.append((ix * spacing_arcmin, iy * spacing_arcmin))
    return np.asarray(cells, dtype=np.float64)


def match_expected_grid(
    cells: tuple[RasterCellOccupancy, ...],
    expected_lm_arcmin: NDArray[np.float64],
    *,
    tolerance_arcmin: float = GRID_MATCH_TOLERANCE_ARCMIN,
) -> int:
    if expected_lm_arcmin.size == 0 or not cells:
        return 0
    recovered = np.asarray([[cell.l_arcmin, cell.m_arcmin] for cell in cells], dtype=np.float64)
    matched = 0
    for target in expected_lm_arcmin:
        if float(np.min(np.hypot(recovered[:, 0] - target[0], recovered[:, 1] - target[1]))) <= (
            tolerance_arcmin
        ):
            matched += 1
    return matched


def scan_passes(scan_per_time: NDArray[np.int32], *, gap: int = 2) -> tuple[tuple[int, ...], ...]:
    """Split HOLORASTER scans on a gap. Input MAIN order is not required."""

    scans = np.unique(scan_per_time[scan_per_time >= 0])
    if scans.size == 0:
        return ()
    groups: list[list[int]] = [[int(scans[0])]]
    for scan in scans[1:]:
        if int(scan) - groups[-1][-1] > int(gap):
            groups.append([int(scan)])
        else:
            groups[-1].append(int(scan))
    return tuple(tuple(group) for group in groups)


def visit_id_per_time(
    observation: HolographyObservation,
    *,
    mode: str = "raster_pass",
) -> NDArray[np.int32]:
    """One visit label per unique pointing time.

    Raster-pass visits follow HOLORASTER scan groups, not MAIN row order
    and not beam radius. Origin dwells inherit the pass of their scan.
    """

    offsets, inverse, _valid, _settled, _moving = observation.pointing_state()
    n_time = int(offsets.shape[0])
    if mode == "per_time":
        return np.arange(n_time, dtype=np.int32)
    visit = np.zeros(n_time, dtype=np.int32)
    scan = observation.block.scan_id
    if mode == "raster_pass" and scan is not None:
        scan = np.asarray(scan, dtype=np.int32).reshape(-1)
        time_index = np.asarray(inverse, dtype=np.int32)
        if scan.size == time_index.size and time_index.size:
            first = np.unique(time_index, return_index=True)[1]
            scan_at_time = np.full(n_time, -1, dtype=np.int32)
            scan_at_time[time_index[first]] = scan[first]
            groups = scan_passes(scan_at_time)
            for index, group in enumerate(groups):
                visit[np.isin(scan_at_time, np.asarray(group, dtype=np.int32))] = int(index)
            if int(np.unique(visit[scan_at_time >= 0]).size) >= 2:
                return visit
    if n_time >= 2:
        mid = n_time // 2
        return np.where(np.arange(n_time) >= mid, 1, 0).astype(np.int32)
    return visit


def occupancy_for_times(
    resolved: ResolvedAntennaPointing,
    time_mask: NDArray[np.bool_],
    *,
    name: str,
    scan_numbers: tuple[int, ...],
    memo195: Memo195LowerCRaster,
    cluster_radius_arcmin: float = 0.15,
) -> RasterPassOccupancy:
    if not np.any(time_mask):
        return RasterPassOccupancy(name, scan_numbers, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    subset = ResolvedAntennaPointing(
        unique_time_s=resolved.unique_time_s[time_mask],
        antenna_id=resolved.antenna_id,
        offset_lm_rad=resolved.offset_lm_rad[time_mask],
        valid=resolved.valid[time_mask],
        settled=resolved.settled[time_mask],
        role=resolved.role[time_mask],
        selected_column=resolved.selected_column,
        offset_sign=resolved.offset_sign,
        join_rule=resolved.join_rule,
        join_tolerance_s=resolved.join_tolerance_s,
        measure_ref=resolved.measure_ref,
        units=resolved.units,
        notes=resolved.notes,
    )
    moving = subset.valid & subset.settled & (subset.role == AntennaPointingRole.MOVING.value)
    cells = _relabel_cells_by_nearest_spacing(
        _merge_raster_cells(
            cluster_antenna_dwells(subset, cluster_radius_arcmin=cluster_radius_arcmin),
            memo195,
            cluster_radius_arcmin=cluster_radius_arcmin,
        ),
        memo195,
    )
    counts = [int(cell.sample_count) for cell in cells]
    return RasterPassOccupancy(
        name=name,
        scan_numbers=scan_numbers,
        n_unique_times=int(np.sum(time_mask)),
        n_settled_moving_samples=int(np.sum(moving)),
        n_clustered_cells=len(cells),
        n_dense_family=sum(1 for cell in cells if cell.family == "dense"),
        n_sparse_family=sum(1 for cell in cells if cell.family == "sparse"),
        square_17_matched=match_expected_grid(
            cells,
            memo195_expected_offsets(memo195.dense_n, memo195.dense_spacing_arcmin, circular=False),
        ),
        square_23_matched=match_expected_grid(
            cells,
            memo195_expected_offsets(
                memo195.sparse_n, memo195.sparse_spacing_arcmin, circular=False
            ),
        ),
        circular_17_matched=match_expected_grid(
            cells,
            memo195_expected_offsets(memo195.dense_n, memo195.dense_spacing_arcmin, circular=True),
        ),
        circular_23_matched=match_expected_grid(
            cells,
            memo195_expected_offsets(
                memo195.sparse_n, memo195.sparse_spacing_arcmin, circular=True
            ),
        ),
        occupancy_min=min(counts) if counts else 0,
        occupancy_max=max(counts) if counts else 0,
    )


def explain_occupancy(report: RasterOccupancyReport) -> str:
    combined = report.combined
    reasons: list[str] = []
    for item in report.passes:
        if item.n_dense_family and item.n_sparse_family == 0:
            reasons.append(
                f"{item.name} is dense-only: {item.n_clustered_cells} cells, "
                f"{item.square_17_matched}/{report.square_17_expected} on the filled 17×17, "
                f"{item.circular_17_matched}/{report.circular_17_expected} on the circular 17×17"
            )
        elif item.n_sparse_family and item.n_dense_family == 0:
            exact = (
                f" (exactly 23²={report.square_23_expected})"
                if item.n_clustered_cells == report.square_23_expected
                else ""
            )
            reasons.append(
                f"{item.name} is sparse-only: {item.n_clustered_cells} cells{exact}, "
                f"{item.square_23_matched}/{report.square_23_expected} "
                "on the filled 23×23 lattice, "
                f"{item.circular_23_matched}/{report.circular_23_expected} "
                "on the circular 23×23"
            )
        else:
            reasons.append(
                f"{item.name} has {item.n_dense_family} dense and "
                f"{item.n_sparse_family} sparse cells"
            )
    pass_dense = sum(item.n_dense_family for item in report.passes)
    pass_sparse = sum(item.n_sparse_family for item in report.passes)
    if report.passes and (
        combined.n_dense_family != pass_dense or combined.n_sparse_family != pass_sparse
    ):
        reasons.append(
            "combined nearest-spacing relabel changes the dense/sparse split from per-pass "
            f"{pass_dense}/{pass_sparse} to {combined.n_dense_family}/{combined.n_sparse_family}"
        )
    if combined.n_dense_family < report.square_17_expected:
        reasons.append(
            "combined dense count is below 17²=289; it is closer to the circular 17×17 "
            f"({report.circular_17_expected} cells) than to the filled square"
        )
    if combined.n_sparse_family != report.square_23_expected:
        reasons.append(
            "combined sparse count is not 23²=529; extra cells are off-lattice fragments "
            "or dense-pass points reassigned to the sparse family"
        )
    if len(report.passes) > 1 and combined.occupancy_max > max(
        item.occupancy_max for item in report.passes
    ):
        reasons.append("combined occupancy exceeds one pass: repeated visits to the same cells")
    if combined.n_clustered_cells > combined.square_17_matched + combined.square_23_matched:
        reasons.append(
            "clustering keeps extra cells off the exact Memo 195 lattice "
            f"(tolerance {GRID_MATCH_TOLERANCE_ARCMIN} arcmin)"
        )
    reasons.append("transition guards remove slew samples and do not create new lattice points")
    if any(
        item.n_sparse_family and item.n_clustered_cells == report.square_23_expected
        for item in report.passes
    ):
        reasons.append(PASS2_MEASURED_RASTER_NOTE)
    return "; ".join(reasons) + "."


def build_occupancy_report(
    resolved: ResolvedAntennaPointing,
    scan_per_time: NDArray[np.int32],
    *,
    memo195: Memo195LowerCRaster,
    cluster_radius_arcmin: float = 0.15,
) -> RasterOccupancyReport:
    passes = []
    for index, scans in enumerate(scan_passes(scan_per_time), start=1):
        mask = np.isin(scan_per_time, np.asarray(scans, dtype=np.int32))
        passes.append(
            occupancy_for_times(
                resolved,
                mask,
                name=f"pass_{index}",
                scan_numbers=scans,
                memo195=memo195,
                cluster_radius_arcmin=cluster_radius_arcmin,
            )
        )
    combined = occupancy_for_times(
        resolved,
        np.ones(resolved.unique_time_s.size, dtype=bool),
        name="combined",
        scan_numbers=tuple(int(scan) for scan in np.unique(scan_per_time[scan_per_time >= 0])),
        memo195=memo195,
        cluster_radius_arcmin=cluster_radius_arcmin,
    )
    report = RasterOccupancyReport(
        passes=tuple(passes),
        combined=combined,
        square_17_expected=int(memo195.dense_n * memo195.dense_n),
        square_23_expected=int(memo195.sparse_n * memo195.sparse_n),
        circular_17_expected=int(
            memo195_expected_offsets(
                memo195.dense_n, memo195.dense_spacing_arcmin, circular=True
            ).shape[0]
        ),
        circular_23_expected=int(
            memo195_expected_offsets(
                memo195.sparse_n, memo195.sparse_spacing_arcmin, circular=True
            ).shape[0]
        ),
        explanation="",
    )
    return RasterOccupancyReport(
        passes=report.passes,
        combined=report.combined,
        square_17_expected=report.square_17_expected,
        square_23_expected=report.square_23_expected,
        circular_17_expected=report.circular_17_expected,
        circular_23_expected=report.circular_23_expected,
        explanation=explain_occupancy(report),
    )


def summarize_alignment(
    residual_s: NDArray[np.float64],
    interval_s: NDArray[np.float64],
    valid: NDArray[np.bool_],
    *,
    overlap_hits: int,
    pointing_overlap_pairs: int,
    max_pointing_interval_s: float,
) -> tuple[TimeAlignmentResiduals, OverlapProof]:
    matched = valid & np.isfinite(residual_s) & np.isfinite(interval_s) & (interval_s > 0.0)
    residual = residual_s[matched]
    half = 0.5 * interval_s[matched]
    n_unmatched = int(np.sum(~valid))
    if residual.size == 0:
        alignment = TimeAlignmentResiduals(
            0, n_unmatched, float("nan"), float("nan"), float("nan"), float("nan")
        )
        fraction = np.array([], dtype=np.float64)
        n_outside = 0
        max_fraction = float("nan")
    else:
        fraction = residual / half
        n_outside = int(np.sum((fraction < -1.0) | (fraction >= 1.0)))
        abs_residual = np.abs(residual)
        alignment = TimeAlignmentResiduals(
            n_matched=int(residual.size),
            n_unmatched=n_unmatched,
            median_s=float(np.median(residual)),
            p95_abs_s=float(np.percentile(abs_residual, 95)),
            max_abs_s=float(np.max(abs_residual)),
            rms_s=float(np.sqrt(np.mean(residual * residual))),
            max_abs_fraction_of_half_interval=float(np.max(np.abs(fraction))),
            n_outside_half_interval=n_outside,
            max_pointing_interval_s=float(max_pointing_interval_s),
        )
        max_fraction = float(np.max(np.abs(fraction)))
    proof = OverlapProof(
        n_visibility_times_in_multiple_intervals=int(overlap_hits),
        n_pointing_interval_overlap_pairs=int(pointing_overlap_pairs),
        max_pointing_interval_s=float(max_pointing_interval_s),
        passed=int(overlap_hits) == 0
        and n_outside == 0
        and (residual.size == 0 or max_fraction < 1.0),
    )
    return alignment, proof


def scan_per_unique_time(
    unique_time_s: NDArray[np.float64],
    row_time_s: NDArray[np.float64],
    scan_number: NDArray[np.int32],
) -> NDArray[np.int32]:
    scans = np.full(unique_time_s.size, -1, dtype=np.int32)
    time_index = np.searchsorted(unique_time_s, row_time_s)
    in_range = time_index < unique_time_s.size
    matched = in_range & (
        unique_time_s[np.clip(time_index, 0, max(unique_time_s.size - 1, 0))] == row_time_s
    )
    if not np.any(matched):
        return scans
    pairs = np.unique(
        np.stack((time_index[matched], scan_number[matched]), axis=1),
        axis=0,
    )
    for index, scan in pairs:
        if scans[int(index)] < 0:
            scans[int(index)] = int(scan)
    return scans


def write_occupancy_map(
    cells: tuple[RasterCellOccupancy, ...],
    path: Path,
    *,
    title: str,
    memo195: Memo195LowerCRaster,
) -> Path:
    import matplotlib.pyplot as plt

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(figsize=(6.2, 6.2))
    square_17 = memo195_expected_offsets(
        memo195.dense_n, memo195.dense_spacing_arcmin, circular=False
    )
    circular_17 = memo195_expected_offsets(
        memo195.dense_n, memo195.dense_spacing_arcmin, circular=True
    )
    square_23 = memo195_expected_offsets(
        memo195.sparse_n, memo195.sparse_spacing_arcmin, circular=False
    )
    axes.scatter(
        square_23[:, 0],
        square_23[:, 1],
        marker="+",
        c="0.88",
        s=16,
        label=f"23×23 square ({square_23.shape[0]})",
        zorder=1,
    )
    axes.scatter(
        square_17[:, 0],
        square_17[:, 1],
        marker="+",
        c="0.70",
        s=22,
        label=f"17×17 square ({square_17.shape[0]})",
        zorder=1,
    )
    axes.scatter(
        circular_17[:, 0],
        circular_17[:, 1],
        facecolors="none",
        edgecolors="#9ecae1",
        s=42,
        linewidths=0.8,
        label=f"17×17 circular ({circular_17.shape[0]})",
        zorder=1,
    )
    for family, color in (("dense", "#1f77b4"), ("sparse", "#ff7f0e"), ("other", "#7f7f7f")):
        selected = [cell for cell in cells if cell.family == family]
        if not selected:
            continue
        axes.scatter(
            [cell.l_arcmin for cell in selected],
            [cell.m_arcmin for cell in selected],
            s=[max(8.0, 0.04 * cell.sample_count) for cell in selected],
            c=color,
            label=f"{family} ({len(selected)})",
            alpha=0.85,
            linewidths=0.0,
        )
    axes.set_xlabel("commanded l / arcmin")
    axes.set_ylabel("commanded m / arcmin")
    axes.set_aspect("equal", adjustable="box")
    axes.axhline(0.0, color="0.7", linewidth=0.6)
    axes.axvline(0.0, color="0.7", linewidth=0.6)
    axes.set_title(title)
    axes.legend(loc="best", fontsize=8)
    figure.tight_layout()
    figure.savefig(destination, dpi=120)
    plt.close(figure)
    return destination


def write_transition_track_plot(
    resolved: ResolvedAntennaPointing,
    sample_reason: NDArray[np.str_],
    antenna_id: int,
    path: Path,
    *,
    antenna_name: str | None = None,
) -> Path:
    import matplotlib.pyplot as plt

    index = int(np.flatnonzero(resolved.antenna_id == antenna_id)[0])
    time_s = resolved.unique_time_s
    offset = resolved.offset_lm_rad[:, index]
    radius = np.rad2deg(np.hypot(offset[:, 0], offset[:, 1])) * 60.0
    reason = sample_reason[:, index]
    first_retained = _first_retained_mask(reason)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(figsize=(8.5, 3.6))
    valid = resolved.valid[:, index]
    axes.plot(time_s[valid], radius[valid], color="0.75", linewidth=0.8, label="joined offset")
    _scatter(
        axes,
        time_s,
        radius,
        reason == HolographyRowReason.DWELL_JUMP.value,
        "C3",
        "x",
        "jump",
    )
    _scatter(
        axes,
        time_s,
        radius,
        reason == HolographyRowReason.TRANSITION_GUARD.value,
        "C1",
        "s",
        "guard",
    )
    _scatter(axes, time_s, radius, first_retained, "C2", "o", "first retained")
    other_ok = (reason == HolographyRowReason.OK.value) & ~first_retained
    _scatter(axes, time_s, radius, other_ok, "C0", ".", "other retained")
    axes.set_xlabel("MAIN TIME / s")
    axes.set_ylabel("offset radius / arcmin")
    axes.set_title(f"Transition guard on {antenna_name or f'antenna {antenna_id}'}")
    axes.legend(loc="best", fontsize=8, ncol=2)
    figure.tight_layout()
    figure.savefig(destination, dpi=120)
    plt.close(figure)
    return destination


def write_pointing_audit_bundle(
    audit: PointingReconstructionAudit,
    destination: Path,
    *,
    moving_plot_antennas: int = 3,
) -> dict[str, Path]:
    """Write the versioned pointing audit, fixture, occupancy maps, and plots."""

    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    payload = pointing_reconstruction_audit_as_dict(audit)
    payload["schema_version"] = POINTING_AUDIT_SCHEMA_VERSION
    if audit.occupancy is not None:
        payload["occupancy"] = occupancy_as_dict(audit.occupancy)
    if audit.overlap_proof is not None:
        payload["overlap_proof"] = overlap_as_dict(audit.overlap_proof)
    audit_path = root / "pointing_audit.json"
    audit_path.write_text(json.dumps(payload, indent=2) + "\n")
    written["audit"] = audit_path
    fixture_path = root / "metadata_fixture.json"
    fixture_path.write_text(json.dumps(real_ms_metadata_fixture(audit), indent=2) + "\n")
    written["fixture"] = fixture_path
    if audit.occupancy is not None:
        summary_path = root / "occupancy_summary.json"
        summary_path.write_text(json.dumps(occupancy_as_dict(audit.occupancy), indent=2) + "\n")
        written["occupancy_summary"] = summary_path
    memo = audit.memo195
    written["occupancy_combined"] = write_occupancy_map(
        audit.raster_cells,
        root / "occupancy_combined.png",
        title="Combined HOLORASTER occupancy",
        memo195=memo,
    )
    if audit.occupancy is not None:
        for item in audit.occupancy.passes:
            mask = np.isin(audit.scan_per_time, np.asarray(item.scan_numbers, dtype=np.int32))
            pass_cells = _cells_for_times(audit.resolved, mask, memo)
            written[item.name] = write_occupancy_map(
                pass_cells,
                root / f"occupancy_{item.name}.png",
                title=f"{item.name} scans {item.scan_numbers[0]}–{item.scan_numbers[-1]}",
                memo195=memo,
            )
    for antenna in audit.moving_antenna_ids[: int(moving_plot_antennas)]:
        name = _antenna_name(audit, antenna)
        written[f"transition_{antenna}"] = write_transition_track_plot(
            audit.resolved,
            audit.sample_reason_grid,
            int(antenna),
            root / f"transition_{name}.png",
            antenna_name=name,
        )
    manifest = {
        "schema_version": POINTING_AUDIT_SCHEMA_VERSION,
        "files": {key: path.name for key, path in written.items()},
        "overlap_proof_passed": bool(audit.overlap_proof.passed) if audit.overlap_proof else False,
        "occupancy_explanation": audit.occupancy.explanation if audit.occupancy else "",
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    written["manifest"] = manifest_path
    return written


def real_ms_metadata_fixture(audit: PointingReconstructionAudit) -> dict[str, object]:
    """Compact real-MS metadata. No visibilities."""

    return {
        "schema_version": POINTING_AUDIT_SCHEMA_VERSION,
        "kind": "thol0001_lower_c_metadata",
        "provenance": dict(audit.provenance),
        "field_name": audit.field_name,
        "selected_column": audit.selected_column,
        "native_channels": pointing_reconstruction_audit_as_dict(audit)["native_channels"],
        "moving_antenna_names": list(audit.moving_antenna_names),
        "reference_antenna_names": list(audit.reference_antenna_names),
        "memo195": pointing_reconstruction_audit_as_dict(audit)["memo195"],
        "cell_spacing_arcmin": dict(audit.cell_spacing_arcmin),
        "offset_range_arcmin": dict(audit.offset_range_arcmin),
        "time_alignment": pointing_reconstruction_audit_as_dict(audit)["time_alignment"],
        "overlap_proof": overlap_as_dict(audit.overlap_proof) if audit.overlap_proof else {},
        "occupancy": occupancy_as_dict(audit.occupancy) if audit.occupancy else {},
        "composition": {key: dict(value) for key, value in audit.composition.items()},
        "unique_time_count": audit.unique_time_count,
        "selected_main_rows": audit.selected_main_rows,
        "notes": list(audit.notes),
        "visibilities": None,
    }


def occupancy_as_dict(report: RasterOccupancyReport) -> dict[str, object]:
    return {
        "square_17_expected": report.square_17_expected,
        "square_23_expected": report.square_23_expected,
        "circular_17_expected": report.circular_17_expected,
        "circular_23_expected": report.circular_23_expected,
        "explanation": report.explanation,
        "combined": _pass_as_dict(report.combined),
        "passes": [_pass_as_dict(item) for item in report.passes],
    }


def overlap_as_dict(proof: OverlapProof) -> dict[str, object]:
    return {
        "n_visibility_times_in_multiple_intervals": proof.n_visibility_times_in_multiple_intervals,
        "n_pointing_interval_overlap_pairs": proof.n_pointing_interval_overlap_pairs,
        "max_pointing_interval_s": proof.max_pointing_interval_s,
        "passed": proof.passed,
    }


def _pass_as_dict(item: RasterPassOccupancy) -> dict[str, object]:
    return {
        "name": item.name,
        "scan_numbers": list(item.scan_numbers),
        "n_unique_times": item.n_unique_times,
        "n_settled_moving_samples": item.n_settled_moving_samples,
        "n_clustered_cells": item.n_clustered_cells,
        "n_dense_family": item.n_dense_family,
        "n_sparse_family": item.n_sparse_family,
        "square_17_matched": item.square_17_matched,
        "square_23_matched": item.square_23_matched,
        "circular_17_matched": item.circular_17_matched,
        "circular_23_matched": item.circular_23_matched,
        "occupancy_min": item.occupancy_min,
        "occupancy_max": item.occupancy_max,
    }


def _cells_for_times(
    resolved: ResolvedAntennaPointing,
    time_mask: NDArray[np.bool_],
    memo195: Memo195LowerCRaster,
) -> tuple[RasterCellOccupancy, ...]:
    report = occupancy_for_times(
        resolved,
        time_mask,
        name="tmp",
        scan_numbers=(),
        memo195=memo195,
    )
    del report
    subset = ResolvedAntennaPointing(
        unique_time_s=resolved.unique_time_s[time_mask],
        antenna_id=resolved.antenna_id,
        offset_lm_rad=resolved.offset_lm_rad[time_mask],
        valid=resolved.valid[time_mask],
        settled=resolved.settled[time_mask],
        role=resolved.role[time_mask],
        selected_column=resolved.selected_column,
        offset_sign=resolved.offset_sign,
        join_rule=resolved.join_rule,
        join_tolerance_s=resolved.join_tolerance_s,
        measure_ref=resolved.measure_ref,
        units=resolved.units,
        notes=resolved.notes,
    )
    return _relabel_cells_by_nearest_spacing(
        _merge_raster_cells(
            cluster_antenna_dwells(subset),
            memo195,
            cluster_radius_arcmin=0.15,
        ),
        memo195,
    )


def _first_retained_mask(reason: NDArray[np.str_]) -> NDArray[np.bool_]:
    first = np.zeros(reason.size, dtype=bool)
    previous_ok = False
    for index, label in enumerate(reason):
        ok = label == HolographyRowReason.OK.value
        if ok and not previous_ok:
            first[index] = True
        previous_ok = ok
    return first


def _scatter(axes, time_s, radius, mask, color, marker, label) -> None:
    if not np.any(mask):
        return
    axes.scatter(time_s[mask], radius[mask], c=color, marker=marker, s=18, label=label, zorder=3)


def _antenna_name(audit: PointingReconstructionAudit, antenna_id: int) -> str:
    if antenna_id in audit.moving_antenna_ids:
        return audit.moving_antenna_names[audit.moving_antenna_ids.index(antenna_id)]
    return f"ant{antenna_id}"
