"""Staged forward-closure ladder for the empirical holography beam.

The one-axis full-Jones comparison is not interpretable until exact-cell
training rows close in copolar. These stages isolate the first operation
that breaks :math:`V \\simeq R_m E_m S R_r^H`.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sl1mjax.holography import HolographyObservation, Memo195LowerCRaster
from sl1mjax.holography_diagonal import _raster_label
from sl1mjax.holography_full_jones import (
    HOLOGRAPHY_FULL_JONES_SCHEMA_VERSION,
    FrozenInterpolationSupport,
    HolographyFullJonesArtifact,
    HolographyFullJonesSample,
    _antenna_jones_planes,
    _predict_vis_jax,
    _recover_row_jones_jax,
    interpolate_holography_full_jones,
    moving_reference_row_geometry,
    predict_moving_reference_from_beam,
    recover_holography_full_jones,
    refactor_on_axis_gauge,
)
from sl1mjax.holography_reference_jones import predict_baseline_coherency
from sl1mjax.polarization import invert_jones

ALGEBRAIC_TOLERANCE = 1.0e-8
EMPIRICAL_COPOLAR_VISIBILITY_CLOSURE = "empirical_copolar_visibility_closure"
FORWARD_CLOSURE_STAGES = (
    "single_sample_algebraic_closure",
    "gauge_refactorization_closure",
    "per_sample_map_closure",
    "cell_aggregation_closure",
    "spatial_support_contract",
    "exact_cell_holdouts",
    "remaining_one_axis_diagnostics",
)
FORWARD_CLOSURE_NOTE = (
    "Exact-cell holdouts are not a full-Jones test until training rows "
    "close in RR/LL. Diagnose through the staged forward-closure ladder."
)


def recover_row_jones(
    visibility: ArrayLike,
    source: ArrayLike,
    r_moving: ArrayLike,
    r_ref: ArrayLike,
    *,
    moving_is_p: bool,
) -> NDArray[np.complex128]:
    """Invert one visibility: :math:`V=R_m E S R_r^H` or the antenna-2 swap."""

    vis = np.asarray(visibility, dtype=np.complex128)
    source_plane = np.asarray(source, dtype=np.complex128)
    r_m = np.asarray(r_moving, dtype=np.complex128)
    r_r = np.asarray(r_ref, dtype=np.complex128)
    if bool(moving_is_p):
        return (
            invert_jones(r_m)
            @ vis
            @ invert_jones(source_plane @ np.conjugate(np.swapaxes(r_r, -1, -2)))
        )
    return invert_jones(r_m) @ np.conjugate(
        np.swapaxes(invert_jones(r_r @ source_plane) @ vis, -1, -2)
    )


def predict_row_visibility(
    beam: ArrayLike,
    source: ArrayLike,
    r_moving: ArrayLike,
    r_ref: ArrayLike,
    *,
    moving_is_p: bool,
) -> NDArray[np.complex128]:
    """Reconstruct one visibility from a recovered beam sample."""

    e_m = np.asarray(beam, dtype=np.complex128)
    source_plane = np.asarray(source, dtype=np.complex128)
    r_m = np.asarray(r_moving, dtype=np.complex128)
    r_r = np.asarray(r_ref, dtype=np.complex128)
    if bool(moving_is_p):
        return predict_baseline_coherency(r_m @ e_m, source_plane, r_r)
    return predict_baseline_coherency(r_r, source_plane, r_m @ e_m)


def complex_relative_error(measured: ArrayLike, predicted: ArrayLike) -> float:
    """Max elementwise :math:`|\\hat V-V|/\\max(|V|,\\varepsilon)`."""

    vis = np.asarray(measured, dtype=np.complex128)
    pred = np.asarray(predicted, dtype=np.complex128)
    scale = np.maximum(np.abs(vis), 1.0e-12)
    delta = np.abs(pred - vis) / scale
    finite = delta[np.isfinite(delta)]
    return float(np.max(finite)) if finite.size else float("nan")


def residual_hands(
    measured: ArrayLike,
    predicted: ArrayLike,
    *,
    stokes_i: ArrayLike,
) -> dict[str, float]:
    vis = np.asarray(measured, dtype=np.complex128)
    pred = np.asarray(predicted, dtype=np.complex128)
    if vis.ndim == 4:
        vis = vis[:, 0]
        pred = pred[:, 0]
    intensity = np.maximum(np.abs(np.asarray(stokes_i, dtype=np.float64).reshape(-1)), 1.0e-3)
    residual = (vis - pred) / intensity[:, None, None]
    finite = np.isfinite(pred[..., 0, 0])
    report = {"n": int(vis.shape[0]), "n_finite": int(np.sum(finite))}
    for name, (row, col) in (("rr", (0, 0)), ("ll", (1, 1)), ("rl", (0, 1)), ("lr", (1, 0))):
        values = residual[finite, row, col]
        values = values[np.isfinite(values)]
        report[f"median_abs_{name}_over_i"] = (
            float(np.median(np.abs(values))) if values.size else float("nan")
        )
        report[f"median_arg_{name}_rad"] = (
            float(np.median(np.angle(values))) if values.size else float("nan")
        )
    return report


def algebraic_closure_cases() -> list[dict[str, object]]:
    """Compact well-conditioned Jones/source cases, including both antenna orders."""

    rng = np.random.default_rng(32)
    cases = []
    source = np.array([[1.3, 0.02 - 0.01j], [0.02 + 0.01j, 1.3]], dtype=np.complex128)
    for moving_is_p in (True, False):
        for name, extra in (
            ("identity_residuals", {}),
            ("nondiagonal_residual", {}),
            ("polarized_source", {"source": source}),
        ):
            r_m = np.eye(2, dtype=np.complex128)
            r_r = np.eye(2, dtype=np.complex128)
            if name == "nondiagonal_residual":
                r_m = np.array([[1.02, 0.03 - 0.02j], [-0.025 + 0.01j, 0.98]], dtype=np.complex128)
                r_r = np.array([[0.97, -0.02 + 0.015j], [0.018, 1.04]], dtype=np.complex128)
            beam = np.array(
                [[0.7 - 0.1j, 0.05 + 0.02j], [-0.04 + 0.03j, 0.65 + 0.08j]],
                dtype=np.complex128,
            )
            beam = beam + 0.01 * rng.normal(size=(2, 2)) + 1j * 0.01 * rng.normal(size=(2, 2))
            plane = extra.get("source", np.eye(2, dtype=np.complex128) * 1.3)
            vis = predict_row_visibility(beam, plane, r_m, r_r, moving_is_p=moving_is_p)
            cases.append(
                {
                    "name": f"{name}_{'p' if moving_is_p else 'q'}",
                    "visibility": vis,
                    "source": plane,
                    "r_moving": r_m,
                    "r_ref": r_r,
                    "moving_is_p": moving_is_p,
                    "beam": beam,
                }
            )
    return cases


def single_sample_algebraic_closure(
    cases: list[dict[str, object]] | None = None,
    *,
    tolerance: float = ALGEBRAIC_TOLERANCE,
) -> dict[str, object]:
    """Recover E from V and reconstruct V with no averaging or interpolation."""

    selected = cases if cases is not None else algebraic_closure_cases()
    errors = []
    failed = []
    for case in selected:
        beam = recover_row_jones(
            case["visibility"],
            case["source"],
            case["r_moving"],
            case["r_ref"],
            moving_is_p=bool(case["moving_is_p"]),
        )
        predicted = predict_row_visibility(
            beam,
            case["source"],
            case["r_moving"],
            case["r_ref"],
            moving_is_p=bool(case["moving_is_p"]),
        )
        error = complex_relative_error(case["visibility"], predicted)
        errors.append(error)
        if not (np.isfinite(error) and error < float(tolerance)):
            failed.append({"name": case["name"], "complex_relative_error": error})
    max_err = float(np.max(errors)) if errors else float("nan")
    passed = not failed
    return {
        "stage": "single_sample_algebraic_closure",
        "status": "pass" if passed else "fail",
        "blocking": not passed,
        "n": len(selected),
        "tolerance": float(tolerance),
        "max_complex_relative_error": max_err,
        "failed": failed,
        "notes": (FORWARD_CLOSURE_NOTE,),
    }


def gauge_refactorization_closure(
    *,
    tolerance: float = ALGEBRAIC_TOLERANCE,
) -> dict[str, object]:
    """Prove :math:`R'=RA`, :math:`E'=A^{-1}E` leaves V unchanged."""

    cases = []
    source = np.array([[1.3, 0.015 + 0.01j], [0.015 - 0.01j, 1.3]], dtype=np.complex128)
    r_m = np.array([[1.01, 0.02 - 0.01j], [-0.015, 0.99]], dtype=np.complex128)
    r_r = np.array([[0.98, -0.025j], [0.02, 1.03]], dtype=np.complex128)
    axis = np.array(
        [[1.04 + 0.03j, 0.05 - 0.02j], [-0.04 + 0.01j, 0.96 - 0.025j]], dtype=np.complex128
    )
    beam = np.array([[0.6, 0.04 + 0.02j], [-0.03 + 0.01j, 0.55]], dtype=np.complex128)
    for moving_is_p in (True, False):
        vis = predict_row_visibility(beam, source, r_m, r_r, moving_is_p=moving_is_p)
        pinned = invert_jones(axis) @ beam
        updated = r_m @ axis
        reconstructed = predict_row_visibility(
            pinned, source, updated, r_r, moving_is_p=moving_is_p
        )
        cases.append(complex_relative_error(vis, reconstructed))
    max_err = float(np.max(cases))
    passed = max_err < float(tolerance)
    return {
        "stage": "gauge_refactorization_closure",
        "status": "pass" if passed else "fail",
        "blocking": not passed,
        "tolerance": float(tolerance),
        "max_complex_relative_error": max_err,
        "tested_nondiagonal_complex_A": True,
        "notes": (FORWARD_CLOSURE_NOTE,),
    }


def _packed_source_planes(
    observation: HolographyObservation,
    rows: NDArray[np.int_],
) -> tuple[NDArray[np.complex128], NDArray[np.complex128], NDArray[np.float64]]:
    from sl1mjax.holography_diagonal import _source_coherency, source_model_stokes_i
    from sl1mjax.polarization import Receptor, pack_coherency

    packed = pack_coherency(
        observation.block.visibility,
        observation.block.correlations,
        (Receptor.R, Receptor.L),
    )
    source = _source_coherency(observation)
    if packed.ndim == 3:
        vis = packed[rows]
        source_planes = source[rows]
    else:
        vis = packed[rows, 0]
        source_planes = source[rows, 0] if source.ndim == 4 else source[rows]
    intensity = source_model_stokes_i(observation)[rows]
    return vis, source_planes, intensity


def _recover_and_predict_rows(
    vis: NDArray[np.complex128],
    source_planes: NDArray[np.complex128],
    r_m: NDArray[np.complex128],
    r_r: NDArray[np.complex128],
    moving_is_p: NDArray[np.bool_],
) -> tuple[NDArray[np.complex128], NDArray[np.complex128], NDArray[np.float64]]:
    import jax.numpy as jnp

    if vis.shape[0] == 0:
        empty = np.zeros((0, 2, 2), dtype=np.complex128)
        return empty, empty, np.zeros(0, dtype=np.float64)
    packed = vis[:, None, :, :]
    source = source_planes[:, None, :, :]
    beams, _ok = _recover_row_jones_jax(
        jnp.asarray(packed),
        jnp.asarray(source),
        jnp.asarray(r_m),
        jnp.asarray(r_r),
        jnp.asarray(moving_is_p),
    )
    beams = np.asarray(beams)[:, 0]
    predicted = np.asarray(
        _predict_vis_jax(
            jnp.asarray(r_m),
            jnp.asarray(beams[:, None, :, :]),
            jnp.asarray(source),
            jnp.asarray(r_r),
            jnp.asarray(moving_is_p),
        )
    )[:, 0]
    scale = np.maximum(np.abs(vis), 1.0e-12)
    errors = np.max(np.abs(predicted - vis) / scale, axis=(-2, -1))
    return beams, predicted, np.asarray(errors, dtype=np.float64)


def _group_mean(values: NDArray, keys: NDArray) -> NDArray:
    uniq, inverse = np.unique(np.asarray(keys), axis=0, return_inverse=True)
    acc = np.zeros((uniq.shape[0],) + values.shape[1:], dtype=values.dtype)
    counts = np.zeros((uniq.shape[0],) + (1,) * (values.ndim - 1), dtype=np.float64)
    np.add.at(acc, inverse, values)
    np.add.at(counts, inverse, 1.0)
    return acc[inverse] / np.maximum(counts[inverse], 1.0)


def compact_representative_rows(
    observation: HolographyObservation,
    *,
    max_rows: int = 64,
    seed: int = 32,
) -> NDArray[np.bool_]:
    """Compact rows covering both raster passes, several movers/refs, and all hands."""

    geometry = moving_reference_row_geometry(observation)
    usable = np.asarray(geometry["usable"], dtype=bool)
    selected = np.zeros(usable.shape, dtype=bool)
    rows = np.flatnonzero(usable)
    if rows.size == 0:
        return selected
    moving = np.asarray(geometry["moving_id"], dtype=np.int32)
    reference = np.asarray(geometry["reference_id"], dtype=np.int32)
    radius = np.asarray(geometry["radius_rad"], dtype=np.float64)
    memo = Memo195LowerCRaster()
    raster = np.where(
        radius[rows] * (180.0 * 60.0 / np.pi) <= memo.dense_radius_arcmin,
        "dense",
        "sparse",
    )
    keys = np.stack(
        [moving[rows], reference[rows], (raster == "dense").astype(np.int32)],
        axis=1,
    )
    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    for key in np.unique(keys, axis=0):
        members = rows[np.all(keys == key, axis=1)]
        order = members[np.argsort(radius[members])]
        take = [int(order[0]), int(order[order.size // 2]), int(order[-1])]
        chosen.extend(take)
    chosen = list(dict.fromkeys(chosen))
    if len(chosen) > int(max_rows):
        origin = rows[radius[rows] <= 1.0e-6]
        keep = set(int(row) for row in origin[: max(4, int(max_rows) // 8)])
        rest = [row for row in chosen if row not in keep]
        extra = rng.choice(rest, size=min(len(rest), int(max_rows) - len(keep)), replace=False)
        chosen = sorted(keep.union(int(row) for row in extra))
    selected[np.asarray(chosen, dtype=np.int64)] = True
    return selected


def _stratify_residuals(
    vis: NDArray[np.complex128],
    predicted: NDArray[np.complex128],
    *,
    intensity: NDArray[np.float64],
    moving: NDArray[np.int32],
    reference: NDArray[np.int32],
    time_index: NDArray[np.int32],
    radius: NDArray[np.float64],
    raster: NDArray,
    beam_amp: NDArray[np.float64],
) -> dict[str, dict[str, dict[str, float]]]:
    radius_bin = np.where(
        radius <= 0.002, "r_le_2mrad", np.where(radius <= 0.006, "r_2_6mrad", "r_gt_6mrad")
    )
    amp_bin = np.where(
        beam_amp >= 0.7, "amp_ge_0.7", np.where(beam_amp >= 0.3, "amp_0.3_0.7", "amp_lt_0.3")
    )
    groups = {
        "mover": moving.astype(str),
        "reference": reference.astype(str),
        "visit": time_index.astype(str),
        "raster": np.asarray(raster).astype(str),
        "beam_radius": radius_bin,
        "beam_amplitude": amp_bin,
    }
    report: dict[str, dict[str, dict[str, float]]] = {}
    for axis, labels in groups.items():
        axis_report = {}
        for label in np.unique(labels):
            keep = labels == label
            if int(np.sum(keep)) < 3:
                continue
            axis_report[str(label)] = residual_hands(
                vis[keep], predicted[keep], stokes_i=intensity[keep]
            )
        report[axis] = axis_report
    return report


def per_sample_map_closure(
    observation: HolographyObservation,
    *,
    residual_jones: Mapping[int, ArrayLike],
    parallactic_angle_rad: ArrayLike,
    row_mask: ArrayLike,
    tolerance: float = ALGEBRAIC_TOLERANCE,
) -> dict[str, object]:
    """Predict selected rows from the Jones recovered on that row alone."""

    mask = np.asarray(row_mask, dtype=bool).reshape(-1)
    geometry = moving_reference_row_geometry(observation)
    rows = np.flatnonzero(mask & geometry["usable"])
    vis, source_planes, intensity = _packed_source_planes(observation, rows)
    _moving, _reference, _time_index, r_m, r_r, moving_is_p = _row_rime_terms(
        observation, residual_jones, parallactic_angle_rad, rows
    )
    beams, predicted, errors = _recover_and_predict_rows(vis, source_planes, r_m, r_r, moving_is_p)
    det_s = np.abs(np.linalg.det(source_planes)) if source_planes.size else np.zeros(0)
    det_m = np.abs(np.linalg.det(r_m)) if r_m.size else np.zeros(0)
    det_r = np.abs(np.linalg.det(r_r)) if r_r.size else np.zeros(0)
    vis_amp = np.max(np.abs(vis), axis=(-2, -1)) if vis.size else np.zeros(0)
    well = (
        np.isfinite(errors)
        & (det_s > 1.0e-8)
        & (det_m > 1.0e-8)
        & (det_r > 1.0e-8)
        & (vis_amp > 1.0e-3)
    )
    well_errors = errors[well]
    max_err = (
        float(np.max(errors[np.isfinite(errors)])) if np.any(np.isfinite(errors)) else float("nan")
    )
    max_well = float(np.max(well_errors)) if well_errors.size else float("nan")
    passed = bool(well_errors.size) and max_well < float(tolerance)
    hands = (
        residual_hands(vis[well], predicted[well], stokes_i=intensity[well])
        if np.any(well)
        else residual_hands(vis, predicted, stokes_i=intensity)
    )
    worst = np.array([], dtype=np.int64)
    if errors.size:
        rank = np.argsort(np.where(np.isfinite(errors), errors, -np.inf))[::-1]
        worst = rows[rank[: min(8, rank.size)]]
    return {
        "stage": "per_sample_map_closure",
        "status": "pass" if passed else "fail",
        "blocking": not passed,
        "n": int(rows.size),
        "n_well_conditioned": int(np.sum(well)),
        "n_ill_conditioned": int(rows.size - np.sum(well)),
        "tolerance": float(tolerance),
        "max_complex_relative_error": max_err,
        "max_well_conditioned_error": max_well,
        "median_complex_relative_error": float(np.median(errors[np.isfinite(errors)]))
        if np.any(np.isfinite(errors))
        else float("nan"),
        "p99_complex_relative_error": float(np.quantile(errors[np.isfinite(errors)], 0.99))
        if np.sum(np.isfinite(errors)) >= 8
        else max_err,
        "worst_row_indices": worst.tolist(),
        "conditioning_note": (
            "Pass/fail uses well-conditioned rows only (|det S|,|det R| > 1e-8 and |V| > 1e-3)."
        ),
        **hands,
        "notes": (FORWARD_CLOSURE_NOTE,),
    }


def _row_rime_terms(
    observation: HolographyObservation,
    residual_jones: Mapping[int, ArrayLike],
    parallactic_angle_rad: ArrayLike,
    rows: NDArray[np.int_],
) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray]:
    import jax.numpy as jnp

    from sl1mjax.holography_full_jones import _sky_frame_residual_jax

    geometry = moving_reference_row_geometry(observation)
    chi = np.asarray(parallactic_angle_rad, dtype=np.float64)
    moving = np.asarray(geometry["moving_id"], dtype=np.int32)[rows]
    reference = np.asarray(geometry["reference_id"], dtype=np.int32)[rows]
    time_index = np.asarray(geometry["time_index"], dtype=np.int32)[rows]
    r_m = _antenna_jones_planes(residual_jones, moving)
    r_r = _antenna_jones_planes(residual_jones, reference)
    r_m = np.asarray(
        _sky_frame_residual_jax(jnp.asarray(r_m), jnp.asarray(chi[time_index, moving]))
    )
    r_r = np.asarray(
        _sky_frame_residual_jax(jnp.asarray(r_r), jnp.asarray(chi[time_index, reference]))
    )
    moving_is_p = np.asarray(observation.block.antenna1, dtype=np.int32)[rows] == moving
    return moving, reference, time_index, r_m, r_r, moving_is_p


def cell_aggregation_closure(
    observation: HolographyObservation,
    *,
    residual_jones: Mapping[int, ArrayLike],
    parallactic_angle_rad: ArrayLike,
    row_mask: ArrayLike,
    copolar_threshold: float = 0.05,
) -> dict[str, object]:
    """Measure the copolar error introduced by each map-aggregation step."""

    import jax.numpy as jnp

    mask = np.asarray(row_mask, dtype=bool).reshape(-1)
    geometry = moving_reference_row_geometry(observation)
    rows = np.flatnonzero(mask & geometry["usable"])
    vis, source_planes, intensity = _packed_source_planes(observation, rows)
    moving, reference, time_index, r_m, r_r, moving_is_p = _row_rime_terms(
        observation, residual_jones, parallactic_angle_rad, rows
    )
    beams, direct, _errors = _recover_and_predict_rows(vis, source_planes, r_m, r_r, moving_is_p)

    def _predict(beam_planes: NDArray[np.complex128]) -> NDArray[np.complex128]:
        if beam_planes.shape[0] == 0:
            return np.zeros((0, 2, 2), dtype=np.complex128)
        return np.asarray(
            _predict_vis_jax(
                jnp.asarray(r_m),
                jnp.asarray(beam_planes[:, None, :, :]),
                jnp.asarray(source_planes[:, None, :, :]),
                jnp.asarray(r_r),
                jnp.asarray(moving_is_p),
            )
        )[:, 0]

    pred_time = _predict(_group_mean(beams, np.stack([moving, time_index], axis=1)))
    cell_l = np.asarray(geometry["cell_l"])[rows]
    cell_m = np.asarray(geometry["cell_m"])[rows]
    pred_cell = _predict(_group_mean(beams, np.stack([moving, cell_l, cell_m], axis=1)))
    steps = {
        "per_row_direct": residual_hands(vis, direct, stokes_i=intensity),
        "reference_combination": residual_hands(vis, pred_time, stokes_i=intensity),
        "visit_combination": residual_hands(vis, pred_cell, stokes_i=intensity),
    }
    production = recover_holography_full_jones(
        observation,
        residual_jones=residual_jones,
        parallactic_angle_rad=parallactic_angle_rad,
        row_mask=mask,
    )
    pred_map = predict_moving_reference_from_beam(
        observation,
        residual_jones=residual_jones,
        artifact=production,
        row_mask=mask,
        parallactic_angle_rad=parallactic_angle_rad,
    )
    pred_plane = pred_map[rows, 0] if pred_map.ndim == 4 else pred_map[rows]
    steps["production_map"] = residual_hands(vis, pred_plane, stokes_i=intensity)
    pinned, residual_p, _origin = refactor_on_axis_gauge(production, residual_jones)
    pred_pin = predict_moving_reference_from_beam(
        observation,
        residual_jones=residual_p,
        artifact=pinned,
        row_mask=mask,
        parallactic_angle_rad=parallactic_angle_rad,
    )
    pred_pin_plane = pred_pin[rows, 0] if pred_pin.ndim == 4 else pred_pin[rows]
    steps["after_on_axis_gauge"] = residual_hands(vis, pred_pin_plane, stokes_i=intensity)
    first_failure = None
    for name, report in steps.items():
        rr = report.get("median_abs_rr_over_i")
        ll = report.get("median_abs_ll_over_i")
        if (
            rr is not None
            and ll is not None
            and np.isfinite(rr)
            and np.isfinite(ll)
            and max(float(rr), float(ll)) >= float(copolar_threshold)
        ):
            first_failure = name
            break
    radius = np.asarray(geometry["radius_rad"], dtype=np.float64)[rows]
    memo = Memo195LowerCRaster()
    raster = np.where(
        radius * (180.0 * 60.0 / np.pi) <= memo.dense_radius_arcmin,
        "dense",
        "sparse",
    )
    beam_amp = np.abs(beams[:, 0, 0]) if beams.size else np.zeros(0, dtype=np.float64)
    failed_pred = {
        "per_row_direct": direct,
        "reference_combination": pred_time,
        "visit_combination": pred_cell,
        "production_map": pred_plane,
        "after_on_axis_gauge": pred_pin_plane,
    }.get(first_failure or "after_on_axis_gauge", pred_pin_plane)
    return {
        "stage": "cell_aggregation_closure",
        "status": "pass" if first_failure is None else "fail",
        "blocking": first_failure is not None,
        "first_failing_operation": first_failure,
        "copolar_threshold": float(copolar_threshold),
        "steps": steps,
        "stratified": _stratify_residuals(
            vis,
            failed_pred,
            intensity=intensity,
            moving=moving,
            reference=reference,
            time_index=time_index,
            radius=radius,
            raster=raster,
            beam_amp=beam_amp,
        ),
        "n": int(rows.size),
        "notes": (FORWARD_CLOSURE_NOTE,),
    }


def dummy_support_artifact(
    offsets_by_antenna: Mapping[int, ArrayLike],
    *,
    frequency_hz: float = 4.564e9,
    copolar_valid: bool = True,
    off_diagonal_valid: bool = True,
) -> HolographyFullJonesArtifact:
    """Identity-Jones samples used only to share the interpolator predicate."""

    memo = Memo195LowerCRaster()
    samples = []
    planes = []
    for antenna, values in offsets_by_antenna.items():
        for offset in np.asarray(values, dtype=np.float64).reshape(-1, 2):
            if not np.all(np.isfinite(offset)):
                continue
            plane = np.eye(2, dtype=np.complex128)
            sample = HolographyFullJonesSample(
                moving_antenna_id=int(antenna),
                unique_time_s=0.0,
                offset_lm_rad=np.asarray(offset, dtype=np.float64),
                frequency_hz=float(frequency_hz),
                jones=plane,
                sigma=np.ones((2, 2), dtype=np.float64),
                n_reference=1,
                weight=1.0,
                copolar_valid=bool(copolar_valid),
                off_diagonal_valid=bool(off_diagonal_valid),
                raster=_raster_label(offset, memo),
            )
            samples.append(sample)
            planes.append(plane)
    if not samples:
        raise ValueError("support artifact needs at least one finite training offset")
    return HolographyFullJonesArtifact(
        samples=tuple(samples),
        jones=np.stack(planes, axis=0),
        valid=np.ones(len(samples), dtype=bool),
        off_diagonal_valid=np.ones(len(samples), dtype=bool),
        calibration_state="casa_parang_true",
        source_name="support_predicate",
        source_model_version="identity",
        reference_combination="support_predicate",
        receptor_convention="circular_R_L",
        offset_sign="commanded_pointing",
        frozen=False,
        schema_version=HOLOGRAPHY_FULL_JONES_SCHEMA_VERSION,
    )


def interpolator_support_category(
    support: FrozenInterpolationSupport,
    antenna_id: int,
    offset_lm_rad: ArrayLike,
    *,
    frequency_hz: float = 4.564e9,
) -> dict[str, object]:
    """Category from the same predicate that produces a finite interpolation."""

    query = np.asarray(offset_lm_rad, dtype=np.float64).reshape(2)
    if not np.all(np.isfinite(query)):
        return {
            "category": "unsupported",
            "finite": False,
            "distance_rad": float("nan"),
            "inside_training_bbox": False,
        }
    offsets = support.offsets_by_antenna.get(int(antenna_id))
    if offsets is None or np.asarray(offsets).size == 0:
        return {
            "category": "unsupported",
            "finite": False,
            "distance_rad": float("nan"),
            "inside_training_bbox": False,
        }
    artifact = dummy_support_artifact({int(antenna_id): offsets}, frequency_hz=frequency_hz)
    _plane, ok, _leak = interpolate_holography_full_jones(
        artifact,
        query,
        moving_antenna_id=int(antenna_id),
        frequency_hz=float(frequency_hz),
    )
    geometry = support.classify_offset(int(antenna_id), query)
    if ok and str(geometry["category"]) == "exact":
        category = "exact"
    elif ok:
        category = "interpolation"
    elif not bool(geometry["inside_training_bbox"]):
        category = "extrapolation"
    else:
        category = "unsupported"
    return {
        "category": category,
        "finite": bool(ok),
        "distance_rad": geometry["distance_rad"],
        "inside_training_bbox": geometry["inside_training_bbox"],
        "n_neighbors": geometry["n_neighbors"],
    }


def spatial_support_contract(
    support: FrozenInterpolationSupport,
    queries: Mapping[str, tuple[int, ArrayLike]],
    *,
    frequency_hz: float = 4.564e9,
) -> dict[str, object]:
    """Require support labels to match finite interpolator output."""

    results = {}
    mismatches = []
    for name, (antenna, offset) in queries.items():
        report = interpolator_support_category(
            support, int(antenna), offset, frequency_hz=frequency_hz
        )
        results[name] = report
        finite = bool(report["finite"])
        supported = str(report["category"]) in {"exact", "interpolation"}
        if finite != supported:
            mismatches.append(name)
    passed = not mismatches
    return {
        "stage": "spatial_support_contract",
        "status": "pass" if passed else "fail",
        "blocking": not passed,
        "mismatches": mismatches,
        "cases": results,
        "notes": (FORWARD_CLOSURE_NOTE,),
    }


def classify_forward_closure(stages: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    """Record the first failed stage. Later stages may still be diagnostic."""

    first_failure = None
    for name in FORWARD_CLOSURE_STAGES:
        report = stages.get(name)
        if report is None:
            continue
        if report.get("status") == "fail" and first_failure is None:
            first_failure = name
    return {
        "first_failure": first_failure,
        "empirical_copolar_visibility_closure": first_failure
        in {None, "exact_cell_holdouts", "remaining_one_axis_diagnostics"},
        "deterministic_closure_passed": first_failure is None
        or first_failure in {"exact_cell_holdouts", "remaining_one_axis_diagnostics"},
        "comparison_interpretable": first_failure is None
        or first_failure in {"exact_cell_holdouts", "remaining_one_axis_diagnostics"},
        "do_not_pool_axes": True,
        "notes": (FORWARD_CLOSURE_NOTE,),
    }
