"""Inner-fold Phase 6 protocol: masks, guard scoring, and run products."""

from __future__ import annotations

import hashlib
import json
import resource
import shutil
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from sl1mjax.beam_aware_imaging import (
    ComponentFamily,
    SkyComponent,
    SkyComponentTable,
    sky_table_from_records,
    sky_table_to_records,
)
from sl1mjax.beam_conventions import BeamCalibrationState, require_beam_calibration_state
from sl1mjax.beam_operator import BeamOperatorConfig
from sl1mjax.calibration_terms import parallactic_angle_rad
from sl1mjax.cassbeam_beam import CASSBEAM_CBAND_MODEL_ID
from sl1mjax.data.canonical import VisibilityBlock
from sl1mjax.finite_pixel import (
    IntegrationPlan,
    adjoint_voltage_from_plan,
    integration_plan_from_table,
    predict_voltage_from_plan,
    predict_voltage_from_plan_value_and_grad,
)
from sl1mjax.inference import InferenceConfig
from sl1mjax.integration_planner import (
    integration_plan_from_planner,
    plan_integration,
)
from sl1mjax.quadtree import QuadtreeLeaf
from sl1mjax.split import interleaved_time_fold_masks
from sl1mjax.voltage_beam import (
    AnalyticAiryVoltageBeam,
    CompositeScalarVoltageBeam,
    VoltageBeamModel,
)
from sl1mjax.voltage_flux_refit import score_visibility_prediction
from sl1mjax.voltage_reconstruction import (
    PRODUCTION_STOKES_I_BEAMS,
    VoltageFitResult,
    VoltageReconstructionConfig,
    VoltageReconstructionResult,
    _holdout_improved,
    _iter_bounded_batches,
)
from sl1mjax.wide_field_sky import (
    CENTRAL_ROOT_SIZE,
    activate_guard_roots,
    is_boundary_guard_leaf,
    is_outer_edge_guard_leaf,
    phase5_render_spec,
    render_intrinsic_stokes_i,
)

FOLD_BIN_SECONDS = 60.0
TRAIN_FOLDS = (0, 1, 2)
HOLDOUT_FOLD = 3
SEALED_FOLD = 4
FIELD_EXPANSION = "field_expansion_required"
GUARD_ACTIVATED = "supported_coarse_activation"
GUARD_QUIET = "inactive"


@dataclass(frozen=True)
class Phase6Folds:
    """Interleaved 60 s time-bin folds with the sealed fold held closed."""

    train: tuple[np.ndarray, ...]
    holdout: tuple[np.ndarray, ...]
    sealed: tuple[np.ndarray, ...]
    bin_seconds: float = FOLD_BIN_SECONDS
    train_folds: tuple[int, ...] = TRAIN_FOLDS
    holdout_fold: int = HOLDOUT_FOLD
    sealed_fold: int = SEALED_FOLD


@dataclass(frozen=True)
class GuardDecision:
    """One outer-guard scoring result."""

    leaf: QuadtreeLeaf
    gradient: float
    near_boundary: bool
    action: str
    predicted_improvement: float


@dataclass(frozen=True)
class GuardReport:
    """Batched inactive-guard screen and any field-expansion signal."""

    decisions: tuple[GuardDecision, ...]
    activated: tuple[QuadtreeLeaf, ...]
    field_expansion: bool
    status: str
    accepted: bool = False


def phase6_folds(
    blocks: Sequence[VisibilityBlock],
    *,
    bin_seconds: float = FOLD_BIN_SECONDS,
) -> Phase6Folds:
    train, holdout, sealed = interleaved_time_fold_masks(
        tuple(blocks),
        bin_seconds=bin_seconds,
        fold_count=5,
        validation_fold=HOLDOUT_FOLD,
        test_fold=SEALED_FOLD,
    )
    validate_phase6_masks(tuple(blocks), train, holdout, sealed)
    return Phase6Folds(train=train, holdout=holdout, sealed=sealed, bin_seconds=bin_seconds)


def validate_phase6_masks(
    blocks: Sequence[VisibilityBlock],
    train: Sequence[np.ndarray],
    holdout: Sequence[np.ndarray],
    sealed: Sequence[np.ndarray] | None = None,
) -> None:
    if len(train) != len(blocks) or len(holdout) != len(blocks):
        raise ValueError("masks must contain one array per block")
    if sealed is not None and len(sealed) != len(blocks):
        raise ValueError("sealed masks must contain one array per block")
    for index, block in enumerate(blocks):
        for name, mask in (("train", train[index]), ("holdout", holdout[index])):
            if mask.shape != block.shape:
                raise ValueError(f"{name}_masks[{index}] must match its visibility block")
        if np.any(train[index] & holdout[index]):
            raise ValueError("train and holdout masks must be disjoint")
        if sealed is not None:
            if sealed[index].shape != block.shape:
                raise ValueError(f"sealed_masks[{index}] must match its visibility block")
            if np.any(train[index] & sealed[index]) or np.any(holdout[index] & sealed[index]):
                raise ValueError("train, holdout, and sealed masks must be disjoint")


def assert_sealed_fold_unused(
    used_masks: Sequence[np.ndarray],
    sealed: Sequence[np.ndarray],
) -> None:
    for used, closed in zip(used_masks, sealed, strict=True):
        if np.any(used & closed):
            raise ValueError("sealed fold samples were used")


def poison_sealed_visibilities(
    blocks: Sequence[VisibilityBlock],
    sealed: Sequence[np.ndarray],
    *,
    amplitude: float = 1.0e6,
) -> tuple[VisibilityBlock, ...]:
    poisoned = []
    for block, mask in zip(blocks, sealed, strict=True):
        visibility = np.asarray(block.visibility, dtype=np.complex128).copy()
        visibility[np.asarray(mask, dtype=bool)] += amplitude
        payload = {key: getattr(block, key) for key in block.__dataclass_fields__}
        payload["visibility"] = visibility
        poisoned.append(VisibilityBlock(**payload))
    return tuple(poisoned)


def score_inactive_guard(
    fit: VoltageFitResult,
    blocks: Sequence[VisibilityBlock],
    beam: VoltageBeamModel,
    *,
    antenna_position_m: np.ndarray,
    calibration_state: BeamCalibrationState | str,
    train_masks: Sequence[np.ndarray],
    holdout_masks: Sequence[np.ndarray],
    activation_threshold: float = 1e-4,
    structure_ratio: float = 0.15,
    batch_size_rows: int = 256,
) -> GuardReport:
    """Score inactive guard roots from the training residual adjoint."""

    del structure_ratio
    state = require_beam_calibration_state(calibration_state)
    batch_config = VoltageReconstructionConfig(
        inference=InferenceConfig(
            batch_size_rows=int(batch_size_rows),
            batch_grouping="times",
        )
    )
    inactive = [
        component
        for component in fit.table.components
        if component.family is ComponentFamily.OUTER_GUARD and not component.active
    ]
    if not inactive:
        return GuardReport((), (), False, GUARD_QUIET)
    print(f"guard inactive={len(inactive)}", flush=True)
    virtual_records = []
    for component in inactive:
        record = {
            "component_id": component.component_id,
            "family": component.family.value,
            "basis_type": component.basis_type.value,
            "l_rad": component.l_rad,
            "m_rad": component.m_rad,
            "stokes_i_jy": 0.0,
            "width_rad": component.width_rad,
            "level": component.level,
            "iy": component.iy,
            "ix": component.ix,
            "parent_id": component.parent_id,
            "active": True,
            "splitting_permitted": False,
            "provenance": dict(component.provenance),
        }
        virtual_records.append(record)
    virtual = sky_table_from_records(
        virtual_records,
        mosaic_phase_centre_rad=fit.table.mosaic_phase_centre_rad,
        source="phase5_virtual_guard",
    )
    plan = integration_plan_from_table(virtual)
    parent_id = {component_id: index for index, component_id in enumerate(plan.parent_id)}
    gradient = np.zeros(plan.parent_count, dtype=np.float64)
    weight_sum = 0.0
    residuals = tuple(fit.residuals)
    n_batch = 0
    for batch in _iter_bounded_batches(blocks, train_masks, batch_config):
        n_batch += 1
        if n_batch == 1 or n_batch % 20 == 0:
            print(f"guard batch {n_batch}", flush=True)
        residual = residuals[batch.source_index]
        packed = np.zeros_like(batch.block.visibility)
        count = int(batch.source_rows.size)
        packed[:count] = residual[batch.source_rows]
        selected = np.asarray(batch.mask, dtype=bool) & batch.block.active
        finite = np.isfinite(batch.block.weight) & (batch.block.weight > 0)
        weight = np.where(selected & finite, batch.block.weight, 0.0)
        weight_sum += float(np.sum(weight))
        adjoint = np.asarray(
            adjoint_voltage_from_plan(
                np.where(weight > 0, packed * weight, 0.0),
                batch.block,
                plan,
                beam,
                antenna_position_m=antenna_position_m,
                calibration_state=state,
                backend="jax",
            ),
            dtype=np.float64,
        )
        gradient += adjoint
    if weight_sum <= 0:
        raise ValueError("train masks contain no finite positive-weight samples")
    gradient *= 2.0 / weight_sum
    shortlist = [
        component
        for component in inactive
        if 0.5 * float(gradient[parent_id[component.component_id]]) ** 2 >= activation_threshold
    ]
    if shortlist:
        print(f"guard curvature {len(shortlist)}", flush=True)
    curvature = {
        component.component_id: _guard_curvature(
            component,
            blocks,
            train_masks,
            beam,
            antenna_position_m=antenna_position_m,
            calibration_state=state,
            mosaic_phase_centre_rad=fit.table.mosaic_phase_centre_rad,
            batch_size_rows=int(batch_size_rows),
        )
        for component in shortlist
    }
    del holdout_masks
    decisions = []
    activated = []
    expansion = False
    for component in inactive:
        leaf = QuadtreeLeaf(
            int(component.level or 0),
            int(component.iy or 0),
            int(component.ix or 0),
        )
        value = float(gradient[parent_id[component.component_id]])
        hessian = max(float(curvature.get(component.component_id, 0.0)), 1e-12)
        improvement = 0.0 if component not in shortlist else max(0.0, 0.5 * value * value / hessian)
        near_inner = is_boundary_guard_leaf(leaf)
        outer_edge = is_outer_edge_guard_leaf(leaf)
        action = GUARD_QUIET
        if improvement >= activation_threshold:
            if outer_edge:
                action = FIELD_EXPANSION
                expansion = True
            else:
                action = GUARD_ACTIVATED
                activated.append(leaf)
        decisions.append(
            GuardDecision(
                leaf=leaf,
                gradient=value,
                near_boundary=near_inner,
                action=action,
                predicted_improvement=improvement,
            )
        )
    status = FIELD_EXPANSION if expansion else (GUARD_ACTIVATED if activated else GUARD_QUIET)
    return GuardReport(tuple(decisions), tuple(activated), expansion, status)


def accept_inactive_guard(
    fit: VoltageFitResult,
    blocks: Sequence[VisibilityBlock],
    beam: VoltageBeamModel,
    *,
    antenna_position_m: np.ndarray,
    calibration_state: BeamCalibrationState | str,
    train_masks: Sequence[np.ndarray],
    holdout_masks: Sequence[np.ndarray],
    refit: Callable[[SkyComponentTable], VoltageFitResult],
    activation_threshold: float = 1e-4,
) -> tuple[VoltageFitResult, GuardReport]:
    """Screen on train, provisionally activate, refit, and accept on holdout."""

    report = score_inactive_guard(
        fit,
        blocks,
        beam,
        antenna_position_m=antenna_position_m,
        calibration_state=calibration_state,
        train_masks=train_masks,
        holdout_masks=holdout_masks,
        activation_threshold=activation_threshold,
    )
    if report.field_expansion or not report.activated:
        return fit, report
    proposed = activate_guard_roots(fit.table, report.activated)
    candidate = refit(proposed)
    if not _holdout_improved(fit, candidate):
        return fit, replace(report, activated=(), status=GUARD_QUIET, accepted=False)
    return candidate, replace(report, status=GUARD_ACTIVATED, accepted=True)


def _guard_curvature(
    component: SkyComponent,
    blocks: Sequence[VisibilityBlock],
    train_masks: Sequence[np.ndarray],
    beam: VoltageBeamModel,
    *,
    antenna_position_m: np.ndarray,
    calibration_state: BeamCalibrationState,
    mosaic_phase_centre_rad: tuple[float, float],
    batch_size_rows: int = 256,
) -> float:
    virtual = sky_table_from_records(
        [
            {
                "component_id": component.component_id,
                "family": component.family.value,
                "basis_type": component.basis_type.value,
                "l_rad": component.l_rad,
                "m_rad": component.m_rad,
                "stokes_i_jy": 1.0,
                "width_rad": component.width_rad,
                "level": component.level,
                "iy": component.iy,
                "ix": component.ix,
                "parent_id": component.parent_id,
                "active": True,
                "splitting_permitted": False,
                "provenance": dict(component.provenance),
            }
        ],
        mosaic_phase_centre_rad=mosaic_phase_centre_rad,
        source="phase5_guard_curvature",
    )
    plan = integration_plan_from_planner(
        virtual,
        plan_integration(
            virtual,
            tuple(blocks),
            beam,
            antenna_position_m=antenna_position_m,
            calibration_state=calibration_state,
        ),
    )
    numerator = 0.0
    denominator = 0.0
    batch_config = VoltageReconstructionConfig(
        inference=InferenceConfig(
            batch_size_rows=max(1, int(batch_size_rows)),
            batch_grouping="times",
        )
    )
    for batch in _iter_bounded_batches(blocks, train_masks, batch_config):
        selected = np.asarray(batch.mask, dtype=bool) & batch.block.active
        finite = np.isfinite(batch.block.weight) & (batch.block.weight > 0)
        weight = np.where(selected & finite, batch.block.weight, 0.0)
        predicted = predict_voltage_from_plan(
            batch.block,
            plan,
            np.ones(plan.parent_count, dtype=np.float64),
            beam,
            antenna_position_m=antenna_position_m,
            calibration_state=calibration_state,
            backend="jax",
        ).visibility
        usable = weight > 0
        numerator += float(np.sum(weight[usable] * np.abs(np.asarray(predicted)[usable]) ** 2))
        denominator += float(np.sum(weight[usable]))
    if denominator <= 0:
        return 0.0
    return 2.0 * numerator / denominator


def apply_guard_report(table: SkyComponentTable, report: GuardReport) -> SkyComponentTable:
    if report.field_expansion:
        raise ValueError("outer guard requests a larger refinable field")
    if not report.activated:
        return table
    return activate_guard_roots(table, report.activated)


def loss_breakdowns(
    blocks: Sequence[VisibilityBlock],
    predictions: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
    *,
    antenna_position_m: np.ndarray,
    pointing_ids: Sequence[str],
) -> list[dict[str, Any]]:
    reports = []
    for block, prediction, mask, pointing_id in zip(
        blocks, predictions, masks, pointing_ids, strict=True
    ):
        payload = score_visibility_prediction(block, prediction, mask=mask)
        payload["pointing_id"] = pointing_id
        payload["by_baseline"] = _baseline_bins(block, prediction, mask)
        payload["by_pa"] = _pa_bins(block, prediction, mask, antenna_position_m)
        reports.append(payload)
    return reports


def _baseline_bins(
    block: VisibilityBlock, prediction: np.ndarray, mask: np.ndarray
) -> list[dict[str, Any]]:
    residual = np.asarray(prediction) - block.visibility
    length = np.hypot(block.uvw_m[:, 0], block.uvw_m[:, 1])
    edges = np.quantile(length, np.linspace(0.0, 1.0, 5)) if length.size else np.array([0.0, 1.0])
    edges = np.unique(edges)
    selected = np.asarray(mask, dtype=bool) & block.active
    rows = []
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        row_mask = (length >= low) & (length <= high)
        sample = np.zeros(block.shape, dtype=bool)
        sample[row_mask] = selected[row_mask]
        power = float(
            np.sum(
                np.where(sample, block.weight, 0.0) * np.abs(np.where(sample, residual, 0.0)) ** 2
            )
        )
        weight = float(np.sum(np.where(sample, block.weight, 0.0)))
        rows.append(
            {
                "uv_min_m": float(low),
                "uv_max_m": float(high),
                "held_out_loss": power / weight if weight > 0 else float("nan"),
            }
        )
    return rows


def _pa_bins(
    block: VisibilityBlock,
    prediction: np.ndarray,
    mask: np.ndarray,
    antenna_position_m: np.ndarray,
) -> list[dict[str, Any]]:
    residual = np.asarray(prediction) - block.visibility
    selected = np.asarray(mask, dtype=bool) & block.active
    chi = parallactic_angle_rad(
        block.time_s,
        block.phase_centre_rad,
        antenna_position_m,
    )
    if chi.ndim > 1:
        chi = np.mean(chi, axis=tuple(range(1, chi.ndim)))
    edges = np.deg2rad(np.linspace(-180.0, 180.0, 7))
    rows = []
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        row_mask = (chi >= low) & (chi < high)
        sample = np.zeros(block.shape, dtype=bool)
        sample[row_mask] = selected[row_mask]
        power = float(
            np.sum(
                np.where(sample, block.weight, 0.0) * np.abs(np.where(sample, residual, 0.0)) ** 2
            )
        )
        weight = float(np.sum(np.where(sample, block.weight, 0.0)))
        rows.append(
            {
                "pa_min_rad": float(low),
                "pa_max_rad": float(high),
                "held_out_loss": power / weight if weight > 0 else float("nan"),
            }
        )
    return rows


def peak_rss_bytes() -> int:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    rss = int(usage.ru_maxrss)
    if rss > 1 << 32:
        return rss
    return rss * 1024


def write_reconstruction_products(
    directory: Path,
    result: VoltageReconstructionResult,
    blocks: Sequence[VisibilityBlock],
    *,
    pointing_ids: Sequence[str],
    antenna_position_m: np.ndarray,
    folds: Phase6Folds,
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    guard: GuardReport | None = None,
    compile_count: int | None = None,
) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    table_path = directory / "component_table.json"
    table_path.write_text(
        json.dumps(
            {
                "mosaic_phase_centre_rad": list(result.table.mosaic_phase_centre_rad),
                "source": result.table.source,
                "components": sky_table_to_records(result.table),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    spec = phase5_render_spec(result.table.mosaic_phase_centre_rad)
    image = render_intrinsic_stokes_i(result.table, spec=spec)
    coordinates = spec.to_arrays()
    np.savez_compressed(
        directory / "intrinsic_stokes_i.npz",
        image=image,
        grid_size=np.asarray(coordinates["grid_size"]),
        pixel_size_rad=np.asarray(coordinates["pixel_size_rad"]),
        phase_centre_ra_rad=np.asarray(coordinates["phase_centre_ra_rad"]),
        phase_centre_dec_rad=np.asarray(coordinates["phase_centre_dec_rad"]),
        half_width_rad=np.asarray(coordinates["half_width_rad"]),
        l_increases_with_x=np.asarray(coordinates["l_increases_with_x"]),
        m_increases_with_y=np.asarray(coordinates["m_increases_with_y"]),
        reference_pixel_y=np.asarray(coordinates["reference_pixel_y"]),
        reference_pixel_x=np.asarray(coordinates["reference_pixel_x"]),
        center_pixel=np.asarray(coordinates["center_pixel"]),
    )
    assert_sealed_fold_unused(folds.train, folds.sealed)
    assert_sealed_fold_unused(folds.holdout, folds.sealed)
    sealed_unused = True
    for pointing_id, block, prediction, residual, sealed in zip(
        pointing_ids,
        blocks,
        result.fit.predictions,
        result.fit.residuals,
        folds.sealed,
        strict=True,
    ):
        closed = np.asarray(sealed, dtype=bool)
        np.savez_compressed(
            directory / f"pointing_{pointing_id}.npz",
            prediction=_seal_array(prediction, closed),
            residual=_seal_array(residual, closed),
            visibility=_seal_array(block.visibility, closed),
            weight=_seal_array(block.weight, closed, fill=0.0),
            time_s=block.time_s,
            sealed_mask=closed,
        )
    train_losses = loss_breakdowns(
        blocks,
        result.fit.predictions,
        folds.train,
        antenna_position_m=antenna_position_m,
        pointing_ids=pointing_ids,
    )
    holdout_losses = loss_breakdowns(
        blocks,
        result.fit.predictions,
        folds.holdout,
        antenna_position_m=antenna_position_m,
        pointing_ids=pointing_ids,
    )
    audit = result.audit
    payload = {
        "beam_mode": result.beam_mode,
        "stop_reason": result.stop_reason,
        "train_loss": result.fit.train_loss,
        "holdout_loss": result.fit.holdout_loss,
        "kkt_residual": result.fit.kkt_residual,
        "converged": result.fit.converged,
        "steps": result.fit.steps,
        "optimization_curve": {
            "steps": list(result.fit.curve_steps),
            "train_loss": list(result.fit.objective_history),
            "holdout_loss": list(result.fit.holdout_history),
            "note": (
                "Train and holdout at validation_interval use the same two-batch "
                "estimator. Final train_loss/holdout_loss are the full mosaic."
            ),
        },
        "elapsed_s": result.elapsed_s,
        "diagnostics": dict(result.diagnostics),
        "train_losses": train_losses,
        "holdout_losses": holdout_losses,
        "guard": _guard_payload(guard),
        "audit_under_resolved": bool(audit.under_resolved),
        "audit_n_findings": len(audit.findings),
        "audit_n_under_resolved": len(audit.under_resolved),
        "compile_count": compile_count,
        "peak_rss_bytes": peak_rss_bytes(),
        "pointing_ids": list(pointing_ids),
        "config": dict(config),
        "manifest": dict(manifest),
        "sealed_fold_unused": sealed_unused,
    }
    (directory / "summary.json").write_text(
        json.dumps(_jsonable(payload), indent=2), encoding="utf-8"
    )
    (directory / "audit_findings.json").write_text(
        json.dumps(
            {
                "n_findings": len(audit.findings),
                "n_under_resolved": len(audit.under_resolved),
                "under_resolved": [
                    {
                        "component_id": item.component_id,
                        "pointing_id": item.pointing_id,
                        "planned_depth": item.planned_depth,
                        "audit_depth": item.audit_depth,
                        "error": item.error,
                        "threshold": item.threshold,
                    }
                    for item in audit.under_resolved
                ],
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    (directory / "integration_plan.json").write_text(
        json.dumps(
            {
                "parent_id": list(result.fit.plan.parent_id),
                "parent_count": result.fit.plan.parent_count,
                "node_count": result.fit.plan.node_count,
                "depths": result.fit.planner_report.depth_by_parent(),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return payload


def _guard_payload(report: GuardReport | None) -> dict[str, Any] | None:
    if report is None:
        return None
    return {
        "status": report.status,
        "field_expansion": report.field_expansion,
        "activated": [str(leaf) for leaf in report.activated],
        "n_scored": len(report.decisions),
        "n_boundary": sum(1 for item in report.decisions if item.near_boundary),
        "accepted": report.accepted,
    }


def _seal_array(array: np.ndarray, sealed: np.ndarray, *, fill: float | None = None) -> np.ndarray:
    out = np.asarray(array).copy()
    if fill is None:
        if np.iscomplexobj(out):
            out[sealed] = np.nan + 1j * np.nan
        else:
            out[sealed] = np.nan
    else:
        out[sealed] = fill
    return out


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def sky_and_plan_from_product(
    directory: Path,
) -> tuple[SkyComponentTable, np.ndarray, IntegrationPlan]:
    """Rebuild fitted flux and the planner-depth plan from a written product."""

    checkpoint = directory / "checkpoint.json"
    plan_path = directory / "integration_plan.json"
    if not checkpoint.is_file() or not plan_path.is_file():
        raise FileNotFoundError(
            f"{directory} needs checkpoint.json and integration_plan.json"
        )
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    table = sky_table_from_records(
        payload["components"],
        mosaic_phase_centre_rad=tuple(payload["mosaic_phase_centre_rad"]),
        source=payload.get("source", "phase6_checkpoint"),
    )
    plan_payload = json.loads(plan_path.read_text(encoding="utf-8"))
    depths = {str(key): int(value) for key, value in plan_payload["depths"].items()}
    plan = integration_plan_from_table(table, depth_by_parent=depths)
    flux = np.array(
        [item.stokes_i_jy for item in table.components if item.active],
        dtype=np.float64,
    )
    if flux.size != plan.parent_count:
        raise ValueError("product flux does not match the rebuilt plan")
    if int(plan_payload["node_count"]) != plan.node_count:
        raise ValueError("rebuilt plan node count does not match integration_plan.json")
    return table, flux, plan


def plan_fingerprint(plan: IntegrationPlan) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(plan.parent_index).tobytes())
    digest.update(np.asarray(plan.weight).tobytes())
    digest.update(np.asarray(plan.node_valid).tobytes())
    digest.update(np.int64(plan.parent_count).tobytes())
    digest.update(np.int64(plan.node_count).tobytes())
    digest.update(str(plan.approximation).encode())
    return digest.hexdigest()


def compare_operator_modes(
    parent_flux: np.ndarray,
    block: VisibilityBlock,
    plan: IntegrationPlan,
    beam: VoltageBeamModel,
    *,
    antenna_position_m: np.ndarray,
    calibration_state: BeamCalibrationState | str = "casa_parang_true",
    config: BeamOperatorConfig | None = None,
    train_mask: np.ndarray | None = None,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    batch_size_rows: int | None = None,
    product: str | None = None,
    source_hashes: Mapping[str, str] | None = None,
    sampled_rows: Sequence[int] | np.ndarray | None = None,
    planner_depths: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Compare one-batch VJP and explicit-JAX loss/gradient (Bacchus gate)."""

    import time

    import jax

    from sl1mjax.voltage_operator_jax import (
        explicit_adjoint_workspace_bytes,
        explicit_kernel_build_count,
    )

    flux = np.asarray(parent_flux, dtype=np.float64).reshape(-1)
    selected = config or BeamOperatorConfig()
    builds_before = explicit_kernel_build_count()

    def _value_and_grad(mode: str) -> tuple[Any, Any]:
        return predict_voltage_from_plan_value_and_grad(
            flux,
            block,
            plan,
            beam,
            antenna_position_m=antenna_position_m,
            calibration_state=calibration_state,
            config=selected,
            train_mask=train_mask,
            operator_mode=mode,
        )

    def _run(mode: str) -> tuple[float, np.ndarray, float]:
        loss, gradient = _value_and_grad(mode)
        jax.block_until_ready(loss)
        jax.block_until_ready(gradient)
        started = time.perf_counter()
        loss, gradient = _value_and_grad(mode)
        jax.block_until_ready(loss)
        jax.block_until_ready(gradient)
        elapsed = time.perf_counter() - started
        return float(loss), np.asarray(gradient, dtype=np.float64), elapsed

    loss_vjp, grad_vjp, vjp_s = _run("vjp")
    loss_exp, grad_exp, explicit_s = _run("explicit_jax")
    builds_after_first = explicit_kernel_build_count()
    _value_and_grad("explicit_jax")
    builds_after = explicit_kernel_build_count()
    delta = grad_exp - grad_vjp
    vjp_norm = float(np.linalg.norm(grad_vjp))
    rel_grad = (
        float(np.linalg.norm(delta) / vjp_norm)
        if vjp_norm > 0
        else float(np.linalg.norm(delta))
    )
    loss_delta = abs(loss_exp - loss_vjp)
    loss_scale = max(abs(loss_vjp), 1.0)
    nodes_per_parent = np.bincount(
        np.asarray(plan.parent_index)[np.asarray(plan.node_valid)],
        minlength=plan.parent_count,
    )
    passed = bool(
        np.isfinite(loss_exp)
        and np.isfinite(loss_vjp)
        and np.all(np.isfinite(grad_exp))
        and np.all(np.isfinite(grad_vjp))
        and loss_delta <= atol + rtol * loss_scale
        and np.allclose(grad_exp, grad_vjp, rtol=rtol, atol=atol)
        and builds_after == builds_after_first
        and builds_after - builds_before <= 1
    )
    depth_values = (
        None
        if planner_depths is None
        else {
            str(depth): int(sum(1 for item in planner_depths.values() if int(item) == depth))
            for depth in sorted({int(item) for item in planner_depths.values()})
        }
    )
    return {
        "passed": passed,
        "parent_count": int(flux.size),
        "node_count": int(plan.node_count),
        "plan_sha256": plan_fingerprint(plan),
        "nodes_per_parent": {
            str(count): int(np.sum(nodes_per_parent == count))
            for count in np.unique(nodes_per_parent)
        },
        "visibility_shape": list(block.visibility.shape),
        "sampled_rows": int(block.uvw_m.shape[0]),
        "sampled_source_rows": (
            None if sampled_rows is None else [int(item) for item in sampled_rows]
        ),
        "sampled_times": int(np.unique(block.time_s).size),
        "batch_size_rows": None if batch_size_rows is None else int(batch_size_rows),
        "pixel_chunk_size": int(selected.pixel_chunk_size),
        "visibility_chunk_size": int(selected.visibility_chunk_size),
        "depth_histogram": depth_values,
        "source_hashes": None if source_hashes is None else dict(source_hashes),
        "product": product,
        "beam_model_id": getattr(beam, "model_id", type(beam).__name__),
        "explicit_adjoint_workspace_bytes": explicit_adjoint_workspace_bytes(
            parent_count=plan.parent_count,
            pixel_chunk_size=selected.pixel_chunk_size,
        ),
        "explicit_kernel_builds": int(builds_after - builds_before),
        "peak_rss_bytes": peak_rss_bytes(),
        "loss_vjp": loss_vjp,
        "loss_explicit": loss_exp,
        "loss_abs_diff": loss_delta,
        "gradient_rel_l2": rel_grad,
        "gradient_max_abs_diff": float(np.max(np.abs(delta))) if delta.size else 0.0,
        "vjp_s": vjp_s,
        "explicit_s": explicit_s,
        "rtol": rtol,
        "atol": atol,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def product_is_complete(directory: Path) -> bool:
    """A product is complete only when both summary and checkpoint exist."""

    return (directory / "summary.json").is_file() and (
        directory / "checkpoint.json"
    ).is_file()


def copy_phase6_products(
    live_output: Path,
    dest_output: Path,
    *,
    stages: Sequence[str] = ("commissioning", "commissioning-c4"),
    beams: Sequence[str] = PRODUCTION_STOKES_I_BEAMS,
    require_complete_source: bool = True,
) -> tuple[str, ...]:
    """Copy live products into dest without overwriting a complete dest beam.

    When ``require_complete_source`` is true, skip incomplete live trees so a
    half-written checkpoint cannot land beside a newer dest summary. When it is
    false, in-progress live files may fill an incomplete dest beam.
    """

    copied: list[str] = []
    for stage in stages:
        for beam in beams:
            source = live_output / stage / beam
            dest = dest_output / stage / beam
            if product_is_complete(dest):
                continue
            if require_complete_source and not product_is_complete(source):
                continue
            if not source.is_dir():
                continue
            dest.mkdir(parents=True, exist_ok=True)
            for item in source.iterdir():
                if item.name.endswith(".tmp") or not item.is_file():
                    continue
                shutil.copy2(item, dest / item.name)
            copied.append(f"{stage}/{beam}")
    return tuple(copied)


def staged_source_manifest(
    dest: Path, *, git_root: Path | None = None
) -> dict[str, Any]:
    """Identity of the staged executable used for resume, compare, and baseline."""

    dest = Path(dest).resolve()
    protocol = dest / "src" / "sl1mjax" / "phase6_protocol.py"
    operator = dest / "src" / "sl1mjax" / "voltage_operator_jax.py"
    lock = dest / "uv.lock"
    payload: dict[str, Any] = {
        "tree": "explicit_staged",
        "dest": str(dest),
        "lock_sha256": sha256_file(lock) if lock.is_file() else None,
        "phase6_protocol_sha256": sha256_file(protocol) if protocol.is_file() else None,
        "voltage_operator_sha256": sha256_file(operator) if operator.is_file() else None,
    }
    if git_root is not None:
        try:
            payload["commit"] = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=git_root, text=True
            ).strip()
            diff = subprocess.check_output(
                ["git", "diff", "HEAD"], cwd=git_root, text=True
            )
            payload["uncommitted_diff_sha256"] = hashlib.sha256(diff.encode()).hexdigest()
        except (OSError, subprocess.CalledProcessError):
            payload["commit"] = "unknown"
    return payload


def write_staged_source_manifest(
    dest: Path,
    output: Path,
    *,
    git_root: Path | None = None,
) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(staged_source_manifest(dest, git_root=git_root), indent=2) + "\n",
        encoding="utf-8",
    )
    return output


def preserve_commissioning_source(live_source: Path, dest_output: Path) -> Path | None:
    """Keep the live commissioning manifest beside dest, never as dest source.json."""

    dest = Path(dest_output) / "commissioning_source.json"
    if dest.is_file():
        return dest
    if not live_source.is_file():
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(live_source, dest)
    return dest


RANKING_HOLD_OUT_THRESHOLD = 5.0e-6
_VLA_DISH_DIAMETER_M = 25.0
_CBAND_REFERENCE_HZ = 4.536e9
_SPEED_OF_LIGHT_M_S = 299_792_458.0


def load_under_resolved_findings(product: Path) -> tuple[dict[str, Any], ...]:
    path = Path(product) / "audit_findings.json"
    if not path.is_file():
        raise FileNotFoundError(f"{product} needs audit_findings.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    findings = payload.get("under_resolved", [])
    if not isinstance(findings, list):
        raise ValueError("audit_findings.json under_resolved must be a list")
    return tuple(item for item in findings if isinstance(item, dict))


def flux_weighted_audit_error(
    table: SkyComponentTable,
    findings: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    """Flux-weighted planner audit error for the under-resolved parents."""

    by_id = {component.component_id: component for component in table.components}
    weights: list[float] = []
    errors: list[float] = []
    for item in findings:
        component = by_id.get(str(item["component_id"]))
        flux = abs(component.stokes_i_jy) if component is not None else 0.0
        weights.append(float(flux))
        errors.append(float(item["error"]))
    weight = np.asarray(weights, dtype=np.float64)
    error = np.asarray(errors, dtype=np.float64)
    total = float(np.sum(weight))
    return {
        "n_under_resolved": float(len(findings)),
        "flux_weighted_error": float(np.sum(weight * error) / total) if total > 0 else float("nan"),
        "unweighted_mean_error": float(np.mean(error)) if error.size else float("nan"),
        "total_abs_flux_jy": total,
        "max_error": float(np.max(error)) if error.size else float("nan"),
    }


def raised_under_resolved_depths(
    depths: Mapping[str, int],
    under_ids: Sequence[str],
    extra_depth: int,
) -> dict[str, int]:
    """Keep the frozen topology; raise only the named parents to ``extra_depth``."""

    selected = set(under_ids)
    return {
        str(component_id): (
            int(extra_depth) if str(component_id) in selected else int(depth)
        )
        for component_id, depth in depths.items()
    }


def choose_seven_point_integration_depth(
    depth3_holdout: float,
    deeper_holdout: Mapping[int, float],
    *,
    threshold: float = RANKING_HOLD_OUT_THRESHOLD,
) -> int:
    """Raise planner ``max_depth`` when deeper quadrature moves holdout by ``threshold``."""

    selected = 3
    for depth in sorted(int(key) for key in deeper_holdout):
        if abs(float(deeper_holdout[depth]) - float(depth3_holdout)) >= threshold:
            selected = depth
    return selected


def airy_first_null_rad(
    frequency_hz: float = _CBAND_REFERENCE_HZ,
    dish_diameter_m: float = _VLA_DISH_DIAMETER_M,
) -> float:
    return 1.22 * _SPEED_OF_LIGHT_M_S / float(frequency_hz) / float(dish_diameter_m)


def locate_under_resolved_parents(
    table: SkyComponentTable,
    findings: Sequence[Mapping[str, Any]],
    *,
    frequency_hz: float = _CBAND_REFERENCE_HZ,
) -> dict[str, Any]:
    """Place under-resolved parents on nulls, raster edges, and bright sky."""

    by_id = {component.component_id: component for component in table.components}
    active = [component for component in table.components if component.active]
    flux = np.array([abs(component.stokes_i_jy) for component in active], dtype=np.float64)
    bright_cut = float(np.quantile(flux, 0.99)) if flux.size else 0.0
    bright = [
        component
        for component in active
        if abs(component.stokes_i_jy) >= bright_cut and bright_cut > 0.0
    ]
    first_null = airy_first_null_rad(frequency_hz)
    counts = {
        "beam_null": 0,
        "raster_boundary": 0,
        "near_bright": 0,
        "other": 0,
    }
    flux_by = {key: 0.0 for key in counts}
    for item in findings:
        component = by_id.get(str(item["component_id"]))
        if component is None:
            counts["other"] += 1
            continue
        tags = _location_tags(component, bright, first_null=first_null)
        if not tags:
            tags = ("other",)
        for tag in tags:
            counts[tag] += 1
            flux_by[tag] += abs(component.stokes_i_jy)
    n = max(len(findings), 1)
    return {
        "n_under_resolved": len(findings),
        "airy_first_null_rad": first_null,
        "bright_flux_cut_jy": bright_cut,
        "counts": counts,
        "flux_jy": flux_by,
        "fractions": {key: value / n for key, value in counts.items()},
    }


def _location_tags(
    component: SkyComponent,
    bright: Sequence[SkyComponent],
    *,
    first_null: float,
) -> tuple[str, ...]:
    tags: list[str] = []
    radius = float(np.hypot(component.l_rad, component.m_rad))
    width = float(component.width_rad)
    for ring in (1.0, 2.0):
        centre = ring * first_null
        if abs(radius - centre) <= max(0.35 * first_null, 0.5 * width):
            tags.append("beam_null")
            break
    if _is_raster_boundary_component(component):
        tags.append("raster_boundary")
    for neighbour in bright:
        if neighbour.component_id == component.component_id:
            tags.append("near_bright")
            break
        separation = float(
            np.hypot(component.l_rad - neighbour.l_rad, component.m_rad - neighbour.m_rad)
        )
        if separation <= 2.0 * max(width, float(neighbour.width_rad), first_null * 0.15):
            tags.append("near_bright")
            break
    return tuple(tags)


def _is_raster_boundary_component(component: SkyComponent) -> bool:
    if component.iy is None or component.ix is None:
        return False
    level = 0 if component.level is None else int(component.level)
    if component.family is ComponentFamily.OUTER_GUARD:
        leaf = QuadtreeLeaf(level, int(component.iy), int(component.ix))
        return is_boundary_guard_leaf(leaf) or is_outer_edge_guard_leaf(leaf)
    last = CENTRAL_ROOT_SIZE - 1
    return int(component.iy) in {0, last} or int(component.ix) in {0, last}


PHASE6_LADDER_STAGES = (
    "commissioning",
    "commissioning-c4",
    "baseline",
    "full_round1",
)
_STAGE_REQUIRES_GUARD = {
    "commissioning": True,
    "commissioning-c4": False,
    "baseline": False,
    "full": True,
    "full_round1": True,
}


@dataclass(frozen=True)
class Phase6ProductReview:
    """One stage/beam product checked against the Phase 6 protocol gates."""

    stage: str
    beam_mode: str
    present: bool
    passed: bool
    failures: tuple[str, ...]
    metrics: Mapping[str, Any]


def review_phase6_product(
    directory: Path,
    *,
    stage: str,
    beam_mode: str,
    require_guard: bool | None = None,
) -> Phase6ProductReview:
    """Read a written product directory and report protocol-gate failures."""

    guard_required = (
        _STAGE_REQUIRES_GUARD.get(stage, True) if require_guard is None else require_guard
    )
    summary_path = directory / "summary.json"
    metrics: dict[str, Any] = {"directory": str(directory)}
    if not summary_path.is_file():
        return Phase6ProductReview(
            stage=stage,
            beam_mode=beam_mode,
            present=False,
            passed=False,
            failures=("missing summary.json",),
            metrics=metrics,
        )
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    recorded_beam = payload.get("beam_mode")
    metrics.update(
        {
            "train_loss": payload.get("train_loss"),
            "holdout_loss": payload.get("holdout_loss"),
            "kkt_residual": payload.get("kkt_residual"),
            "audit_under_resolved": payload.get("audit_under_resolved"),
            "sealed_fold_unused": payload.get("sealed_fold_unused"),
            "stop_reason": payload.get("stop_reason"),
        }
    )
    if recorded_beam != beam_mode:
        failures.append(f"beam_mode {recorded_beam!r} != {beam_mode!r}")
    if recorded_beam == "full_jones" or beam_mode == "full_jones":
        failures.append("full Jones is not a production Stokes-I candidate")
    if recorded_beam not in PRODUCTION_STOKES_I_BEAMS:
        failures.append("beam_mode is not a production Stokes-I candidate")
    if not _finite_number(payload.get("train_loss")):
        failures.append("train_loss is missing or not finite")
    if not _finite_number(payload.get("holdout_loss")):
        failures.append("holdout_loss is missing or not finite")
    if not _finite_number(payload.get("kkt_residual")):
        failures.append("KKT residual is missing or not finite")
    if "audit_under_resolved" not in payload:
        failures.append("audit gate is missing")
    if payload.get("sealed_fold_unused") is not True:
        failures.append("sealed fold 4 was used or unrecorded")
    manifest = payload.get("manifest") or {}
    if manifest.get("sealed_fold") not in {None, SEALED_FOLD}:
        failures.append(f"sealed_fold is {manifest.get('sealed_fold')!r}, expected {SEALED_FOLD}")
    train_folds = tuple(manifest.get("train_folds") or ())
    if SEALED_FOLD in train_folds or manifest.get("holdout_fold") == SEALED_FOLD:
        failures.append("sealed fold 4 appears in train or holdout")
    config = payload.get("config") or {}
    sky_depth = config.get("sky_max_depth")
    integration_depth = config.get("integration_max_depth")
    metrics["sky_max_depth"] = sky_depth
    metrics["integration_max_depth"] = integration_depth
    if integration_depth is not None and int(integration_depth) < 3:
        failures.append("integration max_depth was reduced below 3")
    if (
        sky_depth is not None
        and integration_depth is not None
        and int(sky_depth) >= int(integration_depth)
    ):
        failures.append("sky max_depth reduced the planner integration cap")
    guard = payload.get("guard")
    metrics["guard"] = guard
    if guard_required and not guard:
        failures.append("guard report is missing")
    if isinstance(guard, Mapping):
        if guard.get("field_expansion") or guard.get("status") == FIELD_EXPANSION:
            failures.append("field-expansion abort")
    required_files = ("checkpoint.json", "component_table.json", "integration_plan.json")
    missing = tuple(name for name in required_files if not (directory / name).is_file())
    if missing:
        failures.append(f"missing product files: {', '.join(missing)}")
    return Phase6ProductReview(
        stage=stage,
        beam_mode=beam_mode,
        present=True,
        passed=not failures,
        failures=tuple(failures),
        metrics=metrics,
    )


def review_phase6_output(
    output: Path,
    *,
    stages: Sequence[str] = PHASE6_LADDER_STAGES,
    beams: Sequence[str] = PRODUCTION_STOKES_I_BEAMS,
) -> tuple[Phase6ProductReview, ...]:
    """Review every required stage/beam product under an output root."""

    reviews = []
    for stage in stages:
        for beam in beams:
            reviews.append(
                review_phase6_product(Path(output) / stage / beam, stage=stage, beam_mode=beam)
            )
    return tuple(reviews)


def phase6_output_complete(reviews: Sequence[Phase6ProductReview]) -> bool:
    return bool(reviews) and all(item.passed for item in reviews)


def phase6_ladder_complete(reviews: Sequence[Phase6ProductReview]) -> bool:
    """True when C1/C4/baseline passed and the selected topology pair passed.

    ``full_round1`` is required only for Airy plus the best detailed C1 beam.
    The unselected detailed beam may be absent.
    """

    if not phase6_commissioning_complete(reviews):
        return False
    baseline = [item for item in reviews if item.stage == "baseline"]
    if not baseline or not all(item.passed for item in baseline):
        return False
    by_key = {(item.stage, item.beam_mode): item for item in reviews}
    try:
        topology = select_topology_round_beams(reviews)
    except ValueError:
        return False
    return all(
        (item := by_key.get(("full_round1", beam))) is not None and item.passed
        for beam in topology
    )


def phase6_commissioning_complete(reviews: Sequence[Phase6ProductReview]) -> bool:
    """True when every C1 and C4 production-beam product passed."""

    required = [
        item
        for item in reviews
        if item.stage in {"commissioning", "commissioning-c4"}
    ]
    return bool(required) and all(item.passed for item in required)


def select_topology_round_beams(
    reviews: Sequence[Phase6ProductReview],
) -> tuple[str, ...]:
    """Airy control plus the best detailed C1 beam by holdout loss."""

    c1 = {
        item.beam_mode: item
        for item in reviews
        if item.stage == "commissioning" and item.passed
    }
    if "static_scalar" not in c1:
        raise ValueError("C1 static_scalar must pass before a topology round")
    detailed: list[tuple[float, str]] = []
    for name in ("streamed_scalar", "diagonal_copolar"):
        item = c1.get(name)
        if item is None:
            continue
        holdout = item.metrics.get("holdout_loss")
        if holdout is None:
            continue
        try:
            number = float(holdout)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            detailed.append((number, name))
    if not detailed:
        return ("static_scalar",)
    return ("static_scalar", min(detailed)[1])


EXPLICIT_COMPARE_GEOMETRY = {
    "static_scalar": {
        "batch_size_rows": 64,
        "pixel_chunk_size": 128,
        "visibility_chunk_size": 64,
    },
    "streamed_scalar": {
        "batch_size_rows": 64,
        "pixel_chunk_size": 128,
        "visibility_chunk_size": 64,
    },
    "diagonal_copolar": {
        "batch_size_rows": 64,
        "pixel_chunk_size": 512,
        "visibility_chunk_size": 32,
    },
}
EXPLICIT_COMPARE_BEAM_MODEL_IDS = {
    "static_scalar": AnalyticAiryVoltageBeam.model_id,
    "streamed_scalar": CompositeScalarVoltageBeam.model_id,
    "diagonal_copolar": CASSBEAM_CBAND_MODEL_ID,
}
_COMPARE_REQUIRED_FIELDS = (
    "passed",
    "beam_mode",
    "operator_mode",
    "product",
    "plan_sha256",
    "source_hashes",
    "batch_size_rows",
    "pixel_chunk_size",
    "visibility_chunk_size",
    "beam_model_id",
)


def _compare_report_payloads(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return []
    reports = payload.get("reports")
    if isinstance(reports, list):
        return [item for item in reports if isinstance(item, dict)]
    return [payload]


def validate_explicit_compare_report(
    payload: Mapping[str, Any],
    *,
    beam_mode: str,
    product: Path,
    geometry: Mapping[str, int] | None = None,
) -> tuple[str, ...]:
    """Return failures that keep one compare report from authorizing a beam."""

    selected = geometry or EXPLICIT_COMPARE_GEOMETRY.get(beam_mode, {})
    failures: list[str] = []
    missing = [name for name in _COMPARE_REQUIRED_FIELDS if name not in payload]
    if missing:
        failures.append(f"missing compare fields: {', '.join(missing)}")
        return tuple(failures)
    if payload.get("passed") is not True:
        failures.append("compare did not pass")
    if payload.get("beam_mode") != beam_mode:
        failures.append(
            f"beam_mode {payload.get('beam_mode')!r} != {beam_mode!r}"
        )
    if payload.get("operator_mode") != "explicit_jax":
        failures.append("operator_mode must be explicit_jax")
    recorded = payload.get("product")
    if recorded in {None, ""}:
        failures.append("product path is missing")
    else:
        recorded_path = Path(str(recorded))
        recorded_tail = (
            recorded_path.parts[-2:]
            if len(recorded_path.parts) >= 2
            else (recorded_path.name,)
        )
        expected_tail = (
            product.parts[-2:] if len(product.parts) >= 2 else (product.name,)
        )
        if (
            recorded_path.resolve() != product.resolve()
            and recorded_tail != expected_tail
        ):
            failures.append("product path does not match the commissioned beam")
    hashes = payload.get("source_hashes")
    if not isinstance(hashes, Mapping) or not hashes:
        failures.append("source hashes are missing")
    elif "checkpoint.json" not in hashes:
        failures.append("checkpoint hash is missing")
    else:
        checkpoint = product / "checkpoint.json"
        if not checkpoint.is_file():
            failures.append("commissioned checkpoint is missing")
        elif hashes.get("checkpoint.json") != sha256_file(checkpoint):
            failures.append("checkpoint hash does not match the commissioned product")
        for name in ("integration_plan.json", "summary.json"):
            recorded_hash = hashes.get(name)
            if recorded_hash is None:
                continue
            path = product / name
            if not path.is_file():
                failures.append(f"commissioned {name} is missing")
            elif recorded_hash != sha256_file(path):
                failures.append(f"{name} hash does not match the commissioned product")
    plan_hash = payload.get("plan_sha256")
    if not isinstance(plan_hash, str) or len(plan_hash) != 64:
        failures.append("plan fingerprint is missing")
    elif (product / "checkpoint.json").is_file() and (
        product / "integration_plan.json"
    ).is_file():
        try:
            _table, _flux, plan = sky_and_plan_from_product(product)
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            plan = None
        if plan is not None and plan_hash != plan_fingerprint(plan):
            failures.append("plan fingerprint does not match the commissioned product")
    if payload.get("batch_size_rows") != selected.get("batch_size_rows"):
        failures.append(
            f"batch_size_rows {payload.get('batch_size_rows')!r} "
            f"!= {selected.get('batch_size_rows')}"
        )
    if payload.get("pixel_chunk_size") != selected.get("pixel_chunk_size"):
        failures.append(
            f"pixel_chunk_size {payload.get('pixel_chunk_size')!r} "
            f"!= {selected.get('pixel_chunk_size')}"
        )
    if payload.get("visibility_chunk_size") != selected.get("visibility_chunk_size"):
        failures.append(
            f"visibility_chunk_size {payload.get('visibility_chunk_size')!r} "
            f"!= {selected.get('visibility_chunk_size')}"
        )
    expected_model = EXPLICIT_COMPARE_BEAM_MODEL_IDS.get(beam_mode)
    recorded_model = payload.get("beam_model_id")
    if expected_model is None:
        failures.append(f"no sealed beam_model_id for {beam_mode}")
    elif recorded_model != expected_model:
        failures.append(
            f"beam_model_id {recorded_model!r} != {expected_model!r}"
        )
    return tuple(failures)


def commissioning_ready(
    output: Path,
    *,
    compare_dir: Path | None = None,
    require_compare: bool = False,
    required_beams: Sequence[str] = PRODUCTION_STOKES_I_BEAMS,
) -> dict[str, Any]:
    """C1/C4 review gate, per-beam compare seal, and topology beam pair."""

    reviews = review_phase6_output(
        output, stages=("commissioning", "commissioning-c4"), beams=required_beams
    )
    by_beam: dict[str, dict[str, Any]] = {}
    if compare_dir is not None:
        for path in sorted(Path(compare_dir).glob("operator_compare_*.json")):
            for payload in _compare_report_payloads(path):
                beam = str(payload.get("beam_mode") or "")
                if beam:
                    by_beam[beam] = {"path": str(path), **payload}
    compare_reports: list[dict[str, Any]] = []
    compare_failures: list[str] = []
    if require_compare and compare_dir is None:
        compare_failures.append("compare directory is required")
    elif compare_dir is not None:
        for beam in required_beams:
            product = Path(output) / "commissioning" / beam
            selected = by_beam.get(beam)
            if selected is None:
                compare_failures.append(f"missing compare report for {beam}")
                continue
            failures = validate_explicit_compare_report(
                selected, beam_mode=beam, product=product
            )
            compare_reports.append(
                {
                    "beam_mode": beam,
                    "path": selected.get("path"),
                    "passed": not failures,
                    "failures": list(failures),
                }
            )
            compare_failures.extend(f"{beam}: {item}" for item in failures)
    commissioning = phase6_commissioning_complete(reviews)
    compare_ok = not compare_failures
    ready = commissioning and (compare_ok if require_compare else True)
    return {
        "ready": ready,
        "commissioning_complete": commissioning,
        "compare_passed": (
            False
            if require_compare and compare_dir is None
            else None if compare_dir is None
            else compare_ok
        ),
        "compare_failures": tuple(compare_failures),
        "topology_beams": select_topology_round_beams(reviews) if commissioning else (),
        "reviews": reviews,
        "compare_reports": compare_reports,
    }


def _finite_number(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False
