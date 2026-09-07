"""Reference-treatment, hold-out, and physical checks for diagonal holography.

These reports do not freeze a beam. They do not spatially interpolate or
form an array-average product. Frequency transfer is refused until a
frequency model is supplied.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sl1mjax.beam_conventions import evla195_total_squint_rad
from sl1mjax.holography import (
    AntennaPointingRole,
    HolographyHoldoutAxis,
    HolographyObservation,
    holography_comparison_beams,
    holography_holdout_split,
    moving_reference_row_mask,
)
from sl1mjax.holography_calibration import (
    C147_OFFSET_FIELD_IDS,
    FIRST_BEAM_RECOVERY_UNFROZEN_NOTE,
)
from sl1mjax.holography_diagonal import (
    COMBINED_REFERENCE_ID,
    FIRST_BEAM_RECOVERY_DIAGNOSTIC_NOTE,
    FIRST_RECOVERY_FREQUENCIES_HZ,
    NOMINAL_WEIGHT_NOTE,
    PHASE_GAUGE_REMOVAL_NOTE,
    HolographyDiagonalArtifact,
    HolographyDiagonalSample,
    _artifact_from_samples,
    _complex_rms_scatter,
    _role_grid,
    artifact_from_samples,
    interpolate_holography_diagonal_batch,
    predict_diagonal_from_samples,
    recover_holography_diagonal_per_reference,
    sample_records,
)
from sl1mjax.polarization import Correlation
from sl1mjax.voltage_beam import VoltageBeamModel, beam_coordinates

ON_AXIS_RADIUS_ARCMIN = 0.15
NEAR_NULL_RELATIVE_POWER = 0.05
SQUINT_MAINLOBE_POWER_FRACTION = 0.20
DENSE_SPACING_ARCMIN = 1.72
EXACT_ZERO_OFFSET_ARCMIN = 1.0 / 60.0
PASS_OVERLAP_SPACING_FRACTION = 0.15
CIRCULAR_COHERENCY_NOTE = (
    "Unpolarized Stokes I packs as RR=LL=I, not I/2. "
    "circular_stokes_to_coherency implements CASA RR=I+V."
)
HOLDOUT_LIMITATION_NOTE = (
    "Empirical interpolation uses training samples only. Null-region "
    "scores stay separate because fractional errors are unstable there. "
    "A training-derived complex gauge is applied to Airy, Perley and "
    "CASSBEAM on the same split. Frequency transfer is refused until a "
    "frequency model exists."
)
SPW4_DIAGONAL_STATUS_NOTE = (
    "SPW-4 diagonal recovery is physically credible and unfrozen. The "
    "flux gauge is closed. On-axis voltages near unity show that "
    "calibration, source coherency and inversion share one gauge. "
    "Memo-scale squint is supported. Sampled CASSBEAM squint is a stable "
    "~20% discrepancy, not noise alone. Frequency transfer is untested. "
    "Full Jones stays blocked."
)
NEXT_RECOVERY_ORDER = (
    "consistent_3c147_flux_gauge",
    "scientific_calibration_products",
    "reference_reference_flux_scale",
    "scientific_casa_jax_golden",
    "holoraster_apply_from_data",
    "absolute_and_relative_spw4_recovery",
    "reference_and_repeatability_holdouts",
    "crosshand_floor_coherence",
    "field9_all_antenna_residual_jones",
    "residual_jones_p_convention",
    "field9_channel_support",
    "field9_scan_state",
    "on_axis_beam_identity",
    "field9_time_smooth_residual_jones",
    "scan53_56_chain_jump",
    "field9_estimator_identifiability",
    "field9_stabilized_residual_jones",
    "prediction_equivalent_beam_maps",
    "one_axis_visibility_holdouts",
    "forward_closure",
    "reference_visit_aligned_copolar_transfer",
    "loro_full_versus_diagonal",
    "loro_leakage_sensitivity",
    "highres_cassbeam_direct_visibility_validation",
    "spw4_multichannel_beam_prior",
    "c147_offset_ring_highres_cassbeam",
    "holoraster_cassbeam_comparison_report",
    "cassbeam_diagonal_cband_reference",
    "cassbeam_diagonal_low_order_correction",
    "field9_acquisition_state",
    "hierarchical_scan_state_residual",
    "interleaved_onaxis_beam_transfer",
    "three_c147_qu_nuisance",
    "connected_residual_holdouts",
    "identity_prior_antenna_holdout",
    "residual_jones_holography_recovery",
    "training_only_offdiag_mask",
    "clustered_moving_reference_comparison",
    "independent_spw5_recovery",
    "introduce_and_validate_spatial_representation",
    "predict_unused_c147_offset_ring",
)


@dataclass(frozen=True)
class CopolarHandResidual:
    """Amplitude and phase residuals for one circular hand."""

    n: int
    amplitude_median: float
    amplitude_rms: float
    phase_rms_rad: float
    complex_relative_l2: float


@dataclass(frozen=True)
class CopolarResidualReport:
    rr: CopolarHandResidual
    ll: CopolarHandResidual
    name: str
    notes: tuple[str, ...] = ()


def reference_treatment_report(
    artifact: HolographyDiagonalArtifact,
) -> dict[str, Any]:
    """Scatter, leave-one-out, and per-reference dependence before combining."""

    per_reference = [
        sample
        for sample in artifact.samples
        if sample.valid and sample.reference_antenna_id != COMBINED_REFERENCE_ID
    ]
    groups: dict[tuple[int, int, int], list[HolographyDiagonalSample]] = {}
    for sample in per_reference:
        key = (
            sample.moving_antenna_id,
            int(round(sample.unique_time_s * 1.0e6)),
            int(round(sample.frequency_hz)),
        )
        groups.setdefault(key, []).append(sample)
    cell_reports = []
    flagged_refs: dict[int, list[str]] = {}
    for members in groups.values():
        if len(members) < 2:
            continue
        scatter_r = _complex_rms_scatter([sample.e_r for sample in members if sample.valid_r])
        scatter_l = _complex_rms_scatter([sample.e_l for sample in members if sample.valid_l])
        amp_r = [abs(sample.e_r) for sample in members if sample.valid_r]
        amp_l = [abs(sample.e_l) for sample in members if sample.valid_l]
        phase_r = [float(np.angle(sample.e_r)) for sample in members if sample.valid_r]
        phase_l = [float(np.angle(sample.e_l)) for sample in members if sample.valid_l]
        cell_reports.append(
            {
                "moving_antenna_id": members[0].moving_antenna_id,
                "unique_time_s": members[0].unique_time_s,
                "frequency_hz": members[0].frequency_hz,
                "n_reference": len(members),
                "reference_antenna_ids": [sample.reference_antenna_id for sample in members],
                "scatter_r": scatter_r,
                "scatter_l": scatter_l,
                "amp_scatter_r": _rms_about_median(amp_r),
                "amp_scatter_l": _rms_about_median(amp_l),
                "phase_scatter_r_rad": _circular_phase_rms(phase_r),
                "phase_scatter_l_rad": _circular_phase_rms(phase_l),
            }
        )
        mean_r = (
            np.mean([sample.e_r for sample in members if sample.valid_r])
            if any(sample.valid_r for sample in members)
            else np.nan
        )
        for sample in members:
            if sample.valid_r and np.isfinite(mean_r):
                residual = abs(sample.e_r - mean_r)
                if residual > 5.0 * max(scatter_r, 1.0e-15):
                    flagged_refs.setdefault(sample.reference_antenna_id, []).append(
                        "outlier_versus_reference_mean"
                    )
    time_channel = _reference_time_channel_dependence(per_reference)
    for antenna, reasons in time_channel.items():
        flagged_refs.setdefault(antenna, []).extend(reasons)
    n_cells = len(cell_reports)
    median_scatter_r = (
        float(np.nanmedian([item["scatter_r"] for item in cell_reports]))
        if cell_reports
        else float("nan")
    )
    median_scatter_l = (
        float(np.nanmedian([item["scatter_l"] for item in cell_reports]))
        if cell_reports
        else float("nan")
    )
    return {
        "checks_complete": True,
        "n_per_reference_samples": len(per_reference),
        "n_antenna_time_groups": len(groups),
        "n_antenna_time_groups_with_multiple_references": n_cells,
        "n_cells_with_multiple_references": n_cells,
        "median_reference_scatter_r": median_scatter_r,
        "median_reference_scatter_l": median_scatter_l,
        "cells": cell_reports,
        "flagged_references": {
            str(antenna): sorted(set(reasons)) for antenna, reasons in flagged_refs.items()
        },
        "time_channel_dependence": {
            str(antenna): reasons for antenna, reasons in time_channel.items()
        },
        "notes": (
            FIRST_BEAM_RECOVERY_DIAGNOSTIC_NOTE,
            NOMINAL_WEIGHT_NOTE,
            "References are not assumed interchangeable",
        ),
    }


def leave_one_reference_out_report(
    observation: HolographyObservation,
    visibility: ArrayLike | None = None,
    *,
    mask_unsettled: bool = True,
) -> CopolarResidualReport:
    """Recover without one reference and predict that reference's visibilities."""

    measured = (
        np.asarray(observation.block.visibility, dtype=np.complex128)
        if visibility is None
        else np.asarray(visibility, dtype=np.complex128)
    )
    full = recover_holography_diagonal_per_reference(
        observation, measured, mask_unsettled=mask_unsettled
    )
    references = sorted(
        {
            sample.reference_antenna_id
            for sample in full.samples
            if sample.valid and sample.reference_antenna_id != COMBINED_REFERENCE_ID
        }
    )
    if len(references) < 2:
        raise ValueError("leave-one-reference-out needs at least two reference antennas")
    predicted = np.zeros_like(measured)
    holdout_mask = np.zeros(observation.block.time_s.shape[0], dtype=bool)
    offsets, inverse, pointing_valid, _settled, _moving = observation.pointing_state()
    roles = _role_grid(observation, offsets.shape[:2])
    moving_ref = moving_reference_row_mask(observation.block, observation.pointing)
    for held in references:
        kept = tuple(
            sample
            for sample in full.samples
            if sample.valid and sample.reference_antenna_id != held
        )
        combined = _proxy_samples_without_reference(kept)
        rows = _rows_for_reference(observation, inverse, roles, pointing_valid, held)
        holdout_mask |= rows & moving_ref
        predicted += predict_diagonal_from_samples(
            observation, combined, row_mask=rows & moving_ref
        )
    return copolar_residuals(
        observation,
        measured,
        predicted,
        row_mask=holdout_mask,
        name="leave_one_reference_out",
        notes=("each held-out reference is predicted from the other references",),
    )


def reference_reference_residual_report(
    observation: HolographyObservation,
    visibility: ArrayLike | None = None,
) -> CopolarResidualReport:
    """On-axis reference--reference closures at the same times."""

    measured = (
        np.asarray(observation.block.visibility, dtype=np.complex128)
        if visibility is None
        else np.asarray(visibility, dtype=np.complex128)
    )
    names = tuple(
        item.value if isinstance(item, Correlation) else str(item)
        for item in observation.block.correlations
    )
    rr_index = names.index("RR")
    ll_index = names.index("LL")
    source = observation.source_coherency_visibility
    if source is None and observation.source_model is not None:
        source = observation.source_model.evaluate_coherency(observation.block)
    if source is None:
        from sl1mjax.polarization import circular_stokes_to_coherency

        plane = circular_stokes_to_coherency(observation.stokes_i, 0.0, 0.0, 0.0)
        source = np.broadcast_to(
            plane,
            (observation.block.time_s.shape[0], observation.block.frequency_hz.size, 2, 2),
        ).copy()
    offsets, inverse, pointing_valid, settled, _moving = observation.pointing_state()
    roles = _role_grid(observation, offsets.shape[:2])
    predicted = np.zeros_like(measured)
    mask = np.zeros(observation.block.time_s.shape[0], dtype=bool)
    for row in range(observation.block.time_s.shape[0]):
        time_index = int(inverse[row])
        antenna_p = int(observation.block.antenna1[row])
        antenna_q = int(observation.block.antenna2[row])
        if not (
            pointing_valid[time_index, antenna_p]
            and pointing_valid[time_index, antenna_q]
            and settled[time_index, antenna_p]
            and settled[time_index, antenna_q]
        ):
            continue
        if not (
            roles[time_index, antenna_p] == AntennaPointingRole.REFERENCE.value
            and roles[time_index, antenna_q] == AntennaPointingRole.REFERENCE.value
        ):
            continue
        mask[row] = True
        for channel in range(observation.block.frequency_hz.size):
            predicted[row, channel, rr_index] = source[row, channel, 0, 0]
            predicted[row, channel, ll_index] = source[row, channel, 1, 1]
    return copolar_residuals(
        observation,
        measured,
        predicted,
        row_mask=mask,
        name="reference_reference",
        notes=("identity-gauge reference--reference residual at shared times",),
    )


def combine_after_reference_checks(
    artifact: HolographyDiagonalArtifact,
    report: Mapping[str, Any],
    *,
    observation: HolographyObservation | None = None,
) -> HolographyDiagonalArtifact:
    """Form a robust reference-combined estimate only after the checks."""

    if artifact.frozen:
        raise ValueError("empirical holography diagonal is not frozen")
    if not bool(report.get("checks_complete")):
        raise ValueError("combine only after reference-treatment checks")
    flagged = set(int(antenna) for antenna in report.get("flagged_references", {}))
    groups: dict[tuple[int, int, int], list[HolographyDiagonalSample]] = {}
    for sample in artifact.samples:
        if (
            not sample.valid
            or sample.reference_antenna_id == COMBINED_REFERENCE_ID
            or sample.reference_antenna_id in flagged
        ):
            continue
        key = (
            sample.moving_antenna_id,
            int(round(sample.unique_time_s * 1.0e6)),
            int(round(sample.frequency_hz)),
        )
        groups.setdefault(key, []).append(sample)
    if not groups:
        raise ValueError("no unflagged per-reference samples remain to combine")
    samples: list[HolographyDiagonalSample] = []
    for members in groups.values():
        e_r = [sample.e_r for sample in members if sample.valid_r]
        e_l = [sample.e_l for sample in members if sample.valid_l]
        scatter_r = _complex_rms_scatter(e_r)
        scatter_l = _complex_rms_scatter(e_l)
        first = members[0]
        samples.append(
            HolographyDiagonalSample(
                moving_antenna_id=first.moving_antenna_id,
                unique_time_s=first.unique_time_s,
                offset_lm_rad=first.offset_lm_rad,
                frequency_hz=first.frequency_hz,
                e_r=complex(np.mean(e_r)) if e_r else np.nan + 1j * np.nan,
                e_l=complex(np.mean(e_l)) if e_l else np.nan + 1j * np.nan,
                sigma_r=scatter_r,
                sigma_l=scatter_l,
                n_reference=len(members),
                weight=float("nan"),
                valid=bool(e_r or e_l),
                raster=first.raster,
                reference_antenna_id=COMBINED_REFERENCE_ID,
                scan=first.scan,
                raster_pass=first.raster_pass,
                offset_azelgeo_rad=first.offset_azelgeo_rad,
                flag_rr=all(sample.flag_rr for sample in members),
                flag_ll=all(sample.flag_ll for sample in members),
                weight_rr=float(np.nansum([sample.weight_rr for sample in members])),
                weight_ll=float(np.nansum([sample.weight_ll for sample in members])),
                n_baseline=len(members),
                n_sample=int(np.sum([sample.n_sample for sample in members])),
                valid_r=bool(e_r),
                valid_l=bool(e_l),
                reference_scatter_r=scatter_r,
                reference_scatter_l=scatter_l,
                thermal_scatter_r=_finite_median([sample.thermal_scatter_r for sample in members]),
                thermal_scatter_l=_finite_median([sample.thermal_scatter_l for sample in members]),
                calibration_state=first.calibration_state,
                source_name=first.source_name,
            )
        )
    if observation is None:
        jones = np.stack(
            [
                np.diag(
                    [
                        sample.e_r if sample.valid_r else 0.0,
                        sample.e_l if sample.valid_l else 0.0,
                    ]
                )
                for sample in samples
            ]
        )
        return HolographyDiagonalArtifact(
            samples=tuple(samples),
            jones=jones,
            valid=np.asarray([sample.valid for sample in samples], dtype=bool),
            off_diagonal_valid=np.zeros(len(samples), dtype=bool),
            calibration_state=artifact.calibration_state,
            source_name=artifact.source_name,
            reference_combination="robust_after_reference_checks",
            frozen=False,
            notes=artifact.notes
            + (
                "combined after reference-treatment checks",
                NOMINAL_WEIGHT_NOTE,
                FIRST_BEAM_RECOVERY_UNFROZEN_NOTE,
            ),
        )
    return _artifact_from_samples(
        samples,
        observation,
        reference_combination="robust_after_reference_checks",
        notes=artifact.notes
        + (
            "combined after reference-treatment checks",
            NOMINAL_WEIGHT_NOTE,
        ),
    )


def holdout_prediction_report(
    observation: HolographyObservation,
    visibility: ArrayLike | None = None,
    *,
    axis: HolographyHoldoutAxis | str,
    holdout_antenna_id: int | None = None,
    frequency_model: object | None = None,
    mask_unsettled: bool = True,
) -> CopolarResidualReport:
    """Predict visibilities that were not used in the recovered estimate."""

    selected = HolographyHoldoutAxis(axis)
    if selected is HolographyHoldoutAxis.FREQUENCY and frequency_model is None:
        raise ValueError(
            "recover at 4.564 GHz and predict 4.692 GHz only after a frequency model is introduced"
        )
    measured = (
        np.asarray(observation.block.visibility, dtype=np.complex128)
        if visibility is None
        else np.asarray(visibility, dtype=np.complex128)
    )
    split = holography_holdout_split(observation, selected, holdout_antenna_id=holdout_antenna_id)
    train = recover_holography_diagonal_per_reference(
        _masked_observation(observation, split.train_row_mask),
        measured,
        mask_unsettled=mask_unsettled,
    )
    predicted = predict_diagonal_from_samples(
        observation,
        train.samples,
        row_mask=split.holdout_row_mask,
        frequency_model=frequency_model,
        allow_other_moving_antenna=selected is HolographyHoldoutAxis.MOVING_ANTENNA,
        allow_other_reference=selected is HolographyHoldoutAxis.REFERENCE_ANTENNA,
    )
    notes = list(split.notes)
    if selected is HolographyHoldoutAxis.SPATIAL:
        notes.append(
            "held-out raster cells are predicted only when another measured "
            "sample shares the same cell; no spatial interpolation"
        )
    if selected is HolographyHoldoutAxis.MOVING_ANTENNA:
        notes.append(
            "a held-out moving antenna is predicted only if another mover "
            "recovered the same cell; antenna variation is not averaged away"
        )
    return copolar_residuals(
        observation,
        measured,
        predicted,
        row_mask=split.holdout_row_mask,
        name=split.name,
        notes=tuple(notes),
    )


def repeated_visit_holdout_report(
    observation: HolographyObservation,
    visibility: ArrayLike | None = None,
    *,
    mask_unsettled: bool = True,
) -> CopolarResidualReport:
    """Recover the first visit to each cell and predict later visits."""

    measured = (
        np.asarray(observation.block.visibility, dtype=np.complex128)
        if visibility is None
        else np.asarray(visibility, dtype=np.complex128)
    )
    full = recover_holography_diagonal_per_reference(
        observation, measured, mask_unsettled=mask_unsettled
    )
    scale = 1.0 / 2.908882086657216e-05
    first_times: dict[tuple[int, int, int, int], float] = {}
    for sample in full.samples:
        if not sample.valid:
            continue
        key = (
            sample.moving_antenna_id,
            sample.reference_antenna_id,
            int(round(float(sample.offset_lm_rad[0]) * scale)),
            int(round(float(sample.offset_lm_rad[1]) * scale)),
            int(round(sample.frequency_hz)),
        )
        first_times[key] = (
            sample.unique_time_s
            if key not in first_times
            else min(first_times[key], sample.unique_time_s)
        )
    train = tuple(
        sample
        for sample in full.samples
        if sample.valid
        and np.isclose(
            sample.unique_time_s,
            first_times[
                (
                    sample.moving_antenna_id,
                    sample.reference_antenna_id,
                    int(round(float(sample.offset_lm_rad[0]) * scale)),
                    int(round(float(sample.offset_lm_rad[1]) * scale)),
                    int(round(sample.frequency_hz)),
                )
            ],
            rtol=0.0,
            atol=1.0e-6,
        )
    )
    train_keys = {
        (
            sample.moving_antenna_id,
            sample.reference_antenna_id,
            int(round(sample.unique_time_s * 1.0e6)),
            int(round(sample.frequency_hz)),
        )
        for sample in train
    }
    later = {
        (
            sample.moving_antenna_id,
            sample.reference_antenna_id,
            int(round(sample.unique_time_s * 1.0e6)),
        )
        for sample in full.samples
        if sample.valid
        and (
            sample.moving_antenna_id,
            sample.reference_antenna_id,
            int(round(sample.unique_time_s * 1.0e6)),
            int(round(sample.frequency_hz)),
        )
        not in train_keys
    }
    offsets, inverse, pointing_valid, _settled, _moving = observation.pointing_state()
    roles = _role_grid(observation, offsets.shape[:2])
    holdout = np.zeros(observation.block.time_s.shape[0], dtype=bool)
    for row in range(holdout.size):
        time_index = int(inverse[row])
        antenna_p = int(observation.block.antenna1[row])
        antenna_q = int(observation.block.antenna2[row])
        if not (pointing_valid[time_index, antenna_p] and pointing_valid[time_index, antenna_q]):
            continue
        try:
            moving, reference = _moving_and_reference_roles(antenna_p, antenna_q, roles[time_index])
        except ValueError:
            continue
        time_s = float(observation.pointing.unique_time_s[time_index])
        holdout[row] = (moving, reference, int(round(time_s * 1.0e6))) in later
    predicted = predict_diagonal_from_samples(observation, train, row_mask=holdout)
    return copolar_residuals(
        observation,
        measured,
        predicted,
        row_mask=holdout,
        name="repeated_cell_visits",
        notes=("first visit trains; later visits to the same cell are predicted",),
    )


def predict_c147_offset_fields(
    observation: HolographyObservation,
    raster_artifact: HolographyDiagonalArtifact,
    visibility: ArrayLike | None = None,
) -> CopolarResidualReport:
    """Predict the eight C147-* fields without fitting them."""

    measured = (
        np.asarray(observation.block.visibility, dtype=np.complex128)
        if visibility is None
        else np.asarray(visibility, dtype=np.complex128)
    )
    field_id = np.asarray(observation.block.field_id, dtype=np.int32)
    mask = np.isin(field_id, np.asarray(C147_OFFSET_FIELD_IDS, dtype=np.int32))
    predicted = predict_diagonal_from_samples(observation, raster_artifact.samples, row_mask=mask)
    return copolar_residuals(
        observation,
        measured,
        predicted,
        row_mask=mask,
        name="c147_offset_fields",
        notes=(
            "C147-* fields are not fitted",
            "predicted only at offsets that already have a measured raster cell",
        ),
    )


def compare_models_at_measured_cells(
    artifact: HolographyDiagonalArtifact,
    *,
    beams: Mapping[str, VoltageBeamModel] | None = None,
    calibration_state: str | None = None,
) -> dict[str, Any]:
    """Compare Airy, Perley, CASSBEAM, and empirical voltages at measured cells."""

    models = dict(beams or holography_comparison_beams())
    models.pop("cassbeam_experimental_full_jones", None)
    state = calibration_state or artifact.calibration_state
    usable = unique_measured_cells([sample for sample in artifact.samples if sample.valid])
    scores = {}
    for name, beam in models.items():
        rr = []
        ll = []
        for sample in usable:
            evaluation = beam.evaluate(
                beam_coordinates(
                    np.array([0.0]),
                    np.array([0.0]),
                    np.array([sample.frequency_hz]),
                    parallactic_angle_rad=np.array([0.0]),
                    pointing_offset_lm_rad=sample.offset_lm_rad,
                    antenna_id=np.array([sample.moving_antenna_id], dtype=np.int32),
                ),
                calibration_state=state,
            )
            jones = np.asarray(evaluation.jones).reshape(-1, 2, 2)[0]
            if sample.valid_r:
                rr.append((sample.e_r, complex(jones[0, 0])))
            if sample.valid_l:
                ll.append((sample.e_l, complex(jones[1, 1])))
        scores[name] = {
            "rr": _paired_hand_residual(rr),
            "ll": _paired_hand_residual(ll),
        }
    empirical = {
        "rr": _paired_hand_residual(
            [(sample.e_r, sample.e_r) for sample in usable if sample.valid_r]
        ),
        "ll": _paired_hand_residual(
            [(sample.e_l, sample.e_l) for sample in usable if sample.valid_l]
        ),
    }
    scores["empirical_diagonal_holography"] = empirical
    return {
        "n_cells": len(usable),
        "models": scores,
        "notes": (
            FIRST_BEAM_RECOVERY_DIAGNOSTIC_NOTE,
            "comparison is at unique measured cells only; no interpolation",
            "one representative sample per moving antenna, offset, and frequency",
            "RR and LL residuals are separate; no aggregate MSE",
        ),
    }


def physical_diagonal_diagnostics(
    artifact: HolographyDiagonalArtifact,
) -> dict[str, Any]:
    """On-axis, squint, 1/ν, repeatability, and pass-overlap checks."""

    usable = unique_measured_cells([sample for sample in artifact.samples if sample.valid])
    frequencies = sorted({sample.frequency_hz for sample in usable})
    by_frequency = {
        frequency: [sample for sample in usable if np.isclose(sample.frequency_hz, frequency)]
        for frequency in frequencies
    }
    squint = {}
    for frequency, members in by_frequency.items():
        centroid_r = _centroid(members, "R", power_fraction=SQUINT_MAINLOBE_POWER_FRACTION)
        centroid_l = _centroid(members, "L", power_fraction=SQUINT_MAINLOBE_POWER_FRACTION)
        unmasked_r = _centroid(members, "R", power_fraction=0.0)
        unmasked_l = _centroid(members, "L", power_fraction=0.0)
        measured = float(np.hypot(*(centroid_r - centroid_l)))
        unmasked = float(np.hypot(*(unmasked_r - unmasked_l)))
        memo = float(evla195_total_squint_rad(frequency))
        on_axis = [
            sample
            for sample in members
            if _radius_arcmin(sample.offset_lm_rad) <= ON_AXIS_RADIUS_ARCMIN
        ]
        squint[f"{frequency:.0f}"] = {
            "frequency_hz": frequency,
            "centroid_r_lm_rad": centroid_r.tolist(),
            "centroid_l_lm_rad": centroid_l.tolist(),
            "total_squint_rad": measured,
            "unmasked_total_squint_rad": unmasked,
            "mainlobe_power_fraction": SQUINT_MAINLOBE_POWER_FRACTION,
            "memo_2p4_over_nu_rad": memo,
            "squint_ratio_to_memo": measured / memo if memo > 0.0 else float("nan"),
            "unmasked_squint_ratio_to_memo": unmasked / memo if memo > 0.0 else float("nan"),
            "on_axis": _on_axis_summary(on_axis),
        }
    scaling = {}
    native = [
        frequency
        for frequency in frequencies
        if any(np.isclose(frequency, native) for native in FIRST_RECOVERY_FREQUENCIES_HZ)
    ]
    if len(native) >= 2:
        low, high = sorted(native)[:2]
        low_key = f"{low:.0f}"
        high_key = f"{high:.0f}"
        s_low = squint[low_key]["total_squint_rad"]
        s_high = squint[high_key]["total_squint_rad"]
        scaling = {
            "low_frequency_hz": low,
            "high_frequency_hz": high,
            "measured_ratio": s_low / s_high if s_high > 0.0 else float("nan"),
            "expected_nu_inverse_ratio": high / low,
        }
    all_valid = [sample for sample in artifact.samples if sample.valid]
    return {
        "group_counts": recovery_group_counts(artifact),
        "on_axis_sample_audit": on_axis_sample_audit(all_valid),
        "on_axis_and_squint": squint,
        "frequency_scaling": scaling,
        "repeatability": _moving_antenna_repeatability(usable),
        "pass_nearest_neighbour": pass_nearest_neighbour_report(all_valid),
        "dense_sparse_overlap": _dense_sparse_overlap(usable),
        "phase_smoothness": phase_smoothness_report(usable),
        "notes": (
            FIRST_BEAM_RECOVERY_DIAGNOSTIC_NOTE,
            CIRCULAR_COHERENCY_NOTE,
            HOLDOUT_LIMITATION_NOTE,
            "antenna variation is reported, not averaged away",
            "array-average beam comes later",
            "do not snap pass-2 onto the Memo lattice",
        ),
    }


def recovery_group_counts(artifact: HolographyDiagonalArtifact) -> dict[str, Any]:
    """Name the different groupings that were previously all called cells."""

    valid = [sample for sample in artifact.samples if sample.valid]
    antenna_time = {
        (
            sample.moving_antenna_id,
            int(round(sample.unique_time_s * 1.0e6)),
            int(round(sample.frequency_hz)),
        )
        for sample in valid
    }
    multi_ref = 0
    members: dict[tuple[int, int, int], set[int]] = {}
    for sample in valid:
        key = (
            sample.moving_antenna_id,
            int(round(sample.unique_time_s * 1.0e6)),
            int(round(sample.frequency_hz)),
        )
        members.setdefault(key, set()).add(sample.reference_antenna_id)
    multi_ref = sum(1 for refs in members.values() if len(refs) > 1)
    moving_spatial = unique_measured_cells(valid)
    shared = _shared_spatial_coordinates(valid)
    return {
        "n_per_reference_samples": len(valid),
        "n_antenna_time_groups": len(antenna_time),
        "n_antenna_time_groups_with_multiple_references": multi_ref,
        "n_moving_antenna_spatial_cells": len(moving_spatial),
        "n_shared_spatial_coordinates": shared,
        "notes": (
            "antenna_time_group: one moving antenna at one unique time and frequency",
            "moving_antenna_spatial_cell: one moving antenna at one commanded offset and frequency",
            "shared_spatial_coordinate: one commanded offset/frequency seen by at least "
            "two moving antennas",
        ),
    }


def on_axis_sample_audit(
    samples: list[HolographyDiagonalSample],
    *,
    radius_arcmin: float = ON_AXIS_RADIUS_ARCMIN,
) -> dict[str, Any]:
    """Report whether selected on-axis samples have genuinely zero AZELGEO offset."""

    radii = np.asarray(
        [_radius_arcmin(sample.offset_azelgeo_rad) for sample in samples], dtype=np.float64
    )
    selected = [
        sample for sample, radius in zip(samples, radii, strict=True) if radius <= radius_arcmin
    ]
    selected_radii = np.asarray(
        [_radius_arcmin(sample.offset_azelgeo_rad) for sample in selected], dtype=np.float64
    )
    exact = int(np.sum(selected_radii <= EXACT_ZERO_OFFSET_ARCMIN)) if selected_radii.size else 0
    by_antenna: dict[str, dict[str, float]] = {}
    for sample in selected:
        key = str(sample.moving_antenna_id)
        by_antenna.setdefault(key, {"n": 0, "abs_e_r": [], "abs_e_l": []})
        by_antenna[key]["n"] += 1
        if sample.valid_r:
            by_antenna[key]["abs_e_r"].append(abs(sample.e_r))
        if sample.valid_l:
            by_antenna[key]["abs_e_l"].append(abs(sample.e_l))
    per_antenna = {
        antenna: {
            "n": int(item["n"]),
            "median_abs_e_r": _finite_median(item["abs_e_r"]),
            "median_abs_e_l": _finite_median(item["abs_e_l"]),
            "rl_ratio": (
                _finite_median(item["abs_e_r"]) / _finite_median(item["abs_e_l"])
                if _finite_median(item["abs_e_l"])
                else float("nan")
            ),
        }
        for antenna, item in by_antenna.items()
    }
    return {
        "selection_radius_arcmin": radius_arcmin,
        "exact_zero_threshold_arcmin": EXACT_ZERO_OFFSET_ARCMIN,
        "n_selected": len(selected),
        "n_exact_zero_azelgeo": exact,
        "min_radius_arcmin": float(np.min(selected_radii)) if selected_radii.size else float("nan"),
        "median_radius_arcmin": float(np.median(selected_radii))
        if selected_radii.size
        else float("nan"),
        "max_radius_arcmin": float(np.max(selected_radii)) if selected_radii.size else float("nan"),
        "median_abs_e_r": _finite_median(
            [abs(sample.e_r) for sample in selected if sample.valid_r]
        ),
        "median_abs_e_l": _finite_median(
            [abs(sample.e_l) for sample in selected if sample.valid_l]
        ),
        "per_moving_antenna": per_antenna,
        "notes": (
            "These |E| values equal |V_mr| when S=1 and E_r=1",
            "A scalar source spectrum cannot remove an R/L amplitude difference",
            "Prefer per-antenna on-axis normalisation for beam shape",
        ),
    }


def pass_nearest_neighbour_report(
    samples: list[HolographyDiagonalSample],
) -> dict[str, Any]:
    """Nearest pass-2 neighbour of every pass-1 point. Does not snap either raster."""

    pass1 = _unique_spatial_offsets(
        [sample for sample in samples if sample.raster_pass == "pass-1"]
    )
    pass2 = _unique_spatial_offsets(
        [sample for sample in samples if sample.raster_pass == "pass-2"]
    )
    if pass1.size == 0 or pass2.size == 0:
        return {
            "n_pass1": int(pass1.shape[0]),
            "n_pass2": int(pass2.shape[0]),
            "tolerance_arcmin": float("nan"),
            "n_pairs_within_tolerance": 0,
            "notes": ("one or both raster passes are absent",),
        }
    distances = []
    for offset in pass1:
        distances.append(float(np.min(np.hypot(pass2[:, 0] - offset[0], pass2[:, 1] - offset[1]))))
    distance_arcmin = np.asarray(distances, dtype=np.float64) * 180.0 * 60.0 / np.pi
    dwell = _within_dwell_offset_scatter_arcmin(samples)
    inter = _inter_antenna_offset_scatter_arcmin(samples)
    spacing_frac = PASS_OVERLAP_SPACING_FRACTION * DENSE_SPACING_ARCMIN
    tolerance = float(
        np.nanmax([dwell.get("p95_arcmin", np.nan), inter.get("p95_arcmin", np.nan), spacing_frac])
    )
    if not np.isfinite(tolerance):
        tolerance = spacing_frac
    n_pairs = int(np.sum(distance_arcmin <= tolerance))
    return {
        "n_pass1": int(pass1.shape[0]),
        "n_pass2": int(pass2.shape[0]),
        "median_nearest_arcmin": float(np.median(distance_arcmin)),
        "min_nearest_arcmin": float(np.min(distance_arcmin)),
        "p95_nearest_arcmin": float(np.percentile(distance_arcmin, 95)),
        "n_pairs_within_tolerance": n_pairs,
        "tolerance_arcmin": tolerance,
        "tolerance_derivation": {
            "within_dwell_p95_arcmin": dwell.get("p95_arcmin", float("nan")),
            "inter_antenna_p95_arcmin": inter.get("p95_arcmin", float("nan")),
            "fraction_of_dense_spacing_arcmin": spacing_frac,
            "dense_spacing_arcmin": DENSE_SPACING_ARCMIN,
            "spacing_fraction": PASS_OVERLAP_SPACING_FRACTION,
        },
        "pass1_nearest_zero_arcmin": _nearest_to_origin_arcmin(pass1),
        "pass2_nearest_zero_arcmin": _nearest_to_origin_arcmin(pass2),
        "notes": (
            "pass-2 coordinates are unchanged",
            "pairs identify measurements close enough to test repeatability",
            "this is not exact lattice overlap and does not snap either raster",
        ),
    }


def phase_smoothness_report(samples: list[HolographyDiagonalSample]) -> dict[str, Any]:
    """Neighbour phase jumps after amplitude masks and a per-antenna/reference gauge."""

    thresholds = (0.50, 0.20, 0.10, 0.05)
    by_stratum = {
        "by_moving_antenna": _phase_by_key(samples, lambda sample: sample.moving_antenna_id),
        "by_reference_antenna": _phase_by_key(samples, lambda sample: sample.reference_antenna_id),
        "by_raster_pass": _phase_by_key(samples, lambda sample: sample.raster_pass or "unknown"),
        "by_hand": {
            "R": _phase_jumps(samples, power_fraction=0.20, hand="R", remove_gauge=True),
            "L": _phase_jumps(samples, power_fraction=0.20, hand="L", remove_gauge=True),
        },
    }
    overall = {
        f"above_{int(100 * fraction)}_percent_peak": _phase_jumps(samples, power_fraction=fraction)
        for fraction in thresholds
    }
    gauged = {
        f"above_{int(100 * fraction)}_percent_peak": _phase_jumps(
            samples, power_fraction=fraction, remove_gauge=True
        )
        for fraction in thresholds
    }
    per_antenna = [
        item["median_neighbor_phase_jump_rad"]
        for item in by_stratum["by_moving_antenna"].values()
        if np.isfinite(item.get("median_neighbor_phase_jump_rad", float("nan")))
    ]
    return {
        "mixed_antennas_raw": overall,
        "mixed_antennas_one_global_gauge": gauged,
        "per_antenna_gauge_median_rad": _finite_median(per_antenna),
        "strata": by_stratum,
        "versus_radius_arcmin": _phase_versus_radius(samples),
        "versus_time": _phase_versus_time(samples),
        "notes": (
            "mixed_antennas_* mixes antennas and is a gauge, not main-lobe roughness",
            "per_antenna_gauge_median_rad is the smoothness metric after one phase per antenna",
            "Jumps isolated near nulls should not control interpolation",
            PHASE_GAUGE_REMOVAL_NOTE,
        ),
        "phase_gauge_removed": True,
    }


def calibrate_squint_estimator(
    artifact: HolographyDiagonalArtifact,
    *,
    beam: VoltageBeamModel | None = None,
    calibration_state: str | None = None,
    power_fraction: float = SQUINT_MAINLOBE_POWER_FRACTION,
) -> dict[str, Any]:
    """Run the empirical squint estimator on CASSBEAM at the measured coordinates."""

    from sl1mjax.cassbeam_beam import cassbeam_receptor_mainlobe_separation_arcmin

    usable = unique_measured_cells([sample for sample in artifact.samples if sample.valid])
    if not usable:
        raise ValueError("squint calibration needs valid measured cells")
    frequency = float(usable[0].frequency_hz)
    sampled = _sampled_cassbeam_cells(
        usable,
        beam=beam,
        calibration_state=calibration_state or artifact.calibration_state,
    )
    empirical = _centroid_separation_arcmin(usable, power_fraction=power_fraction)
    sampled_sep = _centroid_separation_arcmin(sampled, power_fraction=power_fraction)
    known = float(cassbeam_receptor_mainlobe_separation_arcmin(frequency))
    memo = float(evla195_total_squint_rad(frequency) * 180.0 * 60.0 / np.pi)
    ratio = empirical / sampled_sep if sampled_sep else float("nan")
    return {
        "frequency_hz": frequency,
        "n_moving_antenna_spatial_cells": len(usable),
        "mainlobe_power_fraction": power_fraction,
        "empirical_separation_arcmin": empirical,
        "sampled_cassbeam_separation_arcmin": sampled_sep,
        "known_cassbeam_mainlobe_separation_arcmin": known,
        "memo_2p4_over_nu_arcmin": memo,
        "sampled_over_known_cassbeam": sampled_sep / known if known else float("nan"),
        "empirical_over_memo": empirical / memo if memo else float("nan"),
        "empirical_over_sampled_cassbeam": ratio,
        "notes": (
            "Same coordinates, missing samples, flags, grouping, and |E|^2 centroid",
            "sampled/known near 0.78-0.85 is estimator bias; near 1 means the data "
            "have smaller squint",
            "Bootstrap the empirical/CASSBEAM ratio over movers and references",
        ),
    }


def bootstrap_empirical_over_cassbeam_squint(
    artifact: HolographyDiagonalArtifact,
    *,
    beam: VoltageBeamModel | None = None,
    calibration_state: str | None = None,
    power_fraction: float = SQUINT_MAINLOBE_POWER_FRACTION,
    n_resample: int = 400,
    seed: int = 0,
) -> dict[str, Any]:
    """Bootstrap empirical/sampled-CASSBEAM squint over movers and references.

    If the uncertainty interval includes one, treat the measurements as
    consistent with CASSBEAM after sampling bias. A stable deficit may be
    a real array-average antenna difference.
    """

    calibration = calibrate_squint_estimator(
        artifact,
        beam=beam,
        calibration_state=calibration_state,
        power_fraction=power_fraction,
    )
    usable = unique_measured_cells([sample for sample in artifact.samples if sample.valid])
    sampled = _sampled_cassbeam_cells(
        usable,
        beam=beam,
        calibration_state=calibration_state or artifact.calibration_state,
    )
    sampled_by_mover = {_mover_cell_key(sample): sample for sample in sampled}
    movers = sorted({sample.moving_antenna_id for sample in usable})
    references = sorted(
        {
            sample.reference_antenna_id
            for sample in artifact.samples
            if sample.valid and sample.reference_antenna_id >= 0
        }
    )
    rng = np.random.default_rng(int(seed))
    mover_ratios = _bootstrap_ratio(
        usable,
        sampled_by_mover,
        group_ids=movers,
        group_of=lambda sample: sample.moving_antenna_id,
        n_resample=n_resample,
        rng=rng,
        power_fraction=power_fraction,
    )
    reference_ratios: list[float] = []
    if references:
        per_ref = [sample for sample in artifact.samples if sample.valid]
        sampled_ref = _sampled_cassbeam_cells(
            unique_measured_cells(per_ref, include_reference=True),
            beam=beam,
            calibration_state=calibration_state or artifact.calibration_state,
        )
        sampled_by_ref = {_reference_cell_key(sample): sample for sample in sampled_ref}
        reference_ratios = _bootstrap_ratio(
            unique_measured_cells(per_ref, include_reference=True),
            sampled_by_ref,
            group_ids=references,
            group_of=lambda sample: sample.reference_antenna_id,
            n_resample=n_resample,
            rng=rng,
            power_fraction=power_fraction,
            key_fn=_reference_cell_key,
        )
    mover_interval = _percentile_interval(mover_ratios)
    reference_interval = _percentile_interval(reference_ratios)
    point = float(calibration["empirical_over_sampled_cassbeam"])
    return {
        **calibration,
        "n_resample": n_resample,
        "seed": int(seed),
        "n_moving_antennas": len(movers),
        "n_reference_antennas": len(references),
        "moving_antenna_ratio": mover_interval,
        "reference_antenna_ratio": reference_interval,
        "includes_one": bool(
            mover_interval["p16"] <= 1.0 <= mover_interval["p84"]
            or (
                np.isfinite(reference_interval["p16"])
                and reference_interval["p16"] <= 1.0 <= reference_interval["p84"]
            )
        ),
        "consistent_with_cassbeam": bool(mover_interval["p16"] <= 1.0 <= mover_interval["p84"]),
        "stable_deficit": bool(np.isfinite(point) and point < 1.0 and mover_interval["p84"] < 1.0),
        "notes": tuple(calibration["notes"])
        + (
            "If the mover bootstrap interval includes 1, treat empirical and CASSBEAM "
            "as consistent",
            "A stable deficit after sampling correction may be a real array-average difference",
        ),
    }


def unique_measured_cells(
    samples: list[HolographyDiagonalSample],
    *,
    offset_atol_rad: float = 2.908882086657216e-05,
    include_reference: bool = False,
) -> list[HolographyDiagonalSample]:
    """One sample per moving antenna, commanded offset, and frequency."""

    scale = 1.0 / float(offset_atol_rad)
    chosen: dict[tuple[int, ...], HolographyDiagonalSample] = {}
    for sample in samples:
        key = (
            sample.moving_antenna_id,
            int(round(float(sample.offset_lm_rad[0]) * scale)),
            int(round(float(sample.offset_lm_rad[1]) * scale)),
            int(round(sample.frequency_hz)),
        )
        if include_reference:
            key = (*key, int(sample.reference_antenna_id))
        if key not in chosen:
            chosen[key] = sample
    return [chosen[key] for key in sorted(chosen)]


def _sampled_cassbeam_cells(
    samples: list[HolographyDiagonalSample],
    *,
    beam: VoltageBeamModel | None,
    calibration_state: str,
) -> list[HolographyDiagonalSample]:
    model = beam or holography_comparison_beams()["cassbeam_diagonal"]
    sampled = []
    for sample in samples:
        evaluation = model.evaluate(
            beam_coordinates(
                np.array([0.0]),
                np.array([0.0]),
                np.array([sample.frequency_hz]),
                parallactic_angle_rad=np.array([0.0]),
                pointing_offset_lm_rad=sample.offset_lm_rad,
                antenna_id=np.array([sample.moving_antenna_id], dtype=np.int32),
            ),
            calibration_state=calibration_state,
        )
        jones = np.asarray(evaluation.jones).reshape(-1, 2, 2)[0]
        valid = np.asarray(evaluation.valid).reshape(-1)[0]
        sampled.append(
            HolographyDiagonalSample(
                moving_antenna_id=sample.moving_antenna_id,
                unique_time_s=sample.unique_time_s,
                offset_lm_rad=sample.offset_lm_rad,
                frequency_hz=sample.frequency_hz,
                e_r=complex(jones[0, 0]) if (sample.valid_r and valid) else np.nan + 1j * np.nan,
                e_l=complex(jones[1, 1]) if (sample.valid_l and valid) else np.nan + 1j * np.nan,
                sigma_r=float("nan"),
                sigma_l=float("nan"),
                n_reference=sample.n_reference,
                weight=float("nan"),
                valid=bool(valid) and (sample.valid_r or sample.valid_l),
                raster=sample.raster,
                reference_antenna_id=sample.reference_antenna_id,
                valid_r=bool(valid and sample.valid_r),
                valid_l=bool(valid and sample.valid_l),
            )
        )
    return sampled


def _mover_cell_key(sample: HolographyDiagonalSample) -> tuple[int, int, int, int]:
    scale = 1.0 / 2.908882086657216e-05
    return (
        sample.moving_antenna_id,
        int(round(float(sample.offset_lm_rad[0]) * scale)),
        int(round(float(sample.offset_lm_rad[1]) * scale)),
        int(round(sample.frequency_hz)),
    )


def _reference_cell_key(sample: HolographyDiagonalSample) -> tuple[int, ...]:
    return (*_mover_cell_key(sample), int(sample.reference_antenna_id))


def _bootstrap_ratio(
    empirical: list[HolographyDiagonalSample],
    sampled_by_key: dict[tuple[int, ...], HolographyDiagonalSample],
    *,
    group_ids: list[int],
    group_of,
    n_resample: int,
    rng: np.random.Generator,
    power_fraction: float,
    key_fn=_mover_cell_key,
) -> list[float]:
    if len(group_ids) < 2:
        ratio = _ratio_for_cells(empirical, sampled_by_key, power_fraction, key_fn)
        return [ratio] if np.isfinite(ratio) else []
    ratios: list[float] = []
    ids = np.asarray(group_ids, dtype=np.int32)
    for _ in range(int(n_resample)):
        drawn = set(int(value) for value in rng.choice(ids, size=ids.size, replace=True))
        subset = [sample for sample in empirical if group_of(sample) in drawn]
        ratio = _ratio_for_cells(subset, sampled_by_key, power_fraction, key_fn)
        if np.isfinite(ratio):
            ratios.append(ratio)
    return ratios


def _ratio_for_cells(
    empirical: list[HolographyDiagonalSample],
    sampled_by_key: dict[tuple[int, ...], HolographyDiagonalSample],
    power_fraction: float,
    key_fn,
) -> float:
    usable = [sample for sample in empirical if sample.valid]
    sampled = [
        sampled_by_key[key_fn(sample)]
        for sample in usable
        if key_fn(sample) in sampled_by_key and sampled_by_key[key_fn(sample)].valid
    ]
    empirical_sep = _centroid_separation_arcmin(usable, power_fraction=power_fraction)
    sampled_sep = _centroid_separation_arcmin(sampled, power_fraction=power_fraction)
    if not np.isfinite(empirical_sep) or not np.isfinite(sampled_sep) or sampled_sep == 0.0:
        return float("nan")
    return float(empirical_sep / sampled_sep)


def _percentile_interval(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "n": 0.0,
            "median": float("nan"),
            "p16": float("nan"),
            "p84": float("nan"),
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": float(array.size),
        "median": float(np.median(array)),
        "p16": float(np.percentile(array, 16)),
        "p84": float(np.percentile(array, 84)),
    }


def first_recovery_artifact_dict(
    artifact: HolographyDiagonalArtifact,
    *,
    reference_report: Mapping[str, Any] | None = None,
    holdouts: Mapping[str, CopolarResidualReport] | None = None,
    model_comparison: Mapping[str, Any] | None = None,
    physical: Mapping[str, Any] | None = None,
    include_samples: bool = True,
) -> dict[str, Any]:
    """Serialise the unfrozen first-recovery diagnostic."""

    payload = {
        "schema": "first_beam_recovery_v1",
        "frozen": False,
        "first_beam_recovery_unfrozen": True,
        "reference_combination": artifact.reference_combination,
        "calibration_state": artifact.calibration_state,
        "source_name": artifact.source_name,
        "n_samples": len(artifact.samples),
        "n_valid": int(sum(1 for sample in artifact.samples if sample.valid)),
        "group_counts": recovery_group_counts(artifact),
        "next_recovery_order": list(NEXT_RECOVERY_ORDER),
        "notes": list(artifact.notes)
        + [
            FIRST_BEAM_RECOVERY_DIAGNOSTIC_NOTE,
            FIRST_BEAM_RECOVERY_UNFROZEN_NOTE,
            NOMINAL_WEIGHT_NOTE,
            PHASE_GAUGE_REMOVAL_NOTE,
            SPW4_DIAGONAL_STATUS_NOTE,
        ],
        "reference_treatment": _summarize_reference_report(reference_report),
        "holdouts": {name: _residual_dict(report) for name, report in (holdouts or {}).items()},
        "measured_cell_model_comparison": dict(model_comparison or {}),
        "physical_diagnostics": dict(physical or {}),
    }
    if include_samples:
        payload["samples"] = sample_records(artifact.samples)
    return payload


def copolar_residuals(
    observation: HolographyObservation,
    measured: ArrayLike,
    predicted: ArrayLike,
    *,
    row_mask: ArrayLike,
    name: str,
    notes: tuple[str, ...] = (),
) -> CopolarResidualReport:
    names = tuple(
        item.value if isinstance(item, Correlation) else str(item)
        for item in observation.block.correlations
    )
    measured_arr = np.asarray(measured, dtype=np.complex128)
    predicted_arr = np.asarray(predicted, dtype=np.complex128)
    rows = np.asarray(row_mask, dtype=bool)
    flags = np.asarray(observation.block.flag, dtype=bool)
    return CopolarResidualReport(
        rr=_hand_residual(measured_arr, predicted_arr, flags, rows, names.index("RR")),
        ll=_hand_residual(measured_arr, predicted_arr, flags, rows, names.index("LL")),
        name=name,
        notes=notes,
    )


def _hand_residual(
    measured: np.ndarray,
    predicted: np.ndarray,
    flag: np.ndarray,
    rows: np.ndarray,
    correlation: int,
) -> CopolarHandResidual:
    usable = (
        rows[:, None]
        & ~flag[:, :, correlation]
        & np.isfinite(predicted[:, :, correlation])
        & (np.abs(predicted[:, :, correlation]) > 0.0)
    )
    if not bool(np.any(usable)):
        return CopolarHandResidual(0, float("nan"), float("nan"), float("nan"), float("nan"))
    meas = measured[:, :, correlation][usable]
    pred = predicted[:, :, correlation][usable]
    ratio = meas / pred
    amp = np.abs(ratio)
    phase = np.angle(ratio)
    resid = meas - pred
    return CopolarHandResidual(
        n=int(meas.size),
        amplitude_median=float(np.median(amp)),
        amplitude_rms=float(np.sqrt(np.mean((amp - 1.0) ** 2))),
        phase_rms_rad=float(np.sqrt(np.mean(phase**2))),
        complex_relative_l2=float(np.linalg.norm(resid) / max(np.linalg.norm(meas), 1.0e-15)),
    )


def _paired_hand_residual(pairs: list[tuple[complex, complex]]) -> dict[str, float]:
    if not pairs:
        return {
            "n": 0,
            "amplitude_median": float("nan"),
            "amplitude_rms": float("nan"),
            "phase_rms_rad": float("nan"),
            "complex_relative_l2": float("nan"),
        }
    measured = np.asarray([item[0] for item in pairs], dtype=np.complex128)
    model = np.asarray([item[1] for item in pairs], dtype=np.complex128)
    usable = np.isfinite(measured) & np.isfinite(model) & (np.abs(model) > 0.0)
    if not bool(np.any(usable)):
        return _paired_hand_residual([])
    ratio = measured[usable] / model[usable]
    resid = measured[usable] - model[usable]
    return {
        "n": int(np.sum(usable)),
        "amplitude_median": float(np.median(np.abs(ratio))),
        "amplitude_rms": float(np.sqrt(np.mean((np.abs(ratio) - 1.0) ** 2))),
        "phase_rms_rad": float(np.sqrt(np.mean(np.angle(ratio) ** 2))),
        "complex_relative_l2": float(
            np.linalg.norm(resid) / max(np.linalg.norm(measured[usable]), 1.0e-15)
        ),
    }


def _summarize_reference_report(report: Mapping[str, Any] | None) -> dict[str, Any]:
    if not report:
        return {}
    summary = dict(report)
    cells = summary.pop("cells", [])
    summary["n_cell_reports"] = len(cells)
    return summary


def _residual_dict(report: CopolarResidualReport) -> dict[str, Any]:
    return {
        "name": report.name,
        "rr": report.rr.__dict__,
        "ll": report.ll.__dict__,
        "notes": list(report.notes),
    }


def _proxy_samples_without_reference(
    samples: tuple[HolographyDiagonalSample, ...],
) -> tuple[HolographyDiagonalSample, ...]:
    groups: dict[tuple[int, int, int], list[HolographyDiagonalSample]] = {}
    for sample in samples:
        key = (
            sample.moving_antenna_id,
            int(round(sample.unique_time_s * 1.0e6)),
            int(round(sample.frequency_hz)),
        )
        groups.setdefault(key, []).append(sample)
    proxies: list[HolographyDiagonalSample] = []
    for members in groups.values():
        mean_r = (
            complex(np.mean([sample.e_r for sample in members if sample.valid_r]))
            if any(sample.valid_r for sample in members)
            else np.nan + 1j * np.nan
        )
        mean_l = (
            complex(np.mean([sample.e_l for sample in members if sample.valid_l]))
            if any(sample.valid_l for sample in members)
            else np.nan + 1j * np.nan
        )
        first = members[0]
        for sample in members:
            proxies.append(
                HolographyDiagonalSample(
                    moving_antenna_id=sample.moving_antenna_id,
                    unique_time_s=sample.unique_time_s,
                    offset_lm_rad=sample.offset_lm_rad,
                    frequency_hz=sample.frequency_hz,
                    e_r=mean_r,
                    e_l=mean_l,
                    sigma_r=_complex_rms_scatter([item.e_r for item in members if item.valid_r]),
                    sigma_l=_complex_rms_scatter([item.e_l for item in members if item.valid_l]),
                    n_reference=len(members),
                    weight=float("nan"),
                    valid=np.isfinite(mean_r) or np.isfinite(mean_l),
                    raster=sample.raster,
                    reference_antenna_id=sample.reference_antenna_id,
                    scan=sample.scan,
                    raster_pass=sample.raster_pass,
                    valid_r=np.isfinite(mean_r),
                    valid_l=np.isfinite(mean_l),
                    calibration_state=first.calibration_state,
                    source_name=first.source_name,
                )
            )
    return tuple(proxies)


def _rows_for_reference(
    observation: HolographyObservation,
    inverse: np.ndarray,
    roles: np.ndarray,
    pointing_valid: np.ndarray,
    reference: int,
) -> NDArray[np.bool_]:
    mask = np.zeros(observation.block.time_s.shape[0], dtype=bool)
    for row in range(mask.size):
        time_index = int(inverse[row])
        antenna_p = int(observation.block.antenna1[row])
        antenna_q = int(observation.block.antenna2[row])
        if not (pointing_valid[time_index, antenna_p] and pointing_valid[time_index, antenna_q]):
            continue
        if roles[time_index, antenna_p] == AntennaPointingRole.REFERENCE.value:
            ref = antenna_p
        elif roles[time_index, antenna_q] == AntennaPointingRole.REFERENCE.value:
            ref = antenna_q
        else:
            continue
        mask[row] = ref == int(reference)
    return mask


def _masked_observation(
    observation: HolographyObservation, row_mask: ArrayLike
) -> HolographyObservation:
    from dataclasses import replace

    mask = np.asarray(row_mask, dtype=bool)
    flag = np.array(observation.block.flag, copy=True)
    flag[~mask] = True
    return HolographyObservation(
        block=replace(observation.block, flag=flag),
        pointing=observation.pointing,
        antenna_position_m=observation.antenna_position_m,
        calibration_state=observation.calibration_state,
        phase_centre_rad=observation.phase_centre_rad,
        source_name=observation.source_name,
        stokes_i=observation.stokes_i,
        source_coherency_visibility=observation.source_coherency_visibility,
        source_model=observation.source_model,
        selected_correlations=observation.selected_correlations,
        selected_frequency_hz=observation.selected_frequency_hz,
        selected_spw_id=observation.selected_spw_id,
        provenance=observation.provenance,
    )


def _reference_time_channel_dependence(
    samples: list[HolographyDiagonalSample],
) -> dict[int, list[str]]:
    flagged: dict[int, list[str]] = {}
    by_ref: dict[int, list[HolographyDiagonalSample]] = {}
    for sample in samples:
        if sample.valid_r:
            by_ref.setdefault(sample.reference_antenna_id, []).append(sample)
    for antenna, members in by_ref.items():
        if len(members) < 4:
            continue
        times = np.asarray([sample.unique_time_s for sample in members], dtype=np.float64)
        freqs = np.asarray([sample.frequency_hz for sample in members], dtype=np.float64)
        values = np.asarray([sample.e_r for sample in members], dtype=np.complex128)
        residual = np.abs(values - np.mean(values))
        if _correlated(times, residual):
            flagged.setdefault(antenna, []).append("residual_depends_on_time")
        if len(np.unique(np.round(freqs))) > 1 and _correlated(freqs, residual):
            flagged.setdefault(antenna, []).append("residual_depends_on_channel")
    return flagged


def _correlated(x: np.ndarray, y: np.ndarray) -> bool:
    if x.size < 4 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return False
    return abs(float(np.corrcoef(x, y)[0, 1])) > 0.8


def _centroid(
    samples: list[HolographyDiagonalSample],
    receptor: str,
    *,
    power_fraction: float = 0.0,
) -> NDArray[np.float64]:
    slot = "e_r" if receptor == "R" else "e_l"
    offsets = np.array([sample.offset_lm_rad for sample in samples], dtype=np.float64)
    power = np.array([abs(getattr(sample, slot)) ** 2 for sample in samples], dtype=np.float64)
    usable = np.isfinite(power) & (power > 0.0)
    if not bool(np.any(usable)):
        return np.array([np.nan, np.nan], dtype=np.float64)
    peak = float(np.max(power[usable]))
    if power_fraction > 0.0:
        usable = usable & (power >= power_fraction * peak)
    if not bool(np.any(usable)):
        return np.array([np.nan, np.nan], dtype=np.float64)
    weight = power[usable]
    return np.asarray(np.sum(offsets[usable] * weight[:, None], axis=0) / np.sum(weight))


def _centroid_separation_arcmin(
    samples: list[HolographyDiagonalSample],
    *,
    power_fraction: float,
) -> float:
    left = _centroid(samples, "R", power_fraction=power_fraction)
    right = _centroid(samples, "L", power_fraction=power_fraction)
    return float(np.hypot(*(left - right)) * 180.0 * 60.0 / np.pi)


def _shared_spatial_coordinates(samples: list[HolographyDiagonalSample]) -> int:
    scale = 1.0 / 2.908882086657216e-05
    groups: dict[tuple[int, int, int], set[int]] = {}
    for sample in samples:
        key = (
            int(round(float(sample.offset_lm_rad[0]) * scale)),
            int(round(float(sample.offset_lm_rad[1]) * scale)),
            int(round(sample.frequency_hz)),
        )
        groups.setdefault(key, set()).add(sample.moving_antenna_id)
    return sum(1 for antennas in groups.values() if len(antennas) > 1)


def _unique_spatial_offsets(samples: list[HolographyDiagonalSample]) -> np.ndarray:
    seen = {}
    scale = 1.0 / 2.908882086657216e-05
    for sample in samples:
        key = (
            int(round(float(sample.offset_lm_rad[0]) * scale)),
            int(round(float(sample.offset_lm_rad[1]) * scale)),
        )
        if key not in seen:
            seen[key] = sample.offset_lm_rad
    if not seen:
        return np.zeros((0, 2), dtype=np.float64)
    return np.array(list(seen.values()), dtype=np.float64)


def _nearest_to_origin_arcmin(offsets: np.ndarray) -> float:
    if offsets.size == 0:
        return float("nan")
    return float(np.min(np.hypot(offsets[:, 0], offsets[:, 1])) * 180.0 * 60.0 / np.pi)


def _within_dwell_offset_scatter_arcmin(
    samples: list[HolographyDiagonalSample],
) -> dict[str, float]:
    groups: dict[tuple[int, int], list[np.ndarray]] = {}
    for sample in samples:
        key = (sample.moving_antenna_id, int(round(sample.unique_time_s * 1.0e6)))
        groups.setdefault(key, []).append(np.asarray(sample.offset_azelgeo_rad, dtype=np.float64))
    rms = []
    for members in groups.values():
        if len(members) < 2:
            continue
        array = np.stack(members, axis=0)
        rms.append(float(np.sqrt(np.mean(np.sum((array - np.mean(array, axis=0)) ** 2, axis=1)))))
    arcmin = np.asarray(rms, dtype=np.float64) * 180.0 * 60.0 / np.pi if rms else np.array([])
    return {
        "n_dwells": len(arcmin),
        "p95_arcmin": float(np.percentile(arcmin, 95)) if arcmin.size else float("nan"),
    }


def _inter_antenna_offset_scatter_arcmin(
    samples: list[HolographyDiagonalSample],
) -> dict[str, float]:
    scale = 1.0 / 2.908882086657216e-05
    groups: dict[tuple[int, int], list[np.ndarray]] = {}
    for sample in samples:
        key = (
            int(round(float(sample.offset_lm_rad[0]) * scale)),
            int(round(float(sample.offset_lm_rad[1]) * scale)),
        )
        groups.setdefault(key, []).append(np.asarray(sample.offset_azelgeo_rad, dtype=np.float64))
    rms = []
    for members in groups.values():
        if len(members) < 2:
            continue
        array = np.stack(members, axis=0)
        rms.append(float(np.sqrt(np.mean(np.sum((array - np.mean(array, axis=0)) ** 2, axis=1)))))
    arcmin = np.asarray(rms, dtype=np.float64) * 180.0 * 60.0 / np.pi if rms else np.array([])
    return {
        "n_commands": len(arcmin),
        "p95_arcmin": float(np.percentile(arcmin, 95)) if arcmin.size else float("nan"),
    }


def _phase_by_key(samples: list[HolographyDiagonalSample], key_fn) -> dict[str, Any]:
    groups: dict[str, list[HolographyDiagonalSample]] = {}
    for sample in samples:
        groups.setdefault(str(key_fn(sample)), []).append(sample)
    return {
        name: _phase_jumps(members, power_fraction=0.20, remove_gauge=True)
        for name, members in groups.items()
    }


def _phase_jumps(
    samples: list[HolographyDiagonalSample],
    *,
    power_fraction: float,
    hand: str = "R",
    remove_gauge: bool = False,
) -> dict[str, float]:
    voltages = []
    offsets = []
    for sample in samples:
        value = sample.e_r if hand == "R" else sample.e_l
        valid = sample.valid_r if hand == "R" else sample.valid_l
        if not valid or not np.isfinite(value):
            continue
        voltages.append(complex(value))
        offsets.append(sample.offset_lm_rad)
    if len(voltages) < 3:
        return {"median_neighbor_phase_jump_rad": float("nan"), "n": 0}
    voltage = np.asarray(voltages, dtype=np.complex128)
    offset = np.asarray(offsets, dtype=np.float64)
    power = np.abs(voltage) ** 2
    peak = float(np.max(power))
    keep = power >= power_fraction * peak if peak > 0.0 else np.ones(power.size, dtype=bool)
    voltage = voltage[keep]
    offset = offset[keep]
    if voltage.size < 3:
        return {"median_neighbor_phase_jump_rad": float("nan"), "n": 0}
    if remove_gauge:
        voltage = voltage * np.exp(-1j * np.angle(np.mean(voltage)))
    phases = _unwrap_connected_phase(offset, np.angle(voltage))
    jumps = []
    for index, point in enumerate(offset):
        others = np.delete(np.arange(offset.shape[0]), index)
        distance = np.hypot(offset[others, 0] - point[0], offset[others, 1] - point[1])
        neighbor = others[int(np.argmin(distance))]
        jumps.append(abs(phases[index] - phases[neighbor]))
    return {
        "median_neighbor_phase_jump_rad": float(np.median(jumps)) if jumps else float("nan"),
        "n": len(jumps),
        "n_samples": int(voltage.size),
    }


def _unwrap_connected_phase(offsets: np.ndarray, phases: np.ndarray) -> np.ndarray:
    remaining = set(range(offsets.shape[0]))
    unwrapped = np.array(phases, copy=True)
    while remaining:
        start = next(iter(remaining))
        queue = [start]
        remaining.remove(start)
        while queue:
            index = queue.pop()
            distance = np.hypot(
                offsets[:, 0] - offsets[index, 0], offsets[:, 1] - offsets[index, 1]
            )
            order = np.argsort(distance)
            added = 0
            for other in order:
                if int(other) not in remaining:
                    continue
                unwrapped[other] = unwrapped[index] + np.angle(
                    np.exp(1j * (phases[other] - phases[index]))
                )
                remaining.remove(int(other))
                queue.append(int(other))
                added += 1
                if added >= 4:
                    break
    return unwrapped


def _phase_versus_radius(samples: list[HolographyDiagonalSample]) -> list[dict[str, float]]:
    rows = []
    edges = (2.0, 6.0, 12.0, 24.0, 60.0)
    radii = np.asarray(
        [_radius_arcmin(sample.offset_lm_rad) for sample in samples], dtype=np.float64
    )
    lower = 0.0
    for upper in edges:
        members = [
            sample for sample, radius in zip(samples, radii, strict=True) if lower <= radius < upper
        ]
        item = _phase_jumps(members, power_fraction=0.20, remove_gauge=True)
        rows.append({"radius_lo_arcmin": lower, "radius_hi_arcmin": upper, **item})
        lower = upper
    return rows


def _phase_versus_time(samples: list[HolographyDiagonalSample]) -> dict[str, float]:
    usable = [sample for sample in samples if sample.valid_r]
    if len(usable) < 4:
        return {"phase_time_abs_correlation": float("nan"), "n": len(usable)}
    times = np.asarray([sample.unique_time_s for sample in usable], dtype=np.float64)
    phases = np.unwrap([float(np.angle(sample.e_r)) for sample in usable])
    if float(np.std(times)) == 0.0 or float(np.std(phases)) == 0.0:
        corr = float("nan")
    else:
        corr = float(np.corrcoef(times, phases)[0, 1])
    return {"phase_time_abs_correlation": abs(corr), "n": len(usable)}


def _on_axis_summary(samples: list[HolographyDiagonalSample]) -> dict[str, float]:
    if not samples:
        return {
            "n": 0,
            "median_abs_e_r": float("nan"),
            "median_abs_e_l": float("nan"),
            "median_phase_r_rad": float("nan"),
            "median_phase_l_rad": float("nan"),
        }
    e_r = np.asarray([sample.e_r for sample in samples if sample.valid_r], dtype=np.complex128)
    e_l = np.asarray([sample.e_l for sample in samples if sample.valid_l], dtype=np.complex128)
    return {
        "n": len(samples),
        "median_abs_e_r": float(np.median(np.abs(e_r))) if e_r.size else float("nan"),
        "median_abs_e_l": float(np.median(np.abs(e_l))) if e_l.size else float("nan"),
        "median_phase_r_rad": float(np.median(np.angle(e_r))) if e_r.size else float("nan"),
        "median_phase_l_rad": float(np.median(np.angle(e_l))) if e_l.size else float("nan"),
    }


def _moving_antenna_repeatability(samples: list[HolographyDiagonalSample]) -> dict[str, Any]:
    groups: dict[tuple[int, int, int], list[HolographyDiagonalSample]] = {}
    scale = 1.0 / 2.908882086657216e-05
    for sample in samples:
        key = (
            int(round(float(sample.offset_lm_rad[0]) * scale)),
            int(round(float(sample.offset_lm_rad[1]) * scale)),
            int(round(sample.frequency_hz)),
        )
        groups.setdefault(key, []).append(sample)
    scatters = []
    for members in groups.values():
        antennas = {sample.moving_antenna_id for sample in members}
        if len(antennas) < 2:
            continue
        per_ant = []
        for antenna in antennas:
            values = [
                sample.e_r
                for sample in members
                if sample.moving_antenna_id == antenna and sample.valid_r
            ]
            if values:
                per_ant.append(complex(np.mean(values)))
        if len(per_ant) >= 2:
            scatters.append(_complex_rms_scatter(per_ant))
    return {
        "n_shared_cells": len(scatters),
        "median_moving_antenna_scatter_r": float(np.median(scatters)) if scatters else float("nan"),
    }


def _dense_sparse_overlap(samples: list[HolographyDiagonalSample]) -> dict[str, Any]:
    dense = [sample for sample in samples if sample.raster == "dense"]
    sparse = [sample for sample in samples if sample.raster == "sparse"]
    scale = 1.0 / 2.908882086657216e-05
    dense_keys = {
        (
            int(round(float(sample.offset_lm_rad[0]) * scale)),
            int(round(float(sample.offset_lm_rad[1]) * scale)),
            int(round(sample.frequency_hz)),
        )
        for sample in dense
    }
    overlap = [
        sample
        for sample in sparse
        if (
            int(round(float(sample.offset_lm_rad[0]) * scale)),
            int(round(float(sample.offset_lm_rad[1]) * scale)),
            int(round(sample.frequency_hz)),
        )
        in dense_keys
    ]
    return {"n_dense": len(dense), "n_sparse": len(sparse), "n_overlap_cells": len(overlap)}


def _phase_smoothness(samples: list[HolographyDiagonalSample]) -> dict[str, float]:
    if len(samples) < 3:
        return {"median_neighbor_phase_jump_rad": float("nan"), "n": 0}
    offsets = np.array([sample.offset_lm_rad for sample in samples], dtype=np.float64)
    phases = np.array(
        [np.angle(sample.e_r) if sample.valid_r else np.nan for sample in samples],
        dtype=np.float64,
    )
    power = np.array(
        [abs(sample.e_r) ** 2 if sample.valid_r else 0.0 for sample in samples],
        dtype=np.float64,
    )
    peak = float(np.max(power)) if power.size else 0.0
    jumps = []
    for index, offset in enumerate(offsets):
        if not np.isfinite(phases[index]) or (
            peak > 0.0 and power[index] < NEAR_NULL_RELATIVE_POWER * peak
        ):
            continue
        others = np.delete(np.arange(offsets.shape[0]), index)
        distance = np.hypot(offsets[others, 0] - offset[0], offsets[others, 1] - offset[1])
        neighbor = others[int(np.argmin(distance))]
        if not np.isfinite(phases[neighbor]):
            continue
        if peak > 0.0 and power[neighbor] < NEAR_NULL_RELATIVE_POWER * peak:
            continue
        jumps.append(abs(np.angle(np.exp(1j * (phases[index] - phases[neighbor])))))
    return {
        "median_neighbor_phase_jump_rad": float(np.median(jumps)) if jumps else float("nan"),
        "n": len(jumps),
    }


def _radius_arcmin(offset: np.ndarray) -> float:
    return float(np.hypot(offset[0], offset[1]) * 180.0 * 60.0 / np.pi)


def _finite_median(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return float("nan")
    return float(np.median(finite))


def _rms_about_median(values: list[float]) -> float:
    if len(values) < 2:
        return float("nan")
    array = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean((array - np.median(array)) ** 2)))


def _circular_phase_rms(values: list[float]) -> float:
    if len(values) < 2:
        return float("nan")
    phasors = np.exp(1j * np.asarray(values, dtype=np.float64))
    mean = np.mean(phasors)
    if abs(mean) <= 0.0:
        return float("nan")
    return float(np.sqrt(np.mean(np.angle(phasors * np.conjugate(mean)) ** 2)))


def _moving_and_reference_roles(
    antenna_p: int,
    antenna_q: int,
    roles: np.ndarray,
) -> tuple[int, int]:
    from sl1mjax.holography_diagonal import _moving_and_reference

    return _moving_and_reference(antenna_p, antenna_q, roles)


def stratified_sample_holdouts(
    artifact: HolographyDiagonalArtifact,
    *,
    include_model_gauges: bool = True,
) -> dict[str, Any]:
    """Score training-only interpolation and model gauges on stored samples.

    Complex voltage residuals equal visibility residuals up to the per-row
    ``S_pq`` already divided out in recovery. Null-region scores stay
    separate. Frequency transfer is not performed.
    """

    usable = [sample for sample in artifact.samples if sample.valid]
    if len(usable) < 8:
        raise ValueError("stratified holdouts need at least eight valid samples")
    cells = unique_measured_cells(usable)
    peaks = _peak_power_by_antenna(usable)
    report = {
        "held_out_references": _holdout_references(usable, peaks),
        "held_out_moving_antennas": _holdout_movers(cells, artifact, peaks),
        "later_visits": _holdout_later_visits(usable, peaks),
        "spatially_interleaved_cells": _holdout_spatial(cells, artifact, peaks),
        "shared_origins_and_cross_pass": _holdout_origins_and_pairs(cells, artifact, peaks),
        "notes": (
            HOLDOUT_LIMITATION_NOTE,
            SPW4_DIAGONAL_STATUS_NOTE,
            "Interpolation never sees holdout samples",
            "Null-adjacent is |E|^2 below 5% of that antenna's peak",
        ),
    }
    if include_model_gauges:
        report["model_gauges"] = _holdout_model_gauges(cells, artifact)
        report["stratified_model_comparison"] = stratified_model_comparison(artifact)
    return report


def stratified_model_comparison(
    artifact: HolographyDiagonalArtifact,
    *,
    beams: Mapping[str, VoltageBeamModel] | None = None,
) -> dict[str, Any]:
    """Decompose Airy / Perley / CASSBEAM residuals at measured cells."""

    from sl1mjax.holography import Memo195LowerCRaster

    models = dict(beams or holography_comparison_beams())
    models.pop("cassbeam_experimental_full_jones", None)
    usable = unique_measured_cells([sample for sample in artifact.samples if sample.valid])
    peaks = _peak_power_by_antenna(usable)
    memo = Memo195LowerCRaster()
    dense = memo.dense_radius_arcmin
    scores = {}
    for name, beam in models.items():
        entries = []
        for sample in usable:
            predicted, inside = _model_voltage(beam, sample, artifact.calibration_state)
            if predicted is None:
                continue
            entries.append(
                _score_entry(sample, predicted[0], predicted[1], peaks, inside_support=inside)
            )
        scores[name] = _stratify_entries(entries, dense_radius_arcmin=dense)
    return {
        "n_cells": len(usable),
        "models": scores,
        "notes": (
            "Whole-raster complex L2 can be dominated by outer or unsupported cells",
            "Null-adjacent scores stay separate",
            "Perley is a scalar power shape and cannot validate voltage phase or R/L",
            SPW4_DIAGONAL_STATUS_NOTE,
        ),
    }


def _holdout_references(
    samples: list[HolographyDiagonalSample],
    peaks: dict[int, tuple[float, float]],
) -> dict[str, Any]:
    groups: dict[tuple[int, int, int, int], list[HolographyDiagonalSample]] = {}
    scale = 1.0 / 2.908882086657216e-05
    for sample in samples:
        key = (
            sample.moving_antenna_id,
            int(round(float(sample.offset_lm_rad[0]) * scale)),
            int(round(float(sample.offset_lm_rad[1]) * scale)),
            int(round(sample.frequency_hz)),
        )
        groups.setdefault(key, []).append(sample)
    references = sorted(
        {
            sample.reference_antenna_id
            for sample in samples
            if sample.reference_antenna_id != COMBINED_REFERENCE_ID
        }
    )
    entries = []
    for held in references:
        for members in groups.values():
            train = [sample for sample in members if sample.reference_antenna_id != held]
            hold = [sample for sample in members if sample.reference_antenna_id == held]
            if not train or not hold:
                continue
            pred_r = complex(np.mean([sample.e_r for sample in train if sample.valid_r]))
            pred_l = complex(np.mean([sample.e_l for sample in train if sample.valid_l]))
            for sample in hold:
                entries.append(_score_entry(sample, pred_r, pred_l, peaks))
    return {
        "name": "held_out_references",
        "n_references": len(references),
        **_stratify_entries(entries),
        "notes": ("each held-out reference is the mean of the other references at that cell",),
    }


def _holdout_movers(
    cells: list[HolographyDiagonalSample],
    artifact: HolographyDiagonalArtifact,
    peaks: dict[int, tuple[float, float]],
) -> dict[str, Any]:
    movers = sorted({sample.moving_antenna_id for sample in cells})
    entries = []
    for held in movers:
        train = unique_measured_cells(
            [
                sample
                for sample in artifact.samples
                if sample.valid and sample.moving_antenna_id != held
            ]
        )
        if len(train) < 4:
            continue
        proxy = artifact_from_samples(
            train,
            calibration_state=artifact.calibration_state,
            source_name=artifact.source_name,
            reference_combination="holdout_train",
        )
        held_cells = [sample for sample in cells if sample.moving_antenna_id == held]
        if not held_cells:
            continue
        pred_r, pred_l, valid = interpolate_holography_diagonal_batch(
            proxy,
            [sample.offset_lm_rad for sample in held_cells],
            moving_antenna_id=np.full(len(held_cells), held, dtype=np.int32),
            frequency_hz=[sample.frequency_hz for sample in held_cells],
            allow_other_moving_antenna=True,
        )
        for sample, right, left, ok in zip(held_cells, pred_r, pred_l, valid, strict=True):
            if ok:
                entries.append(_score_entry(sample, complex(right), complex(left), peaks))
    return {
        "name": "held_out_moving_antennas",
        "n_moving_antennas": len(movers),
        **_stratify_entries(entries),
        "notes": ("held-out movers are interpolated from other movers' training cells only",),
    }


def _holdout_later_visits(
    samples: list[HolographyDiagonalSample],
    peaks: dict[int, tuple[float, float]],
) -> dict[str, Any]:
    scale = 1.0 / 2.908882086657216e-05
    first: dict[tuple[int, int, int, int], HolographyDiagonalSample] = {}
    later: list[HolographyDiagonalSample] = []
    for sample in sorted(samples, key=lambda item: item.unique_time_s):
        key = (
            sample.moving_antenna_id,
            sample.reference_antenna_id,
            int(round(float(sample.offset_lm_rad[0]) * scale)),
            int(round(float(sample.offset_lm_rad[1]) * scale)),
        )
        if key not in first:
            first[key] = sample
        elif sample.unique_time_s > first[key].unique_time_s + 1.0e-6:
            later.append(sample)
    entries = []
    for sample in later:
        key = (
            sample.moving_antenna_id,
            sample.reference_antenna_id,
            int(round(float(sample.offset_lm_rad[0]) * scale)),
            int(round(float(sample.offset_lm_rad[1]) * scale)),
        )
        train = first[key]
        entries.append(_score_entry(sample, train.e_r, train.e_l, peaks))
    return {
        "name": "later_visits",
        "n_first_visits": len(first),
        "n_later_visits": len(later),
        **_stratify_entries(entries),
        "notes": ("first visit trains; later visits to the same cell are predicted",),
    }


def _holdout_spatial(
    cells: list[HolographyDiagonalSample],
    artifact: HolographyDiagonalArtifact,
    peaks: dict[int, tuple[float, float]],
) -> dict[str, Any]:
    keys = [
        (
            sample.moving_antenna_id,
            int(round(float(sample.offset_lm_rad[0]) * 1.0e6)),
            int(round(float(sample.offset_lm_rad[1]) * 1.0e6)),
        )
        for sample in cells
    ]
    unique = sorted(set(keys))
    holdout_keys = set(unique[1::2])
    train = unique_measured_cells(
        [
            sample
            for sample in artifact.samples
            if sample.valid
            and (
                sample.moving_antenna_id,
                int(round(float(sample.offset_lm_rad[0]) * 1.0e6)),
                int(round(float(sample.offset_lm_rad[1]) * 1.0e6)),
            )
            not in holdout_keys
        ]
    )
    proxy = artifact_from_samples(
        train,
        calibration_state=artifact.calibration_state,
        source_name=artifact.source_name,
        reference_combination="holdout_train",
    )
    hold_cells = [
        sample
        for sample in cells
        if (
            sample.moving_antenna_id,
            int(round(float(sample.offset_lm_rad[0]) * 1.0e6)),
            int(round(float(sample.offset_lm_rad[1]) * 1.0e6)),
        )
        in holdout_keys
    ]
    entries = []
    if hold_cells:
        pred_r, pred_l, valid = interpolate_holography_diagonal_batch(
            proxy,
            [sample.offset_lm_rad for sample in hold_cells],
            moving_antenna_id=[sample.moving_antenna_id for sample in hold_cells],
            frequency_hz=[sample.frequency_hz for sample in hold_cells],
        )
        for sample, right, left, ok in zip(hold_cells, pred_r, pred_l, valid, strict=True):
            if ok:
                entries.append(_score_entry(sample, complex(right), complex(left), peaks))
    return {
        "name": "spatially_interleaved_cells",
        "n_train_cells": len(unique) - len(holdout_keys),
        "n_holdout_cells": len(holdout_keys),
        **_stratify_entries(entries),
        "notes": ("checkerboard of quantized cells; interpolator sees training cells only",),
    }


def _holdout_origins_and_pairs(
    cells: list[HolographyDiagonalSample],
    artifact: HolographyDiagonalArtifact,
    peaks: dict[int, tuple[float, float]],
) -> dict[str, Any]:
    origins = [
        sample for sample in cells if _radius_arcmin(sample.offset_lm_rad) <= ON_AXIS_RADIUS_ARCMIN
    ]
    off_axis = unique_measured_cells(
        [
            sample
            for sample in artifact.samples
            if sample.valid and _radius_arcmin(sample.offset_lm_rad) > ON_AXIS_RADIUS_ARCMIN
        ]
    )
    origin_entries = []
    if off_axis:
        proxy = artifact_from_samples(
            off_axis,
            calibration_state=artifact.calibration_state,
            source_name=artifact.source_name,
            reference_combination="holdout_train",
        )
        if origins:
            pred_r, pred_l, valid = interpolate_holography_diagonal_batch(
                proxy,
                [sample.offset_lm_rad for sample in origins],
                moving_antenna_id=[sample.moving_antenna_id for sample in origins],
                frequency_hz=[sample.frequency_hz for sample in origins],
            )
            for sample, right, left, ok in zip(origins, pred_r, pred_l, valid, strict=True):
                if ok:
                    origin_entries.append(
                        _score_entry(sample, complex(right), complex(left), peaks)
                    )
    pairs = _close_cross_pass_pairs(cells)
    pair_entries = []
    pass1 = unique_measured_cells(
        [sample for sample in artifact.samples if sample.valid and sample.raster_pass == "pass-1"]
    )
    if pass1 and pairs:
        proxy = artifact_from_samples(
            pass1,
            calibration_state=artifact.calibration_state,
            source_name=artifact.source_name,
            reference_combination="holdout_train",
        )
        pass2_cells = []
        for _pass1_offset, pass2_offset in pairs:
            for sample in cells:
                if sample.raster_pass != "pass-2":
                    continue
                if np.hypot(*(sample.offset_lm_rad - pass2_offset)) > 1.0e-7:
                    continue
                pass2_cells.append(sample)
        if pass2_cells:
            pred_r, pred_l, valid = interpolate_holography_diagonal_batch(
                proxy,
                [sample.offset_lm_rad for sample in pass2_cells],
                moving_antenna_id=[sample.moving_antenna_id for sample in pass2_cells],
                frequency_hz=[sample.frequency_hz for sample in pass2_cells],
            )
            for sample, right, left, ok in zip(pass2_cells, pred_r, pred_l, valid, strict=True):
                if ok:
                    pair_entries.append(_score_entry(sample, complex(right), complex(left), peaks))
    return {
        "name": "shared_origins_and_cross_pass",
        "n_origin_cells": len(origins),
        "n_cross_pass_pairs": len(pairs),
        "origins": _stratify_entries(origin_entries),
        "cross_pass_pairs": _stratify_entries(pair_entries),
        "notes": (
            "origins are predicted from off-axis training cells",
            "close cross-pass pairs train on pass-1 and predict pass-2",
            "pass-2 coordinates are not snapped",
        ),
    }


def _holdout_model_gauges(
    cells: list[HolographyDiagonalSample],
    artifact: HolographyDiagonalArtifact,
) -> dict[str, Any]:
    models = holography_comparison_beams()
    models.pop("cassbeam_experimental_full_jones", None)
    keys = [
        (
            sample.moving_antenna_id,
            int(round(float(sample.offset_lm_rad[0]) * 1.0e6)),
            int(round(float(sample.offset_lm_rad[1]) * 1.0e6)),
        )
        for sample in cells
    ]
    unique = sorted(set(keys))
    holdout_keys = set(unique[1::2])
    train_cells = [
        sample
        for sample in cells
        if (
            sample.moving_antenna_id,
            int(round(float(sample.offset_lm_rad[0]) * 1.0e6)),
            int(round(float(sample.offset_lm_rad[1]) * 1.0e6)),
        )
        not in holdout_keys
    ]
    hold_cells = [
        sample
        for sample in cells
        if (
            sample.moving_antenna_id,
            int(round(float(sample.offset_lm_rad[0]) * 1.0e6)),
            int(round(float(sample.offset_lm_rad[1]) * 1.0e6)),
        )
        in holdout_keys
    ]
    peaks = _peak_power_by_antenna(cells)
    scores = {}
    for name, beam in models.items():
        train_r, train_l = _model_pairs(beam, train_cells, artifact.calibration_state)
        scale_r = _complex_gauge([item[0] for item in train_r], [item[1] for item in train_r])
        scale_l = _complex_gauge([item[0] for item in train_l], [item[1] for item in train_l])
        entries = []
        for sample in hold_cells:
            predicted, inside = _model_voltage(beam, sample, artifact.calibration_state)
            if predicted is None:
                continue
            entries.append(
                _score_entry(
                    sample,
                    scale_r * predicted[0],
                    scale_l * predicted[1],
                    peaks,
                    inside_support=inside,
                )
            )
        scores[name] = {
            "train_scale_r": [scale_r.real, scale_r.imag],
            "train_scale_l": [scale_l.real, scale_l.imag],
            **_stratify_entries(entries),
        }
    return {
        "n_train_cells": len(train_cells),
        "n_holdout_cells": len(hold_cells),
        "models": scores,
        "notes": (
            "one complex scale per hand from training cells only",
            "holdout cells are never used to form the gauge",
        ),
    }


def _close_cross_pass_pairs(
    samples: list[HolographyDiagonalSample],
) -> list[tuple[np.ndarray, np.ndarray]]:
    report = pass_nearest_neighbour_report(samples)
    tolerance = float(
        report.get("tolerance_arcmin", PASS_OVERLAP_SPACING_FRACTION * DENSE_SPACING_ARCMIN)
    )
    pass1 = _unique_spatial_offsets(
        [sample for sample in samples if sample.raster_pass == "pass-1"]
    )
    pass2 = _unique_spatial_offsets(
        [sample for sample in samples if sample.raster_pass == "pass-2"]
    )
    if pass1.size == 0 or pass2.size == 0:
        return []
    pairs = []
    for offset in pass1:
        distances = np.hypot(pass2[:, 0] - offset[0], pass2[:, 1] - offset[1])
        nearest = int(np.argmin(distances))
        if distances[nearest] * 180.0 * 60.0 / np.pi <= tolerance:
            pairs.append((offset, pass2[nearest]))
    return pairs


def _peak_power_by_antenna(
    samples: list[HolographyDiagonalSample],
) -> dict[int, tuple[float, float]]:
    peaks: dict[int, list[list[float]]] = {}
    for sample in samples:
        peaks.setdefault(sample.moving_antenna_id, [[], []])
        if sample.valid_r:
            peaks[sample.moving_antenna_id][0].append(abs(sample.e_r) ** 2)
        if sample.valid_l:
            peaks[sample.moving_antenna_id][1].append(abs(sample.e_l) ** 2)
    return {
        antenna: (
            float(np.max(power[0])) if power[0] else 0.0,
            float(np.max(power[1])) if power[1] else 0.0,
        )
        for antenna, power in peaks.items()
    }


def _score_entry(
    sample: HolographyDiagonalSample,
    pred_r: complex,
    pred_l: complex,
    peaks: dict[int, tuple[float, float]],
    *,
    inside_support: bool | None = None,
) -> dict[str, Any]:
    peak_r, peak_l = peaks.get(sample.moving_antenna_id, (0.0, 0.0))
    power_r = abs(sample.e_r) ** 2 / peak_r if peak_r > 0.0 else float("nan")
    power_l = abs(sample.e_l) ** 2 / peak_l if peak_l > 0.0 else float("nan")
    return {
        "measured_r": sample.e_r,
        "predicted_r": pred_r,
        "measured_l": sample.e_l,
        "predicted_l": pred_l,
        "valid_r": sample.valid_r,
        "valid_l": sample.valid_l,
        "raster_pass": sample.raster_pass or "unknown",
        "moving_antenna_id": sample.moving_antenna_id,
        "reference_antenna_id": sample.reference_antenna_id,
        "power_r": power_r,
        "power_l": power_l,
        "radius_arcmin": _radius_arcmin(sample.offset_lm_rad),
        "inside_support": inside_support,
    }


def _stratify_entries(
    entries: list[dict[str, Any]],
    *,
    dense_radius_arcmin: float | None = None,
) -> dict[str, Any]:
    if not entries:
        return {"n": 0, "rr": _paired_hand_residual([]), "ll": _paired_hand_residual([])}
    payload = {
        "n": len(entries),
        "rr": _pairs_from_entries(entries, "R"),
        "ll": _pairs_from_entries(entries, "L"),
        "by_pass": {},
        "by_power_region": {},
    }
    for name in sorted({str(item["raster_pass"]) for item in entries}):
        payload["by_pass"][name] = {
            "rr": _pairs_from_entries(
                [item for item in entries if item["raster_pass"] == name], "R"
            ),
            "ll": _pairs_from_entries(
                [item for item in entries if item["raster_pass"] == name], "L"
            ),
        }
    for region, pred in (
        ("above_50_percent", lambda power: power >= 0.50),
        ("above_20_percent", lambda power: power >= 0.20),
        ("above_5_percent", lambda power: power >= 0.05),
        ("null_adjacent", lambda power: power < 0.05),
    ):
        payload["by_power_region"][region] = {
            "rr": _pairs_from_entries(entries, "R", power_pred=pred),
            "ll": _pairs_from_entries(entries, "L", power_pred=pred),
        }
    if dense_radius_arcmin is not None:
        payload["dense_raster"] = {
            "rr": _pairs_from_entries(
                [item for item in entries if item["radius_arcmin"] <= dense_radius_arcmin],
                "R",
            ),
            "ll": _pairs_from_entries(
                [item for item in entries if item["radius_arcmin"] <= dense_radius_arcmin],
                "L",
            ),
        }
        payload["outer_raster"] = {
            "rr": _pairs_from_entries(
                [item for item in entries if item["radius_arcmin"] > dense_radius_arcmin],
                "R",
            ),
            "ll": _pairs_from_entries(
                [item for item in entries if item["radius_arcmin"] > dense_radius_arcmin],
                "L",
            ),
        }
        payload["inside_support"] = {
            "rr": _pairs_from_entries(
                [item for item in entries if item.get("inside_support") is True], "R"
            ),
            "ll": _pairs_from_entries(
                [item for item in entries if item.get("inside_support") is True], "L"
            ),
        }
        payload["outside_support"] = {
            "rr": _pairs_from_entries(
                [item for item in entries if item.get("inside_support") is False], "R"
            ),
            "ll": _pairs_from_entries(
                [item for item in entries if item.get("inside_support") is False], "L"
            ),
        }
    return payload


def _pairs_from_entries(
    entries: list[dict[str, Any]],
    hand: str,
    *,
    power_pred=None,
) -> dict[str, float]:
    pairs = []
    power_key = "power_r" if hand == "R" else "power_l"
    valid_key = "valid_r" if hand == "R" else "valid_l"
    meas_key = "measured_r" if hand == "R" else "measured_l"
    pred_key = "predicted_r" if hand == "R" else "predicted_l"
    for item in entries:
        if not item[valid_key]:
            continue
        power = item[power_key]
        if power_pred is not None and (not np.isfinite(power) or not power_pred(power)):
            continue
        pairs.append((item[meas_key], item[pred_key]))
    return _paired_hand_residual(pairs)


def _model_voltage(
    beam: VoltageBeamModel,
    sample: HolographyDiagonalSample,
    calibration_state: str,
) -> tuple[tuple[complex, complex] | None, bool]:
    evaluation = beam.evaluate(
        beam_coordinates(
            np.array([0.0]),
            np.array([0.0]),
            np.array([sample.frequency_hz]),
            parallactic_angle_rad=np.array([0.0]),
            pointing_offset_lm_rad=sample.offset_lm_rad,
            antenna_id=np.array([sample.moving_antenna_id], dtype=np.int32),
        ),
        calibration_state=calibration_state,
    )
    jones = np.asarray(evaluation.jones).reshape(-1, 2, 2)[0]
    valid = bool(np.asarray(evaluation.valid).reshape(-1)[0])
    if not np.isfinite(jones[0, 0]) or not np.isfinite(jones[1, 1]):
        return None, valid
    return (complex(jones[0, 0]), complex(jones[1, 1])), valid


def _model_pairs(
    beam: VoltageBeamModel,
    samples: list[HolographyDiagonalSample],
    calibration_state: str,
) -> tuple[list[tuple[complex, complex]], list[tuple[complex, complex]]]:
    rr = []
    ll = []
    for sample in samples:
        predicted, _inside = _model_voltage(beam, sample, calibration_state)
        if predicted is None:
            continue
        if sample.valid_r:
            rr.append((sample.e_r, predicted[0]))
        if sample.valid_l:
            ll.append((sample.e_l, predicted[1]))
    return rr, ll


def _complex_gauge(measured: list[complex], model: list[complex]) -> complex:
    left = np.asarray(measured, dtype=np.complex128)
    right = np.asarray(model, dtype=np.complex128)
    usable = np.isfinite(left) & np.isfinite(right) & (np.abs(right) > 0.0)
    if int(np.sum(usable)) == 0:
        return 1.0 + 0.0j
    numerator = complex(np.vdot(right[usable], left[usable]))
    denominator = complex(np.vdot(right[usable], right[usable]))
    if abs(denominator) <= 0.0:
        return 1.0 + 0.0j
    return numerator / denominator
