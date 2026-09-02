"""Four-correlation 3C391 products: one G/K/B apply, then polarisation, then average."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from sl1mjax.calibration import (
    CalibrationSolution,
    apply_calibration,
    identity_solution,
)
from sl1mjax.data.averaging import average_frequency_bins, average_time_bins
from sl1mjax.data.canonical import VisibilityBlock
from sl1mjax.polarization import Correlation, ReceptorBasis
from sl1mjax.voltage_polarization import require_circular_coherency

REQUIRED_CORRELATIONS = (Correlation.RR, Correlation.RL, Correlation.LR, Correlation.LL)
FULLPOL_SCIENCE_FIELDS = (2, 3, 4, 5, 6, 7, 8)
NATIVE_CHANNEL_COUNT = 64
NATIVE_CHANNEL_WIDTH_HZ = 2.0e6
POLCAL_CHANNEL_START = 5
POLCAL_CHANNEL_STOP = 59  # exclusive; CASA spw 0:5~58
POLCAL_UNSUPPORTED_REASON = "polcal_unsupported_channel"
POLCAL_UNSUPPORTED_ANTENNA_REASON = "polcal_unsupported_antenna_dterms"
GKB_ONLY_STATE = "gkb_only"
CASA_FULLPOL_STATE = "casa_fullpol"
GKB_ONLY_FLAG_VERSION = "sl1mjax_gkb_only"
CASA_FULLPOL_FLAG_VERSION = "sl1mjax_post_polcal"
THREE_C286_FIELD_ID = 0
THREE_C286_FRACTIONAL_LINEAR = 0.112
THREE_C286_CASAGUIDE_ANGLE_DEG = 66.0
THREE_C286_Q = THREE_C286_FRACTIONAL_LINEAR * float(np.cos(np.deg2rad(66.0)))
THREE_C286_U = THREE_C286_FRACTIONAL_LINEAR * float(np.sin(np.deg2rad(66.0)))
THREE_C286_FRACTION_ABS = 0.04
THREE_C286_ANGLE_ABS_DEG = 10.0
THREE_C286_V_ABS = 0.05
FULLPOL_STOKES_I_STAGE = "baseline"
FULLPOL_STOKES_I_BEAM = "diagonal_copolar"
SCALAR_WEIGHT_LIMITATION = (
    "Retaining scalar WEIGHT_SPECTRUM after non-diagonal calibration ignores "
    "the induced cross-correlation covariance. Acceptable for this paired first "
    "experiment because both beams see identical weights; not the final "
    "full-polarisation likelihood."
)
CROSS_HAND_CORRELATIONS = (Correlation.RL, Correlation.LR)
PARALLEL_HAND_CORRELATIONS = (Correlation.RR, Correlation.LL)
POLARISATION_TEST_ANCESTOR_ROLE = "polarisation_test_ancestor"
ANCESTOR_REQUIRED_FILES = (
    "checkpoint.json",
    "component_table.json",
    "integration_plan.json",
    "summary.json",
)
PA_BIN_EDGES_DEG = (-180.0, -120.0, -60.0, 0.0, 60.0, 120.0, 180.0)
LEAKAGE_EVIDENCE_RULE = (
    "Improvement in RL/LR is the main evidence for leakage modelling. "
    "A change only in RR/LL is not sufficient evidence for the off-diagonal "
    "Jones terms."
)
DIAGNOSTIC_RUN_KIND = "fullpol_heldout_residual_diagnostic"
DIAGNOSTIC_QUESTION = (
    "Given this provisional fixed I sky and the current calibration "
    "conventions, what does the full-polarisation pipeline do to held-out "
    "four-correlation residuals?"
)
GKB_ONLY_MS_NAME = "3c391_gkb_only_4corr.ms"
CROSS_HAND_MASK_MIN_FRACTION = 0.05


def require_four_circular_correlations(block: VisibilityBlock) -> None:
    require_circular_coherency(block)
    if block.correlations != REQUIRED_CORRELATIONS:
        raise ValueError(
            "full-polarisation products require correlation order "
            f"{tuple(item.value for item in REQUIRED_CORRELATIONS)}, "
            f"got {tuple(item.value for item in block.correlations)}"
        )
    if block.receptor_basis is not ReceptorBasis.CIRCULAR:
        raise ValueError("full-polarisation products require circular receptors")


def polarization_terms_only(solution: CalibrationSolution) -> CalibrationSolution:
    """Identity G/K/B plus imported Kcross/D/X/P for already-corrected DATA."""

    identity = identity_solution(
        antenna_count=solution.antenna_count,
        correlations=solution.correlations,
        frequency_hz=solution.bandpass_frequency_hz,
        time_s=solution.gain_time_s,
        reference_antenna=solution.reference_antenna,
    )
    return replace(
        identity,
        receptors=solution.receptors,
        reference_frequency_hz=solution.reference_frequency_hz,
        antenna_position_m=solution.antenna_position_m,
        cross_hand_delay_s=solution.cross_hand_delay_s,
        cross_hand_delay_valid=solution.cross_hand_delay_valid,
        leakage=solution.leakage,
        leakage_frequency_hz=solution.leakage_frequency_hz,
        leakage_valid=solution.leakage_valid,
        leakage_application=solution.leakage_application,
        rl_phase=solution.rl_phase,
        rl_phase_frequency_hz=solution.rl_phase_frequency_hz,
        rl_phase_valid=solution.rl_phase_valid,
        apply_parallactic_angle=True,
        provenance={
            **dict(solution.provenance),
            "gkb": "identity_on_corrected_data",
            "polarisation_apply": "once",
            "evidence_grade": False,
        },
    )


def unsupported_dterm_antennas(solution: CalibrationSolution) -> tuple[int, ...]:
    """Antennas with no valid D solution on any channel or receptor."""

    if solution.leakage_valid is None:
        return ()
    valid = np.asarray(solution.leakage_valid, dtype=bool)
    per_antenna = np.any(valid, axis=tuple(range(1, valid.ndim)))
    return tuple(int(index) for index in np.flatnonzero(~per_antenna))


def mask_unsupported_polcal_channels(block: VisibilityBlock) -> VisibilityBlock:
    """Keep 64 channels and flag CASA D/X-unsupported edges 0–4 and 59–63."""

    require_four_circular_correlations(block)
    if block.frequency_hz.size != NATIVE_CHANNEL_COUNT:
        raise ValueError(
            f"native polarisation apply expects {NATIVE_CHANNEL_COUNT} channels, "
            f"got {block.frequency_hz.size}"
        )
    flag = np.asarray(block.flag, dtype=bool).copy()
    flag[:, :POLCAL_CHANNEL_START, :] = True
    flag[:, POLCAL_CHANNEL_STOP:, :] = True
    provenance = {
        **dict(block.provenance),
        "polcal_channels": [POLCAL_CHANNEL_START, POLCAL_CHANNEL_STOP - 1],
        "unsupported_channel_reason": POLCAL_UNSUPPORTED_REASON,
        "unsupported_channels": (
            list(range(POLCAL_CHANNEL_START))
            + list(range(POLCAL_CHANNEL_STOP, NATIVE_CHANNEL_COUNT))
        ),
    }
    return replace(block, flag=flag, provenance=provenance)


def apply_polarization_before_averaging(
    block: VisibilityBlock,
    solution: CalibrationSolution,
    *,
    frequency_bins: int | None = None,
    time_bin_seconds: float | None = None,
    mask_unsupported_edges: bool = True,
    extrapolate: bool = True,
) -> VisibilityBlock:
    """Apply Kcross/D/X/P once at native resolution, then optionally average."""

    require_four_circular_correlations(block)
    if int(block.provenance.get("polarisation_applied", 0)) > 0:
        raise ValueError("refusing a second polarisation apply on this block")
    applied = apply_calibration(
        block,
        polarization_terms_only(solution),
        propagate_weights=False,
        extrapolate=extrapolate,
    )
    invalid_d = unsupported_dterm_antennas(solution)
    applied = replace(
        applied,
        provenance={
            **dict(applied.provenance),
            "polarisation_applied": 1,
            "calibration_state": GKB_ONLY_STATE,
            "weight_policy": "preserve_input_weights",
            "leakage_application": solution.leakage_application,
            "unsupported_crosshand_reason": POLCAL_UNSUPPORTED_ANTENNA_REASON,
            "unsupported_dterm_antennas": list(invalid_d),
        },
    )
    if mask_unsupported_edges:
        applied = mask_unsupported_polcal_channels(applied)
    averaged = applied
    if frequency_bins is not None and frequency_bins < averaged.frequency_hz.size:
        averaged = average_frequency_bins(averaged, bin_count=int(frequency_bins))
    if time_bin_seconds is not None and time_bin_seconds > 0:
        averaged = average_time_bins(averaged, bin_seconds=float(time_bin_seconds))
    return replace(
        averaged,
        provenance={
            **dict(averaged.provenance),
            "polarisation_before_averaging": True,
        },
    )


def validate_fullpol_ms_inventory(inventory: Mapping[str, Any]) -> tuple[str, ...]:
    """Fail closed when RL/LR are missing, empty, or selected away."""

    failures: list[str] = []
    polarizations = inventory.get("polarizations") or []
    correlations: list[str] = []
    if polarizations:
        correlations = [str(item) for item in polarizations[0].get("correlations", [])]
    if tuple(correlations) != tuple(item.value for item in REQUIRED_CORRELATIONS):
        failures.append(
            f"correlation order {correlations} != "
            f"{[item.value for item in REQUIRED_CORRELATIONS]}"
        )
    windows = inventory.get("spectral_windows") or []
    if not windows:
        failures.append("spectral window metadata is missing")
    else:
        window = windows[0]
        if int(window.get("channel_count", 0)) != NATIVE_CHANNEL_COUNT:
            failures.append(
                f"channel_count {window.get('channel_count')!r} != {NATIVE_CHANNEL_COUNT}"
            )
        widths = window.get("channel_width_hz") or []
        if widths and any(abs(float(width) - NATIVE_CHANNEL_WIDTH_HZ) > 1.0 for width in widths):
            failures.append("channels are not native 2 MHz")
    field_ids = {int(item) for item in inventory.get("field_ids") or []}
    missing_fields = [field for field in FULLPOL_SCIENCE_FIELDS if field not in field_ids]
    if missing_fields:
        failures.append(f"science fields missing: {missing_fields}")
    columns = set(inventory.get("visibility_columns") or inventory.get("columns") or [])
    if "CORRECTED_DATA" not in columns:
        failures.append("CORRECTED_DATA is missing")
    counts = inventory.get("active_samples") or {}
    by_corr = counts.get("by_correlation") or {}
    for name in ("RL", "LR"):
        if name not in correlations:
            failures.append(f"{name} is absent from the polarisation table")
        elif int(by_corr.get(name, 0)) <= 0:
            failures.append(f"{name} has no active samples")
    by_field = counts.get("by_field") or {}
    for field in FULLPOL_SCIENCE_FIELDS:
        payload = by_field.get(str(field)) or {}
        if int(payload.get("RL", 0)) <= 0 or int(payload.get("LR", 0)) <= 0:
            failures.append(f"field {field} has no usable RL/LR")
    return tuple(failures)


def attach_fullpol_contract(
    inventory: dict[str, Any],
    *,
    calibration_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = dict(inventory)
    if calibration_state is not None:
        payload["calibration_state"] = dict(calibration_state)
    payload["contract_failures"] = list(validate_fullpol_ms_inventory(payload))
    payload["contract_passed"] = not payload["contract_failures"]
    return payload


def parse_flag_version_list(text: str) -> list[dict[str, str]]:
    versions = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        name, _, comment = stripped.partition(" : ")
        versions.append({"name": name.strip(), "comment": comment.strip()})
    return versions


def load_flag_versions(measurement_set: Path) -> list[dict[str, str]]:
    listing = Path(str(Path(measurement_set)) + ".flagversions") / "FLAG_VERSION_LIST"
    if not listing.is_file():
        return []
    return parse_flag_version_list(listing.read_text(encoding="utf-8"))


def load_calibration_state(measurement_set: Path) -> dict[str, Any] | None:
    sidecar = Path(str(measurement_set) + ".calibration_state.json")
    if not sidecar.is_file():
        sidecar = Path(measurement_set).with_name(
            Path(measurement_set).name.replace(".ms", ".calibration_state.json")
        )
    if not sidecar.is_file():
        return None
    import json

    return json.loads(sidecar.read_text(encoding="utf-8"))


def common_active_mask(*blocks: VisibilityBlock) -> np.ndarray:
    """Intersection of active samples. Prefer the named two-mask helpers."""

    if not blocks:
        raise ValueError("common_active_mask requires at least one block")
    mask = np.asarray(blocks[0].active, dtype=bool)
    for block in blocks[1:]:
        other = np.asarray(block.active, dtype=bool)
        if other.shape != mask.shape:
            raise ValueError("compared blocks must share visibility shape")
        mask = mask & other
    return mask


def beam_comparison_mask(jax_block: VisibilityBlock) -> np.ndarray:
    """Diagonal-versus-full-Jones: JAX-prepared active only."""

    return np.asarray(jax_block.active, dtype=bool)


def casa_comparison_mask(
    jax_block: VisibilityBlock, casa_block: VisibilityBlock
) -> np.ndarray:
    """JAX-versus-CASA: intersection. Extra CASA flags stay out of the beam test."""

    require_aligned_visibility_rows(jax_block, casa_block)
    return common_active_mask(jax_block, casa_block)


def require_aligned_visibility_rows(*blocks: VisibilityBlock) -> None:
    if len(blocks) < 2:
        return
    reference = blocks[0]
    for block in blocks[1:]:
        if block.visibility.shape != reference.visibility.shape:
            raise ValueError("compared blocks must share visibility shape")
        if not np.array_equal(block.time_s, reference.time_s):
            raise ValueError("compared blocks must share time_s")
        if not np.array_equal(block.antenna1, reference.antenna1):
            raise ValueError("compared blocks must share antenna1")
        if not np.array_equal(block.antenna2, reference.antenna2):
            raise ValueError("compared blocks must share antenna2")
        if not np.allclose(block.frequency_hz, reference.frequency_hz):
            raise ValueError("compared blocks must share frequency_hz")
        if block.correlations != reference.correlations:
            raise ValueError("compared blocks must share correlations")


def invalid_dterm_crosshand_mask(
    block: VisibilityBlock,
    solution: CalibrationSolution,
    *,
    channel_start: int = POLCAL_CHANNEL_START,
    channel_stop: int = POLCAL_CHANNEL_STOP,
) -> np.ndarray:
    """RL/LR samples on in-band channels involving an antenna with no valid D."""

    require_four_circular_correlations(block)
    invalid = unsupported_dterm_antennas(solution)
    mask = np.zeros(block.shape, dtype=bool)
    if not invalid:
        return mask
    hit = np.isin(block.antenna1, invalid) | np.isin(block.antenna2, invalid)
    names = {correlation: index for index, correlation in enumerate(block.correlations)}
    for correlation in CROSS_HAND_CORRELATIONS:
        mask[:, channel_start:channel_stop, names[correlation]] = hit[:, None]
    return mask


def compare_dterm_crosshand_flags(
    gkb_block: VisibilityBlock,
    jax_block: VisibilityBlock,
    casa_block: VisibilityBlock,
    solution: CalibrationSolution,
    *,
    channel_start: int = POLCAL_CHANNEL_START,
    channel_stop: int = POLCAL_CHANNEL_STOP,
    min_recall: float = 0.95,
) -> dict[str, Any]:
    """Check that post-JAX flags cover CASA's invalid-D RL/LR flags."""

    require_aligned_visibility_rows(gkb_block, jax_block, casa_block)
    domain = invalid_dterm_crosshand_mask(
        gkb_block,
        solution,
        channel_start=channel_start,
        channel_stop=channel_stop,
    )
    gkb_active = np.asarray(gkb_block.active, dtype=bool) & domain
    jax_flagged = (~np.asarray(jax_block.active, dtype=bool)) & domain
    casa_flagged = (~np.asarray(casa_block.active, dtype=bool)) & domain
    casa_extra = gkb_active & casa_flagged
    jax_extra = gkb_active & jax_flagged
    both = int(np.count_nonzero(casa_extra & jax_extra))
    casa_only = int(np.count_nonzero(casa_extra & ~jax_extra))
    jax_only = int(np.count_nonzero(jax_extra & ~casa_extra))
    casa_extra_count = int(np.count_nonzero(casa_extra))
    recall = 1.0 if casa_extra_count == 0 else both / casa_extra_count
    agreed = bool(recall >= min_recall)
    return {
        "unsupported_dterm_antennas": list(unsupported_dterm_antennas(solution)),
        "polcal_channels": [channel_start, channel_stop - 1],
        "domain_samples": int(np.count_nonzero(domain)),
        "casa_extra_on_gkb_active": casa_extra_count,
        "jax_extra_on_gkb_active": int(np.count_nonzero(jax_extra)),
        "both": both,
        "casa_only": casa_only,
        "jax_only": jax_only,
        "recall": recall,
        "min_recall": min_recall,
        "agreed": agreed,
        "weight_limitation": SCALAR_WEIGHT_LIMITATION,
    }


def require_row_locked_fold_masks(
    *fold_masks: np.ndarray,
) -> None:
    """A row's four correlations must share one fold assignment."""

    if not fold_masks:
        raise ValueError("require_row_locked_fold_masks needs at least one fold mask")
    row_hits = [np.any(np.asarray(mask, dtype=bool), axis=(1, 2)) for mask in fold_masks]
    stacked = np.stack(row_hits, axis=0)
    if np.any(np.sum(stacked.astype(np.int32), axis=0) > 1):
        raise ValueError("a row's correlations were split across folds")


def fullpol_phase6_folds(
    blocks: Sequence[VisibilityBlock],
    *,
    poison_sealed: bool = True,
):
    """Five interleaved 60 s folds on four-correlation blocks; fold 4 sealed."""

    from sl1mjax.phase6_protocol import phase6_folds, poison_sealed_visibilities

    folds = phase6_folds(blocks)
    for index in range(len(blocks)):
        require_row_locked_fold_masks(
            folds.train[index], folds.holdout[index], folds.sealed[index]
        )
    prepared = tuple(blocks)
    if poison_sealed:
        prepared = poison_sealed_visibilities(prepared, folds.sealed)
    return folds, prepared


def require_frozen_diagonal_checkpoint(directory: Path) -> Path:
    """Freeze seven-point diagonal_copolar Stokes I; do not wait for topology."""

    from sl1mjax.phase6_protocol import review_phase6_product

    review = review_phase6_product(
        directory,
        stage=FULLPOL_STOKES_I_STAGE,
        beam_mode=FULLPOL_STOKES_I_BEAM,
        require_guard=False,
    )
    if not review.passed:
        raise ValueError(
            "seven-point diagonal_copolar baseline is not ready to freeze: "
            + "; ".join(review.failures)
        )
    checkpoint = Path(directory) / "checkpoint.json"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return checkpoint


def _cassbeam_pin() -> dict[str, Any]:
    from sl1mjax.cassbeam_beam import CASSBEAM_CBAND_MODEL_ID

    manifest = Path(__file__).parent / "data" / "cassbeam_cband" / "manifest.json"
    payload = json.loads(manifest.read_text(encoding="utf-8")) if manifest.is_file() else {}
    return {
        "artifact_id": CASSBEAM_CBAND_MODEL_ID,
        "frozen": payload.get("frozen"),
        "casa_awp2_accepted": payload.get("casa_awp2_accepted"),
        "files": payload.get("files"),
        "native_basis": payload.get("native_basis"),
        "native_column_order": payload.get("native_column_order"),
        "notes": payload.get("notes"),
    }


def _summary_metrics(directory: Path) -> dict[str, Any] | None:
    summary = Path(directory) / "summary.json"
    if not summary.is_file():
        return None
    payload = json.loads(summary.read_text(encoding="utf-8"))
    return {
        "holdout_loss": payload.get("holdout_loss"),
        "train_loss": payload.get("train_loss"),
        "kkt_residual": payload.get("kkt_residual"),
    }


def _percent_worse(reference: float | None, candidate: float | None) -> float | None:
    if reference is None or candidate is None or reference == 0:
        return None
    return 100.0 * (float(candidate) - float(reference)) / float(reference)


def scalar_beam_selection_qualification(source: Path) -> dict[str, Any]:
    """Diagonal is the Jones-test sky, not the winning Stokes-I model."""

    baseline = Path(source).parent
    models = {
        name: metrics
        for name in ("static_scalar", "streamed_scalar", "diagonal_copolar")
        if (metrics := _summary_metrics(baseline / name)) is not None
    }
    airy = models.get("static_scalar") or {}
    streamed = models.get("streamed_scalar") or {}
    diagonal = models.get("diagonal_copolar") or {}
    return {
        "winning_stokes_i_holdout": "static_scalar",
        "diagonal_is_not_winning_stokes_i": True,
        "not_final_cband_stokes_i": True,
        "models": models,
        "diagonal_holdout_vs_airy_percent": _percent_worse(
            airy.get("holdout_loss"), diagonal.get("holdout_loss")
        ),
        "diagonal_holdout_vs_streamed_percent": _percent_worse(
            streamed.get("holdout_loss"), diagonal.get("holdout_loss")
        ),
        "lowest_kkt": "diagonal_copolar",
        "kkt_means": (
            "more stationary training solution, not a more accurate physical beam"
        ),
        "scientific_preference_among_scalar_beams": "undecided",
        "freeze_reason": (
            "Diagonal and full Jones share the same co-polar CASSBEAM response. "
            "The only intended difference is the off-diagonal beam response."
        ),
    }


def freeze_polarisation_stokes_i_ancestor(source: Path, dest: Path) -> dict[str, Any]:
    """Copy the complete diagonal product and label it a polarisation-test ancestor."""

    from sl1mjax.phase6_protocol import sha256_file

    source = Path(source)
    dest = Path(dest)
    require_frozen_diagonal_checkpoint(source)
    dest.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    copied: list[str] = []
    for item in sorted(source.iterdir()):
        if item.name == "freeze.json":
            continue
        target = dest / item.name
        if item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
            for child in sorted(target.rglob("*")):
                if child.is_file():
                    hashes[str(child.relative_to(dest))] = sha256_file(child)
        else:
            shutil.copy2(item, target)
            hashes[item.name] = sha256_file(target)
        copied.append(item.name)
    missing = [name for name in ANCESTOR_REQUIRED_FILES if name not in hashes]
    if missing:
        raise FileNotFoundError(f"ancestor is missing {missing}")
    summary = json.loads((dest / "summary.json").read_text(encoding="utf-8"))
    payload = {
        "role": POLARISATION_TEST_ANCESTOR_ROLE,
        "not_final_cband_stokes_i": True,
        "beam_mode": FULLPOL_STOKES_I_BEAM,
        "source_product": str(source.resolve()),
        "product": str(dest.resolve()),
        "copied_files": copied,
        "hashes": hashes,
        "beam_pin": _cassbeam_pin(),
        "qualification": scalar_beam_selection_qualification(source),
        "conventions": {
            "off_diagonal_sign": "unfrozen",
            "convention_acceptance": "separate_from_scientific_beam_acceptance",
            "scientific_beam_acceptance": "separate_from_convention_acceptance",
            "limited_parallactic_angle": True,
        },
        "stokes_i_folds": dict(summary.get("manifest") or {}),
        "kkt_residual": summary.get("kkt_residual"),
        "kkt_does_not_block": True,
        "kkt_note": (
            "Diagonal KKT does not block the Jones experiment because I remains "
            "identical in both arms. Lower KKT is a more stationary training "
            "solution, not a more accurate physical beam."
        ),
        "weight_limitation": SCALAR_WEIGHT_LIMITATION,
    }
    (dest / "freeze.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def require_polarisation_ancestor(directory: Path) -> tuple[Path, dict[str, Any]]:
    """Require a labelled polarisation-test ancestor, not a raw baseline product."""

    directory = Path(directory)
    freeze_path = directory / "freeze.json"
    if not freeze_path.is_file():
        raise ValueError(
            f"{directory} is not labelled a polarisation-test ancestor; "
            "run scripts/freeze_3c391_polarisation_ancestor.py"
        )
    payload = json.loads(freeze_path.read_text(encoding="utf-8"))
    if payload.get("role") != POLARISATION_TEST_ANCESTOR_ROLE:
        raise ValueError(
            f"freeze role is {payload.get('role')!r}, expected "
            f"{POLARISATION_TEST_ANCESTOR_ROLE}"
        )
    if payload.get("not_final_cband_stokes_i") is not True:
        raise ValueError(
            "ancestor must be labelled not the final C-band Stokes-I reconstruction"
        )
    require_frozen_diagonal_checkpoint(directory)
    return directory, payload


def require_calibrator_gate_report(path: Path) -> dict[str, Any]:
    report_path = Path(path) if Path(path).is_file() else Path(path) / "report.json"
    if not report_path.is_file():
        raise FileNotFoundError(report_path)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    if payload.get("passed") is not True:
        raise ValueError(
            "3C286 calibrator gate has not passed: "
            + "; ".join(payload.get("failures") or ["report.passed is not true"])
        )
    applies = payload.get("polarisation_applies") or {}
    if applies.get("exactly_one_each") is not True:
        raise ValueError("calibrator gate did not confirm polarisation applied exactly once")
    return payload


def fold_mask_digest(masks: Sequence[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for mask in masks:
        array = np.asarray(mask, dtype=np.uint8)
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def mosaic_masked_loss(
    predictions: Sequence[np.ndarray],
    blocks: Sequence[VisibilityBlock],
    masks: Sequence[np.ndarray],
) -> dict[str, float | int]:
    """Mosaic residual power and normalised MSE on the supplied masks."""

    from sl1mjax.voltage_flux_refit import weighted_residual_power

    power = 0.0
    weight = 0.0
    samples = 0
    for prediction, block, mask in zip(predictions, blocks, masks, strict=True):
        selected = np.asarray(mask, dtype=bool) & block.active
        finite_weight = np.isfinite(block.weight) & (block.weight > 0)
        usable = selected & finite_weight
        samples += int(np.count_nonzero(usable))
        residual = np.asarray(prediction) - block.visibility
        sample_power = weighted_residual_power(residual, block.weight, usable)
        if np.isfinite(sample_power):
            power += sample_power
        weight += float(np.sum(np.where(usable, block.weight, 0.0)))
    return {
        "mse": power / weight if weight > 0 else float("nan"),
        "residual_power": power,
        "weight": weight,
        "samples": samples,
    }


def _restricted_masks(
    blocks: Sequence[VisibilityBlock],
    masks: Sequence[np.ndarray],
    correlations: Sequence[Correlation] | None,
) -> tuple[np.ndarray, ...]:
    from sl1mjax.voltage_flux_refit import correlation_mask

    if correlations is None:
        return tuple(np.asarray(mask, dtype=bool) for mask in masks)
    restricted = []
    for block, mask in zip(blocks, masks, strict=True):
        selected = np.zeros(block.shape, dtype=bool)
        for correlation in correlations:
            selected |= correlation_mask(block, mask, correlation)
        restricted.append(selected)
    return tuple(restricted)


def _pa_mosaic_bins(
    predictions: Sequence[np.ndarray],
    blocks: Sequence[VisibilityBlock],
    masks: Sequence[np.ndarray],
    antenna_position_m: np.ndarray,
) -> list[dict[str, Any]]:
    from sl1mjax.calibration_terms import parallactic_angle_rad

    edges = np.deg2rad(np.asarray(PA_BIN_EDGES_DEG, dtype=np.float64))
    rows = []
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        bin_masks = []
        for block, mask in zip(blocks, masks, strict=True):
            selected = np.asarray(mask, dtype=bool) & block.active
            chi = parallactic_angle_rad(
                block.time_s, block.phase_centre_rad, antenna_position_m
            )
            if chi.ndim > 1:
                chi = np.mean(chi, axis=tuple(range(1, chi.ndim)))
            row_ok = (chi >= low) & (chi < high)
            sample = np.zeros(block.shape, dtype=bool)
            sample[row_ok] = selected[row_ok]
            bin_masks.append(sample)
        payload = mosaic_masked_loss(predictions, blocks, bin_masks)
        payload.update(
            {
                "pa_min_rad": float(low),
                "pa_max_rad": float(high),
                "pa_min_deg": float(np.rad2deg(low)),
                "pa_max_deg": float(np.rad2deg(high)),
            }
        )
        rows.append(payload)
    return rows


def fold3_mosaic_score(
    blocks: Sequence[VisibilityBlock],
    predictions: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
    *,
    antenna_position_m: np.ndarray,
    pointing_ids: Sequence[str],
) -> dict[str, Any]:
    """Fold-3 mosaic loss on the JAX beam-comparison mask."""

    hands = {
        "total": None,
        "RR": (Correlation.RR,),
        "LL": (Correlation.LL,),
        "RL": (Correlation.RL,),
        "LR": (Correlation.LR,),
        "RR_LL": PARALLEL_HAND_CORRELATIONS,
        "RL_LR": CROSS_HAND_CORRELATIONS,
    }
    scores: dict[str, Any] = {}
    for name, correlations in hands.items():
        scores[name] = mosaic_masked_loss(
            predictions,
            blocks,
            _restricted_masks(blocks, masks, correlations),
        )
    by_pointing = []
    for index, pointing_id in enumerate(pointing_ids):
        one_blocks = (blocks[index],)
        one_pred = (predictions[index],)
        one_mask = (masks[index],)
        pointing = {
            "pointing_id": pointing_id,
            **{
                name: mosaic_masked_loss(
                    one_pred,
                    one_blocks,
                    _restricted_masks(one_blocks, one_mask, correlations),
                )
                for name, correlations in hands.items()
            },
        }
        pointing["by_channel"] = []
        pointing["by_time"] = []
        block = blocks[index]
        selected = np.asarray(masks[index], dtype=bool) & block.active
        for channel in range(block.frequency_hz.size):
            channel_mask = np.zeros(block.shape, dtype=bool)
            channel_mask[:, channel, :] = selected[:, channel, :]
            payload = mosaic_masked_loss(one_pred, one_blocks, (channel_mask,))
            payload.update(
                {
                    "channel": channel,
                    "frequency_hz": float(block.frequency_hz[channel]),
                }
            )
            pointing["by_channel"].append(payload)
        unique_times, time_index = np.unique(block.time_s, return_inverse=True)
        for slot, time_s in enumerate(unique_times):
            time_mask = np.zeros(block.shape, dtype=bool)
            time_mask[time_index == slot] = selected[time_index == slot]
            payload = mosaic_masked_loss(one_pred, one_blocks, (time_mask,))
            payload["time_s"] = float(time_s)
            pointing["by_time"].append(payload)
        by_pointing.append(pointing)
    by_channel = []
    channel_count = blocks[0].frequency_hz.size if blocks else 0
    for channel in range(channel_count):
        channel_masks = []
        for block, mask in zip(blocks, masks, strict=True):
            selected = np.asarray(mask, dtype=bool) & block.active
            channel_mask = np.zeros(block.shape, dtype=bool)
            channel_mask[:, channel, :] = selected[:, channel, :]
            channel_masks.append(channel_mask)
        payload = mosaic_masked_loss(predictions, blocks, channel_masks)
        payload.update(
            {
                "channel": channel,
                "frequency_hz": float(blocks[0].frequency_hz[channel]),
            }
        )
        by_channel.append(payload)
    scores.update(
        {
            "by_pointing": by_pointing,
            "by_channel": by_channel,
            "by_pa": _pa_mosaic_bins(predictions, blocks, masks, antenna_position_m),
        }
    )
    return scores


def _metric_delta(reference: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    def _sub(left: Any, right: Any) -> float | None:
        if left is None or right is None:
            return None
        left_f = float(left)
        right_f = float(right)
        if not np.isfinite(left_f) or not np.isfinite(right_f):
            return None
        return right_f - left_f

    return {
        "mse": _sub(reference.get("mse"), candidate.get("mse")),
        "residual_power": _sub(reference.get("residual_power"), candidate.get("residual_power")),
        "samples": candidate.get("samples"),
    }


def paired_fold3_delta(
    diagonal: Mapping[str, Any], full_jones: Mapping[str, Any]
) -> dict[str, Any]:
    """``full_jones - diagonal``. Negative MSE is an improvement."""

    payload: dict[str, Any] = {
        name: _metric_delta(diagonal[name], full_jones[name])
        for name in ("total", "RR", "LL", "RL", "LR", "RR_LL", "RL_LR")
    }
    payload["by_pointing"] = []
    for left, right in zip(diagonal["by_pointing"], full_jones["by_pointing"], strict=True):
        row = {
            "pointing_id": left["pointing_id"],
            **{
                name: _metric_delta(left[name], right[name])
                for name in ("total", "RR", "LL", "RL", "LR", "RR_LL", "RL_LR")
            },
        }
        payload["by_pointing"].append(row)
    payload["by_channel"] = [
        {
            "channel": left["channel"],
            "frequency_hz": left["frequency_hz"],
            **_metric_delta(left, right),
        }
        for left, right in zip(diagonal["by_channel"], full_jones["by_channel"], strict=True)
    ]
    payload["by_pa"] = [
        {
            "pa_min_deg": left["pa_min_deg"],
            "pa_max_deg": left["pa_max_deg"],
            **_metric_delta(left, right),
        }
        for left, right in zip(diagonal["by_pa"], full_jones["by_pa"], strict=True)
    ]
    return payload


def leakage_modelling_evidence(delta: Mapping[str, Any]) -> dict[str, Any]:
    """RL/LR improvement is required before crediting off-diagonal Jones terms."""

    rl_lr = (delta.get("RL_LR") or {}).get("mse")
    rr_ll = (delta.get("RR_LL") or {}).get("mse")
    cross_improved = rl_lr is not None and rl_lr < 0.0
    parallel_improved = rr_ll is not None and rr_ll < 0.0
    return {
        "rule": LEAKAGE_EVIDENCE_RULE,
        "rl_lr_mse_delta": rl_lr,
        "rr_ll_mse_delta": rr_ll,
        "cross_hand_improved": cross_improved,
        "parallel_hand_improved": parallel_improved,
        "sufficient_for_off_diagonal_jones": cross_improved,
        "rr_ll_only_is_not_evidence": parallel_improved and not cross_improved,
        "convention_acceptance": "separate_gate",
        "scientific_beam_acceptance": "separate_gate",
        "limited_parallactic_angle": True,
    }


def diagnostic_run_metadata() -> dict[str, Any]:
    """This first full-pol compare is a residual diagnostic, not a freeze."""

    return {
        "kind": DIAGNOSTIC_RUN_KIND,
        "question": DIAGNOSTIC_QUESTION,
        "scientific_validation": False,
        "beam_selection": False,
        "do_not_freeze_full_jones": True,
        "fold_4_sealed": True,
        "not_evidence_about_the_sky": True,
        "convention_acceptance": "separate_gate",
        "scientific_beam_acceptance": "separate_gate",
        "next_after_inspect": "synthetic_and_calibrator_convention_ladder",
    }


def stamp_diagnostic_interpretation(summary: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(summary)
    payload["interpretation"] = diagnostic_run_metadata()
    payload["scientific_validation"] = False
    payload["beam_selection"] = False
    payload["do_not_freeze_full_jones"] = True
    payload["not_evidence_about_the_sky"] = True
    return payload


def _finite_metric(value: Any) -> bool:
    if value is None:
        return False
    return bool(np.isfinite(float(value)))


def abort_fullpol_diagnostic_failures(
    *,
    measurement_set: Path | None = None,
    blocks: Sequence[VisibilityBlock] | None = None,
    train_masks: Sequence[np.ndarray] | None = None,
    holdout_masks: Sequence[np.ndarray] | None = None,
    sealed_masks: Sequence[np.ndarray] | None = None,
    summary: Mapping[str, Any] | None = None,
    regional_started: bool = False,
    fold4_opened: bool = False,
) -> list[str]:
    """Concrete failures that stop the diagnostic before it is mistaken for evidence."""

    failures: list[str] = []
    if measurement_set is not None:
        if Path(measurement_set).name != GKB_ONLY_MS_NAME:
            failures.append(f"wrong MeasurementSet {Path(measurement_set).name}")
        state = load_calibration_state(measurement_set) or {}
        if state.get("calibration_state") != GKB_ONLY_STATE:
            failures.append(
                f"calibration_state is {state.get('calibration_state')!r}, expected gkb_only"
            )
        if state.get("jax_polarisation_input") is not True:
            failures.append("MeasurementSet is not labelled as the JAX polarisation input")
        if "Kcross" in (state.get("applied") or []) or "Df" in (state.get("applied") or []):
            failures.append("polarisation calibration was already applied on the GKB product")
        if state.get("do_not_apply_jax_polarisation") is True:
            failures.append("refusing the CASA full-pol product as a JAX input")
    if regional_started:
        failures.append("run started adaptive regional Q/U")
    if fold4_opened:
        failures.append("run opened sealed fold 4")
    if blocks:
        for index, block in enumerate(blocks):
            names = tuple(item.value for item in block.correlations)
            if names != tuple(item.value for item in REQUIRED_CORRELATIONS):
                failures.append(f"block {index} correlation order is {names}")
            report = correlation_sample_report(block)
            for hand in ("RL", "LR"):
                samples = int((report.get(hand) or {}).get("active_samples") or 0)
                if samples == 0:
                    failures.append(f"block {index} has no active {hand}")
            if holdout_masks is not None:
                holdout = np.asarray(holdout_masks[index], dtype=bool)
                rl = holdout[..., block.correlations.index(Correlation.RL)]
                lr = holdout[..., block.correlations.index(Correlation.LR)]
                denom = max(int(holdout.size // block.visibility.shape[-1]), 1)
                if int(np.count_nonzero(rl)) / denom < CROSS_HAND_MASK_MIN_FRACTION:
                    failures.append(f"block {index} fold-3 RL is almost entirely masked")
                if int(np.count_nonzero(lr)) / denom < CROSS_HAND_MASK_MIN_FRACTION:
                    failures.append(f"block {index} fold-3 LR is almost entirely masked")
    if train_masks is not None and sealed_masks is not None:
        for index, (train, sealed) in enumerate(zip(train_masks, sealed_masks, strict=True)):
            if np.any(np.asarray(train, dtype=bool) & np.asarray(sealed, dtype=bool)):
                failures.append(f"block {index} uses sealed fold 4 in training")
    if holdout_masks is not None and sealed_masks is not None:
        for index, (holdout, sealed) in enumerate(zip(holdout_masks, sealed_masks, strict=True)):
            if np.any(np.asarray(holdout, dtype=bool) & np.asarray(sealed, dtype=bool)):
                failures.append(f"block {index} uses sealed fold 4 in holdout")
    if summary:
        if summary.get("regional_polarization") not in {None, "not_started"}:
            failures.append("regional polarisation was started")
        beams = summary.get("beams") or {}
        sample_counts: list[int] = []
        weight_sums: list[float] = []
        for name, beam in beams.items():
            for key in ("train_loss", "holdout_loss"):
                if key in beam and not _finite_metric(beam.get(key)):
                    failures.append(f"{name} {key} is not finite")
            fold3 = beam.get("fold3") or {}
            for hand in ("total", "RR", "LL", "RL", "LR"):
                metric = fold3.get(hand) or {}
                if metric and not _finite_metric(metric.get("mse")):
                    failures.append(f"{name} fold-3 {hand} loss is not finite")
            total = fold3.get("total") or {}
            if "samples" in total:
                sample_counts.append(int(total["samples"]))
            if "weight" in total:
                weight_sums.append(float(total["weight"]))
        if len(set(sample_counts)) > 1:
            failures.append("diagonal and full-Jones arms used different samples")
        if len(weight_sums) > 1 and any(
            abs(item - weight_sums[0]) > 1.0e-9 * max(abs(weight_sums[0]), 1.0)
            for item in weight_sums[1:]
        ):
            failures.append("diagonal and full-Jones arms used different weights")
        folds = summary.get("folds") or {}
        if folds.get("sealed") != 4 or folds.get("poisoned") is not True:
            failures.append("fold 4 is not sealed and poisoned")
    return failures


def require_fullpol_diagnostic_ok(**kwargs: Any) -> None:
    failures = abort_fullpol_diagnostic_failures(**kwargs)
    if failures:
        raise ValueError("stop full-pol diagnostic: " + "; ".join(failures))


def inspect_fold3_diagnostic(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Report fold-3 residuals by hand, pointing, channel, and PA. Not a freeze."""

    beams = summary.get("beams") or {}
    inspection: dict[str, Any] = {
        "interpretation": diagnostic_run_metadata(),
        "beams": {},
    }
    for name, beam in beams.items():
        fold3 = beam.get("fold3") or {}
        inspection["beams"][name] = {
            "q": beam.get("q"),
            "u": beam.get("u"),
            "v": beam.get("v"),
            "total": fold3.get("total"),
            "RR": fold3.get("RR"),
            "LL": fold3.get("LL"),
            "RL": fold3.get("RL"),
            "LR": fold3.get("LR"),
            "RR_LL": fold3.get("RR_LL"),
            "RL_LR": fold3.get("RL_LR"),
            "by_pointing": [
                {
                    "pointing_id": row.get("pointing_id"),
                    "total": row.get("total"),
                    "RR": row.get("RR"),
                    "LL": row.get("LL"),
                    "RL": row.get("RL"),
                    "LR": row.get("LR"),
                }
                for row in fold3.get("by_pointing") or []
            ],
            "by_channel": fold3.get("by_channel") or [],
            "by_pa": fold3.get("by_pa") or [],
        }
    comparison = summary.get("comparison") or {}
    inspection["delta_full_jones_minus_diagonal"] = comparison.get(
        "delta_full_jones_minus_diagonal"
    )
    inspection["leakage_evidence"] = comparison.get("leakage_evidence")
    inspection["structured_focus"] = "large pointing, PA, or channel deltas locate convention tests"
    return inspection


def three_c286_expectation() -> dict[str, float]:
    return {
        "q": THREE_C286_Q,
        "u": THREE_C286_U,
        "v": 0.0,
        "fractional_linear": THREE_C286_FRACTIONAL_LINEAR,
        "casaguide_angle_deg": THREE_C286_CASAGUIDE_ANGLE_DEG,
    }


def evaluate_three_c286_gate(floor: Mapping[str, Any]) -> dict[str, Any]:
    """Pass/fail the casaguide 11.2% / 66° / V~0 recovery."""

    expected = three_c286_expectation()
    q = float(floor["q"])
    u = float(floor["u"])
    v = float(floor["v"])
    fraction = float(floor["fractional_linear"])
    if "casaguide_angle_deg" in floor:
        angle = float(floor["casaguide_angle_deg"])
    else:
        angle = float(np.rad2deg(float(floor["casaguide_angle_rad"])))
    failures: list[str] = []
    if abs(fraction - expected["fractional_linear"]) > THREE_C286_FRACTION_ABS:
        failures.append(
            f"Q/I,U/I fraction {fraction:.4f} != {expected['fractional_linear']:.3f}"
        )
    if abs(((angle - expected["casaguide_angle_deg"] + 180.0) % 360.0) - 180.0) > (
        THREE_C286_ANGLE_ABS_DEG
    ):
        failures.append(f"casaguide angle {angle:.2f} deg != 66")
    if abs(v - expected["v"]) > THREE_C286_V_ABS:
        failures.append(f"V/I {v:.4f} is not approximately zero")
    return {
        "expected": expected,
        "observed": {
            "q": q,
            "u": u,
            "v": v,
            "fractional_linear": fraction,
            "casaguide_angle_deg": angle,
        },
        "tolerances": {
            "fractional_linear_abs": THREE_C286_FRACTION_ABS,
            "casaguide_angle_abs_deg": THREE_C286_ANGLE_ABS_DEG,
            "v_abs": THREE_C286_V_ABS,
        },
        "passed": not failures,
        "failures": failures,
    }


def correlation_sample_report(
    block: VisibilityBlock, mask: np.ndarray | None = None
) -> dict[str, dict[str, float | int]]:
    selected = beam_comparison_mask(block) if mask is None else np.asarray(mask, dtype=bool)
    report: dict[str, dict[str, float | int]] = {}
    for index, correlation in enumerate(block.correlations):
        hand = selected[..., index]
        vis = block.visibility[..., index]
        report[correlation.value] = {
            "active_samples": int(np.count_nonzero(hand)),
            "median_abs": (
                float(np.median(np.abs(vis[hand]))) if np.any(hand) else 0.0
            ),
        }
    return report
