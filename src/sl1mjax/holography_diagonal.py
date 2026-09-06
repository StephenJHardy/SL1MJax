"""Recover complex diagonal holography beams from moving--reference visibilities.

This inverts the compact-calibrator RIME on moving--reference baselines
when the reference antenna is treated as on-axis identity. It does not
freeze a beam and does not recover off-diagonal Jones terms.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sl1mjax.beam_operator import unique_visibility_times
from sl1mjax.holography import (
    AntennaPointingRole,
    HolographyObservation,
    Memo195LowerCRaster,
    moving_reference_row_mask,
)
from sl1mjax.holography_pointing_maps import scan_passes
from sl1mjax.holography_reference_jones import reference_jones_for_antenna
from sl1mjax.polarization import Correlation, circular_stokes_to_coherency
from sl1mjax.voltage_beam import BeamCoordinates, BeamEvaluation

HOLOGRAPHY_DIAGONAL_SCHEMA_VERSION = 1
ARRAY_AVERAGE_ANTENNA_ID = -1
COMBINED_REFERENCE_ID = -1
THOL0001_REFERENCE_ANTENNA_NAMES = (
    "ea02",
    "ea03",
    "ea07",
    "ea12",
    "ea14",
    "ea24",
    "ea26",
)
FIRST_RECOVERY_FREQUENCIES_HZ = (4.564e9, 4.692e9)
NOMINAL_WEIGHT_NOTE = (
    "Nominal WEIGHT=4e6 must not define the uncertainty. Use reference "
    "scatter and repeated-sample thermal scatter."
)
FIRST_BEAM_RECOVERY_DIAGNOSTIC_NOTE = (
    "first_beam_recovery is an unfrozen diagnostic until the source and "
    "calibration gates finish. It is not a frozen beam. Compatibility "
    "G tables must not produce the scientific holography beam."
)
ABSOLUTE_RECOVERY_NOTE = (
    "Absolute recovery divides each HOLORASTER visibility by that row's "
    "field-10 MODEL_DATA coherency. CASA setjy fluxd is integrated-flux "
    "provenance only. A scalar Stokes I must not stand in for S_pq."
)
RELATIVE_ONAXIS_NORMALIZATION_NOTE = (
    "The relative product divides each moving antenna by its measured "
    "on-axis E_R and E_L. It is robust against residual scalar gain and "
    "must not conceal an inconsistent field-0/field-9 flux gauge."
)
RESTORE_ONAXIS_AGREEMENT_NOTE = (
    "Agreement between the absolute product and the relative product after "
    "restoring the on-axis normalization is an important gate."
)
PHASE_GAUGE_REMOVAL_NOTE = (
    "Phase smoothness removes one constant phase per moving antenna. That "
    "gauge removal stays in provenance. Report phase by antenna, reference, "
    "hand and pass. Near-null phase is excluded from smoothness decisions."
)
ON_AXIS_NORMALIZATION_RADIUS_ARCMIN = 0.15


@dataclass(frozen=True)
class HolographyDiagonalSample:
    """One recovered diagonal Jones sample at a raster cell.

    A per-reference sample keeps ``reference_antenna_id``. Combined
    estimates use ``COMBINED_REFERENCE_ID`` only after reference checks.
    """

    moving_antenna_id: int
    unique_time_s: float
    offset_lm_rad: NDArray[np.float64]
    frequency_hz: float
    e_r: complex
    e_l: complex
    sigma_r: float
    sigma_l: float
    n_reference: int
    weight: float
    valid: bool
    raster: str
    reference_antenna_id: int = COMBINED_REFERENCE_ID
    scan: int = -1
    raster_pass: str = ""
    offset_azelgeo_rad: NDArray[np.float64] | None = None
    flag_rr: bool = False
    flag_ll: bool = False
    weight_rr: float = float("nan")
    weight_ll: float = float("nan")
    n_baseline: int = 1
    n_sample: int = 1
    valid_r: bool = True
    valid_l: bool = True
    reference_scatter_r: float = float("nan")
    reference_scatter_l: float = float("nan")
    thermal_scatter_r: float = float("nan")
    thermal_scatter_l: float = float("nan")
    exclusion_reason: str = ""
    calibration_state: str = ""
    source_name: str = ""
    normalization: str = "absolute"
    on_axis_e_r: complex = np.nan + 1j * np.nan
    on_axis_e_l: complex = np.nan + 1j * np.nan

    def __post_init__(self) -> None:
        offset = np.asarray(self.offset_lm_rad, dtype=np.float64).reshape(2)
        object.__setattr__(self, "offset_lm_rad", offset)
        azel = (
            offset.copy()
            if self.offset_azelgeo_rad is None
            else np.asarray(self.offset_azelgeo_rad, dtype=np.float64).reshape(2)
        )
        object.__setattr__(self, "offset_azelgeo_rad", azel)


@dataclass(frozen=True)
class HolographyDiagonalArtifact:
    """Unfrozen empirical diagonal holography product.

    Axes of ``jones`` are ``(sample, receptor_out, receptor_in)`` with
    off-diagonals identically zero. Support outside the sampled raster is
    false.
    """

    samples: tuple[HolographyDiagonalSample, ...]
    jones: NDArray[np.complex128]
    valid: NDArray[np.bool_]
    off_diagonal_valid: NDArray[np.bool_]
    calibration_state: str
    source_name: str
    reference_combination: str
    schema_version: int = HOLOGRAPHY_DIAGONAL_SCHEMA_VERSION
    frozen: bool = False
    notes: tuple[str, ...] = ()
    first_beam_recovery_unfrozen: bool = True

    def __post_init__(self) -> None:
        if int(self.schema_version) != HOLOGRAPHY_DIAGONAL_SCHEMA_VERSION:
            raise ValueError("unsupported holography diagonal schema")
        if self.frozen:
            raise ValueError("empirical holography diagonal is not frozen")
        if not self.first_beam_recovery_unfrozen:
            raise ValueError("first_beam_recovery must remain an unfrozen diagnostic")
        jones = np.asarray(self.jones, dtype=np.complex128)
        valid = np.asarray(self.valid, dtype=bool)
        leakage = np.asarray(self.off_diagonal_valid, dtype=bool)
        n = len(self.samples)
        if jones.shape != (n, 2, 2):
            raise ValueError("diagonal jones must have shape (sample, 2, 2)")
        if valid.shape != (n,) or leakage.shape != (n,):
            raise ValueError("validity masks must have one value per sample")
        if np.any(np.abs(jones[:, 0, 1]) > 0.0) or np.any(np.abs(jones[:, 1, 0]) > 0.0):
            raise ValueError("diagonal artifact cannot store off-diagonal Jones")
        if np.any(leakage):
            raise ValueError("diagonal artifact must clear off-diagonal support")
        object.__setattr__(self, "jones", jones)
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "off_diagonal_valid", leakage)
        object.__setattr__(self, "samples", tuple(self.samples))
        object.__setattr__(self, "notes", tuple(self.notes))

    def power_stokes_i(self) -> NDArray[np.float64]:
        """Derived Stokes-I power. Does not replace the complex voltages."""

        return 0.5 * (np.abs(self.jones[:, 0, 0]) ** 2 + np.abs(self.jones[:, 1, 1]) ** 2)

    def receptor_centroid_lm_rad(self, receptor: str) -> NDArray[np.float64]:
        """Flux-weighted ``(l, m)`` centroid of ``|E|^2``. Derived diagnostic."""

        if receptor not in {"R", "L"}:
            raise ValueError("receptor must be 'R' or 'L'")
        slot = 0 if receptor == "R" else 1
        offsets = np.array([sample.offset_lm_rad for sample in self.samples], dtype=np.float64)
        power = np.abs(self.jones[:, slot, slot]) ** 2
        usable = self.valid & np.isfinite(power) & (power > 0.0)
        if not bool(np.any(usable)):
            return np.array([np.nan, np.nan], dtype=np.float64)
        weight = power[usable]
        return np.asarray(np.sum(offsets[usable] * weight[:, None], axis=0) / np.sum(weight))


def artifact_from_samples(
    samples: list[HolographyDiagonalSample] | tuple[HolographyDiagonalSample, ...],
    *,
    calibration_state: str,
    source_name: str,
    reference_combination: str,
    notes: tuple[str, ...] = (),
) -> HolographyDiagonalArtifact:
    """Build an unfrozen artifact from already-recovered samples."""

    jones = []
    valid = []
    for sample in samples:
        plane = np.zeros((2, 2), dtype=np.complex128)
        if sample.valid_r:
            plane[0, 0] = sample.e_r
        if sample.valid_l:
            plane[1, 1] = sample.e_l
        jones.append(plane)
        valid.append(sample.valid)
    return HolographyDiagonalArtifact(
        samples=tuple(samples),
        jones=np.stack(jones, axis=0) if samples else np.zeros((0, 2, 2), dtype=np.complex128),
        valid=np.asarray(valid, dtype=bool),
        off_diagonal_valid=np.zeros(len(samples), dtype=bool),
        calibration_state=calibration_state,
        source_name=source_name,
        reference_combination=reference_combination,
        frozen=False,
        first_beam_recovery_unfrozen=True,
        notes=notes or (FIRST_BEAM_RECOVERY_DIAGNOSTIC_NOTE, NOMINAL_WEIGHT_NOTE),
    )


def on_axis_voltage_by_moving_antenna(
    samples: tuple[HolographyDiagonalSample, ...] | list[HolographyDiagonalSample],
    *,
    radius_arcmin: float = ON_AXIS_NORMALIZATION_RADIUS_ARCMIN,
) -> dict[int, tuple[complex, complex]]:
    """Median on-axis ``E_R`` and ``E_L`` for each moving antenna."""

    radius_rad = float(radius_arcmin) * np.pi / (180.0 * 60.0)
    grouped: dict[int, list[HolographyDiagonalSample]] = {}
    for sample in samples:
        if not sample.valid:
            continue
        offset = np.asarray(
            sample.offset_azelgeo_rad
            if sample.offset_azelgeo_rad is not None
            else sample.offset_lm_rad,
            dtype=np.float64,
        )
        if float(np.hypot(offset[0], offset[1])) > radius_rad:
            continue
        grouped.setdefault(sample.moving_antenna_id, []).append(sample)
    scales: dict[int, tuple[complex, complex]] = {}
    for antenna, members in grouped.items():
        members = sorted(
            members,
            key=lambda sample: float(
                np.hypot(
                    np.asarray(
                        sample.offset_azelgeo_rad
                        if sample.offset_azelgeo_rad is not None
                        else sample.offset_lm_rad
                    )[0],
                    np.asarray(
                        sample.offset_azelgeo_rad
                        if sample.offset_azelgeo_rad is not None
                        else sample.offset_lm_rad
                    )[1],
                )
            ),
        )
        real_r = [sample.e_r for sample in members if sample.valid_r and np.isfinite(sample.e_r)]
        real_l = [sample.e_l for sample in members if sample.valid_l and np.isfinite(sample.e_l)]
        e_r = (
            complex(np.mean(np.asarray(real_r, dtype=np.complex128)))
            if real_r
            else np.nan + 1j * np.nan
        )
        e_l = (
            complex(np.mean(np.asarray(real_l, dtype=np.complex128)))
            if real_l
            else np.nan + 1j * np.nan
        )
        if np.isfinite(e_r) or np.isfinite(e_l):
            scales[antenna] = (e_r, e_l)
    return scales


def normalize_holography_diagonal_on_axis(
    artifact: HolographyDiagonalArtifact,
    *,
    radius_arcmin: float = ON_AXIS_NORMALIZATION_RADIUS_ARCMIN,
) -> HolographyDiagonalArtifact:
    """Per-moving-antenna relative beam from measured on-axis ``E_R`` and ``E_L``."""

    if artifact.frozen:
        raise ValueError("empirical holography diagonal is not frozen")
    scales = on_axis_voltage_by_moving_antenna(artifact.samples, radius_arcmin=radius_arcmin)
    if not scales:
        raise ValueError("relative beam needs a measured on-axis sample")
    samples: list[HolographyDiagonalSample] = []
    for sample in artifact.samples:
        scale = scales.get(sample.moving_antenna_id)
        if scale is None:
            samples.append(
                replace(
                    sample,
                    valid=False,
                    valid_r=False,
                    valid_l=False,
                    exclusion_reason=sample.exclusion_reason or "missing_onaxis_normalization",
                    normalization="per_antenna_onaxis",
                )
            )
            continue
        e_r_scale, e_l_scale = scale
        e_r = (
            sample.e_r / e_r_scale
            if sample.valid_r and np.isfinite(e_r_scale) and e_r_scale != 0
            else sample.e_r
        )
        e_l = (
            sample.e_l / e_l_scale
            if sample.valid_l and np.isfinite(e_l_scale) and e_l_scale != 0
            else sample.e_l
        )
        samples.append(
            replace(
                sample,
                e_r=e_r,
                e_l=e_l,
                sigma_r=sample.sigma_r / abs(e_r_scale)
                if sample.valid_r and np.isfinite(e_r_scale) and e_r_scale != 0
                else sample.sigma_r,
                sigma_l=sample.sigma_l / abs(e_l_scale)
                if sample.valid_l and np.isfinite(e_l_scale) and e_l_scale != 0
                else sample.sigma_l,
                normalization="per_antenna_onaxis",
                on_axis_e_r=e_r_scale,
                on_axis_e_l=e_l_scale,
            )
        )
    return artifact_from_samples(
        samples,
        calibration_state=artifact.calibration_state,
        source_name=artifact.source_name,
        reference_combination=artifact.reference_combination,
        notes=artifact.notes
        + (
            RELATIVE_ONAXIS_NORMALIZATION_NOTE,
            RESTORE_ONAXIS_AGREEMENT_NOTE,
            f"on_axis_radius_arcmin={radius_arcmin}",
        ),
    )


def restore_holography_diagonal_on_axis(
    artifact: HolographyDiagonalArtifact,
) -> HolographyDiagonalArtifact:
    """Multiply a relative beam back by its stored on-axis ``E_R`` and ``E_L``."""

    if artifact.frozen:
        raise ValueError("empirical holography diagonal is not frozen")
    samples: list[HolographyDiagonalSample] = []
    for sample in artifact.samples:
        if sample.normalization != "per_antenna_onaxis":
            raise ValueError("restore requires a per-antenna on-axis relative beam")
        e_r = (
            sample.e_r * sample.on_axis_e_r
            if sample.valid_r and np.isfinite(sample.on_axis_e_r)
            else sample.e_r
        )
        e_l = (
            sample.e_l * sample.on_axis_e_l
            if sample.valid_l and np.isfinite(sample.on_axis_e_l)
            else sample.e_l
        )
        samples.append(
            replace(
                sample,
                e_r=e_r,
                e_l=e_l,
                sigma_r=(
                    sample.sigma_r * abs(sample.on_axis_e_r)
                    if sample.valid_r and np.isfinite(sample.on_axis_e_r)
                    else sample.sigma_r
                ),
                sigma_l=(
                    sample.sigma_l * abs(sample.on_axis_e_l)
                    if sample.valid_l and np.isfinite(sample.on_axis_e_l)
                    else sample.sigma_l
                ),
                normalization="absolute",
            )
        )
    return artifact_from_samples(
        samples,
        calibration_state=artifact.calibration_state,
        source_name=artifact.source_name,
        reference_combination=artifact.reference_combination,
        notes=artifact.notes + (RESTORE_ONAXIS_AGREEMENT_NOTE, ABSOLUTE_RECOVERY_NOTE),
    )


def compare_absolute_and_restored_relative(
    absolute: HolographyDiagonalArtifact,
    relative: HolographyDiagonalArtifact,
) -> dict[str, float]:
    """Gate: restoring on-axis normalization recovers the absolute voltages."""

    restored = restore_holography_diagonal_on_axis(relative)
    by_key = {
        (
            sample.moving_antenna_id,
            sample.reference_antenna_id,
            float(sample.unique_time_s),
            float(sample.frequency_hz),
        ): sample
        for sample in restored.samples
    }
    delta_r: list[float] = []
    delta_l: list[float] = []
    for sample in absolute.samples:
        if not sample.valid:
            continue
        key = (
            sample.moving_antenna_id,
            sample.reference_antenna_id,
            float(sample.unique_time_s),
            float(sample.frequency_hz),
        )
        other = by_key.get(key)
        if other is None or not other.valid:
            continue
        if sample.valid_r and other.valid_r:
            denom = max(abs(sample.e_r), 1.0e-15)
            delta_r.append(abs(sample.e_r - other.e_r) / denom)
        if sample.valid_l and other.valid_l:
            denom = max(abs(sample.e_l), 1.0e-15)
            delta_l.append(abs(sample.e_l - other.e_l) / denom)
    return {
        "n_compared_r": float(len(delta_r)),
        "n_compared_l": float(len(delta_l)),
        "median_relative_abs_r": float(np.median(delta_r)) if delta_r else float("nan"),
        "median_relative_abs_l": float(np.median(delta_l)) if delta_l else float("nan"),
        "max_relative_abs_r": float(np.max(delta_r)) if delta_r else float("nan"),
        "max_relative_abs_l": float(np.max(delta_l)) if delta_l else float("nan"),
    }


def recover_holography_diagonal_per_reference(
    observation: HolographyObservation,
    visibility: ArrayLike | None = None,
    *,
    mask_unsettled: bool = True,
    reference_antenna_id: int | None = None,
    reference_jones: ArrayLike | Mapping[int, ArrayLike] | None = None,
) -> HolographyDiagonalArtifact:
    """Recover one diagonal sample per moving--reference baseline.

    Uses the calibrated on-axis identity gauge unless ``reference_jones``
    is supplied. A per-antenna map is looked up by reference id.
    The reference identifier is preserved. References are not combined.
    The artifact is never frozen.
    """

    estimates, exclusions = _diagonal_row_estimates(
        observation,
        visibility,
        mask_unsettled=mask_unsettled,
        reference_antenna_id=reference_antenna_id,
        reference_jones=reference_jones,
    )
    if not estimates:
        raise ValueError("diagonal recovery found no usable RR/LL samples")
    groups: dict[tuple[int, int, int, int], list[_RowEstimate]] = {}
    for item in estimates:
        key = (item.moving, item.reference, item.time_index, item.channel)
        groups.setdefault(key, []).append(item)
    memo = Memo195LowerCRaster()
    pass_by_scan = _raster_pass_by_scan(observation)
    samples: list[HolographyDiagonalSample] = []
    for key in sorted(groups):
        members = groups[key]
        samples.append(
            _sample_from_row_estimates(
                members,
                memo=memo,
                raster_pass=pass_by_scan.get(members[0].scan, ""),
                observation=observation,
                combined_reference=False,
            )
        )
    samples.extend(
        _excluded_sample(item, memo=memo, observation=observation, pass_by_scan=pass_by_scan)
        for item in exclusions
    )
    samples.sort(
        key=lambda sample: (
            sample.moving_antenna_id,
            sample.reference_antenna_id,
            sample.unique_time_s,
            sample.frequency_hz,
        )
    )
    return _artifact_from_samples(
        samples,
        observation,
        reference_combination="per_reference_identity_gauge",
        notes=(
            "empirical diagonal; not frozen",
            FIRST_BEAM_RECOVERY_DIAGNOSTIC_NOTE,
            NOMINAL_WEIGHT_NOTE,
            "one sample per moving--reference baseline; references not combined",
            "off-diagonal support is cleared",
            (
                "reference Jones supplied"
                if reference_jones is not None
                else "reference antennas treated as on-axis identity"
            ),
        ),
    )


def recover_holography_diagonal(
    observation: HolographyObservation,
    visibility: ArrayLike | None = None,
    *,
    mask_unsettled: bool = True,
    reference_antenna_id: int | None = None,
    reference_jones: ArrayLike | Mapping[int, ArrayLike] | None = None,
) -> HolographyDiagonalArtifact:
    """Recover complex R and L voltages from moving--reference baselines.

    The reference Jones defaults to identity. Supply ``reference_jones``
    from on-axis reference--reference baselines to remove that gauge.
    A per-antenna map is looked up by reference id. Off-diagonal terms
    are not estimated. The artifact is never frozen.

    This path still inverse-variance combines references. The first
    measured-cell run should call ``recover_holography_diagonal_per_reference``
    and only combine after the reference-treatment checks.
    """

    estimates, _exclusions = _diagonal_row_estimates(
        observation,
        visibility,
        mask_unsettled=mask_unsettled,
        reference_antenna_id=reference_antenna_id,
        reference_jones=reference_jones,
    )
    if not estimates:
        raise ValueError("diagonal recovery found no usable RR/LL samples")
    groups: dict[tuple[int, int, int], list[_RowEstimate]] = {}
    for item in estimates:
        key = (item.moving, item.time_index, item.channel)
        groups.setdefault(key, []).append(item)
    memo = Memo195LowerCRaster()
    pass_by_scan = _raster_pass_by_scan(observation)
    samples: list[HolographyDiagonalSample] = []
    for key in sorted(groups):
        members = groups[key]
        samples.append(
            _sample_from_row_estimates(
                members,
                memo=memo,
                raster_pass=pass_by_scan.get(members[0].scan, ""),
                observation=observation,
                combined_reference=reference_antenna_id is None,
            )
        )
    return _artifact_from_samples(
        samples,
        observation,
        reference_combination="inverse_variance_moving_reference",
        notes=(
            "empirical diagonal; not frozen",
            FIRST_BEAM_RECOVERY_DIAGNOSTIC_NOTE,
            NOMINAL_WEIGHT_NOTE,
            "off-diagonal support is cleared",
            (
                "reference Jones supplied"
                if reference_jones is not None
                else "reference antennas treated as on-axis identity"
            ),
        ),
    )


def average_holography_diagonal(
    artifact: HolographyDiagonalArtifact,
    *,
    offset_atol_rad: float = 2.908882086657216e-05,
) -> HolographyDiagonalArtifact:
    """Inverse-variance average of matching commanded-offset diagonal cells.

    A cell needs at least two moving antennas. One-antenna cells stay
    antenna-specific and are not relabelled as an array average.
    """

    if artifact.frozen:
        raise ValueError("empirical holography diagonal is not frozen")
    groups: dict[tuple[int, int, int], list[HolographyDiagonalSample]] = {}
    scale = 1.0 / float(offset_atol_rad)
    for sample in artifact.samples:
        if not sample.valid or sample.moving_antenna_id < 0:
            continue
        key = (
            int(round(float(sample.offset_lm_rad[0]) * scale)),
            int(round(float(sample.offset_lm_rad[1]) * scale)),
            int(round(sample.frequency_hz)),
        )
        groups.setdefault(key, []).append(sample)
    if not groups:
        raise ValueError("array-average needs antenna-specific holography samples")
    samples: list[HolographyDiagonalSample] = []
    planes = []
    valid = []
    included: set[int] = set()
    for key in sorted(groups):
        members = groups[key]
        if len({sample.moving_antenna_id for sample in members}) < 2:
            continue
        included.update(sample.moving_antenna_id for sample in members)
        combined = _combine_references(
            [
                (
                    sample.e_r,
                    sample.e_l,
                    1.0 / max(sample.sigma_r, 1.0e-15) ** 2,
                    1.0 / max(sample.sigma_l, 1.0e-15) ** 2,
                )
                for sample in members
            ]
        )
        offset = np.mean([sample.offset_lm_rad for sample in members], axis=0)
        sample = HolographyDiagonalSample(
            moving_antenna_id=ARRAY_AVERAGE_ANTENNA_ID,
            unique_time_s=float(np.mean([member.unique_time_s for member in members])),
            offset_lm_rad=np.asarray(offset, dtype=np.float64),
            frequency_hz=float(members[0].frequency_hz),
            e_r=combined[0],
            e_l=combined[1],
            sigma_r=combined[2],
            sigma_l=combined[3],
            n_reference=combined[4],
            weight=combined[5],
            valid=combined[6],
            raster=members[0].raster,
            valid_r=combined[7],
            valid_l=combined[8],
            reference_scatter_r=combined[2],
            reference_scatter_l=combined[3],
            calibration_state=artifact.calibration_state,
            source_name=artifact.source_name,
        )
        samples.append(sample)
        plane = np.zeros((2, 2), dtype=np.complex128)
        plane[0, 0] = sample.e_r
        plane[1, 1] = sample.e_l
        planes.append(plane)
        valid.append(sample.valid)
    if not samples:
        raise ValueError("array-average needs at least two moving antennas at one cell")
    antennas = ",".join(str(antenna) for antenna in sorted(included))
    return HolographyDiagonalArtifact(
        samples=tuple(samples),
        jones=np.stack(planes, axis=0),
        valid=np.asarray(valid, dtype=bool),
        off_diagonal_valid=np.zeros(len(samples), dtype=bool),
        calibration_state=artifact.calibration_state,
        source_name=artifact.source_name,
        reference_combination="inverse_variance_array_average",
        frozen=False,
        notes=artifact.notes
        + (
            "array-average of antenna-specific samples; not frozen",
            f"included_moving_antennas={antennas}",
        ),
    )


def interpolate_holography_diagonal(
    artifact: HolographyDiagonalArtifact,
    offset_lm_rad: ArrayLike,
    *,
    moving_antenna_id: int,
    frequency_hz: float,
    allow_other_moving_antenna: bool = False,
) -> tuple[complex, complex, bool]:
    """Inverse-distance in commanded ``(l, m)``, then linear in frequency.

    Does not extrapolate outside the sampled spatial box or frequency span.
    Training samples only: the caller must omit holdout cells from ``artifact``.
    """

    query = np.asarray(offset_lm_rad, dtype=np.float64).reshape(2)
    exact = _diagonal_candidates(
        artifact,
        moving_antenna_id,
        frequency_hz,
        allow_other_moving_antenna=allow_other_moving_antenna,
    )
    if exact:
        return _spatial_diagonal(exact, query)
    by_frequency: dict[int, list[HolographyDiagonalSample]] = {}
    for sample in _diagonal_candidates(
        artifact,
        moving_antenna_id,
        None,
        allow_other_moving_antenna=allow_other_moving_antenna,
    ):
        by_frequency.setdefault(int(round(sample.frequency_hz)), []).append(sample)
    frequencies: list[float] = []
    voltages: list[tuple[complex, complex]] = []
    for members in by_frequency.values():
        e_r, e_l, ok = _spatial_diagonal(members, query)
        if ok:
            frequencies.append(float(members[0].frequency_hz))
            voltages.append((e_r, e_l))
    if len(frequencies) < 2:
        return 0.0 + 0.0j, 0.0 + 0.0j, False
    order = np.argsort(frequencies)
    freqs = np.asarray(frequencies, dtype=np.float64)[order]
    if float(frequency_hz) < float(freqs[0]) - 1.0 or float(frequency_hz) > float(freqs[-1]) + 1.0:
        return 0.0 + 0.0j, 0.0 + 0.0j, False
    e_r = _interp_complex_delay(frequency_hz, freqs, np.asarray([voltages[i][0] for i in order]))
    e_l = _interp_complex_delay(frequency_hz, freqs, np.asarray([voltages[i][1] for i in order]))
    return e_r, e_l, True


def _diagonal_candidates(
    artifact: HolographyDiagonalArtifact,
    moving_antenna_id: int,
    frequency_hz: float | None,
    *,
    allow_other_moving_antenna: bool = False,
) -> list[HolographyDiagonalSample]:
    matching = [
        sample
        for sample in artifact.samples
        if sample.valid
        and (
            frequency_hz is None
            or np.isclose(sample.frequency_hz, frequency_hz, rtol=0.0, atol=1.0)
        )
    ]
    specific = [sample for sample in matching if sample.moving_antenna_id == moving_antenna_id]
    if specific:
        return specific
    if allow_other_moving_antenna:
        return matching
    return [sample for sample in matching if sample.moving_antenna_id < 0]


def _spatial_diagonal(
    candidates: list[HolographyDiagonalSample],
    query: np.ndarray,
) -> tuple[complex, complex, bool]:
    selected = _local_neighbors(candidates, query)
    if not selected:
        return 0.0 + 0.0j, 0.0 + 0.0j, False
    offsets = np.array([sample.offset_lm_rad for sample in selected], dtype=np.float64)
    distances = np.hypot(offsets[:, 0] - query[0], offsets[:, 1] - query[1])
    if float(np.min(distances)) <= 1.0e-15:
        sample = selected[int(np.argmin(distances))]
        return sample.e_r, sample.e_l, True
    weights = 1.0 / np.maximum(distances, 1.0e-15)
    weights = weights / np.sum(weights)
    e_r = complex(np.sum(weights * np.array([sample.e_r for sample in selected])))
    e_l = complex(np.sum(weights * np.array([sample.e_l for sample in selected])))
    return e_r, e_l, True


def _local_neighbors(candidates: list, query: np.ndarray, *, k: int = 4) -> list:
    if not candidates:
        return []
    by_raster: dict[str, list] = {}
    for sample in candidates:
        by_raster.setdefault(str(getattr(sample, "raster", "dense")), []).append(sample)
    chosen: list = []
    for members in by_raster.values():
        offsets = np.array([sample.offset_lm_rad for sample in members], dtype=np.float64)
        distances = np.hypot(offsets[:, 0] - query[0], offsets[:, 1] - query[1])
        spacing = _median_spacing(offsets)
        radius = 1.5 * spacing if spacing > 0.0 else 1.0e-15
        keep = np.flatnonzero(distances <= radius + 1.0e-15)
        if keep.size == 0:
            continue
        order = keep[np.argsort(distances[keep])][:k]
        chosen.extend(members[int(index)] for index in order)
    return chosen


def interpolate_holography_diagonal_batch(
    artifact: HolographyDiagonalArtifact,
    offset_lm_rad: ArrayLike,
    *,
    moving_antenna_id: ArrayLike,
    frequency_hz: ArrayLike,
    allow_other_moving_antenna: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized training-only interpolation for many query cells.

    Spacing is computed once per training set. Queries never enter the
    neighbor cloud.
    """

    queries = np.asarray(offset_lm_rad, dtype=np.float64).reshape(-1, 2)
    movers = np.asarray(moving_antenna_id, dtype=np.int32).reshape(-1)
    frequencies = np.asarray(frequency_hz, dtype=np.float64).reshape(-1)
    if queries.shape[0] != movers.size or queries.shape[0] != frequencies.size:
        raise ValueError("offset, moving_antenna_id, and frequency_hz must match")
    e_r = np.full(queries.shape[0], np.nan + 1j * np.nan, dtype=np.complex128)
    e_l = np.full(queries.shape[0], np.nan + 1j * np.nan, dtype=np.complex128)
    ok = np.zeros(queries.shape[0], dtype=bool)
    groups: dict[tuple[int, int], list[int]] = {}
    for index, (mover, frequency) in enumerate(zip(movers, frequencies, strict=True)):
        groups.setdefault((int(mover), int(round(float(frequency)))), []).append(index)
    for (mover, _freq_key), rows in groups.items():
        frequency = float(frequencies[rows[0]])
        candidates = _diagonal_candidates(
            artifact,
            mover,
            frequency,
            allow_other_moving_antenna=allow_other_moving_antenna,
        )
        if not candidates:
            continue
        pred_r, pred_l, valid = _spatial_diagonal_batch(
            candidates, queries[np.asarray(rows, dtype=np.int64)]
        )
        e_r[rows] = pred_r
        e_l[rows] = pred_l
        ok[rows] = valid
    return e_r, e_l, ok


def _spatial_diagonal_batch(
    candidates: list[HolographyDiagonalSample],
    queries: np.ndarray,
    *,
    k: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_query = int(queries.shape[0])
    e_r = np.full(n_query, np.nan + 1j * np.nan, dtype=np.complex128)
    e_l = np.full(n_query, np.nan + 1j * np.nan, dtype=np.complex128)
    ok = np.zeros(n_query, dtype=bool)
    by_raster: dict[str, list[HolographyDiagonalSample]] = {}
    for sample in candidates:
        by_raster.setdefault(str(getattr(sample, "raster", "dense")), []).append(sample)
    for members in by_raster.values():
        offsets = np.asarray([sample.offset_lm_rad for sample in members], dtype=np.float64)
        values_r = np.asarray([sample.e_r for sample in members], dtype=np.complex128)
        values_l = np.asarray([sample.e_l for sample in members], dtype=np.complex128)
        spacing = _median_spacing(offsets)
        radius = 1.5 * spacing if spacing > 0.0 else 1.0e-15
        dist = np.hypot(
            queries[:, None, 0] - offsets[None, :, 0],
            queries[:, None, 1] - offsets[None, :, 1],
        )
        exact = dist <= 1.0e-15
        has_exact = np.any(exact, axis=1)
        nearest_exact = np.argmin(np.where(exact, dist, np.inf), axis=1)
        e_r[has_exact] = values_r[nearest_exact[has_exact]]
        e_l[has_exact] = values_l[nearest_exact[has_exact]]
        ok[has_exact] = True
        usable = (~has_exact)[:, None] & (dist <= radius + 1.0e-15)
        if not bool(np.any(usable)):
            continue
        ranked = np.where(usable, dist, np.inf)
        order = np.argpartition(ranked, min(k, ranked.shape[1] - 1), axis=1)[:, :k]
        gather = np.take_along_axis(ranked, order, axis=1)
        finite = np.isfinite(gather)
        weights = np.zeros_like(gather)
        np.divide(1.0, np.maximum(gather, 1.0e-15), out=weights, where=finite)
        weight_sum = np.sum(weights, axis=1, keepdims=True)
        finite_row = (weight_sum[:, 0] > 0.0) & ~has_exact
        if not bool(np.any(finite_row)):
            continue
        np.divide(weights, weight_sum, out=weights, where=weight_sum > 0.0)
        picked_r = values_r[order]
        picked_l = values_l[order]
        e_r[finite_row] = np.sum(weights[finite_row] * picked_r[finite_row], axis=1)
        e_l[finite_row] = np.sum(weights[finite_row] * picked_l[finite_row], axis=1)
        ok[finite_row] = True
    return e_r, e_l, ok


def _unique_offsets(offsets: np.ndarray, *, atol_rad: float = 1.0e-12) -> np.ndarray:
    points = np.asarray(offsets, dtype=np.float64).reshape(-1, 2)
    if points.shape[0] == 0:
        return points
    scaled = np.round(points / float(atol_rad))
    _, index = np.unique(scaled, axis=0, return_index=True)
    return points[np.sort(index)]


def _median_spacing(offsets: np.ndarray) -> float:
    points = _unique_offsets(offsets)
    if points.shape[0] < 2:
        return 0.0
    delta = points[:, None, :] - points[None, :, :]
    dist = np.hypot(delta[..., 0], delta[..., 1])
    np.fill_diagonal(dist, np.inf)
    return float(np.median(np.min(dist, axis=1)))


def _interp_complex_delay(
    frequency_hz: float,
    frequencies: np.ndarray,
    values: np.ndarray,
) -> complex:
    phases = np.unwrap(np.angle(values))
    amplitudes = np.abs(values)
    design = np.column_stack((np.ones(frequencies.size), frequencies))
    coeff, *_ = np.linalg.lstsq(design, phases, rcond=None)
    residual = phases - (coeff[0] + coeff[1] * frequencies)
    amplitude = float(np.interp(frequency_hz, frequencies, amplitudes))
    leftover = float(np.interp(frequency_hz, frequencies, residual))
    phase = float(coeff[0] + coeff[1] * frequency_hz + leftover)
    return complex(amplitude * np.exp(1j * phase))


def _source_coherency(observation: HolographyObservation) -> NDArray[np.complex128]:
    if observation.source_coherency_visibility is not None:
        return np.asarray(observation.source_coherency_visibility, dtype=np.complex128)
    if observation.source_model is not None:
        return observation.source_model.evaluate_coherency(observation.block)
    plane = circular_stokes_to_coherency(observation.stokes_i, 0.0, 0.0, 0.0)
    n_row = int(observation.block.time_s.shape[0])
    n_chan = int(observation.block.frequency_hz.size)
    return np.broadcast_to(plane, (n_row, n_chan, 2, 2)).copy()


def source_model_stokes_i(observation: HolographyObservation) -> NDArray[np.float64]:
    """Per-row source Stokes I. Never the measured, beam-attenuated visibility.

    Uses the packed source coherency: ``|(S_RR + S_LL) / 2|``.
    """

    source = _source_coherency(observation)
    if source.ndim == 4:
        plane = source[:, 0]
    elif source.ndim == 3:
        plane = source
    else:
        plane = np.broadcast_to(source, (int(observation.block.time_s.size), 2, 2))
    return np.abs(0.5 * (plane[..., 0, 0] + plane[..., 1, 1])).astype(np.float64)


def four_hand_active_rows(observation: HolographyObservation) -> NDArray[np.bool_]:
    """Rows with every selected channel and correlation active."""

    return np.all(np.asarray(observation.block.active, dtype=bool), axis=(1, 2))


def copolar_hand_active_rows(
    observation: HolographyObservation,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Independent RR and LL sample masks. Incomplete cross-hands are allowed."""

    from sl1mjax.polarization import Correlation

    active = np.asarray(observation.block.active, dtype=bool)
    correlations = tuple(observation.block.correlations)
    rr = correlations.index(Correlation.RR)
    ll = correlations.index(Correlation.LL)
    return np.all(active[:, :, rr], axis=1), np.all(active[:, :, ll], axis=1)


def holography_row_exclusion_counts(observation: HolographyObservation) -> dict[str, object]:
    """Vectorized row-removal accounting. Does not walk MAIN row-by-row."""

    _offsets, inverse, pointing_valid, settled, _moving = observation.pointing_state()
    time_index = np.asarray(inverse, dtype=np.int32)
    antenna_p = np.asarray(observation.block.antenna1, dtype=np.int32)
    antenna_q = np.asarray(observation.block.antenna2, dtype=np.int32)
    pointing_ok = pointing_valid[time_index, antenna_p] & pointing_valid[time_index, antenna_q]
    settled_ok = settled[time_index, antenna_p] & settled[time_index, antenna_q]
    flag = np.asarray(observation.block.flag, dtype=bool)
    weight = np.asarray(observation.block.weight, dtype=np.float64)
    visibility = np.asarray(observation.block.visibility)
    any_flag = np.any(flag, axis=(1, 2))
    invalid_weight = np.any(~np.isfinite(weight) | (weight <= 0.0), axis=(1, 2))
    incomplete = ~four_hand_active_rows(observation)
    missing_pointing = ~pointing_ok
    unsettled = pointing_ok & ~settled_ok
    exclusive = np.full(antenna_p.size, "kept", dtype="U24")
    exclusive[incomplete] = "incomplete_coherency"
    exclusive[invalid_weight] = "invalid_weight"
    exclusive[any_flag] = "flagged"
    exclusive[unsettled] = "unsettled"
    exclusive[missing_pointing] = "missing_pointing"
    return {
        "n_rows": int(antenna_p.size),
        "n_missing_pointing": int(np.sum(missing_pointing)),
        "n_unsettled": int(np.sum(unsettled)),
        "n_any_flag": int(np.sum(any_flag)),
        "n_invalid_weight": int(np.sum(invalid_weight)),
        "n_incomplete_coherency": int(np.sum(incomplete)),
        "n_settled_four_hand": int(np.sum(pointing_ok & settled_ok & ~incomplete)),
        "exclusive": {
            label: int(np.sum(exclusive == label))
            for label in (
                "missing_pointing",
                "unsettled",
                "flagged",
                "invalid_weight",
                "incomplete_coherency",
                "kept",
            )
        },
        "nonfinite_visibility": int(
            np.sum(~np.isfinite(visibility.real) | ~np.isfinite(visibility.imag))
        ),
    }


def four_hand_sample_weight(observation: HolographyObservation) -> NDArray[np.float64]:
    """Conservative per-row weight: the weakest of the four-hand samples."""

    return np.min(np.asarray(observation.block.weight, dtype=np.float64), axis=(1, 2))


def _moving_and_reference(
    antenna_p: int,
    antenna_q: int,
    roles: np.ndarray,
) -> tuple[int, int]:
    role_p = str(roles[antenna_p])
    role_q = str(roles[antenna_q])
    if role_p == AntennaPointingRole.MOVING.value and role_q == AntennaPointingRole.REFERENCE.value:
        return antenna_p, antenna_q
    if role_q == AntennaPointingRole.MOVING.value and role_p == AntennaPointingRole.REFERENCE.value:
        return antenna_q, antenna_p
    raise ValueError("row is not a moving--reference baseline")


def _combine_references(
    estimates: list[tuple[complex, complex, float, float]],
) -> tuple[complex, complex, float, float, int, float, bool, bool, bool]:
    e_r = np.asarray([item[0] for item in estimates], dtype=np.complex128)
    e_l = np.asarray([item[1] for item in estimates], dtype=np.complex128)
    w_r = np.asarray([item[2] for item in estimates], dtype=np.float64)
    w_l = np.asarray([item[3] for item in estimates], dtype=np.float64)
    sum_r = float(np.sum(w_r))
    sum_l = float(np.sum(w_l))
    valid_r = sum_r > 0.0
    valid_l = sum_l > 0.0
    if not valid_r and not valid_l:
        return (
            np.nan + 1j * np.nan,
            np.nan + 1j * np.nan,
            float("nan"),
            float("nan"),
            len(estimates),
            0.0,
            False,
            False,
            False,
        )
    mean_r = complex(np.sum(w_r * e_r) / sum_r) if valid_r else np.nan + 1j * np.nan
    mean_l = complex(np.sum(w_l * e_l) / sum_l) if valid_l else np.nan + 1j * np.nan
    scatter_r = (
        float(np.sqrt(np.sum(w_r * np.abs(e_r - mean_r) ** 2) / sum_r)) if valid_r else float("nan")
    )
    scatter_l = (
        float(np.sqrt(np.sum(w_l * np.abs(e_l - mean_l) ** 2) / sum_l)) if valid_l else float("nan")
    )
    sigma_r = scatter_r if scatter_r > 0.0 else float("nan")
    sigma_l = scatter_l if scatter_l > 0.0 else float("nan")
    return (
        mean_r,
        mean_l,
        float(sigma_r),
        float(sigma_l),
        len(estimates),
        0.5 * (sum_r + sum_l),
        valid_r or valid_l,
        valid_r,
        valid_l,
    )


def _raster_label(offset_lm_rad: np.ndarray, memo: Memo195LowerCRaster) -> str:
    radius = float(np.hypot(offset_lm_rad[0], offset_lm_rad[1]) * 180.0 * 60.0 / np.pi)
    if radius <= memo.dense_radius_arcmin:
        return "dense"
    return "sparse"


@dataclass(frozen=True)
class _RowEstimate:
    moving: int
    reference: int
    time_index: int
    channel: int
    time_s: float
    frequency_hz: float
    offset: NDArray[np.float64]
    scan: int
    e_r: complex
    e_l: complex
    weight_rr: float
    weight_ll: float
    flag_rr: bool
    flag_ll: bool
    exclusion_reason: str = ""


@dataclass(frozen=True)
class MeasuredCellDiagonalVoltageBeam:
    """Lookup-only empirical diagonal. Refuses interpolation and averaging."""

    samples: tuple[HolographyDiagonalSample, ...]
    model_id: str = "measured_cell_diagonal"
    offset_atol_rad: float = 1.0e-12
    frequency_atol_hz: float = 1.0
    identity_for_unmatched_reference: bool = True

    def evaluate(
        self,
        coordinates: BeamCoordinates,
        *,
        calibration_state: str,
    ) -> BeamEvaluation:
        del calibration_state
        l_rad = np.asarray(coordinates.l_rad, dtype=np.float64).reshape(-1)
        m_rad = np.asarray(coordinates.m_rad, dtype=np.float64).reshape(-1)
        frequency = np.asarray(coordinates.frequency_hz, dtype=np.float64).reshape(-1)
        pointing = coordinates.pointing_offset_lm_rad
        if pointing is None:
            offset = np.stack((l_rad, m_rad), axis=1)
        else:
            offset = np.broadcast_to(
                np.asarray(pointing, dtype=np.float64).reshape(2), (l_rad.size, 2)
            )
        antennas = (
            np.asarray(coordinates.antenna_id, dtype=np.int32).reshape(-1)
            if coordinates.antenna_id is not None
            else np.full(1, -1, dtype=np.int32)
        )
        n_ant = int(antennas.size)
        n_dir = int(l_rad.size)
        n_chan = int(frequency.size)
        jones = np.zeros((n_ant, n_dir, n_chan, 2, 2), dtype=np.complex128)
        valid = np.zeros((n_ant, n_dir, n_chan), dtype=bool)
        for ant_index, antenna in enumerate(antennas):
            for dir_index in range(n_dir):
                for chan_index, freq in enumerate(frequency):
                    sample = self._lookup(int(antenna), offset[dir_index], float(freq))
                    if sample is None:
                        if self.identity_for_unmatched_reference and _is_on_axis(
                            offset[dir_index], self.offset_atol_rad
                        ):
                            jones[ant_index, dir_index, chan_index] = np.eye(2)
                            valid[ant_index, dir_index, chan_index] = True
                        continue
                    if sample.valid_r:
                        jones[ant_index, dir_index, chan_index, 0, 0] = sample.e_r
                    if sample.valid_l:
                        jones[ant_index, dir_index, chan_index, 1, 1] = sample.e_l
                    valid[ant_index, dir_index, chan_index] = sample.valid_r or sample.valid_l
        return BeamEvaluation(
            jones=jones,
            valid=valid,
            provenance={
                "model_id": self.model_id,
                "interpolation": False,
                "array_average": False,
                "lookup": "measured_cell",
            },
            off_diagonal_valid=np.zeros_like(valid),
        )

    def _lookup(
        self, antenna_id: int, offset: np.ndarray, frequency_hz: float
    ) -> HolographyDiagonalSample | None:
        matches = [
            sample
            for sample in self.samples
            if sample.valid
            and sample.moving_antenna_id == antenna_id
            and np.allclose(sample.offset_lm_rad, offset, rtol=0.0, atol=self.offset_atol_rad)
            and np.isclose(sample.frequency_hz, frequency_hz, rtol=0.0, atol=self.frequency_atol_hz)
        ]
        if not matches:
            return None
        if len({sample.reference_antenna_id for sample in matches}) == 1:
            return matches[0]
        return None


def predict_diagonal_from_samples(
    observation: HolographyObservation,
    samples: tuple[HolographyDiagonalSample, ...] | list[HolographyDiagonalSample],
    *,
    row_mask: ArrayLike | None = None,
    frequency_model: object | None = None,
    holdout_frequency_hz: float | None = None,
    allow_other_moving_antenna: bool = False,
    allow_other_reference: bool = False,
) -> NDArray[np.complex128]:
    """Predict RR/LL from recovered measured cells. No spatial interpolation.

    Predicting a second frequency requires an explicit ``frequency_model``.
    """

    if holdout_frequency_hz is not None and frequency_model is None:
        raise ValueError(
            "recover at 4.564 GHz and predict 4.692 GHz only after a frequency model is introduced"
        )
    names = tuple(
        item.value if isinstance(item, Correlation) else str(item)
        for item in observation.block.correlations
    )
    rr_index = names.index("RR")
    ll_index = names.index("LL")
    source = _source_coherency(observation)
    predicted = np.zeros(observation.block.visibility.shape, dtype=np.complex128)
    offsets, inverse, pointing_valid, _settled, _moving = observation.pointing_state()
    roles = _role_grid(observation, offsets.shape[:2])
    by_key = _samples_by_moving_reference_cell(samples)
    rows = (
        np.flatnonzero(np.asarray(row_mask, dtype=bool))
        if row_mask is not None
        else np.arange(observation.block.time_s.shape[0])
    )
    for row in rows:
        time_index = int(inverse[row])
        antenna_p = int(observation.block.antenna1[row])
        antenna_q = int(observation.block.antenna2[row])
        if not (pointing_valid[time_index, antenna_p] and pointing_valid[time_index, antenna_q]):
            continue
        try:
            moving, reference = _moving_and_reference(antenna_p, antenna_q, roles[time_index])
        except ValueError:
            continue
        offset = np.asarray(offsets[time_index, moving], dtype=np.float64)
        time_s = float(observation.pointing.unique_time_s[time_index])
        for channel in range(observation.block.frequency_hz.size):
            frequency = float(observation.block.frequency_hz[channel])
            if holdout_frequency_hz is not None and not np.isclose(
                frequency, holdout_frequency_hz, rtol=0.0, atol=1.0
            ):
                continue
            sample = _lookup_prediction_sample(
                by_key,
                moving,
                reference,
                time_s,
                offset,
                frequency,
                allow_other_moving_antenna=allow_other_moving_antenna,
                allow_other_reference=allow_other_reference,
            )
            if sample is None:
                continue
            s_rr = source[row, channel, 0, 0]
            s_ll = source[row, channel, 1, 1]
            v_rr = sample.e_r * s_rr if sample.valid_r else np.nan + 1j * np.nan
            v_ll = sample.e_l * s_ll if sample.valid_l else np.nan + 1j * np.nan
            if moving != antenna_p:
                v_rr = np.conjugate(v_rr)
                v_ll = np.conjugate(v_ll)
            predicted[row, channel, rr_index] = v_rr
            predicted[row, channel, ll_index] = v_ll
    return predicted


def samples_from_records(records: list[dict[str, object]]) -> list[HolographyDiagonalSample]:
    """Rebuild measured-cell samples from ``sample_records`` JSON."""

    samples = []
    for record in records:
        e_r = _complex_from_parts(record.get("e_r_real"), record.get("e_r_imag"))
        e_l = _complex_from_parts(record.get("e_l_real"), record.get("e_l_imag"))
        samples.append(
            HolographyDiagonalSample(
                moving_antenna_id=int(record["moving_antenna_id"]),
                unique_time_s=float(record["unique_time_s"]),
                offset_lm_rad=np.asarray(record["offset_lm_rad"], dtype=np.float64),
                frequency_hz=float(record["frequency_hz"]),
                e_r=e_r,
                e_l=e_l,
                sigma_r=float(record.get("sigma_r", float("nan"))),
                sigma_l=float(record.get("sigma_l", float("nan"))),
                n_reference=int(record.get("n_reference", 1)),
                weight=float(record.get("weight", float("nan"))),
                valid=bool(record.get("valid", False)),
                raster=str(record.get("raster", "")),
                reference_antenna_id=int(record.get("reference_antenna_id", COMBINED_REFERENCE_ID)),
                scan=int(record.get("scan", -1)),
                raster_pass=str(record.get("raster_pass", "")),
                offset_azelgeo_rad=np.asarray(
                    record.get("offset_azelgeo_rad", record["offset_lm_rad"]),
                    dtype=np.float64,
                ),
                flag_rr=bool(record.get("flag_rr", False)),
                flag_ll=bool(record.get("flag_ll", False)),
                weight_rr=float(record.get("weight_rr", float("nan"))),
                weight_ll=float(record.get("weight_ll", float("nan"))),
                n_baseline=int(record.get("n_baseline", 1)),
                n_sample=int(record.get("n_sample", 1)),
                valid_r=bool(record.get("valid_r", False)),
                valid_l=bool(record.get("valid_l", False)),
                reference_scatter_r=float(record.get("reference_scatter_r", float("nan"))),
                reference_scatter_l=float(record.get("reference_scatter_l", float("nan"))),
                thermal_scatter_r=float(record.get("thermal_scatter_r", float("nan"))),
                thermal_scatter_l=float(record.get("thermal_scatter_l", float("nan"))),
                exclusion_reason=str(record.get("exclusion_reason", "")),
                calibration_state=str(record.get("calibration_state", "")),
                source_name=str(record.get("source_name", "")),
                normalization=str(record.get("normalization", "absolute")),
                on_axis_e_r=_complex_from_parts(
                    record.get("on_axis_e_r_real"), record.get("on_axis_e_r_imag")
                ),
                on_axis_e_l=_complex_from_parts(
                    record.get("on_axis_e_l_real"), record.get("on_axis_e_l_imag")
                ),
            )
        )
    return samples


def _complex_from_parts(real: object, imag: object) -> complex:
    if real is None or imag is None:
        return np.nan + 1j * np.nan
    return complex(float(real), float(imag))


def sample_records(
    samples: tuple[HolographyDiagonalSample, ...] | list[HolographyDiagonalSample],
) -> list[dict[str, object]]:
    """JSON-safe records for every recovered or excluded measured cell."""

    records = []
    for sample in samples:
        records.append(
            {
                "moving_antenna_id": sample.moving_antenna_id,
                "reference_antenna_id": sample.reference_antenna_id,
                "scan": sample.scan,
                "raster_pass": sample.raster_pass,
                "unique_time_s": sample.unique_time_s,
                "frequency_hz": sample.frequency_hz,
                "offset_azelgeo_rad": [
                    float(sample.offset_azelgeo_rad[0]),
                    float(sample.offset_azelgeo_rad[1]),
                ],
                "offset_lm_rad": [float(sample.offset_lm_rad[0]), float(sample.offset_lm_rad[1])],
                "e_r_real": float(np.real(sample.e_r)) if np.isfinite(sample.e_r) else None,
                "e_r_imag": float(np.imag(sample.e_r)) if np.isfinite(sample.e_r) else None,
                "e_l_real": float(np.real(sample.e_l)) if np.isfinite(sample.e_l) else None,
                "e_l_imag": float(np.imag(sample.e_l)) if np.isfinite(sample.e_l) else None,
                "flag_rr": sample.flag_rr,
                "flag_ll": sample.flag_ll,
                "weight_rr": sample.weight_rr,
                "weight_ll": sample.weight_ll,
                "n_baseline": sample.n_baseline,
                "n_sample": sample.n_sample,
                "n_reference": sample.n_reference,
                "reference_scatter_r": sample.reference_scatter_r,
                "reference_scatter_l": sample.reference_scatter_l,
                "thermal_scatter_r": sample.thermal_scatter_r,
                "thermal_scatter_l": sample.thermal_scatter_l,
                "sigma_r": sample.sigma_r,
                "sigma_l": sample.sigma_l,
                "valid": sample.valid,
                "valid_r": sample.valid_r,
                "valid_l": sample.valid_l,
                "raster": sample.raster,
                "exclusion_reason": sample.exclusion_reason,
                "calibration_state": sample.calibration_state,
                "source_name": sample.source_name,
                "normalization": sample.normalization,
                "on_axis_e_r_real": (
                    float(np.real(sample.on_axis_e_r)) if np.isfinite(sample.on_axis_e_r) else None
                ),
                "on_axis_e_r_imag": (
                    float(np.imag(sample.on_axis_e_r)) if np.isfinite(sample.on_axis_e_r) else None
                ),
                "on_axis_e_l_real": (
                    float(np.real(sample.on_axis_e_l)) if np.isfinite(sample.on_axis_e_l) else None
                ),
                "on_axis_e_l_imag": (
                    float(np.imag(sample.on_axis_e_l)) if np.isfinite(sample.on_axis_e_l) else None
                ),
            }
        )
    return records


def _diagonal_row_estimates(
    observation: HolographyObservation,
    visibility: ArrayLike | None,
    *,
    mask_unsettled: bool,
    reference_antenna_id: int | None,
    reference_jones: ArrayLike | Mapping[int, ArrayLike] | None,
) -> tuple[list[_RowEstimate], list[_RowEstimate]]:
    measured = (
        np.asarray(observation.block.visibility, dtype=np.complex128)
        if visibility is None
        else np.asarray(visibility, dtype=np.complex128)
    )
    if measured.shape != observation.block.visibility.shape:
        raise ValueError("visibility must match the observation visibility shape")
    names = tuple(
        item.value if isinstance(item, Correlation) else str(item)
        for item in observation.block.correlations
    )
    if "RR" not in names or "LL" not in names:
        raise ValueError("diagonal recovery needs RR and LL")
    rr_index = names.index("RR")
    ll_index = names.index("LL")
    source = _source_coherency(observation)
    moving_ref = moving_reference_row_mask(observation.block, observation.pointing)
    active = observation.active_row_mask(
        mask_unsettled=mask_unsettled,
        moving_reference_only=True,
    )
    rows = np.flatnonzero(moving_ref & active)
    if rows.size == 0:
        raise ValueError("diagonal recovery needs usable moving--reference rows")
    offsets, inverse, pointing_valid, _settled, _moving = observation.pointing_state()
    roles = _role_grid(observation, offsets.shape[:2])
    unique_times, _ = unique_visibility_times(observation.block.time_s)
    estimates: list[_RowEstimate] = []
    exclusions: list[_RowEstimate] = []
    for row in rows:
        time_index = int(inverse[row])
        antenna_p = int(observation.block.antenna1[row])
        antenna_q = int(observation.block.antenna2[row])
        if not (pointing_valid[time_index, antenna_p] and pointing_valid[time_index, antenna_q]):
            continue
        moving, reference = _moving_and_reference(antenna_p, antenna_q, roles[time_index])
        if reference_antenna_id is not None and reference != int(reference_antenna_id):
            continue
        ref_jones = reference_jones_for_antenna(reference_jones, reference)
        offset = np.asarray(offsets[time_index, moving], dtype=np.float64)
        scan = int(observation.block.scan_id[row])
        for channel in range(observation.block.frequency_hz.size):
            s_rr = source[row, channel, 0, 0]
            s_ll = source[row, channel, 1, 1]
            flag = observation.block.flag[row, channel]
            flag_rr = bool(flag[rr_index])
            flag_ll = bool(flag[ll_index])
            weight_rr = float(observation.block.weight[row, channel, rr_index])
            weight_ll = float(observation.block.weight[row, channel, ll_index])
            if flag_rr:
                weight_rr = 0.0
            if flag_ll:
                weight_ll = 0.0
            reason = ""
            if abs(s_rr) < 1.0e-15 or abs(s_ll) < 1.0e-15:
                reason = "empty_source"
            elif weight_rr <= 0.0 and weight_ll <= 0.0:
                reason = "flagged_rr_and_ll"
            e_r = np.nan + 1j * np.nan
            e_l = np.nan + 1j * np.nan
            if reason == "":
                if weight_rr > 0.0:
                    e_r = measured[row, channel, rr_index] / s_rr
                    if moving != antenna_p:
                        e_r = np.conjugate(e_r)
                    if abs(ref_jones[0, 0]) > 0.0:
                        e_r = e_r / np.conjugate(ref_jones[0, 0])
                if weight_ll > 0.0:
                    e_l = measured[row, channel, ll_index] / s_ll
                    if moving != antenna_p:
                        e_l = np.conjugate(e_l)
                    if abs(ref_jones[1, 1]) > 0.0:
                        e_l = e_l / np.conjugate(ref_jones[1, 1])
            item = _RowEstimate(
                moving=int(moving),
                reference=int(reference),
                time_index=time_index,
                channel=channel,
                time_s=float(unique_times[time_index]),
                frequency_hz=float(observation.block.frequency_hz[channel]),
                offset=offset,
                scan=scan,
                e_r=complex(e_r),
                e_l=complex(e_l),
                weight_rr=weight_rr,
                weight_ll=weight_ll,
                flag_rr=flag_rr,
                flag_ll=flag_ll,
                exclusion_reason=reason,
            )
            if reason:
                exclusions.append(item)
            else:
                estimates.append(item)
    return estimates, exclusions


def _sample_from_row_estimates(
    members: list[_RowEstimate],
    *,
    memo: Memo195LowerCRaster,
    raster_pass: str,
    observation: HolographyObservation,
    combined_reference: bool,
) -> HolographyDiagonalSample:
    combined = _combine_references(
        [(item.e_r, item.e_l, item.weight_rr, item.weight_ll) for item in members]
    )
    references = {item.reference for item in members}
    thermal_r, thermal_l = _thermal_scatter(members)
    sigma_r = combined[2]
    sigma_l = combined[3]
    if combined_reference and len(references) > 1:
        ref_scatter_r = combined[2]
        ref_scatter_l = combined[3]
    else:
        ref_scatter_r = float("nan")
        ref_scatter_l = float("nan")
        sigma_r = thermal_r
        sigma_l = thermal_l
    first = members[0]
    return HolographyDiagonalSample(
        moving_antenna_id=int(first.moving),
        unique_time_s=first.time_s,
        offset_lm_rad=first.offset,
        frequency_hz=first.frequency_hz,
        e_r=combined[0],
        e_l=combined[1],
        sigma_r=sigma_r,
        sigma_l=sigma_l,
        n_reference=len(references),
        weight=combined[5],
        valid=combined[6],
        raster=_raster_label(first.offset, memo),
        reference_antenna_id=COMBINED_REFERENCE_ID if combined_reference else int(first.reference),
        scan=int(first.scan),
        raster_pass=raster_pass,
        offset_azelgeo_rad=first.offset,
        flag_rr=all(item.flag_rr for item in members),
        flag_ll=all(item.flag_ll for item in members),
        weight_rr=float(np.sum([item.weight_rr for item in members])),
        weight_ll=float(np.sum([item.weight_ll for item in members])),
        n_baseline=len({(item.moving, item.reference) for item in members}),
        n_sample=len(members),
        valid_r=combined[7],
        valid_l=combined[8],
        reference_scatter_r=ref_scatter_r,
        reference_scatter_l=ref_scatter_l,
        thermal_scatter_r=thermal_r,
        thermal_scatter_l=thermal_l,
        calibration_state=observation.calibration_state,
        source_name=observation.source_name,
    )


def _excluded_sample(
    item: _RowEstimate,
    *,
    memo: Memo195LowerCRaster,
    observation: HolographyObservation,
    pass_by_scan: dict[int, str],
) -> HolographyDiagonalSample:
    return HolographyDiagonalSample(
        moving_antenna_id=int(item.moving),
        unique_time_s=item.time_s,
        offset_lm_rad=item.offset,
        frequency_hz=item.frequency_hz,
        e_r=item.e_r,
        e_l=item.e_l,
        sigma_r=float("nan"),
        sigma_l=float("nan"),
        n_reference=1,
        weight=0.0,
        valid=False,
        raster=_raster_label(item.offset, memo),
        reference_antenna_id=int(item.reference),
        scan=int(item.scan),
        raster_pass=pass_by_scan.get(item.scan, ""),
        offset_azelgeo_rad=item.offset,
        flag_rr=item.flag_rr,
        flag_ll=item.flag_ll,
        weight_rr=item.weight_rr,
        weight_ll=item.weight_ll,
        n_baseline=1,
        n_sample=1,
        valid_r=False,
        valid_l=False,
        exclusion_reason=item.exclusion_reason,
        calibration_state=observation.calibration_state,
        source_name=observation.source_name,
    )


def _artifact_from_samples(
    samples: list[HolographyDiagonalSample],
    observation: HolographyObservation,
    *,
    reference_combination: str,
    notes: tuple[str, ...],
) -> HolographyDiagonalArtifact:
    jones = []
    valid = []
    for sample in samples:
        plane = np.zeros((2, 2), dtype=np.complex128)
        if sample.valid_r:
            plane[0, 0] = sample.e_r
        if sample.valid_l:
            plane[1, 1] = sample.e_l
        jones.append(plane)
        valid.append(sample.valid)
    return HolographyDiagonalArtifact(
        samples=tuple(samples),
        jones=np.stack(jones, axis=0),
        valid=np.asarray(valid, dtype=bool),
        off_diagonal_valid=np.zeros(len(samples), dtype=bool),
        calibration_state=observation.calibration_state,
        source_name=observation.source_name,
        reference_combination=reference_combination,
        frozen=False,
        first_beam_recovery_unfrozen=True,
        notes=notes,
    )


def _thermal_scatter(members: list[_RowEstimate]) -> tuple[float, float]:
    usable_r = [item.e_r for item in members if item.weight_rr > 0.0 and np.isfinite(item.e_r)]
    usable_l = [item.e_l for item in members if item.weight_ll > 0.0 and np.isfinite(item.e_l)]
    return _complex_rms_scatter(usable_r), _complex_rms_scatter(usable_l)


def _complex_rms_scatter(values: list[complex]) -> float:
    if len(values) < 2:
        return float("nan")
    array = np.asarray(values, dtype=np.complex128)
    return float(np.sqrt(np.mean(np.abs(array - np.mean(array)) ** 2)))


def _raster_pass_by_scan(observation: HolographyObservation) -> dict[int, str]:
    scans = np.asarray(observation.block.scan_id, dtype=np.int32)
    mapping: dict[int, str] = {}
    for index, group in enumerate(scan_passes(scans), start=1):
        for scan in group:
            mapping[int(scan)] = f"pass-{index}"
    return mapping


def _role_grid(observation: HolographyObservation, shape: tuple[int, int]) -> np.ndarray:
    roles = np.full(shape, "unknown", dtype="U16")
    for index, antenna in enumerate(observation.pointing.antenna_id):
        roles[:, int(antenna)] = observation.pointing.role[:, index]
    return roles


def _is_on_axis(offset: np.ndarray, atol: float) -> bool:
    return bool(np.hypot(offset[0], offset[1]) <= atol)


def _samples_by_moving_reference_cell(
    samples: tuple[HolographyDiagonalSample, ...] | list[HolographyDiagonalSample],
) -> dict[tuple[int, int, int], list[HolographyDiagonalSample]]:
    groups: dict[tuple[int, int, int], list[HolographyDiagonalSample]] = {}
    for sample in samples:
        if not sample.valid:
            continue
        key = (
            sample.moving_antenna_id,
            sample.reference_antenna_id,
            int(round(sample.frequency_hz)),
        )
        groups.setdefault(key, []).append(sample)
    return groups


def _lookup_prediction_sample(
    groups: dict[tuple[int, int, int], list[HolographyDiagonalSample]],
    moving: int,
    reference: int,
    time_s: float,
    offset: np.ndarray,
    frequency_hz: float,
    *,
    allow_other_moving_antenna: bool = False,
    allow_other_reference: bool = False,
) -> HolographyDiagonalSample | None:
    freq_key = int(round(frequency_hz))
    candidates = groups.get((moving, reference, freq_key), [])
    exact = [
        sample
        for sample in candidates
        if np.isclose(sample.unique_time_s, time_s, rtol=0.0, atol=1.0e-6)
    ]
    if exact:
        return exact[0]
    same_cell = [
        sample
        for sample in candidates
        if np.allclose(sample.offset_lm_rad, offset, rtol=0.0, atol=1.0e-12)
    ]
    if same_cell:
        return same_cell[0]
    if allow_other_reference:
        other_refs = [
            sample
            for key, members in groups.items()
            if key[0] == moving and key[2] == freq_key
            for sample in members
            if np.allclose(sample.offset_lm_rad, offset, rtol=0.0, atol=1.0e-12)
        ]
        if other_refs:
            return other_refs[0]
    if not allow_other_moving_antenna:
        return None
    others = [
        sample
        for key, members in groups.items()
        if key[1] == reference and key[2] == freq_key
        for sample in members
        if np.allclose(sample.offset_lm_rad, offset, rtol=0.0, atol=1.0e-12)
    ]
    if others:
        return others[0]
    return None
