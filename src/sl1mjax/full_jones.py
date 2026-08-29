"""Phase 6 full-Jones acquisition contract, composition, and outer field.

Phase 6A selects the C-band artifact or generator and the convention-oracle
sample list. It does not freeze a table. Phase 6B may import complex Jones
only after that freeze. Phases 6C and 6D are fail-closed rules so a later
backend cannot double-count squint or hard-splice Jones onto the outer
Airy field. Imaging still uses the static Airy path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from functools import lru_cache
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sl1mjax.beam_conventions import (
    JONES_RECEPTOR_ORDER,
    BeamCalibrationState,
    artifact_by_id,
    beam_requires_identity_on_axis,
    require_beam_calibration_state,
)
from sl1mjax.polarization import Receptor
from sl1mjax.voltage_beam import BeamCoordinates, BeamEvaluation

FULL_JONES_PIN_CATALOG_VERSION = "phase-6-bacchus-cassbeam-2026-08-29"
FULL_JONES_PIN_SCHEMA_VERSION = 1
FULL_JONES_MODEL_ID = "full_jones_unfrozen"

_PIN_PATH = Path(__file__).with_name("data") / "vla_cband_full_jones_pin.json"


class TermPresence(StrEnum):
    """Whether a physical term is already inside a full-Jones artifact."""

    PRESENT = "present"
    ABSENT = "absent"
    UNKNOWN = "unknown"


class TransmitReceiveConvention(StrEnum):
    """Which side of the measurement equation the native Jones occupies."""

    RECEIVE = "receive"
    TRANSMIT = "transmit"


class DirectionAxisOrientation(StrEnum):
    """Native direction-cosine orientation of an imported Jones table."""

    L_EAST_M_NORTH = "l_east_m_north"


class AntennaAveraging(StrEnum):
    """Whether the artifact is array-average or antenna-specific."""

    ARRAY_AVERAGE = "array_average"
    ANTENNA_SPECIFIC = "antenna_specific"


class FullJonesOuterFieldPolicy(StrEnum):
    """What happens beyond validated full-Jones support.

    ``tapered_scalar_composite`` returns to the Phase 3 scalar composite on
    the diagonal and marks off-diagonal response unsupported. ``unsupported``
    zeros the whole Jones. ``hard_splice`` copies arbitrary complex elements
    onto Airy or Perley and is refused.
    """

    TAPERED_SCALAR_COMPOSITE = "tapered_scalar_composite"
    UNSUPPORTED = "unsupported"
    HARD_SPLICE = "hard_splice"


@dataclass(frozen=True)
class FullJonesContents:
    """Which Jones terms an artifact already contains."""

    squint: TermPresence = TermPresence.UNKNOWN
    off_diagonal_leakage: TermPresence = TermPresence.UNKNOWN
    on_axis_g: TermPresence = TermPresence.UNKNOWN
    on_axis_d: TermPresence = TermPresence.UNKNOWN
    on_axis_x: TermPresence = TermPresence.UNKNOWN
    on_axis_p: TermPresence = TermPresence.UNKNOWN

    def unknown_terms(self) -> tuple[str, ...]:
        """Return names still marked unknown."""

        return tuple(
            name
            for name, value in (
                ("squint", self.squint),
                ("off_diagonal_leakage", self.off_diagonal_leakage),
                ("on_axis_g", self.on_axis_g),
                ("on_axis_d", self.on_axis_d),
                ("on_axis_x", self.on_axis_x),
                ("on_axis_p", self.on_axis_p),
            )
            if value is TermPresence.UNKNOWN
        )


@dataclass(frozen=True)
class FullJonesReferencePin:
    """Pinned conventions for one C-band full-Jones artifact or generator.

    A pin may exist while still unfrozen. Evaluation is refused until
    ``require_frozen_full_jones_reference`` succeeds.
    """

    artifact_id: str
    generator_or_path: str
    native_quantity: str
    native_basis: str
    receptor_order: tuple[Receptor, Receptor]
    transmit_receive: TransmitReceiveConvention | None
    direction_axis_orientation: DirectionAxisOrientation | None
    frequency_support_hz: tuple[float, float] | None
    direction_support: str
    antenna_averaging: AntennaAveraging | None
    contents: FullJonesContents
    outer_field_policy: FullJonesOuterFieldPolicy
    generator_version: str | None = None
    input_checksum: str | None = None
    output_checksum: str | None = None
    frozen: bool = False
    unpinned_fields: tuple[str, ...] = field(default_factory=tuple)
    notes: str = ""

    def missing_freeze_fields(self) -> tuple[str, ...]:
        """Return fields that still block a scientific freeze."""

        missing: list[str] = []
        if self.native_quantity != "voltage_jones_2x2":
            missing.append("native_quantity")
        if self.native_basis != "circular":
            missing.append("native_basis")
        if self.receptor_order != JONES_RECEPTOR_ORDER:
            missing.append("receptor_order")
        if self.transmit_receive is not TransmitReceiveConvention.RECEIVE:
            missing.append("transmit_receive")
        if (
            self.direction_axis_orientation
            is not DirectionAxisOrientation.L_EAST_M_NORTH
        ):
            missing.append("direction_axis_orientation")
        if self.frequency_support_hz is None:
            missing.append("frequency_support_hz")
        if self.antenna_averaging is None:
            missing.append("antenna_averaging")
        if self.generator_version is None and self.input_checksum is None:
            missing.append("generator_version_or_input_checksum")
        if self.output_checksum is None:
            missing.append("output_checksum")
        if self.outer_field_policy is FullJonesOuterFieldPolicy.HARD_SPLICE:
            missing.append("outer_field_policy")
        missing.extend(f"contents.{name}" for name in self.contents.unknown_terms())
        missing.extend(self.unpinned_fields)
        return tuple(dict.fromkeys(missing))


@dataclass(frozen=True)
class FullJonesAcquisitionPlan:
    """Phase 6A route selection. Not a frozen Jones table."""

    catalog_version: str
    schema_version: int
    frozen: bool
    convention_oracle_route: str
    scientific_table_route: str
    scientific_fallback_route: str
    refused_as_cband: tuple[str, ...]
    notes: str
    bagofwinds_compact_products_found: bool
    cassbeam_generic_template_is_cband: bool
    cassbeam_runtime_in_imager: bool
    cassbeam_host_package: str | None
    holography_request_frequencies_mhz: tuple[int, ...]


@dataclass(frozen=True)
class OrientationOracleSampleSpec:
    """Compact correlation-aware samples that close Phase 5 conventions.

    The list is the acquisition request. It does not invent Jones values.
    """

    frequencies_hz: tuple[float, ...]
    parallactic_angles_rad: tuple[float, ...]
    direction_roles: tuple[str, ...]
    correlations: tuple[str, ...]
    sky_stokes: tuple[str, ...]
    required_closures: tuple[str, ...]


@dataclass(frozen=True)
class OuterFieldComposition:
    """Jones after the Phase 6D outer-field rule."""

    jones: NDArray[np.complex128]
    valid: NDArray[np.bool_]
    off_diagonal_valid: NDArray[np.bool_]


@lru_cache(maxsize=1)
def load_full_jones_acquisition_plan() -> FullJonesAcquisitionPlan:
    """Load the Phase 6A route selection."""

    payload = json.loads(_PIN_PATH.read_text(encoding="utf-8"))
    if payload.get("catalog_version") != FULL_JONES_PIN_CATALOG_VERSION:
        raise ValueError("unexpected full-Jones pin catalog version")
    if int(payload.get("schema_version", -1)) != FULL_JONES_PIN_SCHEMA_VERSION:
        raise ValueError("unexpected full-Jones pin schema version")
    bagofwinds = payload["bagofwinds_search"]
    cassbeam = payload["cassbeam_generator"]
    holography = payload["holography_request"]
    return FullJonesAcquisitionPlan(
        catalog_version=str(payload["catalog_version"]),
        schema_version=int(payload["schema_version"]),
        frozen=bool(payload["frozen"]),
        convention_oracle_route=str(payload["convention_oracle_route"]),
        scientific_table_route=str(payload["scientific_table_route"]),
        scientific_fallback_route=str(payload["scientific_fallback_route"]),
        refused_as_cband=tuple(payload["refused_as_cband"]),
        notes=str(payload["notes"]),
        bagofwinds_compact_products_found=bool(
            bagofwinds["compact_products_found"]
        ),
        cassbeam_generic_template_is_cband=bool(
            cassbeam["generic_template_is_cband"]
        ),
        cassbeam_runtime_in_imager=bool(cassbeam["runtime_in_imager"]),
        cassbeam_host_package=(
            str(cassbeam["host_package"]) if cassbeam.get("host_package") else None
        ),
        holography_request_frequencies_mhz=tuple(
            int(value) for value in holography["frequencies_mhz"]
        ),
    )


def full_jones_reference_is_frozen() -> bool:
    """True only after a C-band artifact and its conventions are pinned."""

    return load_full_jones_acquisition_plan().frozen


def orientation_oracle_sample_spec() -> OrientationOracleSampleSpec:
    """Samples a correlation-aware CASSBEAM or CASA export must supply."""

    return OrientationOracleSampleSpec(
        frequencies_hz=(4.564e9, 4.692e9),
        parallactic_angles_rad=(0.0, 0.5 * np.pi),
        direction_roles=(
            "beam_centre",
            "half_power",
            "squint_positive_l",
            "squint_negative_l",
            "first_sidelobe",
        ),
        correlations=("RR", "RL", "LR", "LL"),
        sky_stokes=("I", "Q", "U", "V"),
        required_closures=(
            "R/L sign",
            "feed-frame position angle",
            "parallactic rotation",
            "squint separation",
        ),
    )


def unfrozen_full_jones_pin(artifact_id: str) -> FullJonesReferencePin:
    """Return the inventoried but unfrozen pin for an identified C-band route."""

    plan = load_full_jones_acquisition_plan()
    if artifact_id in plan.refused_as_cband:
        raise ValueError(
            f"{artifact_id} is not a C-band full-Jones reference"
        )
    artifact = artifact_by_id(artifact_id)
    scalar_quantities = {
        "stokes_i_power",
        "stokes_i_fwhm",
        "diagonal_voltage_rr_ll",
        "receptor_power_rr_ll",
    }
    if artifact.frozen_reference:
        raise ValueError(f"{artifact_id} is already a frozen scalar reference")
    if artifact.usable_for_cband is False or artifact.quantity in scalar_quantities:
        raise ValueError(
            f"{artifact_id} cannot be promoted to a measured full-Jones beam"
        )
    return FullJonesReferencePin(
        artifact_id=artifact.artifact_id,
        generator_or_path=artifact.epoch,
        native_quantity=artifact.quantity,
        native_basis="unspecified",
        receptor_order=JONES_RECEPTOR_ORDER,
        transmit_receive=None,
        direction_axis_orientation=None,
        frequency_support_hz=artifact.frequency_hz,
        direction_support=artifact.direction_support,
        antenna_averaging=None,
        contents=FullJonesContents(),
        outer_field_policy=FullJonesOuterFieldPolicy.TAPERED_SCALAR_COMPOSITE,
        frozen=False,
        unpinned_fields=artifact.unpinned_requirements,
        notes=artifact.notes,
    )


def require_frozen_full_jones_reference(
    pin: FullJonesReferencePin | None = None,
) -> FullJonesReferencePin:
    """Accept a frozen pin or refuse to evaluate a full-Jones backend."""

    selected = pin if pin is not None else unfrozen_full_jones_pin(
        load_full_jones_acquisition_plan().scientific_table_route
    )
    plan = load_full_jones_acquisition_plan()
    if selected.artifact_id in plan.refused_as_cband:
        raise ValueError(
            f"{selected.artifact_id} is not a C-band full-Jones reference"
        )
    missing = selected.missing_freeze_fields()
    if (not selected.frozen) or missing or (not plan.frozen):
        detail = ", ".join(missing) if missing else "catalog is not frozen"
        raise ValueError(
            "full-Jones reference is not frozen; refuse to evaluate "
            f"({detail})"
        )
    return selected


def refuse_analytic_squint_composition(contents: FullJonesContents) -> None:
    """Refuse ``DiagonalSquintVoltageBeam`` when the artifact already has squint."""

    if contents.squint is not TermPresence.ABSENT:
        raise ValueError(
            "full-Jones artifact must replace analytic squint; do not "
            "multiply by DiagonalSquintVoltageBeam when squint is "
            f"{contents.squint.value}"
        )


def refuse_on_axis_double_count(
    contents: FullJonesContents,
    calibration_state: BeamCalibrationState | str,
) -> None:
    """Refuse on-axis D/X/P/G that would be applied twice."""

    if not beam_requires_identity_on_axis(calibration_state):
        return
    extras = tuple(
        name
        for name, value in (
            ("G", contents.on_axis_g),
            ("D", contents.on_axis_d),
            ("X", contents.on_axis_x),
            ("P", contents.on_axis_p),
        )
        if value is not TermPresence.ABSENT
    )
    if extras:
        state = require_beam_calibration_state(calibration_state)
        raise ValueError(
            f"{state.value} requires E(0)=I; artifact still contains "
            f"on-axis {'/'.join(extras)}"
        )


def apply_full_jones_outer_field(
    full_jones: ArrayLike,
    full_valid: ArrayLike,
    scalar_jones: ArrayLike,
    scalar_valid: ArrayLike,
    *,
    policy: FullJonesOuterFieldPolicy,
    taper_weight: ArrayLike | None = None,
) -> OuterFieldComposition:
    """Apply the Phase 6D outer-field rule.

    Off-diagonal elements stay unsupported outside the artifact. The
    default taper is a sharp cutoff: weight 1 on full-Jones support and
    0 elsewhere. A supplied weight must lie in ``[0, 1]``.
    """

    if policy is FullJonesOuterFieldPolicy.HARD_SPLICE:
        raise ValueError(
            "do not hard-splice arbitrary complex Jones onto Airy or Perley"
        )
    jones = np.asarray(full_jones, dtype=np.complex128)
    valid = np.asarray(full_valid, dtype=bool)
    scalar = np.asarray(scalar_jones, dtype=np.complex128)
    scalar_ok = np.asarray(scalar_valid, dtype=bool)
    if jones.shape != scalar.shape or jones.ndim != 5 or jones.shape[-2:] != (2, 2):
        raise ValueError(
            "full and scalar Jones must have shape "
            "(antenna, direction, channel, 2, 2)"
        )
    if valid.shape != jones.shape[:3] or scalar_ok.shape != jones.shape[:3]:
        raise ValueError("validity must match Jones (antenna, direction, channel)")
    if policy is FullJonesOuterFieldPolicy.UNSUPPORTED:
        out = np.where(valid[..., None, None], jones, 0.0)
        return OuterFieldComposition(
            jones=np.asarray(out, dtype=np.complex128),
            valid=valid,
            off_diagonal_valid=valid,
        )
    if policy is not FullJonesOuterFieldPolicy.TAPERED_SCALAR_COMPOSITE:
        raise ValueError(f"unknown outer-field policy {policy!r}")
    if taper_weight is None:
        weight = valid.astype(np.float64)
    else:
        weight = np.asarray(taper_weight, dtype=np.float64)
        if weight.shape != valid.shape:
            raise ValueError("taper_weight must match (antenna, direction, channel)")
        if np.any(~np.isfinite(weight)) or np.any(weight < 0.0) or np.any(weight > 1.0):
            raise ValueError("taper_weight must be finite and in [0, 1]")
    blended = weight[..., None, None] * jones + (1.0 - weight[..., None, None]) * scalar
    blended = np.array(blended, dtype=np.complex128, copy=True)
    outside = ~valid
    blended[outside, 0, 1] = 0.0
    blended[outside, 1, 0] = 0.0
    usable = valid | scalar_ok
    blended = np.where(usable[..., None, None], blended, 0.0)
    return OuterFieldComposition(
        jones=np.asarray(blended, dtype=np.complex128),
        valid=np.asarray(usable, dtype=bool),
        off_diagonal_valid=valid,
    )


@dataclass(frozen=True)
class FullJonesVoltageBeam:
    """Fail-closed Phase 6B evaluator.

    The class exists so composition and outer-field rules can be tested
    against the public ``VoltageBeamModel`` contract. It refuses to
    interpolate or invent Jones until the pin is frozen.
    """

    pin: FullJonesReferencePin
    model_id: str = FULL_JONES_MODEL_ID

    def evaluate(
        self,
        coordinates: BeamCoordinates,
        *,
        calibration_state: BeamCalibrationState | str,
    ) -> BeamEvaluation:
        del coordinates
        require_beam_calibration_state(calibration_state)
        require_frozen_full_jones_reference(self.pin)
        raise RuntimeError("frozen full-Jones evaluation is not implemented")
