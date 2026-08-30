"""Beam-aware imaging baselines and the finite-pixel sky contract.

Phase 0 pins the completed voltage-beam evaluator and the current
point-centre 3C391 transfer so later work cannot silently upgrade
scientific status. Phase 1 keeps finite component shapes when the
voltage path consumes a sealed sky. The old point-centre operator
remains a named diagnostic mode.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass, field, replace
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from sl1mjax.beam_conventions import (
    CIRCULAR_P_JONES,
    JONES_RECEPTOR_ORDER,
    ON_AXIS_DI_JONES_ORDER,
    analytic_squint_is_evidence_grade,
    antenna_frame_polarization_is_physically_verified,
)
from sl1mjax.casa_awp2_oracle import (
    polarization_oracle_is_frozen,
    power_oracle_is_frozen,
)
from sl1mjax.cassbeam_beam import (
    CASA_PARANG_PARALLACTIC_BASIS,
    MAX_NEAREST_NODE_SEPARATION_HZ,
    UNCALIBRATED_PARALLACTIC_BASIS,
    diagonal_copolar_is_casa_accepted,
    load_cassbeam_cband_artifact,
    voltage_beam_for_mode,
)
from sl1mjax.composite import (
    MosaicPointComponent,
    MosaicQuadtreeComponent,
    MosaicSkyComponent,
)
from sl1mjax.coordinates import radec_to_lmn
from sl1mjax.full_jones import full_jones_reference_is_frozen
from sl1mjax.quadtree import (
    QuadtreeGrid,
    QuadtreeLeaf,
    QuadtreeTopology,
    quadtree_sky_from_regular_grid,
)

_MANIFEST_PATH = Path(__file__).with_name("data") / "beam_aware_imaging_manifest.json"
_TRANSFER_REPORT_PATH = (
    Path(__file__).with_name("data") / "3c391_voltage_beam_transfer_20260830.json"
)
BEAM_AWARE_IMAGING_MANIFEST_ID = "beam-aware-imaging-phase-0-2026-08-30"
BEAM_AWARE_IMAGING_SCHEMA_VERSION = 1
FIXED_SKY_TRANSFER_REPORT_ID = "3c391_voltage_beam_transfer_20260830"
LIVE_UNFROZEN_GATES = {
    "full_jones_reference_is_frozen": full_jones_reference_is_frozen,
    "diagonal_copolar_is_casa_accepted": diagonal_copolar_is_casa_accepted,
    "power_oracle_is_frozen": power_oracle_is_frozen,
    "polarization_oracle_is_frozen": polarization_oracle_is_frozen,
}
SCIENCE_STATUS_TO_GATE = {
    "full_jones_frozen": "full_jones_reference_is_frozen",
    "diagonal_copolar_casa_accepted": "diagonal_copolar_is_casa_accepted",
    "casa_awp2_stage1_frozen": "power_oracle_is_frozen",
    "casa_awp2_stage2_implemented": "polarization_oracle_is_frozen",
}
EXPECTED_STRING_GATES = {
    "casa_awp2_stage2": "not implemented until Stage 1 is frozen",
    "perley_interpolated_frequency_policy": "declared and refused",
    "analytic_airy_0.06_fwhm_half_offset": "unused and not evidence-grade",
}
ARCSEC_RAD = np.deg2rad(1.0 / 3600.0)
WIDTH_ARCSEC_TOLERANCE = 1.0e-6
DEFAULT_FAMILY_BY_NAME = {
    "central": "central_tree",
    "coarse": "coarse_field",
    "catalogue": "catalogue",
    "catalog": "catalogue",
    "outer": "outer_guard",
    "guard": "outer_guard",
    "outer_guard": "outer_guard",
}


class ComponentFamily(StrEnum):
    """Scientific role of a fitted sky component."""

    CENTRAL_TREE = "central_tree"
    COARSE_FIELD = "coarse_field"
    OUTER_GUARD = "outer_guard"
    CATALOGUE = "catalogue"


class SkyBasisType(StrEnum):
    """Declared within-component brightness distribution."""

    UNIFORM_SQUARE = "uniform_square"
    DELTA = "delta"
    GAUSSIAN = "gaussian"


class VoltageIntegrationMode(StrEnum):
    """How the voltage operator uses a finite sky component."""

    POINT_CENTRE = "point_centre"
    ANALYTIC_SQUARE = "analytic_square"
    SUBCELL_2X2 = "subcell_2x2"
    SUBCELL_4X4 = "subcell_4x4"


class QuadtreeOverlapPolicy(StrEnum):
    """Whether a family may keep a parent and its children active."""

    PREFIX_FREE = "prefix_free"
    OVERLAPPING_HISTORICAL = "overlapping_historical"


def overlap_policy_for_family(family: ComponentFamily | str) -> QuadtreeOverlapPolicy:
    """Return whether a family is a prefix-free tree or a historical overlay."""

    selected = ComponentFamily(family)
    if selected is ComponentFamily.COARSE_FIELD:
        return QuadtreeOverlapPolicy.OVERLAPPING_HISTORICAL
    return QuadtreeOverlapPolicy.PREFIX_FREE


@dataclass(frozen=True)
class SkyComponent:
    """One fitted sky atom with its declared basis and integrated flux."""

    component_id: str
    family: ComponentFamily
    basis_type: SkyBasisType
    l_rad: float
    m_rad: float
    stokes_i_jy: float
    width_rad: float = 0.0
    level: int | None = None
    iy: int | None = None
    ix: int | None = None
    parent_id: str | None = None
    active: bool = True
    splitting_permitted: bool = False
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "family", ComponentFamily(self.family))
        object.__setattr__(self, "basis_type", SkyBasisType(self.basis_type))
        if not self.component_id.strip():
            raise ValueError("component_id must be non-empty")
        if not np.isfinite(self.l_rad) or not np.isfinite(self.m_rad):
            raise ValueError("component centres must be finite")
        if not np.isfinite(self.stokes_i_jy):
            raise ValueError("stokes_i_jy must be finite")
        if self.basis_type is SkyBasisType.DELTA:
            if self.width_rad != 0.0:
                raise ValueError("delta components must have width_rad == 0")
            return
        if not np.isfinite(self.width_rad) or self.width_rad <= 0.0:
            raise ValueError(f"{self.basis_type.value} components require a positive width_rad")


@dataclass(frozen=True)
class SkyConversionReport:
    """Bookkeeping for checkpoint or mosaic-component conversion."""

    input_atom_count: int
    kept_atom_count: int
    dropped_zero_flux_count: int
    missing_component_names: tuple[str, ...]
    total_input_flux_jy: float
    total_kept_flux_jy: float
    flux_by_family: dict[str, float]
    count_by_family: dict[str, int]
    count_by_width_arcsec: dict[str, int]
    discarded_finite_widths: bool


@dataclass(frozen=True)
class SkyComponentTable:
    """Pointing-independent fitted sky used by the voltage path."""

    components: tuple[SkyComponent, ...]
    mosaic_phase_centre_rad: tuple[float, float]
    report: SkyConversionReport
    source: str = "mosaic_components"

    def __post_init__(self) -> None:
        if not self.components:
            raise ValueError("sky table must contain at least one component")
        identifiers = [component.component_id for component in self.components]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("component identifiers must be unique")
        _reject_prefix_conflicts(self.components)

    def active(self) -> tuple[SkyComponent, ...]:
        return tuple(component for component in self.components if component.active)

    def flux_jy(self, *, include_inactive: bool = False) -> float:
        selected = self.components if include_inactive else self.active()
        return float(sum(component.stokes_i_jy for component in selected))


@dataclass(frozen=True)
class PointCentreAtoms:
    """Named diagnostic packing for the current delta voltage operator.

    Widths and basis types are retained so later finite-pixel modes can
    consume the same table. The operator still uses a delta kernel.
    """

    component_id: tuple[str, ...]
    l_rad: NDArray[np.float64]
    m_rad: NDArray[np.float64]
    stokes_i_jy: NDArray[np.float64]
    width_rad: NDArray[np.float64]
    basis_type: tuple[str, ...]
    parent_index: NDArray[np.int32]
    mode: VoltageIntegrationMode = VoltageIntegrationMode.POINT_CENTRE
    dropped_zero_flux_count: int = 0

    @property
    def flux(self) -> NDArray[np.float64]:
        return self.stokes_i_jy


@lru_cache(maxsize=1)
def load_beam_aware_imaging_manifest() -> dict[str, Any]:
    """Load the Phase 0 development manifest."""

    payload = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("beam-aware imaging manifest must be a JSON object")
    if payload.get("manifest_id") != BEAM_AWARE_IMAGING_MANIFEST_ID:
        raise ValueError("unexpected beam-aware imaging manifest id")
    if int(payload.get("schema_version", -1)) != BEAM_AWARE_IMAGING_SCHEMA_VERSION:
        raise ValueError("unexpected beam-aware imaging manifest schema")
    return dict(payload)


@lru_cache(maxsize=1)
def load_fixed_sky_transfer_report() -> dict[str, Any]:
    """Load the compact 2026-08-30 point-centre transfer extract."""

    payload = json.loads(_TRANSFER_REPORT_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("fixed-sky transfer report must be a JSON object")
    if payload.get("report_id") != FIXED_SKY_TRANSFER_REPORT_ID:
        raise ValueError("unexpected fixed-sky transfer report id")
    if int(payload.get("schema_version", -1)) != 1:
        raise ValueError("unexpected fixed-sky transfer report schema")
    return dict(payload)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_phase0_baseline(manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    """Reproduce the Phase 0 scientific-status and artifact gates."""

    payload = manifest if manifest is not None else load_beam_aware_imaging_manifest()
    status = payload["scientific_status"]
    conventions = payload["conventions"]
    cassbeam = payload["cassbeam"]
    gates = payload["unfrozen_or_unimplemented_gates"]
    errors: list[str] = []

    live_gates = {name: bool(function()) for name, function in LIVE_UNFROZEN_GATES.items()}
    for name, value in live_gates.items():
        if name not in gates:
            errors.append(f"unfrozen-gate pin {name} is missing")
        elif bool(gates[name]) != value:
            errors.append(f"{name} is {value}; unimplemented-gate pin is {gates[name]}")
    for science_key, gate_key in SCIENCE_STATUS_TO_GATE.items():
        if bool(status[science_key]) != live_gates[gate_key]:
            errors.append(
                f"{science_key} is {status[science_key]}; live {gate_key} is {live_gates[gate_key]}"
            )
        if bool(status[science_key]) != bool(gates.get(gate_key)):
            errors.append(
                f"{science_key}={status[science_key]} contradicts {gate_key}={gates.get(gate_key)}"
            )
    for name, expected in EXPECTED_STRING_GATES.items():
        if gates.get(name) != expected:
            errors.append(f"{name} is {gates.get(name)!r}; expected {expected!r}")
    live = {
        **live_gates,
        **{key: live_gates[gate] for key, gate in SCIENCE_STATUS_TO_GATE.items()},
        "analytic_squint_evidence_grade": analytic_squint_is_evidence_grade(),
        "antenna_frame_polarization_verified": (
            antenna_frame_polarization_is_physically_verified()
        ),
    }
    for key in (
        "analytic_squint_evidence_grade",
        "antenna_frame_polarization_verified",
    ):
        if bool(status[key]) != bool(live[key]):
            errors.append(f"{key} is {live[key]}; manifest pins {status[key]}")

    if conventions["on_axis_di_jones_order"] != ON_AXIS_DI_JONES_ORDER:
        errors.append("on-axis DI Jones order drifted")
    if conventions["circular_p_jones"] != CIRCULAR_P_JONES:
        errors.append("circular P Jones convention drifted")
    if tuple(conventions["jones_receptor_order"]) != tuple(
        receptor.value for receptor in JONES_RECEPTOR_ORDER
    ):
        errors.append("Jones receptor order drifted")
    if conventions["casa_parang_parallactic_basis"] != CASA_PARANG_PARALLACTIC_BASIS:
        errors.append("casa_parang_true parallactic basis drifted")
    if conventions["uncalibrated_parallactic_basis"] != UNCALIBRATED_PARALLACTIC_BASIS:
        errors.append("uncalibrated parallactic basis drifted")

    artifact = load_cassbeam_cband_artifact()
    if artifact.pin.frozen:
        errors.append("CASSBEAM pin is frozen; Phase 0 forbids a silent upgrade")
    if bool(artifact.manifest.get("frozen")):
        errors.append("CASSBEAM packaged manifest is frozen")
    if float(cassbeam["max_nearest_node_separation_hz"]) != MAX_NEAREST_NODE_SEPARATION_HZ:
        errors.append("CASSBEAM nearest-node pad drifted")
    table = artifact.tables[0]
    if int(table.l_origin_index) != int(cassbeam["l_origin_index"]):
        errors.append("CASSBEAM l origin drifted")
    if int(table.m_origin_index) != int(cassbeam["m_origin_index"]):
        errors.append("CASSBEAM m origin drifted")
    packaged_nodes = artifact.manifest.get("frequencies_mhz", ())
    if not isinstance(packaged_nodes, (list, tuple)):
        errors.append("CASSBEAM packaged frequency nodes are not a sequence")
    elif [int(value) for value in packaged_nodes] != [
        int(value) for value in cassbeam["frequencies_mhz"]
    ]:
        errors.append("CASSBEAM frequency nodes drifted")

    factory_refused = False
    try:
        voltage_beam_for_mode("full_jones")
    except ValueError as error:
        factory_refused = "not frozen" in str(error)
    if not factory_refused:
        errors.append("production full-Jones factory did not refuse the unfrozen artifact")
    if not status["full_jones_factory_refuses_unfrozen"]:
        errors.append("manifest must keep full_jones_factory_refuses_unfrozen true")
    if status["transfer_ranking_is_scientific_beam_selection"]:
        errors.append("fixed-sky ranking must not be marked as scientific beam selection")
    if not status["transfer_operator_discarded_finite_widths"]:
        errors.append("Phase 0 must record that the transfer operator discarded widths")
    if status["default_imaging_beam"] != "static_airy":
        errors.append("default imaging beam must remain static_airy")
    try:
        reproduce_fixed_sky_transfer_totals(payload)
    except ValueError as error:
        errors.append(str(error))

    if errors:
        raise ValueError("Phase 0 baseline gate failed: " + "; ".join(errors))
    return {
        "accepted": True,
        "manifest_id": payload["manifest_id"],
        "scientific_status": dict(status),
        "live_status": live,
        "full_jones_factory_refused": True,
    }


def interpret_fixed_sky_transfer(summary: dict[str, Any]) -> dict[str, Any]:
    """Reproduce the 2026-08-30 point-centre transfer interpretation."""

    beam_names = (
        "static_scalar",
        "streamed_scalar",
        "diagonal_copolar",
        "full_jones_unfrozen",
    )
    totals: dict[str, list[float]] = {name: [] for name in beam_names}
    rr: dict[str, list[float]] = {name: [] for name in beam_names}
    ll: dict[str, list[float]] = {name: [] for name in beam_names}
    for pointing in summary.get("pointings", {}).values():
        for name, scores in pointing.get("beams", {}).items():
            if name not in totals:
                continue
            totals[name].append(float(scores["total"]))
            correlations = scores.get("correlations", {})
            rr_loss = correlations.get("RR", {}).get("held_out_loss")
            ll_loss = correlations.get("LL", {}).get("held_out_loss")
            if rr_loss is not None:
                rr[name].append(float(rr_loss))
            if ll_loss is not None:
                ll[name].append(float(ll_loss))

    def _mean(values: list[float]) -> float:
        return float(np.mean(values)) if values else float("nan")

    means = {name: _mean(values) for name, values in totals.items() if values}
    ranking = sorted(means, key=lambda name: means[name])
    airy = means.get("static_scalar")
    composite = means.get("streamed_scalar")
    diagonal = means.get("diagonal_copolar")
    full = means.get("full_jones_unfrozen")
    return {
        "mean_held_out_loss": means,
        "mean_rr": {name: _mean(values) for name, values in rr.items() if values},
        "mean_ll": {name: _mean(values) for name, values in ll.items() if values},
        "ranking_best_first": ranking,
        "scalar_shape_matters": bool(
            airy is not None and composite is not None and composite < airy
        ),
        "squint_or_rl_structure_matters": bool(
            composite is not None and diagonal is not None and diagonal < composite
        ),
        "leakage_matters": bool(
            diagonal is not None and full is not None and full < diagonal
        ),
        "no_detailed_beam_beats_airy": bool(
            airy is not None
            and all(
                means.get(name, airy) >= airy
                for name in ("streamed_scalar", "diagonal_copolar", "full_jones_unfrozen")
                if name in means
            )
        ),
        "cross_hand_in_data": False,
        "do_not_freeze_full_jones": True,
    }


def compact_transfer_report_as_summary(report: dict[str, Any]) -> dict[str, Any]:
    """Rebuild the transfer-script summary shape from the compact extract."""

    pointings: dict[str, Any] = {}
    for name, payload in report["pointings"].items():
        beams = {}
        for beam, total in payload["totals"].items():
            beams[beam] = {
                "total": float(total),
                "correlations": {
                    "RR": {"held_out_loss": float(payload["rr"][beam]), "in_data": True},
                    "LL": {"held_out_loss": float(payload["ll"][beam]), "in_data": True},
                    "RL": {"held_out_loss": None, "in_data": False},
                    "LR": {"held_out_loss": None, "in_data": False},
                },
            }
        pointings[name] = {
            "radius_arcmin": float(payload["radius_arcmin"]),
            "leakage_atom_fraction": float(payload["leakage_atom_fraction"]),
            "leakage_flux_fraction": float(payload["leakage_flux_fraction"]),
            "beams": beams,
        }
    return {"pointings": pointings}


def reproduce_fixed_sky_transfer_totals(
    manifest: dict[str, Any] | None = None,
    report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Reproduce the 2026-08-30 interpretation from the pinned compact report."""

    payload = manifest if manifest is not None else load_beam_aware_imaging_manifest()
    pinned = report if report is not None else load_fixed_sky_transfer_report()
    transfer = payload["fixed_sky_transfer"]
    digest = _sha256_file(_TRANSFER_REPORT_PATH)
    if digest != transfer["report_sha256"]:
        raise ValueError("compact transfer report hash drifted from the Phase 0 pin")
    if pinned["provenance"]["summary_sha256"] != transfer["summary_sha256"]:
        raise ValueError("original summary hash drifted from the Phase 0 pin")
    interpretation = interpret_fixed_sky_transfer(compact_transfer_report_as_summary(pinned))
    expected = pinned["interpretation"]
    for key, value in expected.items():
        actual = interpretation[key]
        if isinstance(value, dict):
            if set(actual) != set(value):
                raise ValueError(
                    f"fixed-sky {key} keys are {sorted(actual)}; "
                    f"compact report pins {sorted(value)}"
                )
            for name, pinned_value in value.items():
                if abs(float(actual[name]) - float(pinned_value)) > 1.0e-9 * max(
                    1.0, abs(float(pinned_value))
                ):
                    raise ValueError(
                        f"fixed-sky {key}[{name}] is {actual[name]!r}; "
                        f"compact report pins {pinned_value!r}"
                    )
        elif actual != value:
            raise ValueError(
                f"fixed-sky {key} is {actual!r}; compact report pins {value!r}"
            )
    means = interpretation["mean_held_out_loss"]
    leakage_delta = float(means["diagonal_copolar"]) - float(means["full_jones_unfrozen"])
    if leakage_delta <= 0.0:
        raise ValueError("pinned full-Jones mean is not below diagonal")
    if abs(leakage_delta - 0.1315208193) > 1.0e-9:
        raise ValueError(f"full-Jones versus diagonal delta drifted: {leakage_delta}")
    if not interpretation["leakage_matters"]:
        raise ValueError("original leakage_matters flag must remain true")
    return interpretation


def family_from_component_name(
    name: str,
    *,
    family_by_name: dict[str, str] | None = None,
) -> ComponentFamily:
    """Map a mosaic component name onto a sky-table family."""

    mapping = {**DEFAULT_FAMILY_BY_NAME, **(family_by_name or {})}
    try:
        return ComponentFamily(mapping[name])
    except KeyError as error:
        raise ValueError(
            f"unknown mosaic component name {name!r}; pass family_by_name"
        ) from error


def leaf_component_id(
    family: ComponentFamily | str,
    leaf: QuadtreeLeaf,
    *,
    source: str,
) -> str:
    """Stable identifier for one quadtree leaf inside one source dictionary."""

    if not source.strip():
        raise ValueError("source component name must be non-empty")
    return f"{ComponentFamily(family).value}:{source}:{leaf.level}:{leaf.iy}:{leaf.ix}"


def delta_component_id(family: ComponentFamily | str, source: str, index: int) -> str:
    """Stable identifier for one catalogue or other delta atom."""

    if not source.strip():
        raise ValueError("source component name must be non-empty")
    return f"{ComponentFamily(family).value}:{source}:delta:{index}"


def width_arcsec(width_rad: float) -> float:
    """Convert a square width to arcseconds."""

    return float(width_rad / ARCSEC_RAD)


def width_arcsec_label(width_rad: float) -> str:
    """Bucket a width onto the documented 4/8/16/60 arcsec scales when it matches."""

    if width_rad == 0.0:
        return "delta"
    arcsec = width_arcsec(width_rad)
    for candidate in (4.0, 8.0, 16.0, 60.0, 64.0):
        if abs(arcsec - candidate) <= WIDTH_ARCSEC_TOLERANCE:
            return f"{candidate:g}"
    return f"{arcsec:.6g}"


def sky_table_from_mosaic_components(
    components: tuple[MosaicSkyComponent, ...],
    *,
    mosaic_phase_centre_rad: tuple[float, float] = (0.0, 0.0),
    family_by_name: dict[str, str] | None = None,
    include_zero_flux: bool = True,
    source: str = "mosaic_components",
    expected_component_names: tuple[str, ...] = (),
) -> SkyComponentTable:
    """Convert sealed mosaic dictionaries into the basis-preserving table."""

    if not components:
        raise ValueError("components must contain at least one sky component")
    present = {component.name for component in components}
    missing = tuple(name for name in expected_component_names if name not in present)
    converted: list[SkyComponent] = []
    input_count = 0
    input_flux = 0.0
    dropped_zeros = 0
    for component in components:
        family = family_from_component_name(component.name, family_by_name=family_by_name)
        atoms = _atoms_from_mosaic_component(component, family)
        for atom in atoms:
            input_count += 1
            input_flux += atom.stokes_i_jy
            if not include_zero_flux and atom.stokes_i_jy <= 0.0:
                dropped_zeros += 1
                continue
            converted.append(atom)
    table = SkyComponentTable(
        components=tuple(converted),
        mosaic_phase_centre_rad=_phase_centre(mosaic_phase_centre_rad),
        report=_conversion_report(
            converted,
            input_atom_count=input_count,
            dropped_zero_flux_count=dropped_zeros,
            missing_component_names=missing,
            total_input_flux_jy=input_flux,
            discarded_finite_widths=False,
        ),
        source=source,
    )
    if include_zero_flux and abs(table.report.total_kept_flux_jy - input_flux) > 1e-12:
        raise ValueError("conversion changed total intrinsic flux")
    return table


def sky_table_to_records(table: SkyComponentTable) -> list[dict[str, Any]]:
    """Serialize table rows for round-trip tests and manifests."""

    records = []
    for component in table.components:
        records.append(
            {
                "component_id": component.component_id,
                "family": component.family.value,
                "basis_type": component.basis_type.value,
                "l_rad": component.l_rad,
                "m_rad": component.m_rad,
                "stokes_i_jy": component.stokes_i_jy,
                "width_rad": component.width_rad,
                "level": component.level,
                "iy": component.iy,
                "ix": component.ix,
                "parent_id": component.parent_id,
                "active": component.active,
                "splitting_permitted": component.splitting_permitted,
                "provenance": dict(component.provenance),
            }
        )
    return records


def sky_table_from_records(
    records: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    *,
    mosaic_phase_centre_rad: tuple[float, float] = (0.0, 0.0),
    source: str = "records",
) -> SkyComponentTable:
    """Rebuild a table from :func:`sky_table_to_records` output."""

    components = tuple(
        SkyComponent(
            component_id=str(row["component_id"]),
            family=ComponentFamily(row["family"]),
            basis_type=SkyBasisType(row["basis_type"]),
            l_rad=float(row["l_rad"]),
            m_rad=float(row["m_rad"]),
            stokes_i_jy=float(row["stokes_i_jy"]),
            width_rad=float(row.get("width_rad", 0.0)),
            level=None if row.get("level") is None else int(row["level"]),
            iy=None if row.get("iy") is None else int(row["iy"]),
            ix=None if row.get("ix") is None else int(row["ix"]),
            parent_id=None if row.get("parent_id") in (None, "") else str(row["parent_id"]),
            active=bool(row.get("active", True)),
            splitting_permitted=bool(row.get("splitting_permitted", False)),
            provenance=dict(row.get("provenance") or {}),
        )
        for row in records
    )
    return SkyComponentTable(
        components=components,
        mosaic_phase_centre_rad=_phase_centre(mosaic_phase_centre_rad),
        report=_conversion_report(
            components,
            input_atom_count=len(components),
            dropped_zero_flux_count=0,
            missing_component_names=(),
            total_input_flux_jy=float(sum(item.stokes_i_jy for item in components)),
            discarded_finite_widths=False,
        ),
        source=source,
    )


def point_centre_atoms(
    table: SkyComponentTable,
    *,
    include_zero_flux: bool = False,
    include_inactive: bool = False,
) -> PointCentreAtoms:
    """Pack the table for the named point-centre voltage diagnostic.

    The baseline contract keeps atoms with ``stokes_i_jy > 0``. Exact zeros
    and negative fluxes are omitted unless ``include_zero_flux`` is true.
    """

    selected = []
    dropped_zeros = 0
    for component in table.components:
        if not include_inactive and not component.active:
            continue
        if not include_zero_flux and component.stokes_i_jy <= 0.0:
            dropped_zeros += 1
            continue
        selected.append(component)
    if not selected:
        raise ValueError("frozen sky has no positive-flux atoms")
    return PointCentreAtoms(
        component_id=tuple(component.component_id for component in selected),
        l_rad=np.asarray([component.l_rad for component in selected], dtype=np.float64),
        m_rad=np.asarray([component.m_rad for component in selected], dtype=np.float64),
        stokes_i_jy=np.asarray(
            [component.stokes_i_jy for component in selected], dtype=np.float64
        ),
        width_rad=np.asarray([component.width_rad for component in selected], dtype=np.float64),
        basis_type=tuple(component.basis_type.value for component in selected),
        parent_index=np.arange(len(selected), dtype=np.int32),
        dropped_zero_flux_count=dropped_zeros,
    )


def prepare_voltage_sky(
    table: SkyComponentTable,
    *,
    mode: VoltageIntegrationMode | str = VoltageIntegrationMode.POINT_CENTRE,
    include_zero_flux: bool = False,
) -> PointCentreAtoms:
    """Consume a sky table without dropping finite component shapes."""

    selected = VoltageIntegrationMode(mode)
    if selected is VoltageIntegrationMode.POINT_CENTRE:
        return point_centre_atoms(table, include_zero_flux=include_zero_flux)
    raise ValueError(f"voltage integration mode {selected.value!r} is not implemented")


def mosaic_components_from_checkpoint(
    checkpoint: Path,
    protocol: dict[str, Any] | Path,
    mosaic_phase_centre_rad: tuple[float, float],
    *,
    search_roots: tuple[Path, ...] = (),
) -> tuple[MosaicSkyComponent, ...]:
    """Load the sealed mosaic dictionaries from a composite checkpoint."""

    if isinstance(protocol, Path):
        payload = json.loads(protocol.read_text(encoding="utf-8"))
        roots = (protocol.resolve().parent.parent.parent, *search_roots)
    else:
        payload = dict(protocol)
        roots = search_roots
    payload = _resolve_protocol_paths(payload, Path.cwd(), *roots)
    frozen_directory = Path(payload["frozen_directory"])
    frozen_summary = json.loads((frozen_directory / "summary.json").read_text(encoding="utf-8"))
    central_topology = _load_topology(
        frozen_directory / "consensus_topology.csv",
        root_size=int(frozen_summary["root_size"]),
        root_pixel_size_rad=np.deg2rad(float(frozen_summary["root_pixel_arcsec"]) / 3600.0),
    )
    with np.load(checkpoint) as stored:
        components: list[MosaicSkyComponent] = [
            MosaicQuadtreeComponent(
                "central",
                central_topology,
                np.asarray(stored["flux_central"]),
            )
        ]
        missing: list[str] = []
        if "flux_coarse" in stored:
            coarse = quadtree_sky_from_regular_grid(
                int(payload["coarse_size"]),
                np.deg2rad(float(payload["coarse_pixel_arcsec"]) / 3600.0),
                np.asarray(stored["flux_coarse"]),
            )
            components.append(MosaicQuadtreeComponent("coarse", coarse.topology, coarse.flux))
        elif "coarse_size" in payload:
            missing.append("coarse")
        if "flux_catalogue" in stored:
            atom_metadata = payload["catalog_atoms"]
            ra = np.deg2rad([float(atom["ra_deg"]) for atom in atom_metadata])
            dec = np.deg2rad([float(atom["dec_deg"]) for atom in atom_metadata])
            l_rad, m_rad, _n = radec_to_lmn(
                mosaic_phase_centre_rad[0],
                mosaic_phase_centre_rad[1],
                ra,
                dec,
            )
            components.append(
                MosaicPointComponent(
                    "catalogue",
                    l_rad,
                    m_rad,
                    np.asarray(stored["flux_catalogue"]),
                )
            )
        elif payload.get("catalog_atoms"):
            missing.append("catalogue")
    if missing:
        raise ValueError(f"checkpoint is missing expected components: {missing}")
    return tuple(components)


def sky_table_from_checkpoint(
    checkpoint: Path,
    protocol: dict[str, Any] | Path,
    mosaic_phase_centre_rad: tuple[float, float],
    *,
    search_roots: tuple[Path, ...] = (),
    include_zero_flux: bool = True,
) -> SkyComponentTable:
    """Convert a sealed composite checkpoint without discarding finite widths."""

    components = mosaic_components_from_checkpoint(
        checkpoint,
        protocol,
        mosaic_phase_centre_rad,
        search_roots=search_roots,
    )
    expected = tuple(component.name for component in components)
    return sky_table_from_mosaic_components(
        components,
        mosaic_phase_centre_rad=mosaic_phase_centre_rad,
        include_zero_flux=include_zero_flux,
        source=str(checkpoint),
        expected_component_names=expected,
    )


def _resolve_protocol_paths(protocol: dict[str, Any], *roots: Path) -> dict[str, Any]:
    resolved = dict(protocol)
    frozen = resolved.get("frozen_directory")
    if not frozen:
        return resolved
    path = Path(str(frozen))
    for candidate in (path, *(root / path for root in roots)):
        if (candidate / "summary.json").is_file():
            resolved["frozen_directory"] = str(candidate)
            return resolved
    return resolved


def _load_topology(
    path: Path,
    *,
    root_size: int,
    root_pixel_size_rad: float,
) -> QuadtreeTopology:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = tuple(csv.DictReader(stream))
    leaves = tuple(
        QuadtreeLeaf(int(row["level"]), int(row["iy"]), int(row["ix"])) for row in rows
    )
    if not leaves:
        raise ValueError("topology CSV must contain at least one leaf")
    return QuadtreeTopology(QuadtreeGrid(root_size, root_pixel_size_rad), leaves)


def _phase_centre(phase_centre_rad: tuple[float, float]) -> tuple[float, float]:
    if len(phase_centre_rad) != 2:
        raise ValueError("mosaic_phase_centre_rad must contain two values")
    return (float(phase_centre_rad[0]), float(phase_centre_rad[1]))


def _atoms_from_mosaic_component(
    component: MosaicSkyComponent,
    family: ComponentFamily,
) -> tuple[SkyComponent, ...]:
    flux = np.asarray(component.flux, dtype=np.float64).reshape(-1)
    if isinstance(component, MosaicPointComponent):
        l_rad = np.asarray(component.l_rad, dtype=np.float64).reshape(-1)
        m_rad = np.asarray(component.m_rad, dtype=np.float64).reshape(-1)
        if flux.size != l_rad.size or l_rad.size != m_rad.size:
            raise ValueError(f"component {component.name!r} flux must match its atoms")
        atoms = []
        for index, (l_value, m_value, flux_value) in enumerate(
            zip(l_rad, m_rad, flux, strict=True)
        ):
            atoms.append(
                SkyComponent(
                    component_id=delta_component_id(family, component.name, index),
                    family=family,
                    basis_type=SkyBasisType.DELTA,
                    l_rad=float(l_value),
                    m_rad=float(m_value),
                    stokes_i_jy=float(flux_value),
                    width_rad=0.0,
                    splitting_permitted=False,
                    provenance={
                        "mosaic_name": component.name,
                        "source_index": int(index),
                    },
                )
            )
        return tuple(atoms)

    if not isinstance(component, MosaicQuadtreeComponent):
        raise TypeError(f"unsupported sky component {type(component)!r}")
    l_rad, m_rad = component.topology.centers()
    widths = component.topology.widths_rad()
    if flux.size != l_rad.size:
        raise ValueError(f"component {component.name!r} flux must match its leaves")
    splitting = family is ComponentFamily.CENTRAL_TREE
    atoms = []
    for leaf, l_value, m_value, width, flux_value in zip(
        component.topology.leaves, l_rad, m_rad, widths, flux, strict=True
    ):
        identifier = leaf_component_id(family, leaf, source=component.name)
        parent_id = (
            None
            if leaf.level == 0
            else leaf_component_id(family, leaf.parent(), source=component.name)
        )
        atoms.append(
            SkyComponent(
                component_id=identifier,
                family=family,
                basis_type=SkyBasisType.UNIFORM_SQUARE,
                l_rad=float(l_value),
                m_rad=float(m_value),
                stokes_i_jy=float(flux_value),
                width_rad=float(width),
                level=int(leaf.level),
                iy=int(leaf.iy),
                ix=int(leaf.ix),
                parent_id=parent_id,
                splitting_permitted=splitting,
                provenance={"mosaic_name": component.name},
            )
        )
    return tuple(atoms)


def _conversion_report(
    components: tuple[SkyComponent, ...] | list[SkyComponent],
    *,
    input_atom_count: int,
    dropped_zero_flux_count: int,
    missing_component_names: tuple[str, ...],
    total_input_flux_jy: float,
    discarded_finite_widths: bool,
) -> SkyConversionReport:
    flux_by_family: dict[str, float] = {}
    count_by_family: dict[str, int] = {}
    count_by_width: dict[str, int] = {}
    for component in components:
        family = component.family.value
        flux_by_family[family] = flux_by_family.get(family, 0.0) + component.stokes_i_jy
        count_by_family[family] = count_by_family.get(family, 0) + 1
        label = width_arcsec_label(component.width_rad)
        count_by_width[label] = count_by_width.get(label, 0) + 1
    return SkyConversionReport(
        input_atom_count=int(input_atom_count),
        kept_atom_count=len(components),
        dropped_zero_flux_count=int(dropped_zero_flux_count),
        missing_component_names=tuple(missing_component_names),
        total_input_flux_jy=float(total_input_flux_jy),
        total_kept_flux_jy=float(sum(component.stokes_i_jy for component in components)),
        flux_by_family=flux_by_family,
        count_by_family=count_by_family,
        count_by_width_arcsec=count_by_width,
        discarded_finite_widths=bool(discarded_finite_widths),
    )


def _reject_prefix_conflicts(components: tuple[SkyComponent, ...]) -> None:
    """Refuse an active parent and child in a prefix-free square family."""

    by_group: dict[tuple[ComponentFamily, str], list[SkyComponent]] = {}
    for component in components:
        if not component.active:
            continue
        if component.basis_type is not SkyBasisType.UNIFORM_SQUARE:
            continue
        if component.level is None or component.iy is None or component.ix is None:
            continue
        source = str(component.provenance.get("mosaic_name") or "")
        by_group.setdefault((component.family, source), []).append(component)
    for (family, _source), members in by_group.items():
        leaves = {
            QuadtreeLeaf(level, iy, ix): item
            for item in members
            for level, iy, ix in ((item.level, item.iy, item.ix),)
            if level is not None and iy is not None and ix is not None
        }
        for leaf, component in leaves.items():
            for ancestor in leaf.ancestors():
                if ancestor in leaves:
                    raise ValueError(
                        f"family {family.value} is not prefix-free: "
                        f"{component.component_id} has active ancestor "
                        f"{leaves[ancestor].component_id}"
                    )


def table_with_replaced_flux(
    table: SkyComponentTable,
    flux: NDArray[np.floating] | list[float] | tuple[float, ...],
) -> SkyComponentTable:
    """Write a new flux vector onto the existing geometry."""

    values = np.asarray(flux, dtype=np.float64).reshape(-1)
    if values.size != len(table.components):
        raise ValueError("flux length must match the sky table")
    updated = tuple(
        replace(component, stokes_i_jy=float(value))
        for component, value in zip(table.components, values, strict=True)
    )
    return SkyComponentTable(
        components=updated,
        mosaic_phase_centre_rad=table.mosaic_phase_centre_rad,
        report=_conversion_report(
            updated,
            input_atom_count=len(updated),
            dropped_zero_flux_count=0,
            missing_component_names=(),
            total_input_flux_jy=float(values.sum()),
            discarded_finite_widths=False,
        ),
        source=table.source,
    )
