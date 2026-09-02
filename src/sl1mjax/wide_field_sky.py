"""Prefix-free Phase 5 sky: refinable centre, inactive outer guard, catalogue."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from sl1mjax.beam_aware_imaging import (
    ComponentFamily,
    SkyBasisType,
    SkyComponent,
    SkyComponentTable,
    delta_component_id,
    leaf_component_id,
    sky_table_from_mosaic_components,
    sky_table_from_records,
    sky_table_to_records,
)
from sl1mjax.catalog import CatalogGuardAtom, RadioCatalogSource
from sl1mjax.composite import MosaicQuadtreeComponent
from sl1mjax.quadtree import QuadtreeGrid, QuadtreeLeaf, QuadtreeSky, QuadtreeTopology

OUTER_ROOT_SIZE = 64
OUTER_PIXEL_ARCSEC = 64.0
CENTRAL_OUTER_SPAN = 26
CENTRAL_ROOT_SIZE = 104
CENTRAL_PIXEL_ARCSEC = 16.0
OUTER_MARGIN = (OUTER_ROOT_SIZE - CENTRAL_OUTER_SPAN) // 2


def outer_pixel_size_rad() -> float:
    return float(np.deg2rad(OUTER_PIXEL_ARCSEC / 3600.0))


def central_pixel_size_rad() -> float:
    return float(np.deg2rad(CENTRAL_PIXEL_ARCSEC / 3600.0))


def central_half_width_rad() -> float:
    return 0.5 * CENTRAL_OUTER_SPAN * outer_pixel_size_rad()


def outer_grid() -> QuadtreeGrid:
    return QuadtreeGrid(OUTER_ROOT_SIZE, outer_pixel_size_rad())


def central_grid() -> QuadtreeGrid:
    return QuadtreeGrid(CENTRAL_ROOT_SIZE, central_pixel_size_rad())


def is_central_outer_root(iy: int, ix: int) -> bool:
    return (
        OUTER_MARGIN <= iy < OUTER_MARGIN + CENTRAL_OUTER_SPAN
        and OUTER_MARGIN <= ix < OUTER_MARGIN + CENTRAL_OUTER_SPAN
    )


def outer_guard_leaves() -> tuple[QuadtreeLeaf, ...]:
    return tuple(
        QuadtreeLeaf(0, iy, ix)
        for iy in range(OUTER_ROOT_SIZE)
        for ix in range(OUTER_ROOT_SIZE)
        if not is_central_outer_root(iy, ix)
    )


def is_boundary_guard_leaf(leaf: QuadtreeLeaf) -> bool:
    """True for the inner guard ring around the refinable 26×26 field."""

    if leaf.level != 0:
        return False
    iy_touch = leaf.iy in {OUTER_MARGIN - 1, OUTER_MARGIN + CENTRAL_OUTER_SPAN}
    ix_touch = leaf.ix in {OUTER_MARGIN - 1, OUTER_MARGIN + CENTRAL_OUTER_SPAN}
    in_central_iy = OUTER_MARGIN <= leaf.iy < OUTER_MARGIN + CENTRAL_OUTER_SPAN
    in_central_ix = OUTER_MARGIN <= leaf.ix < OUTER_MARGIN + CENTRAL_OUTER_SPAN
    return (iy_touch and in_central_ix) or (ix_touch and in_central_iy) or (iy_touch and ix_touch)


def is_outer_edge_guard_leaf(leaf: QuadtreeLeaf) -> bool:
    """True for the perimeter of the complete 64×64 outer guard."""

    if leaf.level != 0:
        return False
    last = OUTER_ROOT_SIZE - 1
    return leaf.iy in {0, last} or leaf.ix in {0, last}


@dataclass(frozen=True)
class Phase5Geometry:
    """Declared Phase 5 prefix-free layout."""

    central_root_size: int = CENTRAL_ROOT_SIZE
    central_pixel_size_rad: float = central_pixel_size_rad()
    outer_root_size: int = OUTER_ROOT_SIZE
    outer_pixel_size_rad: float = outer_pixel_size_rad()
    central_outer_span: int = CENTRAL_OUTER_SPAN


def catalogue_components_from_pinned_json(path: str | Path) -> tuple[SkyComponent, ...]:
    """Load the frozen NVSS/VLASS guard atoms with published initial fluxes."""

    import json

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    atoms = []
    for item in payload["atoms"]:
        source = item["source"]
        atoms.append(
            CatalogGuardAtom(
                source=RadioCatalogSource(
                    name=source["name"],
                    ra_deg=float(source["ra_deg"]),
                    dec_deg=float(source["dec_deg"]),
                    reference_frequency_hz=float(source["reference_frequency_hz"]),
                    integrated_flux_jy=float(source["integrated_flux_jy"]),
                    catalog=source["catalog"],
                    reference_url=source["reference_url"],
                    peak_flux_jy=source.get("peak_flux_jy"),
                    major_axis_arcsec=source.get("major_axis_arcsec"),
                    minor_axis_arcsec=source.get("minor_axis_arcsec"),
                    position_angle_deg=source.get("position_angle_deg"),
                    spectral_index=source.get("spectral_index"),
                    epoch=source.get("epoch") or "",
                ),
                l_rad=float(item["l_rad"]),
                m_rad=float(item["m_rad"]),
                offset_arcmin=float(item["offset_arcmin"]),
                initial_flux_jy=float(item["initial_flux_jy"]),
                maximum_apparent_flux_jy=float(item["maximum_apparent_flux_jy"]),
                maximum_beam_power=float(item["maximum_beam_power"]),
            )
        )
    return catalogue_components_from_atoms(atoms)


def catalogue_components_from_atoms(
    atoms: Sequence[CatalogGuardAtom],
) -> tuple[SkyComponent, ...]:
    components = []
    for index, atom in enumerate(atoms):
        components.append(
            SkyComponent(
                component_id=delta_component_id(
                    ComponentFamily.CATALOGUE, atom.source.catalog, index
                ),
                family=ComponentFamily.CATALOGUE,
                basis_type=SkyBasisType.DELTA,
                l_rad=float(atom.l_rad),
                m_rad=float(atom.m_rad),
                stokes_i_jy=float(atom.initial_flux_jy),
                width_rad=0.0,
                active=True,
                splitting_permitted=False,
                provenance={
                    "mosaic_name": atom.source.catalog,
                    "source_name": atom.source.name,
                    "catalog": atom.source.catalog,
                    "reference_url": atom.source.reference_url,
                    "reference_frequency_hz": atom.source.reference_frequency_hz,
                    "published_flux_jy": atom.source.integrated_flux_jy,
                    "spectral_index": atom.source.spectral_index,
                    "initial_flux_jy": atom.initial_flux_jy,
                    "offset_arcmin": atom.offset_arcmin,
                },
            )
        )
    return tuple(components)


def phase5_starting_table(
    *,
    mosaic_phase_centre_rad: tuple[float, float],
    catalogue: Sequence[SkyComponent] = (),
) -> SkyComponentTable:
    """Central 104×104 zeros, inactive 64″ guard, pinned catalogue deltas."""

    central_sky = QuadtreeSky(
        central_grid(),
        central_grid().root_leaves(),
        np.zeros(CENTRAL_ROOT_SIZE * CENTRAL_ROOT_SIZE, dtype=np.float64),
    )
    central = sky_table_from_mosaic_components(
        (MosaicQuadtreeComponent("central", central_sky.topology, central_sky.flux),),
        mosaic_phase_centre_rad=mosaic_phase_centre_rad,
        source="phase5_starting_sky",
    )
    guard_leaves = outer_guard_leaves()
    guard_grid = outer_grid()
    guard_flux = np.zeros(len(guard_leaves), dtype=np.float64)
    guard_topology = QuadtreeTopology(guard_grid, guard_leaves)
    records = sky_table_to_records(central)
    for leaf, flux in zip(guard_topology.leaves, guard_flux, strict=True):
        l_rad, m_rad = guard_grid.leaf_center_rad(leaf)
        records.append(
            {
                "component_id": leaf_component_id(
                    ComponentFamily.OUTER_GUARD, leaf, source="guard"
                ),
                "family": ComponentFamily.OUTER_GUARD.value,
                "basis_type": SkyBasisType.UNIFORM_SQUARE.value,
                "l_rad": float(l_rad),
                "m_rad": float(m_rad),
                "stokes_i_jy": float(flux),
                "width_rad": float(guard_grid.leaf_width_rad(leaf.level)),
                "level": int(leaf.level),
                "iy": int(leaf.iy),
                "ix": int(leaf.ix),
                "parent_id": None,
                "active": False,
                "splitting_permitted": False,
                "provenance": {"mosaic_name": "guard", "role": "sentinel"},
            }
        )
    for component in catalogue:
        if component.family is ComponentFamily.COARSE_FIELD:
            raise ValueError("Phase 5 sky refuses the overlapping coarse field")
        if component.family is not ComponentFamily.CATALOGUE:
            raise ValueError("phase5 catalogue extras must be catalogue atoms")
        records.append(_component_record(component))
    table = sky_table_from_records(
        records, mosaic_phase_centre_rad=mosaic_phase_centre_rad, source="phase5_starting_sky"
    )
    _require_phase5_geometry(table)
    return table


def _require_phase5_geometry(table: SkyComponentTable) -> None:
    central = [
        item
        for item in table.components
        if item.family is ComponentFamily.CENTRAL_TREE and item.active
    ]
    guard = [item for item in table.components if item.family is ComponentFamily.OUTER_GUARD]
    if len(central) != CENTRAL_ROOT_SIZE * CENTRAL_ROOT_SIZE:
        raise ValueError("Phase 5 central sky must be the 104 by 104 root grid")
    if any(not np.isclose(item.width_rad, central_pixel_size_rad()) for item in central):
        raise ValueError("Phase 5 central roots must be 16 arcsec")
    if any(item.stokes_i_jy != 0.0 for item in central):
        raise ValueError("Phase 5 starting central roots must be zero")
    if any(item.active for item in guard):
        raise ValueError("Phase 5 starting outer-guard roots must be inactive")
    if any(item.stokes_i_jy != 0.0 for item in guard):
        raise ValueError("Phase 5 starting outer-guard roots must be zero")
    if any(item.family is ComponentFamily.COARSE_FIELD for item in table.components):
        raise ValueError("Phase 5 sky refuses the overlapping coarse field")
    expected_guard = OUTER_ROOT_SIZE * OUTER_ROOT_SIZE - CENTRAL_OUTER_SPAN * CENTRAL_OUTER_SPAN
    if len(guard) != expected_guard:
        raise ValueError(
            "Phase 5 outer guard must be the 64 by 64 frame minus the central 26 by 26"
        )


def _component_record(component: SkyComponent) -> dict[str, Any]:
    return {
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


def activate_guard_roots(
    table: SkyComponentTable, leaves: Sequence[QuadtreeLeaf]
) -> SkyComponentTable:
    wanted = {leaf for leaf in leaves}
    records = sky_table_to_records(table)
    for record in records:
        if record["family"] != ComponentFamily.OUTER_GUARD.value:
            continue
        leaf = QuadtreeLeaf(int(record["level"]), int(record["iy"]), int(record["ix"]))
        if leaf in wanted:
            record["active"] = True
            record["splitting_permitted"] = False
    return sky_table_from_records(
        records,
        mosaic_phase_centre_rad=table.mosaic_phase_centre_rad,
        source=table.source,
    )


PHASE5_RENDER_HALF_WIDTH_RAD = max(
    0.5 * OUTER_ROOT_SIZE * outer_pixel_size_rad(),
    float(np.deg2rad(40.0 / 60.0)),
)


@dataclass(frozen=True)
class IntrinsicImageSpec:
    """Frozen mosaic-frame image coordinates shared by every beam candidate."""

    grid_size: int
    pixel_size_rad: float
    phase_centre_rad: tuple[float, float]
    half_width_rad: float
    l_increases_with_x: bool = False
    m_increases_with_y: bool = True
    reference_pixel: tuple[float, float] = (0.0, 0.0)

    def to_arrays(self) -> dict[str, np.ndarray | float | int | bool]:
        center = 0.5 * (self.grid_size - 1)
        return {
            "grid_size": int(self.grid_size),
            "pixel_size_rad": float(self.pixel_size_rad),
            "phase_centre_ra_rad": float(self.phase_centre_rad[0]),
            "phase_centre_dec_rad": float(self.phase_centre_rad[1]),
            "half_width_rad": float(self.half_width_rad),
            "l_increases_with_x": bool(self.l_increases_with_x),
            "m_increases_with_y": bool(self.m_increases_with_y),
            "reference_pixel_y": float(self.reference_pixel[0]),
            "reference_pixel_x": float(self.reference_pixel[1]),
            "center_pixel": float(center),
        }


def phase5_render_spec(
    phase_centre_rad: tuple[float, float],
    *,
    pixel_size_rad: float | None = None,
) -> IntrinsicImageSpec:
    """One frozen render grid covering the 64″ field and pinned catalogue atoms."""

    width = central_pixel_size_rad() if pixel_size_rad is None else float(pixel_size_rad)
    half = float(PHASE5_RENDER_HALF_WIDTH_RAD)
    grid_size = int(np.ceil(2.0 * half / width)) + 1
    if grid_size % 2 == 0:
        grid_size += 1
    center = 0.5 * (grid_size - 1)
    return IntrinsicImageSpec(
        grid_size=grid_size,
        pixel_size_rad=width,
        phase_centre_rad=(float(phase_centre_rad[0]), float(phase_centre_rad[1])),
        half_width_rad=half,
        reference_pixel=(center, center),
    )


def render_intrinsic_stokes_i(
    table: SkyComponentTable,
    *,
    spec: IntrinsicImageSpec | None = None,
    grid_size: int | None = None,
    pixel_size_rad: float | None = None,
) -> NDArray[np.float64]:
    """Paint active squares and deltas onto one common mosaic-frame image."""

    selected = spec
    if selected is None:
        if grid_size is None and pixel_size_rad is None:
            selected = phase5_render_spec(table.mosaic_phase_centre_rad)
        else:
            width = central_pixel_size_rad() if pixel_size_rad is None else float(pixel_size_rad)
            size = (
                int(grid_size)
                if grid_size is not None
                else (int(np.ceil(2.0 * PHASE5_RENDER_HALF_WIDTH_RAD / width)) + 1)
            )
            if size % 2 == 0:
                size += 1
            center = 0.5 * (size - 1)
            selected = IntrinsicImageSpec(
                grid_size=size,
                pixel_size_rad=width,
                phase_centre_rad=table.mosaic_phase_centre_rad,
                half_width_rad=0.5 * (size - 1) * width,
                reference_pixel=(center, center),
            )
    width = float(selected.pixel_size_rad)
    size = int(selected.grid_size)
    origin = 0.5 * (size - 1) * width
    image = np.zeros((size, size), dtype=np.float64)
    for component in table.components:
        if not component.active or component.stokes_i_jy == 0.0:
            continue
        if component.basis_type is SkyBasisType.DELTA or component.width_rad == 0.0:
            ix = int(round((-component.l_rad + origin) / width))
            iy = int(round((component.m_rad + origin) / width))
            if 0 <= iy < size and 0 <= ix < size:
                image[iy, ix] += component.stokes_i_jy
            continue
        half = 0.5 * component.width_rad
        ix0 = int(np.floor((-component.l_rad - half + origin) / width + 0.5))
        ix1 = int(np.floor((-component.l_rad + half + origin) / width + 0.5))
        iy0 = int(np.floor((component.m_rad - half + origin) / width + 0.5))
        iy1 = int(np.floor((component.m_rad + half + origin) / width + 0.5))
        iy0 = max(iy0, 0)
        ix0 = max(ix0, 0)
        iy1 = min(iy1, size)
        ix1 = min(ix1, size)
        area = max(iy1 - iy0, 0) * max(ix1 - ix0, 0)
        if area <= 0:
            continue
        image[iy0:iy1, ix0:ix1] += component.stokes_i_jy / area
    return image
