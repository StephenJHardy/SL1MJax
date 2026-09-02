from __future__ import annotations

import numpy as np
import pytest

from sl1mjax.beam_aware_imaging import ComponentFamily, SkyBasisType, SkyComponent
from sl1mjax.quadtree import QuadtreeLeaf
from sl1mjax.wide_field_sky import (
    CENTRAL_OUTER_SPAN,
    CENTRAL_ROOT_SIZE,
    OUTER_MARGIN,
    OUTER_ROOT_SIZE,
    catalogue_components_from_pinned_json,
    central_grid,
    central_half_width_rad,
    central_pixel_size_rad,
    is_boundary_guard_leaf,
    is_central_outer_root,
    is_outer_edge_guard_leaf,
    outer_grid,
    outer_pixel_size_rad,
    phase5_render_spec,
    phase5_starting_table,
    render_intrinsic_stokes_i,
)

_PHASE = (np.deg2rad(282.35), np.deg2rad(-0.93))


def test_phase5_central_nests_inside_the_64_arcsec_outer_grid() -> None:
    assert CENTRAL_ROOT_SIZE == 104
    assert OUTER_ROOT_SIZE == 64
    assert np.isclose(CENTRAL_OUTER_SPAN * outer_pixel_size_rad(), 104 * central_pixel_size_rad())
    outer = outer_grid()
    central = central_grid()
    south_west_outer = QuadtreeLeaf(0, OUTER_MARGIN, OUTER_MARGIN)
    south_west_child = QuadtreeLeaf(2, OUTER_MARGIN * 4, OUTER_MARGIN * 4)
    np.testing.assert_allclose(
        outer.leaf_center_rad(south_west_child),
        central.leaf_center_rad(QuadtreeLeaf(0, 0, 0)),
        rtol=0.0,
        atol=1e-15,
    )
    del south_west_outer


def test_phase5_starting_table_is_prefix_free_with_inactive_guard() -> None:
    table = phase5_starting_table(mosaic_phase_centre_rad=_PHASE)
    central = [item for item in table.components if item.family is ComponentFamily.CENTRAL_TREE]
    guard = [item for item in table.components if item.family is ComponentFamily.OUTER_GUARD]
    assert len(central) == 104 * 104
    assert all(item.active and item.stokes_i_jy == 0.0 for item in central)
    assert all(not item.active and item.stokes_i_jy == 0.0 for item in guard)
    assert len(guard) == 64 * 64 - 26 * 26
    assert not any(item.family is ComponentFamily.COARSE_FIELD for item in table.components)
    half = central_half_width_rad()
    for item in central:
        assert abs(item.l_rad) <= half + 0.5 * central_pixel_size_rad() + 1e-15
        assert abs(item.m_rad) <= half + 0.5 * central_pixel_size_rad() + 1e-15
    for item in guard:
        assert not is_central_outer_root(int(item.iy), int(item.ix))
        assert (
            abs(item.l_rad) > half - 0.5 * outer_pixel_size_rad() - 1e-12
            or abs(item.m_rad) > half - 0.5 * outer_pixel_size_rad() - 1e-12
        )


def test_phase5_starting_table_refuses_coarse_field() -> None:
    with pytest.raises(ValueError, match="coarse field"):
        phase5_starting_table(
            mosaic_phase_centre_rad=_PHASE,
            catalogue=(
                SkyComponent(
                    component_id="coarse_field:coarse:0:0:0",
                    family=ComponentFamily.COARSE_FIELD,
                    basis_type=SkyBasisType.UNIFORM_SQUARE,
                    l_rad=0.0,
                    m_rad=0.0,
                    stokes_i_jy=1.0,
                    width_rad=central_pixel_size_rad(),
                    level=0,
                    iy=0,
                    ix=0,
                ),
            ),
        )


def test_phase5_pinned_catalogue_uses_published_extrapolated_flux() -> None:
    catalogue = catalogue_components_from_pinned_json("config/3c391_radio_guard_catalog.json")
    table = phase5_starting_table(mosaic_phase_centre_rad=_PHASE, catalogue=catalogue)
    atoms = [item for item in table.components if item.family is ComponentFamily.CATALOGUE]
    assert len(atoms) == 3
    assert all(item.stokes_i_jy > 0 for item in atoms)
    assert {item.provenance["source_name"] for item in atoms} == {
        "VLASS_J184932.56-003805.2",
        "VLASS_J184734.26-011244.4",
        "VLASS_J184959.25-013255.8",
    }
    spec = phase5_render_spec(_PHASE)
    image = render_intrinsic_stokes_i(table, spec=spec)
    np.testing.assert_allclose(np.sum(image), sum(item.stokes_i_jy for item in atoms), rtol=1e-6)
    other = phase5_starting_table(mosaic_phase_centre_rad=_PHASE)
    assert render_intrinsic_stokes_i(other, spec=spec).shape == image.shape
    assert spec.grid_size == image.shape[0]
    assert spec.phase_centre_rad == _PHASE


def test_phase5_outer_edge_is_not_the_inner_boundary() -> None:
    inner = QuadtreeLeaf(0, OUTER_MARGIN - 1, OUTER_MARGIN)
    outer = QuadtreeLeaf(0, 0, OUTER_MARGIN)
    corner = QuadtreeLeaf(0, 0, 0)
    assert is_boundary_guard_leaf(inner)
    assert not is_outer_edge_guard_leaf(inner)
    assert is_outer_edge_guard_leaf(outer)
    assert is_outer_edge_guard_leaf(corner)
    assert not is_boundary_guard_leaf(outer)
