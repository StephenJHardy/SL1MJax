from __future__ import annotations

import numpy as np
import pytest

from sl1mjax.polarization import Correlation
from sl1mjax.quadtree import (
    QuadtreeGrid,
    QuadtreeLeaf,
    QuadtreeSky,
    leaves_exceeding_error_bound,
    predict_quadtree_stokes_i,
    quadtree_sky_from_regular_grid,
    wide_field_error_bounds,
)
from sl1mjax.rime import predict_stokes_i, square_wide_field_error_bound
from sl1mjax.sky import GaussianApproximation, RegularGrid, SquarePixelBasis


def test_level_zero_matches_regular_grid_coordinates() -> None:
    grid = QuadtreeGrid(root_size=4, root_pixel_size_rad=2.5e-3)
    regular = RegularGrid(size=4, pixel_size_rad=2.5e-3)
    expected_l, expected_m = regular.coordinates

    leaves = grid.root_leaves()
    actual_l = np.asarray([grid.leaf_center_rad(leaf)[0] for leaf in leaves])
    actual_m = np.asarray([grid.leaf_center_rad(leaf)[1] for leaf in leaves])

    np.testing.assert_allclose(actual_l, expected_l)
    np.testing.assert_allclose(actual_m, expected_m)
    assert all(grid.leaf_width_rad(leaf.level) == 2.5e-3 for leaf in leaves)


def test_children_exactly_tile_parent_square() -> None:
    grid = QuadtreeGrid(root_size=2, root_pixel_size_rad=4e-3)
    parent = QuadtreeLeaf(0, 0, 0)
    parent_l, parent_m = grid.leaf_center_rad(parent)
    parent_width = grid.leaf_width_rad(0)

    children = parent.children()
    assert len(children) == 4
    assert all(child.parent() == parent for child in children)
    for child in children:
        child_l, child_m = grid.leaf_center_rad(child)
        child_width = grid.leaf_width_rad(child.level)
        assert child_width == pytest.approx(parent_width / 2)
        assert abs(child_l - parent_l) == pytest.approx(child_width / 2)
        assert abs(child_m - parent_m) == pytest.approx(child_width / 2)


def test_split_conserves_flux_and_is_undone_by_merge() -> None:
    sky = quadtree_sky_from_regular_grid(2, 4e-3, [1.0, 2.0, 3.0, 4.0])
    leaf = QuadtreeLeaf(0, 0, 0)

    split_sky = sky.split(leaf)
    assert leaf not in split_sky.leaves
    assert set(leaf.children()).issubset(split_sky.leaves)
    np.testing.assert_allclose(split_sky.flux.sum(), sky.flux.sum())

    merged_sky = split_sky.merge(leaf)
    assert merged_sky.leaves == sky.leaves
    np.testing.assert_allclose(merged_sky.flux, sky.flux)


def test_split_with_explicit_unequal_child_flux() -> None:
    sky = quadtree_sky_from_regular_grid(2, 4e-3, [4.0, 0.0, 0.0, 0.0])
    leaf = QuadtreeLeaf(0, 0, 0)

    split_sky = sky.split(leaf, child_flux=[1.0, 1.0, 1.0, 1.0])
    np.testing.assert_allclose(split_sky.flux.sum(), 4.0)

    with pytest.raises(ValueError, match="four values"):
        sky.split(leaf, child_flux=[1.0, 1.0, 1.0])


def test_split_rejects_absent_leaf_and_merge_rejects_incomplete_children() -> None:
    sky = quadtree_sky_from_regular_grid(2, 4e-3, [1.0, 2.0, 3.0, 4.0])

    with pytest.raises(ValueError, match="not present"):
        sky.split(QuadtreeLeaf(0, 5, 5))

    split_sky = sky.split(QuadtreeLeaf(0, 0, 0))
    incomplete = split_sky.split(QuadtreeLeaf(1, 0, 0))
    with pytest.raises(ValueError, match="missing children"):
        incomplete.merge(QuadtreeLeaf(0, 0, 0))

    with pytest.raises(ValueError, match="already present"):
        split_sky.merge(QuadtreeLeaf(0, 1, 1))


def test_sky_rejects_duplicate_or_mismatched_leaves() -> None:
    grid = QuadtreeGrid(2, 4e-3)
    leaf = QuadtreeLeaf(0, 0, 0)
    with pytest.raises(ValueError, match="one value per leaf"):
        QuadtreeSky(grid, (leaf,), np.asarray([1.0, 2.0]))
    with pytest.raises(ValueError, match="unique"):
        QuadtreeSky(grid, (leaf, leaf), np.asarray([1.0, 2.0]))
    with pytest.raises(ValueError, match="out of bounds"):
        QuadtreeSky(grid, (QuadtreeLeaf(0, 9, 9),), np.asarray([1.0]))


def test_sky_canonicalizes_leaf_and_flux_order() -> None:
    grid = QuadtreeGrid(2, 4e-3)
    leaves = (QuadtreeLeaf(0, 1, 1), QuadtreeLeaf(0, 0, 0))
    sky = QuadtreeSky(grid, leaves, np.asarray([9.0, 1.0]))
    assert sky.leaves == (QuadtreeLeaf(0, 0, 0), QuadtreeLeaf(0, 1, 1))
    np.testing.assert_allclose(sky.flux, [1.0, 9.0])


def _uvw_and_frequency() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    uvw_m = np.asarray([[120.0, -70.0, 35.0], [-50.0, 90.0, -20.0]])
    frequency_hz = np.asarray([1.1e9])
    antenna1 = np.asarray([0, 0])
    antenna2 = np.asarray([1, 1])
    return uvw_m, frequency_hz, antenna1, antenna2


def test_predict_quadtree_matches_manual_per_level_prediction() -> None:
    sky = quadtree_sky_from_regular_grid(2, 4e-3, [1.0, 2.0, 3.0, 4.0])
    sky = sky.split(QuadtreeLeaf(0, 0, 0))
    uvw_m, frequency_hz, antenna1, antenna2 = _uvw_and_frequency()
    correlations = (Correlation.I,)

    actual = predict_quadtree_stokes_i(
        sky, uvw_m, frequency_hz, antenna1, antenna2, correlations
    )

    l, m = sky.centers()
    levels = np.asarray([leaf.level for leaf in sky.leaves])
    expected = np.zeros_like(np.asarray(actual))
    for level in sorted(set(levels.tolist())):
        mask = levels == level
        expected = expected + np.asarray(
            predict_stokes_i(
                sky.flux[mask],
                l[mask],
                m[mask],
                uvw_m,
                frequency_hz,
                antenna1,
                antenna2,
                correlations,
                pixel_basis=SquarePixelBasis(1.0, GaussianApproximation.WIDE_FIELD),
                pixel_size_rad=sky.grid.leaf_width_rad(int(level)),
            )
        )
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)


def test_predict_quadtree_split_reproduces_unsplit_response_under_paraxial() -> None:
    """An equal-flux split must not change the predicted visibility (paraxial)."""

    sky = quadtree_sky_from_regular_grid(2, 4e-3, [1.6, 0.0, 0.0, 0.0])
    split_sky = sky.split(QuadtreeLeaf(0, 0, 0))
    uvw_m, frequency_hz, antenna1, antenna2 = _uvw_and_frequency()
    correlations = (Correlation.I,)

    before = predict_quadtree_stokes_i(
        sky,
        uvw_m,
        frequency_hz,
        antenna1,
        antenna2,
        correlations,
        approximation=GaussianApproximation.PARAXIAL,
    )
    after = predict_quadtree_stokes_i(
        split_sky,
        uvw_m,
        frequency_hz,
        antenna1,
        antenna2,
        correlations,
        approximation=GaussianApproximation.PARAXIAL,
    )
    np.testing.assert_allclose(after, before, rtol=1e-12, atol=1e-12)


def test_predict_quadtree_rejects_empty_sky() -> None:
    grid = QuadtreeGrid(2, 4e-3)
    empty = QuadtreeSky(grid, (), np.asarray([]))
    uvw_m, frequency_hz, antenna1, antenna2 = _uvw_and_frequency()
    with pytest.raises(ValueError, match="no leaves"):
        predict_quadtree_stokes_i(
            empty, uvw_m, frequency_hz, antenna1, antenna2, (Correlation.I,)
        )


def test_wide_field_error_bounds_matches_per_leaf_analytic_formula() -> None:
    sky = quadtree_sky_from_regular_grid(2, 2e-2, [1.0, 2.0, 3.0, 4.0])
    max_w = 250.0

    actual = wide_field_error_bounds(sky, max_w)

    l, m = sky.centers()
    widths = sky.widths_rad()
    expected = np.asarray(
        [
            square_wide_field_error_bound(width, l_value, m_value, max_w)
            for width, l_value, m_value in zip(widths, l, m, strict=True)
        ]
    )
    np.testing.assert_allclose(actual, expected)


def test_leaves_exceeding_error_bound_flags_coarse_pixels_not_fine_ones() -> None:
    """Splitting a flagged coarse pixel should bring its children under tolerance."""

    sky = quadtree_sky_from_regular_grid(2, 4e-3, [1.0, 2.0, 3.0, 4.0])
    coarse_leaf = QuadtreeLeaf(0, 0, 0)
    max_w = 100.0
    tolerance = 1e-3

    flagged = leaves_exceeding_error_bound(sky, max_w, tolerance)
    assert coarse_leaf in flagged

    refined = sky.split(coarse_leaf)
    still_flagged = leaves_exceeding_error_bound(refined, max_w, tolerance)
    assert not any(child in still_flagged for child in coarse_leaf.children())
