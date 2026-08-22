from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sl1mjax.direct_operator import DirectDFTConfig
from sl1mjax.polarization import Correlation
from sl1mjax.quadtree import (
    QuadtreeGrid,
    QuadtreeLeaf,
    QuadtreeSky,
    QuadtreeTopology,
    leaves_exceeding_error_bound,
    predict_quadtree_stokes_i,
    predict_quadtree_stokes_i_explicit,
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
        sky.flux, sky.topology, uvw_m, frequency_hz, antenna1, antenna2, correlations
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
        sky.flux,
        sky.topology,
        uvw_m,
        frequency_hz,
        antenna1,
        antenna2,
        correlations,
        approximation=GaussianApproximation.PARAXIAL,
    )
    after = predict_quadtree_stokes_i(
        split_sky.flux,
        split_sky.topology,
        uvw_m,
        frequency_hz,
        antenna1,
        antenna2,
        correlations,
        approximation=GaussianApproximation.PARAXIAL,
    )
    np.testing.assert_allclose(after, before, rtol=1e-12, atol=1e-12)


def test_predict_quadtree_rejects_empty_topology() -> None:
    grid = QuadtreeGrid(2, 4e-3)
    empty = QuadtreeTopology(grid, ())
    uvw_m, frequency_hz, antenna1, antenna2 = _uvw_and_frequency()
    with pytest.raises(ValueError, match="no leaves"):
        predict_quadtree_stokes_i(
            jnp.asarray([]), empty, uvw_m, frequency_hz, antenna1, antenna2, (Correlation.I,)
        )


def test_wide_field_error_bounds_matches_per_leaf_analytic_formula() -> None:
    sky = quadtree_sky_from_regular_grid(2, 2e-2, [1.0, 2.0, 3.0, 4.0])
    max_w = 250.0

    actual = wide_field_error_bounds(sky.topology, max_w)

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

    flagged = leaves_exceeding_error_bound(sky.topology, max_w, tolerance)
    assert coarse_leaf in flagged

    refined = sky.split(coarse_leaf)
    still_flagged = leaves_exceeding_error_bound(refined.topology, max_w, tolerance)
    assert not any(child in still_flagged for child in coarse_leaf.children())


# --- Regression tests for reviewer findings on PR #1 ---


def test_flux_can_be_optimized_with_jax_grad_against_a_fixed_topology() -> None:
    """Finding 1: topology is static host data; flux must trace under jax.grad."""

    sky = quadtree_sky_from_regular_grid(2, 4e-3, [0.7, 0.2, 1.1, 0.4])
    topology = sky.topology  # built once, host-side; never touched by the traced call
    uvw_m, frequency_hz, antenna1, antenna2 = _uvw_and_frequency()
    correlations = (Correlation.I,)

    def loss(flux: jax.Array) -> jax.Array:
        prediction = predict_quadtree_stokes_i(
            flux, topology, uvw_m, frequency_hz, antenna1, antenna2, correlations
        )
        return jnp.real(jnp.sum(prediction * jnp.conj(prediction)))

    flux0 = jnp.asarray(sky.flux)
    automatic = np.asarray(jax.grad(loss)(flux0))

    epsilon = 1e-6
    finite = np.empty(4)
    for index in range(4):
        offset = np.zeros(4)
        offset[index] = epsilon
        finite[index] = (float(loss(flux0 + offset)) - float(loss(flux0 - offset))) / (2 * epsilon)
    np.testing.assert_allclose(automatic, finite, rtol=2e-6, atol=2e-7)


def test_predict_quadtree_slices_beam_weights_per_level() -> None:
    """Finding 2: component-indexed beam_weights must be masked per level, not
    forwarded whole -- otherwise a level's leaf subset shape mismatches the
    full-tree beam_weights shape as soon as a tree has more than one level."""

    sky = quadtree_sky_from_regular_grid(2, 4e-3, [1.0, 2.0, 3.0, 4.0])
    sky = sky.split(QuadtreeLeaf(0, 0, 0))  # 3 level-0 leaves + 4 level-1 leaves = 7
    uvw_m, frequency_hz, antenna1, antenna2 = _uvw_and_frequency()
    correlations = (Correlation.I,)
    rng = np.random.default_rng(0)
    beam_weights = rng.uniform(0.5, 1.5, size=(len(sky.leaves), frequency_hz.size))

    actual = predict_quadtree_stokes_i(
        sky.flux,
        sky.topology,
        uvw_m,
        frequency_hz,
        antenna1,
        antenna2,
        correlations,
        beam_weights=beam_weights,
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
                beam_weights=beam_weights[mask],
            )
        )
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)


def test_predict_quadtree_rejects_mismatched_beam_weight_rows() -> None:
    sky = quadtree_sky_from_regular_grid(2, 4e-3, [1.0, 2.0, 3.0, 4.0])
    uvw_m, frequency_hz, antenna1, antenna2 = _uvw_and_frequency()
    wrong_shape_beam_weights = np.ones((3, frequency_hz.size))

    with pytest.raises(ValueError, match="one row per leaf"):
        predict_quadtree_stokes_i(
            sky.flux,
            sky.topology,
            uvw_m,
            frequency_hz,
            antenna1,
            antenna2,
            (Correlation.I,),
            beam_weights=wrong_shape_beam_weights,
        )


def test_split_rejects_child_flux_that_does_not_conserve_total() -> None:
    """Finding 3: explicit child_flux must sum to the parent's flux."""

    sky = quadtree_sky_from_regular_grid(2, 4e-3, [4.0, 0.0, 0.0, 0.0])
    leaf = QuadtreeLeaf(0, 0, 0)

    with pytest.raises(ValueError, match="sum to the parent"):
        sky.split(leaf, child_flux=[1.0, 1.0, 0.5, 0.5])  # sums to 3.0, not 4.0

    with pytest.raises(ValueError, match="non-negative"):
        sky.split(leaf, child_flux=[5.0, -1.0, 0.0, 0.0])

    with pytest.raises(ValueError, match="finite"):
        sky.split(leaf, child_flux=[np.nan, 4.0, 0.0, 0.0])


def test_sky_rejects_negative_or_non_finite_flux() -> None:
    grid = QuadtreeGrid(2, 4e-3)
    leaves = grid.root_leaves()
    with pytest.raises(ValueError, match="non-negative"):
        QuadtreeSky(grid, leaves, np.asarray([1.0, -0.5, 2.0, 3.0]))
    with pytest.raises(ValueError, match="finite"):
        QuadtreeSky(grid, leaves, np.asarray([1.0, np.inf, 2.0, 3.0]))


def test_leaf_set_must_be_prefix_free() -> None:
    """Finding 4: a parent and any of its descendants may not both be active
    leaves -- their sky areas overlap and prediction would double-count."""

    grid = QuadtreeGrid(2, 4e-3)
    parent = QuadtreeLeaf(0, 0, 0)
    child = parent.children()[0]
    grandchild = child.children()[0]

    with pytest.raises(ValueError, match="prefix-free"):
        QuadtreeTopology(grid, (parent, child, QuadtreeLeaf(0, 0, 1)))

    with pytest.raises(ValueError, match="prefix-free"):
        QuadtreeSky(grid, (parent, grandchild), np.asarray([1.0, 2.0]))


def test_topology_is_hashable_and_compares_equal_regardless_of_construction_order() -> None:
    """Finding 5: QuadtreeTopology (unlike QuadtreeSky) has no NumPy fields,
    so equality and hashing work as a canonicalized dataclass would suggest."""

    grid = QuadtreeGrid(2, 4e-3)
    leaves_a = (QuadtreeLeaf(0, 1, 1), QuadtreeLeaf(0, 0, 0))
    leaves_b = (QuadtreeLeaf(0, 0, 0), QuadtreeLeaf(0, 1, 1))

    topology_a = QuadtreeTopology(grid, leaves_a)
    topology_b = QuadtreeTopology(grid, leaves_b)

    assert topology_a == topology_b
    assert hash(topology_a) == hash(topology_b)
    assert len({topology_a, topology_b}) == 1


def test_explicit_quadtree_prediction_and_gradient_match_autodiff() -> None:
    sky = quadtree_sky_from_regular_grid(2, 4e-3, [1.0, 2.0, 3.0, 4.0])
    sky = sky.split(QuadtreeLeaf(0, 0, 0), child_flux=[0.1, 0.2, 0.3, 0.4])
    uvw_m, frequency_hz, antenna1, antenna2 = _uvw_and_frequency()
    correlations = (Correlation.I,)
    beam_weights = np.linspace(0.7, 1.0, len(sky.leaves))[:, None]
    config = DirectDFTConfig(visibility_chunk_size=1, pixel_chunk_size=2)

    def native_loss(flux: jax.Array) -> jax.Array:
        prediction = predict_quadtree_stokes_i(
            flux,
            sky.topology,
            uvw_m,
            frequency_hz,
            antenna1,
            antenna2,
            correlations,
            beam_weights=beam_weights,
        )
        return jnp.real(jnp.vdot(prediction, prediction))

    def explicit_loss(flux: jax.Array) -> jax.Array:
        prediction = predict_quadtree_stokes_i_explicit(
            flux,
            sky.topology,
            uvw_m,
            frequency_hz,
            antenna1,
            antenna2,
            correlations,
            config=config,
            beam_weights=beam_weights,
        )
        return jnp.real(jnp.vdot(prediction, prediction))

    flux = jnp.asarray(sky.flux)
    np.testing.assert_allclose(explicit_loss(flux), native_loss(flux), rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(
        jax.grad(explicit_loss)(flux),
        jax.grad(native_loss)(flux),
        rtol=1e-12,
        atol=1e-12,
    )
