"""Flux-conserving quadtree of square pixels, with split/merge operations.

Leaves are keyed by deterministic ``(level, iy, ix)`` coordinates on a fixed
root grid, per ``docs/hierarchical_pixels_proposal.md``. Level 0 matches
``RegularGrid`` exactly; each further level halves the pixel width and
doubles the per-axis leaf count. Splitting a leaf replaces it with four
children whose flux sums to the parent's, so total sky flux is conserved
under any sequence of splits and merges.

Topology (``QuadtreeTopology``: which leaves exist) and flux (the physical
brightness carried by each leaf) are kept as separate types. Topology is
static, host-side data with no NumPy array fields, so it is safe to close
over as a constant inside ``jax.jit``/``jax.grad``. Flux should be passed to
``predict_quadtree_stokes_i`` as a plain JAX array aligned with the
topology's leaf order, so it can be optimized directly; ``QuadtreeSky``
bundles topology with a NumPy-backed flux array purely for host-side
bookkeeping (split/merge, error-bound diagnostics) and is not meant to be
constructed from traced values.

This module intentionally stops short of the proposal's scientific
residual/Haar split score, which needs a fitted model and visibility
residuals. It does wire up the cheaper, purely geometric companion gate
from ``square_wide_field_error_bound``: a leaf can need splitting not
because the sky beneath it is complex, but because the wide-field square
kernel's own curvature approximation has become unreliable at that width
and elevation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from sl1mjax.direct_operator import DirectDFTConfig, predict_stokes_i_explicit
from sl1mjax.polarization import Correlation
from sl1mjax.rime import predict_stokes_i, square_wide_field_error_bound
from sl1mjax.sky import GaussianApproximation, SquarePixelBasis


@dataclass(frozen=True, order=True)
class QuadtreeLeaf:
    """A deterministic ``(level, iy, ix)`` address on the quadtree's root grid."""

    level: int
    iy: int
    ix: int

    def __post_init__(self) -> None:
        if self.level < 0 or self.iy < 0 or self.ix < 0:
            raise ValueError("level, iy, and ix must be non-negative")

    def children(self) -> tuple[QuadtreeLeaf, QuadtreeLeaf, QuadtreeLeaf, QuadtreeLeaf]:
        """The four leaves that exactly tile this leaf's square at the next level.

        Order is index-raster: south row first, then east column first, matching
        ``(2 iy, 2 ix)``, ``(2 iy, 2 ix + 1)``, ``(2 iy + 1, 2 ix)``,
        ``(2 iy + 1, 2 ix + 1)``. On the FITS grid (``iy`` north, ``ix`` west)
        that is sky SE, SW, NE, NW. Residual Haar scores use
        :meth:`haar_children` instead, which is celestial NW, NE, SW, SE.
        """

        return (
            QuadtreeLeaf(self.level + 1, 2 * self.iy, 2 * self.ix),
            QuadtreeLeaf(self.level + 1, 2 * self.iy, 2 * self.ix + 1),
            QuadtreeLeaf(self.level + 1, 2 * self.iy + 1, 2 * self.ix),
            QuadtreeLeaf(self.level + 1, 2 * self.iy + 1, 2 * self.ix + 1),
        )

    def haar_children(self) -> tuple[QuadtreeLeaf, QuadtreeLeaf, QuadtreeLeaf, QuadtreeLeaf]:
        """Children in celestial NW, NE, SW, SE order for residual Haar details."""

        return (
            QuadtreeLeaf(self.level + 1, 2 * self.iy + 1, 2 * self.ix + 1),
            QuadtreeLeaf(self.level + 1, 2 * self.iy + 1, 2 * self.ix),
            QuadtreeLeaf(self.level + 1, 2 * self.iy, 2 * self.ix + 1),
            QuadtreeLeaf(self.level + 1, 2 * self.iy, 2 * self.ix),
        )

    def parent(self) -> QuadtreeLeaf:
        if self.level == 0:
            raise ValueError("level-0 leaves have no parent")
        return QuadtreeLeaf(self.level - 1, self.iy // 2, self.ix // 2)

    def ancestors(self) -> tuple[QuadtreeLeaf, ...]:
        """This leaf's parent, grandparent, ... up to (excluding) the root level."""

        chain = []
        current = self
        while current.level > 0:
            current = current.parent()
            chain.append(current)
        return tuple(chain)


@dataclass(frozen=True)
class QuadtreeGrid:
    """The root grid a quadtree's leaves are addressed relative to.

    Level 0 reproduces ``RegularGrid(root_size, root_pixel_size_rad)``
    exactly, including its axis convention (l decreases with column index,
    FITS ``CDELT1 < 0``) and center alignment.
    """

    root_size: int
    root_pixel_size_rad: float

    def __post_init__(self) -> None:
        if self.root_size < 1:
            raise ValueError("root_size must be at least one")
        if not np.isfinite(self.root_pixel_size_rad) or self.root_pixel_size_rad <= 0:
            raise ValueError("root_pixel_size_rad must be finite and positive")
        # Every finer level subdivides within the root grid's own footprint,
        # so bounding level 0 (which matches RegularGrid exactly) is enough
        # to keep every leaf's center inside the valid direction-cosine disk.
        edge = (self.root_size - 1) * self.root_pixel_size_rad / 2
        if edge >= 1 / np.sqrt(2):
            raise ValueError("grid extends outside valid direction cosines")

    def leaf_width_rad(self, level: int) -> float:
        if level < 0:
            raise ValueError("level must be non-negative")
        return self.root_pixel_size_rad / float(2**level)

    def leaves_per_axis(self, level: int) -> int:
        return self.root_size * int(2**level)

    def leaf_center_rad(self, leaf: QuadtreeLeaf) -> tuple[float, float]:
        n = self.leaves_per_axis(leaf.level)
        width = self.leaf_width_rad(leaf.level)
        l = -((leaf.ix - (n - 1) / 2) * width)
        m = (leaf.iy - (n - 1) / 2) * width
        return l, m

    def contains(self, leaf: QuadtreeLeaf) -> bool:
        n = self.leaves_per_axis(leaf.level)
        return 0 <= leaf.iy < n and 0 <= leaf.ix < n

    def root_leaves(self) -> tuple[QuadtreeLeaf, ...]:
        """Level-0 leaves in the same row-major order as ``RegularGrid.coordinates``."""

        return tuple(
            QuadtreeLeaf(0, iy, ix) for iy in range(self.root_size) for ix in range(self.root_size)
        )


def _validate_leaf_set(grid: QuadtreeGrid, leaves: tuple[QuadtreeLeaf, ...]) -> None:
    """Shared structural checks for a leaf set: unique, in bounds, prefix-free.

    Prefix-free means no active leaf may have another active leaf among its
    ancestors. Two leaves in an ancestor/descendant relationship cover
    overlapping sky area, so allowing both would double-count part of the
    image and make split/merge bookkeeping (which assumes a partition of
    the sky) unsound.
    """

    if len(set(leaves)) != len(leaves):
        raise ValueError("leaves must be unique")
    for leaf in leaves:
        if not grid.contains(leaf):
            raise ValueError(f"leaf {leaf} is out of bounds for the grid")
    leaf_set = set(leaves)
    for leaf in leaves:
        conflicting_ancestors = [ancestor for ancestor in leaf.ancestors() if ancestor in leaf_set]
        if conflicting_ancestors:
            raise ValueError(
                f"leaf set is not prefix-free: {leaf} has active ancestor "
                f"{conflicting_ancestors[0]} covering overlapping sky area"
            )


@dataclass(frozen=True)
class QuadtreeTopology:
    """Which leaves exist, with no flux data.

    Genuinely comparable and hashable (no NumPy array fields), and safe to
    close over as a constant inside ``jax.jit``/``jax.grad``: constructing
    one never touches JAX tracers, so it can be built once per optimization
    epoch and reused. Pass leaf flux separately, as a plain array aligned
    with ``leaves``, to ``predict_quadtree_stokes_i``.
    """

    grid: QuadtreeGrid
    leaves: tuple[QuadtreeLeaf, ...]

    def __post_init__(self) -> None:
        _validate_leaf_set(self.grid, self.leaves)
        object.__setattr__(self, "leaves", tuple(sorted(self.leaves)))

    def centers(self) -> tuple[np.ndarray, np.ndarray]:
        coordinates = [self.grid.leaf_center_rad(leaf) for leaf in self.leaves]
        l = np.asarray([value[0] for value in coordinates], dtype=np.float64)
        m = np.asarray([value[1] for value in coordinates], dtype=np.float64)
        return l, m

    def widths_rad(self) -> np.ndarray:
        return np.asarray(
            [self.grid.leaf_width_rad(leaf.level) for leaf in self.leaves],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class QuadtreeSky:
    """Host-side bookkeeping bundle: a quadtree topology plus its physical flux.

    This is a convenience container for split/merge and diagnostics, not a
    JAX-traceable object: ``flux`` is eagerly converted to a NumPy array, so
    constructing one from a value under ``jax.grad``/``jax.jit`` will raise
    ``TracerArrayConversionError``. For an optimizable sky, keep the
    topology fixed for an optimization epoch (``sky.topology``) and pass
    flux directly to ``predict_quadtree_stokes_i`` as a JAX array.

    Unlike ``QuadtreeTopology``, this class does not support meaningful
    equality or hashing (dataclass equality on a NumPy array field returns
    an elementwise boolean array, and ``hash()`` raises); don't rely on
    either. Flux must be finite and non-negative, matching the physical
    "integrated Jy per leaf" convention used throughout the rest of the
    package.
    """

    grid: QuadtreeGrid
    leaves: tuple[QuadtreeLeaf, ...]
    flux: np.ndarray

    def __post_init__(self) -> None:
        flux_array = np.asarray(self.flux, dtype=np.float64)
        if flux_array.shape != (len(self.leaves),):
            raise ValueError("flux must have exactly one value per leaf")
        if not np.all(np.isfinite(flux_array)):
            raise ValueError("flux must be finite")
        if np.any(flux_array < 0):
            raise ValueError("flux must be non-negative")
        _validate_leaf_set(self.grid, self.leaves)
        order = sorted(range(len(self.leaves)), key=lambda index: self.leaves[index])
        object.__setattr__(self, "leaves", tuple(self.leaves[index] for index in order))
        object.__setattr__(self, "flux", flux_array[order])

    @property
    def topology(self) -> QuadtreeTopology:
        return QuadtreeTopology(self.grid, self.leaves)

    def centers(self) -> tuple[np.ndarray, np.ndarray]:
        return self.topology.centers()

    def widths_rad(self) -> np.ndarray:
        return self.topology.widths_rad()

    def split(self, leaf: QuadtreeLeaf, child_flux: ArrayLike | None = None) -> QuadtreeSky:
        """Replace ``leaf`` with its four children, conserving total flux.

        By default each child gets one quarter of the parent's flux. Pass
        ``child_flux`` (four values, in ``children()`` order) to seed an
        unequal split, e.g. from a four-child lookahead solve; the four
        values must be finite, non-negative, and sum to the parent's flux.
        """

        if leaf not in self.leaves:
            raise ValueError(f"leaf {leaf} is not present in this sky")
        index = self.leaves.index(leaf)
        parent_flux = float(self.flux[index])
        if child_flux is None:
            child_values = np.full(4, parent_flux / 4.0)
        else:
            child_values = np.asarray(child_flux, dtype=np.float64)
            if child_values.shape != (4,):
                raise ValueError("child_flux must contain exactly four values")
            if not np.all(np.isfinite(child_values)):
                raise ValueError("child_flux must be finite")
            if np.any(child_values < 0):
                raise ValueError("child_flux must be non-negative")
            total = float(child_values.sum())
            if not np.isclose(total, parent_flux, rtol=1e-9, atol=1e-12):
                raise ValueError(
                    "child_flux must sum to the parent's flux: "
                    f"got {total!r}, expected {parent_flux!r}"
                )
        children = leaf.children()
        remaining_leaves = self.leaves[:index] + self.leaves[index + 1 :]
        remaining_flux = np.concatenate((self.flux[:index], self.flux[index + 1 :]))
        new_leaves = remaining_leaves + children
        new_flux = np.concatenate((remaining_flux, child_values))
        return QuadtreeSky(self.grid, new_leaves, new_flux)

    def merge(self, parent: QuadtreeLeaf) -> QuadtreeSky:
        """Replace all four children of ``parent`` with ``parent``, summing their flux."""

        if parent in self.leaves:
            raise ValueError(f"leaf {parent} is already present; nothing to merge")
        children = parent.children()
        missing = [child for child in children if child not in self.leaves]
        if missing:
            raise ValueError(f"cannot merge {parent}: missing children {missing}")
        child_indices = [self.leaves.index(child) for child in children]
        combined_flux = float(np.sum(self.flux[child_indices]))
        keep = np.ones(len(self.leaves), dtype=bool)
        keep[child_indices] = False
        remaining_leaves = tuple(leaf for leaf, kept in zip(self.leaves, keep, strict=True) if kept)
        remaining_flux = self.flux[keep]
        new_leaves = remaining_leaves + (parent,)
        new_flux = np.concatenate((remaining_flux, np.asarray([combined_flux])))
        return QuadtreeSky(self.grid, new_leaves, new_flux)


def quadtree_sky_from_regular_grid(
    root_size: int, root_pixel_size_rad: float, flux: ArrayLike
) -> QuadtreeSky:
    """Build a single-level quadtree sky matching a flat ``RegularGrid`` image."""

    grid = QuadtreeGrid(root_size, root_pixel_size_rad)
    flux_array = np.asarray(flux, dtype=np.float64).reshape(-1)
    if flux_array.shape != (root_size * root_size,):
        raise ValueError("flux must contain exactly one value per root cell")
    return QuadtreeSky(grid, grid.root_leaves(), flux_array)


def _broadcast_circular_contrast(
    circular_contrast: ArrayLike | None, leaf_count: int
) -> Array | None:
    if circular_contrast is None:
        return None
    contrast = jnp.asarray(circular_contrast)
    if contrast.size == 1:
        return jnp.broadcast_to(contrast.reshape(()), (leaf_count,))
    if contrast.shape != (leaf_count,):
        raise ValueError("circular_contrast must be scalar or one value per leaf")
    return contrast


def _prediction_inputs(
    flux: ArrayLike,
    topology: QuadtreeTopology,
    beam_weights: ArrayLike | None,
    beam_weights_rr: ArrayLike | None,
    beam_weights_ll: ArrayLike | None,
    centers_lm: tuple[ArrayLike, ArrayLike] | None = None,
    circular_contrast: ArrayLike | None = None,
) -> tuple[
    Array,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, ArrayLike | None],
]:
    if not topology.leaves:
        raise ValueError("quadtree topology has no leaves")
    flux_array = jnp.asarray(flux)
    if flux_array.shape != (len(topology.leaves),):
        raise ValueError("flux must have exactly one value per leaf")
    if centers_lm is None:
        l, m = topology.centers()
    else:
        l = np.asarray(centers_lm[0], dtype=np.float64).ravel()
        m = np.asarray(centers_lm[1], dtype=np.float64).ravel()
        if l.shape != (len(topology.leaves),) or m.shape != l.shape:
            raise ValueError("centers_lm must contain one l and m per leaf")
        if not np.all(np.isfinite(l)) or not np.all(np.isfinite(m)):
            raise ValueError("centers_lm must be finite")
        if np.any(l * l + m * m >= 1):
            raise ValueError("centers_lm must lie inside the visible hemisphere")
    levels = np.asarray([leaf.level for leaf in topology.leaves])
    component_arrays = {
        "beam_weights": beam_weights,
        "beam_weights_rr": beam_weights_rr,
        "beam_weights_ll": beam_weights_ll,
        "circular_contrast": _broadcast_circular_contrast(
            circular_contrast, len(topology.leaves)
        ),
    }
    for name, value in component_arrays.items():
        if value is not None:
            value_array = jnp.asarray(value)
            if value_array.ndim < 1 or value_array.shape[0] != len(topology.leaves):
                raise ValueError(f"{name} must have exactly one row per leaf")
    return flux_array, l, m, levels, component_arrays


def _level_beam_kwargs(
    component_arrays: dict[str, ArrayLike | None], mask: np.ndarray
) -> dict[str, Array]:
    return {
        name: jnp.asarray(value)[mask]
        for name, value in component_arrays.items()
        if value is not None
    }


def predict_quadtree_stokes_i(
    flux: ArrayLike,
    topology: QuadtreeTopology,
    uvw_m: ArrayLike,
    frequency_hz: ArrayLike,
    antenna1: ArrayLike,
    antenna2: ArrayLike,
    correlations: tuple[Correlation, ...],
    *,
    approximation: GaussianApproximation = GaussianApproximation.WIDE_FIELD,
    beam_weights: ArrayLike | None = None,
    beam_weights_rr: ArrayLike | None = None,
    beam_weights_ll: ArrayLike | None = None,
    centers_lm: tuple[ArrayLike, ArrayLike] | None = None,
    circular_contrast: ArrayLike | None = None,
    **predict_kwargs: Any,
) -> Array:
    """Predict correlations for a quadtree sky.

    ``flux`` is a plain array (one value per leaf, in ``topology.leaves``
    order) rather than part of ``topology``, so it can be a traced JAX
    array under ``jax.grad``/``jax.jit`` with the topology held fixed as a
    constant.

    Leaves are grouped by depth so every group shares one pixel width and
    can reuse the existing single-width square-pixel operator (and its
    ``_operator_factory`` cache) with one call per level, summing the
    results. This is the proposal's recommended least-invasive multi-width
    implementation: it avoids a genuinely per-component width kernel.

    ``beam_weights``, ``beam_weights_rr``, and ``beam_weights_ll`` are
    component-indexed (one row per leaf) like ``flux``, so they are sliced
    by the same per-level mask before being forwarded; passing them through
    ``predict_kwargs`` unsliced would mismatch shapes against a level's
    leaf subset for any tree with more than one level present.
    """

    flux_array, l, m, levels, component_arrays = _prediction_inputs(
        flux,
        topology,
        beam_weights,
        beam_weights_rr,
        beam_weights_ll,
        centers_lm,
        circular_contrast,
    )
    total: Array | None = None
    for level in sorted(set(levels.tolist())):
        mask = levels == level
        level_kwargs = dict(predict_kwargs)
        level_kwargs.update(_level_beam_kwargs(component_arrays, mask))
        contribution = predict_stokes_i(
            flux_array[mask],
            l[mask],
            m[mask],
            uvw_m,
            frequency_hz,
            antenna1,
            antenna2,
            correlations,
            pixel_basis=SquarePixelBasis(1.0, approximation),
            pixel_size_rad=topology.grid.leaf_width_rad(int(level)),
            **level_kwargs,
        )
        total = contribution if total is None else total + contribution
    assert total is not None
    return total


def predict_quadtree_stokes_i_explicit(
    flux: ArrayLike,
    topology: QuadtreeTopology,
    uvw_m: ArrayLike,
    frequency_hz: ArrayLike,
    antenna1: ArrayLike,
    antenna2: ArrayLike,
    correlations: tuple[Correlation, ...],
    *,
    approximation: GaussianApproximation = GaussianApproximation.WIDE_FIELD,
    fixed_gains: ArrayLike | None = None,
    config: DirectDFTConfig | None = None,
    beam_weights: ArrayLike | None = None,
    beam_weights_rr: ArrayLike | None = None,
    beam_weights_ll: ArrayLike | None = None,
    centers_lm: tuple[ArrayLike, ArrayLike] | None = None,
    circular_contrast: ArrayLike | None = None,
) -> Array:
    """Predict a quadtree sky with the streamed explicit DFT and adjoint.

    Each level is a separate fixed-width operator call. Flux and primary-beam
    arrays remain component-indexed and are sliced to the same level before
    prediction. The returned sum is linear in leaf flux, so JAX combines the
    per-level custom VJPs into one gradient over the canonical leaf vector.
    """

    flux_array, l, m, levels, component_arrays = _prediction_inputs(
        flux,
        topology,
        beam_weights,
        beam_weights_rr,
        beam_weights_ll,
        centers_lm,
        circular_contrast,
    )
    total: Array | None = None
    for level in sorted(set(levels.tolist())):
        mask = levels == level
        contribution = predict_stokes_i_explicit(
            flux_array[mask],
            l[mask],
            m[mask],
            uvw_m,
            frequency_hz,
            antenna1,
            antenna2,
            correlations,
            fixed_gains=fixed_gains,
            pixel_basis=SquarePixelBasis(1.0, approximation),
            pixel_size_rad=topology.grid.leaf_width_rad(int(level)),
            config=config,
            **_level_beam_kwargs(component_arrays, mask),
        )
        total = contribution if total is None else total + contribution
    assert total is not None
    return total


def wide_field_error_bounds(topology: QuadtreeTopology, max_w_wavelengths: float) -> np.ndarray:
    """Per-leaf analytic upper bound on the wide-field kernel's curvature error.

    Purely geometric: no visibility evaluation is needed, only each leaf's
    own width and position and the dataset's largest ``|w|`` baseline
    component. See ``square_wide_field_error_bound`` for the derivation.
    """

    l, m = topology.centers()
    widths = topology.widths_rad()
    return np.asarray(square_wide_field_error_bound(widths, l, m, max_w_wavelengths))


def leaves_exceeding_error_bound(
    topology: QuadtreeTopology, max_w_wavelengths: float, tolerance: float
) -> tuple[QuadtreeLeaf, ...]:
    """Leaves whose analytic wide-field truncation error exceeds ``tolerance``.

    This is a companion to (not a replacement for) a scientific residual or
    Haar-detail split score: a leaf can be flagged here purely because the
    kernel itself has become unreliable at its current width and elevation,
    even over sky regions with no genuine structure to resolve.
    """

    bounds = wide_field_error_bounds(topology, max_w_wavelengths)
    return tuple(
        leaf for leaf, bound in zip(topology.leaves, bounds, strict=True) if bound > tolerance
    )
