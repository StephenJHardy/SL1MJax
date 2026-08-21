"""Flux-conserving quadtree of square pixels, with split/merge operations.

Leaves are keyed by deterministic ``(level, iy, ix)`` coordinates on a fixed
root grid, per ``docs/hierarchical_pixels_proposal.md``. Level 0 matches
``RegularGrid`` exactly; each further level halves the pixel width and
doubles the per-axis leaf count. Splitting a leaf replaces it with four
children whose flux sums to the parent's, so total sky flux is conserved
under any sequence of splits and merges.

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

import numpy as np
from jax import Array
from jax.typing import ArrayLike

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
        """The four leaves that exactly tile this leaf's square at the next level."""

        return (
            QuadtreeLeaf(self.level + 1, 2 * self.iy, 2 * self.ix),
            QuadtreeLeaf(self.level + 1, 2 * self.iy, 2 * self.ix + 1),
            QuadtreeLeaf(self.level + 1, 2 * self.iy + 1, 2 * self.ix),
            QuadtreeLeaf(self.level + 1, 2 * self.iy + 1, 2 * self.ix + 1),
        )

    def parent(self) -> QuadtreeLeaf:
        if self.level == 0:
            raise ValueError("level-0 leaves have no parent")
        return QuadtreeLeaf(self.level - 1, self.iy // 2, self.ix // 2)


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


@dataclass(frozen=True)
class QuadtreeSky:
    """A flux-conserving quadtree sky: deterministic leaves plus integrated flux.

    Leaves are canonicalized to ``(level, iy, ix)`` order on construction, so
    two skies with the same leaves and flux always compare and hash the same
    way regardless of split/merge history.
    """

    grid: QuadtreeGrid
    leaves: tuple[QuadtreeLeaf, ...]
    flux: np.ndarray

    def __post_init__(self) -> None:
        flux_array = np.asarray(self.flux, dtype=np.float64)
        if flux_array.shape != (len(self.leaves),):
            raise ValueError("flux must have exactly one value per leaf")
        if len(set(self.leaves)) != len(self.leaves):
            raise ValueError("leaves must be unique")
        for leaf in self.leaves:
            if not self.grid.contains(leaf):
                raise ValueError(f"leaf {leaf} is out of bounds for the grid")
        order = sorted(range(len(self.leaves)), key=lambda index: self.leaves[index])
        object.__setattr__(self, "leaves", tuple(self.leaves[index] for index in order))
        object.__setattr__(self, "flux", flux_array[order])

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

    def split(self, leaf: QuadtreeLeaf, child_flux: ArrayLike | None = None) -> QuadtreeSky:
        """Replace ``leaf`` with its four children, conserving total flux.

        By default each child gets one quarter of the parent's flux. Pass
        ``child_flux`` (four values, in ``children()`` order) to seed an
        unequal split, e.g. from a four-child lookahead solve; total flux is
        conserved as long as the four values sum to the parent's.
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


def predict_quadtree_stokes_i(
    sky: QuadtreeSky,
    uvw_m: ArrayLike,
    frequency_hz: ArrayLike,
    antenna1: ArrayLike,
    antenna2: ArrayLike,
    correlations: tuple[Correlation, ...],
    *,
    approximation: GaussianApproximation = GaussianApproximation.WIDE_FIELD,
    **predict_kwargs: Any,
) -> Array:
    """Predict correlations for a quadtree sky.

    Leaves are grouped by depth so every group shares one pixel width and
    can reuse the existing single-width square-pixel operator (and its
    ``_operator_factory`` cache) with one call per level, summing the
    results. This is the proposal's recommended least-invasive multi-width
    implementation: it avoids a genuinely per-component width kernel.
    """

    if not sky.leaves:
        raise ValueError("quadtree sky has no leaves")
    l, m = sky.centers()
    levels = np.asarray([leaf.level for leaf in sky.leaves])
    total: Array | None = None
    for level in sorted(set(levels.tolist())):
        mask = levels == level
        contribution = predict_stokes_i(
            sky.flux[mask],
            l[mask],
            m[mask],
            uvw_m,
            frequency_hz,
            antenna1,
            antenna2,
            correlations,
            pixel_basis=SquarePixelBasis(1.0, approximation),
            pixel_size_rad=sky.grid.leaf_width_rad(int(level)),
            **predict_kwargs,
        )
        total = contribution if total is None else total + contribution
    assert total is not None
    return total


def wide_field_error_bounds(sky: QuadtreeSky, max_w_wavelengths: float) -> np.ndarray:
    """Per-leaf analytic upper bound on the wide-field kernel's curvature error.

    Purely geometric: no visibility evaluation is needed, only each leaf's
    own width and position and the dataset's largest ``|w|`` baseline
    component. See ``square_wide_field_error_bound`` for the derivation.
    """

    l, m = sky.centers()
    widths = sky.widths_rad()
    return np.asarray(square_wide_field_error_bound(widths, l, m, max_w_wavelengths))


def leaves_exceeding_error_bound(
    sky: QuadtreeSky, max_w_wavelengths: float, tolerance: float
) -> tuple[QuadtreeLeaf, ...]:
    """Leaves whose analytic wide-field truncation error exceeds ``tolerance``.

    This is a companion to (not a replacement for) a scientific residual or
    Haar-detail split score: a leaf can be flagged here purely because the
    kernel itself has become unreliable at its current width and elevation,
    even over sky regions with no genuine structure to resolve.
    """

    bounds = wide_field_error_bounds(sky, max_w_wavelengths)
    return tuple(leaf for leaf, bound in zip(sky.leaves, bounds, strict=True) if bound > tolerance)
