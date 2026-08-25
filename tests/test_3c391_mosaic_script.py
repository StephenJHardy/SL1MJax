from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import numpy as np

from sl1mjax.quadtree import (
    QuadtreeGrid,
    QuadtreeLeaf,
    QuadtreeTopology,
)

_SCRIPT = Path(__file__).parents[1] / "scripts/image_3c391_mosaic.py"
_SPEC = importlib.util.spec_from_file_location("image_3c391_mosaic", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
mosaic_script = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mosaic_script)


def test_embedded_topology_preserves_leaf_centres(tmp_path: Path) -> None:
    source_grid = QuadtreeGrid(2, 2e-4)
    source_leaves = tuple(
        leaf
        for leaf in source_grid.root_leaves()
        if leaf != QuadtreeLeaf(0, 0, 0)
    ) + QuadtreeLeaf(0, 0, 0).children()
    source = QuadtreeTopology(source_grid, source_leaves)
    topology_path = tmp_path / "topology.csv"
    with topology_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("level", "iy", "ix"))
        for leaf in source.leaves:
            writer.writerow((leaf.level, leaf.iy, leaf.ix))

    embedded = mosaic_script._load_embedded_topology(
        topology_path,
        source_root_size=2,
        destination_root_size=4,
        root_pixel_size_rad=2e-4,
    )

    assert len(embedded.leaves) == len(source.leaves) + 12
    source_centres = set(zip(*source.centers(), strict=True))
    embedded_centres = set(zip(*embedded.centers(), strict=True))
    assert source_centres <= embedded_centres


def test_hierarchical_initial_flux_conserves_each_root_flux() -> None:
    grid = QuadtreeGrid(2, 2e-4)
    base = QuadtreeTopology(grid, grid.root_leaves())
    split_parent = QuadtreeLeaf(0, 0, 0)
    hierarchy = QuadtreeTopology(
        grid,
        tuple(leaf for leaf in base.leaves if leaf != split_parent)
        + split_parent.children(),
    )
    base_flux = np.asarray([0.8, 0.2, 0.1, 0.4])

    initial = mosaic_script._initial_hierarchical_flux(
        base,
        base_flux,
        hierarchy,
    )

    assert np.sum(initial) == np.sum(base_flux)
    by_leaf = dict(zip(hierarchy.leaves, initial, strict=True))
    for child in split_parent.children():
        assert by_leaf[child] == 0.2
