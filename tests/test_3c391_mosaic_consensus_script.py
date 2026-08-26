from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import numpy as np

from sl1mjax.quadtree import QuadtreeGrid, QuadtreeLeaf, QuadtreeTopology

_SCRIPT = Path(__file__).parents[1] / "scripts/refit_3c391_mosaic_consensus.py"
_SPEC = importlib.util.spec_from_file_location("refit_3c391_mosaic_consensus", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
refit_script = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(refit_script)


def test_load_sky_aligns_csv_flux_with_sorted_topology(tmp_path: Path) -> None:
    grid = QuadtreeGrid(2, 2e-4)
    parent = QuadtreeLeaf(0, 0, 0)
    leaves = tuple(leaf for leaf in grid.root_leaves() if leaf != parent) + parent.children()
    topology = QuadtreeTopology(grid, leaves)
    expected = {leaf: float(index + 1) / 10 for index, leaf in enumerate(topology.leaves)}
    path = tmp_path / "topology.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("level", "iy", "ix", "flux_jy"))
        for leaf in reversed(topology.leaves):
            writer.writerow((leaf.level, leaf.iy, leaf.ix, expected[leaf]))

    sky = refit_script._load_sky(
        path,
        root_size=2,
        root_pixel_size_rad=2e-4,
    )

    assert sky.topology == topology
    np.testing.assert_allclose(sky.flux, [expected[leaf] for leaf in topology.leaves])
