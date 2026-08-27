from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

from sl1mjax.beam import VLAPrimaryBeam
from sl1mjax.data.canonical import VisibilityBlock
from sl1mjax.direct_operator import DirectDFTConfig
from sl1mjax.polarization import Correlation, ReceptorBasis
from sl1mjax.quadtree import (
    QuadtreeGrid,
    QuadtreeTopology,
    predict_quadtree_stokes_i_explicit,
)

SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "search_3c391_native_spatial_variability.py"
)


def _module():
    spec = importlib.util.spec_from_file_location(
        "search_3c391_native_spatial_variability",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _block() -> VisibilityBlock:
    rows = 8
    shape = (rows, 3, 1)
    return VisibilityBlock(
        uvw_m=np.column_stack(
            (
                np.linspace(-100.0, 100.0, rows),
                np.linspace(50.0, -80.0, rows),
                np.linspace(-20.0, 30.0, rows),
            )
        ),
        frequency_hz=4.5e9 + np.arange(shape[1]) * 2e6,
        visibility=np.zeros(shape, dtype=np.complex128),
        model_visibility=np.zeros(shape, dtype=np.complex128),
        weight=np.ones(shape),
        flag=np.zeros(shape, dtype=bool),
        time_s=np.repeat(np.arange(4, dtype=np.float64) * 10.0, 2),
        interval_s=np.full(rows, 10.0),
        antenna1=np.tile([0, 1], 4),
        antenna2=np.tile([1, 2], 4),
        correlations=(Correlation.RR,),
        receptor_basis=ReceptorBasis.CIRCULAR,
    )


def test_spatial_prefilter_is_holdout_independent_apparent_flux_ranking() -> None:
    module = _module()
    block = _block()
    grid = QuadtreeGrid(3, 8e-4)
    topology = QuadtreeTopology(grid, grid.root_leaves())
    flux = np.asarray([0.01, 0.2, 0.03, 0.1, 0.05, 0.0, 0.02, 0.08, 0.04])

    selected, metadata = module._select_spatial_candidates(
        topology,
        flux,
        block,
        VLAPrimaryBeam(),
        mosaic_phase_centre_rad=block.phase_centre_rad,
        maximum_candidates=4,
        minimum_static_flux_jy=0.01,
    )

    apparent = [metadata[leaf]["apparent_static_flux_jy"] for leaf in selected]
    assert len(selected) == 4
    assert apparent == sorted(apparent, reverse=True)
    assert all(metadata[leaf]["static_flux_jy"] >= 0.01 for leaf in selected)


def test_single_leaf_injection_response_matches_full_topology_atom() -> None:
    module = _module()
    block = _block()
    grid = QuadtreeGrid(2, 7e-4)
    topology = QuadtreeTopology(grid, grid.root_leaves())
    leaf = topology.leaves[2]
    direct = DirectDFTConfig(
        visibility_chunk_size=4,
        pixel_chunk_size=4,
        precision="float64",
    )
    beam = VLAPrimaryBeam()

    isolated = module._unit_leaf_response(
        block,
        topology,
        leaf,
        beam,
        block.phase_centre_rad,
        direct,
    )
    l_rad, m_rad = topology.centers()
    beam_weights = beam.power_weights(l_rad, m_rad, block.frequency_hz)
    flux = np.zeros(len(topology.leaves))
    flux[2] = 1.0
    full = np.asarray(
        predict_quadtree_stokes_i_explicit(
            flux,
            topology,
            block.uvw_m,
            block.frequency_hz,
            block.antenna1,
            block.antenna2,
            block.correlations,
            config=direct,
            beam_weights=beam_weights,
        )
    )

    np.testing.assert_allclose(isolated, full, rtol=1e-12, atol=1e-12)
