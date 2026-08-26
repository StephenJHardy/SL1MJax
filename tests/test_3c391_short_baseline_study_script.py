from __future__ import annotations

import importlib.util
from dataclasses import replace
from pathlib import Path

import numpy as np

from sl1mjax.data.canonical import VisibilityBlock
from sl1mjax.polarization import Correlation, ReceptorBasis
from sl1mjax.quadtree import quadtree_sky_from_regular_grid

SCRIPT = Path(__file__).parents[1] / "scripts" / "study_3c391_short_baselines.py"


def _module():
    spec = importlib.util.spec_from_file_location("short_baseline_study", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _block() -> VisibilityBlock:
    rows = 10
    shape = (rows, 1, 1)
    uv_klambda = np.asarray([0.5, 0.6, 1.0, 1.2, 2.0, 2.5, 4.0, 5.0, 8.0, 10.0])
    uv_m = uv_klambda * 1000.0 * 299_792_458.0 / 1.0e9
    return VisibilityBlock(
        uvw_m=np.column_stack(
            (uv_m, np.zeros(rows), np.zeros(rows))
        ),
        frequency_hz=np.asarray([1.0e9]),
        visibility=np.ones(shape, dtype=np.complex128),
        weight=np.ones(shape),
        flag=np.zeros(shape, dtype=bool),
        time_s=np.arange(rows, dtype=np.float64),
        antenna1=np.zeros(rows, dtype=np.int32),
        antenna2=np.ones(rows, dtype=np.int32),
        correlations=(Correlation.I,),
        receptor_basis=ReceptorBasis.STOKES,
    )


def test_time_half_masks_are_disjoint_and_cover_active_samples() -> None:
    module = _module()
    block = _block()

    first, second, boundaries = module.time_half_masks((block,))

    assert boundaries == (4.5,)
    assert not np.any(first[0] & second[0])
    np.testing.assert_array_equal(first[0] | second[0], block.active)
    assert np.count_nonzero(first[0]) == np.count_nonzero(second[0]) == 5


def test_interleaved_masks_keep_complete_time_bins_together() -> None:
    module = _module()
    block = _block()

    even, odd, boundaries = module.interleaved_time_bin_masks(
        (block,),
        bin_seconds=2.0,
    )

    assert boundaries == (None,)
    assert not np.any(even[0] & odd[0])
    np.testing.assert_array_equal(even[0] | odd[0], block.active)
    for time_bin in np.unique(np.floor(block.time_s / 2.0).astype(int)):
        rows = np.floor(block.time_s / 2.0).astype(int) == time_bin
        assert np.all(even[0][rows]) or np.all(odd[0][rows])


def test_extended_airy_beam_preserves_farther_sidelobes() -> None:
    module = _module()
    frequency = np.asarray([4.6e9])
    radius = np.sin(np.deg2rad(20.0 / 60.0))

    standard = module._beam("airy").power(radius, 0.0, frequency)
    extended = module._beam("airy_extended").power(radius, 0.0, frequency)

    assert standard[0] == 0.0
    assert extended[0] > 0.0


def test_metrics_detects_an_exact_candidate_prediction() -> None:
    module = _module()
    block = _block()
    mask = block.active
    wrong = np.zeros(block.shape, dtype=np.complex128)
    exact = block.visibility.copy()

    comparison = module._comparison((block,), (wrong,), (exact,), (mask,))

    assert comparison["all"]["base"]["weighted_complex_mse"] == 1.0
    assert comparison["all"]["candidate"]["weighted_complex_mse"] == 0.0
    assert comparison["all"]["relative_weighted_complex_mse_change"] == -1.0


def test_positive_visibility_atom_recovers_real_flux() -> None:
    module = _module()
    block = _block()
    base = np.zeros(block.shape, dtype=np.complex128)
    atom = np.full(block.shape, 0.5 + 0.25j)
    expected_flux = 1.7
    observed = replace(block, visibility=base + expected_flux * atom)

    fitted = module.fit_positive_visibility_atom(
        (observed,),
        (base,),
        (atom,),
        (observed.active,),
    )

    np.testing.assert_allclose(fitted, expected_flux)


def test_mosaic_beam_sensitivity_weights_normalize_detectable_flux() -> None:
    module = _module()
    block = replace(_block(), frequency_hz=np.asarray([4.6e9]))
    sky = quadtree_sky_from_regular_grid(
        32,
        np.deg2rad(60.0 / 3600.0),
        np.zeros(32**2),
    )

    weights = module.mosaic_beam_sensitivity_weights(
        (block,),
        sky.topology,
        (block.active,),
        (0.0, 0.0),
        module._beam("airy"),
    )

    assert weights.shape == (32**2,)
    assert np.max(weights) == 1.0
    assert weights[0] == 0.0
    assert weights[32 * 16 + 16] > 0.9
