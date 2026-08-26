from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from sl1mjax.data.canonical import VisibilityBlock
from sl1mjax.polarization import Correlation, ReceptorBasis
from sl1mjax.quadtree import quadtree_sky_from_regular_grid

SCRIPT = Path(__file__).parents[1] / "scripts" / "fit_3c391_composite.py"


def _module():
    spec = importlib.util.spec_from_file_location("fit_3c391_composite", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _block() -> VisibilityBlock:
    rows = 20
    shape = (rows, 1, 1)
    return VisibilityBlock(
        uvw_m=np.column_stack(
            (
                np.linspace(10.0, 1_000.0, rows),
                np.zeros(rows),
                np.zeros(rows),
            )
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
        phase_centre_rad=(4.2, -0.3),
    )


def test_five_fold_time_masks_are_disjoint_complete_and_bin_coherent() -> None:
    module = _module()
    block = _block()

    train, validation, test = module.interleaved_time_fold_masks(
        (block,),
        bin_seconds=2.0,
    )

    assert not np.any(train[0] & validation[0])
    assert not np.any(train[0] & test[0])
    assert not np.any(validation[0] & test[0])
    np.testing.assert_array_equal(train[0] | validation[0] | test[0], block.active)
    assert np.count_nonzero(train[0]) == 12
    assert np.count_nonzero(validation[0]) == 4
    assert np.count_nonzero(test[0]) == 4
    time_bin = np.floor(block.time_s / 2.0).astype(int)
    for value in np.unique(time_bin):
        rows = time_bin == value
        membership = (
            np.all(train[0][rows]),
            np.all(validation[0][rows]),
            np.all(test[0][rows]),
        )
        assert sum(membership) == 1


def test_component_templates_cover_all_model_variants() -> None:
    module = _module()
    block = _block()
    central = quadtree_sky_from_regular_grid(2, 2e-4, np.zeros(4)).topology
    source = module.RadioCatalogSource(
        name="outer",
        ra_deg=282.4,
        dec_deg=-0.6,
        reference_frequency_hz=1.4e9,
        integrated_flux_jy=0.7,
        catalog="test",
        reference_url="https://example.invalid/catalog",
    )
    catalog_atoms = (
        module.CatalogGuardAtom(
            source=source,
            l_rad=0.01,
            m_rad=0.02,
            offset_arcmin=75.0,
            initial_flux_jy=0.3,
            maximum_apparent_flux_jy=0.01,
            maximum_beam_power=0.03,
        ),
    )

    templates = module._component_templates(
        central,
        block.phase_centre_rad,
        coarse_size=4,
        coarse_pixel_arcsec=60.0,
        catalog_atoms=catalog_atoms,
    )

    assert set(templates) == {"central", "coarse", "catalogue"}
    assert templates["central"].flux.shape == (4,)
    assert templates["coarse"].flux.shape == (16,)
    assert templates["catalogue"].flux.shape == (1,)
    assert templates["catalogue"].l_rad[0] == 0.01
    assert templates["catalogue"].m_rad[0] == 0.02
    assert templates["catalogue"].flux[0] == 0.3
    assert module.VARIANT_COMPONENTS["full"] == (
        "central",
        "coarse",
        "catalogue",
    )

    fitted_catalogue = module.MosaicPointComponent(
        "catalogue",
        templates["catalogue"].l_rad,
        templates["catalogue"].m_rad,
        np.asarray([0.25]),
    )
    assert module._catalogue_flux_payload((fitted_catalogue,), catalog_atoms) == {"outer": 0.25}
    assert module._catalogue_flux_payload((templates["central"],), catalog_atoms) == {}


def test_metrics_use_only_the_requested_partition() -> None:
    module = _module()
    block = _block()
    prediction = block.visibility.copy()
    prediction[10:] = 0.0
    first = np.zeros(block.shape, dtype=bool)
    first[:10] = True
    second = np.zeros(block.shape, dtype=bool)
    second[10:] = True

    exact = module._metrics((block,), (prediction,), (first,))
    wrong = module._metrics((block,), (prediction,), (second,))

    assert exact["weighted_complex_mse"] == 0.0
    assert wrong["weighted_complex_mse"] == 1.0


def test_initial_candidate_loads_flux_without_predictions(tmp_path) -> None:
    module = _module()
    central = quadtree_sky_from_regular_grid(2, 2e-4, np.zeros(4)).topology
    component = module.MosaicQuadtreeComponent("central", central, np.zeros(4))
    np.savez(tmp_path / "full_lambda_0.0003.npz", flux_central=np.arange(4.0))

    loaded = module._load_initial_components(
        tmp_path,
        "full_lambda_0.0003",
        (component,),
    )

    np.testing.assert_array_equal(loaded[0].flux, np.arange(4.0))
