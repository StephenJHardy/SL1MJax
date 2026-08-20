import json
from pathlib import Path

import numpy as np
import pytest

from sl1mjax.data.canonical import VisibilityBlock
from sl1mjax.data.synthetic import default_sources, simulate_dataset
from sl1mjax.diagnostics import dirty_image_and_psf
from sl1mjax.imaging import ImagingConfig, reconstruct
from sl1mjax.inference import InferenceConfig
from sl1mjax.polarization import Correlation, ReceptorBasis
from sl1mjax.sky import RegularGrid

CASA_FIXTURE = Path(__file__).parent / "fixtures" / "casa_vla_golden.npz"


def test_casa_visibility_dirty_image_and_psf_have_matched_astrometry_and_flux() -> None:
    case = "east"
    metadata = json.loads(CASA_FIXTURE.with_suffix(".json").read_text())["cases"][case]
    prefix = f"{case}_"
    with np.load(CASA_FIXTURE) as fixture:
        visibility = fixture[prefix + "visibility"]
        rows = visibility.shape[0]
        block = VisibilityBlock(
            uvw_m=fixture[prefix + "uvw_m"],
            frequency_hz=fixture[prefix + "frequency_hz"],
            visibility=visibility,
            weight=fixture[prefix + "weight"],
            flag=fixture[prefix + "flag"],
            time_s=np.arange(rows, dtype=np.float64),
            antenna1=fixture[prefix + "antenna1"],
            antenna2=fixture[prefix + "antenna2"],
            correlations=tuple(
                Correlation(value) for value in metadata["correlations"]
            ),
            receptor_basis=ReceptorBasis(metadata["receptor_basis"]),
            phase_centre_rad=tuple(metadata["phase_centre_rad"]),
        )
    grid = RegularGrid(65, np.deg2rad(2 / 3600))
    dirty, psf = dirty_image_and_psf(block, grid, chunk_size=128)
    l, m = grid.coordinates
    source = metadata["truth"]["sources"][0]
    peak = int(np.argmax(dirty))
    astrometric_error = np.hypot(l[peak] - source["l"], m[peak] - source["m"])

    assert astrometric_error < grid.pixel_size_rad
    assert dirty.ravel()[peak] == pytest.approx(source["flux_jy"], rel=0.02)
    assert tuple(map(int, np.unravel_index(np.argmax(psf), psf.shape))) == (32, 32)
    assert psf[32, 32] == pytest.approx(1.0, abs=1e-14)


def test_reconstruction_reports_low_independent_holdout_error() -> None:
    grid = RegularGrid(8, np.deg2rad(10 / 3600))
    block = simulate_dataset(
        grid, rows=256, channels=2, seed=108
    ).blocks[0]
    result = reconstruct(
        block,
        ImagingConfig(
            size=grid.size,
            pixel_size_rad=grid.pixel_size_rad,
            inference=InferenceConfig(
                steps=160,
                learning_rate=0.12,
                sparsity_weight=1e-5,
                initial_intensity=0.02,
                patience=200,
            ),
            holdout_fraction=0.2,
            split_seed=31,
        ),
    )
    truth_peak = max(source.flux for source in default_sources(grid))

    assert result.train_loss < 1e-3
    assert result.holdout_loss < 2e-3
    assert np.max(result.image) == pytest.approx(truth_peak, rel=0.1)
    diagnostics = result.diagnostics()
    assert diagnostics["split"] == {
        "strategy": "uv_cell",
        "seed": 31,
        "holdout_fraction": 0.2,
    }
    assert diagnostics["metrics"]["holdout_weighted_complex_mse"] == result.holdout_loss
