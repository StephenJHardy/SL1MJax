from __future__ import annotations

import numpy as np
import pytest

from sl1mjax.data.synthetic import default_sources, simulate_dataset
from sl1mjax.imaging import ImagingConfig, reconstruct
from sl1mjax.inference import InferenceConfig
from sl1mjax.sky import PIXEL_MODEL_NAMES, RegularGrid, pixel_basis_from_name


@pytest.mark.parametrize("pixel_model", PIXEL_MODEL_NAMES)
def test_matched_pixel_mode_recovers_synthetic_grid(pixel_model: str) -> None:
    grid = RegularGrid(6, np.deg2rad(12 / 3600))
    pixel_basis = pixel_basis_from_name(pixel_model, gaussian_sigma_pixels=0.45)
    block = simulate_dataset(
        grid,
        pixel_basis=pixel_basis,
        rows=192,
        channels=2,
        seed=14,
    ).blocks[0]
    result = reconstruct(
        block,
        ImagingConfig(
            size=grid.size,
            pixel_size_rad=grid.pixel_size_rad,
            pixel_basis=pixel_basis,
            inference=InferenceConfig(
                steps=250,
                learning_rate=0.12,
                sparsity_weight=0.0,
                smoothness_weight=0.0,
                chunk_size=1024,
                patience=300,
            ),
            holdout_fraction=0.2,
            split_seed=3,
        ),
    )
    truth = np.zeros((grid.size, grid.size))
    l, m = grid.coordinates
    for source in default_sources(grid):
        index = np.argmin((l - source.l) ** 2 + (m - source.m) ** 2)
        truth.ravel()[index] = source.flux

    relative_image_error = np.linalg.norm(result.image - truth) / np.linalg.norm(truth)
    assert relative_image_error < 0.05
    assert result.holdout_loss < 2e-4
    assert block.provenance["pixel_basis"]["kind"] == pixel_basis.kind
