from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sl1mjax.data.canonical import VisibilityBlock
from sl1mjax.data.synthetic import PointSource, simulate_dataset
from sl1mjax.diagnostics import evaluate_residuals
from sl1mjax.direct_operator import DirectDFTConfig
from sl1mjax.imaging import ImagingConfig, reconstruct
from sl1mjax.inference import InferenceConfig
from sl1mjax.output import write_products
from sl1mjax.polarization import ReceptorBasis
from sl1mjax.sky import RegularGrid
from sl1mjax.split import random_row_split


def _case() -> tuple[RegularGrid, VisibilityBlock]:
    grid = RegularGrid(9, np.deg2rad(8 / 3600))
    l, m = grid.coordinates
    centre = grid.size * grid.size // 2
    block = simulate_dataset(
        grid,
        basis=ReceptorBasis.LINEAR,
        sources=(PointSource(1.0, float(l[centre]), float(m[centre])),),
        rows=48,
        channels=2,
        noise_std=0.0,
        seed=23,
    ).blocks[0]
    return grid, block


def test_missing_point_source_appears_positive_in_every_residual_map() -> None:
    grid, block = _case()
    split = random_row_split(block, holdout_fraction=0.25, seed=4)
    evaluation = evaluate_residuals(
        block,
        np.zeros(block.shape, dtype=np.complex128),
        grid,
        split.train,
        split.holdout,
        config=DirectDFTConfig(
            visibility_chunk_size=16,
            pixel_chunk_size=32,
        ),
    )
    centre = (grid.size // 2, grid.size // 2)

    assert evaluation.full_dirty[centre] == pytest.approx(1.0, abs=1e-12)
    assert evaluation.train_dirty[centre] == pytest.approx(1.0, abs=1e-12)
    assert evaluation.holdout_dirty[centre] == pytest.approx(1.0, abs=1e-12)
    assert evaluation.psf[centre] == pytest.approx(1.0, abs=1e-12)
    peak_index = tuple(
        map(
            int,
            np.unravel_index(
                np.argmax(evaluation.full_dirty),
                evaluation.full_dirty.shape,
            ),
        )
    )
    assert peak_index == centre
    diagnostics = evaluation.diagnostics
    assert diagnostics["sign_convention"] == "observed_minus_model"
    assert diagnostics["visibility"]["full"]["normalized_residual_power"] == (
        pytest.approx(1.0)
    )
    grouped = diagnostics["visibility"]["grouped_full"]
    assert len(grouped["channels"]) == 2
    assert len(grouped["correlations"]) == 4
    assert len(grouped["worst_baselines"]) <= 10


def test_exact_prediction_has_zero_residual_images_and_statistics() -> None:
    grid, block = _case()
    split = random_row_split(block, holdout_fraction=0.25, seed=4)

    evaluation = evaluate_residuals(
        block,
        block.visibility,
        grid,
        split.train,
        split.holdout,
        config=DirectDFTConfig(
            visibility_chunk_size=16,
            pixel_chunk_size=32,
        ),
    )

    np.testing.assert_allclose(evaluation.full_dirty, 0.0, atol=1e-14)
    np.testing.assert_allclose(evaluation.train_dirty, 0.0, atol=1e-14)
    np.testing.assert_allclose(evaluation.holdout_dirty, 0.0, atol=1e-14)
    assert (
        evaluation.diagnostics["visibility"]["full"]["weighted_complex_mse"]
        == 0.0
    )


def test_standard_products_include_residual_fits_and_grouped_diagnostics(
    tmp_path: Path,
) -> None:
    grid, block = _case()
    result = reconstruct(
        block,
        ImagingConfig(
            size=grid.size,
            pixel_size_rad=grid.pixel_size_rad,
            split_strategy="random_row",
            inference=InferenceConfig(
                steps=2,
                patience=3,
                validation_interval=1,
                operator_mode="explicit",
                direct_dft=DirectDFTConfig(
                    visibility_chunk_size=16,
                    pixel_chunk_size=32,
                    precision="float32",
                ),
            ),
        ),
    )

    products = write_products(result, tmp_path / "image.fits")

    assert len(products) == 8
    assert all(path.exists() for path in products)
    diagnostics = json.loads((tmp_path / "image.json").read_text())
    assert "residual_evaluation" in diagnostics
    with np.load(tmp_path / "image.residuals.npz") as stored:
        assert {
            "full_residual_dirty",
            "train_residual_dirty",
            "holdout_residual_dirty",
            "psf",
        } <= set(stored.files)


def test_all_visibility_fit_has_no_holdout_split() -> None:
    grid, block = _case()
    result = reconstruct(
        block,
        ImagingConfig(
            size=grid.size,
            pixel_size_rad=grid.pixel_size_rad,
            holdout_fraction=0.0,
            inference=InferenceConfig(
                steps=2,
                patience=3,
                validation_interval=1,
                operator_mode="explicit",
                direct_dft=DirectDFTConfig(
                    visibility_chunk_size=16,
                    pixel_chunk_size=32,
                    precision="float32",
                ),
            ),
        ),
    )
    diagnostics = result.diagnostics()
    assert diagnostics["split"]["strategy"] == "all"
    assert diagnostics["split"]["holdout_fraction"] == 0.0
    assert result.residual_evaluation is not None
    assert result.residual_evaluation.diagnostics["visibility"]["holdout"][
        "active_count"
    ] == 0
