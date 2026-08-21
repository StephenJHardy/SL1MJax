from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from sl1mjax.cli import main
from sl1mjax.data.canonical import read_dataset
from sl1mjax.inference import InferenceConfig, load_checkpoint


def test_simulate_to_canonical_zarr_and_image_products(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    canonical = tmp_path / "synthetic.zarr"
    image = tmp_path / "image.fits"

    assert (
        main(
            [
                "simulate",
                str(canonical),
                "--basis",
                "linear",
                "--size",
                "6",
                "--pixel-arcsec",
                "12",
                "--rows",
                "48",
                "--channels",
                "2",
                "--seed",
                "4",
            ]
        )
        == 0
    )
    assert capsys.readouterr().out.strip() == str(canonical)
    assert (canonical / "manifest.json").is_file()

    dataset = read_dataset(canonical)
    assert len(dataset.blocks) == 1
    block = dataset.blocks[0]
    assert block.shape == (48, 2, 4)
    assert tuple(value.value for value in block.correlations) == ("XX", "XY", "YX", "YY")
    assert block.provenance["generator"] == "sl1mjax"
    assert block.provenance["seed"] == 4

    assert (
        main(
            [
                "image",
                str(canonical),
                str(image),
                "--size",
                "6",
                "--pixel-arcsec",
                "12",
                "--steps",
                "50",
                "--learning-rate",
                "0.12",
                "--sparsity-weight",
                "0",
                "--chunk-size",
                "128",
                "--patience",
                "60",
                "--holdout-fraction",
                "0.25",
                "--split-seed",
                "9",
            ]
        )
        == 0
    )

    diagnostics_path = image.with_suffix(".json")
    residuals_path = image.with_suffix(".residuals.npz")
    checkpoint_path = image.with_suffix(".checkpoint.npz")
    evaluation_paths = tuple(
        image.with_name(f"{image.stem}.{label}.fits")
        for label in (
            "full-residual-dirty",
            "train-residual-dirty",
            "holdout-residual-dirty",
            "psf",
        )
    )
    reported = capsys.readouterr().out.strip().splitlines()
    assert reported == [
        str(image),
        str(diagnostics_path),
        str(residuals_path),
        str(checkpoint_path),
        *(str(path) for path in evaluation_paths),
    ]
    product_paths = (
        image,
        diagnostics_path,
        residuals_path,
        checkpoint_path,
        *evaluation_paths,
    )
    assert all(path.is_file() for path in product_paths)

    with fits.open(image) as hdus:
        assert hdus[0].data.shape == (6, 6)
        assert np.all(np.isfinite(hdus[0].data))
        assert np.all(hdus[0].data > 0)
        assert hdus[0].header["BUNIT"] == "Jy/pixel"
        assert hdus[0].header["CTYPE1"] == "RA---SIN"
        assert hdus[0].header["CTYPE2"] == "DEC--SIN"

    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    assert diagnostics["split"] == {
        "strategy": "uv_cell",
        "seed": 9,
            "holdout_fraction": 0.25,
    }
    assert diagnostics["correlations"] == ["XX", "XY", "YX", "YY"]
    assert diagnostics["metrics"]["steps"] == 50
    assert np.isfinite(diagnostics["metrics"]["train_weighted_complex_mse"])
    assert np.isfinite(diagnostics["metrics"]["holdout_weighted_complex_mse"])
    assert diagnostics["residual_evaluation"]["sign_convention"] == (
        "observed_minus_model"
    )

    with np.load(residuals_path) as residuals:
        assert set(residuals.files) == {
            "prediction",
            "residual",
            "correlations",
            "full_residual_dirty",
            "train_residual_dirty",
            "holdout_residual_dirty",
            "psf",
        }
        assert residuals["prediction"].shape == block.shape
        assert residuals["residual"].shape == block.shape
        np.testing.assert_allclose(
            residuals["residual"],
            residuals["prediction"] - block.visibility,
        )
        np.testing.assert_array_equal(
            residuals["correlations"],
            np.array(["XX", "XY", "YX", "YY"]),
        )

    raw, _optimizer_state, step = load_checkpoint(
        checkpoint_path,
        InferenceConfig(
            steps=50,
            learning_rate=0.12,
            sparsity_weight=0.0,
            chunk_size=128,
            patience=60,
        ),
        parameter_count=36,
    )
    assert raw.shape == (36,)
    assert np.all(np.isfinite(raw))
    assert step == 50
