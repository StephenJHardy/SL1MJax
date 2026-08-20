"""Score CASA CLEAN as a visibility predictor against the 3C391 fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

from sl1mjax.data.canonical import VisibilityBlock, read_dataset
from sl1mjax.diagnostics import ResidualEvaluation, evaluate_residuals
from sl1mjax.direct_operator import DirectDFTConfig, predict_stokes_i_explicit
from sl1mjax.objective import (
    normalized_weighted_complex_mse,
    weighted_complex_mse,
)
from sl1mjax.sky import DeltaPixelBasis, RegularGrid
from sl1mjax.split import VisibilitySplit, random_row_split, uv_cell_split


def _split(
    block: VisibilityBlock, *, strategy: str, holdout_fraction: float, seed: int
) -> VisibilitySplit:
    if holdout_fraction == 0:
        return VisibilitySplit(
            block.active.copy(), np.zeros(block.shape, dtype=bool), "all"
        )
    if strategy == "uv_cell":
        return uv_cell_split(block, holdout_fraction=holdout_fraction, seed=seed)
    if strategy == "random_row":
        return random_row_split(block, holdout_fraction=holdout_fraction, seed=seed)
    raise ValueError("split_strategy must be uv_cell or random_row")


def _visibility_metrics(
    block: VisibilityBlock, prediction: np.ndarray, split: VisibilitySplit
) -> dict[str, float]:
    holdout_mse = (
        float("nan")
        if not np.any(split.holdout)
        else float(
            weighted_complex_mse(
                prediction, block.visibility, block.weight, ~split.holdout
            )
        )
    )
    holdout_normalized = (
        float("nan")
        if not np.any(split.holdout)
        else float(
            normalized_weighted_complex_mse(
                prediction, block.visibility, block.weight, ~split.holdout
            )
        )
    )
    return {
        "train_weighted_complex_mse": float(
            weighted_complex_mse(
                prediction, block.visibility, block.weight, ~split.train
            )
        ),
        "holdout_weighted_complex_mse": holdout_mse,
        "train_normalized_weighted_complex_mse": float(
            normalized_weighted_complex_mse(
                prediction, block.visibility, block.weight, ~split.train
            )
        ),
        "holdout_normalized_weighted_complex_mse": holdout_normalized,
    }


def _predict_image(
    image: np.ndarray,
    block: VisibilityBlock,
    grid: RegularGrid,
    config: DirectDFTConfig,
) -> np.ndarray:
    l, m = grid.coordinates
    return np.asarray(
        predict_stokes_i_explicit(
            image.ravel(),
            l,
            m,
            block.uvw_m,
            block.frequency_hz,
            block.antenna1,
            block.antenna2,
            block.correlations,
            pixel_basis=DeltaPixelBasis(),
            pixel_size_rad=grid.pixel_size_rad,
            config=config,
        )
    )


def _plot(
    casa_clean: np.ndarray,
    sl1mjax: np.ndarray,
    path: Path,
) -> None:
    difference = sl1mjax - casa_clean
    limit = float(
        np.percentile(
            np.abs(np.concatenate((casa_clean.ravel(), sl1mjax.ravel()))),
            99.5,
        )
    )
    difference_limit = float(np.percentile(np.abs(difference), 99.5))
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for axis, image, label in zip(
        axes[:2],
        (casa_clean, sl1mjax),
        ("CASA multiscale model residual dirty", "SL1MJax residual dirty"),
        strict=True,
    ):
        shown = axis.imshow(
            image, origin="lower", cmap="coolwarm", vmin=-limit, vmax=limit
        )
        axis.set_title(label)
        figure.colorbar(shown, ax=axis, fraction=0.046)
    shown = axes[2].imshow(
        difference,
        origin="lower",
        cmap="coolwarm",
        vmin=-difference_limit,
        vmax=difference_limit,
    )
    axes[2].set_title("SL1MJax − CASA CLEAN")
    figure.colorbar(shown, ax=axes[2], fraction=0.046)
    figure.suptitle("3C391 C1 residual dirty images from visibility models")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "fixture",
        type=Path,
        nargs="?",
        default=Path("outputs/3c391_imaging_fixture.zarr"),
    )
    parser.add_argument(
        "--casa-model",
        type=Path,
        default=Path(
            "outputs/3c391_casa_imaging_128/3c391_c1_multiscale.model.fits"
        ),
    )
    parser.add_argument(
        "--sl1mjax-reconstruction",
        type=Path,
        default=Path(
            "outputs/3c391_regularization_best_random/"
            "casa_corrected_reconstruction.fits"
        ),
    )
    parser.add_argument(
        "--casa-model-visibilities",
        type=Path,
        help="Optional NPZ with CASA MODEL_DATA predictions matching the fixture.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/3c391_casa_clean_visibility_score"),
    )
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--pixel-arcsec", type=float, default=4.0)
    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=0.0,
        help="Visibility split for scoring. Use 0 to score all visibilities.",
    )
    parser.add_argument("--split-seed", type=int, default=17)
    parser.add_argument(
        "--split-strategy",
        choices=("uv_cell", "random_row"),
        default="random_row",
    )
    parser.add_argument("--visibility-tile-size", type=int, default=4096)
    parser.add_argument("--pixel-tile-size", type=int, default=4096)
    arguments = parser.parse_args()

    dataset = read_dataset(arguments.fixture)
    block = dataset.blocks[0]
    grid = RegularGrid(arguments.size, np.deg2rad(arguments.pixel_arcsec / 3600))
    config = DirectDFTConfig(
        visibility_chunk_size=arguments.visibility_tile_size,
        pixel_chunk_size=arguments.pixel_tile_size,
        precision="float32",
    )
    with fits.open(arguments.casa_model) as hdus:
        casa_model = np.asarray(np.squeeze(hdus[0].data), dtype=np.float64)
    with fits.open(arguments.sl1mjax_reconstruction) as hdus:
        sl1mjax_image = np.asarray(np.squeeze(hdus[0].data), dtype=np.float64)
    if casa_model.shape != (arguments.size, arguments.size):
        raise ValueError(
            f"CASA model shape {casa_model.shape} does not match {arguments.size}"
        )
    if sl1mjax_image.shape != casa_model.shape:
        raise ValueError("SL1MJax reconstruction must match the CASA model grid")

    split = _split(
        block,
        strategy=arguments.split_strategy,
        holdout_fraction=arguments.holdout_fraction,
        seed=arguments.split_seed,
    )
    casa_prediction = _predict_image(casa_model, block, grid, config)
    sl1mjax_prediction = _predict_image(sl1mjax_image, block, grid, config)
    predictions = {
        "casa_multiscale_model_dft": casa_prediction,
        "sl1mjax_regularized": sl1mjax_prediction,
    }
    if arguments.casa_model_visibilities is not None:
        stored = np.load(arguments.casa_model_visibilities)
        predictions["casa_modelcolumn"] = np.asarray(stored["prediction"])

    evaluations: dict[str, ResidualEvaluation] = {}
    metrics: dict[str, object] = {}
    for name, prediction in predictions.items():
        if prediction.shape != block.shape:
            raise ValueError(f"{name} prediction shape {prediction.shape} != {block.shape}")
        evaluation = evaluate_residuals(
            block, prediction, grid, split.train, split.holdout, config=config
        )
        evaluations[name] = evaluation
        metrics[name] = {
            "image_total_flux_jy": (
                float(np.sum(casa_model))
                if name.startswith("casa")
                else float(np.sum(sl1mjax_image))
            ),
            **_visibility_metrics(block, prediction, split),
            "residual_evaluation": evaluation.diagnostics,
        }

    arguments.output.mkdir(parents=True, exist_ok=True)
    np.savez(
        arguments.output / "predictions.npz",
        **{name: value for name, value in predictions.items()},
    )
    for name, evaluation in evaluations.items():
        np.savez(
            arguments.output / f"{name}.residuals.npz",
            full_residual_dirty=evaluation.full_dirty,
            train_residual_dirty=evaluation.train_dirty,
            holdout_residual_dirty=evaluation.holdout_dirty,
            psf=evaluation.psf,
        )
    _plot(
        evaluations["casa_multiscale_model_dft"].full_dirty,
        evaluations["sl1mjax_regularized"].full_dirty,
        arguments.output / "residual_dirty_comparison.png",
    )
    summary = {
        "fixture": arguments.fixture.name,
        "casa_model": arguments.casa_model.name,
        "sl1mjax_reconstruction": arguments.sl1mjax_reconstruction.name,
        "split_strategy": arguments.split_strategy,
        "holdout_fraction": arguments.holdout_fraction,
        "models": metrics,
    }
    (arguments.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
