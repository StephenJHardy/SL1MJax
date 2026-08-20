"""Form the nearest-interpolation 3C391 residual dirty-image baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

from sl1mjax.data.canonical import VisibilityBlock, read_dataset
from sl1mjax.diagnostics import ResidualEvaluation, evaluate_residuals
from sl1mjax.direct_operator import DirectDFTConfig
from sl1mjax.objective import (
    normalized_weighted_complex_mse,
    weighted_complex_mse,
)
from sl1mjax.sky import RegularGrid
from sl1mjax.split import VisibilitySplit, random_row_split, uv_cell_split


def _split(
    block: VisibilityBlock, *, strategy: str, holdout_fraction: float, seed: int
) -> VisibilitySplit:
    if strategy == "uv_cell":
        return uv_cell_split(block, holdout_fraction=holdout_fraction, seed=seed)
    if strategy == "random_row":
        return random_row_split(block, holdout_fraction=holdout_fraction, seed=seed)
    raise ValueError("split_strategy must be uv_cell or random_row")


def _write_fits(
    image: np.ndarray,
    path: Path,
    grid: RegularGrid,
    phase_centre_rad: tuple[float, float],
    *,
    unit: str,
) -> None:
    header = fits.Header()
    header["BUNIT"] = unit
    header["CTYPE1"] = "RA---SIN"
    header["CTYPE2"] = "DEC--SIN"
    header["CRPIX1"] = (grid.size + 1) / 2
    header["CRPIX2"] = (grid.size + 1) / 2
    header["CRVAL1"] = np.rad2deg(phase_centre_rad[0])
    header["CRVAL2"] = np.rad2deg(phase_centre_rad[1])
    header["CDELT1"] = -np.rad2deg(grid.pixel_size_rad)
    header["CDELT2"] = np.rad2deg(grid.pixel_size_rad)
    fits.PrimaryHDU(np.asarray(image, dtype=np.float64), header=header).writeto(
        path, overwrite=True
    )


def _visibility_metrics(
    block: VisibilityBlock, prediction: np.ndarray, split: VisibilitySplit
) -> dict[str, float]:
    return {
        "train_weighted_complex_mse": float(
            weighted_complex_mse(prediction, block.visibility, block.weight, ~split.train)
        ),
        "holdout_weighted_complex_mse": float(
            weighted_complex_mse(
                prediction, block.visibility, block.weight, ~split.holdout
            )
        ),
        "train_normalized_weighted_complex_mse": float(
            normalized_weighted_complex_mse(
                prediction, block.visibility, block.weight, ~split.train
            )
        ),
        "holdout_normalized_weighted_complex_mse": float(
            normalized_weighted_complex_mse(
                prediction, block.visibility, block.weight, ~split.holdout
            )
        ),
    }


def _case_metrics(
    block: VisibilityBlock,
    prediction: np.ndarray,
    evaluation: ResidualEvaluation,
    split: VisibilitySplit,
) -> dict[str, object]:
    return {
        **_visibility_metrics(block, prediction, split),
        "residual_evaluation": evaluation.diagnostics,
    }


def _plot_residual_dirty(
    casa: np.ndarray,
    nearest: np.ndarray,
    path: Path,
) -> None:
    difference = nearest - casa
    limit = float(
        np.percentile(np.abs(np.concatenate((casa.ravel(), nearest.ravel()))), 99.5)
    )
    difference_limit = float(np.percentile(np.abs(difference), 99.5))
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for axis, image, label in zip(
        axes[:2],
        (casa, nearest),
        ("CASA corrected residual dirty", "Nearest-gain residual dirty"),
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
    axes[2].set_title("Nearest − CASA")
    figure.colorbar(shown, ax=axes[2], fraction=0.046)
    figure.suptitle("3C391 C1 residual dirty image, nearest-gain baseline")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _evaluate_case(
    block: VisibilityBlock,
    prediction: np.ndarray,
    grid: RegularGrid,
    *,
    strategy: str,
    holdout_fraction: float,
    seed: int,
    config: DirectDFTConfig,
) -> tuple[ResidualEvaluation, VisibilitySplit]:
    if prediction.shape != block.shape:
        raise ValueError(
            f"prediction shape {prediction.shape} does not match block {block.shape}"
        )
    split = _split(
        block, strategy=strategy, holdout_fraction=holdout_fraction, seed=seed
    )
    evaluation = evaluate_residuals(
        block, prediction, grid, split.train, split.holdout, config=config
    )
    return evaluation, split


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "fixture",
        type=Path,
        nargs="?",
        default=Path("outputs/3c391_imaging_fixture.zarr"),
    )
    parser.add_argument(
        "--reconstruction",
        type=Path,
        default=Path("outputs/3c391_explicit_full"),
        help="Directory containing nearest-gain reconstruction residual NPZ files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/3c391_interpolation_baseline"),
    )
    parser.add_argument("--size", type=int, default=96)
    parser.add_argument("--pixel-arcsec", type=float, default=4.0)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=17)
    parser.add_argument(
        "--split-strategy",
        choices=("uv_cell", "random_row"),
        default="uv_cell",
    )
    parser.add_argument("--precision", choices=("float32", "float64"), default="float32")
    parser.add_argument("--visibility-tile-size", type=int, default=256)
    parser.add_argument("--pixel-tile-size", type=int, default=1024)
    arguments = parser.parse_args()

    dataset = read_dataset(arguments.fixture)
    if len(dataset.blocks) != 2:
        raise ValueError("3C391 imaging fixture must contain exactly two blocks")
    if dataset.provenance.get("case_order") != ["casa_corrected", "jax_calibrated"]:
        raise ValueError("fixture case_order must be casa_corrected then jax_calibrated")
    casa_prediction = np.asarray(
        np.load(arguments.reconstruction / "casa_corrected_reconstruction.residuals.npz")[
            "prediction"
        ]
    )
    nearest_prediction = np.asarray(
        np.load(arguments.reconstruction / "jax_calibrated_reconstruction.residuals.npz")[
            "prediction"
        ]
    )
    grid = RegularGrid(arguments.size, np.deg2rad(arguments.pixel_arcsec / 3600))
    dft = DirectDFTConfig(
        visibility_chunk_size=arguments.visibility_tile_size,
        pixel_chunk_size=arguments.pixel_tile_size,
        precision=arguments.precision,
    )
    casa_eval, casa_split = _evaluate_case(
        dataset.blocks[0],
        casa_prediction,
        grid,
        strategy=arguments.split_strategy,
        holdout_fraction=arguments.holdout_fraction,
        seed=arguments.split_seed,
        config=dft,
    )
    nearest_eval, nearest_split = _evaluate_case(
        dataset.blocks[1],
        nearest_prediction,
        grid,
        strategy=arguments.split_strategy,
        holdout_fraction=arguments.holdout_fraction,
        seed=arguments.split_seed,
        config=dft,
    )
    arguments.output.mkdir(parents=True, exist_ok=True)
    for stem, evaluation, phase_centre in (
        (
            "casa_corrected",
            casa_eval,
            dataset.blocks[0].phase_centre_rad,
        ),
        (
            "nearest_gain",
            nearest_eval,
            dataset.blocks[1].phase_centre_rad,
        ),
    ):
        for label, image, unit in (
            ("full-residual-dirty", evaluation.full_dirty, "Jy/beam"),
            ("train-residual-dirty", evaluation.train_dirty, "Jy/beam"),
            ("holdout-residual-dirty", evaluation.holdout_dirty, "Jy/beam"),
            ("psf", evaluation.psf, "1"),
        ):
            _write_fits(
                image,
                arguments.output / f"{stem}.{label}.fits",
                grid,
                phase_centre,
                unit=unit,
            )
        np.savez(
            arguments.output / f"{stem}.residuals.npz",
            full_residual_dirty=evaluation.full_dirty,
            train_residual_dirty=evaluation.train_dirty,
            holdout_residual_dirty=evaluation.holdout_dirty,
            psf=evaluation.psf,
        )
    _plot_residual_dirty(
        casa_eval.full_dirty,
        nearest_eval.full_dirty,
        arguments.output / "residual_dirty_comparison.png",
    )
    summary = {
        "role": "nearest-gain interpolation baseline",
        "interpolation": "nearest",
        "fixture": arguments.fixture.name,
        "reconstruction": arguments.reconstruction.name,
        "fixture_provenance": dict(dataset.provenance),
        "block_shape": list(dataset.blocks[1].shape),
        "configuration": {
            "size": arguments.size,
            "pixel_arcsec": arguments.pixel_arcsec,
            "holdout_fraction": arguments.holdout_fraction,
            "split_strategy": arguments.split_strategy,
            "split_seed": arguments.split_seed,
            "precision": arguments.precision,
            "visibility_tile_size": arguments.visibility_tile_size,
            "pixel_tile_size": arguments.pixel_tile_size,
        },
        "casa_corrected": _case_metrics(
            dataset.blocks[0], casa_prediction, casa_eval, casa_split
        ),
        "nearest_gain": _case_metrics(
            dataset.blocks[1], nearest_prediction, nearest_eval, nearest_split
        ),
    }
    (arguments.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
