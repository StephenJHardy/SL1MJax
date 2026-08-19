"""Run the larger 3C391 comparison from a portable averaged fixture."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sl1mjax.data.canonical import VisibilityBlock, read_dataset
from sl1mjax.direct_operator import DirectDFTConfig
from sl1mjax.imaging import ImagingConfig, ImagingResult, reconstruct
from sl1mjax.inference import InferenceConfig
from sl1mjax.output import write_products
from sl1mjax.sky import PIXEL_MODEL_NAMES, pixel_basis_from_name


def _select_even_rows(
    block: VisibilityBlock, maximum_rows: int | None
) -> VisibilityBlock:
    if maximum_rows is None or maximum_rows >= block.shape[0]:
        return block
    if maximum_rows < 1:
        raise ValueError("maximum_rows must be positive")
    rows = np.linspace(
        0, block.shape[0] - 1, maximum_rows, dtype=np.int64
    )
    field_id = block.field_id
    scan_id = block.scan_id
    state_id = block.state_id
    observation_id = block.observation_id
    feed1 = block.feed1
    feed2 = block.feed2
    interval_s = block.interval_s
    assert field_id is not None
    assert scan_id is not None
    assert state_id is not None
    assert observation_id is not None
    assert feed1 is not None
    assert feed2 is not None
    assert interval_s is not None
    return replace(
        block,
        uvw_m=block.uvw_m[rows],
        visibility=block.visibility[rows],
        weight=block.weight[rows],
        flag=block.flag[rows],
        time_s=block.time_s[rows],
        antenna1=block.antenna1[rows],
        antenna2=block.antenna2[rows],
        field_id=field_id[rows],
        scan_id=scan_id[rows],
        state_id=state_id[rows],
        observation_id=observation_id[rows],
        feed1=feed1[rows],
        feed2=feed2[rows],
        interval_s=interval_s[rows],
    )


def _image_metrics(
    actual: np.ndarray, reference: np.ndarray
) -> dict[str, float]:
    normalized_rms = np.sqrt(
        np.sum(np.square(actual - reference))
        / np.maximum(np.sum(np.square(reference)), np.finfo(float).tiny)
    )
    return {
        "normalized_rms": float(normalized_rms),
        "correlation": float(
            np.corrcoef(actual.ravel(), reference.ravel())[0, 1]
        ),
        "reference_peak": float(np.max(reference)),
        "actual_peak": float(np.max(actual)),
        "reference_total_flux": float(np.sum(reference)),
        "actual_total_flux": float(np.sum(actual)),
    }


def _plot(
    casa: np.ndarray, jax_calibrated: np.ndarray, path: Path
) -> None:
    difference = jax_calibrated - casa
    limit = float(
        np.percentile(
            np.abs(np.concatenate((casa.ravel(), jax_calibrated.ravel()))),
            99.5,
        )
    )
    difference_limit = float(np.percentile(np.abs(difference), 99.5))
    figure, axes = plt.subplots(
        1, 3, figsize=(14, 4.5), constrained_layout=True
    )
    for axis, image, label in zip(
        axes[:2],
        (casa, jax_calibrated),
        ("CASA corrected", "SL1MJax calibrated"),
        strict=True,
    ):
        shown = axis.imshow(
            image,
            origin="lower",
            cmap="inferno",
            vmin=-0.05 * limit,
            vmax=limit,
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
    axes[2].set_title("SL1MJax − CASA")
    figure.colorbar(shown, ax=axes[2], fraction=0.046)
    figure.suptitle("3C391 C1 full averaged explicit reconstruction")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _result_metrics(result: ImagingResult) -> dict[str, float | int | bool]:
    return {
        "elapsed_s": result.elapsed_s,
        "steps": result.inference.steps,
        "best_step": result.inference.best_step,
        "converged": result.inference.converged,
        "train_loss": result.train_loss,
        "holdout_loss": result.holdout_loss,
        "train_normalized_loss": result.train_normalized_loss,
        "holdout_normalized_loss": result.holdout_normalized_loss,
        "peak_flux": float(np.max(result.image)),
        "total_flux": float(np.sum(result.image)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("fixture", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/3c391_explicit_full"),
    )
    parser.add_argument("--size", type=int, default=96)
    parser.add_argument("--pixel-arcsec", type=float, default=4.0)
    parser.add_argument("--maximum-rows", type=int)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--sparsity-weight", type=float, default=1e-4)
    parser.add_argument("--smoothness-weight", type=float, default=0.0)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument(
        "--split-strategy",
        choices=("uv_cell", "random_row"),
        default="uv_cell",
    )
    parser.add_argument("--validation-interval", type=int, default=10)
    parser.add_argument("--pixel-model", choices=PIXEL_MODEL_NAMES, default="delta")
    parser.add_argument("--gaussian-sigma-pixels", type=float, default=0.5)
    parser.add_argument("--precision", choices=("float32", "float64"), default="float32")
    parser.add_argument("--visibility-tile-size", type=int, default=256)
    parser.add_argument("--pixel-tile-size", type=int, default=1024)
    arguments = parser.parse_args()

    dataset = read_dataset(arguments.fixture)
    if len(dataset.blocks) != 2:
        raise ValueError("3C391 imaging fixture must contain exactly two blocks")
    expected_order = ["casa_corrected", "jax_calibrated"]
    if dataset.provenance.get("case_order") != expected_order:
        raise ValueError(f"fixture case_order must be {expected_order}")
    casa = _select_even_rows(dataset.blocks[0], arguments.maximum_rows)
    jax_calibrated = _select_even_rows(
        dataset.blocks[1], arguments.maximum_rows
    )
    if casa.shape != jax_calibrated.shape:
        raise ValueError("fixture blocks must have matching shapes")
    direct_config = DirectDFTConfig(
        visibility_chunk_size=arguments.visibility_tile_size,
        pixel_chunk_size=arguments.pixel_tile_size,
        precision=arguments.precision,
    )
    configuration = ImagingConfig(
        size=arguments.size,
        pixel_size_rad=np.deg2rad(arguments.pixel_arcsec / 3600),
        pixel_basis=pixel_basis_from_name(
            arguments.pixel_model,
            gaussian_sigma_pixels=arguments.gaussian_sigma_pixels,
        ),
        inference=InferenceConfig(
            steps=arguments.steps,
            learning_rate=arguments.learning_rate,
            sparsity_weight=arguments.sparsity_weight,
            smoothness_weight=arguments.smoothness_weight,
            chunk_size=arguments.visibility_tile_size,
            initial_intensity=1e-3,
            patience=arguments.steps + 1,
            validation_interval=arguments.validation_interval,
            operator_mode="explicit",
            direct_dft=direct_config,
        ),
        holdout_fraction=arguments.holdout_fraction,
        split_seed=17,
        split_strategy=arguments.split_strategy,
    )
    casa_result = reconstruct(casa, configuration)
    jax_result = reconstruct(jax_calibrated, configuration)
    arguments.output.mkdir(parents=True, exist_ok=True)
    write_products(
        casa_result, arguments.output / "casa_corrected_reconstruction.fits"
    )
    write_products(
        jax_result, arguments.output / "jax_calibrated_reconstruction.fits"
    )
    _plot(
        casa_result.image,
        jax_result.image,
        arguments.output / "reconstruction_comparison.png",
    )
    scalar_visibility_count = casa.shape[0] * casa.shape[1]
    summary = {
        "fixture": arguments.fixture.name,
        "fixture_provenance": dict(dataset.provenance),
        "block_shape": list(casa.shape),
        "scalar_visibility_count": scalar_visibility_count,
        "pixel_count": arguments.size**2,
        "visibility_pixel_products_per_pass": (
            scalar_visibility_count * arguments.size**2
        ),
        "configuration": {
            "size": arguments.size,
            "pixel_arcsec": arguments.pixel_arcsec,
            "pixel_model": arguments.pixel_model,
            "steps": arguments.steps,
            "sparsity_weight": arguments.sparsity_weight,
            "smoothness_weight": arguments.smoothness_weight,
            "holdout_fraction": arguments.holdout_fraction,
            "split_strategy": arguments.split_strategy,
            "validation_interval": arguments.validation_interval,
            "precision": arguments.precision,
            "visibility_tile_size": arguments.visibility_tile_size,
            "pixel_tile_size": arguments.pixel_tile_size,
        },
        "casa_corrected": _result_metrics(casa_result),
        "jax_calibrated": _result_metrics(jax_result),
        "image_comparison": _image_metrics(
            jax_result.image, casa_result.image
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
