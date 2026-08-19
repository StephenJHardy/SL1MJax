"""End-to-end gradient imaging for one compatible visibility block."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any, Literal

import numpy as np

from sl1mjax.data.canonical import VisibilityBlock
from sl1mjax.direct_operator import predict_stokes_i_explicit
from sl1mjax.inference import InferenceConfig, InferenceResult, infer_regular_grid
from sl1mjax.objective import (
    normalized_weighted_complex_mse,
    weighted_complex_mse,
)
from sl1mjax.rime import predict_stokes_i
from sl1mjax.sky import DeltaPixelBasis, PixelBasis, RegularGrid
from sl1mjax.split import random_row_split, uv_cell_split


@dataclass(frozen=True)
class ImagingConfig:
    size: int = 16
    pixel_size_rad: float = np.deg2rad(5 / 3600)
    pixel_basis: PixelBasis = DeltaPixelBasis()
    inference: InferenceConfig = InferenceConfig()
    holdout_fraction: float = 0.2
    split_seed: int = 0
    split_strategy: Literal["uv_cell", "random_row"] = "uv_cell"


@dataclass(frozen=True)
class ImagingResult:
    image: np.ndarray
    prediction: np.ndarray
    residual: np.ndarray
    inference: InferenceResult
    train_loss: float
    holdout_loss: float
    train_normalized_loss: float
    holdout_normalized_loss: float
    elapsed_s: float
    grid: RegularGrid
    configuration: ImagingConfig
    provenance: dict[str, Any]
    correlations: tuple[str, ...]
    phase_centre_rad: tuple[float, float]

    def diagnostics(self) -> dict[str, Any]:
        configuration = asdict(self.configuration)
        return {
            "configuration": configuration,
            "provenance": self.provenance,
            "correlations": list(self.correlations),
            "metrics": {
                "train_weighted_complex_mse": self.train_loss,
                "holdout_weighted_complex_mse": self.holdout_loss,
                "train_normalized_weighted_complex_mse": (
                    self.train_normalized_loss
                ),
                "holdout_normalized_weighted_complex_mse": (
                    self.holdout_normalized_loss
                ),
                "steps": self.inference.steps,
                "best_step": self.inference.best_step,
                "converged": self.inference.converged,
                "elapsed_s": self.elapsed_s,
                "peak_flux": float(np.max(self.image)),
                "total_flux": float(np.sum(self.image)),
            },
            "history": {
                "objective": list(self.inference.objective_history),
                "data": list(self.inference.data_history),
                "prior": list(self.inference.prior_history),
                "holdout": list(self.inference.holdout_history),
                "holdout_steps": list(self.inference.holdout_steps),
            },
            "split": {
                "strategy": self.configuration.split_strategy,
                "seed": self.configuration.split_seed,
            },
        }


def reconstruct(
    block: VisibilityBlock, configuration: ImagingConfig | None = None
) -> ImagingResult:
    config = configuration or ImagingConfig()
    grid = RegularGrid(config.size, config.pixel_size_rad)
    if config.split_strategy == "uv_cell":
        split = uv_cell_split(
            block,
            holdout_fraction=config.holdout_fraction,
            seed=config.split_seed,
        )
    elif config.split_strategy == "random_row":
        split = random_row_split(
            block,
            holdout_fraction=config.holdout_fraction,
            seed=config.split_seed,
        )
    else:
        raise ValueError("split_strategy must be uv_cell or random_row")
    started = perf_counter()
    inference = infer_regular_grid(
        block,
        grid,
        split.train,
        config.inference,
        holdout_mask=split.holdout,
        pixel_basis=config.pixel_basis,
    )
    elapsed = perf_counter() - started
    l, m = grid.coordinates
    if config.inference.operator_mode == "explicit":
        prediction = np.asarray(
            predict_stokes_i_explicit(
                inference.image.ravel(),
                l,
                m,
                block.uvw_m,
                block.frequency_hz,
                block.antenna1,
                block.antenna2,
                block.correlations,
                pixel_basis=config.pixel_basis,
                pixel_size_rad=grid.pixel_size_rad,
                config=config.inference.direct_dft,
            )
        )
    else:
        prediction = np.asarray(
            predict_stokes_i(
                inference.image.ravel(),
                l,
                m,
                block.uvw_m,
                block.frequency_hz,
                block.antenna1,
                block.antenna2,
                block.correlations,
                chunk_size=config.inference.chunk_size,
                pixel_basis=config.pixel_basis,
                pixel_size_rad=grid.pixel_size_rad,
            )
        )
    residual = prediction - block.visibility
    train_loss = float(
        weighted_complex_mse(prediction, block.visibility, block.weight, ~split.train)
    )
    holdout_loss = float(
        weighted_complex_mse(prediction, block.visibility, block.weight, ~split.holdout)
    )
    train_normalized_loss = float(
        normalized_weighted_complex_mse(
            prediction, block.visibility, block.weight, ~split.train
        )
    )
    holdout_normalized_loss = float(
        normalized_weighted_complex_mse(
            prediction, block.visibility, block.weight, ~split.holdout
        )
    )
    return ImagingResult(
        image=inference.image,
        prediction=prediction,
        residual=residual,
        inference=inference,
        train_loss=train_loss,
        holdout_loss=holdout_loss,
        train_normalized_loss=train_normalized_loss,
        holdout_normalized_loss=holdout_normalized_loss,
        elapsed_s=elapsed,
        grid=grid,
        configuration=config,
        provenance=dict(block.provenance),
        correlations=tuple(value.value for value in block.correlations),
        phase_centre_rad=block.phase_centre_rad,
    )
