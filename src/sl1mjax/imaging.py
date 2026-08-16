"""End-to-end gradient imaging for one compatible visibility block."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any

import numpy as np

from sl1mjax.data.canonical import VisibilityBlock
from sl1mjax.inference import InferenceConfig, InferenceResult, infer_regular_grid
from sl1mjax.objective import weighted_complex_mse
from sl1mjax.rime import predict_stokes_i
from sl1mjax.sky import RegularGrid
from sl1mjax.split import uv_cell_split


@dataclass(frozen=True)
class ImagingConfig:
    size: int = 16
    pixel_size_rad: float = np.deg2rad(5 / 3600)
    inference: InferenceConfig = InferenceConfig()
    holdout_fraction: float = 0.2
    split_seed: int = 0


@dataclass(frozen=True)
class ImagingResult:
    image: np.ndarray
    prediction: np.ndarray
    residual: np.ndarray
    inference: InferenceResult
    train_loss: float
    holdout_loss: float
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
                "steps": self.inference.steps,
                "converged": self.inference.converged,
                "elapsed_s": self.elapsed_s,
                "peak_flux": float(np.max(self.image)),
                "total_flux": float(np.sum(self.image)),
            },
            "history": {
                "objective": list(self.inference.objective_history),
                "data": list(self.inference.data_history),
                "prior": list(self.inference.prior_history),
            },
            "split": {"strategy": "uv_cell", "seed": self.configuration.split_seed},
        }


def reconstruct(
    block: VisibilityBlock, configuration: ImagingConfig | None = None
) -> ImagingResult:
    config = configuration or ImagingConfig()
    grid = RegularGrid(config.size, config.pixel_size_rad)
    split = uv_cell_split(
        block,
        holdout_fraction=config.holdout_fraction,
        seed=config.split_seed,
    )
    started = perf_counter()
    inference = infer_regular_grid(block, grid, split.train, config.inference)
    elapsed = perf_counter() - started
    l, m = grid.coordinates
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
        )
    )
    residual = prediction - block.visibility
    train_loss = float(
        weighted_complex_mse(prediction, block.visibility, block.weight, ~split.train)
    )
    holdout_loss = float(
        weighted_complex_mse(prediction, block.visibility, block.weight, ~split.holdout)
    )
    return ImagingResult(
        image=inference.image,
        prediction=prediction,
        residual=residual,
        inference=inference,
        train_loss=train_loss,
        holdout_loss=holdout_loss,
        elapsed_s=elapsed,
        grid=grid,
        configuration=config,
        provenance=dict(block.provenance),
        correlations=tuple(value.value for value in block.correlations),
        phase_centre_rad=block.phase_centre_rad,
    )
