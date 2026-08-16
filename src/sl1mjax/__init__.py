"""Differentiable radio-interferometric sky–instrument modelling."""

from jax import config as _jax_config

_jax_config.update("jax_enable_x64", True)

from sl1mjax.data import (  # noqa: E402
    VisibilityBlock,
    VisibilityDataset,
    read_dataset,
    write_dataset,
)
from sl1mjax.imaging import ImagingConfig, ImagingResult, reconstruct  # noqa: E402
from sl1mjax.inference import (  # noqa: E402
    InferenceConfig,
    InferenceResult,
    infer_regular_grid,
)
from sl1mjax.polarization import Correlation, ReceptorBasis  # noqa: E402
from sl1mjax.rime import predict_stokes_i  # noqa: E402
from sl1mjax.sky import RegularGrid  # noqa: E402

__all__ = [
    "Correlation",
    "ImagingConfig",
    "ImagingResult",
    "InferenceConfig",
    "InferenceResult",
    "ReceptorBasis",
    "RegularGrid",
    "VisibilityBlock",
    "VisibilityDataset",
    "infer_regular_grid",
    "predict_stokes_i",
    "read_dataset",
    "reconstruct",
    "write_dataset",
]

__version__ = "0.1.0"
