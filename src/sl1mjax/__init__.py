"""Differentiable radio-interferometric sky–instrument modelling."""

from jax import config as _jax_config

_jax_config.update("jax_enable_x64", True)

from sl1mjax.beam import gaussian_primary_beam  # noqa: E402
from sl1mjax.coordinates import lmn_to_radec, radec_to_lmn  # noqa: E402
from sl1mjax.data import (  # noqa: E402
    VisibilityBlock,
    VisibilityDataset,
    read_dataset,
    write_dataset,
)
from sl1mjax.diagnostics import dirty_image_and_psf  # noqa: E402
from sl1mjax.imaging import ImagingConfig, ImagingResult, reconstruct  # noqa: E402
from sl1mjax.inference import (  # noqa: E402
    InferenceConfig,
    InferenceResult,
    infer_regular_grid,
)
from sl1mjax.polarization import Correlation, ReceptorBasis  # noqa: E402
from sl1mjax.rime import predict_stokes_i  # noqa: E402
from sl1mjax.sky import (  # noqa: E402
    COMPOUND_N4_BASIS,
    CompoundPixelBasis,
    DeltaPixelBasis,
    GaussianApproximation,
    GaussianPixelBasis,
    RegularGrid,
    pixel_basis_from_name,
)

__all__ = [
    "Correlation",
    "COMPOUND_N4_BASIS",
    "CompoundPixelBasis",
    "DeltaPixelBasis",
    "GaussianApproximation",
    "GaussianPixelBasis",
    "ImagingConfig",
    "ImagingResult",
    "InferenceConfig",
    "InferenceResult",
    "ReceptorBasis",
    "RegularGrid",
    "VisibilityBlock",
    "VisibilityDataset",
    "dirty_image_and_psf",
    "gaussian_primary_beam",
    "infer_regular_grid",
    "lmn_to_radec",
    "pixel_basis_from_name",
    "predict_stokes_i",
    "radec_to_lmn",
    "read_dataset",
    "reconstruct",
    "write_dataset",
]

__version__ = "0.1.0"
