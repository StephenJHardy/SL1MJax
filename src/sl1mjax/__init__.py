"""Differentiable radio-interferometric sky–instrument modelling."""

from jax import config as _jax_config

_jax_config.update("jax_enable_x64", True)

from sl1mjax.beam import gaussian_primary_beam  # noqa: E402
from sl1mjax.calibration import (  # noqa: E402
    CalibrationSolution,
    apply_calibration,
    identity_solution,
)
from sl1mjax.calibration_diagnostics import diagnose_calibration  # noqa: E402
from sl1mjax.calibration_inference import (  # noqa: E402
    CalibrationFitResult,
    CalibrationSolveConfig,
    solve_staged_calibration,
)
from sl1mjax.calibration_terms import (  # noqa: E402
    CalibrationChain,
    CalibrationCoordinates,
    GainCurveTerm,
    OpacityTerm,
    RequantizerTerm,
)
from sl1mjax.coordinates import lmn_to_radec, radec_to_lmn  # noqa: E402
from sl1mjax.data import (  # noqa: E402
    VisibilityBlock,
    VisibilityDataset,
    read_dataset,
    write_dataset,
)
from sl1mjax.diagnostics import (  # noqa: E402
    ResidualEvaluation,
    dirty_image_and_psf,
    evaluate_residuals,
)
from sl1mjax.direct_operator import (  # noqa: E402
    DirectDFTConfig,
    direct_scalar_adjoint,
    direct_scalar_visibility,
    predict_stokes_i_explicit,
)
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
    "CalibrationFitResult",
    "CalibrationChain",
    "CalibrationCoordinates",
    "CalibrationSolution",
    "CalibrationSolveConfig",
    "COMPOUND_N4_BASIS",
    "CompoundPixelBasis",
    "DeltaPixelBasis",
    "DirectDFTConfig",
    "GaussianApproximation",
    "GaussianPixelBasis",
    "GainCurveTerm",
    "ImagingConfig",
    "ImagingResult",
    "InferenceConfig",
    "InferenceResult",
    "ReceptorBasis",
    "RequantizerTerm",
    "RegularGrid",
    "ResidualEvaluation",
    "OpacityTerm",
    "VisibilityBlock",
    "VisibilityDataset",
    "apply_calibration",
    "diagnose_calibration",
    "dirty_image_and_psf",
    "direct_scalar_adjoint",
    "direct_scalar_visibility",
    "evaluate_residuals",
    "gaussian_primary_beam",
    "infer_regular_grid",
    "identity_solution",
    "lmn_to_radec",
    "pixel_basis_from_name",
    "predict_stokes_i",
    "predict_stokes_i_explicit",
    "radec_to_lmn",
    "read_dataset",
    "reconstruct",
    "solve_staged_calibration",
    "write_dataset",
]

__version__ = "0.1.0"
