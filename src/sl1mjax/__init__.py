"""Differentiable radio-interferometric sky–instrument modelling."""

from jax import config as _jax_config

_jax_config.update("jax_enable_x64", True)

from sl1mjax.beam import (  # noqa: E402
    VLABeamCatalog,
    VLAPrimaryBeam,
    gaussian_primary_beam,
    primary_beam_from_name,
)
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
from sl1mjax.quadtree import (  # noqa: E402
    QuadtreeGrid,
    QuadtreeLeaf,
    QuadtreeSky,
    QuadtreeTopology,
    leaves_exceeding_error_bound,
    predict_quadtree_stokes_i,
    quadtree_sky_from_regular_grid,
    wide_field_error_bounds,
)
from sl1mjax.rime import (  # noqa: E402
    predict_stokes_i,
    square_wide_field_error_bound,
)
from sl1mjax.sky import (  # noqa: E402
    COMPOUND_N4_BASIS,
    CompoundPixelBasis,
    DeltaPixelBasis,
    GaussianApproximation,
    GaussianPixelBasis,
    RegularGrid,
    SquarePixelBasis,
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
    "QuadtreeGrid",
    "QuadtreeLeaf",
    "QuadtreeSky",
    "QuadtreeTopology",
    "ReceptorBasis",
    "RequantizerTerm",
    "RegularGrid",
    "VLABeamCatalog",
    "VLAPrimaryBeam",
    "ResidualEvaluation",
    "OpacityTerm",
    "SquarePixelBasis",
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
    "leaves_exceeding_error_bound",
    "lmn_to_radec",
    "pixel_basis_from_name",
    "primary_beam_from_name",
    "predict_quadtree_stokes_i",
    "predict_stokes_i",
    "predict_stokes_i_explicit",
    "quadtree_sky_from_regular_grid",
    "radec_to_lmn",
    "read_dataset",
    "reconstruct",
    "solve_staged_calibration",
    "square_wide_field_error_bound",
    "wide_field_error_bounds",
    "write_dataset",
]

__version__ = "0.1.0"
