from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from sl1mjax.data.canonical import VisibilityBlock
from sl1mjax.polarization import Correlation, ReceptorBasis
from sl1mjax.resolution import estimate_synthesized_beam, resolution_limited_max_depth


def _block() -> VisibilityBlock:
    uvw_m = np.asarray(
        [
            [-1200.0, -600.0, 0.0],
            [-900.0, 800.0, 0.0],
            [700.0, -1000.0, 0.0],
            [1300.0, 500.0, 0.0],
        ]
    )
    shape = (4, 1, 1)
    return VisibilityBlock(
        uvw_m=uvw_m,
        frequency_hz=np.asarray([1.0e9]),
        visibility=np.zeros(shape, dtype=np.complex128),
        weight=np.ones(shape),
        flag=np.zeros(shape, dtype=bool),
        time_s=np.arange(4, dtype=np.float64),
        antenna1=np.zeros(4, dtype=np.int32),
        antenna2=np.ones(4, dtype=np.int32),
        correlations=(Correlation.I,),
        receptor_basis=ReceptorBasis.STOKES,
    )


def test_weighted_uv_beam_matches_psf_curvature_formula() -> None:
    block = _block()
    beam = estimate_synthesized_beam((block,))
    uv = block.uvw_m[:, :2] * 1.0e9 / 299_792_458.0
    eigenvalues = np.linalg.eigvalsh(uv.T @ uv / uv.shape[0])
    factor = np.sqrt(2.0 * np.log(2.0)) / np.pi

    assert beam.major_fwhm_rad == pytest.approx(factor / np.sqrt(eigenvalues[0]))
    assert beam.minor_fwhm_rad == pytest.approx(factor / np.sqrt(eigenvalues[1]))


def test_beam_estimate_uses_only_selected_samples() -> None:
    block = _block()
    mask = np.ones(block.shape, dtype=bool)
    mask[-1] = False

    selected = estimate_synthesized_beam((block,), (mask,))
    block_with_flag = replace(block, flag=~mask)
    flagged = estimate_synthesized_beam((block_with_flag,))

    assert selected == flagged


@pytest.mark.parametrize(
    ("root_arcsec", "beam_arcsec", "maximum_pixels", "expected_depth"),
    [
        (16.0, 17.0, 5.0, 2),
        (16.0, 17.0, 4.0, 1),
        (4.0, 17.0, 5.0, 0),
        (64.0, 16.0, 5.0, 4),
    ],
)
def test_resolution_depth_limit(
    root_arcsec: float,
    beam_arcsec: float,
    maximum_pixels: float,
    expected_depth: int,
) -> None:
    assert (
        resolution_limited_max_depth(
            np.deg2rad(root_arcsec / 3600.0),
            np.deg2rad(beam_arcsec / 3600.0),
            maximum_pixels_per_beam=maximum_pixels,
        )
        == expected_depth
    )
