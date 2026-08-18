from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from sl1mjax.coordinates import radec_to_lmn
from sl1mjax.imaging import ImagingConfig, ImagingResult
from sl1mjax.inference import InferenceConfig, InferenceResult
from sl1mjax.output import write_products
from sl1mjax.polarization import Correlation
from sl1mjax.rime import SPEED_OF_LIGHT_M_S, predict_stokes_i
from sl1mjax.sky import RegularGrid


def _predict(uvw_m: np.ndarray, frequencies: np.ndarray, l: float, m: float) -> np.ndarray:
    return np.asarray(
        predict_stokes_i(
            np.asarray([1.0]),
            np.asarray([l]),
            np.asarray([m]),
            uvw_m,
            frequencies,
            np.asarray([0]),
            np.asarray([1]),
            (Correlation.I,),
        )
    )[0, :, 0]


def test_phase_increment_scales_exactly_with_channel_frequency() -> None:
    frequencies = np.array([0.8e9, 1.6e9])
    l = 2e-4
    visibility = _predict(np.array([[80.0, 0.0, 0.0]]), frequencies, l, 0.0)
    expected_increment = (
        2 * np.pi * 80.0 * l * np.diff(frequencies)[0] / SPEED_OF_LIGHT_M_S
    )
    np.testing.assert_allclose(
        np.angle(visibility[1] / visibility[0]),
        expected_increment,
        rtol=1e-13,
        atol=1e-13,
    )


def test_w_term_is_isolated_and_matches_wide_field_identity() -> None:
    l, m, w = 0.3, -0.2, 17.0
    n = np.sqrt(1 - l * l - m * m)
    with_w = _predict(
        np.array([[0.0, 0.0, w]]),
        np.array([SPEED_OF_LIGHT_M_S]),
        l,
        m,
    )[0]
    coplanar = _predict(
        np.zeros((1, 3)),
        np.array([SPEED_OF_LIGHT_M_S]),
        l,
        m,
    )[0]
    np.testing.assert_allclose(
        with_w / coplanar,
        np.exp(2j * np.pi * w * (n - 1)),
        rtol=1e-14,
        atol=1e-14,
    )
    assert abs(with_w - coplanar) > 0.1


def test_fits_wcs_and_grid_share_east_north_handedness(tmp_path: Path) -> None:
    grid = RegularGrid(3, np.deg2rad(2 / 3600))
    inference = InferenceResult(
        image=np.zeros((3, 3)),
        raw_parameters=np.zeros(9),
        optimizer_state=None,
        objective_history=(1.0,),
        data_history=(1.0,),
        prior_history=(0.0,),
        steps=1,
        converged=False,
    )
    phase_centre = (np.deg2rad(180.0), np.deg2rad(45.0))
    result = ImagingResult(
        image=inference.image,
        prediction=np.zeros((1, 1, 1), dtype=np.complex128),
        residual=np.zeros((1, 1, 1), dtype=np.complex128),
        inference=inference,
        train_loss=0.0,
        holdout_loss=0.0,
        elapsed_s=0.0,
        grid=grid,
        configuration=ImagingConfig(
            size=3,
            pixel_size_rad=grid.pixel_size_rad,
            inference=InferenceConfig(steps=1),
        ),
        provenance={},
        correlations=("I",),
        phase_centre_rad=phase_centre,
    )
    image_path, *_ = write_products(result, tmp_path / "image.fits")
    with fits.open(image_path) as hdus:
        wcs = WCS(hdus[0].header)
        west_ra, west_dec = np.deg2rad(wcs.pixel_to_world_values(2, 1))
        east_ra, east_dec = np.deg2rad(wcs.pixel_to_world_values(0, 1))
        north_ra, north_dec = np.deg2rad(wcs.pixel_to_world_values(1, 2))

    assert radec_to_lmn(*phase_centre, east_ra, east_dec)[0] > 0
    assert radec_to_lmn(*phase_centre, west_ra, west_dec)[0] < 0
    assert radec_to_lmn(*phase_centre, north_ra, north_dec)[1] > 0
    l, m = (value.reshape(3, 3) for value in grid.coordinates)
    assert l[1, 0] > 0 and l[1, 2] < 0 and m[2, 1] > 0
