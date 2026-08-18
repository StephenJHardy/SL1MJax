import numpy as np
import pytest

from sl1mjax.beam import gaussian_primary_beam
from sl1mjax.data.synthetic import PointSource, simulate_dataset
from sl1mjax.objective import weighted_complex_mse
from sl1mjax.polarization import ReceptorBasis
from sl1mjax.rime import SPEED_OF_LIGHT_M_S
from sl1mjax.sky import RegularGrid


def test_complex_noise_and_inverse_variance_weights_match_requested_statistics() -> None:
    noise_std = 0.2
    block = simulate_dataset(
        RegularGrid(4, 1e-5),
        basis=ReceptorBasis.LINEAR,
        sources=(PointSource(0.0, 0.0, 0.0),),
        rows=6_000,
        noise_std=noise_std,
        seed=91,
    ).blocks[0]
    assert np.std(block.visibility.real) == pytest.approx(
        noise_std / np.sqrt(2), rel=0.025
    )
    assert np.std(block.visibility.imag) == pytest.approx(
        noise_std / np.sqrt(2), rel=0.025
    )
    np.testing.assert_array_equal(block.weight, 1 / noise_std**2)


def test_flags_remove_arbitrary_data_and_zero_weights_remove_samples() -> None:
    observed = np.array([[[1 + 2j], [3 + 4j], [5 + 6j]]])
    predicted = observed.copy()
    predicted[0, 0, 0] += 1e12
    predicted[0, 1, 0] -= 1e12j
    flag = np.array([[[True], [False], [False]]])
    weight = np.array([[[1.0], [0.0], [2.0]]])
    assert float(weighted_complex_mse(predicted, observed, weight, flag)) == 0.0


def test_gaussian_primary_beam_is_separate_radial_and_frequency_scaled() -> None:
    dish_diameter = 25.0
    low_frequency = 1.0e9
    low_fwhm = 1.02 * SPEED_OF_LIGHT_M_S / (low_frequency * dish_diameter)
    half_power_radius = np.sin(low_fwhm / 2)
    attenuation = gaussian_primary_beam(
        [0.0, half_power_radius, half_power_radius],
        [0.0, 0.0, 0.0],
        [low_frequency, low_frequency, 2 * low_frequency],
        dish_diameter_m=dish_diameter,
    )
    np.testing.assert_allclose(attenuation[:2], [1.0, 0.5], rtol=1e-14)
    assert attenuation[2] == pytest.approx(0.5**4, rel=1e-13)


def test_primary_beam_rejects_nonphysical_inputs() -> None:
    with pytest.raises(ValueError):
        gaussian_primary_beam(0, 0, 1e9, dish_diameter_m=0)
    with pytest.raises(ValueError):
        gaussian_primary_beam(0, 0, -1, dish_diameter_m=25)
    with pytest.raises(ValueError):
        gaussian_primary_beam(1, 0, 1e9, dish_diameter_m=25)
