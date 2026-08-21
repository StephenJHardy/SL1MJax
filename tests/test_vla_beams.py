from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from sl1mjax.beam import (
    VLABeamCatalog,
    VLAPrimaryBeam,
    _j1,
    gaussian_primary_beam,
    primary_beam_from_name,
)
from sl1mjax.data.synthetic import PointSource, simulate_dataset
from sl1mjax.direct_operator import DirectDFTConfig, predict_stokes_i_explicit
from sl1mjax.imaging import ImagingConfig, reconstruct
from sl1mjax.inference import InferenceConfig
from sl1mjax.polarization import Correlation, ReceptorBasis
from sl1mjax.rime import SPEED_OF_LIGHT_M_S, predict_stokes_i
from sl1mjax.sky import RegularGrid


def test_j1_has_the_standard_first_zero() -> None:
    assert _j1(0.0) == pytest.approx(0.0, abs=1e-15)
    assert _j1(3.83170597) == pytest.approx(0.0, abs=1e-6)


def test_gaussian_primary_beam_wrapper_still_matches_half_power() -> None:
    dish = 25.0
    frequency = 1.0e9
    fwhm = 1.02 * SPEED_OF_LIGHT_M_S / (frequency * dish)
    radius = np.sin(fwhm / 2)
    attenuation = gaussian_primary_beam(
        [0.0, radius],
        [0.0, 0.0],
        frequency,
        dish_diameter_m=dish,
    )
    np.testing.assert_allclose(attenuation, [1.0, 0.5], rtol=1e-14)


def test_airy_beam_is_unity_on_axis_and_tapers_off_axis() -> None:
    beam = VLAPrimaryBeam(kind="airy")
    frequency = 4.6e9
    on_axis = beam.power(0.0, 0.0, frequency)
    mid = beam.power(np.sin(np.deg2rad(0.04)), 0.0, frequency)
    far = beam.power(np.sin(np.deg2rad(0.12)), 0.0, frequency)
    assert on_axis == pytest.approx(1.0, rel=1e-12)
    assert 0.0 < float(far) < float(mid) < 1.0


def test_airy_beam_is_zero_beyond_the_catalog_maximum_radius() -> None:
    catalog = VLABeamCatalog()
    beam = VLAPrimaryBeam(kind="airy", catalog=catalog)
    frequency = 1.0e9
    just_inside = np.sin(catalog.airy_max_radius_rad(frequency) * 0.99)
    just_outside = np.sin(catalog.airy_max_radius_rad(frequency) * 1.01)
    assert beam.power(just_inside, 0.0, frequency) > 0
    assert beam.power(just_outside, 0.0, frequency) == 0.0


def test_airy_frequency_scaling_shrinks_the_beam() -> None:
    beam = VLAPrimaryBeam(kind="airy")
    offset = np.sin(np.deg2rad(0.05))
    low = float(beam.power(offset, 0.0, 1.5e9))
    high = float(beam.power(offset, 0.0, 6.0e9))
    assert high < low


def test_squint_offsets_rr_and_ll_in_opposite_directions() -> None:
    frequency = 1.0e9
    catalog = VLABeamCatalog()
    beam = VLAPrimaryBeam(kind="gaussian", apply_squint=True, catalog=catalog)
    offset = float(catalog.squint_offset_rad(frequency))
    l_peak = np.sin(offset)
    rr = beam.power([-l_peak, 0.0, l_peak], [0.0, 0.0, 0.0], frequency, receptor="RR")
    ll = beam.power([-l_peak, 0.0, l_peak], [0.0, 0.0, 0.0], frequency, receptor="LL")
    assert rr[2] > rr[0]
    assert ll[0] > ll[2]
    assert rr[2] == pytest.approx(1.0, rel=1e-6)
    assert ll[0] == pytest.approx(1.0, rel=1e-6)
    stokes_i = beam.power(0.0, 0.0, frequency, receptor="I")
    assert float(stokes_i) < 1.0
    assert float(stokes_i) == pytest.approx(0.5 * (rr[1] + ll[1]), rel=1e-12)


def test_pointing_offset_recenters_the_beam() -> None:
    beam = VLAPrimaryBeam(kind="gaussian", pointing_lm=(0.01, -0.02))
    assert beam.power(0.01, -0.02, 3e9) == pytest.approx(1.0, rel=1e-12)
    assert beam.power(0.0, 0.0, 3e9) < 1.0


def test_power_weights_have_pixel_channel_shape() -> None:
    grid = RegularGrid(5, np.deg2rad(4 / 3600))
    l, m = grid.coordinates
    frequencies = np.asarray([4.5e9, 4.6e9, 4.7e9])
    weights = VLAPrimaryBeam(kind="airy").power_weights(l, m, frequencies)
    assert weights.shape == (25, 3)
    np.testing.assert_allclose(weights[12], 1.0, rtol=1e-12)


def test_predict_applies_the_power_beam_to_a_point_source() -> None:
    grid = RegularGrid(5, 2e-3)
    l, m = grid.coordinates
    intensity = np.zeros(l.size)
    off_axis = 6
    intensity[off_axis] = 2.0
    beam = VLAPrimaryBeam(kind="gaussian")
    weights = beam.power_weights(l, m, [1.2e9])
    uvw_m = np.asarray([[12.0, -8.0, 3.0]])
    frequency = np.asarray([1.2e9])
    antenna1 = np.asarray([0], dtype=np.int32)
    antenna2 = np.asarray([1], dtype=np.int32)
    correlations = (Correlation.RR, Correlation.LL)
    expected_scale = float(weights[off_axis, 0])
    bare = np.asarray(
        predict_stokes_i(
            intensity, l, m, uvw_m, frequency, antenna1, antenna2, correlations
        )
    )
    attenuated = np.asarray(
        predict_stokes_i(
            intensity,
            l,
            m,
            uvw_m,
            frequency,
            antenna1,
            antenna2,
            correlations,
            beam_weights=weights,
        )
    )
    explicit = np.asarray(
        predict_stokes_i_explicit(
            intensity,
            l,
            m,
            uvw_m,
            frequency,
            antenna1,
            antenna2,
            correlations,
            beam_weights=weights,
            config=DirectDFTConfig(visibility_chunk_size=8, pixel_chunk_size=8),
        )
    )
    np.testing.assert_allclose(attenuated, expected_scale * bare, rtol=1e-10)
    np.testing.assert_allclose(explicit, attenuated, rtol=1e-9)


def test_squinted_predict_makes_rr_and_ll_differ() -> None:
    grid = RegularGrid(7, 3e-3)
    l, m = grid.coordinates
    intensity = np.zeros(l.size)
    intensity[np.argmax(l)] = 1.5
    beam = VLAPrimaryBeam(kind="gaussian", apply_squint=True)
    _, rr, ll = (
        None,
        beam.power_weights(l, m, [1.0e9], receptor="RR"),
        beam.power_weights(l, m, [1.0e9], receptor="LL"),
    )
    uvw_m = np.asarray([[20.0, 5.0, -2.0], [-11.0, 17.0, 4.0]])
    frequency = np.asarray([1.0e9])
    prediction = np.asarray(
        predict_stokes_i(
            intensity,
            l,
            m,
            uvw_m,
            frequency,
            np.asarray([0, 0], dtype=np.int32),
            np.asarray([1, 2], dtype=np.int32),
            (Correlation.RR, Correlation.LL),
            beam_weights_rr=rr,
            beam_weights_ll=ll,
        )
    )
    assert prediction.shape == (2, 1, 2)
    assert not np.allclose(prediction[..., 0], prediction[..., 1])


def test_reconstruct_recovers_a_source_observed_through_the_same_beam() -> None:
    grid = RegularGrid(6, np.deg2rad(6 / 3600))
    l, m = grid.coordinates
    centre = (grid.size // 2) * grid.size + (grid.size // 2)
    geometry = simulate_dataset(
        grid,
        basis=ReceptorBasis.CIRCULAR,
        sources=(PointSource(1.0, float(l[centre]), float(m[centre])),),
        rows=80,
        channels=2,
        seed=21,
    ).blocks[0]
    beam = VLAPrimaryBeam(kind="gaussian")
    intensity = np.zeros(l.size)
    intensity[centre] = 1.0
    observed = np.asarray(
        predict_stokes_i_explicit(
            intensity,
            l,
            m,
            geometry.uvw_m,
            geometry.frequency_hz,
            geometry.antenna1,
            geometry.antenna2,
            geometry.correlations,
            beam_weights=beam.power_weights(l, m, geometry.frequency_hz),
            config=DirectDFTConfig(
                visibility_chunk_size=32,
                pixel_chunk_size=32,
                precision="float32",
            ),
        )
    )
    block = replace(geometry, visibility=observed)
    result = reconstruct(
        block,
        ImagingConfig(
            size=grid.size,
            pixel_size_rad=grid.pixel_size_rad,
            holdout_fraction=0.0,
            primary_beam=beam,
            inference=InferenceConfig(
                steps=160,
                learning_rate=0.12,
                sparsity_weight=0.0,
                patience=100,
                validation_interval=10,
                operator_mode="explicit",
                direct_dft=DirectDFTConfig(
                    visibility_chunk_size=32,
                    pixel_chunk_size=32,
                    precision="float32",
                ),
            ),
        ),
    )
    assert result.train_loss < 5e-3
    assert result.image.ravel()[centre] == pytest.approx(1.0, rel=0.2)
    assert result.diagnostics()["configuration"]["primary_beam"]["kind"] == "gaussian"


def test_primary_beam_from_name_rejects_unknown_models() -> None:
    assert primary_beam_from_name("none") is None
    assert primary_beam_from_name("airy").kind == "airy"
    with pytest.raises(ValueError):
        primary_beam_from_name("cosine")


def test_catalog_rejects_nonphysical_geometry() -> None:
    with pytest.raises(ValueError):
        VLABeamCatalog(blockage_diameter_m=30.0)
    with pytest.raises(ValueError):
        VLAPrimaryBeam(kind="cosine")  # type: ignore[arg-type]
