import numpy as np

from sl1mjax.coordinates import lmn_to_radec, radec_to_lmn


def test_phase_centre_maps_to_zero_direction_cosines() -> None:
    ra0, dec0 = np.deg2rad([359.9, 45.0])
    l, m, n = radec_to_lmn(ra0, dec0, [ra0], [dec0])
    np.testing.assert_allclose(l, 0.0, atol=1e-15)
    np.testing.assert_allclose(m, 0.0, atol=1e-15)
    np.testing.assert_allclose(n, 1.0, atol=1e-15)


def test_exact_transform_round_trips_ra_wrap_and_high_declination() -> None:
    ra0, dec0 = np.deg2rad([359.9, 75.0])
    ra = np.deg2rad([0.1, 359.7, 0.0])
    dec = np.deg2rad([75.2, 74.8, 75.05])
    l, m, n = radec_to_lmn(ra0, dec0, ra, dec)
    restored_ra, restored_dec = lmn_to_radec(ra0, dec0, l, m, n)
    angular_ra_error = np.angle(np.exp(1j * (restored_ra - ra)))
    np.testing.assert_allclose(angular_ra_error, 0.0, atol=1e-14)
    np.testing.assert_allclose(restored_dec, dec, atol=1e-14)


def test_east_and_north_offsets_have_positive_l_and_m() -> None:
    ra0, dec0 = np.deg2rad([180.0, 45.0])
    offset = np.deg2rad(5 / 3600)
    east = radec_to_lmn(ra0, dec0, [ra0 + offset / np.cos(dec0)], [dec0])
    north = radec_to_lmn(ra0, dec0, [ra0], [dec0 + offset])
    assert east[0][0] > 0
    assert abs(east[1][0]) < 1e-9
    assert north[1][0] > 0
    assert abs(north[0][0]) < 1e-15
