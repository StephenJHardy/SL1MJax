from __future__ import annotations

import numpy as np

from sl1mjax.evla_c_independent_rime import (
    independent_apply_jones,
    independent_circular_stokes,
    independent_dual_antenna_visibility,
    independent_geometric_fringe,
    independent_holoraster_visibility,
    independent_offset_field_visibility,
    independent_parallactic_jones,
    independent_sky_from_feed,
    manufactured_asymmetric_jones,
    pack_cassbeam_native_columns,
    source_lm_is_negative_commanded,
)
from sl1mjax.evla_c_validation_refresh import ALGEBRA_ATOL
from sl1mjax.holography_beam_prior import predict_vis_numpy, sky_frame_residual_numpy
from sl1mjax.holography_c147_offset_ring import (
    apply_geometric_fringe,
    geometric_fringe_phase,
    predict_dual_antenna_numpy,
    predict_offset_field_visibilities,
)
from sl1mjax.holography_holoraster_coordinates import source_lm_feed_from_commanded_azelgeo
from sl1mjax.polarization import apply_jones_to_coherency, circular_stokes_to_coherency


def test_circular_stokes_i_q_u_v_independent() -> None:
    i_only = independent_circular_stokes(8.0)
    np.testing.assert_allclose(i_only[0, 0], 8.0)
    np.testing.assert_allclose(i_only[1, 1], 8.0)
    np.testing.assert_allclose(i_only[0, 1], 0.0)
    plus_q = independent_circular_stokes(8.0, 0.5, 0.0, 0.0)
    np.testing.assert_allclose(plus_q[0, 1], 0.5)
    np.testing.assert_allclose(plus_q[1, 0], 0.5)
    plus_u = independent_circular_stokes(8.0, 0.0, 0.25, 0.0)
    np.testing.assert_allclose(plus_u[0, 1], 0.25j)
    np.testing.assert_allclose(plus_u[1, 0], -0.25j)
    plus_v = independent_circular_stokes(8.0, 0.0, 0.0, 0.1)
    np.testing.assert_allclose(plus_v[0, 0], 8.1)
    np.testing.assert_allclose(plus_v[1, 1], 7.9)
    packed = circular_stokes_to_coherency(8.0, 0.5, 0.25, 0.1)
    np.testing.assert_allclose(
        independent_circular_stokes(8.0, 0.5, 0.25, 0.1), packed, atol=ALGEBRA_ATOL
    )


def test_asymmetric_jones_and_reversed_baseline() -> None:
    jp = manufactured_asymmetric_jones()
    jq = manufactured_asymmetric_jones(scale=0.8 - 0.15j)
    sky = independent_circular_stokes(8.0, 0.2, -0.1, 0.05)
    vis = independent_apply_jones(jp, sky, jq)
    production = apply_jones_to_coherency(sky, jp, jq)
    np.testing.assert_allclose(vis, production, atol=ALGEBRA_ATOL)
    reversed_vis = independent_apply_jones(jq, sky, jp)
    np.testing.assert_allclose(
        reversed_vis, np.conjugate(np.swapaxes(vis, -1, -2)), atol=ALGEBRA_ATOL
    )


def test_nonzero_parallactic_matches_production() -> None:
    feed = np.broadcast_to(manufactured_asymmetric_jones(), (3, 2, 2)).copy()
    chi = np.array([0.2, -0.4, 1.1], dtype=np.float64)
    sky = independent_sky_from_feed(feed, chi)
    production = sky_frame_residual_numpy(feed, chi)
    np.testing.assert_allclose(sky, production, atol=ALGEBRA_ATOL)
    para = independent_parallactic_jones(chi)
    np.testing.assert_allclose(para[:, 0, 0] * para[:, 1, 1], 1.0, atol=ALGEBRA_ATOL)


def test_holoraster_mover_on_both_baseline_sides() -> None:
    n = 5
    r_m = np.broadcast_to(manufactured_asymmetric_jones(), (n, 2, 2)).copy()
    r_r = np.broadcast_to(manufactured_asymmetric_jones(scale=0.95 + 0.05j), (n, 2, 2)).copy()
    beam = np.broadcast_to(manufactured_asymmetric_jones(scale=1.02), (n, 2, 2)).copy()
    source = np.broadcast_to(independent_circular_stokes(8.0, 0.01, -0.02, 0.0), (n, 2, 2)).copy()
    moving_is_p = np.array([True, False, True, False, True])
    vis = independent_holoraster_visibility(r_m, beam, source, r_r, moving_is_p)
    production = predict_vis_numpy(r_m, beam, source, r_r, moving_is_p)
    np.testing.assert_allclose(vis, production, atol=ALGEBRA_ATOL)
    assert vis.shape[0] == n


def test_offset_field_fringe_matches_production() -> None:
    n = 4
    r_p = np.broadcast_to(manufactured_asymmetric_jones(), (n, 1, 2, 2)).copy()
    r_q = np.broadcast_to(manufactured_asymmetric_jones(scale=0.9), (n, 1, 2, 2)).copy()
    e_p = np.broadcast_to(manufactured_asymmetric_jones(scale=1.05), (n, 1, 2, 2)).copy()
    e_q = np.broadcast_to(manufactured_asymmetric_jones(scale=0.97), (n, 1, 2, 2)).copy()
    source = independent_circular_stokes(8.0, 0.0, 0.0, 0.0)
    uvw = np.array(
        [[100.0, -40.0, 12.0], [80.0, 20.0, -5.0], [-30.0, 90.0, 8.0], [10.0, 15.0, 1.0]]
    )
    freq = np.array([4.564e9])
    lm = np.array([[1.0e-3, -5.0e-4], [8.0e-4, 2.0e-4], [-6.0e-4, 7.0e-4], [2.0e-4, -3.0e-4]])
    vis = independent_offset_field_visibility(
        r_p, e_p, source, e_q, r_q, uvw_m=uvw, frequency_hz=freq, sky_lm_rad=lm
    )
    production = predict_offset_field_visibilities(
        r_p, e_p, source, e_q, r_q, uvw_m=uvw, frequency_hz=freq, sky_lm_rad=lm
    )
    np.testing.assert_allclose(vis, production, atol=ALGEBRA_ATOL)
    np.testing.assert_allclose(
        independent_geometric_fringe(uvw, freq, lm),
        geometric_fringe_phase(uvw, freq, lm),
        atol=ALGEBRA_ATOL,
    )
    bare = independent_dual_antenna_visibility(r_p, e_p, source, e_q, r_q)
    np.testing.assert_allclose(
        bare, predict_dual_antenna_numpy(r_p, e_p, source, e_q, r_q), atol=ALGEBRA_ATOL
    )


def test_deliberate_missing_conjugation_fails() -> None:
    jp = manufactured_asymmetric_jones()
    jq = manufactured_asymmetric_jones(scale=0.7 + 0.4j)
    sky = independent_circular_stokes(8.0, 0.3, 0.1, 0.0)
    correct = independent_apply_jones(jp, sky, jq)
    wrong = jp @ sky @ jq
    assert not np.allclose(wrong, correct, atol=ALGEBRA_ATOL)


def test_deliberate_missing_parallactic_fails() -> None:
    feed = np.broadcast_to(manufactured_asymmetric_jones(), (2, 2, 2)).copy()
    chi = np.array([0.7, -0.5])
    correct = independent_sky_from_feed(feed, chi)
    assert not np.allclose(correct, feed, atol=ALGEBRA_ATOL)


def test_deliberate_fringe_sign_flip_fails() -> None:
    uvw = np.array([[120.0, -30.0, 10.0]])
    freq = np.array([4.5e9])
    lm = np.array([[0.001, 0.0004]])
    correct = independent_geometric_fringe(uvw, freq, lm)
    flipped = np.conjugate(correct)
    assert not np.allclose(flipped, correct, atol=ALGEBRA_ATOL)
    vis = np.ones((1, 1, 2, 2), dtype=np.complex128)
    applied = apply_geometric_fringe(vis, uvw, freq, lm)
    assert not np.allclose(applied, vis, atol=ALGEBRA_ATOL)


def test_source_coordinate_sign_is_negative_commanded() -> None:
    commanded = np.array([[0.001, -0.002], [0.0, 0.003]])
    source = source_lm_is_negative_commanded(commanded)
    np.testing.assert_allclose(source, source_lm_feed_from_commanded_azelgeo(commanded))
    assert not np.allclose(source, commanded)


def test_cassbeam_column_packing() -> None:
    columns = np.array([[1.0, 0.1, 0.02, -0.03, 0.04, 0.05, 0.9, -0.2]])
    jones = pack_cassbeam_native_columns(columns)
    np.testing.assert_allclose(jones[0, 0, 0], 1.0 + 0.1j)
    np.testing.assert_allclose(jones[0, 1, 0], 0.02 - 0.03j)
    np.testing.assert_allclose(jones[0, 0, 1], 0.04 + 0.05j)
    np.testing.assert_allclose(jones[0, 1, 1], 0.9 - 0.2j)
