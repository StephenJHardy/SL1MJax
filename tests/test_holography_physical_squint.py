from __future__ import annotations

import inspect

import numpy as np
import pytest
from test_holography_highres_cassbeam import _FREQ_HZ, _write_test_catalog

from sl1mjax.holography_beam_prior import evaluate_holoraster_cassbeam
from sl1mjax.holography_cassbeam_correction import (
    ARCMIN_TO_RAD,
    IDENTITY_CORRECTION,
    apply_feed_frame_correction,
    cassbeam_hand_peaks_rad,
    catalog_diagonal_feed_lookup,
    gaussian_diagonal_lookup,
)
from sl1mjax.holography_highres_cassbeam import DEFAULT_CONVENTION
from sl1mjax.holography_physical_squint import (
    IDENTITY_PHYSICAL,
    PhysicalBeamState,
    apply_physical_feed_frame,
    native_hand_centers_rad,
    native_separation_rad,
    physical_path_stages,
    physical_query_coordinates,
    refuse_envelope_or_phase,
    refuse_old_squint_scale_coupling,
    refuse_pointing_fit,
    require_physical_identity_matches_comparison,
    target_hand_centers_rad,
)


def test_identity_query_is_the_commanded_coordinate() -> None:
    offset = np.array([[0.0, 0.0], [1.2e-4, -8.0e-5], [3.0e-4, 2.0e-4]])
    query_r, query_l = physical_query_coordinates(offset, IDENTITY_PHYSICAL)
    np.testing.assert_allclose(query_r, offset)
    np.testing.assert_allclose(query_l, offset)
    assert IDENTITY_PHYSICAL.is_identity()


def test_width_does_not_move_target_centres() -> None:
    native = IDENTITY_PHYSICAL
    wide = PhysicalBeamState(width=1.08)
    t0 = target_hand_centers_rad(native)
    t1 = target_hand_centers_rad(wide)
    np.testing.assert_allclose(t0[0], t1[0])
    np.testing.assert_allclose(t0[1], t1[1])
    offset = np.array([[2.0e-4, 1.0e-4]])
    q0_r, q0_l = physical_query_coordinates(offset, native)
    q1_r, q1_l = physical_query_coordinates(offset, wide)
    assert not np.allclose(q0_r, q1_r)
    assert not np.allclose(q0_l, q1_l)


def test_delta_leaves_common_centre_and_width() -> None:
    native = native_separation_rad()
    scaled = PhysicalBeamState(delta_lm_rad=1.25 * native)
    t_r, t_l = target_hand_centers_rad(scaled)
    right, left = native_hand_centers_rad()
    np.testing.assert_allclose(0.5 * (t_r + t_l), 0.5 * (right + left))
    np.testing.assert_allclose(t_r - t_l, 1.25 * native)
    assert scaled.width == 1.0


def test_pointing_translates_both_hands_equally() -> None:
    shift = np.array([1.0e-5, -2.0e-5])
    moved = PhysicalBeamState(pointing_lm_rad=(float(shift[0]), float(shift[1])))
    t_r, t_l = target_hand_centers_rad(moved)
    n_r, n_l = target_hand_centers_rad(IDENTITY_PHYSICAL)
    np.testing.assert_allclose(t_r - n_r, shift)
    np.testing.assert_allclose(t_l - n_l, shift)
    np.testing.assert_allclose(t_r - t_l, n_r - n_l)


def test_identity_matches_catalog_lookup_and_frozen_rime(tmp_path) -> None:
    catalog = _write_test_catalog(tmp_path)
    lookup = catalog_diagonal_feed_lookup(catalog, _FREQ_HZ)
    base = np.array([[5.0e-4, -3.0e-4], [2.0e-4, 4.0e-4], [-4.0e-4, 1.0e-4]])
    offset = np.vstack([base, base + 4.0e-13, base - 3.0e-13])
    n = offset.shape[0]
    chi_m = np.linspace(0.15, 0.55, n)
    chi_r = np.linspace(-0.25, 0.20, n)
    moving = np.array([4, 5, 4, 5, 4, 5, 4, 5, 5], dtype=np.int32)
    reference = np.full(n, 2, dtype=np.int32)
    moving_is_p = np.array([True, False, True, False, True, False, True, False, True])
    source = np.zeros((n, 2, 2), dtype=np.complex128)
    source[:, 0, 0] = 8.0 + 0.15 * np.arange(n)
    source[:, 1, 1] = 7.2 - 0.08 * np.arange(n)
    residual = {
        2: np.array([[0.96 + 0.01j, 0.02], [-0.03, 1.04 - 0.02j]], dtype=np.complex128),
        4: np.array([[1.07, 0.02 + 0.01j], [-0.01, 0.94]], dtype=np.complex128),
        5: np.array([[0.91 - 0.02j, -0.03], [0.02, 1.11 + 0.01j]], dtype=np.complex128),
    }
    physical = physical_path_stages(
        offset,
        lookup,
        IDENTITY_PHYSICAL,
        residual_jones=residual,
        moving_id=moving,
        reference_id=reference,
        moving_is_p=moving_is_p,
        chi_moving=chi_m,
        chi_reference=chi_r,
        source=source,
    )
    comparison = evaluate_holoraster_cassbeam(
        catalog=catalog,
        frequencies_hz=np.asarray([_FREQ_HZ]),
        convention=DEFAULT_CONVENTION,
        offset_lm_rad=offset,
        chi_moving=chi_m,
        chi_reference=chi_r,
        moving_id=moving,
        reference_id=reference,
        moving_is_p=moving_is_p,
        residual_jones=residual,
        source=source,
        off_diagonal=False,
    )
    report = require_physical_identity_matches_comparison(physical, comparison)
    assert report["first_divergent_stage"] is None
    looked = lookup(offset)
    np.testing.assert_allclose(physical["feed"], looked, rtol=0.0, atol=1.0e-12)
    rr_ok = np.ones(n, dtype=bool)
    rr_ok[1] = False
    ll_ok = np.ones(n, dtype=bool)
    ll_ok[2] = False
    assert bool(np.any(~rr_ok) and np.any(~ll_ok))


def test_identity_matches_legacy_feed_frame_identity() -> None:
    peak_r, peak_l = cassbeam_hand_peaks_rad()
    lookup = gaussian_diagonal_lookup(peak_r, peak_l, 8.0 * ARCMIN_TO_RAD)
    offset = np.array([[0.0, 0.0], [3.0 * ARCMIN_TO_RAD, -2.0 * ARCMIN_TO_RAD]])
    physical = apply_physical_feed_frame(offset, lookup, IDENTITY_PHYSICAL)
    legacy = apply_feed_frame_correction(offset, lookup, IDENTITY_CORRECTION)
    np.testing.assert_allclose(physical, legacy, rtol=0.0, atol=1.0e-12)


def test_physical_query_is_not_the_old_coupled_warp() -> None:
    offset = np.array([[4.0e-4, -1.0e-4]])
    wide = PhysicalBeamState(width=1.04)
    q_r, _q_l = physical_query_coordinates(offset, wide)
    peak_r, _peak_l = native_hand_centers_rad()
    old = (offset - (0.85 - 1.0) * peak_r) / 1.04
    assert not np.allclose(q_r, old)
    with pytest.raises(RuntimeError, match="coupled"):
        refuse_old_squint_scale_coupling()
    with pytest.raises(RuntimeError, match="does not fit"):
        refuse_pointing_fit()
    with pytest.raises(RuntimeError, match="outside"):
        refuse_envelope_or_phase("rl_squint_scale")


def test_kernels_have_no_row_loop() -> None:
    assert "for row" not in inspect.getsource(physical_query_coordinates)
    assert "for row" not in inspect.getsource(apply_physical_feed_frame)
    assert "for row" not in inspect.getsource(physical_path_stages)


def test_invert_moving_jones_recovers_a_known_beam() -> None:
    from sl1mjax.holography_beam_prior import predict_vis_numpy, sky_frame_residual_numpy
    from sl1mjax.holography_physical_squint_experiment import (
        invert_moving_sky_jones,
        sky_jones_to_feed_frame,
    )
    from sl1mjax.polarization import circular_parallactic_jones

    n = 6
    chi = np.linspace(-0.3, 0.4, n)
    para = circular_parallactic_jones(chi)
    feed = np.zeros((n, 2, 2), dtype=np.complex128)
    feed[:, 0, 0] = 0.8 + 0.05j * np.arange(n)
    feed[:, 1, 1] = 0.7 - 0.03j * np.arange(n)
    conjugate = np.conjugate(np.swapaxes(para, -1, -2))
    sky = conjugate @ feed @ para
    residual = np.zeros((n, 2, 2), dtype=np.complex128)
    residual[:, 0, 0] = 1.05 + 0.02j
    residual[:, 1, 1] = 0.97 - 0.01j
    r_m = sky_frame_residual_numpy(residual, chi)
    r_r = sky_frame_residual_numpy(np.eye(2), -0.5 * chi)
    source = np.zeros((n, 2, 2), dtype=np.complex128)
    source[:, 0, 0] = 8.0 + 0.2 * np.arange(n)
    source[:, 1, 1] = 7.5 - 0.1 * np.arange(n)
    moving_is_p = np.array([True, False, True, False, True, False])
    vis = predict_vis_numpy(r_m, sky, source, r_r, moving_is_p)[:, 0]
    recovered = invert_moving_sky_jones(vis, source, r_m, r_r, moving_is_p)
    np.testing.assert_allclose(recovered, sky, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(
        sky_jones_to_feed_frame(recovered, chi), feed, rtol=0.0, atol=1.0e-12
    )


def test_common_support_requires_both_hands() -> None:
    from sl1mjax.holography_physical_squint_experiment import common_support_maps

    offset = np.array(
        [[0.0, 0.0], [1.0e-4, 0.0], [1.0e-4, 0.0], [2.0e-4, 0.0], [3.0e-4, 0.0]],
        dtype=np.float64,
    )
    rr = np.array([1.0, 0.8, 0.7, 0.4, 0.2])
    ll = np.array([0.9, 0.75, 0.7, 0.35, 0.15])
    w = np.ones(5)
    rr_ok = np.array([True, True, True, True, False])
    ll_ok = np.array([True, True, True, False, True])
    maps = common_support_maps(offset, rr, ll, w, w, rr_ok, ll_ok)
    assert maps["n_cells"] == 2
    assert "for row" not in inspect.getsource(common_support_maps)
