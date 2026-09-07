from __future__ import annotations

import inspect

import numpy as np
import pytest

from sl1mjax.holography_cassbeam_correction import (
    ARCMIN_TO_RAD,
    IDENTITY_CORRECTION,
    CorrectionState,
    apply_feed_frame_correction,
    apply_parallactic_after_correction,
    cassbeam_hand_peaks_rad,
    gaussian_diagonal_lookup,
    identity_predictions,
    multiplicative_envelope,
    next_term,
    predict_moving_reference_vis,
    refuse_additive_map,
    require_identity_matches_baseline,
    with_accepted_term,
)
from sl1mjax.holography_diagonal_correction import require_boresight_unity
from sl1mjax.polarization import circular_parallactic_jones


def _lookup():
    peak_r, peak_l = cassbeam_hand_peaks_rad()
    return gaussian_diagonal_lookup(peak_r, peak_l, 8.0 * ARCMIN_TO_RAD)


def test_identity_matches_the_feed_frame_lookup() -> None:
    lookup = _lookup()
    offset = np.array([[0.0, 0.0], [3.0 * ARCMIN_TO_RAD, -2.0 * ARCMIN_TO_RAD]])
    corrected = apply_feed_frame_correction(offset, lookup, IDENTITY_CORRECTION)
    baseline = lookup(offset)
    require_identity_matches_baseline(corrected, baseline)
    copied = identity_predictions(baseline)
    require_identity_matches_baseline(copied, baseline)
    with pytest.raises(ValueError, match="frozen baseline"):
        require_identity_matches_baseline(corrected * 1.01, baseline)


def test_every_warp_keeps_boresight_unity() -> None:
    lookup = _lookup()
    origin = np.zeros((1, 2))
    cass0 = lookup(origin)[0]
    states = (
        with_accepted_term(IDENTITY_CORRECTION, "rl_squint_scale", squint_scale=1.25),
        with_accepted_term(
            with_accepted_term(IDENTITY_CORRECTION, "rl_squint_scale", squint_scale=1.25),
            "beam_width",
            width_scale=1.08,
        ),
        CorrectionState(pointing_l_rad=1.0 * ARCMIN_TO_RAD, pointing_m_rad=-0.4 * ARCMIN_TO_RAD),
        CorrectionState(sidelobe_radius_scale=0.92, sidelobe_amplitude=1.3),
        CorrectionState(azimuthal=(0.05, -0.04, 0.02, 0.01)),
    )
    for state in states:
        corrected = apply_feed_frame_correction(origin, lookup, state)[0]
        require_boresight_unity(corrected[0, 0] / cass0[0, 0], corrected[1, 1] / cass0[1, 1])
        env_r, env_l = multiplicative_envelope(origin, state)
        assert env_r[0] == pytest.approx(1.0)
        assert env_l[0] == pytest.approx(1.0)


def test_squint_scale_moves_hand_peaks_without_a_flux_offset() -> None:
    lookup = _lookup()
    peak_r, _peak_l = cassbeam_hand_peaks_rad()
    grid = np.linspace(-1.2 * ARCMIN_TO_RAD, 1.2 * ARCMIN_TO_RAD, 49)
    ll, mm = np.meshgrid(grid, grid, indexing="ij")
    offset = np.stack([ll.reshape(-1), mm.reshape(-1)], axis=1)
    identity = apply_feed_frame_correction(offset, lookup, IDENTITY_CORRECTION)
    scaled = apply_feed_frame_correction(
        offset,
        lookup,
        with_accepted_term(IDENTITY_CORRECTION, "rl_squint_scale", squint_scale=1.3),
    )
    ident_peak = offset[int(np.argmax(np.abs(identity[:, 0, 0])))]
    scaled_peak = offset[int(np.argmax(np.abs(scaled[:, 0, 0])))]
    assert np.linalg.norm(scaled_peak - 1.3 * peak_r) < np.linalg.norm(ident_peak - 1.3 * peak_r)


def test_parallactic_rotation_happens_after_the_feed_frame_correction() -> None:
    lookup = _lookup()
    offset = np.array([[4.0 * ARCMIN_TO_RAD, 1.0 * ARCMIN_TO_RAD]])
    chi = np.array([0.4])
    feed = apply_feed_frame_correction(offset, lookup, IDENTITY_CORRECTION)
    sky = apply_parallactic_after_correction(feed, chi)
    para = circular_parallactic_jones(chi)
    conjugate = np.conjugate(np.swapaxes(para, -1, -2))
    np.testing.assert_allclose(sky, conjugate @ feed @ para)
    cosine, sine = np.cos(chi[0]), np.sin(chi[0])
    rotated = np.array(
        [
            [
                offset[0, 0] * cosine + offset[0, 1] * sine,
                -offset[0, 0] * sine + offset[0, 1] * cosine,
            ]
        ]
    )
    rotated_lookup = apply_feed_frame_correction(rotated, lookup, IDENTITY_CORRECTION)
    assert not np.allclose(feed, rotated_lookup)


def test_predictions_are_complex_visibilities() -> None:
    jones = np.zeros((2, 2, 2), dtype=np.complex128)
    jones[:, 0, 0] = [0.8 + 0.1j, 0.5 - 0.2j]
    jones[:, 1, 1] = [0.7 - 0.05j, 0.6 + 0.0j]
    source = np.eye(2, dtype=np.complex128) * 8.0
    vis = predict_moving_reference_vis(jones, source, np.array([True, False]))
    np.testing.assert_allclose(vis[0, 0, 0], 8.0 * jones[0, 0, 0])
    np.testing.assert_allclose(vis[1, 1, 1], 8.0 * np.conjugate(jones[1, 1, 1]))
    with pytest.raises(RuntimeError, match="multiplicative envelopes"):
        refuse_additive_map("pixel_map")


def test_nested_terms_must_follow_the_ladder() -> None:
    assert next_term(IDENTITY_CORRECTION) == "rl_squint_scale"
    squint = with_accepted_term(IDENTITY_CORRECTION, "rl_squint_scale", squint_scale=1.2)
    assert next_term(squint) == "beam_width"
    with pytest.raises(ValueError, match="next nested term"):
        with_accepted_term(IDENTITY_CORRECTION, "beam_width", width_scale=1.05)
    with pytest.raises(RuntimeError, match="amplitude-only"):
        CorrectionState(accepted_terms=("identity", "outer_phase"))


def test_feed_frame_kernels_have_no_row_loop() -> None:
    source = inspect.getsource(apply_feed_frame_correction)
    assert "for row" not in source
    assert "for _row" not in source
    assert "for row" not in inspect.getsource(predict_moving_reference_vis)
