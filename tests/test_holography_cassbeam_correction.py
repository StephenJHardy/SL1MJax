from __future__ import annotations

import inspect
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from sl1mjax.cassbeam_highres import diagonal_projection
from sl1mjax.holography_beam_prior import (
    CASSBEAM_QUERY_DECIMALS,
    compare_holoraster_stages,
    evaluate_holoraster_cassbeam,
    predict_from_moving_beams,
    unique_feed_frame_jones,
    unique_native_jones,
)
from sl1mjax.holography_cassbeam_correction import (
    ARCMIN_TO_RAD,
    FROZEN_HYPERPARAMETERS,
    IDENTITY_CORRECTION,
    CorrectionSamples,
    CorrectionState,
    accepted_prefix,
    apply_feed_frame_correction,
    apply_parallactic_after_correction,
    cassbeam_hand_peaks_rad,
    catalog_diagonal_feed_lookup,
    correction_path_stages,
    gaussian_diagonal_lookup,
    identity_predictions,
    identity_residual_jones,
    ladder_result_to_dict,
    multiplicative_envelope,
    next_term,
    predict_from_state,
    predict_moving_reference_vis,
    refuse_additive_map,
    require_aligned_frozen_prediction,
    require_identity_matches_baseline,
    run_first_ladder,
    shrink_azimuthal_coefficients,
    with_accepted_term,
)
from sl1mjax.holography_highres_cassbeam import (
    DEFAULT_CONVENTION,
    apply_axis_convention,
    normalize_after_jones_convention,
)
from test_holography_highres_cassbeam import _FREQ_HZ, _write_test_catalog
from sl1mjax.holography_diagonal_correction import (
    SPW4_HOLDOUT_MOVING_ANTENNA_NAMES,
    require_boresight_unity,
)
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
    offset = np.array([[0.0, 0.0], [1.0e-4, 0.0]])
    moving = np.array([4, 5], dtype=np.int32)
    reference = np.array([2, 2], dtype=np.int32)
    vis = np.zeros((2, 2, 2), dtype=np.complex128)
    vis[:, 0, 0] = [1.0, 0.5]
    require_aligned_frozen_prediction(
        offset, moving, reference, vis, offset, moving, reference, vis
    )
    with pytest.raises(ValueError, match="antenna order"):
        require_aligned_frozen_prediction(
            offset, moving, reference, vis, offset, moving[::-1], reference, vis
        )


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
        accepted_prefix(
            "pointing_offset",
            pointing_l_rad=1.0 * ARCMIN_TO_RAD,
            pointing_m_rad=-0.4 * ARCMIN_TO_RAD,
        ),
        accepted_prefix(
            "first_sidelobe_radius_amplitude",
            sidelobe_radius_scale=0.92,
            sidelobe_amplitude=1.3,
        ),
        accepted_prefix("low_order_azimuthal", azimuthal=(0.05, -0.04, 0.02, 0.01)),
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
    with pytest.raises(ValueError, match="can change only after"):
        CorrectionState(width_scale=1.08)
    with pytest.raises(ValueError, match="prefix of the first ladder"):
        CorrectionState(accepted_terms=("identity", "beam_width"))


def test_feed_frame_kernels_have_no_row_loop() -> None:
    source = inspect.getsource(apply_feed_frame_correction)
    assert "for row" not in source
    assert "for _row" not in source
    assert "for row" not in inspect.getsource(predict_moving_reference_vis)
    assert "for row" not in inspect.getsource(run_first_ladder)
    assert "for row" not in inspect.getsource(predict_from_state)
    assert "for row" not in inspect.getsource(evaluate_holoraster_cassbeam)
    assert "for row" not in inspect.getsource(unique_feed_frame_jones)
    assert "unique_feed_frame_jones" in inspect.getsource(catalog_diagonal_feed_lookup)


def _antenna_names() -> tuple[str, ...]:
    names = [""] * 29
    for index in range(1, 29):
        names[index] = f"ea{index:02d}"
    return tuple(names)


def _manufactured_samples(*, squint_scale: float = 1.25) -> tuple[CorrectionSamples, object]:
    lookup = _lookup()
    train_cells = np.array(
        [[2.0, 0.0], [0.0, 2.0], [-2.0, 0.0]],
        dtype=np.float64,
    ) * ARCMIN_TO_RAD
    hold_cells = np.array(
        [[3.0, 1.0], [-1.0, 3.0], [2.0, -2.0]],
        dtype=np.float64,
    ) * ARCMIN_TO_RAD
    train_movers = np.array([4, 5], dtype=np.int32)
    hold_movers = np.array([8, 13, 18, 22, 28], dtype=np.int32)
    offsets = []
    moving = []
    train = []
    spatial = []
    mover = []
    for cell in train_cells:
        for antenna in train_movers:
            offsets.append(cell)
            moving.append(antenna)
            train.append(True)
            spatial.append(False)
            mover.append(False)
        for antenna in hold_movers:
            offsets.append(cell)
            moving.append(antenna)
            train.append(False)
            spatial.append(False)
            mover.append(True)
    for cell in hold_cells:
        for antenna in train_movers:
            offsets.append(cell)
            moving.append(antenna)
            train.append(False)
            spatial.append(True)
            mover.append(False)
    offset = np.asarray(offsets, dtype=np.float64)
    n = offset.shape[0]
    dummy = np.zeros((n, 2, 2), dtype=np.complex128)
    dummy[:, 0, 0] = 1.0
    dummy[:, 1, 1] = 1.0
    moving_id = np.asarray(moving, dtype=np.int32)
    reference_id = np.full(n, 2, dtype=np.int32)
    samples = CorrectionSamples(
        offset_lm_rad=offset,
        measured=dummy,
        baseline=dummy,
        weight=np.ones((n, 2, 2)),
        source=np.eye(2, dtype=np.complex128) * 8.0,
        moving_is_p=np.ones(n, dtype=bool),
        moving_id=moving_id,
        reference_id=reference_id,
        residual_jones=identity_residual_jones(np.concatenate([moving_id, reference_id])),
        parallactic_angle_rad=np.zeros(n, dtype=np.float64),
        reference_parallactic_angle_rad=np.zeros(n, dtype=np.float64),
        antenna_names=_antenna_names(),
        train=np.asarray(train, dtype=bool),
        spatial_holdout=np.asarray(spatial, dtype=bool),
        mover_holdout=np.asarray(mover, dtype=bool),
        main_lobe=np.ones(n, dtype=bool),
        mid=np.zeros(n, dtype=bool),
        outer=np.zeros(n, dtype=bool),
        field_id=np.full(n, 10, dtype=np.int32),
    )
    truth = (
        IDENTITY_CORRECTION
        if abs(float(squint_scale) - 1.0) <= 1.0e-15
        else with_accepted_term(
            IDENTITY_CORRECTION, "rl_squint_scale", squint_scale=float(squint_scale)
        )
    )
    baseline = predict_from_state(samples, lookup, IDENTITY_CORRECTION)
    measured = predict_from_state(samples, lookup, truth)
    filled = replace(
        samples,
        measured=measured,
        baseline=baseline,
    )
    return filled, lookup


def test_ladder_accepts_squint_and_stops_without_opening_spw5() -> None:
    samples, lookup = _manufactured_samples(squint_scale=1.25)
    result = run_first_ladder(samples, lookup, n_boot=64, seed=3)
    assert result.identity_matches_baseline is True
    assert result.holdout_scores_preserved is True
    assert "rl_squint_scale" in result.accepted.accepted_terms
    assert result.accepted.squint_scale == pytest.approx(1.25, abs=0.03)
    assert result.selection_record[0].term == "rl_squint_scale"
    assert result.selection_record[0].decision == "accept_candidate"
    assert set(result.selection_record[0].region_residual_power) >= {
        "main_lobe",
        "mid",
        "outer_diagnostic",
    }
    assert result.development_refit is not None
    assert result.development_refit.accepted_terms == result.accepted.accepted_terms
    if result.stopped_at is not None:
        assert result.stopped_at != "rl_squint_scale"
    assert result.selection_record[0].mover_units.names == SPW4_HOLDOUT_MOVING_ANTENNA_NAMES
    assert set(FROZEN_HYPERPARAMETERS) >= {
        "squint_scale_grid",
        "width_scale_grid",
        "pointing_arcmin_grid",
        "sidelobe_radius_grid",
        "sidelobe_amplitude_grid",
        "paired_delta_bootstrap",
        "bootstrap_seed",
    }
    recorded = ladder_result_to_dict(result)
    assert recorded["hyperparameters"]["bootstrap_seed"] == 3
    assert recorded["hyperparameters"]["paired_delta_bootstrap"] == 64
    assert recorded["selection_record"][0]["mover_units"]["names"] == list(
        SPW4_HOLDOUT_MOVING_ANTENNA_NAMES
    )
    assert recorded["identity_matches_baseline"] is True


def test_ladder_keeps_identity_when_the_data_are_the_baseline() -> None:
    samples, lookup = _manufactured_samples(squint_scale=1.0)
    result = run_first_ladder(samples, lookup, n_boot=64, seed=5)
    assert result.accepted.is_identity()
    assert result.stopped_at == "rl_squint_scale"
    assert result.selection_record[0].decision == "keep_baseline"


def test_development_samples_cannot_include_c147_or_spw5() -> None:
    samples, _lookup = _manufactured_samples()
    with pytest.raises(RuntimeError, match="SPW 5"):
        replace(samples, frequency_hz=4.692e9, spectral_window_id=5)
    fields = np.full(samples.train.size, 10, dtype=np.int32)
    fields[np.flatnonzero(samples.train)[0]] = 3
    with pytest.raises(RuntimeError, match="C147"):
        replace(samples, field_id=fields)
    with pytest.raises(ValueError, match="channel 32"):
        replace(samples, frequency_hz=4.532e9, spectral_window_id=4)
    with pytest.raises(ValueError, match="SPW 4"):
        replace(samples, spectral_window_id=3)
    with pytest.raises(ValueError, match="exactly one channel"):
        replace(samples, measured=np.zeros((samples.train.size, 2, 2, 2), dtype=np.complex128))
    overlap = samples.spatial_holdout.copy()
    overlap[np.flatnonzero(samples.train)[0]] = True
    with pytest.raises(ValueError, match="disjoint"):
        replace(samples, spatial_holdout=overlap)
    empty = np.zeros(samples.train.size, dtype=bool)
    with pytest.raises(ValueError, match="non-empty"):
        replace(samples, train=empty)


def test_identity_reproduces_the_frozen_two_antenna_rime() -> None:
    samples, lookup = _manufactured_samples(squint_scale=1.0)
    residual = dict(samples.residual_jones)
    residual[4] = np.array([[1.07, 0.02], [-0.01, 0.94]], dtype=np.complex128)
    residual[2] = np.array([[0.96, 0.0], [0.03, 1.04]], dtype=np.complex128)
    chi_m = np.linspace(0.0, 0.4, samples.train.size)
    chi_r = np.linspace(-0.1, 0.15, samples.train.size)
    dirty = replace(
        samples,
        residual_jones=residual,
        parallactic_angle_rad=chi_m,
        reference_parallactic_angle_rad=chi_r,
    )
    identity = predict_from_state(dirty, lookup, IDENTITY_CORRECTION)
    feed = apply_feed_frame_correction(dirty.offset_lm_rad, lookup, IDENTITY_CORRECTION)
    sky = apply_parallactic_after_correction(feed, dirty.parallactic_angle_rad)
    expected = predict_from_moving_beams(
        dirty.residual_jones,
        dirty.moving_id,
        dirty.reference_id,
        dirty.moving_is_p,
        dirty.parallactic_angle_rad,
        dirty.reference_parallactic_angle_rad,
        sky,
        dirty.source,
    )
    np.testing.assert_allclose(identity, expected[:, 0])
    assert not np.allclose(identity, dirty.baseline)


def test_azimuthal_envelope_cannot_change_sign() -> None:
    offset = np.array([[8.0 * ARCMIN_TO_RAD, 0.0], [0.0, 8.0 * ARCMIN_TO_RAD]])
    shrunk = shrink_azimuthal_coefficients(offset, (2.5, 2.5, 2.5, 2.5))
    state = accepted_prefix("low_order_azimuthal", azimuthal=shrunk)
    angle = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    radius = 16.0 * ARCMIN_TO_RAD
    env_r, env_l = multiplicative_envelope(
        np.stack([radius * np.sin(angle), radius * np.cos(angle)], axis=1),
        state,
    )
    assert np.all(env_r > 0.0)
    assert np.all(env_l > 0.0)
    assert max(abs(item) for item in shrunk) < 2.5


def _raw_catalog_diagonal_lookup(catalog, frequency_hz: float):
    plane = catalog.plane(float(frequency_hz))
    origin = plane.origin_native()

    def lookup(offset_lm_rad):
        offset = np.asarray(offset_lm_rad, dtype=np.float64).reshape(-1, 2)
        l_q, m_q = apply_axis_convention(offset, DEFAULT_CONVENTION)
        native, ok = plane.lookup_native(l_q, m_q)
        looked = diagonal_projection(
            normalize_after_jones_convention(native, origin, DEFAULT_CONVENTION)
        )
        return np.where(np.asarray(ok, dtype=bool)[:, None, None], looked, np.nan)

    return lookup


def test_canonical_cassbeam_paths_agree_on_perturbed_catalog(tmp_path: Path) -> None:
    catalog = _write_test_catalog(tmp_path)
    base = np.array(
        [[5.0e-4, -3.0e-4], [2.0e-4, 4.0e-4], [-4.0e-4, 1.0e-4]],
        dtype=np.float64,
    )
    perturbation = 4.0e-13
    offset = np.vstack([base, base + perturbation, base - 3.0e-13])
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
    freqs = np.asarray([_FREQ_HZ], dtype=np.float64)
    native, _valid, inverse = unique_native_jones(
        offset, chi_m, freqs, catalog, DEFAULT_CONVENTION
    )
    assert native.shape[0] == base.shape[0]
    assert CASSBEAM_QUERY_DECIMALS == 12
    np.testing.assert_array_equal(inverse[:3], inverse[3:6])

    unique_feed, unique_ok = unique_feed_frame_jones(
        offset, catalog, freqs, DEFAULT_CONVENTION
    )
    assert bool(np.all(unique_ok))
    lookup = catalog_diagonal_feed_lookup(catalog, _FREQ_HZ)
    looked = lookup(offset)
    np.testing.assert_allclose(looked, unique_feed[:, 0], rtol=0.0, atol=0.0, equal_nan=True)

    comparison = evaluate_holoraster_cassbeam(
        catalog=catalog,
        frequencies_hz=freqs,
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
    correction = correction_path_stages(
        offset,
        lookup,
        IDENTITY_CORRECTION,
        residual_jones=residual,
        moving_id=moving,
        reference_id=reference,
        moving_is_p=moving_is_p,
        chi_moving=chi_m,
        chi_reference=chi_r,
        source=source,
    )
    report = compare_holoraster_stages(comparison, correction, atol=1.0e-12)
    assert report["first_divergent_stage"] is None
    require_identity_matches_baseline(correction["visibility"], comparison.visibility[:, 0])

    raw = _raw_catalog_diagonal_lookup(catalog, _FREQ_HZ)(offset)
    raw_vs_unique = compare_holoraster_stages(
        {"feed": unique_feed, "sky": unique_feed, "visibility": unique_feed},
        {"feed": raw, "sky": raw, "visibility": raw},
        atol=1.0e-12,
    )
    assert raw_vs_unique["first_divergent_stage"] == "feed"
    assert raw_vs_unique["feed"]["RR"]["max_abs"] > 1.0e-12 or raw_vs_unique["feed"]["LL"][
        "max_abs"
    ] > 1.0e-12
