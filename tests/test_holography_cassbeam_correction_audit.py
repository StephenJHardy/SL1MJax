from __future__ import annotations

import inspect

import numpy as np
import pytest

from sl1mjax.holography_cassbeam_correction import (
    ARCMIN_TO_RAD,
    IDENTITY_CORRECTION,
    CorrectionState,
    accepted_prefix,
    apply_feed_frame_correction,
    cassbeam_hand_peaks_rad,
)
from sl1mjax.holography_cassbeam_correction_audit import (
    ACCEPTED_SQUINT_SCALE,
    ACCEPTED_WIDTH_SCALE,
    accepted_validation_state,
    hand_residual_power,
    interpret_physical_coherence,
    joint_squint_width_surface,
    leave_one_mover_sensitivity,
    model_offset_grid,
    refuse_reopening_rejected_ladder_terms,
    render_feed_frame_beams,
    split_region_hand_losses,
    squint_from_feed_jones,
    squint_moves_toward_holography,
    warped_hand_peaks_rad,
    warped_squint_separation_arcmin,
)
from test_holography_cassbeam_correction import _lookup, _manufactured_samples


def test_rejected_terms_cannot_be_reopened() -> None:
    refuse_reopening_rejected_ladder_terms(accepted_validation_state())
    refuse_reopening_rejected_ladder_terms(IDENTITY_CORRECTION)
    pointing = accepted_prefix(
        "pointing_offset",
        squint_scale=ACCEPTED_SQUINT_SCALE,
        width_scale=ACCEPTED_WIDTH_SCALE,
        pointing_l_rad=1.0e-5,
    )
    with pytest.raises(RuntimeError, match="stay rejected"):
        refuse_reopening_rejected_ladder_terms(pointing)


def test_warped_peaks_match_the_corrected_feed_frame_argmax() -> None:
    lookup = _lookup()
    state = accepted_validation_state()
    offset, _n = model_offset_grid(half_arcmin=2.0, step_arcmin=0.05)
    jones = render_feed_frame_beams(lookup, state, offset)
    peak_r = offset[int(np.argmax(np.abs(jones[:, 0, 0])))]
    peak_l = offset[int(np.argmax(np.abs(jones[:, 1, 1])))]
    expect_r, expect_l = warped_hand_peaks_rad(state)
    assert np.linalg.norm(peak_r - expect_r) < 0.08 * ARCMIN_TO_RAD
    assert np.linalg.norm(peak_l - expect_l) < 0.08 * ARCMIN_TO_RAD
    identity = apply_feed_frame_correction(offset, lookup, IDENTITY_CORRECTION)
    cass_r, cass_l = cassbeam_hand_peaks_rad()
    ident_sep = warped_squint_separation_arcmin(IDENTITY_CORRECTION)
    assert ident_sep == pytest.approx(float(np.hypot(*(cass_r - cass_l)) / ARCMIN_TO_RAD))
    corrected = squint_from_feed_jones(offset, jones, series="model_corrected")
    identity_sq = squint_from_feed_jones(offset, identity, series="model_identity")
    assert float(corrected["separation_arcmin"]) < float(identity_sq["separation_arcmin"])
    assert not squint_moves_toward_holography(0.515, 0.411, 0.35)


def test_split_region_hand_losses_and_leave_one_mover() -> None:
    samples, lookup = _manufactured_samples(squint_scale=1.25)
    truth = accepted_prefix("rl_squint_scale", squint_scale=1.25)
    from sl1mjax.holography_cassbeam_correction import predict_from_state

    candidate = predict_from_state(samples, lookup, truth)
    identity = samples.baseline
    losses = split_region_hand_losses(samples, candidate)
    assert set(losses) == {"train", "spatial_holdout", "mover_holdout"}
    assert losses["train"]["main_lobe"]["loss"] < losses["train"]["main_lobe"]["rr"] + 1.0
    assert losses["train"]["main_lobe"]["rr"] < hand_residual_power(
        samples.measured[samples.train],
        identity[samples.train],
        samples.weight[samples.train],
        hand="rr",
    )
    movers = leave_one_mover_sensitivity(samples, candidate, identity, n_boot=32, seed=2)
    assert movers["all_five_improve"] is True
    assert movers["loo_all_improve"] is True
    assert movers["movers"]["names"] == ["ea08", "ea13", "ea18", "ea22", "ea28"]
    assert len(movers["leave_one_out"]) == 5


def test_joint_surface_recovers_an_interior_manufactured_minimum() -> None:
    samples, lookup = _manufactured_samples(squint_scale=1.10)
    surface = joint_squint_width_surface(
        samples,
        lookup,
        squint_grid=np.array([1.00, 1.10, 1.20]),
        width_grid=np.array([0.96, 1.00, 1.04]),
    )
    assert surface["min_squint"] == pytest.approx(1.10)
    assert surface["min_width"] == pytest.approx(1.00)
    assert surface["identifiable_interior"] is True
    assert surface["minimum_on_grid_edge"] is False


def test_coherence_gate_keeps_spw5_closed_when_squint_moves_away() -> None:
    report = interpret_physical_coherence(
        measured_squint_arcmin=0.515,
        cassbeam_squint_arcmin=0.411,
        corrected_squint_arcmin=0.35,
        surface={"identifiable_interior": True, "accepted_on_grid_edge": True},
        movers={"unit_gate_passes": True},
    )
    assert report["physically_coherent"] is False
    assert report["freeze_spw4_family"] is False
    assert report["open_spw5_transfer"] is False
    assert report["description"] == "validation-selected SPW-4 diagonal correction"


def test_audit_kernels_have_no_row_loop() -> None:
    assert "for row" not in inspect.getsource(render_feed_frame_beams)
    assert "for row" not in inspect.getsource(split_region_hand_losses)
    assert "for row" not in inspect.getsource(leave_one_mover_sensitivity)
