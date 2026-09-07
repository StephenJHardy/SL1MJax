from __future__ import annotations

import inspect

import numpy as np

from sl1mjax.holography_cassbeam_correction import (
    ARCMIN_TO_RAD,
    CorrectionSamples,
    cassbeam_hand_peaks_rad,
    gaussian_diagonal_lookup,
    identity_residual_jones,
)
from sl1mjax.holography_physical_squint import (
    IDENTITY_PHYSICAL,
    PhysicalBeamState,
    native_separation_rad,
)
from sl1mjax.holography_physical_squint_experiment import (
    candidate_state,
    common_support_maps,
    delta_from_parallel_perp,
    fit_width_on_train,
    interpret_experiment,
    invert_moving_sky_jones,
    measure_direct_squint,
    mover_direction_stable,
    native_direction_unit,
    recovered_feed_jones,
    score_model,
    sky_jones_to_feed_frame,
)


def _antenna_names() -> tuple[str, ...]:
    names = [""] * 29
    for index in range(1, 29):
        names[index] = f"ea{index:02d}"
    return tuple(names)


def _lookup():
    peak_r, peak_l = cassbeam_hand_peaks_rad()
    return gaussian_diagonal_lookup(peak_r, peak_l, 8.0 * ARCMIN_TO_RAD)


def _physical_samples(state: PhysicalBeamState) -> tuple[CorrectionSamples, object]:
    from sl1mjax.holography_physical_squint import physical_path_stages

    lookup = _lookup()
    train_cells = np.array([[2.0, 0.0], [0.0, 2.0], [-2.0, 0.0]]) * ARCMIN_TO_RAD
    hold_cells = np.array([[3.0, 1.0], [-1.0, 3.0], [2.0, -2.0]]) * ARCMIN_TO_RAD
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
    residual = identity_residual_jones(np.concatenate([moving_id, reference_id]))
    samples = CorrectionSamples(
        offset_lm_rad=offset,
        measured=dummy,
        baseline=dummy,
        weight=np.ones((n, 2, 2)),
        source=np.eye(2, dtype=np.complex128) * 8.0,
        moving_is_p=np.ones(n, dtype=bool),
        moving_id=moving_id,
        reference_id=reference_id,
        residual_jones=residual,
        parallactic_angle_rad=np.linspace(-0.2, 0.3, n),
        reference_parallactic_angle_rad=np.linspace(0.1, -0.15, n),
        antenna_names=_antenna_names(),
        train=np.asarray(train, dtype=bool),
        spatial_holdout=np.asarray(spatial, dtype=bool),
        mover_holdout=np.asarray(mover, dtype=bool),
        main_lobe=np.ones(n, dtype=bool),
        mid=np.zeros(n, dtype=bool),
        outer=np.zeros(n, dtype=bool),
        field_id=np.full(n, 10, dtype=np.int32),
    )
    baseline = physical_path_stages(
        samples.offset_lm_rad,
        lookup,
        IDENTITY_PHYSICAL,
        residual_jones=residual,
        moving_id=samples.moving_id,
        reference_id=samples.reference_id,
        moving_is_p=samples.moving_is_p,
        chi_moving=samples.parallactic_angle_rad,
        chi_reference=samples.reference_parallactic_angle_rad,
        source=samples.source,
    )["visibility"]
    measured = physical_path_stages(
        samples.offset_lm_rad,
        lookup,
        state,
        residual_jones=residual,
        moving_id=samples.moving_id,
        reference_id=samples.reference_id,
        moving_is_p=samples.moving_is_p,
        chi_moving=samples.parallactic_angle_rad,
        chi_reference=samples.reference_parallactic_angle_rad,
        source=samples.source,
    )["visibility"]
    from dataclasses import replace

    return replace(samples, measured=measured, baseline=baseline), lookup


def test_candidate_states_are_the_predeclared_models() -> None:
    native = native_separation_rad()
    emp = 1.2 * native
    assert candidate_state("no_squint", empirical_delta_rad=emp, width=1.04).delta()[0] == 0.0
    assert candidate_state("native", empirical_delta_rad=emp, width=1.04).is_identity()
    wide = candidate_state("native_plus_width", empirical_delta_rad=emp, width=1.04)
    assert wide.width == 1.04
    np.testing.assert_allclose(wide.delta(), native)
    empir = candidate_state("empirical", empirical_delta_rad=emp, width=1.0)
    np.testing.assert_allclose(empir.delta(), emp)
    assert empir.width == 1.0


def test_parallel_perp_recovers_native_separation() -> None:
    np.testing.assert_allclose(
        delta_from_parallel_perp(parallel_scale=1.0, perp_arcmin=0.0),
        native_separation_rad(),
    )
    unit = native_direction_unit()
    perp = delta_from_parallel_perp(parallel_scale=1.0, perp_arcmin=0.2)
    np.testing.assert_allclose(np.dot(perp - native_separation_rad(), unit), 0.0, atol=1.0e-15)


def test_common_support_ignores_hand_only_cells() -> None:
    offset = np.array(
        [[0.0, 0.0], [1.0e-4, 0.0], [4.0e-4, 0.0], [-4.0e-4, 0.0]],
        dtype=np.float64,
    )
    rr = np.array([1.0, 0.8, 10.0, 0.1])
    ll = np.array([0.9, 0.75, 0.1, 10.0])
    w = np.ones(4)
    maps = common_support_maps(
        offset,
        rr,
        ll,
        w,
        w,
        np.array([True, True, True, False]),
        np.array([True, True, False, True]),
    )
    assert maps["n_cells"] == 2
    assert float(np.max(np.abs(maps["offset"][:, 0]))) < 2.0e-4


def test_fit_width_recovers_an_interior_truth() -> None:
    samples, lookup = _physical_samples(PhysicalBeamState(width=1.04))
    fitted = fit_width_on_train(
        samples,
        lookup,
        delta_lm_rad=IDENTITY_PHYSICAL.delta(),
        grid=np.array([0.96, 1.00, 1.04, 1.08]),
    )
    assert fitted["width"] == 1.04
    assert fitted["interior"] is True
    native = score_model(
        samples,
        samples.baseline,
        samples.baseline,
        rr_ok=np.ones(samples.train.size, dtype=bool),
        ll_ok=np.ones(samples.train.size, dtype=bool),
        n_boot=32,
        seed=0,
    )
    assert abs(native["spatial"]["delta"]) <= 1.0e-12


def test_invert_kernels_have_no_row_loop() -> None:
    assert "for row" not in inspect.getsource(invert_moving_sky_jones)
    assert "for row" not in inspect.getsource(sky_jones_to_feed_frame)
    assert "for row" not in inspect.getsource(recovered_feed_jones)
    assert "for row" not in inspect.getsource(fit_width_on_train)
    assert "for row" not in inspect.getsource(measure_direct_squint)


def test_interpret_keeps_native_when_empirical_is_unresolved() -> None:
    spatial = {"delta": 0.01, "delta_lo": -0.01, "delta_hi": 0.02, "improves": False}
    moving = {"delta": 0.02, "delta_lo": 0.0, "delta_hi": 0.04, "improves": False}
    native_vs_none = {
        "spatial": {"delta": -0.02, "delta_lo": -0.03, "delta_hi": -0.01, "improves": True},
        "moving": {"delta": -0.01, "delta_lo": -0.02, "delta_hi": -0.005, "improves": True},
        "losses": {"mover_holdout": {"all": {"rr_minus_ll": 0.2}}},
        "baseline_rr_minus_ll": 0.4,
        "movers": {"unit_gate_passes": True},
    }
    empirical = {
        "spatial": spatial,
        "moving": moving,
        "movers": {"unit_gate_passes": False},
        "losses": {"mover_holdout": {"all": {"rr_minus_ll": 0.3}}},
    }
    direct = {
        "magnitude_arcmin": 0.52,
        "mover_bootstrap": {"magnitude_lo": 0.4},
        "by_mover": [
            {"position_angle_rad": 0.6},
            {"position_angle_rad": 0.65},
            {"position_angle_rad": 0.55},
        ],
    }
    width = {"interior": True, "width": 1.04}
    scored_width = {
        "spatial": {"improves": True},
        "moving": {"improves": True},
        "movers": {"unit_gate_passes": True},
    }
    out = interpret_experiment(
        native_vs_none=native_vs_none,
        empirical_vs_native=empirical,
        width_native_vs_native=scored_width,
        width_empirical_vs_empirical=scored_width,
        direct=direct,
        width_native=width,
        width_empirical=width,
        identity_ok=True,
        centroid_direction_ok=True,
    )
    assert out["outcome"] == "native_squint_detected_and_adequate"
    assert out["store_empirical_correction"] is False
    assert out["keep_native_squint"] is True
    assert out["width_survived"] is True
    assert mover_direction_stable(direct["by_mover"]) is True
