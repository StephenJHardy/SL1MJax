from __future__ import annotations

import numpy as np
import pytest

from sl1mjax.holography_beam_prior import (
    INJECTION_VOLTAGE,
    SPW4_MULTICHANNEL_BEAM_PRIOR,
    ComplexMoments,
    accumulate_hand_moments,
    all_conventions,
    assert_origin_uninjected,
    assert_split_isolation,
    bootstrap_complex_from_moments,
    bootstrap_increment_from_moments,
    classify_beam_prior_decision,
    contiguous_channel_block_masks,
    convention_equivalence_classes,
    fit_frequency_smooth_alpha,
    fit_per_antenna_if_supported,
    fit_spatial_shrinkage,
    fit_unit_and_scalar,
    flatten_row_channel,
    inject_voltage_leakage,
    moments_from_template,
    predict_vis_numpy,
    residual_power_at,
    select_convention_class,
    stack_moments,
    supported_prior_from_decision,
    vis_planes,
)
from sl1mjax.holography_highres_cassbeam import (
    DEFAULT_CONVENTION,
    antenna_frame_lm,
    apply_axis_convention,
    convention_ladder,
    resample_cluster_indices,
)


def _moments_from_alpha(
    alpha: complex, n_cluster: int = 20, n_per: int = 6
) -> list[ComplexMoments]:
    rng = np.random.default_rng(4)
    n = n_cluster * n_per
    t = np.zeros((n, 2, 2), dtype=np.complex128)
    t[:, 0, 1] = 0.05 + 0.01j
    t[:, 1, 0] = 0.04 - 0.012j
    noise = 0.002 * (rng.standard_normal(t.shape) + 1j * rng.standard_normal(t.shape))
    residual = alpha * t + noise
    weight = np.ones_like(t, dtype=np.float64)
    clusters = np.repeat(np.arange(n_cluster), n_per)
    return accumulate_hand_moments(
        t,
        residual,
        weight,
        cluster_ids=clusters,
        channel_ids=np.zeros(n, dtype=np.int64),
        dwell_ids=clusters,
        mover_ids=np.zeros(n, dtype=np.int64),
        reference_ids=np.ones(n, dtype=np.int64),
        cell_ids=np.zeros(n, dtype=np.int64),
    )


def test_split_isolation_refuses_overlapping_masks() -> None:
    train = np.array([True, True, False, False])
    hold = np.array([False, True, True, False])
    with pytest.raises(ValueError, match="contaminated split"):
        assert_split_isolation(train, hold)
    assert_split_isolation(train, np.array([False, False, True, True]))


def test_score_moments_use_only_selected_rows() -> None:
    t = np.zeros((6, 2, 2), dtype=np.complex128)
    r = np.zeros_like(t)
    t[:, 0, 1] = 1.0
    r[:, 0, 1] = np.array([1, 1, 1, 10, 10, 10], dtype=np.float64)
    w = np.ones_like(t, dtype=np.float64)
    train = accumulate_hand_moments(
        t[:3],
        r[:3],
        w[:3],
        cluster_ids=np.arange(3),
        channel_ids=np.zeros(3, dtype=np.int64),
        dwell_ids=np.arange(3),
        mover_ids=np.zeros(3, dtype=np.int64),
        reference_ids=np.ones(3, dtype=np.int64),
        cell_ids=np.zeros(3, dtype=np.int64),
    )
    mixed = accumulate_hand_moments(
        t,
        r,
        w,
        cluster_ids=np.arange(6),
        channel_ids=np.zeros(6, dtype=np.int64),
        dwell_ids=np.arange(6),
        mover_ids=np.zeros(6, dtype=np.int64),
        reference_ids=np.ones(6, dtype=np.int64),
        cell_ids=np.zeros(6, dtype=np.int64),
    )
    assert stack_moments(train).residual_power(0.0) == pytest.approx(3.0)
    assert stack_moments(mixed).residual_power(0.0) > 200.0


def test_cluster_bootstrap_keeps_replacement_multiplicities() -> None:
    labels = np.array([1, 1, 2, 2, 2], dtype=np.int64)

    class _Fixed:
        def choice(self, unique, size, replace):
            assert replace is True
            return np.array([1, 1], dtype=np.int64)

    index = resample_cluster_indices(labels, _Fixed())
    assert index.tolist() == [0, 1, 0, 1]


def test_complex_alpha_uncertainty_uses_real_and_imaginary_intervals() -> None:
    moments = _moments_from_alpha(0.0 + 0.7j)
    report = bootstrap_complex_from_moments(moments, n_boot=200, seed=2)
    assert report["consistent_with_zero"] is False
    assert report["consistent_with_one"] is False
    lo_i, hi_i = report["imag_ci95"]
    lo_r, hi_r = report["real_ci95"]
    assert lo_i < 0.7 < hi_i
    assert lo_r <= 0.15
    assert hi_r >= -0.15


def test_mount_frame_does_not_spatially_rotate_azelgeo() -> None:
    offset = np.array([[0.003, 0.001]], dtype=np.float64)
    l_m, m_m = apply_axis_convention(offset, DEFAULT_CONVENTION)
    l_s, m_s = apply_axis_convention(offset, DEFAULT_CONVENTION)
    l_rot, m_rot = antenna_frame_lm(l_s, m_s, np.array([0.5 * np.pi]))
    assert l_m[0] == pytest.approx(0.003)
    assert m_m[0] == pytest.approx(0.001)
    assert DEFAULT_CONVENTION.rotate_spatial is False
    assert not np.isclose(l_rot[0], l_m[0]) or not np.isclose(m_rot[0], m_m[0])


def test_convention_family_is_complete_and_finite() -> None:
    ladder = all_conventions()
    assert ladder == convention_ladder()
    assert len(ladder) == 128
    assert DEFAULT_CONVENTION in ladder
    names = [item.name for item in ladder]
    assert len(set(names)) == 128


def test_flatten_keeps_every_native_channel() -> None:
    vis = np.zeros((5, 64, 2, 2), dtype=np.complex128)
    vis[:, 7, 0, 1] = 3.0
    flat, rows, channels = flatten_row_channel(vis)
    assert flat.shape == (5 * 64, 2, 2)
    assert rows.size == 5 * 64
    assert channels.tolist()[:64] == list(range(64))
    assert np.count_nonzero(flat[:, 0, 1]) == 5
    assert vis_planes(vis).shape[1] == 64


def test_contiguous_channel_block_is_not_a_row_union() -> None:
    masks = contiguous_channel_block_masks(64, hold_start=48, hold_width=16)
    assert int(np.sum(masks["holdout"])) == 16
    assert int(np.sum(masks["train"])) == 48
    assert not bool(np.any(masks["train"] & masks["holdout"]))
    assert masks["hold_start"] == 48
    assert masks["hold_stop"] == 64


def test_nested_models_and_injection_increment() -> None:
    train = _moments_from_alpha(0.8 + 0.1j)
    scalar = fit_unit_and_scalar(train)
    assert abs(complex(scalar["alpha_hat"]) - (0.8 + 0.1j)) < 0.15
    assert scalar["power_hat"] < scalar["power_0"]
    smooth = fit_frequency_smooth_alpha(train)
    assert smooth["alpha_channel"].shape[0] >= 1
    spatial = fit_spatial_shrinkage(train)
    assert spatial["n_cells"] == 1
    antenna = fit_per_antenna_if_supported(train)
    assert antenna["n_movers"] == 1

    uninjected = _moments_from_alpha(0.05 + 0.0j, n_cluster=16)
    injected = _moments_from_alpha(0.05 + 0.3 + 0.0j, n_cluster=16)
    increment = bootstrap_increment_from_moments(uninjected, injected, n_boot=80, seed=1)
    assert increment["inconsistent_with_zero"] is True
    assert increment["real"] == pytest.approx(0.3, abs=0.08)


def test_injection_preserves_uninjected_origin_rows() -> None:
    vis = np.ones((4, 8, 2, 2), dtype=np.complex128)
    template = np.full_like(vis, 0.2 + 0.1j)
    origin = np.array([True, False, False, True])
    copies = inject_voltage_leakage(vis, template, 0.03, origin_mask=origin)
    assert_origin_uninjected(origin, copies, vis)
    assert copies[1, 0, 0, 1] != vis[1, 0, 0, 1]
    with pytest.raises(ValueError, match="origin rows were injected"):
        broken = np.array(copies)
        broken[0] += 1.0
        assert_origin_uninjected(origin, broken, vis)
    assert INJECTION_VOLTAGE == (0.001, 0.003, 0.01, 0.03)


def test_convention_equivalence_and_selection() -> None:
    t_a = np.zeros((10, 2, 2), dtype=np.complex128)
    t_a[:, 0, 1] = 0.04
    t_a[:, 1, 0] = 0.03
    t_b = t_a * (1.0 + 1.0e-6)
    t_c = t_a * 3.0
    weight = np.ones_like(t_a, dtype=np.float64)
    classes = convention_equivalence_classes({"a": t_a, "b": t_b, "c": t_c}, weight)
    assert any(set(group) >= {"a", "b"} for group in classes)
    locked = select_convention_class({"a": 1.0, "c": 4.0}, classes, margin=0.02)
    assert locked["status"] == "locked"
    tied = select_convention_class({"a": 1.0, "c": 1.001}, [["a"], ["c"]], margin=0.02)
    assert tied["status"] == "convention_unresolved"


def test_classifier_fail_closed_and_scientific_nondetection() -> None:
    failed = classify_beam_prior_decision(
        software_ok=False,
        split_clean=True,
        injection_detectable={"0.003": True},
        holdout_alpha=None,
        a1_improves=False,
        hat_improves=False,
        spatial_improves=False,
        rr_ll_regression=False,
    )
    assert failed["process_failure"] is True
    assert failed["decision"] == "software_gate_failed"
    contaminated = classify_beam_prior_decision(
        software_ok=True,
        split_clean=False,
        injection_detectable={"0.003": True},
        holdout_alpha=None,
        a1_improves=False,
        hat_improves=False,
        spatial_improves=False,
        rr_ll_regression=False,
    )
    assert contaminated["decision"] == "contaminated_split"
    limit = classify_beam_prior_decision(
        software_ok=True,
        split_clean=True,
        injection_detectable={"0.001": False, "0.003": False, "0.01": False, "0.03": False},
        holdout_alpha={"consistent_with_zero": True, "real": 0.0, "imag": 0.0},
        a1_improves=False,
        hat_improves=False,
        spatial_improves=False,
        rr_ll_regression=False,
    )
    assert limit["decision"] == "upper_limit_only"
    assert limit["process_failure"] is False
    assert limit["scientific_decision"] is True
    assert limit["full_jones_frozen"] is False
    assert limit["spw5_closed"] is True
    assert limit["most_important_next_artifact"] == SPW4_MULTICHANNEL_BEAM_PRIOR


def test_prior_contains_only_supported_quantities() -> None:
    prior = supported_prior_from_decision(
        decision="upper_limit_only",
        scalar={"alpha_hat": 0.4 + 0.1j},
        smooth={"alpha_channel": np.ones(4, dtype=np.complex128)},
        spatial={"alpha_cell": {0: 1.0}},
        antenna={"alpha_mover": {1: 0.2}, "array_alpha": 0.1},
        holdout_alpha={
            "real": 0.02,
            "imag": 0.0,
            "real_ci95": (-0.05, 0.08),
            "imag_ci95": (-0.04, 0.04),
        },
        provenance={"spw": 4},
        antenna_supported=True,
    )
    assert prior.frozen is False
    assert prior.full_jones_frozen is False
    assert prior.spw5_opened is False
    assert "off_diagonal_upper_abs" in prior.supported_quantities
    assert "off_diagonal_mean" not in prior.supported_quantities
    assert prior.off_diagonal_mean is None
    assert prior.off_diagonal_upper_abs is not None


def test_predict_vis_numpy_is_batched() -> None:
    n = 5
    identity = np.broadcast_to(np.eye(2, dtype=np.complex128), (n, 2, 2)).copy()
    beam = np.zeros((n, 3, 2, 2), dtype=np.complex128)
    beam[:, :, 0, 0] = 1.0
    beam[:, :, 1, 1] = 1.0
    beam[:, :, 0, 1] = 0.01
    source = np.zeros((n, 3, 2, 2), dtype=np.complex128)
    source[:, :, 0, 0] = 7.0
    source[:, :, 1, 1] = 7.0
    vis = predict_vis_numpy(identity, beam, source, identity, np.ones(n, dtype=bool))
    assert vis.shape == (n, 3, 2, 2)
    assert vis[0, 1, 0, 0] == pytest.approx(7.0)


def test_moments_from_template_do_not_average_channels() -> None:
    n_row, n_chan = 4, 8
    template = np.zeros((n_row, n_chan, 2, 2), dtype=np.complex128)
    residual = np.zeros_like(template)
    template[:, :, 0, 1] = 0.02
    residual[:, 2, 0, 1] = 0.02
    residual[:, 5, 0, 1] = 0.06
    weight = np.ones_like(template, dtype=np.float64)
    geometry = {
        "time_index": np.arange(n_row),
        "moving_id": np.zeros(n_row, dtype=np.int64),
        "reference_id": np.ones(n_row, dtype=np.int64),
        "cell_l": np.zeros(n_row, dtype=np.int64),
        "cell_m": np.zeros(n_row, dtype=np.int64),
    }
    moments = moments_from_template(
        template,
        residual,
        weight,
        geometry,
        np.arange(n_row),
        n_channel=n_chan,
    )
    channels = {item.channel for item in moments}
    assert channels == set(range(n_chan))
    power = residual_power_at(moments, np.ones(n_chan, dtype=np.complex128))
    assert np.isfinite(power)
    coarse = moments_from_template(
        template,
        residual,
        weight,
        geometry,
        np.arange(n_row),
        n_channel=n_chan,
        spatial_groups=False,
    )
    assert len(coarse) <= len(moments)
