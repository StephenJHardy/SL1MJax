from __future__ import annotations

import inspect

import numpy as np
import pytest

from sl1mjax.coordinates import radec_to_lmn
from sl1mjax.holography import source_relative_lm_rad
from sl1mjax.holography_beam_prior import ComplexMoments, accumulate_hand_moments
from sl1mjax.casa_awp2_oracle import geometric_visibility_phase
from sl1mjax.holography_c147_offset_ring import (
    C147_OFFSET_RING,
    apply_channel_holdout,
    apply_geometric_fringe,
    geometric_fringe_phase,
    diagnose_directional_disagreement,
    ns_ew_masks,
    predict_offset_field_visibilities,
    template_geometry_transforms,
    FieldSkyRecord,
    SourceSkyRecord,
    antenna_power_share,
    channel32_smoke_gates,
    classify_c147_offset_ring,
    clustered_null_upper_limit,
    diagonal_rr_ll_closure,
    declare_field_partitions,
    field_partition_masks,
    locked_convention,
    predict_dual_antenna_numpy,
    reconstruct_field_offsets,
    refuse_convention_search,
    refuse_placeholder_upper_limit,
    scale_compatible,
    select_3c147_sky_direction,
    select_training_qu,
    source_relative_offsets,
    upper_limit_to_dict,
)
from sl1mjax.holography_highres_cassbeam import DEFAULT_CONVENTION, convention_ladder


def _misleading_fields() -> list[FieldSkyRecord]:
    """Names lie. Geometry is an 8-point ring around 3C147."""

    ra0, dec0 = 1.4660765716752369, 0.8726646259971648
    radius = np.deg2rad(0.25)
    names = ("C147-S", "C147-SE", "C147-E", "C147-NE", "C147-N", "C147-NW", "C147-W", "C147-SW")
    fields = [
        FieldSkyRecord(0, "J0542+4951", (ra0, dec0), code="D"),
    ]
    for index, name in enumerate(names):
        angle = index * np.pi / 4.0
        l_rad = radius * np.sin(angle)
        m_rad = radius * np.cos(angle)
        ra, dec = _lm_to_radec(ra0, dec0, l_rad, m_rad)
        fields.append(FieldSkyRecord(index + 1, name, (ra, dec)))
    return fields


def _lm_to_radec(ra0: float, dec0: float, l_rad: float, m_rad: float) -> tuple[float, float]:
    n_rad = np.sqrt(max(0.0, 1.0 - l_rad * l_rad - m_rad * m_rad))
    from sl1mjax.coordinates import lmn_to_radec

    ra, dec = lmn_to_radec(ra0, dec0, l_rad, m_rad, n_rad)
    return float(np.asarray(ra)), float(np.asarray(dec))


def _moments(alpha: complex, n_cluster: int = 12, n_per: int = 4) -> list[ComplexMoments]:
    rng = np.random.default_rng(2)
    n = n_cluster * n_per
    t = np.zeros((n, 2, 2), dtype=np.complex128)
    t[:, 0, 1] = 0.04 + 0.01j
    t[:, 1, 0] = 0.035 - 0.008j
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
        dwell_ids=np.zeros(n, dtype=np.int64),
        mover_ids=np.zeros(n, dtype=np.int64),
        reference_ids=np.ones(n, dtype=np.int64),
        cell_ids=np.zeros(n, dtype=np.int64),
        hands=("rl", "lr"),
    )


def test_offsets_come_from_phase_dir_not_field_names() -> None:
    fields = _misleading_fields()
    sources = [SourceSkyRecord("J0542+4951", fields[0].phase_centre_rad, field_id=0)]
    source, origin = select_3c147_sky_direction(fields, sources)
    assert origin == "source_table_agrees_field_0"
    geos = reconstruct_field_offsets(fields, source, source_from=origin)
    pointed_north = next(item for item in geos if item.field_id == 1)
    assert pointed_north.name == "C147-S"
    assert pointed_north.m_rad < 0.0
    assert abs(pointed_north.l_rad) < 0.05 * pointed_north.radius_rad
    expected = radec_to_lmn(
        pointed_north.phase_centre_rad[0],
        pointed_north.phase_centre_rad[1],
        source[0],
        source[1],
    )
    assert np.isclose(pointed_north.l_rad, float(np.asarray(expected[0])))
    assert np.isclose(pointed_north.m_rad, float(np.asarray(expected[1])))


def test_source_and_field0_must_agree() -> None:
    fields = _misleading_fields()
    ra, dec = fields[0].phase_centre_rad
    sources = [SourceSkyRecord("3C147", (ra + np.deg2rad(10.0 / 3600.0), dec), field_id=0)]
    with pytest.raises(ValueError, match="disagree"):
        select_3c147_sky_direction(fields, sources)


def test_partitions_use_geometry_and_keep_fields_atomic() -> None:
    geos = reconstruct_field_offsets(
        _misleading_fields(),
        (1.4660765716752369, 0.8726646259971648),
        source_from="field_0_phase_dir",
    )
    partition = declare_field_partitions(geos)
    assert set(partition.training + partition.inner_holdout + partition.sealed_holdout) == set(
        range(1, 9)
    )
    assert len(partition.sealed_holdout) == 2
    assert len(partition.inner_holdout) == 2
    assert len(partition.training) == 4
    sealed = [item for item in geos if item.field_id in partition.sealed_holdout]
    dang = abs(
        (sealed[0].position_angle_rad - sealed[1].position_angle_rad + np.pi) % (2 * np.pi) - np.pi
    )
    assert abs(dang - np.pi) < np.deg2rad(5.0)
    field_id = np.repeat(np.arange(1, 9), 5)
    masks = field_partition_masks(field_id, partition)
    assert not np.any(masks["training"] & masks["inner_holdout"])
    assert not np.any(masks["training"] & masks["sealed_holdout"])
    assert not np.any(masks["inner_holdout"] & masks["sealed_holdout"])
    for field in range(1, 9):
        owned = (
            int(np.sum(masks["training"][field_id == field]))
            + int(np.sum(masks["inner_holdout"][field_id == field]))
            + int(np.sum(masks["sealed_holdout"][field_id == field]))
        )
        assert owned == 5


def test_source_relative_offsets_match_commanded_pointing() -> None:
    sky = np.array([[0.01, -0.02], [0.03, 0.00]], dtype=np.float64)
    delta = np.array([[0.001, 0.0], [0.0, -0.004]], dtype=np.float64)
    got = source_relative_offsets(sky, delta)
    for i in range(2):
        expected = source_relative_lm_rad(sky[i : i + 1], delta[i : i + 1]).reshape(2)
        assert np.allclose(got[i], expected)


def test_geometric_fringe_matches_casa_and_is_unity_on_axis() -> None:
    uvw = np.array([[120.0, -40.0, 8.0], [30.0, 90.0, -2.0]], dtype=np.float64)
    freq = np.array([4.564e9], dtype=np.float64)
    on_axis = geometric_fringe_phase(uvw, freq, np.array([0.0, 0.0]))
    assert np.allclose(on_axis, 1.0)
    lm = np.array([0.001, -0.0005], dtype=np.float64)
    got = geometric_fringe_phase(uvw, freq, lm)
    for i in range(2):
        assert np.allclose(got[i, 0], geometric_visibility_phase(uvw[i : i + 1], float(freq[0]), *lm))
    vis = np.ones((2, 1, 2, 2), dtype=np.complex128)
    phased = apply_geometric_fringe(vis, uvw, freq, lm)
    assert np.allclose(phased[:, 0, 0, 0], got[:, 0])
    assert "for row" not in inspect.getsource(geometric_fringe_phase)


def test_channel_holdout_is_dropped_before_fits() -> None:
    cube = np.arange(8, dtype=np.float64).reshape(2, 4)
    train = np.array([True, True, True, False])
    kept = apply_channel_holdout(cube, train)
    assert kept.shape == (2, 3)
    assert np.array_equal(kept[0], [0.0, 1.0, 2.0])
    with pytest.raises(ValueError, match="channel holdout"):
        apply_channel_holdout(cube, np.array([True, False]))


def test_manufactured_offset_field_closes_only_with_fringe() -> None:
    identity = np.eye(2, dtype=np.complex128)
    sky = np.array([[8.0, 0.0], [0.0, 8.0]], dtype=np.complex128)
    uvw = np.array([[1800.0, -400.0, 30.0], [900.0, 1200.0, -15.0]], dtype=np.float64)
    freq = np.array([4.564e9], dtype=np.float64)
    lm = np.array([0.001054, 0.0], dtype=np.float64)
    jones = np.stack([identity, identity])
    bare = predict_dual_antenna_numpy(jones, jones, sky, jones, jones)
    manufactured = apply_geometric_fringe(bare, uvw, freq, lm)
    recovered = predict_offset_field_visibilities(
        jones,
        jones,
        sky,
        jones,
        jones,
        uvw_m=uvw,
        frequency_hz=freq,
        sky_lm_rad=lm,
    )
    weight = np.ones_like(manufactured, dtype=np.float64)
    missing = diagonal_rr_ll_closure(manufactured, bare, weight)
    closed = diagonal_rr_ll_closure(manufactured, recovered, weight)
    assert missing["rr"]["relative_power"] > 0.5
    assert closed["rr"]["relative_power"] < 1.0e-12
    assert closed["ll"]["relative_power"] < 1.0e-12
    on_axis = apply_geometric_fringe(bare, uvw, freq, np.array([0.0, 0.0]))
    assert np.allclose(on_axis, bare)


def test_ns_ew_masks_follow_reconstructed_azimuth() -> None:
    offset = np.array(
        [[0.0, 0.001], [0.001, 0.0], [0.0, -0.001], [-0.001, 0.0]],
        dtype=np.float64,
    )
    masks = ns_ew_masks(offset)
    assert np.array_equal(masks["north_south"], [True, False, True, False])
    assert np.array_equal(masks["east_west"], [False, True, False, True])
    transforms = template_geometry_transforms(np.array([[1.0, 0.2j], [0.1, 1.0]]))
    assert transforms["conjugate"][0, 1] == -0.2j
    assert transforms["swap_rl_lr"][0, 1] == 0.1


def test_directional_diagnosis_does_not_select_a_model() -> None:
    n = 40
    offset = np.repeat(
        np.array([[0.001, 0.0], [-0.001, 0.0], [0.0, 0.001], [0.0, -0.001]], dtype=np.float64),
        n // 4,
        axis=0,
    )
    diag = np.zeros((n, 2, 2), dtype=np.complex128)
    diag[:, 0, 0] = 8.0
    diag[:, 1, 1] = 8.0
    increment = np.zeros((n, 2, 2), dtype=np.complex128)
    increment[:, 0, 1] = 0.03 + 0.01j
    increment[:, 1, 0] = 0.025 - 0.008j
    measured = diag + increment
    weight = np.ones_like(measured, dtype=np.float64)
    report = diagnose_directional_disagreement(measured, diag + increment, diag, weight, offset)
    assert report["model_selected"] is False
    assert report["convention_search_reopened"] is False
    assert report["native_by_axis"]["east_west"]["rl_correlation"] > 0.9
    assert "1" in report["azimuthal_phasers"]


def test_dual_antenna_predict_is_vectorized_rime() -> None:
    r_p = np.array([[1.1, 0.01], [0.0, 0.9]], dtype=np.complex128)
    r_q = np.array([[0.95, 0.0], [0.02, 1.05]], dtype=np.complex128)
    e_p = np.array([[0.8, 0.05], [0.04, 0.7]], dtype=np.complex128)
    e_q = np.array([[0.75, 0.03], [0.02, 0.72]], dtype=np.complex128)
    sky = np.array([[8.0, 0.1], [0.1, 8.0]], dtype=np.complex128)
    pred = predict_dual_antenna_numpy(
        np.stack([r_p, r_p]),
        np.stack([e_p, e_p]),
        sky,
        np.stack([e_q, e_q]),
        np.stack([r_q, r_q]),
    )
    manual = r_p @ e_p @ sky @ e_q.conj().T @ r_q.conj().T
    assert pred.shape == (2, 1, 2, 2)
    assert np.allclose(pred[0, 0], manual)
    source = inspect.getsource(predict_dual_antenna_numpy)
    assert "for " not in source
    assert "for row" not in source


def test_locked_convention_is_default_and_ladder_is_refused() -> None:
    assert locked_convention() == DEFAULT_CONVENTION
    refuse_convention_search(None)
    refuse_convention_search((DEFAULT_CONVENTION,))
    with pytest.raises(ValueError, match="does not search"):
        refuse_convention_search(convention_ladder())


def test_clustered_upper_limit_is_real_and_refuses_placeholder() -> None:
    moments = _moments(0.0 + 0.0j)
    limit = clustered_null_upper_limit(moments, template_median_abs=0.04, n_perm=80, seed=3)
    refuse_placeholder_upper_limit(limit)
    assert limit.status == "computed"
    assert limit.coherent_voltage_ul95 > 0.0
    assert limit.coherent_voltage_ul95 != 8.0 or limit.n_clusters >= 8
    assert limit.n_clusters >= 8
    with pytest.raises(ValueError, match="placeholder|insufficient"):
        clustered_null_upper_limit([], template_median_abs=0.04)
    with pytest.raises(ValueError, match="not computed"):
        refuse_placeholder_upper_limit({"status": "placeholder", "coherent_voltage_ul95": 8.0})


def test_classifier_records_scientific_null_and_keeps_spw5_closed() -> None:
    limit = clustered_null_upper_limit(_moments(0.0j), template_median_abs=0.04, n_perm=60)
    gate = classify_c147_offset_ring(
        software_ok=True,
        split_clean=True,
        smoke_ok=True,
        rr_ll_regression=False,
        inner_rl_improves=False,
        inner_lr_improves=False,
        sealed_rl_improves=False,
        sealed_lr_improves=False,
        scale_ok=True,
        antenna_ok=True,
        unit_better_than_diag=False,
        scaled_better_than_unit=False,
        upper_limit=limit,
    )
    assert gate["decision"] == "no_model_selected"
    assert gate["scientific_decision"] is True
    assert gate["process_failure"] is False
    assert gate["full_jones_frozen"] is False
    assert gate["spw5_closed"] is True
    assert gate["most_important_next_artifact"] == "cassbeam_diagonal_low_order_correction"
    failed = classify_c147_offset_ring(
        software_ok=False,
        split_clean=True,
        smoke_ok=True,
        rr_ll_regression=False,
        inner_rl_improves=True,
        inner_lr_improves=True,
        sealed_rl_improves=True,
        sealed_lr_improves=True,
        scale_ok=True,
        antenna_ok=True,
        unit_better_than_diag=True,
        scaled_better_than_unit=False,
        upper_limit=None,
    )
    assert failed["decision"] == "software_gate_failed"
    assert failed["process_failure"] is True


def test_selected_model_requires_heldout_crosshands_and_no_rr_ll_regression() -> None:
    limit = clustered_null_upper_limit(_moments(0.0j), template_median_abs=0.04, n_perm=40)
    gate = classify_c147_offset_ring(
        software_ok=True,
        split_clean=True,
        smoke_ok=True,
        rr_ll_regression=True,
        inner_rl_improves=True,
        inner_lr_improves=True,
        sealed_rl_improves=True,
        sealed_lr_improves=True,
        scale_ok=True,
        antenna_ok=True,
        unit_better_than_diag=True,
        scaled_better_than_unit=False,
        upper_limit=limit,
    )
    assert gate["selected_model"] is None
    selected = classify_c147_offset_ring(
        software_ok=True,
        split_clean=True,
        smoke_ok=True,
        rr_ll_regression=False,
        inner_rl_improves=True,
        inner_lr_improves=True,
        sealed_rl_improves=True,
        sealed_lr_improves=True,
        scale_ok=True,
        antenna_ok=True,
        unit_better_than_diag=True,
        scaled_better_than_unit=False,
        upper_limit=None,
    )
    assert selected["selected_model"] == "full_jones_unit"
    assert selected["spw5_closed"] is False


def test_smoke_gate_blocks_full_spw4() -> None:
    blocked = channel32_smoke_gates(
        geometry_ok=True,
        calibration_hashed=True,
        predictions_finite=True,
        diagonal_closure_ok=False,
    )
    assert blocked["passed"] is False
    assert blocked["continue_to_spw4"] is False
    assert blocked["channel"] == 32
    ok = channel32_smoke_gates(
        geometry_ok=True,
        calibration_hashed=True,
        predictions_finite=True,
        diagonal_closure_ok=True,
    )
    assert ok["continue_to_spw4"] is True


def test_scale_and_antenna_leverage_helpers() -> None:
    assert scale_compatible([1.0 + 0.05j, 0.9 - 0.04j])["compatible"] is True
    assert scale_compatible([1.0 + 0.0j, -1.0 + 0.0j])["compatible"] is False
    moments = _moments(0.2 + 0.0j)
    share = antenna_power_share(moments)
    assert share["n_antennas"] == 2
    assert select_training_qu({(0.0, 0.0): 2.0, (0.001, 0.0): 1.0}) == (0.001, 0.0)
    payload = upper_limit_to_dict(
        clustered_null_upper_limit(moments, template_median_abs=0.04, n_perm=40)
    )
    assert payload["status"] == "computed"
