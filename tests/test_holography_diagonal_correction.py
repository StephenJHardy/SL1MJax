from __future__ import annotations

import inspect

import numpy as np
import pytest

from sl1mjax.data.canonical import VisibilityBlock
from sl1mjax.holography import (
    THOL0001_SPW4_CHANNEL_32_HZ,
    AntennaPointingRole,
    HolographyObservation,
    ResolvedAntennaPointing,
)
from sl1mjax.holography_calibration import C147_OFFSET_FIELD_IDS, HELD_OUT_REFERENCE_ANTENNA
from sl1mjax.holography_diagonal import THOL0001_REFERENCE_ANTENNA_NAMES
from sl1mjax.holography_diagonal_correction import (
    FIRST_LADDER_TERMS,
    HOLORASTER_FIELD_ID,
    MIN_MOVERS_IMPROVING,
    SPW4_CORRECTION_PROTOCOL_VERSION,
    SPW4_HOLDOUT_MOVING_ANTENNA_NAMES,
    SPW4_HOLDOUT_REFERENCE_ANTENNA_NAMES,
    SPW4_TRAINING_FREQUENCY_HZ,
    SPW5_FREQUENCY_HZ,
    THOL0001_MOVING_ANTENNA_NAMES,
    MoverPairedDeltas,
    PairedLossDifference,
    bootstrap_paired_delta,
    complex_visibility_loss,
    copolar_loss_parts,
    frozen_protocol_payload,
    mover_paired_deltas,
    protocol_as_mapping,
    reduce_cluster_parts,
    refuse_c147_training,
    refuse_magnitude_loss,
    refuse_phase_in_first_ladder,
    refuse_spw5,
    refuse_visibility_bootstrap,
    require_boresight_unity,
    require_disjoint_antenna_cover,
    score_paired_holdout,
    select_nested_correction,
    spatial_holdout_from_offsets,
    spw4_correction_holdouts,
)
from sl1mjax.polarization import Correlation, ReceptorBasis

_PHASE = (1.4660765716752369, 0.8726646259971648)
_CORR = (Correlation.RR, Correlation.RL, Correlation.LR, Correlation.LL)


def _antenna_names() -> tuple[str, ...]:
    names = [""] * 29
    for index in range(1, 29):
        names[index] = f"ea{index:02d}"
    return tuple(names)


def _arcmin(l_arcmin: float, m_arcmin: float) -> tuple[float, float]:
    return (np.deg2rad(l_arcmin / 60.0), np.deg2rad(m_arcmin / 60.0))


def _toy_observation() -> HolographyObservation:
    """Four antennas, two raster cells, one unused C147 row."""

    antenna1 = np.array([4, 8, 4, 8, 4, 4], dtype=np.int32)
    antenna2 = np.array([2, 2, 2, 2, 24, 2], dtype=np.int32)
    time_s = np.array([0.0, 0.0, 10.0, 10.0, 20.0, 30.0], dtype=np.float64)
    field_id = np.array([10, 10, 10, 10, 10, 3], dtype=np.int32)
    n_row = time_s.size
    block = VisibilityBlock(
        uvw_m=np.broadcast_to(np.array([30.0, -12.0, 8.0]), (n_row, 3)).copy(),
        frequency_hz=np.array([THOL0001_SPW4_CHANNEL_32_HZ]),
        visibility=np.zeros((n_row, 1, 4), dtype=np.complex128),
        weight=np.ones((n_row, 1, 4), dtype=np.float64),
        flag=np.zeros((n_row, 1, 4), dtype=bool),
        time_s=time_s,
        antenna1=antenna1,
        antenna2=antenna2,
        correlations=_CORR,
        receptor_basis=ReceptorBasis.CIRCULAR,
        phase_centre_rad=_PHASE,
        field_id=field_id,
        spectral_window_id=4,
    )
    unique_time = np.array([0.0, 10.0, 20.0, 30.0], dtype=np.float64)
    n_time = unique_time.size
    n_ant = 25
    offsets = np.zeros((n_time, n_ant, 2), dtype=np.float64)
    role = np.full((n_time, n_ant), AntennaPointingRole.UNKNOWN.value, dtype="U16")
    for time_index in range(n_time):
        role[time_index, 2] = AntennaPointingRole.REFERENCE.value
        role[time_index, 24] = AntennaPointingRole.REFERENCE.value
        role[time_index, 4] = AntennaPointingRole.MOVING.value
        role[time_index, 8] = AntennaPointingRole.MOVING.value
    offsets[0, 4] = _arcmin(10.0, 0.0)
    offsets[0, 8] = _arcmin(10.0, 0.0)
    offsets[1, 4] = _arcmin(20.0, 0.0)
    offsets[1, 8] = _arcmin(20.0, 0.0)
    offsets[2, 4] = _arcmin(10.0, 0.0)
    offsets[3, 4] = _arcmin(10.0, 0.0)
    pointing = ResolvedAntennaPointing(
        unique_time_s=unique_time,
        antenna_id=np.arange(n_ant, dtype=np.int32),
        offset_lm_rad=offsets,
        valid=np.ones((n_time, n_ant), dtype=bool),
        settled=np.ones((n_time, n_ant), dtype=bool),
        role=role,
        selected_column="DIRECTION",
        offset_sign="commanded_pointing",
        join_rule="interval",
        join_tolerance_s=0.0,
        measure_ref="AZELGEO",
        units="rad",
    )
    positions = np.zeros((n_ant, 3), dtype=np.float64)
    positions[:, 0] = np.linspace(-1.6e6, -1.5e6, n_ant)
    positions[:, 1] = -5.04e6
    positions[:, 2] = 3.55e6
    return HolographyObservation(
        block=block,
        pointing=pointing,
        antenna_position_m=positions,
        calibration_state="casa_parang_true",
        phase_centre_rad=_PHASE,
        stokes_i=8.0,
    )


def _delta(
    axis: str,
    point: float,
    lo: float,
    hi: float,
    *,
    kind: str,
    n: int = 20,
) -> PairedLossDifference:
    return PairedLossDifference(
        axis=axis,
        delta=point,
        delta_lo=lo,
        delta_hi=hi,
        n_clusters=n,
        n_boot=400,
        cluster_kind=kind,
    )


def _movers(
    delta: np.ndarray,
    mainlobe_delta: np.ndarray | None = None,
    mainlobe_baseline: np.ndarray | None = None,
) -> MoverPairedDeltas:
    values = np.asarray(delta, dtype=np.float64)
    return MoverPairedDeltas(
        names=SPW4_HOLDOUT_MOVING_ANTENNA_NAMES,
        delta=values,
        mainlobe_delta=np.zeros(5) if mainlobe_delta is None else mainlobe_delta,
        mainlobe_baseline=(
            np.full(5, 0.01) if mainlobe_baseline is None else mainlobe_baseline
        ),
    )


def test_frozen_partition_covers_published_antennas() -> None:
    require_disjoint_antenna_cover()
    assert SPW4_HOLDOUT_MOVING_ANTENNA_NAMES == THOL0001_MOVING_ANTENNA_NAMES[3::4]
    assert SPW4_HOLDOUT_REFERENCE_ANTENNA_NAMES == THOL0001_REFERENCE_ANTENNA_NAMES[5:]
    assert HELD_OUT_REFERENCE_ANTENNA in SPW4_HOLDOUT_REFERENCE_ANTENNA_NAMES
    payload = frozen_protocol_payload()
    assert payload["protocol_version"] == SPW4_CORRECTION_PROTOCOL_VERSION
    assert payload["training_frequency_hz"] == SPW4_TRAINING_FREQUENCY_HZ
    assert payload["spw5_status"] == "sealed"
    assert payload["phase_in_first_ladder"] is False
    assert payload["boresight"] == "C_R(0)=C_L(0)=1"
    assert payload["selection_rule"] == "paired_delta_L_ci95_below_zero"
    assert payload["visibility_bootstrap"] is False
    assert payload["min_movers_improving"] == MIN_MOVERS_IMPROVING
    assert payload["first_ladder"] == list(FIRST_LADDER_TERMS)
    protocol_as_mapping(payload)


def test_first_ladder_refuses_phase_spw5_and_flux_gauge() -> None:
    refuse_phase_in_first_ladder("rl_squint_scale")
    with pytest.raises(RuntimeError, match="amplitude-only"):
        refuse_phase_in_first_ladder("outer_phase")
    with pytest.raises(RuntimeError, match="amplitude-only"):
        refuse_phase_in_first_ladder("phase_offset")
    refuse_spw5(frequency_hz=SPW4_TRAINING_FREQUENCY_HZ)
    with pytest.raises(RuntimeError, match="SPW 5"):
        refuse_spw5(opened=True)
    with pytest.raises(RuntimeError, match="SPW 5"):
        refuse_spw5(frequency_hz=SPW5_FREQUENCY_HZ)
    require_boresight_unity(1.0, 1.0)
    with pytest.raises(ValueError, match="flux gauge"):
        require_boresight_unity(1.02, 1.0)
    refuse_c147_training([HOLORASTER_FIELD_ID, 0, 9])
    with pytest.raises(RuntimeError, match="C147"):
        refuse_c147_training(C147_OFFSET_FIELD_IDS[:1])


def test_spatial_checkerboard_is_row_order_independent() -> None:
    first = np.array([_arcmin(10.0, 0.0), _arcmin(20.0, 0.0)])
    second = np.array([_arcmin(20.0, 0.0), _arcmin(10.0, 0.0)])
    hold_first = spatial_holdout_from_offsets(first)
    hold_second = spatial_holdout_from_offsets(second)
    np.testing.assert_array_equal(hold_first, [False, True])
    np.testing.assert_array_equal(hold_second, [True, False])


def test_frozen_masks_isolate_training_spatial_and_mover_holdouts() -> None:
    holdouts = spw4_correction_holdouts(_toy_observation(), _antenna_names())
    np.testing.assert_array_equal(holdouts.train, [True, False, False, False, False, False])
    np.testing.assert_array_equal(
        holdouts.spatial_holdout, [False, False, True, False, False, False]
    )
    np.testing.assert_array_equal(holdouts.mover_holdout, [False, True, False, False, False, False])
    np.testing.assert_array_equal(
        holdouts.reference_holdout, [False, False, False, False, True, False]
    )
    np.testing.assert_array_equal(
        holdouts.spatial_mover_holdout, [False, False, False, True, False, False]
    )
    np.testing.assert_array_equal(holdouts.unused_c147, [False, False, False, False, False, True])
    assert not np.any(holdouts.train & holdouts.unused_c147)


def test_selection_uses_paired_delta_and_per_mover_gates() -> None:
    better_spatial = _delta("spatial", -0.02, -0.03, -0.01, kind="spatial_cell")
    overlapping_spatial = _delta("spatial", -0.01, -0.03, 0.005, kind="spatial_cell")
    better_mover = _delta("moving", -0.015, -0.02, -0.008, kind="moving_antenna", n=5)
    worse_mover = _delta("moving", 0.01, -0.002, 0.02, kind="moving_antenna", n=5)
    better_ref = _delta("reference", -0.04, -0.05, -0.03, kind="reference_antenna", n=2)
    four_of_five = _movers([-0.02, -0.01, -0.015, -0.008, 0.002])
    three_of_five = _movers([-0.02, -0.01, -0.015, 0.004, 0.003])
    mainlobe_hit = _movers(
        [-0.02, -0.01, -0.015, -0.008, -0.004],
        mainlobe_delta=np.array([0.0, 0.0, 0.05, 0.0, 0.0]),
        mainlobe_baseline=np.array([0.01, 0.01, 0.01, 0.01, 0.01]),
    )
    assert better_spatial.improves()
    assert not overlapping_spatial.improves()
    assert four_of_five.passes()
    assert not three_of_five.passes()
    assert not mainlobe_hit.passes()
    missing = _movers([-0.02, -0.01, np.nan, -0.008, -0.004])
    assert not missing.passes()
    assert (
        select_nested_correction(better_spatial, better_mover, missing) == "keep_baseline"
    )
    assert (
        select_nested_correction(better_spatial, better_mover, four_of_five) == "accept_candidate"
    )
    assert select_nested_correction(better_spatial, worse_mover, four_of_five) == "keep_baseline"
    assert (
        select_nested_correction(overlapping_spatial, better_mover, four_of_five)
        == "keep_baseline"
    )
    assert select_nested_correction(better_spatial, better_mover, three_of_five) == "keep_baseline"
    assert (
        select_nested_correction(
            overlapping_spatial,
            worse_mover,
            four_of_five,
            reference=better_ref,
        )
        == "keep_baseline"
    )


def test_paired_delta_keeps_shared_cluster_covariance() -> None:
    rng = np.random.default_rng(4)
    n_cluster = 40
    shared = 0.08 + 0.04 * rng.standard_normal(n_cluster)
    baseline = np.maximum(shared, 1.0e-3)
    candidate = np.maximum(baseline - 0.012 + 0.001 * rng.standard_normal(n_cluster), 1.0e-4)
    denom = np.ones(n_cluster)
    unpaired_hi = float(np.quantile(candidate / denom, 0.975))
    unpaired_point_baseline = float(np.sum(baseline) / np.sum(denom))
    paired = bootstrap_paired_delta(
        candidate,
        denom,
        baseline,
        denom,
        axis="spatial",
        cluster_kind="spatial_cell",
        n_boot=400,
        seed=4,
    )
    assert unpaired_hi > unpaired_point_baseline
    assert paired.improves()
    assert paired.delta_hi < 0.0


def test_complex_loss_is_not_a_magnitude_ratio() -> None:
    measured = np.zeros((3, 2, 2), dtype=np.complex128)
    predicted = np.zeros((3, 2, 2), dtype=np.complex128)
    measured[:, 0, 0] = [1.0 + 0.2j, 2.0 - 0.1j, 0.5]
    predicted[:, 0, 0] = measured[:, 0, 0]
    measured[:, 1, 1] = measured[:, 0, 0]
    predicted[:, 1, 1] = predicted[:, 0, 0]
    weight = np.ones((3, 2, 2))
    assert complex_visibility_loss(measured, predicted, weight) == pytest.approx(0.0)
    predicted[:, 0, 0] = np.abs(measured[:, 0, 0])
    predicted[:, 1, 1] = np.abs(measured[:, 1, 1])
    assert complex_visibility_loss(measured, predicted, weight) > 0.0
    with pytest.raises(RuntimeError, match="complex visibilities"):
        refuse_magnitude_loss("db_ratio")
    with pytest.raises(RuntimeError, match="individual visibilities"):
        refuse_visibility_bootstrap("visibility_row")
    with pytest.raises(ValueError, match="exactly one channel"):
        copolar_loss_parts(
            np.zeros((3, 2, 2, 2), dtype=np.complex128),
            np.zeros((3, 2, 2, 2), dtype=np.complex128),
            np.ones((3, 2, 2, 2)),
        )


def test_mover_paired_deltas_report_every_held_out_antenna() -> None:
    names = _antenna_names()
    n = 10
    measured = np.zeros((n, 2, 2), dtype=np.complex128)
    baseline = np.zeros((n, 2, 2), dtype=np.complex128)
    candidate = np.zeros((n, 2, 2), dtype=np.complex128)
    measured[:, 0, 0] = 1.0
    measured[:, 1, 1] = 1.0
    baseline[:, 0, 0] = 0.8
    baseline[:, 1, 1] = 0.8
    candidate[:, 0, 0] = 0.95
    candidate[:, 1, 1] = 0.95
    weight = np.ones((n, 2, 2))
    movers = np.repeat(
        np.array([8, 13, 18, 22, 28], dtype=np.int32),
        2,
    )
    report = mover_paired_deltas(
        measured, candidate, baseline, weight, movers, names
    )
    assert report.names == SPW4_HOLDOUT_MOVING_ANTENNA_NAMES
    assert report.n_improving == 5
    assert report.passes()


def test_cluster_reduction_does_not_treat_rows_as_units() -> None:
    numer = np.array([1.0, 1.0, 3.0, 3.0, 3.0])
    denom = np.ones(5)
    labels = np.array([10, 10, 11, 11, 11])
    ids, clustered_n, clustered_d = reduce_cluster_parts(numer, denom, labels)
    np.testing.assert_array_equal(ids, [10, 11])
    np.testing.assert_allclose(clustered_n, [2.0, 9.0])
    np.testing.assert_allclose(clustered_d, [2.0, 3.0])
    scored = score_paired_holdout(
        np.broadcast_to(np.eye(2, dtype=np.complex128), (5, 2, 2)).copy() * 1.0,
        np.broadcast_to(np.eye(2, dtype=np.complex128), (5, 2, 2)).copy() * 0.9,
        np.broadcast_to(np.eye(2, dtype=np.complex128), (5, 2, 2)).copy() * 0.5,
        np.ones((5, 2, 2)),
        labels,
        axis="spatial",
        cluster_kind="spatial_cell",
        n_boot=64,
        seed=1,
    )
    assert scored.n_clusters == 2
    assert scored.cluster_kind == "spatial_cell"


def test_holdout_builder_has_no_row_loop() -> None:
    source = inspect.getsource(spw4_correction_holdouts)
    assert "for row" not in source
    assert "for _row" not in source


def test_protocol_payload_cannot_open_phase_or_spw5() -> None:
    payload = frozen_protocol_payload()
    payload["spw5_status"] = "open"
    with pytest.raises(RuntimeError, match="SPW 5"):
        protocol_as_mapping(payload)
    payload = frozen_protocol_payload()
    payload["phase_in_first_ladder"] = True
    with pytest.raises(RuntimeError, match="phase"):
        protocol_as_mapping(payload)
    payload = frozen_protocol_payload()
    payload["selection_rule"] = "unpaired_candidate_interval"
    with pytest.raises(RuntimeError, match="paired"):
        protocol_as_mapping(payload)
    payload = frozen_protocol_payload()
    payload["visibility_bootstrap"] = True
    with pytest.raises(RuntimeError, match="visibilities"):
        protocol_as_mapping(payload)
