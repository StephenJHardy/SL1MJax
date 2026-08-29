from __future__ import annotations

import numpy as np
import pytest

from sl1mjax.beam_conventions import (
    CASSBEAM_UNPINNED_REQUIREMENTS,
    HOLOGRAPHY_UNPINNED_REQUIREMENTS,
    JONES_RECEPTOR_ORDER,
    BeamCalibrationState,
)
from sl1mjax.full_jones import (
    FULL_JONES_PIN_CATALOG_VERSION,
    AntennaAveraging,
    DirectionAxisOrientation,
    FullJonesContents,
    FullJonesOuterFieldPolicy,
    FullJonesReferencePin,
    FullJonesVoltageBeam,
    TermPresence,
    TransmitReceiveConvention,
    apply_full_jones_outer_field,
    full_jones_reference_is_frozen,
    load_full_jones_acquisition_plan,
    orientation_oracle_sample_spec,
    refuse_analytic_squint_composition,
    refuse_on_axis_double_count,
    require_frozen_full_jones_reference,
    unfrozen_full_jones_pin,
)
from sl1mjax.voltage_beam import beam_coordinates


def _identity_and_leakage() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    full = np.zeros((1, 2, 1, 2, 2), dtype=np.complex128)
    full[0, 0, 0] = np.array([[1.0, 0.2], [0.3, 1.0]])
    full[0, 1, 0] = np.array([[0.4, 0.5], [0.6, 0.7]])
    full_valid = np.array([[[True], [False]]])
    scalar = np.zeros_like(full)
    scalar[..., 0, 0] = 0.8
    scalar[..., 1, 1] = 0.9
    scalar_valid = np.array([[[True], [True]]])
    return full, full_valid, scalar, scalar_valid


def _frozen_pin(**overrides: object) -> FullJonesReferencePin:
    values: dict[str, object] = {
        "artifact_id": "cassbeam_go",
        "generator_or_path": "pinned-not-real",
        "native_quantity": "voltage_jones_2x2",
        "native_basis": "circular",
        "receptor_order": JONES_RECEPTOR_ORDER,
        "transmit_receive": TransmitReceiveConvention.RECEIVE,
        "direction_axis_orientation": DirectionAxisOrientation.L_EAST_M_NORTH,
        "frequency_support_hz": (4.052e9, 7.948e9),
        "direction_support": "validated main lobe",
        "antenna_averaging": AntennaAveraging.ARRAY_AVERAGE,
        "contents": FullJonesContents(
            squint=TermPresence.PRESENT,
            off_diagonal_leakage=TermPresence.PRESENT,
            on_axis_g=TermPresence.ABSENT,
            on_axis_d=TermPresence.ABSENT,
            on_axis_x=TermPresence.ABSENT,
            on_axis_p=TermPresence.ABSENT,
        ),
        "outer_field_policy": FullJonesOuterFieldPolicy.TAPERED_SCALAR_COMPOSITE,
        "generator_version": "unspecified",
        "input_checksum": "sha256:0",
        "output_checksum": "sha256:1",
        "frozen": True,
        "unpinned_fields": (),
    }
    values.update(overrides)
    return FullJonesReferencePin(**values)  # type: ignore[arg-type]


def test_phase_6a_selects_routes_and_stays_unfrozen() -> None:
    plan = load_full_jones_acquisition_plan()
    assert plan.catalog_version == FULL_JONES_PIN_CATALOG_VERSION
    assert plan.frozen is False
    assert full_jones_reference_is_frozen() is False
    assert plan.convention_oracle_route == "cassbeam_go"
    assert plan.scientific_table_route == "perley2016_cband_holography_grids"
    assert plan.scientific_fallback_route == "cassbeam_go"
    assert "iheanetu2019_lband" in plan.refused_as_cband
    assert "jagannathan2021_atoz_plumber" in plan.refused_as_cband
    cassbeam = unfrozen_full_jones_pin("cassbeam_go")
    holography = unfrozen_full_jones_pin("perley2016_cband_holography_grids")
    assert cassbeam.frozen is False
    assert cassbeam.unpinned_fields == CASSBEAM_UNPINNED_REQUIREMENTS
    assert holography.unpinned_fields == HOLOGRAPHY_UNPINNED_REQUIREMENTS
    assert cassbeam.outer_field_policy is (
        FullJonesOuterFieldPolicy.TAPERED_SCALAR_COMPOSITE
    )
    assert plan.bagofwinds_compact_products_found is False
    assert plan.cassbeam_generic_template_is_cband is False
    assert plan.cassbeam_runtime_in_imager is False
    assert plan.cassbeam_host_package == "1.1-4build2"
    assert plan.holography_request_frequencies_mhz == (4564, 4692)


def test_phase_6a_refuses_scalar_and_wrong_band_promotion() -> None:
    with pytest.raises(ValueError, match="not a C-band"):
        unfrozen_full_jones_pin("iheanetu2019_lband")
    with pytest.raises(ValueError, match="not a C-band"):
        unfrozen_full_jones_pin("jagannathan2021_atoz_plumber")
    with pytest.raises(ValueError, match="already a frozen scalar"):
        unfrozen_full_jones_pin("perley2016_cband_stokes_i")
    with pytest.raises(ValueError, match="already a frozen scalar"):
        unfrozen_full_jones_pin("sl1mjax_airy")
    with pytest.raises(ValueError, match="cannot be promoted"):
        unfrozen_full_jones_pin("evla195_diagonal_squint")


def test_orientation_oracle_requests_correlation_aware_samples() -> None:
    spec = orientation_oracle_sample_spec()
    assert spec.frequencies_hz == (4.564e9, 4.692e9)
    assert spec.parallactic_angles_rad == (0.0, 0.5 * np.pi)
    assert spec.correlations == ("RR", "RL", "LR", "LL")
    assert "R/L sign" in spec.required_closures
    assert "feed-frame position angle" in spec.required_closures
    assert "squint separation" in spec.required_closures


def test_full_jones_backend_refuses_unfrozen_evaluation() -> None:
    pin = unfrozen_full_jones_pin("cassbeam_go")
    beam = FullJonesVoltageBeam(pin=pin)
    coordinates = beam_coordinates(0.0, 0.0, 4.6e9)
    with pytest.raises(ValueError, match="not frozen"):
        beam.evaluate(coordinates, calibration_state="casa_parang_true")
    with pytest.raises(ValueError, match="not frozen"):
        require_frozen_full_jones_reference(pin)
    complete = _frozen_pin()
    assert complete.missing_freeze_fields() == ()
    with pytest.raises(ValueError, match="catalog is not frozen"):
        require_frozen_full_jones_reference(complete)


def test_composition_refuses_double_counted_squint_and_on_axis_jones() -> None:
    unknown = FullJonesContents()
    with pytest.raises(ValueError, match="replace analytic squint"):
        refuse_analytic_squint_composition(unknown)
    with pytest.raises(ValueError, match="replace analytic squint"):
        refuse_analytic_squint_composition(
            FullJonesContents(squint=TermPresence.PRESENT)
        )
    refuse_analytic_squint_composition(FullJonesContents(squint=TermPresence.ABSENT))
    leaking = FullJonesContents(
        squint=TermPresence.ABSENT,
        on_axis_d=TermPresence.PRESENT,
        on_axis_x=TermPresence.ABSENT,
        on_axis_p=TermPresence.UNKNOWN,
        on_axis_g=TermPresence.ABSENT,
    )
    with pytest.raises(ValueError, match="E\\(0\\)=I"):
        refuse_on_axis_double_count(leaking, BeamCalibrationState.CASA_PARANG_TRUE)
    refuse_on_axis_double_count(leaking, BeamCalibrationState.UNCALIBRATED)
    clean = FullJonesContents(
        squint=TermPresence.PRESENT,
        off_diagonal_leakage=TermPresence.PRESENT,
        on_axis_g=TermPresence.ABSENT,
        on_axis_d=TermPresence.ABSENT,
        on_axis_x=TermPresence.ABSENT,
        on_axis_p=TermPresence.ABSENT,
    )
    refuse_on_axis_double_count(clean, "casa_parang_true")


def test_outer_field_taper_returns_to_scalar_and_drops_off_diagonals() -> None:
    full, full_valid, scalar, scalar_valid = _identity_and_leakage()
    with pytest.raises(ValueError, match="hard-splice"):
        apply_full_jones_outer_field(
            full,
            full_valid,
            scalar,
            scalar_valid,
            policy=FullJonesOuterFieldPolicy.HARD_SPLICE,
        )
    closed = apply_full_jones_outer_field(
        full,
        full_valid,
        scalar,
        scalar_valid,
        policy=FullJonesOuterFieldPolicy.UNSUPPORTED,
    )
    assert bool(closed.valid[0, 0, 0])
    assert not bool(closed.valid[0, 1, 0])
    assert closed.jones[0, 1, 0].tolist() == [[0.0, 0.0], [0.0, 0.0]]
    composed = apply_full_jones_outer_field(
        full,
        full_valid,
        scalar,
        scalar_valid,
        policy=FullJonesOuterFieldPolicy.TAPERED_SCALAR_COMPOSITE,
    )
    assert composed.jones[0, 0, 0].tolist() == full[0, 0, 0].tolist()
    assert composed.jones[0, 1, 0, 0, 0] == pytest.approx(0.8)
    assert composed.jones[0, 1, 0, 1, 1] == pytest.approx(0.9)
    assert composed.jones[0, 1, 0, 0, 1] == pytest.approx(0.0)
    assert composed.jones[0, 1, 0, 1, 0] == pytest.approx(0.0)
    assert bool(composed.off_diagonal_valid[0, 0, 0])
    assert not bool(composed.off_diagonal_valid[0, 1, 0])
    assert bool(composed.valid[0, 1, 0])
    half = apply_full_jones_outer_field(
        full,
        full_valid,
        scalar,
        scalar_valid,
        policy=FullJonesOuterFieldPolicy.TAPERED_SCALAR_COMPOSITE,
        taper_weight=np.array([[[0.25], [0.0]]]),
    )
    assert half.jones[0, 0, 0, 0, 0] == pytest.approx(0.25 * 1.0 + 0.75 * 0.8)
    assert half.jones[0, 0, 0, 0, 1] == pytest.approx(0.25 * 0.2)
