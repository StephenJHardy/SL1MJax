from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from sl1mjax.calibration import apply_calibration, identity_solution
from sl1mjax.calibration_terms import (
    CalibrationChain,
    CalibrationCoordinates,
    GainCurveTerm,
    OpacityTerm,
    airmass_from_elevation,
    compare_jones,
    elevation_rad,
    prior_baseline_jones,
    read_calibration_chain,
    write_calibration_chain,
)
from sl1mjax.data.canonical import VisibilityBlock
from sl1mjax.data.metadata import (
    AntennaRecord,
    CalibrationDeviceRecord,
    ObservationMetadata,
    SpectralWindowRecord,
    SwitchedPowerRecord,
    WeatherRecord,
)
from sl1mjax.polarization import Correlation, ReceptorBasis
from sl1mjax.vla_priors import (
    GainCurveCatalogEntry,
    estimate_vla_zenith_opacity,
    generate_vla_gain_curve,
    generate_vla_requantizer,
)

FIXTURE = Path(__file__).parent / "fixtures/vla_priors_srdp_golden.json"


def _coordinates() -> CalibrationCoordinates:
    time_s = np.asarray([60_000.0 * 86400.0])
    julian_date = time_s[0] / 86400.0 + 2_400_000.5
    gmst = np.deg2rad(
        np.mod(280.46061837 + 360.98564736629 * (julian_date - 2_451_545.0), 360)
    )
    return CalibrationCoordinates(
        time_s,
        np.asarray([6.0e9, 6.1e9]),
        3,
        (float(gmst), 0.0),
        np.asarray([[6_378_137.0, 0.0, 0.0], [6_378_137.0, 0.0, 0.0]]),
        2,
    )


def _metadata() -> ObservationMetadata:
    return ObservationMetadata(
        antennas=(
            AntennaRecord(0, "ea01", "W01", (6_378_137.0, 0.0, 0.0), 25.0, "ALT-AZ"),
            AntennaRecord(1, "ea02", "E01", (6_378_137.0, 0.0, 0.0), 25.0, "ALT-AZ"),
        ),
        spectral_windows=(
            SpectralWindowRecord(
                3,
                "C",
                6.05e9,
                (6.0e9, 6.1e9),
                (1.0e8, 1.0e8),
            ),
        ),
        weather=(
            WeatherRecord(
                60_000.0 * 86400.0,
                60.0,
                -1,
                temperature_k=285.0,
                dew_point_k=275.0,
            ),
        ),
        switched_power=(
            SwitchedPowerRecord(60_000.0 * 86400.0, 20.0, 0, 0, 3, (), (), (0.5, 0.6)),
            SwitchedPowerRecord(60_000.0 * 86400.0, 20.0, 1, 0, 3, (), (), (0.7, 0.8)),
        ),
        calibration_devices=(
            CalibrationDeviceRecord(
                60_000.0 * 86400.0,
                60.0,
                0,
                0,
                3,
                (1.0, 1.0),
            ),
        ),
    )


def test_elevation_airmass_and_analytic_prior_terms() -> None:
    coordinates = _coordinates()
    np.testing.assert_allclose(elevation_rad(coordinates), np.pi / 2, atol=2e-8)
    np.testing.assert_allclose(
        airmass_from_elevation(elevation_rad(coordinates)), 1.0, atol=2e-8
    )
    coefficients = np.zeros((2, 1, 2, 4))
    coefficients[..., 0] = 1.1
    gain_curve = GainCurveTerm(coefficients, np.asarray([3]), np.ones((2, 1, 2), bool))
    opacity = OpacityTerm(
        np.full((2, 1, 2), 0.02), np.asarray([3]), np.ones((2, 1, 2), bool)
    )

    gc_jones, gc_valid = gain_curve.evaluate(coordinates)
    opacity_jones, opacity_valid = opacity.evaluate(coordinates)

    np.testing.assert_allclose(gc_jones, 1.1)
    np.testing.assert_allclose(opacity_jones, np.exp(-0.01), rtol=1e-8)
    assert np.all(gc_valid & opacity_valid)
    assert np.all(np.diff(np.exp(-0.5 * 0.02 * np.asarray([1.0, 2.0]))) < 0)


def test_requantizer_validity_and_chain_composition() -> None:
    coordinates = _coordinates()
    metadata = _metadata()
    rq = generate_vla_requantizer(metadata, chunk_size=1)
    coefficients = np.zeros((2, 1, 2, 4))
    coefficients[..., 0] = 2.0
    gain_curve = GainCurveTerm(coefficients, np.asarray([3]), np.ones((2, 1, 2), bool))
    chain = CalibrationChain(
        (gain_curve, rq), coordinates.antenna_position_m, {"test": True}
    )

    value, valid = chain.evaluate(coordinates)
    np.testing.assert_allclose(
        value[:, :, 0, :], np.broadcast_to([1.0, 1.2], (1, 2, 2))
    )
    np.testing.assert_allclose(
        value[:, :, 1, :], np.broadcast_to([1.4, 1.6], (1, 2, 2))
    )
    assert np.all(valid)
    baseline, baseline_valid = prior_baseline_jones(
        chain, coordinates, np.asarray([0]), np.asarray([1])
    )
    np.testing.assert_allclose(
        baseline, np.broadcast_to([1.4, 1.92], (1, 2, 2))
    )
    assert np.all(baseline_valid)

    expired = replace(rq, interval_s=np.full_like(rq.interval_s, 0.5))
    shifted = replace(coordinates, time_s=coordinates.time_s + 1)
    _, expired_valid = expired.evaluate(shifted)
    assert not np.any(expired_valid)
    unbounded = replace(rq, interval_s=np.zeros_like(rq.interval_s))
    _, unbounded_valid = unbounded.evaluate(shifted)
    assert np.all(unbounded_valid)


def test_chain_serialization_and_backward_compatible_apply(tmp_path: Path) -> None:
    coordinates = _coordinates()
    metadata = _metadata()
    gain_curve = generate_vla_gain_curve(
        metadata, observation_time_s=coordinates.time_s[0]
    )
    opacity = estimate_vla_zenith_opacity(
        metadata, observation_time_s=coordinates.time_s[0]
    )
    rq = generate_vla_requantizer(metadata)
    chain = CalibrationChain(
        (gain_curve, opacity, rq), coordinates.antenna_position_m, {"version": 1}
    )
    write_calibration_chain(chain, tmp_path / "priors")
    restored = read_calibration_chain(tmp_path / "priors")
    expected, expected_valid = chain.evaluate(coordinates)
    actual, actual_valid = restored.evaluate(coordinates)
    np.testing.assert_allclose(actual, expected)
    np.testing.assert_array_equal(actual_valid, expected_valid)

    solution = identity_solution(
        antenna_count=2,
        correlations=(Correlation.RR, Correlation.LL),
        frequency_hz=coordinates.frequency_hz,
        time_s=coordinates.time_s,
    )
    baseline, _ = prior_baseline_jones(
        chain, coordinates, np.asarray([0]), np.asarray([1])
    )
    block = VisibilityBlock(
        np.zeros((1, 3)),
        coordinates.frequency_hz,
        np.asarray(baseline),
        np.ones((1, 2, 2)),
        np.zeros((1, 2, 2), bool),
        coordinates.time_s,
        np.asarray([0]),
        np.asarray([1]),
        (Correlation.RR, Correlation.LL),
        ReceptorBasis.CIRCULAR,
        model_visibility=np.ones((1, 2, 2), complex),
        phase_centre_rad=coordinates.phase_centre_rad,
        spectral_window_id=3,
    )
    corrected = apply_calibration(block, solution, priors=chain)
    legacy = apply_calibration(
        replace(block, visibility=np.ones_like(block.visibility)), solution
    )
    np.testing.assert_allclose(corrected.visibility, 1.0)
    np.testing.assert_allclose(legacy.visibility, 1.0)
    assert corrected.provenance["calibration"]["prior_terms"] == [
        "gain_curve",
        "opacity",
        "requantizer",
    ]


def test_generators_catalog_weather_rq_and_metadata_roundtrip() -> None:
    metadata = _metadata()
    custom_catalog = (
        GainCurveCatalogEntry(
            0,
            np.inf,
            4e9,
            8e9,
            (
                (1.0, 0.01, 0.0, 0.0),
                (1.0, 0.01, 0.0, 0.0),
            ),
            "test",
        ),
    )
    gain_curve = generate_vla_gain_curve(
        metadata,
        observation_time_s=60_000.0 * 86400.0,
        catalog=custom_catalog,
    )
    opacity = estimate_vla_zenith_opacity(
        metadata, observation_time_s=60_000.0 * 86400.0
    )
    rq = generate_vla_requantizer(metadata, chunk_size=1)

    assert gain_curve.valid.all()
    assert gain_curve.provenance["sources"] == ["test"]
    assert opacity.valid.all()
    assert opacity.provenance["measured_pwv_mm"] is not None
    np.testing.assert_allclose(rq.gain, [[0.5, 0.6], [0.7, 0.8]])
    assert ObservationMetadata.from_dict(metadata.to_dict()) == metadata


def test_prior_chain_remains_differentiable_for_solved_model() -> None:
    coordinates = _coordinates()
    rq = generate_vla_requantizer(_metadata())
    fixed, _ = rq.evaluate(coordinates)
    fixed_array = jnp.asarray(fixed)

    def loss(log_amplitude: jax.Array) -> jax.Array:
        predicted = fixed_array * jnp.exp(log_amplitude)
        return jnp.sum(jnp.abs(predicted - 1.0) ** 2)

    gradient = jax.grad(loss)(jnp.asarray(0.0))
    assert np.isfinite(float(gradient))


def test_jones_comparator_reports_values_and_flags() -> None:
    comparison = compare_jones(
        np.asarray([1.0, 1.01]),
        np.asarray([1.0, 1.0]),
        np.asarray([True, False]),
        np.asarray([True, True]),
    )
    assert comparison.relative_rms == 0.0
    assert comparison.compared_count == 1
    assert comparison.flag_mismatch_count == 1


def test_seven_case_srdp_oracle_fixture_is_complete() -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))

    assert payload["purpose"].startswith("CASA comparison oracle")
    assert len(payload["datasets"]) == 7
    for dataset in payload["datasets"]:
        roles = {table["role"] for table in dataset["tables"]}
        assert roles == {"gain_curve", "opacity", "requantizer"}
        for table in dataset["tables"]:
            assert table["fparam"]
            assert len(table["fparam"]) == len(table["flag"])
            assert table["spw_reference_frequency_hz"]
        gain_curve = next(
            table for table in dataset["tables"] if table["role"] == "gain_curve"
        )
        coefficients = np.asarray(gain_curve["fparam"])[:, 0, :]
        assert np.all(np.isfinite(coefficients))
        np.testing.assert_allclose(coefficients[:, :4], coefficients[:, 4:])
        requantizer = next(
            table for table in dataset["tables"] if table["role"] == "requantizer"
        )
        rq_values = np.asarray(requantizer["fparam"])[:, 0, :]
        np.testing.assert_allclose(rq_values[:, (1, 3)], 1.0)


def test_authoritative_gain_catalog_passes_seven_case_casa_gate() -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    for dataset in payload["datasets"]:
        table = next(
            table
            for table in dataset["tables"]
            if table["role"] == "gain_curve"
        )
        match = re.search(r"\.(\d{5}\.\d+)\.ms", table["source_table"])
        assert match is not None
        frequencies = table["spw_reference_frequency_hz"]
        antenna_count = max(table["antenna_id"]) + 1
        metadata = ObservationMetadata(
            antennas=tuple(
                AntennaRecord(
                    antenna,
                    f"ea{antenna + 1:02d}",
                    "",
                    (6_378_137.0, 0.0, 0.0),
                    25.0,
                    "ALT-AZ",
                )
                for antenna in range(antenna_count)
            ),
            spectral_windows=tuple(
                SpectralWindowRecord(spw, str(spw), frequency, (frequency,), ())
                for spw, frequency in enumerate(frequencies)
            ),
        )
        generated = generate_vla_gain_curve(
            metadata, observation_time_s=float(match.group(1)) * 86400.0
        )
        generated_coefficients = np.asarray(
            [
                generated.coefficients[antenna, spw].reshape(-1)
                for antenna, spw in zip(
                    table["antenna_id"],
                    table["spectral_window_id"],
                    strict=True,
                )
            ]
        )
        reference_coefficients = np.asarray(table["fparam"])[:, 0, :]
        relative_rms = np.sqrt(
            np.sum((generated_coefficients - reference_coefficients) ** 2)
            / np.sum(reference_coefficients**2)
        )
        assert relative_rms < 1e-3, dataset["obs_id"]
