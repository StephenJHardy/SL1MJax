from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sl1mjax.cassbeam_evla_c import evla_c_feedtaper_db
from sl1mjax.evla_c_metrics import (
    copolar_nonregression,
    hand_scores,
    main_lobe_nonregression_limit,
    residual_power,
)
from sl1mjax.evla_c_publication import (
    frequency_series_from_reports,
    holoraster_channel32_from_report,
    supersession_ledger,
)
from sl1mjax.evla_c_sensitivity import (
    classify_full_jones,
    inject_offdiagonal_visibility,
    manufactured_nontemplate_leakage,
    pooled_crosshand_delta,
)
from sl1mjax.evla_c_validation_refresh import (
    EXISTING_EVLA_C_ROOT,
    GENERIC_BEAM_ROOT,
    PUBLICATION_CHANNEL_HZ,
    PUBLICATION_CHANNELS,
    RESIDUAL_JONES_FREQUENCY_POLICY,
    RESIDUAL_JONES_NATIVE_CHANNEL,
    empty_status,
    experiment_manifest,
    publication_frequency_hz,
    publication_frequency_mhz,
    refuse_frozen_write,
    update_task,
    write_json_atomic,
)


def test_publication_frequencies_are_native_chan_freq() -> None:
    assert PUBLICATION_CHANNELS == (0, 8, 16, 24, 32, 40, 48, 56, 63)
    assert publication_frequency_hz(32) == 4.564e9
    assert publication_frequency_mhz(0) == 4500
    assert publication_frequency_mhz(63) == 4626
    assert publication_frequency_hz(8) == PUBLICATION_CHANNEL_HZ[8]
    with pytest.raises(ValueError):
        publication_frequency_hz(1)


def test_evla_c_taper_is_frequency_dependent() -> None:
    t0 = evla_c_feedtaper_db(publication_frequency_hz(0))
    t32 = evla_c_feedtaper_db(publication_frequency_hz(32))
    t63 = evla_c_feedtaper_db(publication_frequency_hz(63))
    assert t0 < t32 < t63
    np.testing.assert_allclose(t32, 12.2115, atol=1.0e-6)


def test_refuse_frozen_and_existing_evla_paths() -> None:
    with pytest.raises(RuntimeError):
        refuse_frozen_write(GENERIC_BEAM_ROOT / "spw4" / "dummy.dat")
    with pytest.raises(RuntimeError):
        refuse_frozen_write(EXISTING_EVLA_C_ROOT / "manifest.json")


def test_residual_jones_policy_is_explicit() -> None:
    manifest = experiment_manifest()
    assert manifest["residual_jones"]["native_channel"] == RESIDUAL_JONES_NATIVE_CHANNEL
    assert manifest["residual_jones"]["frequency_policy"] == RESIDUAL_JONES_FREQUENCY_POLICY
    assert manifest["residual_jones"]["silent_extrapolation"] is False
    assert manifest["nearest_plane_substitution"] is False
    assert manifest["production_accepted"] is False
    assert manifest["spw5_opened"] is False


def test_status_updates_are_atomic(tmp_path: Path) -> None:
    status = empty_status()
    status = update_task(status, "phase0_inventory", state="complete", artifact="manifest.json")
    path = write_json_atomic(tmp_path / "status.json", status)
    loaded = path.read_text(encoding="utf-8")
    assert "phase0_inventory" in loaded
    assert status["tasks"]["phase0_inventory"]["state"] == "complete"


def _vis(
    n: int, *, rl: complex = 0.0, pred_rl: complex = 0.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    meas = np.zeros((n, 2, 2), dtype=np.complex128)
    pred = np.zeros((n, 2, 2), dtype=np.complex128)
    wgt = np.ones((n, 2, 2), dtype=np.float64)
    meas[:, 0, 0] = 8.0
    meas[:, 1, 1] = 8.0
    meas[:, 0, 1] = rl
    meas[:, 1, 0] = np.conjugate(rl)
    pred[:, 0, 0] = 8.0
    pred[:, 1, 1] = 8.0
    pred[:, 0, 1] = pred_rl
    pred[:, 1, 0] = np.conjugate(pred_rl)
    return meas, pred, wgt


def test_residual_power_and_nonregression() -> None:
    meas, pred, wgt = _vis(8, rl=0.2, pred_rl=0.0)
    power = residual_power(meas, pred, wgt, "RL")
    assert power == pytest.approx(1.0)
    scores = hand_scores(meas, pred, wgt, source_i_jy=8.0)
    assert scores["RR"]["residual_power"] == pytest.approx(0.0)
    assert scores["RL"]["n"] == 8
    limit = main_lobe_nonregression_limit(0.01)
    assert limit == pytest.approx(0.002)
    assert main_lobe_nonregression_limit(0.05) == pytest.approx(0.005)
    diag = {"RR": {"residual_power": 0.01}, "LL": {"residual_power": 0.01}}
    full = {"RR": {"residual_power": 0.011}, "LL": {"residual_power": 0.011}}
    assert copolar_nonregression(full, diag)["passes"] is True
    worse = {"RR": {"residual_power": 0.02}, "LL": {"residual_power": 0.02}}
    assert copolar_nonregression(worse, diag)["passes"] is False


def test_classification_requires_both_partitions_and_does_not_promote() -> None:
    improve = {"delta": -0.01, "delta_lo": -0.02, "delta_hi": -0.005, "improves": True}
    none = {"delta": 0.0, "delta_lo": -0.001, "delta_hi": 0.002, "improves": False}
    spatial = {"RL": improve, "LR": improve, "pooled_RL_LR": improve}
    mover = {"RL": improve, "LR": improve, "pooled_RL_LR": improve}
    report = classify_full_jones(
        spatial=spatial,
        mover=mover,
        mover_units={"at_least_four": True},
        copolar={"passes": True},
        injection_detects_unit=True,
        injection_zero_is_null=True,
        numerical_ok=True,
    )
    assert report["outcome"] == "supported_on_spw4_development"
    assert report["production_accepted"] is False
    weak = classify_full_jones(
        spatial={"RL": none, "LR": none, "pooled_RL_LR": none},
        mover={"RL": none, "LR": none, "pooled_RL_LR": none},
        mover_units={"at_least_four": False},
        copolar={"passes": True},
        injection_detects_unit=True,
        injection_zero_is_null=True,
        numerical_ok=True,
    )
    assert weak["outcome"] == "inconclusive_sensitivity"
    blocked = classify_full_jones(
        spatial=spatial,
        mover=mover,
        mover_units={"at_least_four": True},
        copolar={"passes": True},
        injection_detects_unit=True,
        injection_zero_is_null=True,
        numerical_ok=False,
    )
    assert blocked["outcome"] == "blocked_implementation_or_contract"


def test_injection_and_nontemplate_are_distinct() -> None:
    meas, pred, _wgt = _vis(6, rl=0.1, pred_rl=0.02)
    template = pred - np.zeros_like(pred)
    template[:, 0, 1] = 0.02
    injected = inject_offdiagonal_visibility(meas, template, amplitude=1.0, phase_rad=0.0)
    np.testing.assert_allclose(injected[:, 0, 1], meas[:, 0, 1] + 0.02)
    other = manufactured_nontemplate_leakage(meas, seed=1, amplitude=0.05)
    assert not np.allclose(other, injected)


def test_residual_power_multiplicity_matches_row_repeat() -> None:
    meas, pred, wgt = _vis(8, rl=0.2, pred_rl=0.05)
    clusters = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64)
    chosen = np.array([0, 2, 2, 3], dtype=np.int64)
    multiplicity = np.bincount(chosen, minlength=4)[clusters]
    rows = np.concatenate([np.flatnonzero(clusters == item) for item in chosen])
    weighted = residual_power(meas, pred, wgt, "RL", multiplicity=multiplicity)
    repeated = residual_power(meas[rows], pred[rows], wgt[rows], "RL")
    assert weighted == pytest.approx(repeated, abs=1.0e-12)


def test_pooled_bootstrap_flags_clear_improvement() -> None:
    rng = np.random.default_rng(0)
    n = 40
    meas = np.zeros((n, 2, 2), dtype=np.complex128)
    diag = np.zeros((n, 2, 2), dtype=np.complex128)
    full = np.zeros((n, 2, 2), dtype=np.complex128)
    wgt = np.ones((n, 2, 2), dtype=np.float64)
    meas[:, 0, 0] = 8.0
    meas[:, 1, 1] = 8.0
    signal = 0.2 + 0.05j
    meas[:, 0, 1] = signal + 0.01 * rng.normal(size=n)
    meas[:, 1, 0] = np.conjugate(meas[:, 0, 1])
    full[:, 0, 1] = signal
    full[:, 1, 0] = np.conjugate(signal)
    clusters = np.repeat(np.arange(8), 5)
    report = pooled_crosshand_delta(meas, full, diag, wgt, clusters, n_boot=80, seed=0)
    assert report["pooled_RL_LR"]["improves"] is True
    assert report["RL"]["improves"] is True


def test_publication_channel32_document_is_unfrozen() -> None:
    report = {
        "channel": 32,
        "frequency_hz": 4.564e9,
        "n_development": 12,
        "n_full_raster_usable": 20,
        "source_i_jy": 8.0,
        "scores": {
            "arms": {
                "evla_c_source_diagonal": {
                    "train": {
                        "all": {
                            "RR": {"n": 12, "residual_power": 0.05, "slope_real": 1.0},
                            "LL": {"n": 12, "residual_power": 0.06, "slope_real": 1.0},
                            "RL": {"n": 12, "residual_power": 0.99, "slope_real": 0.0},
                            "LR": {"n": 12, "residual_power": 0.99, "slope_real": 0.0},
                        },
                        "main_lobe": {
                            "RR": {"n": 8, "residual_power": 0.04, "median_abs_over_i": 0.05},
                            "LL": {"n": 8, "residual_power": 0.05, "median_abs_over_i": 0.05},
                        },
                    },
                    "spatial_holdout": {"all": {}},
                },
                "evla_c_source_full_jones": {
                    "train": {
                        "all": {
                            "RR": {"n": 12, "residual_power": 0.05},
                            "LL": {"n": 12, "residual_power": 0.06},
                            "RL": {"n": 12, "residual_power": 0.90},
                            "LR": {"n": 12, "residual_power": 0.91},
                        }
                    }
                },
            }
        },
        "sensitivity": {"spatial": {"pooled_RL_LR": {"improves": False}}},
    }
    document = holoraster_channel32_from_report(report)
    assert document["full_jones_frozen"] is False
    assert document["production_factory_modified"] is False
    assert document["model_selected"] is False
    assert document["coordinate_query"] == "source_lm_feed"
    assert document["n_development"] == 12
    ledger = supersession_ledger()
    assert ledger["production_accepted"] is False
    statuses = {item["item"]: item["status"] for item in ledger["entries"]}
    assert statuses["SPW 5"] == "sealed"


def test_identity_match_on_shared_source_lm(tmp_path: Path) -> None:
    from sl1mjax.evla_c_holoraster_compare import identity_vs_prior_evla_diagonal

    n = 5
    lm = np.linspace(-1e-3, 1e-3, n * 2).reshape(n, 2)
    pred = np.zeros((n, 2, 2), dtype=np.complex128)
    pred[:, 0, 0] = 0.8 + 0.01j
    pred[:, 1, 1] = 0.7 - 0.02j
    new_path = tmp_path / "new.npz"
    old_path = tmp_path / "old.npz"
    np.savez_compressed(
        new_path,
        measured=pred,
        weight=np.ones((n, 2, 2)),
        source_lm_feed=lm,
        moving_id=np.arange(n, dtype=np.int32),
        reference_id=np.zeros(n, dtype=np.int32),
        evla_c_source_diagonal=pred,
        evla_c_source_full_jones=pred,
    )
    rr_ll = np.column_stack((pred[:, 0, 0], pred[:, 1, 1]))
    np.savez_compressed(
        old_path,
        source_lm_feed=lm,
        moving_id=np.arange(n, dtype=np.int32),
        reference_id=np.zeros(n, dtype=np.int32),
        evla_c_source_lm_rr_ll=rr_ll,
    )
    report = identity_vs_prior_evla_diagonal(new_path, old_path, atol=1.0e-12)
    assert report["status"] == "pass"
    assert report["n_matched"] == n


def test_offset_ring_adapter_uses_inner_holdout() -> None:
    from sl1mjax.evla_c_publication import offset_ring_from_c147_report

    report = {
        "fields": [{"radius_arcmin": 3.61}, {"radius_arcmin": 3.62}],
        "channels": [
            {
                "channel": 32,
                "scores": {
                    "inner_holdout": {
                        "diagonal": {
                            "RR": {"n": 10, "residual_power": 0.012},
                            "LL": {"n": 10, "residual_power": 0.009},
                        }
                    }
                },
            }
        ],
    }
    adapted = offset_ring_from_c147_report(report)
    assert adapted["closure"]["passed"] is True
    assert adapted["historical_holdout"] is True
    assert adapted["q_u_is_frequency_holdout"] is False


def test_frequency_series_uses_main_lobe_not_whole_raster() -> None:
    root = Path("docs/evla_c_full_jones_validation_refresh/channels")
    reports = [
        json.loads((root / name).read_text(encoding="utf-8"))
        for name in ("channel00_report.json", "channel08_report.json", "channel32_report.json")
        if (root / name).is_file()
    ]
    if len(reports) < 2:
        pytest.skip("refresh channel reports are not in the working tree")
    series = frequency_series_from_reports(reports)
    by_ch = {int(item["channel"]): item for item in series["channels"]}
    ch0 = by_ch[0]
    assert ch0["region"] == "main_lobe"
    assert ch0["diagonal"]["rr"]["residual_power"] == pytest.approx(0.01606, abs=2e-4)
    assert ch0["diagonal_all"]["rr"]["residual_power"] > ch0["diagonal"]["rr"]["residual_power"]


def test_c05_uses_refresh_outcome() -> None:
    from sl1mjax.beam_validation_outputs import ValidationBundle
    from sl1mjax.beam_validation_statistics import build_claims

    hand = {
        "n": 12,
        "slope_real": 1.0,
        "slope_imag": 0.0,
        "correlation_abs": 0.99,
        "residual_power": 0.006,
        "median_abs_ratio": 1.05,
        "median_abs_obs": 0.2,
        "median_abs_pred": 0.01,
    }
    cross = {**hand, "residual_power": 0.99, "correlation_abs": 0.1}
    channel32 = {
        "diagonal": {"rr": hand, "ll": hand, "rl": cross, "lr": cross},
        "experimental_full_jones": {"rr": hand, "ll": hand, "rl": cross, "lr": cross},
        "regions": {
            "main_lobe": {"median_abs_rr_over_i": 0.05, "median_abs_ll_over_i": 0.05},
            "mid": {"median_abs_rr_over_i": 0.06, "median_abs_ll_over_i": 0.06},
            "outer_diagnostic": {
                "median_abs_rr_over_i": 0.03,
                "median_abs_ll_over_i": 0.03,
            },
            "all": {"median_abs_rr_over_i": 0.04, "median_abs_ll_over_i": 0.04},
            "hand_residual_power": {
                "main_lobe": {"rr": hand, "ll": hand},
                "mid": {
                    "rr": {**hand, "residual_power": 0.05},
                    "ll": {**hand, "residual_power": 0.05},
                },
                "outer_diagnostic": {
                    "rr": {**hand, "residual_power": 0.3, "correlation_abs": 0.8},
                    "ll": {**hand, "residual_power": 0.3, "correlation_abs": 0.8},
                },
                "all": {"rr": hand, "ll": hand},
            },
        },
        "classification": {
            "outcome": "inconclusive_sensitivity",
            "reason": "unit injection not recovered",
        },
    }
    squint = {
        "measured": {
            "estimator": "mainlobe_20pct_peak",
            "publication_estimator": True,
            "full_raster_is_publication_estimator": False,
            "separation_arcmin": 0.5,
            "memo195_separation_arcmin": 0.526,
            "full_raster_separation_arcmin": 0.7,
        },
        "cassbeam": {
            "estimator": "mainlobe_20pct_peak",
            "publication_estimator": True,
            "full_raster_is_publication_estimator": False,
            "separation_arcmin": 0.41,
        },
    }
    bundle = ValidationBundle(
        root=Path("."),
        manifest={"publication_version": "vla_c_band_beam_validation_v2", "files": {}},
        claims={"claims": []},
        observation={},
        calibration={},
        convention_gates={},
        holoraster_channel32=channel32,
        holoraster_frequency={"channels": []},
        squint=squint,
        offset_ring={
            "closure": {
                "rr": {"relative_power": 0.01, "n": 8},
                "ll": {"relative_power": 0.01, "n": 8},
                "passed": True,
            },
            "radius_arcmin": 3.61,
            "q_u_is_frequency_holdout": False,
        },
        residual_geometry={},
        residual_strata={},
        radial_coherence={},
        antenna_coherence={},
        bright_source_examples={},
        frequency_series={},
        offset_ring_fields={},
        crosshand_quadrants={},
        coordinate_feed_comparison={
            "commanded_source_max_abs_error_rad": 0.0,
            "coordinate_contract": {"axis_map": "az_to_plus_l_el_to_plus_m_no_swap"},
            "paired_scores": {
                "evla_c_source_lm_vs_generic_source_lm": {
                    "spatial": {
                        "improves": True,
                        "delta": -0.005,
                        "delta_lo": -0.008,
                        "delta_hi": -0.003,
                    },
                    "moving": {
                        "improves": True,
                        "delta": -0.006,
                        "delta_lo": -0.007,
                        "delta_hi": -0.004,
                    },
                }
            },
        },
    )
    claims = build_claims(bundle)
    by_id = {item["id"]: item for item in claims["claims"]}
    assert claims["full_jones"] == "inconclusive_sensitivity"
    assert by_id["C05"]["status"] == "warn"


def test_injection_grid_covers_predeclared_amplitudes() -> None:
    from sl1mjax.evla_c_sensitivity import INJECTION_AMPLITUDES, INJECTION_PHASES_DEG

    assert INJECTION_AMPLITUDES == (0.0, 0.3, 1.0, 3.0)
    assert INJECTION_PHASES_DEG == (0.0, 90.0, 180.0)
