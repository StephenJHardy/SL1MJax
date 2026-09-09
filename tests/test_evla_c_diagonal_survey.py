from __future__ import annotations

import json
from pathlib import Path

import pytest

from sl1mjax.evla_c_diagonal_survey import (
    BEAM_ROOT,
    CATALOG_ID,
    PUBLICATION_BUNDLE_VERSION,
    RESIDUAL_JONES_POLICY,
    SCIENTIFIC_SPW45_MS,
    build_slot_status,
    execution_from_measurement_set,
    slot_product_stem,
    calibration_policy,
    classify_execution_band,
    empty_status,
    experiment_manifest,
    holoraster_calibration_policy,
    nearest_supported_channel,
    pass_a_channel,
    pass_b_channels,
    planned_slots,
    refuse_frozen_write,
    refuse_residual_jones_apply,
    survey_holoraster_row_cache,
    taper_in_documented_range,
    update_task,
    work_ms_path,
    write_json_atomic,
)


def test_pass_channels_match_64chan_rule() -> None:
    assert pass_a_channel(64) == 32
    assert pass_b_channels(64) == (8, 24, 32, 40, 56)


def test_pass_channels_scale_and_round_down() -> None:
    assert pass_a_channel(32) == 16
    assert pass_b_channels(32) == (4, 12, 16, 20, 28)
    assert pass_a_channel(63) == 31
    assert pass_b_channels(63) == (7, 23, 31, 39, 55)


def test_unsupported_pass_a_uses_nearest_lower_tie() -> None:
    assert nearest_supported_channel(64, [31, 33], preferred=32) == 31
    assert nearest_supported_channel(64, [32, 40]) == 32
    with pytest.raises(ValueError, match="no supported"):
        nearest_supported_channel(64, [])


def test_residual_jones_is_refused() -> None:
    with pytest.raises(RuntimeError, match="not applied"):
        refuse_residual_jones_apply()
    assert experiment_manifest()["residual_jones_policy"] == RESIDUAL_JONES_POLICY
    assert experiment_manifest()["full_jones_is_gate"] is False
    assert experiment_manifest()["publication_bundle"] == PUBLICATION_BUNDLE_VERSION
    assert experiment_manifest()["catalog_id"] == CATALOG_ID


def test_taper_range_and_frozen_paths() -> None:
    assert taper_in_documented_range(4.564e9)
    assert taper_in_documented_range(7.9e9)
    assert not taper_in_documented_range(3.0e9)
    with pytest.raises(RuntimeError):
        refuse_frozen_write(Path("src/sl1mjax/data/vla_c_band_beam_validation_v2/manifest.json"))
    with pytest.raises(RuntimeError):
        refuse_frozen_write(BEAM_ROOT.parent / "cassbeam_evla_c_g1024_p32_ninepub_20260908" / "x")


def test_band_classifier_uses_frequencies_not_names() -> None:
    assert classify_execution_band(3.99e9, 6.01e9) == "lower_c"
    assert classify_execution_band(5.99e9, 8.01e9) == "upper_c_candidate"
    assert classify_execution_band(8.0e9, 12.0e9) == "not_c_band"
    assert classify_execution_band(4.0e9, 8.0e9) == "ambiguous"


def test_window_classifier_ignores_auxiliary_lower_windows() -> None:
    from sl1mjax.evla_c_diagonal_survey import classify_spectral_windows, window_role

    windows = [
        {
            "spectral_window_id": index,
            "freq_min_hz": 5.988e9 + index * 0.128e9,
            "freq_max_hz": 6.114e9 + index * 0.128e9,
        }
        for index in range(16)
    ]
    windows.append({"spectral_window_id": 16, "freq_min_hz": 4.832e9, "freq_max_hz": 4.958e9})
    windows.append({"spectral_window_id": 17, "freq_min_hz": 4.960e9, "freq_max_hz": 5.086e9})
    assert classify_spectral_windows(windows) == "upper_c_candidate"
    assert window_role(windows[16], "upper_c_candidate") == "auxiliary_lower_c_frequency"
    assert window_role(windows[0], "upper_c_candidate") == "science"


def test_slot_status_explains_unscored_science_windows() -> None:
    table = json.loads(Path("docs/evla_c_diagonal_survey/frequency_table.json").read_text())
    scored = [
        {
            "frequency_hz": 4.564e9,
            "status": "scientifically-qualified",
            "main_lobe_both_hands_accepted": True,
        }
    ]
    status = build_slot_status(table, scored)
    assert status["n_slots"] > status["n_scored"]
    assert status["n_scored"] == 1
    assert any(row["state"] == "scientifically-qualified" for row in status["slots"])
    assert any(
        row["execution"] == "lower_c"
        and row["spectral_window_id"] == 4
        and row["pass"] == "B"
        and row["reason"] == "pass_b_score_pending"
        for row in status["slots"]
    )
    assert any(
        row["execution"] == "upper_c"
        and row["state"] == "planned"
        and "3c138" in row["reason"]
        for row in status["slots"]
    )
    ready = build_slot_status(
        table,
        scored,
        upper_c_inventory={
            "uses_3c138": True,
            "uses_3c286": False,
            "lower_c_scan_list_reusable": False,
        },
    )
    assert any(
        row["execution"] == "upper_c"
        and row["reason"] == "upper_c_work_copy_or_score_pending"
        for row in ready["slots"]
    )
    assert all(row["state"] != "complete" or row["scored"] for row in status["slots"])
    assert not any(row["state"] == "0" or row["reason"] == "zero" for row in status["slots"])
    assert status["pass_a_survey_complete"] is False


def test_pass_b_becomes_pass_a_only_after_centre_survey() -> None:
    table = json.loads(Path("docs/evla_c_diagonal_survey/frequency_table.json").read_text())
    scored = []
    for execution in ("lower_c", "upper_c"):
        for window in table["executions"][execution]["windows"]:
            if window.get("role") != "science":
                continue
            scored.append(
                {
                    "frequency_hz": window["pass_a_frequency_hz"],
                    "status": "complete",
                    "main_lobe_both_hands_accepted": False,
                }
            )
    status = build_slot_status(
        table,
        scored,
        upper_c_inventory={
            "uses_3c138": True,
            "uses_3c286": False,
            "lower_c_scan_list_reusable": False,
        },
    )
    assert status["pass_a_survey_complete"] is True
    assert status["pass_b_scope"] == "lower_c_spw_4_6_only"
    assert status["acceptance"] == "pass_a_complete_pass_b_limited"
    assert status["numerical_qualification"] == "incomplete_no_band_wide_convergence"
    assert status["upper_c_archive_integrity"] == "unverified"
    assert any(row["state"] == "pass_a_only" for row in status["slots"])
    assert all(
        row["reason"] == "pass_b_not_opened_centre_survey_complete"
        for row in status["slots"]
        if row["state"] == "pass_a_only"
    )
    assert not any(row["state"] == "planned" for row in status["slots"])
    upper_scored = build_slot_status(
        table,
        [
            {
                "frequency_hz": 6.052e9,
                "status": "complete",
                "main_lobe_both_hands_accepted": False,
            }
        ],
        upper_c_inventory={
            "uses_3c138": True,
            "uses_3c286": False,
            "lower_c_scan_list_reusable": False,
        },
    )
    assert any(
        row["execution"] == "upper_c"
        and abs(float(row["frequency_hz"]) - 6.052e9) < 1.0
        and row["scored"]
        and row["state"] == "complete"
        for row in upper_scored["slots"]
    )


def test_upper_c_slot_products_do_not_reuse_lower_c_names() -> None:
    assert execution_from_measurement_set(Path("upper_c_spw00.work.ms")) == "upper_c"
    assert execution_from_measurement_set(Path("lower_c_spw00.work.ms")) == "lower_c"
    root = Path("/tmp/survey-products")
    assert slot_product_stem(
        root, execution="lower_c", spectral_window_id=0, channel=32
    ).name == "spw00_channel32"
    assert slot_product_stem(
        root, execution="upper_c", spectral_window_id=0, channel=32
    ).name == "upper_c_spw00_channel32"


def test_planned_slots_keep_empty_pass_b_members() -> None:
    freqs = [4.5e9 + 2.0e6 * index for index in range(64)]
    slots = planned_slots(
        [
            {
                "spectral_window_id": 4,
                "n_chan": 64,
                "chan_freq_hz": freqs,
                "pass_a_channel": 32,
                "pass_b_channels": [8, 24, 32, 40, 56],
            }
        ]
    )
    assert [row["channel"] for row in slots if row["pass"] == "A"] == [32]
    assert [row["channel"] for row in slots if row["pass"] == "B"] == [8, 24, 32, 40, 56]


def test_work_copy_path_is_not_scientific_ms() -> None:
    path = work_ms_path("lower_c", 3)
    assert path != SCIENTIFIC_SPW45_MS
    assert path.name == "lower_c_spw03.work.ms"
    assert "evla_c_diagonal_survey_v1" in path.as_posix()
    upper = work_ms_path("upper_c", 0)
    assert upper.name == "upper_c_spw00.work.ms"
    assert upper != path
    assert survey_holoraster_row_cache(SCIENTIFIC_SPW45_MS, 4) is None
    cached = survey_holoraster_row_cache(path, 3)
    assert cached is not None
    assert cached.name.endswith(".holoraster_ddid3.rows.npy")
    assert survey_holoraster_row_cache(Path("docs/evla_c_diagonal_survey"), 0) is None


def test_cal_wrapper_refuses_upper_c_work_ms(tmp_path: Path) -> None:
    import os
    import subprocess

    work = tmp_path / "work_ms" / "upper_c_spw00.work.ms"
    work.mkdir(parents=True)
    result = subprocess.run(
        ["bash", "scripts/run_thol0001_survey_spw_calibration.sh", "0", "upper_c"],
        env={**os.environ, "OUT": str(tmp_path)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "3C286" in result.stderr


def test_upper_c_calibration_policy_uses_3c138_scans() -> None:
    inventory = json.loads(
        Path("docs/evla_c_diagonal_survey/upper_c_scan_inventory.json").read_text()
    )
    policy = holoraster_calibration_policy(inventory)
    assert policy["uses_3c138"] is True
    assert policy["uses_3c286"] is False
    assert policy["lower_c_scan_list_reusable"] is False
    assert policy["flux_scans"] == "3,53"
    assert policy["solve_flux_scan"] == "3"
    assert policy["held_out_flux_scan"] == "53"
    assert policy["flux_scans"] != "2,51"
    assert "15" in policy["phase_scans"].split(",")
    assert "101" not in policy["phase_scans"].split(",")
    assert policy["check_source"] == "J0521+1638"


def test_cal_wrapper_accepts_upper_c_with_policy(tmp_path: Path) -> None:
    import os
    import stat
    import subprocess

    inventory = json.loads(
        Path("docs/evla_c_diagonal_survey/upper_c_scan_inventory.json").read_text()
    )
    policy = holoraster_calibration_policy(inventory)
    (tmp_path / "upper_c_calibration_policy.json").write_text(json.dumps(policy))
    work = tmp_path / "work_ms" / "upper_c_spw00.work.ms"
    work.mkdir(parents=True)
    casa = tmp_path / "casa"
    casa.write_text("#!/bin/bash\nexit 0\n")
    casa.chmod(casa.stat().st_mode | stat.S_IEXEC)
    result = subprocess.run(
        ["bash", "scripts/run_thol0001_survey_spw_calibration.sh", "0", "upper_c"],
        env={**os.environ, "OUT": str(tmp_path), "CASA": str(casa)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "calibrating" in result.stdout


def test_calibration_policy_refuses_residual_jones() -> None:
    policy = calibration_policy()
    assert policy["residual_jones"] == "not_applied"
    assert policy["channel32_residual_jones"]["apply"] is False
    assert policy["new_spws"]["require_new_work_copies"] is True
    assert policy["reuse_incompatible_tables"] is False


def test_survey_catalog_refuses_nearest_plane() -> None:
    from sl1mjax.evla_c_survey_compare import SURVEY_MODEL_ID, require_survey_frequency

    class _Catalog:
        def require_exact_mhz(self, frequency_hz: float) -> int:
            raise ValueError("no exact high-res CASSBEAM plane at 4700 MHz")

    with pytest.raises(ValueError, match="no exact"):
        require_survey_frequency(_Catalog(), 4.700e9)  # type: ignore[arg-type]
    assert SURVEY_MODEL_ID == "cassbeam_evla_c_diagonal_survey_v1"


def test_holoraster_audit_is_cached_per_spw(tmp_path: Path) -> None:
    from sl1mjax.evla_c_survey_compare import holoraster_audit_for_slot

    calls: list[tuple[object, ...]] = []

    class _Cmp:
        pass

    def _fake_audit(measurement_set, *, target_frequency_hz):
        calls.append((measurement_set, target_frequency_hz))
        return {"n": len(calls)}

    import sl1mjax.evla_c_survey_compare as survey

    original = survey.audit_holography_measurement_set
    survey.audit_holography_measurement_set = _fake_audit
    try:
        cmp = _Cmp()
        first = holoraster_audit_for_slot(
            cmp, tmp_path / "a.ms", frequency_hz=4.052e9, spectral_window_id=0
        )
        second = holoraster_audit_for_slot(
            cmp, tmp_path / "a.ms", frequency_hz=4.004e9, spectral_window_id=0
        )
        other = holoraster_audit_for_slot(
            cmp, tmp_path / "a.ms", frequency_hz=4.180e9, spectral_window_id=1
        )
    finally:
        survey.audit_holography_measurement_set = original
    assert first is second
    assert first is not other
    assert len(calls) == 2


def test_survey_extract_does_not_use_spw4_seal() -> None:
    import inspect

    from sl1mjax import evla_c_survey_compare as survey
    from sl1mjax.holography_diagonal_correction import holoraster_correction_holdouts

    source = inspect.getsource(survey.load_holoraster_slot)
    assert "spw4_correction_holdouts" not in source
    assert "refuse_spw5" not in source
    assert holoraster_correction_holdouts is survey.holoraster_correction_holdouts


def test_survey_score_identity_prediction() -> None:
    import numpy as np

    from sl1mjax.evla_c_survey_compare import score_extract

    vis = np.ones((8, 2, 2), dtype=np.complex128)
    vis[:, 0, 1] = 0.0
    vis[:, 1, 0] = 0.0
    weight = np.ones((8, 2, 2), dtype=np.float64)
    main = np.zeros(8, dtype=bool)
    main[:4] = True
    mid = ~main
    report = score_extract(
        {
            "spectral_window_id": 5,
            "channel": 32,
            "frequency_hz": 4.692e9,
            "data_column": "CORRECTED_DATA",
            "n_development": 8,
            "intensity": np.full(8, 20.0),
            "measured": vis,
            "weight": weight,
            "source_lm": np.array(
                [[0.0, 0.0], [1e-4, 0.0], [0.0, 1e-4], [1e-4, 1e-4],
                 [0.01, 0.0], [0.0, 0.01], [0.01, 0.01], [0.02, 0.0]]
            ),
            "regions": {
                "main_lobe": main,
                "mid": mid,
                "outer_diagnostic": np.zeros(8, dtype=bool),
            },
        },
        vis,
    )
    assert report["main_lobe_both_hands_accepted"] is True
    assert report["empirical_main_lobe_accepted"] is True
    assert report["numerically_qualified"] is False
    assert report["regions"]["main_lobe"]["RR"]["residual_power"] == 0.0
    assert "fixed_geometry" in report


def test_survey_phase5_maps_and_null_interval(tmp_path: Path) -> None:
    import numpy as np

    from sl1mjax.evla_c_survey_plots import (
        first_null_from_radial,
        maps_from_export,
        slot_diagnostics,
        write_plot_products,
    )

    rng = np.random.default_rng(0)
    n = 400
    l_m = rng.normal(0.0, 0.004, size=(n, 2))
    radius = np.hypot(l_m[:, 0], l_m[:, 1])
    amp = np.exp(-0.5 * (radius / 0.002) ** 2)
    measured = np.zeros((n, 2, 2), dtype=np.complex128)
    predicted = np.zeros((n, 2, 2), dtype=np.complex128)
    measured[:, 0, 0] = amp + 0.01
    measured[:, 1, 1] = amp + 0.01
    predicted[:, 0, 0] = amp
    predicted[:, 1, 1] = amp
    export = {
        "source_lm_feed": l_m,
        "measured": measured,
        "predicted": predicted,
        "weight": np.ones((n, 2, 2), dtype=np.float64),
        "frequency_hz": np.array([4.564e9]),
    }
    maps = maps_from_export(export)
    assert maps["rr_measured"].shape == maps["weight"].shape
    assert np.isfinite(np.nanmax(np.abs(maps["rr_measured"])))
    diag = slot_diagnostics(export, report={"source_i_jy": 8.0})
    assert diag["nulls"]["RR"]["status"] in {"coarse_minimum", "unresolved"}
    assert diag["coarse_radial_minima"]["RR"]["quantity"] == "coarse_radial_minimum"
    assert first_null_from_radial(diag["radial"]["RR"])["status"] in {
        "coarse_minimum",
        "unresolved",
    }
    assert "model_first_minima" in diag
    assert "measured_first_null" in diag
    np.savez(tmp_path / "spw04_channel32_export.npz", **export)
    (tmp_path / "spw04_channel32_report.json").write_text(
        '{"source_i_jy": 8.0, "status": "scientifically-qualified", '
        '"spectral_window_id": 4, "channel": 32}',
        encoding="utf-8",
    )
    result = write_plot_products(tmp_path, tmp_path / "plots")
    assert result["n_slots"] == 1
    assert (tmp_path / "plots" / "phase5_diagnostics.json").is_file()
    assert (tmp_path / "plots" / "spw04_channel32_signed_cuts.png").is_file()
    assert (tmp_path / "plots" / "nulls_vs_frequency.png").is_file()
    assert "rr_diag" in diag["cuts"]
    assert np.isfinite(np.real(diag["cuts"]["rr_m0"]["measured"])).any()


def test_survey_v3_bundle_refuses_v2_and_writes_handoff(tmp_path: Path) -> None:
    import json
    import shutil

    from sl1mjax.evla_c_survey_publication import write_survey_v3_bundle

    ledger = Path("docs/evla_c_diagonal_survey")
    with pytest.raises(RuntimeError, match="frozen"):
        write_survey_v3_bundle(
            ledger_dir=ledger,
            output_dir=tmp_path / "vla_c_band_beam_validation_v2",
        )
    destination = tmp_path / "vla_c_band_beam_validation_v3"
    if (ledger / "suitability_table.json").is_file():
        write_survey_v3_bundle(ledger_dir=ledger, output_dir=destination)
        manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["publication_version"] == "vla_c_band_beam_validation_v3"
        assert manifest["production_accepted"] is False
        assert (destination / "imaging_handoff.json").is_file()
        assert (destination / "supersession_ledger.json").is_file()
        handoff = json.loads((destination / "imaging_handoff.json").read_text(encoding="utf-8"))
        assert handoff["interpolation_policy"] == "refused"
        assert 4536 in handoff["imaging_node_mhz"]
        assert "run_3c391_phase6_bacchus.py" in handoff["mosaic_command"]
        assert "evla_c_diagonal_survey_v1" in handoff["mosaic_command"]
    else:
        shutil.copytree(ledger, tmp_path / "ledger")


def test_survey_v3_notebook_and_render_leave_v2_alone(tmp_path: Path) -> None:
    import importlib.util
    import sys

    def _load(name: str, relative: str):
        path = Path(relative)
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module

    writer = _load("survey_v3_nb", "scripts/write_vla_c_band_beam_validation_v3_notebook.py")
    renderer = _load("survey_v3_md", "scripts/render_vla_c_band_beam_validation_v3.py")
    path = writer.write_notebook(tmp_path / "vla_c_band_beam_validation_v3.ipynb")
    text = path.read_text(encoding="utf-8")
    assert "vla_c_band_beam_validation_v3" in text
    page = renderer.render(Path("src/sl1mjax/data/vla_c_band_beam_validation_v3"))
    assert "does not replace" in page
    assert "evla_c_diagonal_survey_v1" in page
    sys.argv = [
        "render",
        "--output",
        str(tmp_path / "docs" / "vla_c_band_beam_validation_v2.md"),
    ]
    with pytest.raises(RuntimeError, match="frozen v2"):
        renderer.main()


def test_live_v3_switch_requires_snapshot_and_frozen_v2(tmp_path: Path, monkeypatch) -> None:
    import importlib.util

    path = Path("scripts/switch_vla_c_band_beam_validation_live_v3.py")
    spec = importlib.util.spec_from_file_location("switch_live_v3", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "SNAPSHOT", tmp_path / "missing")
    with pytest.raises(RuntimeError, match="snapshot missing"):
        module._require_snapshot()
    monkeypatch.setattr(module, "SNAPSHOT", Path("docs/vla_c_band_beam_validation_v2_snapshot"))
    module._require_snapshot()
    module._require_frozen_v2()


def test_status_round_trip(tmp_path: Path) -> None:
    status = update_task(empty_status(), "phase0_manifest", state="complete", artifact="x")
    path = write_json_atomic(tmp_path / "status.json", status)
    assert path.is_file()
    assert status["tasks"]["phase0_manifest"]["state"] == "complete"


def test_coarse_bins_cannot_relabel_a_later_minimum_as_first() -> None:
    import numpy as np

    from sl1mjax.beam_validation_statistics import RADIAL_EDGES_ARCMIN
    from sl1mjax.evla_c_survey_plots import (
        coarse_radial_minimum,
        first_magnitude_minimum,
        rebin_radial_medians,
    )

    radius = np.linspace(0.0, 40.0, 801)
    amp = np.interp(
        radius,
        [0.0, 3.0, 10.5, 16.0, 22.5, 30.0, 40.0],
        [1.0, 0.55, 0.02, 0.14, 0.01, 0.08, 0.07],
    )
    first = first_magnitude_minimum(radius, amp)
    assert first["status"] == "resolved"
    assert 9.5 <= float(first["r_arcmin"]) <= 11.5
    coarse = coarse_radial_minimum(
        rebin_radial_medians(radius, amp, RADIAL_EDGES_ARCMIN)
    )
    assert coarse["quantity"] == "coarse_radial_minimum"
    assert coarse["status"] == "coarse_minimum"
    assert float(coarse["r_lo_arcmin"]) >= 20.0


def test_fixed_geometry_mask_ignores_contaminated_amplitude() -> None:
    import numpy as np

    from sl1mjax.evla_c_survey_compare import (
        fixed_radius_region_masks,
        score_fixed_geometry,
    )

    offset = np.zeros((6, 2), dtype=np.float64)
    offset[:3, 0] = np.deg2rad(2.0 / 60.0)
    offset[3:, 0] = np.deg2rad(30.0 / 60.0)
    masks = fixed_radius_region_masks(offset)
    assert int(masks["main_lobe"].sum()) == 3
    assert int(masks["outer_diagnostic"].sum()) == 3
    vis = np.ones((6, 2, 2), dtype=np.complex128)
    vis[0, 0, 0] = 0.05
    pred = np.ones((6, 2, 2), dtype=np.complex128)
    weight = np.ones((6, 2, 2), dtype=np.float64)
    scored = score_fixed_geometry(
        measured=vis,
        predicted=pred,
        weight=weight,
        offset_lm=offset,
        source_i_jy=1.0,
    )
    assert scored["n_main_lobe"] == 3
    assert scored["numerically_qualified"] is False
