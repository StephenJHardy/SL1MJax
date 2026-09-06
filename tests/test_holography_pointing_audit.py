from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sl1mjax.holography import (
    HolographyRowReason,
    holography_execution_provenance,
    identities_from_holography_path,
    inventory_from_dict,
    load_synthetic_holography_pointing_table,
    pointing_table_overlap_pairs,
    resolve_antenna_pointing,
    synthetic_memo195_pointing_table,
)
from sl1mjax.holography_calibration import (
    C147_OFFSET_FIELD_IDS,
    D_CODE_FIELD_ID,
    calibration_field_policy,
    empirical_variance_from_repeat_scatter,
    field_may_enter_d_solve,
    product_manifest,
)
from sl1mjax.holography_commissioning import (
    _factorize_groups,
    _moving_by_scan,
    select_commissioning_fields,
)
from sl1mjax.holography_pointing_audit import (
    HolographyMainMetadata,
    format_pointing_reconstruction_audit,
    is_map_antenna_surface,
    pointing_reconstruction_audit_as_dict,
    reconstruct_holography_pointing,
    select_native_channels,
    unique_times_within_groups,
)
from sl1mjax.holography_pointing_maps import (
    PASS2_MEASURED_RASTER_NOTE,
    memo195_expected_offsets,
    scan_passes,
    write_pointing_audit_bundle,
)

_PHASE = (np.deg2rad(84.0), np.deg2rad(50.0))


def _main_from_times(
    times: np.ndarray,
    *,
    data_desc_id: np.ndarray,
    state_id: np.ndarray | None = None,
    field_id: np.ndarray | None = None,
    scan_number: np.ndarray | None = None,
    extra_unmatched: float | None = 200.0,
) -> HolographyMainMetadata:
    rows_time: list[float] = []
    rows_ant1: list[int] = []
    rows_ant2: list[int] = []
    rows_ddid: list[int] = []
    rows_state: list[int] = []
    rows_field: list[int] = []
    rows_scan: list[int] = []
    states = np.zeros(times.size, dtype=np.int32) if state_id is None else state_id
    fields = np.zeros(times.size, dtype=np.int32) if field_id is None else field_id
    scans = np.ones(times.size, dtype=np.int32) if scan_number is None else scan_number
    for time, ddid, state, field, scan in zip(
        times, data_desc_id, states, fields, scans, strict=True
    ):
        for antenna1, antenna2 in ((0, 1), (0, 2), (1, 2)):
            rows_time.append(float(time))
            rows_ant1.append(antenna1)
            rows_ant2.append(antenna2)
            rows_ddid.append(int(ddid))
            rows_state.append(int(state))
            rows_field.append(int(field))
            rows_scan.append(int(scan))
    if extra_unmatched is not None:
        rows_time.append(float(extra_unmatched))
        rows_ant1.append(0)
        rows_ant2.append(1)
        rows_ddid.append(int(data_desc_id[0]))
        rows_state.append(0)
        rows_field.append(0)
        rows_scan.append(99)
    return HolographyMainMetadata(
        time_s=np.asarray(rows_time, dtype=np.float64),
        interval_s=np.full(len(rows_time), 10.0, dtype=np.float64),
        antenna1=np.asarray(rows_ant1, dtype=np.int32),
        antenna2=np.asarray(rows_ant2, dtype=np.int32),
        field_id=np.asarray(rows_field, dtype=np.int32),
        state_id=np.asarray(rows_state, dtype=np.int32),
        scan_number=np.asarray(rows_scan, dtype=np.int32),
        data_desc_id=np.asarray(rows_ddid, dtype=np.int32),
        field_names=("HOLORASTER", "J0542+4951"),
        state_modes=("MAP_ANTENNA_SURFACE#ON_SOURCE", "OBSERVE_TARGET"),
        antenna_names=("ea00", "ea01", "ea02"),
        phase_centre_rad=_PHASE,
        main_row_count=10_000,
    )


def test_path_identities_stay_separate_from_observation_project() -> None:
    path = "/media/stephen/astro/vla/extracted/THOL0001.sb31628704.eb31629959.57401.169024456016.ms"
    identities = identities_from_holography_path(path)
    assert identities["project"] == "THOL0001"
    assert identities["scheduling_block"] == "sb31628704"
    assert identities["execution_block"] == "eb31629959"
    provenance = holography_execution_provenance(path, observation_project="uid://evla/pdb/4538748")
    assert provenance["project"] == "THOL0001"
    assert provenance["observation_project"] == "uid://evla/pdb/4538748"
    restored = inventory_from_dict(
        {
            "project": "THOL0001",
            "scheduling_block": "sb31628704",
            "execution_block": "eb31629959",
            "source_name": "3C147",
            "observation_utc": "2016-01-14T04:03:28Z",
            "path": path,
            "archive_sha256": "abc",
            "table_presence": {"POINTING": True},
            "correlations": ["RR"],
            "antenna_count": 1,
            "observation_project": "uid://evla/pdb/4538748",
        }
    )
    assert restored.project == "THOL0001"
    assert restored.observation_project == "uid://evla/pdb/4538748"


def test_unique_times_are_sorted_within_ddid_not_globally() -> None:
    time_s = np.array([30.0, 10.0, 20.0, 20.0, 10.0, 30.0], dtype=np.float64)
    group_id = np.array([5, 4, 4, 5, 5, 4], dtype=np.int32)
    times, groups = unique_times_within_groups(time_s, group_id)
    assert groups.tolist() == [4, 4, 4, 5, 5, 5]
    assert times.tolist() == [10.0, 20.0, 30.0, 10.0, 20.0, 30.0]
    assert not np.all(np.diff(times) > 0.0)


def test_native_channels_select_4p564_and_4p692() -> None:
    spw4 = 4.500e9 + 2.0e6 * np.arange(64)
    spw5 = 4.628e9 + 2.0e6 * np.arange(64)
    selected = select_native_channels(
        {3: np.array([3.988e9]), 4: spw4, 5: spw5},
        ((3, 3, 0), (4, 4, 0), (5, 5, 0)),
    )
    assert [item.data_desc_id for item in selected] == [4, 5]
    assert selected[0].channel == 32
    assert selected[1].channel == 32
    assert selected[0].frequency_hz == pytest.approx(4.564e9)
    assert selected[1].frequency_hz == pytest.approx(4.692e9)


def test_on_source_is_not_used_for_reconstruction_validity() -> None:
    table = load_synthetic_holography_pointing_table()
    object.__setattr__(table, "on_source", np.zeros(table.time_s.size, dtype=bool))
    times = np.unique(table.time_s)
    with_flag = resolve_antenna_pointing(
        table,
        times,
        np.array([0, 1, 2], dtype=np.int32),
        selected_column="POINTING_OFFSET",
        phase_centre_rad=_PHASE,
        use_on_source=True,
    )
    ignored = resolve_antenna_pointing(
        table,
        times,
        np.array([0, 1, 2], dtype=np.int32),
        selected_column="POINTING_OFFSET",
        phase_centre_rad=_PHASE,
        use_on_source=False,
        settled_jump_arcmin=8.0,
    )
    assert any("ON_SOURCE was not used" in note for note in ignored.notes)
    assert bool(np.any(ignored.settled))
    assert with_flag.notes != ignored.notes


def test_map_antenna_surface_dwell_and_transition_guards() -> None:
    table = synthetic_memo195_pointing_table(phase_centre_rad=_PHASE)
    unique = np.unique(table.time_s)
    # Interleave two DDIDs and shuffle MAIN order.
    times = np.repeat(unique[::-1], 2)
    data_desc = np.tile(np.array([4, 5], dtype=np.int32), unique.size)
    state = np.zeros(times.size, dtype=np.int32)
    # The Memo 195 fixture parks a mid-slew sample at unique[4]; STATE, not ON_SOURCE,
    # must keep it out of the settled raster.
    state[times == unique[4]] = 1
    main = _main_from_times(times, data_desc_id=data_desc, state_id=state)
    audit = reconstruct_holography_pointing(
        table,
        main,
        data_desc_ids=(4, 5),
        selected_column="POINTING_OFFSET",
        settled_jump_arcmin=8.0,
        transition_guard_samples=1,
        provenance=holography_execution_provenance(
            "THOL0001.sb31628704.eb31629959.ms",
            observation_project="uid://evla/pdb/4538748",
        ),
        native_channels=select_native_channels(
            {4: 4.500e9 + 2.0e6 * np.arange(64), 5: 4.628e9 + 2.0e6 * np.arange(64)},
            ((4, 4, 0), (5, 5, 0)),
        ),
    )
    assert audit.provenance["project"] == "THOL0001"
    assert audit.provenance["observation_project"] == "uid://evla/pdb/4538748"
    assert audit.moving_antenna_ids == (1, 2)
    assert audit.reference_antenna_ids == (0,)
    assert audit.moving_antenna_names == ("ea01", "ea02")
    assert audit.reference_antenna_names == ("ea00",)
    assert audit.memo195_dense_status == "pass"
    assert audit.memo195_sparse_status == "pass"
    assert {cell.family for cell in audit.raster_cells} >= {"dense", "sparse"}
    assert audit.unique_times_by_ddid[4] == unique.size + 1
    assert audit.unique_times_by_ddid[5] == unique.size
    assert audit.sample_reasons[HolographyRowReason.MISSING_POINTING.value] >= 1
    assert audit.sample_reasons[HolographyRowReason.NOT_MAP_ANTENNA_SURFACE.value] >= 1
    assert audit.row_reasons[HolographyRowReason.NOT_MOVING_REFERENCE.value] >= 1
    assert audit.row_reasons[HolographyRowReason.MISSING_POINTING.value] >= 1
    assert audit.composition["field"] == {"HOLORASTER": audit.selected_main_rows}
    assert "MAP_ANTENNA_SURFACE#ON_SOURCE" in audit.composition["state"]
    assert set(audit.composition["data_desc_id"]) == {"4", "5"}
    assert audit.time_alignment.n_matched > 0
    assert audit.offset_range_arcmin["radius_max"] > 0.0
    payload = pointing_reconstruction_audit_as_dict(audit)
    restored = json.loads(json.dumps(payload))
    assert restored["selected_column"] == "POINTING_OFFSET"
    assert "DATA" not in json.dumps(restored)
    text = format_pointing_reconstruction_audit(audit)
    assert "OBSERVATION.PROJECT uid://evla/pdb/4538748" in text
    assert "moving antennas ea01, ea02" in text


def test_non_holoraster_and_other_ddids_are_excluded() -> None:
    table = synthetic_memo195_pointing_table(phase_centre_rad=_PHASE)
    unique = np.unique(table.time_s)
    times = np.concatenate((unique, unique[:1]))
    data_desc = np.concatenate(
        (np.full(unique.size, 4, dtype=np.int32), np.array([9], dtype=np.int32))
    )
    field = np.concatenate((np.zeros(unique.size, dtype=np.int32), np.array([1], dtype=np.int32)))
    main = _main_from_times(times, data_desc_id=data_desc, field_id=field, extra_unmatched=None)
    audit = reconstruct_holography_pointing(
        table,
        main,
        data_desc_ids=(4,),
        selected_column="POINTING_OFFSET",
        settled_jump_arcmin=8.0,
    )
    assert audit.composition["field"] == {"HOLORASTER": audit.selected_main_rows}
    assert audit.composition["data_desc_id"] == {"4": audit.selected_main_rows}


def test_map_antenna_surface_token() -> None:
    assert is_map_antenna_surface("MAP_ANTENNA_SURFACE#ON_SOURCE") is True
    assert is_map_antenna_surface("OBSERVE_TARGET#ON_SOURCE") is False


def test_versioned_real_ms_fixture_has_no_visibilities() -> None:
    root = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "sl1mjax"
        / "data"
        / "holography_thol0001_lower_c"
    )
    fixture = json.loads((root / "metadata_fixture.json").read_text())
    summary = json.loads((root / "occupancy_summary.json").read_text())
    assert fixture["kind"] == "thol0001_lower_c_metadata"
    assert fixture["visibilities"] is None
    assert fixture["overlap_proof"]["passed"] is True
    assert fixture["time_alignment"]["n_outside_half_interval"] == 0
    assert fixture["time_alignment"]["max_abs_fraction_of_half_interval"] < 1.0
    assert summary["combined"]["n_dense_family"] == 191
    assert summary["combined"]["n_sparse_family"] == 606
    assert (root / "occupancy_pass_1.png").is_file()
    assert (root / "transition_ea04.png").is_file()
    vis = json.loads((root / "visibility_summary.json").read_text())
    assert vis["n_channels"] == 64
    assert vis["channel_32_only"] is False
    assert vis["source_ms_unchanged"] is True
    assert vis["fields"]["C147-*"]["looks_like_intended_leakage_scan"] is True
    occupancy = json.loads((root / "occupancy_summary.json").read_text())
    assert occupancy["snap_pass2_to_memo_lattice"] is False
    assert occupancy["pass2_geometry"] == "complete measured raster, not Memo-lattice raster"


def test_memo195_circular_grid_is_smaller_than_the_square() -> None:
    square = memo195_expected_offsets(17, 1.72, circular=False)
    circular = memo195_expected_offsets(17, 1.72, circular=True)
    assert square.shape[0] == 289
    assert circular.shape[0] < 289
    assert circular.shape[0] == 197


def test_scan_passes_split_on_a_gap() -> None:
    first = np.arange(18, 51, 2, dtype=np.int32)
    second = np.arange(57, 102, 2, dtype=np.int32)
    grouped = scan_passes(np.concatenate((first, second)))
    assert grouped == (tuple(first.tolist()), tuple(second.tolist()))


def test_alignment_stays_inside_half_interval_and_records_no_query_overlaps() -> None:
    table = synthetic_memo195_pointing_table(phase_centre_rad=_PHASE)
    unique = np.unique(table.time_s)
    times = np.repeat(unique, 2)
    data_desc = np.tile(np.array([4, 5], dtype=np.int32), unique.size)
    main = _main_from_times(times, data_desc_id=data_desc, extra_unmatched=None)
    audit = reconstruct_holography_pointing(
        table,
        main,
        data_desc_ids=(4, 5),
        selected_column="POINTING_OFFSET",
        settled_jump_arcmin=8.0,
    )
    assert audit.time_alignment.max_abs_fraction_of_half_interval < 1.0
    assert audit.time_alignment.n_outside_half_interval == 0
    assert audit.overlap_proof.n_visibility_times_in_multiple_intervals == 0
    assert audit.overlap_proof.passed is True
    assert pointing_table_overlap_pairs(table) == 0


def test_d_solve_excludes_c147_offset_and_keeps_onaxis_d_field() -> None:
    policy = calibration_field_policy()
    assert D_CODE_FIELD_ID in policy.leakage_field_ids
    assert all(field_may_enter_d_solve(field, "J0542+4951") for field in (0, 9))
    assert all(not field_may_enter_d_solve(field, "C147-N") for field in C147_OFFSET_FIELD_IDS)
    assert PASS2_MEASURED_RASTER_NOTE in policy.notes
    assert policy.holoraster_gain_baselines == "reference_reference"


def test_empirical_variance_ignores_nominal_weight() -> None:
    values = np.array([1 + 0j, 1.1 + 0j, 0.9 + 0j, 5 + 0j, 5.2 + 0j], dtype=np.complex128)
    groups = np.array([0, 0, 0, 1, 1], dtype=np.int32)
    variance = empirical_variance_from_repeat_scatter(values, groups)
    assert np.isfinite(variance)
    assert variance < 1.0


def test_product_manifest_records_parang_calwt_and_no_offset_d(tmp_path: Path) -> None:
    table = tmp_path / "K0.cal"
    table.write_text("k\n")
    payload = product_manifest(
        name="diagonal",
        tables={"K0": table},
        model={"3C147": "Perley-Butler 2017"},
        parang=False,
        calwt=False,
        flag_version="sl1mjax_diagonal_input",
        notes=("test",),
    )
    assert payload["parang"] is False
    assert payload["calwt"] is False
    assert payload["c147_offset_used_for_d"] is False
    assert payload["snap_pass2_to_memo_lattice"] is False
    assert payload["tables"]["K0"]["sha256"]


def test_occupancy_explanation_names_circular_and_relabel() -> None:
    from sl1mjax.holography_pointing_maps import (
        RasterOccupancyReport,
        RasterPassOccupancy,
        explain_occupancy,
    )

    dense_pass = RasterPassOccupancy(
        "pass_1",
        (18, 50),
        10,
        10,
        275,
        275,
        0,
        275,
        7,
        196,
        7,
        120,
        420,
    )
    sparse_pass = RasterPassOccupancy(
        "pass_2",
        (57, 101),
        10,
        10,
        529,
        0,
        529,
        9,
        289,
        5,
        259,
        120,
        460,
    )
    combined = RasterPassOccupancy(
        "combined",
        (18, 101),
        20,
        20,
        797,
        191,
        606,
        277,
        289,
        197,
        259,
        120,
        540,
    )
    text = explain_occupancy(
        RasterOccupancyReport(
            (dense_pass, sparse_pass),
            combined,
            289,
            529,
            197,
            377,
            "",
        )
    )
    assert "pass_1 is dense-only" in text
    assert "exactly 23²=529" in text
    assert "191/606" in text
    assert "circular 17×17" in text
    assert "complete measured raster" in text


def test_group_factorization_and_moving_scan_lookup() -> None:
    scan = np.array([18, 18, 57, 57], dtype=np.int32)
    field = np.array([10, 10, 10, 10], dtype=np.int32)
    state = np.zeros(4, dtype=np.int32)
    ddid = np.array([4, 5, 4, 5], dtype=np.int32)
    keys, inverse = _factorize_groups(scan, field, state, ddid)
    assert keys.shape[0] == 4
    assert inverse.tolist() == [0, 1, 2, 3]

    class _Roles:
        unique_time_s = np.array([10.0, 20.0], dtype=np.float64)
        antenna_id = np.array([0, 1, 2], dtype=np.int32)
        role = np.array([["reference", "moving", "moving"], ["reference", "moving", "reference"]])

    found = _moving_by_scan(
        _Roles(),
        np.array([18, 18, 57], dtype=np.int32),
        np.array([10.0, 10.0, 20.0], dtype=np.float64),
        ("ea00", "ea01", "ea02"),
    )
    assert found["18"] == ["ea01", "ea02"]
    assert found["57"] == ["ea01"]


def test_commissioning_fields_keep_3c147_d_scan_holoraster_and_3c286() -> None:
    names = (
        "J0542+4951",
        "C147-N",
        "C147-NW",
        "C147-W",
        "C147-SW",
        "C147-S",
        "C147-SE",
        "C147-E",
        "C147-NE",
        "J0542+4951",
        "HOLORASTER",
        "J1331+3030",
        "IGNORED",
    )
    assert select_commissioning_fields(names) == tuple(range(12))


def test_pointing_bundle_writes_versioned_outputs(tmp_path: Path) -> None:
    table = synthetic_memo195_pointing_table(phase_centre_rad=_PHASE)
    unique = np.unique(table.time_s)
    times = np.repeat(unique, 2)
    data_desc = np.tile(np.array([4, 5], dtype=np.int32), unique.size)
    main = _main_from_times(times, data_desc_id=data_desc, extra_unmatched=None)
    audit = reconstruct_holography_pointing(
        table,
        main,
        data_desc_ids=(4, 5),
        selected_column="POINTING_OFFSET",
        settled_jump_arcmin=8.0,
    )
    written = write_pointing_audit_bundle(audit, tmp_path / "bundle")
    assert (tmp_path / "bundle" / "pointing_audit.json").is_file()
    assert (tmp_path / "bundle" / "metadata_fixture.json").is_file()
    assert "visibilities" in json.loads((tmp_path / "bundle" / "metadata_fixture.json").read_text())
    fixture = json.loads((tmp_path / "bundle" / "metadata_fixture.json").read_text())
    assert fixture["visibilities"] is None
    assert written["occupancy_combined"].is_file()
    assert (tmp_path / "bundle" / "occupancy_summary.json").is_file()
