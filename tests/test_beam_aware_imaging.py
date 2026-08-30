from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from sl1mjax.beam_aware_imaging import (
    BEAM_AWARE_IMAGING_MANIFEST_ID,
    ComponentFamily,
    SkyBasisType,
    SkyComponent,
    VoltageIntegrationMode,
    family_from_component_name,
    interpret_fixed_sky_transfer,
    load_beam_aware_imaging_manifest,
    load_fixed_sky_transfer_report,
    overlap_policy_for_family,
    point_centre_atoms,
    prepare_voltage_sky,
    reproduce_fixed_sky_transfer_totals,
    sky_table_from_checkpoint,
    sky_table_from_mosaic_components,
    sky_table_from_records,
    sky_table_to_records,
    validate_phase0_baseline,
    width_arcsec_label,
)
from sl1mjax.beam_conventions import BeamCalibrationState
from sl1mjax.beam_operator import (
    BeamOperatorConfig,
    SkyStokesPlanes,
    predict_voltage_beam,
)
from sl1mjax.cassbeam_beam import voltage_beam_for_mode
from sl1mjax.composite import MosaicPointComponent, MosaicQuadtreeComponent
from sl1mjax.data.canonical import VisibilityBlock
from sl1mjax.polarization import Correlation, ReceptorBasis
from sl1mjax.quadtree import QuadtreeLeaf, quadtree_sky_from_regular_grid
from sl1mjax.voltage_beam import AnalyticAiryVoltageBeam
from sl1mjax.voltage_operator_jax import predict_voltage_beam_jax

SCRIPT = Path(__file__).parents[1] / "scripts" / "diagnose_3c391_voltage_beam_transfer.py"
_ANTENNA_POSITION_M = np.array(
    [
        [-1_601_162.0, -5_042_003.0, 3_553_983.0],
        [-1_601_100.0, -5_042_100.0, 3_553_900.0],
        [-1_601_200.0, -5_042_190.0, 3_554_000.0],
    ]
)


def _transfer_module():
    spec = importlib.util.spec_from_file_location(
        "diagnose_3c391_voltage_beam_transfer",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _mixed_width_components() -> tuple[
    MosaicQuadtreeComponent, MosaicQuadtreeComponent, MosaicPointComponent
]:
    root_width = np.deg2rad(16.0 / 3600.0)
    coarse_width = np.deg2rad(60.0 / 3600.0)
    central = quadtree_sky_from_regular_grid(2, root_width, [0.4, 0.0, 0.3, 0.2])
    split = central.split(QuadtreeLeaf(0, 0, 0), [0.1, 0.1, 0.1, 0.1])
    fine_parent = QuadtreeLeaf(1, 0, 0)
    refined = split.split(fine_parent)
    coarse = quadtree_sky_from_regular_grid(2, coarse_width, [0.05, 0.0, 0.07, 0.08])
    catalogue = MosaicPointComponent(
        "catalogue",
        np.array([0.01, -0.02, 0.03]),
        np.array([0.0, 0.01, -0.01]),
        np.array([1.2, 0.0, 0.4]),
    )
    return (
        MosaicQuadtreeComponent("central", refined.topology, refined.flux),
        MosaicQuadtreeComponent("coarse", coarse.topology, coarse.flux),
        catalogue,
    )


def test_phase0_manifest_matches_live_scientific_status() -> None:
    payload = load_beam_aware_imaging_manifest()
    assert payload["manifest_id"] == BEAM_AWARE_IMAGING_MANIFEST_ID
    status = payload["scientific_status"]
    assert status["default_imaging_beam"] == "static_airy"
    assert status["full_jones_frozen"] is False
    assert status["transfer_operator_discarded_finite_widths"] is True
    assert status["transfer_ranking_is_scientific_beam_selection"] is False
    assert payload["conventions"]["calibration_state"] == (
        BeamCalibrationState.CASA_PARANG_TRUE.value
    )
    gate = validate_phase0_baseline(payload)
    assert gate["accepted"]
    assert gate["full_jones_factory_refused"]


def test_phase0_full_jones_factory_still_refuses() -> None:
    with pytest.raises(ValueError, match="not frozen"):
        voltage_beam_for_mode("full_jones")
    validate_phase0_baseline()


def test_phase0_reproduces_pinned_fixed_sky_totals() -> None:
    report = load_fixed_sky_transfer_report()
    interpretation = reproduce_fixed_sky_transfer_totals()
    means = interpretation["mean_held_out_loss"]
    assert interpretation["ranking_best_first"] == [
        "static_scalar",
        "full_jones_unfrozen",
        "diagonal_copolar",
        "streamed_scalar",
    ]
    assert interpretation["no_detailed_beam_beats_airy"] is True
    assert interpretation["leakage_matters"] is True
    assert interpretation["cross_hand_in_data"] is False
    assert interpretation["do_not_freeze_full_jones"] is True
    assert means["static_scalar"] == pytest.approx(279977.1190987142)
    assert means["diagonal_copolar"] == pytest.approx(287177.1592272951)
    assert means["full_jones_unfrozen"] == pytest.approx(287177.0277064758)
    assert means["streamed_scalar"] == pytest.approx(290475.0001851497)
    assert means["diagonal_copolar"] - means["full_jones_unfrozen"] == pytest.approx(
        0.1315208193
    )
    assert report["provenance"]["sky_atoms_positive"] == 5869
    assert report["provenance"]["sky_flux_jy"] == pytest.approx(25.533968640130965)
    assert report["pointings"]["C1"]["totals"]["diagonal_copolar"] < report["pointings"][
        "C1"
    ]["totals"]["static_scalar"]
    assert report["pointings"]["C4"]["totals"]["diagonal_copolar"] > report["pointings"][
        "C4"
    ]["totals"]["static_scalar"]
    provenance = report["provenance"]
    assert provenance["command"]
    assert provenance["revision_note"]
    assert provenance["script_sha256"]
    assert provenance["inputs"]["sky_checkpoint"]["sha256"]
    assert provenance["inputs"]["sky_protocol"]["sha256"]
    assert len(provenance["inputs"]["native_holdouts"]["zarr_tree_sha256"]) == 7


def test_phase0_refuses_a_silent_scientific_upgrade() -> None:
    payload = load_beam_aware_imaging_manifest()
    upgraded = {
        **payload,
        "scientific_status": {
            **payload["scientific_status"],
            "full_jones_frozen": True,
        },
    }
    with pytest.raises(ValueError, match="full_jones_frozen"):
        validate_phase0_baseline(upgraded)


def test_phase0_validates_explicit_unfrozen_gate_pins() -> None:
    payload = load_beam_aware_imaging_manifest()
    upgraded = {
        **payload,
        "unfrozen_or_unimplemented_gates": {
            **payload["unfrozen_or_unimplemented_gates"],
            "power_oracle_is_frozen": True,
        },
    }
    with pytest.raises(ValueError, match="power_oracle_is_frozen"):
        validate_phase0_baseline(upgraded)


def test_phase0_rr_ll_only_data_have_no_cross_hand_score() -> None:
    module = _transfer_module()
    dummy = np.ones((4, 2, 2), dtype=np.complex128)
    block = VisibilityBlock(
        uvw_m=np.ones((4, 3)),
        frequency_hz=np.array([4.564e9, 4.692e9]),
        visibility=dummy,
        weight=np.ones_like(dummy),
        flag=np.zeros(dummy.shape, dtype=bool),
        time_s=np.linspace(1.0e9, 1.0e9 + 300.0, 4),
        antenna1=np.zeros(4, dtype=np.int32),
        antenna2=np.ones(4, dtype=np.int32),
        correlations=(Correlation.RR, Correlation.LL),
        receptor_basis=ReceptorBasis.CIRCULAR,
        phase_centre_rad=(0.3, -0.01),
    )
    scores = module.score_prediction(
        block,
        0.5 * dummy,
        antenna_position_m=_ANTENNA_POSITION_M,
        pointing_radius_arcmin=1.5,
        leakage_atom_fraction=0.8,
    )
    assert scores["correlations"]["RR"]["in_data"] is True
    assert scores["correlations"]["LL"]["in_data"] is True
    assert scores["correlations"]["RL"]["in_data"] is False
    assert scores["correlations"]["LR"]["held_out_loss"] is None
    interpretation = interpret_fixed_sky_transfer(
        {"pointings": {"C1": {"beams": {"static_scalar": scores}}}}
    )
    assert interpretation["cross_hand_in_data"] is False


def test_phase0_jax_point_operator_still_matches_numpy() -> None:
    l_rad = np.array([0.0, np.sin(np.deg2rad(0.03))])
    m_rad = np.zeros(2)
    flux = np.array([1.1, 0.3])
    frequency = np.array([4.536e9, 4.662e9])
    time_s = np.array([5.0e9, 5.0e9, 5.0e9 + 1800.0])
    dummy = np.zeros((3, 2, 2), dtype=np.complex128)
    block = VisibilityBlock(
        uvw_m=np.array([[20.0, -8.0, 1.0], [-11.0, 14.0, 2.0], [6.0, 9.0, -3.0]]),
        frequency_hz=frequency,
        visibility=dummy,
        weight=np.ones_like(dummy, dtype=np.float64),
        flag=np.zeros(dummy.shape, dtype=bool),
        time_s=time_s,
        antenna1=np.array([0, 0, 1], dtype=np.int32),
        antenna2=np.array([1, 2, 2], dtype=np.int32),
        correlations=(Correlation.RR, Correlation.LL),
        receptor_basis=ReceptorBasis.CIRCULAR,
        phase_centre_rad=(np.deg2rad(282.35), np.deg2rad(-0.93)),
    )
    kwargs = dict(
        block=block,
        l_rad=l_rad,
        m_rad=m_rad,
        sky=SkyStokesPlanes(stokes_i=flux),
        beam=AnalyticAiryVoltageBeam(),
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
        config=BeamOperatorConfig(visibility_chunk_size=2, pixel_chunk_size=1),
    )
    numpy_result = predict_voltage_beam(**kwargs)
    jax_result = predict_voltage_beam_jax(**kwargs)
    np.testing.assert_allclose(
        jax_result.visibility, numpy_result.visibility, rtol=1e-9, atol=1e-11
    )


def test_phase1_conversion_preserves_widths_types_and_flux() -> None:
    components = _mixed_width_components()
    table = sky_table_from_mosaic_components(
        components,
        mosaic_phase_centre_rad=(0.3, -0.01),
        expected_component_names=("central", "coarse", "catalogue", "missing"),
    )
    assert table.report.missing_component_names == ("missing",)
    assert table.report.discarded_finite_widths is False
    assert table.report.dropped_zero_flux_count == 0
    input_flux = float(sum(float(np.asarray(item.flux).sum()) for item in components))
    assert table.flux_jy() == pytest.approx(input_flux)
    counts = table.report.count_by_width_arcsec
    assert counts["16"] == 3
    assert counts["8"] == 3
    assert counts["4"] == 4
    assert counts["60"] == 4
    assert counts["delta"] == 3
    assert table.report.count_by_family["central_tree"] == 10
    assert table.report.count_by_family["coarse_field"] == 4
    assert table.report.count_by_family["catalogue"] == 3
    catalogue = [item for item in table.components if item.family is ComponentFamily.CATALOGUE]
    assert all(item.basis_type is SkyBasisType.DELTA for item in catalogue)
    assert all(item.width_rad == 0.0 for item in catalogue)
    np.testing.assert_allclose([item.l_rad for item in catalogue], [0.01, -0.02, 0.03])
    np.testing.assert_allclose([item.m_rad for item in catalogue], [0.0, 0.01, -0.01])
    squares = [item for item in table.components if item.basis_type is SkyBasisType.UNIFORM_SQUARE]
    assert all(item.width_rad > 0.0 for item in squares)
    assert overlap_policy_for_family("coarse_field").value == "overlapping_historical"


def test_phase1_round_trip_records() -> None:
    table = sky_table_from_mosaic_components(_mixed_width_components())
    restored = sky_table_from_records(
        sky_table_to_records(table),
        mosaic_phase_centre_rad=table.mosaic_phase_centre_rad,
    )
    assert [item.component_id for item in restored.components] == [
        item.component_id for item in table.components
    ]
    np.testing.assert_allclose(
        [item.l_rad for item in restored.components],
        [item.l_rad for item in table.components],
    )
    np.testing.assert_allclose(
        [item.width_rad for item in restored.components],
        [item.width_rad for item in table.components],
    )
    np.testing.assert_allclose(
        [item.stokes_i_jy for item in restored.components],
        [item.stokes_i_jy for item in table.components],
    )
    assert [item.basis_type for item in restored.components] == [
        item.basis_type for item in table.components
    ]


def test_phase1_point_centre_mode_keeps_widths_and_drops_zeros() -> None:
    table = sky_table_from_mosaic_components(_mixed_width_components())
    atoms = prepare_voltage_sky(table, mode=VoltageIntegrationMode.POINT_CENTRE)
    assert atoms.mode is VoltageIntegrationMode.POINT_CENTRE
    assert atoms.dropped_zero_flux_count == 3
    assert np.all(atoms.stokes_i_jy > 0.0)
    assert "uniform_square" in atoms.basis_type
    assert "delta" in atoms.basis_type
    assert np.any(atoms.width_rad > 0.0)
    catalogue = [item for item in table.components if item.family is ComponentFamily.CATALOGUE]
    packed = point_centre_atoms(table, include_zero_flux=True)
    np.testing.assert_allclose(
        packed.l_rad[-3:],
        [item.l_rad for item in catalogue],
    )


def test_phase1_rejects_unknown_basis_and_invalid_widths() -> None:
    with pytest.raises(ValueError, match="unknown mosaic component name"):
        family_from_component_name("unmapped")
    with pytest.raises(ValueError):
        SkyBasisType("hexagonal")
    with pytest.raises(ValueError, match="width_rad"):
        SkyComponent(
            "bad-delta",
            ComponentFamily.CATALOGUE,
            SkyBasisType.DELTA,
            0.0,
            0.0,
            1.0,
            width_rad=1.0e-4,
        )
    with pytest.raises(ValueError, match="positive width_rad"):
        SkyComponent(
            "bad-square",
            ComponentFamily.CENTRAL_TREE,
            SkyBasisType.UNIFORM_SQUARE,
            0.0,
            0.0,
            1.0,
            width_rad=0.0,
        )


def test_phase1_rejects_active_parent_and_child_in_one_family() -> None:
    width = np.deg2rad(16.0 / 3600.0)
    records = [
        {
            "component_id": "central_tree:0:0:0",
            "family": "central_tree",
            "basis_type": "uniform_square",
            "l_rad": 0.0,
            "m_rad": 0.0,
            "stokes_i_jy": 1.0,
            "width_rad": width,
            "level": 0,
            "iy": 0,
            "ix": 0,
            "parent_id": None,
            "active": True,
            "splitting_permitted": True,
            "provenance": {},
        },
        {
            "component_id": "central_tree:1:0:0",
            "family": "central_tree",
            "basis_type": "uniform_square",
            "l_rad": width / 4.0,
            "m_rad": -width / 4.0,
            "stokes_i_jy": 0.25,
            "width_rad": width / 2.0,
            "level": 1,
            "iy": 0,
            "ix": 0,
            "parent_id": "central_tree:0:0:0",
            "active": True,
            "splitting_permitted": True,
            "provenance": {},
        },
    ]
    with pytest.raises(ValueError, match="not prefix-free"):
        sky_table_from_records(records)
    records[0]["provenance"] = {"mosaic_name": "central"}
    records[1]["provenance"] = {"mosaic_name": "guard"}
    records[0]["component_id"] = "central_tree:central:0:0:0"
    records[1]["component_id"] = "central_tree:guard:1:0:0"
    records[1]["parent_id"] = "central_tree:guard:0:0:0"
    separate = sky_table_from_records(records)
    assert len(separate.components) == 2
    child_only = sky_table_from_mosaic_components(_mixed_width_components()[:1])
    assert all(
        item.level is None or item.level > 0 or item.parent_id is None
        for item in child_only.components
    )


def test_phase1_transfer_script_uses_the_table_without_dropping_widths() -> None:
    module = _transfer_module()
    components = _mixed_width_components()
    l_rad, m_rad, flux = module.flatten_positive_sky(components)
    table = sky_table_from_mosaic_components(components)
    atoms = prepare_voltage_sky(table)
    np.testing.assert_allclose(l_rad, atoms.l_rad)
    np.testing.assert_allclose(m_rad, atoms.m_rad)
    np.testing.assert_allclose(flux, atoms.stokes_i_jy)
    assert np.any(atoms.width_rad > 0.0)
    assert width_arcsec_label(np.deg2rad(16.0 / 3600.0)) == "16"


def test_phase1_point_centre_mode_drops_nonpositive_flux() -> None:
    table = sky_table_from_mosaic_components(
        (
            MosaicPointComponent(
                "catalogue",
                np.array([0.0, 0.01, -0.02]),
                np.array([0.0, 0.0, 0.01]),
                np.array([1.2, 0.0, -0.4]),
            ),
        )
    )
    atoms = prepare_voltage_sky(table)
    np.testing.assert_allclose(atoms.l_rad, [0.0])
    np.testing.assert_allclose(atoms.stokes_i_jy, [1.2])
    assert atoms.dropped_zero_flux_count == 2


def test_phase1_component_ids_include_source_dictionary_name() -> None:
    first = MosaicPointComponent(
        "nvss",
        np.array([0.01]),
        np.array([0.0]),
        np.array([1.0]),
    )
    second = MosaicPointComponent(
        "vlass",
        np.array([0.02]),
        np.array([0.01]),
        np.array([0.5]),
    )
    table = sky_table_from_mosaic_components(
        (first, second),
        family_by_name={"nvss": "catalogue", "vlass": "catalogue"},
    )
    identifiers = [item.component_id for item in table.components]
    assert identifiers == ["catalogue:nvss:delta:0", "catalogue:vlass:delta:0"]
    mixed = _mixed_width_components()
    central = sky_table_from_mosaic_components(mixed[:1])
    assert all(
        item.component_id.startswith("central_tree:central:")
        for item in central.components
    )


def _sealed_checkpoint_paths() -> tuple[Path, Path]:
    root = Path(__file__).parents[1]
    return (
        root / "outputs/3c391_composite_catalogue_stage3/protocol.json",
        root / "outputs/3c391_recovery_policy_fit_zero/sealed_active_only.npz",
    )


@pytest.mark.skipif(
    not all(path.is_file() for path in _sealed_checkpoint_paths()),
    reason="sealed 3C391 checkpoint is not present in this checkout",
)
def test_phase1_sealed_checkpoint_matches_documented_counts() -> None:
    protocol_path, checkpoint = _sealed_checkpoint_paths()
    native = protocol_path.parents[1] / "3c391_native_averaging_ablation" / "native_C1.zarr"
    if native.is_dir():
        from sl1mjax.data.canonical import read_dataset

        phase_centre = read_dataset(native).blocks[0].phase_centre_rad
    else:
        phase_centre = (np.deg2rad(282.35), np.deg2rad(-0.93))
    table = sky_table_from_checkpoint(checkpoint, protocol_path, phase_centre)
    atoms = prepare_voltage_sky(table)
    pinned = load_beam_aware_imaging_manifest()["sky_checkpoint"]
    assert len(table.components) == 15635
    assert atoms.l_rad.size == pinned["transfer_positive_atoms"]
    assert atoms.stokes_i_jy.sum() == pytest.approx(pinned["transfer_positive_flux_jy"])
    assert table.report.count_by_width_arcsec["4"] == pinned["central"]["topology_leaves"]["4"]
    assert table.report.count_by_width_arcsec["8"] == pinned["central"]["topology_leaves"]["8"]
    assert table.report.count_by_width_arcsec["16"] == pinned["central"]["topology_leaves"][
        "16"
    ]
    assert table.report.count_by_width_arcsec["60"] == 64 * 64
    assert table.report.count_by_width_arcsec["delta"] == 3
    assert np.all(atoms.stokes_i_jy > 0.0)
