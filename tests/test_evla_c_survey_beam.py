from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from sl1mjax.cassbeam_beam import CASSBEAM_CBAND_MODEL_ID, voltage_beam_for_mode
from sl1mjax.cassbeam_highres import HighresCassbeamCatalog
from sl1mjax.data.canonical import VisibilityBlock
from sl1mjax.evla_c_diagonal_survey import CATALOG_ID
from sl1mjax.evla_c_survey_beam import (
    IMAGING_NODE_MHZ,
    NATIVE_3C391_MHZ,
    OPT_IN_SURVEY_BEAM,
    EvlaCSurveyVoltageBeam,
    survey_catalog_digest,
    survey_catalog_record,
    voltage_beam_for_survey_catalog,
)
from sl1mjax.evla_c_survey_compare import SURVEY_MODEL_ID
from sl1mjax.inference import InferenceConfig
from sl1mjax.integration_planner import IntegrationTolerance
from sl1mjax.phase6_protocol import (
    phase6_folds,
    write_reconstruction_products,
    write_smoke_reconstruction_products,
)
from sl1mjax.polarization import Correlation, ReceptorBasis
from sl1mjax.voltage_beam import BeamCoordinates
from sl1mjax.voltage_reconstruction import (
    PRODUCTION_STOKES_I_BEAMS,
    VoltageReconstructionConfig,
    reconstruct_voltage_stokes_i,
    starting_central_table,
    stokes_i_beam,
)

_TEST_RASTER = (9, 9, 2, 2)
_FREQ_HZ = 4.536e9
_PHASE = (np.deg2rad(282.35), np.deg2rad(-0.93))
_ANTENNA_POSITION_M = np.array(
    [
        [-1_601_162.0, -5_042_003.0, 3_553_983.0],
        [-1_601_100.0, -5_042_100.0, 3_553_900.0],
        [-1_601_200.0, -5_042_190.0, 3_554_000.0],
        [-1_601_050.0, -5_042_200.0, 3_553_850.0],
    ]
)
_POINTINGS = ("C1", "C2", "C3", "C4", "C5", "C6", "C7")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_survey_catalog(root: Path, *, frequency_mhz: int = 4536) -> Path:
    (root / "reference").mkdir(parents=True)
    (root / "spw4").mkdir(parents=True)
    geom = root / "reference" / "vla_geom"
    geom.write_text("12.5\n", encoding="utf-8")
    stem = f"evla-cband-{frequency_mhz}-g1024-p32"
    params = root / "spw4" / f"{stem}.params"
    params.write_text(
        f"freq = {frequency_mhz / 1000.0:.3f}\ngridsize = 16\npixelsperbeam = 4\n",
        encoding="utf-8",
    )
    size = 9
    l_origin, m_origin = 3, 4
    native = np.zeros((size, size, 2, 2), dtype=np.complex128)
    for j in range(size):
        for i in range(size):
            dl = float(i - l_origin)
            dm = float(j - m_origin)
            fall = 0.01 * (dl * dl + dm * dm)
            native[j, i] = np.array(
                [[1.0 - fall, 0.0], [0.0, 1.0 - 0.8 * fall]],
                dtype=np.complex128,
            )
    rows = []
    for j in range(size):
        for i in range(size):
            plane = native[j, i]
            rows.append(
                [
                    plane[0, 0].real,
                    plane[0, 0].imag,
                    plane[1, 0].real,
                    plane[1, 0].imag,
                    plane[0, 1].real,
                    plane[0, 1].imag,
                    plane[1, 1].real,
                    plane[1, 1].imag,
                ]
            )
    data = root / "spw4" / f"{stem}.jones.dat"
    np.savetxt(data, np.asarray(rows, dtype=np.float64))
    files = {
        "reference/vla_geom": _sha256(geom),
        f"spw4/{stem}.jones.dat": _sha256(data),
        f"spw4/{stem}.params": _sha256(params),
    }
    manifest = {
        "model_id": SURVEY_MODEL_ID,
        "raster": {"shape": [9, 9, 2, 2], "science_normalization": "inv(E(0)) @ E(s)"},
        "planes": [
            {
                "frequency_mhz": frequency_mhz,
                "shape": [9, 9, 2, 2],
                "native_columns": [
                    "Re_RR",
                    "Im_RR",
                    "Re_LR",
                    "Im_LR",
                    "Re_RL",
                    "Im_RL",
                    "Re_LL",
                    "Im_LL",
                ],
                "data": f"spw4/{stem}.jones.dat",
                "params": f"spw4/{stem}.params",
            }
        ],
        "files_sha256": files,
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def _block(*, phase_centre_rad=_PHASE) -> VisibilityBlock:
    uvw = np.array(
        [
            [30.0, 8.0, 2.0],
            [180.0, -40.0, -6.0],
            [900.0, 120.0, 15.0],
            [40.0, -700.0, -12.0],
            [220.0, 60.0, -4.0],
            [70.0, -90.0, 3.0],
        ],
        dtype=np.float64,
    )
    dummy = np.ones((uvw.shape[0], 1, 2), dtype=np.complex128) * 0.2
    return VisibilityBlock(
        uvw_m=uvw,
        frequency_hz=np.array([_FREQ_HZ]),
        visibility=dummy,
        weight=np.ones_like(dummy, dtype=np.float64),
        flag=np.zeros(dummy.shape, dtype=bool),
        time_s=5.0e9 + np.arange(uvw.shape[0], dtype=np.float64) * 8_000.0,
        antenna1=np.array([0, 0, 1, 0, 1, 2], dtype=np.int32),
        antenna2=np.array([1, 2, 3, 2, 3, 3], dtype=np.int32),
        correlations=(Correlation.RR, Correlation.LL),
        receptor_basis=ReceptorBasis.CIRCULAR,
        phase_centre_rad=phase_centre_rad,
    )


def test_survey_selector_refuses_wrong_id_and_digest(tmp_path: Path) -> None:
    root = _write_survey_catalog(tmp_path)
    digest = survey_catalog_digest(root)
    with pytest.raises(ValueError, match="unknown survey catalog"):
        voltage_beam_for_survey_catalog(
            root=root,
            digest=digest,
            catalog_id="diagonal_copolar",
            expected_raster=_TEST_RASTER,
        )
    with pytest.raises(ValueError, match="digest mismatch"):
        voltage_beam_for_survey_catalog(
            root=root,
            digest="0" * 64,
            expected_raster=_TEST_RASTER,
        )
    with pytest.raises(ValueError, match="requires survey_catalog_root"):
        stokes_i_beam(OPT_IN_SURVEY_BEAM)
    beam = voltage_beam_for_survey_catalog(
        root=root,
        digest=digest,
        expected_raster=_TEST_RASTER,
    )
    assert beam.model_id.startswith(f"{CATALOG_ID}:")
    assert beam.off_diagonal is False
    assert OPT_IN_SURVEY_BEAM not in PRODUCTION_STOKES_I_BEAMS


def test_production_factory_stays_on_packaged_cassbeam() -> None:
    packaged = voltage_beam_for_mode("diagonal_copolar")
    assert packaged.model_id == CASSBEAM_CBAND_MODEL_ID
    assert stokes_i_beam("diagonal_copolar").model_id == CASSBEAM_CBAND_MODEL_ID
    with pytest.raises(ValueError, match="unknown Stokes-I beam candidate"):
        stokes_i_beam("evla_c_diagonal_survey_v1_typo")


def test_survey_beam_refuses_nearest_frequency_and_stays_diagonal(tmp_path: Path) -> None:
    root = _write_survey_catalog(tmp_path)
    digest = survey_catalog_digest(root)
    beam = stokes_i_beam(
        OPT_IN_SURVEY_BEAM,
        survey_catalog_root=root,
        survey_catalog_digest=digest,
        survey_expected_raster=_TEST_RASTER,
    )
    assert isinstance(beam, EvlaCSurveyVoltageBeam)
    coordinates = BeamCoordinates(
        l_rad=np.array([0.0]),
        m_rad=np.array([0.0]),
        frequency_hz=np.array([4.548e9]),
        parallactic_angle_rad=np.array([0.1]),
        antenna_id=np.array([0], dtype=np.int32),
    )
    with pytest.raises(ValueError, match="nearest-frequency"):
        beam.evaluate(coordinates, calibration_state="casa_parang_true")
    on_grid = BeamCoordinates(
        l_rad=np.array([0.0]),
        m_rad=np.array([0.0]),
        frequency_hz=np.array([_FREQ_HZ]),
        parallactic_angle_rad=np.array([0.0]),
        antenna_id=np.array([0], dtype=np.int32),
    )
    evaluated = beam.evaluate(on_grid, calibration_state="casa_parang_true")
    assert evaluated.jones.shape[-2:] == (2, 2)
    assert evaluated.jones[0, 0, 0, 0, 1] == 0.0
    assert evaluated.jones[0, 0, 0, 1, 0] == 0.0
    assert evaluated.provenance["nearest_plane_substitution"] is False
    assert evaluated.provenance["airy_fallback"] is False
    record = survey_catalog_record(
        root=root,
        scored_slots=[{"frequency_hz": _FREQ_HZ}],
    )
    assert record["interpolation_policy"] == "refused"
    assert record["planes"][0]["support"] == "empirically_compared"
    assert record["native_3c391_mhz"] == list(NATIVE_3C391_MHZ)
    assert record["imaging_node_mhz"] == list(IMAGING_NODE_MHZ)


def test_survey_beam_matches_hand_coded_stokes_i_rime(tmp_path: Path) -> None:
    root = _write_survey_catalog(tmp_path)
    beam = voltage_beam_for_survey_catalog(
        root=root,
        digest=survey_catalog_digest(root),
        expected_raster=_TEST_RASTER,
    )
    coordinates = BeamCoordinates(
        l_rad=np.array([0.0]),
        m_rad=np.array([0.0]),
        frequency_hz=np.array([_FREQ_HZ]),
        parallactic_angle_rad=np.array([0.0, 0.0]),
        antenna_id=np.array([0, 1], dtype=np.int32),
    )
    evaluated = beam.evaluate(coordinates, calibration_state="casa_parang_true")
    left = evaluated.jones[0, 0, 0]
    right = evaluated.jones[1, 0, 0]
    stokes_i = 8.0
    source = 0.5 * stokes_i * np.eye(2, dtype=np.complex128)
    predicted = left @ source @ right.conj().T
    assert evaluated.provenance["off_diagonal"] is False
    np.testing.assert_allclose(left, np.eye(2), atol=1.0e-12)
    np.testing.assert_allclose(predicted[0, 0], 0.5 * stokes_i)
    np.testing.assert_allclose(predicted[1, 1], 0.5 * stokes_i)
    np.testing.assert_allclose(predicted[0, 1], 0.0)
    np.testing.assert_allclose(predicted[1, 0], 0.0)


def test_survey_adapter_seven_pointing_reconstruction(tmp_path: Path) -> None:
    root = _write_survey_catalog(tmp_path)
    digest = survey_catalog_digest(root)
    beam = voltage_beam_for_survey_catalog(
        root=root,
        digest=digest,
        expected_raster=_TEST_RASTER,
    )
    offsets = np.linspace(-3.0e-4, 3.0e-4, num=7)
    blocks = tuple(
        _block(phase_centre_rad=(_PHASE[0] + offset, _PHASE[1] - 0.5 * offset))
        for offset in offsets
    )
    table = starting_central_table(
        root_size=2,
        root_pixel_size_rad=np.deg2rad(16.0 / 3600.0),
        mosaic_phase_centre_rad=_PHASE,
        flux=np.array([0.05, 0.05, 0.05, 1.2]),
    )
    config = VoltageReconstructionConfig(
        root_size=2,
        root_pixel_size_rad=np.deg2rad(16.0 / 3600.0),
        inference=InferenceConfig(
            solver="proximal_sgd",
            batch_grouping="times",
            steps=4,
            learning_rate=0.2,
            sparsity_weight=0.0,
            patience=4,
            validation_interval=2,
            min_delta=1e-12,
            batch_size_rows=8,
            kkt_tolerance=1.0,
        ),
        tolerance=IntegrationTolerance(max_depth=1, forced_feature_depth=0),
        max_rounds=0,
        max_depth=1,
        kkt_max_batches=1,
        operator_mode="explicit_jax",
    )
    checkpoints: list[dict[str, object]] = []

    def on_checkpoint(fit, _state) -> None:
        checkpoints.append(
            {
                "train_loss": fit.train_loss,
                "components": fit.table.source,
            }
        )

    result = reconstruct_voltage_stokes_i(
        table,
        blocks,
        beam,
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
        config=config,
        beam_mode=OPT_IN_SURVEY_BEAM,
        pointing_ids=_POINTINGS,
        on_checkpoint=on_checkpoint,
    )
    assert result.beam_mode == OPT_IN_SURVEY_BEAM
    assert np.isfinite(result.fit.train_loss)
    assert checkpoints
    directory = tmp_path / "products"
    write_reconstruction_products(
        directory,
        result,
        blocks,
        pointing_ids=_POINTINGS,
        antenna_position_m=_ANTENNA_POSITION_M,
        folds=phase6_folds(blocks),
        config={"steps": 4},
        manifest={"model_id": beam.model_id, "catalog_digest": digest},
    )
    assert (directory / "summary.json").is_file()
    for pointing in _POINTINGS:
        assert (directory / f"pointing_{pointing}.npz").is_file()
    smoke_dir = tmp_path / "smoke_products"
    smoke_payload = write_smoke_reconstruction_products(
        smoke_dir,
        result,
        blocks,
        pointing_ids=_POINTINGS,
        config={"steps": 4},
        manifest={"model_id": beam.model_id, "catalog_digest": digest},
    )
    assert smoke_payload["fold_products"] == "skipped_insufficient_time_bins"
    assert (smoke_dir / "intrinsic_stokes_i.npz").is_file()
    catalog = HighresCassbeamCatalog(
        root, expected_model_id=SURVEY_MODEL_ID, expected_raster=_TEST_RASTER
    )
    assert catalog.frequency_mhz_list() == (4536,)
