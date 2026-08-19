from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

import sl1mjax.data.ms as ms_module
from sl1mjax.data.metadata import CalibratorRole
from sl1mjax.polarization import Correlation, ReceptorBasis


class FakeTable:
    def __init__(
        self,
        columns: dict[str, Any],
        *,
        cell_defined: dict[str, bool] | None = None,
    ) -> None:
        self.columns = columns
        self.cell_defined = cell_defined or {}
        self.entered = False
        self.closed = False

    def __enter__(self) -> FakeTable:
        self.entered = True
        return self

    def __exit__(self, *_args: object) -> None:
        self.closed = True

    def colnames(self) -> list[str]:
        return list(self.columns)

    def getcol(self, name: str) -> Any:
        return self.columns[name]

    def getcell(self, name: str, row: int) -> Any:
        return self.columns[name][row]

    def iscelldefined(self, name: str, row: int) -> bool:
        assert 0 <= row < self.nrows()
        return self.cell_defined.get(name, True)

    def nrows(self) -> int:
        first = next(iter(self.columns.values()))
        return len(first)


class RaggedFakeTable(FakeTable):
    def __init__(
        self,
        columns: dict[str, Any],
        *,
        ragged_columns: set[str],
        cell_defined: dict[str, bool] | None = None,
    ) -> None:
        super().__init__(columns, cell_defined=cell_defined)
        self.ragged_columns = ragged_columns

    def getcol(self, name: str) -> Any:
        if name in self.ragged_columns:
            raise RuntimeError(f"{name} contains variable-shaped cells")
        return super().getcol(name)


class FakeTables:
    def __init__(self, source: Path, *, weight_spectrum_defined: bool) -> None:
        shape = (6, 2, 4)
        sample = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
        self.visibility = sample + 1j * (sample + 0.5)
        self.flags = np.zeros(shape, dtype=bool)
        self.flags[4, 1, 2] = True
        self.flag_row = np.zeros(6, dtype=bool)
        self.weight_spectrum = sample + 10.0
        self.row_weight = np.arange(24, dtype=np.float64).reshape(6, 4) + 1.0
        self.field_id = np.array([1, 0, 0, 1, 1, 0], dtype=np.int32)
        self.ddid = np.array([1, 0, 1, 0, 1, 0], dtype=np.int32)
        self.uvw = np.arange(18, dtype=np.float64).reshape(6, 3)
        self.times = np.arange(6, dtype=np.float64) + 100.0
        self.antenna1 = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)
        self.antenna2 = np.array([1, 2, 2, 3, 3, 0], dtype=np.int32)
        self.scan = np.array([3, 3, 4, 4, 5, 5], dtype=np.int32)
        self.channel_frequencies = (
            np.array([1.0e9, 1.01e9]),
            np.array([1.5e9, 1.51e9]),
        )
        self.phase_centres = (
            np.array([[[0.1, -0.2]]]),
            np.array([[[0.3, -0.4]]]),
        )
        self.calls: list[tuple[str, bool, bool]] = []
        self.tables = {
            source.name: FakeTable(
                {
                    "CORRECTED_DATA": self.visibility,
                    "MODEL_DATA": 0.5 * self.visibility,
                    "FLAG": self.flags,
                    "FLAG_ROW": self.flag_row,
                    "UVW": self.uvw,
                    "TIME": self.times,
                    "ANTENNA1": self.antenna1,
                    "ANTENNA2": self.antenna2,
                    "SCAN_NUMBER": self.scan,
                    "STATE_ID": np.array([1, 0, 0, 1, 1, 0], dtype=np.int32),
                    "OBSERVATION_ID": np.zeros(6, dtype=np.int32),
                    "FEED1": np.zeros(6, dtype=np.int32),
                    "FEED2": np.zeros(6, dtype=np.int32),
                    "INTERVAL": np.full(6, 10.0),
                    "FIELD_ID": self.field_id,
                    "DATA_DESC_ID": self.ddid,
                    "WEIGHT_SPECTRUM": self.weight_spectrum,
                    "WEIGHT": self.row_weight,
                },
                cell_defined={"WEIGHT_SPECTRUM": weight_spectrum_defined},
            ),
            "DATA_DESCRIPTION": FakeTable(
                {
                    "SPECTRAL_WINDOW_ID": np.array([1, 0], dtype=np.int32),
                    "POLARIZATION_ID": np.array([1, 0], dtype=np.int32),
                }
            ),
            "SPECTRAL_WINDOW": FakeTable(
                {"CHAN_FREQ": self.channel_frequencies}
            ),
            "POLARIZATION": FakeTable(
                {
                    "CORR_TYPE": (
                        np.array([9, 10, 11, 12], dtype=np.int32),
                        np.array([5, 6, 7, 8], dtype=np.int32),
                    )
                }
            ),
            "FIELD": FakeTable(
                {
                    "NAME": np.array(["target", "calibrator"]),
                    "SOURCE_ID": np.array([0, 1]),
                    "PHASE_DIR": self.phase_centres,
                    "DELAY_DIR": self.phase_centres,
                    "REFERENCE_DIR": self.phase_centres,
                }
            ),
            "ANTENNA": FakeTable(
                {
                    "NAME": np.array(["ea01", "ea02", "ea03", "ea04"]),
                    "STATION": np.array(["W01", "E01", "N01", "W02"]),
                    "POSITION": np.arange(12, dtype=float).reshape(4, 3),
                    "DISH_DIAMETER": np.full(4, 25.0),
                    "MOUNT": np.array(["ALT-AZ"] * 4),
                }
            ),
            "STATE": FakeTable(
                {
                    "OBS_MODE": np.array(
                        ["OBSERVE_TARGET#ON_SOURCE", "CALIBRATE_PHASE#ON_SOURCE"]
                    )
                }
            ),
            "OBSERVATION": FakeTable(
                {
                    "TELESCOPE_NAME": np.array(["EVLA"]),
                    "OBSERVER": np.array(["tester"]),
                    "PROJECT": np.array(["TEST"]),
                    "TIME_RANGE": np.array([[100.0, 106.0]]),
                }
            ),
            "FEED": FakeTable(
                {
                    "ANTENNA_ID": np.arange(4),
                    "FEED_ID": np.zeros(4, dtype=int),
                    "SPECTRAL_WINDOW_ID": np.zeros(4, dtype=int),
                    "POLARIZATION_TYPE": np.array([["R", "L"]] * 4),
                    "RECEPTOR_ANGLE": np.zeros((4, 2)),
                }
            ),
        }

    def table(self, path: str, *, readonly: bool, ack: bool) -> FakeTable:
        self.calls.append((path, readonly, ack))
        key = Path(path).name
        return self.tables[key]


class RaggedFakeTables(FakeTables):
    def __init__(self, source: Path) -> None:
        super().__init__(source, weight_spectrum_defined=True)
        self.channel_frequencies = (
            np.array([1.0e9, 1.01e9, 1.02e9]),
            np.array([1.5e9, 1.51e9]),
        )
        ragged_visibility = []
        ragged_flags = []
        ragged_weight = []
        for row, ddid in enumerate(self.ddid):
            channels = 3 if ddid == 1 else 2
            values = (
                np.arange(channels * 4, dtype=np.float64).reshape(channels, 4)
                + row * 100
            )
            ragged_visibility.append(values + 1j * (values + 0.5))
            ragged_flags.append(np.zeros((channels, 4), dtype=bool))
            ragged_weight.append(values + 10.0)
        self.ragged_visibility = tuple(ragged_visibility)
        self.ragged_flags = tuple(ragged_flags)
        self.ragged_weight = tuple(ragged_weight)
        self.tables[source.name] = RaggedFakeTable(
            {
                "CORRECTED_DATA": self.ragged_visibility,
                "FLAG": self.ragged_flags,
                "UVW": self.uvw,
                "TIME": self.times,
                "ANTENNA1": self.antenna1,
                "ANTENNA2": self.antenna2,
                "SCAN_NUMBER": self.scan,
                "FIELD_ID": self.field_id,
                "DATA_DESC_ID": self.ddid,
                "WEIGHT_SPECTRUM": self.ragged_weight,
                "WEIGHT": self.row_weight,
            },
            ragged_columns={"CORRECTED_DATA", "FLAG", "WEIGHT_SPECTRUM"},
        )
        self.tables["SPECTRAL_WINDOW"] = FakeTable(
            {"CHAN_FREQ": self.channel_frequencies}
        )


@pytest.fixture
def fake_ms_path(tmp_path: Path) -> Path:
    return tmp_path / "observation.ms"


def test_extracts_multiple_fields_ddids_and_spws_with_weight_spectrum(
    monkeypatch: pytest.MonkeyPatch, fake_ms_path: Path
) -> None:
    fake = FakeTables(fake_ms_path, weight_spectrum_defined=True)
    monkeypatch.setattr(ms_module, "_tables", lambda: fake)

    dataset = ms_module.extract_measurement_set(fake_ms_path)

    field_ddid_pairs = []
    for block in dataset.blocks:
        assert block.field_id is not None
        field_ddid_pairs.append((block.field_id[0], block.data_description_id))
    assert field_ddid_pairs == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ]
    for block in dataset.blocks:
        assert block.field_id is not None
        field = int(block.field_id[0])
        ddid = block.data_description_id
        selected = (fake.field_id == field) & (fake.ddid == ddid)
        expected_spw = (1, 0)[ddid]
        expected_polid = (1, 0)[ddid]

        assert block.spectral_window_id == expected_spw
        assert block.polarization_id == expected_polid
        np.testing.assert_array_equal(
            block.frequency_hz, fake.channel_frequencies[expected_spw]
        )
        np.testing.assert_array_equal(block.visibility, fake.visibility[selected])
        np.testing.assert_array_equal(block.weight, fake.weight_spectrum[selected])
        np.testing.assert_array_equal(block.flag, fake.flags[selected])
        np.testing.assert_array_equal(block.uvw_m, fake.uvw[selected])
        assert block.phase_centre_rad == tuple(fake.phase_centres[field].reshape(-1, 2)[0])
        assert block.provenance["field_id"] == field
        assert block.provenance["data_description_id"] == ddid
        assert block.provenance["source_column"] == "CORRECTED_DATA"

        if ddid == 0:
            assert block.receptor_basis is ReceptorBasis.CIRCULAR
            assert block.correlations == (
                Correlation.RR,
                Correlation.RL,
                Correlation.LR,
                Correlation.LL,
            )
        else:
            assert block.receptor_basis is ReceptorBasis.LINEAR
            assert block.correlations == (
                Correlation.XX,
                Correlation.XY,
                Correlation.YX,
                Correlation.YY,
            )

    assert dataset.provenance["extractor"] == "sl1mjax"
    assert dataset.provenance["source"] == str(fake_ms_path.resolve())
    assert dataset.metadata is not None
    assert dataset.metadata.antennas[0].name == "ea01"
    assert dataset.metadata.fields[1].name == "calibrator"
    assert dataset.metadata.fields[1].roles == (CalibratorRole.PHASE,)
    assert len(fake.calls) == 10
    assert all(readonly and not ack for _, readonly, ack in fake.calls)
    assert all(table.entered and table.closed for table in fake.tables.values())


def test_selects_field_by_role_and_preserves_model_and_row_metadata(
    monkeypatch: pytest.MonkeyPatch, fake_ms_path: Path
) -> None:
    fake = FakeTables(fake_ms_path, weight_spectrum_defined=True)
    monkeypatch.setattr(ms_module, "_tables", lambda: fake)

    dataset = ms_module.extract_measurement_set(
        fake_ms_path,
        model_column="MODEL_DATA",
        roles=(CalibratorRole.PHASE,),
    )

    assert len(dataset.blocks) == 2
    for block in dataset.blocks:
        selected = (fake.field_id == 1) & (fake.ddid == block.data_description_id)
        assert block.model_visibility is not None
        np.testing.assert_array_equal(
            block.model_visibility,
            0.5 * fake.visibility[selected],
        )
        np.testing.assert_array_equal(block.interval_s, 10.0)
        np.testing.assert_array_equal(block.state_id, 1)


def test_falls_back_to_row_weights_and_broadcasts_over_channels(
    monkeypatch: pytest.MonkeyPatch, fake_ms_path: Path
) -> None:
    fake = FakeTables(fake_ms_path, weight_spectrum_defined=False)
    monkeypatch.setattr(ms_module, "_tables", lambda: fake)

    dataset = ms_module.extract_measurement_set(
        fake_ms_path,
        fields=(1,),
        data_description_ids=(1,),
    )

    assert len(dataset.blocks) == 1
    block = dataset.blocks[0]
    selected = (fake.field_id == 1) & (fake.ddid == 1)
    expected = np.broadcast_to(
        fake.row_weight[selected, None, :], fake.visibility[selected].shape
    )
    np.testing.assert_array_equal(block.weight, expected)
    assert block.weight.shape == block.visibility.shape == (2, 2, 4)


def test_extracts_ddids_with_different_channel_counts(
    monkeypatch: pytest.MonkeyPatch, fake_ms_path: Path
) -> None:
    fake = RaggedFakeTables(fake_ms_path)
    monkeypatch.setattr(ms_module, "_tables", lambda: fake)

    dataset = ms_module.extract_measurement_set(fake_ms_path, fields=(0,))

    assert len(dataset.blocks) == 2
    for block in dataset.blocks:
        selected_rows = np.flatnonzero(
            (fake.field_id == 0) & (fake.ddid == block.data_description_id)
        )
        expected_visibility = np.stack(
            [fake.ragged_visibility[row] for row in selected_rows]
        )
        expected_weight = np.stack([fake.ragged_weight[row] for row in selected_rows])
        assert block.visibility.shape[1] == block.frequency_hz.size
        np.testing.assert_array_equal(block.visibility, expected_visibility)
        np.testing.assert_array_equal(block.weight, expected_weight)


def test_explicit_channel_and_row_selection_is_preserved_in_provenance(
    monkeypatch: pytest.MonkeyPatch, fake_ms_path: Path
) -> None:
    fake = FakeTables(fake_ms_path, weight_spectrum_defined=True)
    monkeypatch.setattr(ms_module, "_tables", lambda: fake)

    dataset = ms_module.extract_measurement_set(
        fake_ms_path,
        fields=(1,),
        data_description_ids=(1,),
        channels=(1,),
        row_stride=2,
    )

    block = dataset.blocks[0]
    selected_rows = np.flatnonzero((fake.field_id == 1) & (fake.ddid == 1))[::2]
    np.testing.assert_array_equal(
        block.visibility, fake.visibility[selected_rows][:, [1], :]
    )
    assert block.shape == (1, 1, 4)
    assert block.provenance["row_stride"] == 2
    np.testing.assert_array_equal(block.provenance["channel_indices"], [1])
