"""Opt-in release gate for a local calibrated VLA MeasurementSet."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from sl1mjax.data.canonical import read_dataset, write_dataset
from sl1mjax.data.ms import extract_measurement_set
from sl1mjax.imaging import ImagingConfig, reconstruct
from sl1mjax.inference import InferenceConfig

_MEASUREMENT_SET = os.environ.get("SL1MJAX_TEST_MS")

pytestmark = pytest.mark.skipif(
    not _MEASUREMENT_SET,
    reason="set SL1MJAX_TEST_MS to run the calibrated VLA release gate",
)


def test_calibrated_vla_extracts_and_images_without_casa_runtime(tmp_path: Path) -> None:
    field = os.environ.get("SL1MJAX_TEST_FIELD")
    ddid = os.environ.get("SL1MJAX_TEST_DDID")
    dataset = extract_measurement_set(
        Path(_MEASUREMENT_SET or ""),
        data_column=os.environ.get("SL1MJAX_TEST_COLUMN", "CORRECTED_DATA"),
        fields=None if field is None else (int(field),),
        data_description_ids=None if ddid is None else (int(ddid),),
        channels=(int(os.environ.get("SL1MJAX_TEST_CHANNEL", "0")),),
        row_stride=int(os.environ.get("SL1MJAX_TEST_ROW_STRIDE", "100")),
    )
    canonical = tmp_path / "vla.zarr"
    write_dataset(dataset, canonical)
    reloaded = read_dataset(canonical)
    result = reconstruct(
        reloaded.blocks[0],
        ImagingConfig(
            size=8,
            pixel_size_rad=np.deg2rad(5 / 3600),
            inference=InferenceConfig(steps=10, chunk_size=1024, patience=20),
        ),
    )
    assert np.isfinite(result.train_loss)
    assert np.isfinite(result.holdout_loss)
    assert result.residual.shape == reloaded.blocks[0].shape
