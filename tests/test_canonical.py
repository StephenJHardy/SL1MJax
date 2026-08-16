from dataclasses import replace
from pathlib import Path

import numpy as np

from sl1mjax.data.canonical import (
    VisibilityBlock,
    VisibilityDataset,
    read_dataset,
    write_dataset,
)
from sl1mjax.polarization import Correlation, ReceptorBasis


def _block(
    *,
    correlations: tuple[Correlation, ...],
    basis: ReceptorBasis,
    offset: float,
    ddid: int,
    spw: int,
    polid: int,
) -> VisibilityBlock:
    rows = 3
    channels = 2
    shape = (rows, channels, len(correlations))
    real = np.arange(np.prod(shape), dtype=np.float64).reshape(shape) + offset
    return VisibilityBlock(
        uvw_m=np.array(
            [[offset, 2.0, -3.0], [4.0, offset + 5.0, 6.0], [-7.0, 8.0, offset]]
        ),
        frequency_hz=np.array([1.0e9 + offset, 1.01e9 + offset]),
        visibility=real + 1j * (real + 0.25),
        weight=np.full(shape, offset + 1.0),
        flag=np.zeros(shape, dtype=bool),
        time_s=np.array([10.0, 20.0, 30.0]) + offset,
        antenna1=np.array([0, 1, 2]),
        antenna2=np.array([1, 2, 3]),
        field_id=np.array([4, 4, 4]),
        scan_id=np.array([7, 7, 8]),
        correlations=correlations,
        receptor_basis=basis,
        phase_centre_rad=(0.12 + offset / 100, -0.34),
        data_description_id=ddid,
        spectral_window_id=spw,
        polarization_id=polid,
        provenance={
            "source": f"observation-{ddid}.ms",
            "selection": {"ddid": ddid, "channels": [0, 1]},
        },
    )


def _assert_blocks_equal(actual: VisibilityBlock, expected: VisibilityBlock) -> None:
    assert actual.correlations == expected.correlations
    assert actual.receptor_basis is expected.receptor_basis
    assert actual.phase_centre_rad == expected.phase_centre_rad
    assert actual.data_description_id == expected.data_description_id
    assert actual.spectral_window_id == expected.spectral_window_id
    assert actual.polarization_id == expected.polarization_id
    assert actual.provenance == expected.provenance
    for name in (
        "uvw_m",
        "frequency_hz",
        "visibility",
        "weight",
        "flag",
        "time_s",
        "antenna1",
        "antenna2",
        "field_id",
        "scan_id",
    ):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))


def test_multi_block_zarr_round_trip_preserves_data_metadata_and_provenance(
    tmp_path: Path,
) -> None:
    blocks = (
        _block(
            correlations=(Correlation.YY, Correlation.XX),
            basis=ReceptorBasis.LINEAR,
            offset=1.0,
            ddid=3,
            spw=5,
            polid=7,
        ),
        _block(
            correlations=(Correlation.RL, Correlation.RR, Correlation.LL),
            basis=ReceptorBasis.CIRCULAR,
            offset=2.0,
            ddid=9,
            spw=11,
            polid=13,
        ),
        _block(
            correlations=(Correlation.V, Correlation.I),
            basis=ReceptorBasis.STOKES,
            offset=3.0,
            ddid=15,
            spw=17,
            polid=19,
        ),
    )
    original = VisibilityDataset(
        blocks,
        provenance={
            "pipeline": "integration-test",
            "inputs": ["first.ms", "second.ms"],
            "parameters": {"average": False},
        },
    )
    path = tmp_path / "canonical.zarr"

    write_dataset(original, path)
    restored = read_dataset(path)

    assert restored.provenance == original.provenance
    assert len(restored.blocks) == 3
    for actual, expected in zip(restored.blocks, original.blocks, strict=True):
        _assert_blocks_equal(actual, expected)


def test_active_mask_excludes_every_invalid_visibility_or_weight_condition() -> None:
    block = _block(
        correlations=(Correlation.XX, Correlation.XY, Correlation.YY),
        basis=ReceptorBasis.LINEAR,
        offset=1.0,
        ddid=0,
        spw=0,
        polid=0,
    )
    visibility = block.visibility.copy()
    weight = block.weight.copy()
    flag = block.flag.copy()
    flag[0, 0, 0] = True
    weight[0, 0, 1] = 0.0
    weight[0, 0, 2] = -1.0
    weight[0, 1, 0] = np.nan
    weight[0, 1, 1] = np.inf
    visibility[0, 1, 2] = np.nan + 1j
    visibility[1, 0, 0] = 1.0 + np.inf * 1j
    masked = VisibilityBlock(
        uvw_m=block.uvw_m,
        frequency_hz=block.frequency_hz,
        visibility=visibility,
        weight=weight,
        flag=flag,
        time_s=block.time_s,
        antenna1=block.antenna1,
        antenna2=block.antenna2,
        field_id=block.field_id,
        scan_id=block.scan_id,
        correlations=block.correlations,
        receptor_basis=block.receptor_basis,
    )
    expected = np.ones(block.shape, dtype=bool)
    expected.flat[:7] = False

    np.testing.assert_array_equal(masked.active, expected)


def test_numpy_provenance_values_are_normalized_for_json(tmp_path: Path) -> None:
    block = replace(
        _block(
            correlations=(Correlation.XX, Correlation.YY),
            basis=ReceptorBasis.LINEAR,
            offset=1.0,
            ddid=0,
            spw=0,
            polid=0,
        ),
        provenance={"ddid": np.int32(3), "channels": np.array([0, 1])},
    )
    path = tmp_path / "numpy-provenance.zarr"
    write_dataset(VisibilityDataset((block,)), path)

    restored = read_dataset(path).blocks[0]
    assert restored.provenance == {"ddid": 3, "channels": [0, 1]}
