import json
from pathlib import Path

import numpy as np

from sl1mjax.data.synthetic import PointSource, simulate_dataset
from sl1mjax.inference import InferenceConfig, infer_regular_grid
from sl1mjax.polarization import ReceptorBasis
from sl1mjax.rime import predict_stokes_i
from sl1mjax.sky import RegularGrid


def test_casa_vla_configuration_uv_coverage_and_resolution_scale() -> None:
    statistics = json.loads(
        (
            Path(__file__).parent / "fixtures" / "casa_vla_sampling.json"
        ).read_text()
    )["configurations"]
    p99 = [statistics[name]["uv_distance_m_p99"] for name in ("A", "C", "D")]
    resolution = [
        statistics[name]["nominal_resolution_arcsec_p99"] for name in ("A", "C", "D")
    ]
    assert p99[0] > p99[1] > p99[2]
    assert resolution[0] < resolution[1] < resolution[2]
    assert p99[0] / p99[1] > 5
    assert p99[1] / p99[2] > 2


def test_off_grid_source_reconstruction_has_bounded_astrometry_flux_and_residual() -> None:
    grid = RegularGrid(12, np.deg2rad(6 / 3600))
    pixel = grid.pixel_size_rad
    source = PointSource(1.0, 0.35 * pixel, -0.4 * pixel)
    block = simulate_dataset(
        grid,
        basis=ReceptorBasis.CIRCULAR,
        sources=(source,),
        rows=384,
        channels=2,
        max_baseline_m=3_000,
        seed=14,
    ).blocks[0]
    result = infer_regular_grid(
        block,
        grid,
        block.active,
        InferenceConfig(
            steps=300,
            learning_rate=0.1,
            sparsity_weight=1e-5,
            initial_intensity=0.001,
            patience=350,
            chunk_size=127,
        ),
    )
    l, m = grid.coordinates
    peak = int(np.argmax(result.image))
    astrometric_error_pixels = np.hypot(
        (l[peak] - source.l) / pixel,
        (m[peak] - source.m) / pixel,
    )
    prediction = np.asarray(
        predict_stokes_i(
            result.image.ravel(),
            l,
            m,
            block.uvw_m,
            block.frequency_hz,
            block.antenna1,
            block.antenna2,
            block.correlations,
        )
    )
    normalized_complex_rms = np.linalg.norm(prediction - block.visibility) / np.linalg.norm(
        block.visibility
    )
    peak_row, peak_column = np.unravel_index(peak, result.image.shape)
    compact_flux = np.sum(
        result.image[
            max(0, peak_row - 1) : peak_row + 2,
            max(0, peak_column - 1) : peak_column + 2,
        ]
    )
    assert astrometric_error_pixels < 1
    assert abs(compact_flux - source.flux) < 0.25
    assert normalized_complex_rms < 0.15
