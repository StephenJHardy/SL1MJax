from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
from test_3c391_composite_script import _block

SCRIPT = Path(__file__).parents[1] / "scripts" / "analyze_3c391_flagged_visibilities.py"
if str(SCRIPT.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT.parent))


def _module():
    spec = importlib.util.spec_from_file_location("analyze_flagged_visibilities", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_flatten_samples_separates_amplitude_and_phase_changes() -> None:
    module = _module()
    block = _block()
    prediction = np.ones(block.shape, dtype=np.complex128)
    visibility = prediction.copy()
    visibility[0, 0, 0] = 2.0
    visibility[1, 0, 0] = 1.0 + 1.0j
    block = module.replace(block, visibility=visibility)
    scores = np.zeros(block.shape, dtype=np.float64)

    features = module._flatten_samples(
        (block,),
        (prediction,),
        (scores,),
        start_time_s=float(np.min(block.time_s)),
        ratio_snr_threshold=0.0,
    )

    first = int(np.argmax(features["observed_real_jy"]))
    second = int(np.argmax(features["observed_imag_jy"]))
    assert features["radial_residual_sigma"][first] > 0
    assert abs(features["tangential_residual_sigma"][first]) < 1e-12
    assert features["tangential_residual_sigma"][second] > 0


def test_cohort_summary_describes_tail_morphology_and_power() -> None:
    module = _module()
    features = {
        "score": np.asarray([0.0, 7.0, 8.0, 1.0]),
        "weight": np.ones(4),
        "residual_amplitude_jy": np.asarray([1.0, 3.0, 4.0, 1.0]),
        "weighted_residual_amplitude": np.asarray([1.0, 3.0, 4.0, 1.0]),
        "model_snr": np.full(4, 10.0),
        "log_amplitude_ratio": np.zeros(4),
        "phase_error_deg": np.zeros(4),
        "radial_residual_sigma": np.asarray([0.0, 3.0, 0.0, 0.0]),
        "tangential_residual_sigma": np.asarray([0.0, 0.0, 4.0, 0.0]),
        "pointing": np.zeros(4, dtype=int),
        "time_bin": np.zeros(4, dtype=int),
        "channel": np.zeros(4, dtype=int),
        "correlation": np.zeros(4, dtype=int),
    }

    summary = module._cohort_summary(features, score_threshold=6.0)

    assert summary["tail_count"] == 2
    assert summary["tail_fraction"] == 0.5
    assert summary["tail_morphology"]["amplitude_dominated_fraction"] == 0.5
    assert summary["tail_morphology"]["phase_dominated_fraction"] == 0.5
    assert summary["tail_morphology"]["cross_baseline_coherent_fraction"] == 1.0


def test_amplitude_comparison_writes_matched_panels(tmp_path: Path) -> None:
    module = _module()
    predicted = np.geomspace(1e-3, 2.0, 200)

    def cohort(scale: float) -> dict[str, np.ndarray]:
        return {
            "predicted_real_jy": predicted,
            "predicted_imag_jy": np.zeros_like(predicted),
            "observed_real_jy": predicted * scale,
            "observed_imag_jy": np.zeros_like(predicted),
        }

    output = tmp_path / "comparison.jpg"
    module._plot_amplitude_comparison(output, cohort(1.0), cohort(1.1))

    assert output.exists()
    assert output.stat().st_size > 0


def test_phase_comparison_writes_matched_principal_squares(tmp_path: Path) -> None:
    module = _module()
    phase = np.linspace(-np.pi, np.pi, 200)

    def cohort(offset: float) -> dict[str, np.ndarray]:
        predicted = np.exp(1j * phase)
        observed = np.exp(1j * (phase + offset))
        return {
            "predicted_real_jy": predicted.real,
            "predicted_imag_jy": predicted.imag,
            "observed_real_jy": observed.real,
            "observed_imag_jy": observed.imag,
            "model_snr": np.full(phase.size, 10.0),
            "observed_snr": np.full(phase.size, 10.0),
        }

    output = tmp_path / "phase-comparison.jpg"
    module._plot_phase_comparison(
        output,
        cohort(0.0),
        cohort(0.1),
        snr_threshold=3.0,
    )

    assert output.exists()
    assert output.stat().st_size > 0
