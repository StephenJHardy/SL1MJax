"""Render the documentation page from the compact bundle and shared figures."""

from __future__ import annotations

import argparse
from pathlib import Path

from sl1mjax.beam_validation_outputs import load_bundle
from sl1mjax.beam_validation_plots import write_all_figures
from sl1mjax.beam_validation_statistics import (
    claim_table,
    copolar_correlation,
    copolar_slope,
    crosshand_non_detection,
    offset_ring_diagonal_closure,
    publication_squint_pair,
    residual_power_table,
)

ASSET_DIR = Path("docs/assets/vla_c_band_beam_validation")
DOC_PATH = Path("docs/vla_c_band_beam_validation.md")


def _pct(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def _slope(value: complex) -> str:
    sign = "+" if value.imag >= 0.0 else "-"
    return f"{value.real:.3f}{sign}{abs(value.imag):.3f}i"


def render_markdown(bundle, figures: list[Path]) -> str:
    residuals = residual_power_table(bundle.holoraster_channel32)
    squint = publication_squint_pair(bundle.squint)
    ring = offset_ring_diagonal_closure(bundle.offset_ring)
    cross = crosshand_non_detection(bundle.holoraster_channel32)
    claims = claim_table(bundle)
    lines = [
        "# VLA C-band beam validation",
        "",
        "This page is rendered from the compact SPW-4 validation bundle and the",
        "shared plotting functions. It is not an independently written narrative.",
        "The executable account is `notebooks/vla_c_band_beam_validation.ipynb`.",
        "",
        f"Bundle `{bundle.manifest['publication_version']}` checksum",
        f"`{bundle.manifest['bundle_sha256']}`.",
        "",
        "## Executive summary",
        "",
        "CASSBEAM is the reference diagonal C-band beam inside a stated validity",
        "domain. It is not an unqualified high-dynamic-range model of the whole",
        "raster.",
        "",
        "| Region | Support class | Channel-32 residual power |",
        "|---|---|---|",
        f"| Main lobe | **accepted** | RR {_pct(residuals['main_lobe']['rr'])}, LL {_pct(residuals['main_lobe']['ll'])} |",
        f"| Mid beam | **qualified** | RR {_pct(residuals['mid']['rr'])}, LL {_pct(residuals['mid']['ll'])} |",
        f"| Outer raster | **diagnostic** | RR {_pct(residuals['outer_diagnostic']['rr'])}, LL {_pct(residuals['outer_diagnostic']['ll'])} |",
        "",
        f"Complex slopes are {_slope(copolar_slope(bundle.holoraster_channel32, 'rr'))} (RR) and",
        f"{_slope(copolar_slope(bundle.holoraster_channel32, 'll'))} (LL), with correlations",
        f"{copolar_correlation(bundle.holoraster_channel32, 'rr'):.3f} /",
        f"{copolar_correlation(bundle.holoraster_channel32, 'll'):.3f}.",
        "",
        f"The C147-* ring at {ring['radius_arcmin']:.2f}′ gives RR/LL residual power",
        f"{_pct(ring['rr'])} / {_pct(ring['ll'])} after the geometric fringe.",
        "That Q/U result is not a frequency holdout.",
        "",
        "The cross-hand result is a sensitivity-limited non-detection:",
        f"RL/LR correlation {cross['rl_correlation']:.3f} / {cross['lr_correlation']:.3f},",
        f"residual power {cross['rl_residual_power']:.3f} / {cross['lr_residual_power']:.3f}.",
        "",
        "Publication squint uses the 20%-of-peak main-lobe estimator:",
        f"measured {float(squint['measured']['separation_arcmin']):.3f}′,",
        f"Memo 195 {float(squint['measured']['memo195_separation_arcmin']):.3f}′,",
        f"CASSBEAM {float(squint['cassbeam']['separation_arcmin']):.3f}′.",
        "",
        "![Observation timeline](assets/vla_c_band_beam_validation/01_observation_timeline.png)",
        "",
        "![Raster occupancy](assets/vla_c_band_beam_validation/02_raster_occupancy.png)",
        "",
        "![Main-lobe CASSBEAM scatter](assets/vla_c_band_beam_validation/08_cassbeam_scatter_main_lobe.png)",
        "",
        "![Spatial residual maps](assets/vla_c_band_beam_validation/10_spatial_measured_cassbeam_residual.png)",
        "",
        "![Residual versus radius](assets/vla_c_band_beam_validation/12_residual_vs_radius.png)",
        "",
        "![Publication squint](assets/vla_c_band_beam_validation/13_squint_mainlobe_20pct.png)",
        "",
        "![Offset ring](assets/vla_c_band_beam_validation/17_c147_offset_ring.png)",
        "",
        "![Experimental cross-hands](assets/vla_c_band_beam_validation/18_experimental_crosshand.png)",
        "",
        "## Claim registry",
        "",
        "| ID | Status | Claim |",
        "|---|---|---|",
    ]
    for row in claims:
        lines.append(f"| {row['id']} | {row['status']} | {row['title']} |")
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- One published SPW-4 frequency; SPW 5 remains sealed.",
            "- Outer-raster residual power is too large for an unqualified HDR model.",
            "- Full Jones is an experimental non-detection below the THOL0001 floor.",
            "- The next artifact is a low-order diagonal correction aimed at the 10′ ring.",
            "",
        ]
    )
    _ = figures
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--asset-dir", type=Path, default=ASSET_DIR)
    parser.add_argument("--output", type=Path, default=DOC_PATH)
    arguments = parser.parse_args()
    bundle = load_bundle(arguments.bundle)
    written = write_all_figures(bundle, arguments.asset_dir)
    arguments.output.write_text(render_markdown(bundle, written))
    print(arguments.output)


if __name__ == "__main__":
    main()
