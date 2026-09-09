"""Render the documentation page from the compact bundle and shared figures."""

from __future__ import annotations

import argparse
from pathlib import Path

from sl1mjax.beam_validation_outputs import load_bundle
from sl1mjax.beam_validation_plots import write_all_figures
from sl1mjax.beam_validation_statistics import (
    claim_table,
    classify_diagonal_region_support,
    copolar_correlation,
    copolar_slope,
    crosshand_non_detection,
    frequency_copolar_series,
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
    squint = publication_squint_pair(bundle.squint)
    ring = offset_ring_diagonal_closure(bundle.offset_ring)
    cross = crosshand_non_detection(bundle.holoraster_channel32)
    claims = claim_table(bundle)
    residuals = residual_power_table(bundle.holoraster_channel32)
    support = classify_diagonal_region_support(
        {
            name: {
                "rr": {"residual_power": residuals[name]["rr"]},
                "ll": {"residual_power": residuals[name]["ll"]},
            }
            for name in ("main_lobe", "mid", "outer_diagnostic")
        }
    )
    frequency = frequency_copolar_series(bundle.holoraster_frequency)
    classification = dict(bundle.holoraster_channel32.get("classification") or {})
    outcome = str(
        classification.get("outcome")
        or bundle.claims.get("full_jones")
        or "not_retested_after_coordinate_feed_update"
    )
    coordinate = bundle.coordinate_feed_comparison
    paired = coordinate.get("paired_scores") or {}
    generic_coordinate = paired.get("generic_source_lm_vs_generic_commanded") or {}
    evla_feed = paired.get("evla_c_source_lm_vs_generic_source_lm") or {}
    development_squint = (
        ((coordinate.get("map_squint") or {}).get("source_lm_labels") or {}).get(
            "independent_masks"
        )
        or {}
    )
    evla_squint = coordinate.get("evla_plane_centroids") or {}
    n_dev = int(bundle.holoraster_channel32.get("n_development") or 0)
    freq_lo = min((row["frequency_hz"] for row in frequency), default=4.564e9)
    freq_hi = max((row["frequency_hz"] for row in frequency), default=4.564e9)

    def region_row(label: str, key: str, class_key: str) -> str:
        row = residuals[key]
        klass = support[class_key]
        return (
            f"| {label} | **{klass}** | "
            f"RR {_pct(row['rr'])}, LL {_pct(row['ll'])} | "
            f"RR {_pct(row['rr_rms'])}, LL {_pct(row['ll_rms'])} | "
            f"RR {_pct(row['rr_median_abs_over_i'])}, "
            f"LL {_pct(row['ll_median_abs_over_i'])} |"
        )

    def paired_text(record, axis: str) -> str:
        row = record.get(axis) or {}
        return (
            f"{float(row.get('delta', float('nan'))):+.4f} "
            f"[{float(row.get('delta_lo', float('nan'))):+.4f}, "
            f"{float(row.get('delta_hi', float('nan'))):+.4f}]"
        )

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
        "This refresh tests EVLA-C CASSBEAM at `source_lm_feed` on THOL0001 SPW 4",
        f"native channels {freq_lo/1e6:.0f}–{freq_hi/1e6:.0f} MHz",
        f"({len(frequency) or 1} sampled channels). It is not a whole-C-band validation",
        "and does not freeze a production beam. Residual power is not a flux error;",
        r"the RMS visibility column is $\sqrt{L}$.",
        "",
        f"Channel 32 uses {n_dev:,} development rows. Diagonal support:",
        "",
        "| Region | Support class | Residual power | RMS visibility | Median $|\\Delta V|/I$ |",
        "|---|---|---|---|---|",
        region_row("Main lobe", "main_lobe", "main_lobe"),
        region_row("Mid beam", "mid", "mid_beam"),
        region_row("Outer raster", "outer_diagnostic", "outer_raster"),
        "",
        "Outer CASSBEAM is coherent but not unqualified:",
        f"correlation {residuals['outer_diagnostic']['rr_correlation']:.3f} / "
        f"{residuals['outer_diagnostic']['ll_correlation']:.3f}.",
        "A common scalar beam cancels voltage phase in Stokes I. Outer phase",
        "maps are exploratory.",
        "",
        f"Full-Jones outcome: `{outcome}`.",
        f"Channel-32 experimental RL/LR residual power {cross['rl_residual_power']:.3f} / "
        f"{cross['lr_residual_power']:.3f}.",
        "Residual Jones is the field-9 channel-32 plane applied to all publication",
        "channels; that frequency limit is explicit.",
        "",
        f"EVLA-C diagonal slopes are {_slope(copolar_slope(bundle.holoraster_channel32, 'rr'))} (RR) and",
        f"{_slope(copolar_slope(bundle.holoraster_channel32, 'll'))} (LL), with correlations",
        f"{copolar_correlation(bundle.holoraster_channel32, 'rr'):.3f} /",
        f"{copolar_correlation(bundle.holoraster_channel32, 'll'):.3f}.",
        "",
        f"The C147-* ring at {ring['radius_arcmin']:.2f}′ gives RR/LL residual power",
        f"{_pct(ring['rr'])} / {_pct(ring['ll'])} after the geometric fringe.",
        "Those field partitions are historical development diagnostics, not a new sealed holdout.",
        "",
        "Publication squint uses the 20%-of-peak main-lobe estimator:",
        f"measured {float(squint['measured']['separation_arcmin']):.3f}′,",
        f"Memo 195 {float(squint['measured']['memo195_separation_arcmin']):.3f}′,",
        f"CASSBEAM {float(squint['cassbeam']['separation_arcmin']):.3f}′.",
        "",
        "## Current EVLA-C / source-in-beam panels",
        "",
        "These figures are recomputed from the refresh exports. They are not the",
        "historical generic/commanded comparison.",
        "",
        "![On-axis V/I](assets/vla_c_band_beam_validation/05_onaxis_amplitude.png)",
        "",
        "![Main-lobe CASSBEAM scatter](assets/vla_c_band_beam_validation/08_cassbeam_scatter_main_lobe.png)",
        "",
        "![Spatial residual maps](assets/vla_c_band_beam_validation/10_spatial_measured_cassbeam_residual.png)",
        "",
        "![Residual versus radius](assets/vla_c_band_beam_validation/12_residual_vs_radius.png)",
        "",
        "![Publication squint](assets/vla_c_band_beam_validation/13_squint_mainlobe_20pct.png)",
        "",
        "![Residual strata](assets/vla_c_band_beam_validation/15_residual_strata.png)",
        "",
        "![Frequency series](assets/vla_c_band_beam_validation/16_spw4_frequency.png)",
        "",
        "![Offset ring](assets/vla_c_band_beam_validation/17_c147_offset_ring.png)",
        "",
        "![Experimental cross-hands](assets/vla_c_band_beam_validation/18_experimental_crosshand.png)",
        "",
        "![SPW 5 sealed](assets/vla_c_band_beam_validation/19_spw5_sealed.png)",
        "",
        "![Amplitude in dB](assets/vla_c_band_beam_validation/20_amplitude_db.png)",
        "",
        "![Signed complex cuts](assets/vla_c_band_beam_validation/21_signed_complex_cuts.png)",
        "",
        "![Masked phase](assets/vla_c_band_beam_validation/22_masked_phase.png)",
        "",
        "![Radial coherence](assets/vla_c_band_beam_validation/23_radial_coherence.png)",
        "",
        "![Per-antenna coherence](assets/vla_c_band_beam_validation/24_antenna_coherence.png)",
        "",
        "![Bright-source examples](assets/vla_c_band_beam_validation/25_bright_source_examples.png)",
        "",
        "## Observation and calibration provenance (F01–F04 reused)",
        "",
        "These panels validate the observation, occupancy, antenna roles, and",
        "CASA/JAX apply operator. They do not validate HOLORASTER coordinates or",
        "the EVLA-C beam.",
        "",
        "![Observation timeline](assets/vla_c_band_beam_validation/01_observation_timeline.png)",
        "",
        "![Raster occupancy](assets/vla_c_band_beam_validation/02_raster_occupancy.png)",
        "",
        "![Antenna roles](assets/vla_c_band_beam_validation/03_antenna_roles.png)",
        "",
        "![CASA/JAX operator residual](assets/vla_c_band_beam_validation/04_casa_jax_operator_residual.png)",
        "",
        "## Appendix: coordinate/feed controls (labelled historical models)",
        "",
        r"The HOLORASTER metadata gives the commanded antenna displacement. The",
        r"beam must be queried at the source relative to that pointing:",
        r"$s_{\rm feed}=-\Delta_{\rm commanded}$.",
        "The following paired scores keep generic and EVLA-C models on matched",
        "development rows. They are controls, not a second unseen validation.",
        "",
        "| Change | Spatial paired ΔL [95% CI] | Mover paired ΔL [95% CI] |",
        "|---|---:|---:|",
        (
            "| Correct coordinate, generic feed | "
            f"{paired_text(generic_coordinate, 'spatial')} | "
            f"{paired_text(generic_coordinate, 'moving')} |"
        ),
        (
            "| EVLA-C feed at corrected coordinate | "
            f"{paired_text(evla_feed, 'spatial')} | "
            f"{paired_text(evla_feed, 'moving')} |"
        ),
        "",
        "The corrected-frame vector plot uses the training-only independent-mask",
        f"measured estimate ({float(development_squint.get('separation_arcmin', float('nan'))):.3f}′)",
        "to compare direction. It does not replace the all-data publication estimate",
        f"({float(squint['measured']['separation_arcmin']):.3f}′). The EVLA-C plane is",
        f"{float(evla_squint.get('separation_arcmin', float('nan'))):.3f}′.",
        "",
        "![Coordinate and feed impact](assets/vla_c_band_beam_validation/26_coordinate_feed_impact.png)",
        "",
        "![Updated measured-versus-predicted scatter](assets/vla_c_band_beam_validation/27_coordinate_feed_scatter.png)",
        "",
        "![Updated residual maps](assets/vla_c_band_beam_validation/28_coordinate_feed_residual_maps.png)",
        "",
        "![Corrected-frame squint vectors](assets/vla_c_band_beam_validation/29_coordinate_feed_squint.png)",
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
            "- Use EVLA-C CASSBEAM at source-in-beam coordinates as the leading SPW-4 development prior.",
            "- Keep generic VLA at commanded coordinates as a historical ablation.",
            "- Mid beam is qualified; attach substantial model uncertainty outside it.",
            "- Bright out-of-field sources can be misestimated by factors of two or more.",
            "- Do not infer a general outer phase correction from the current phase means.",
            "- Outer-field corrections must transfer across movers and spatial holdouts;",
            "  otherwise use source-specific nuisance terms or peeling.",
            f"- Tested SPW-4 frequencies: {freq_lo/1e6:.0f}–{freq_hi/1e6:.0f} MHz; SPW 5 remains sealed.",
            f"- Full-Jones outcome is `{outcome}`; this does not accept a production beam.",
            "- Residual Jones has no per-channel axis; the channel-32 plane is applied explicitly.",
            "- C147-* partitions are historical development diagnostics.",
            "- The next bounded full-Jones step is a residual-Jones frequency axis;",
            "  do not retune the current gates. The next diagonal step remains a",
            "  validation-selected low-order correction or a sealed frequency-transfer",
            "  test, not a production full-Jones freeze.",
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
