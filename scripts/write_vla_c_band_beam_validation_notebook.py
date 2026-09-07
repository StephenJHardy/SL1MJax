"""Write the source VLA C-band beam-validation notebook.

Cells load the compact bundle. They do not hard-code scientific numbers.
"""

from __future__ import annotations

from pathlib import Path

import nbformat

NOTEBOOK_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "vla_c_band_beam_validation.ipynb"

MARKDOWN = [
    (
        "# VLA C-band beam validation\n"
        "\n"
        "This notebook is the executable scientific account of the THOL0001 SPW-4 "
        "CASSBEAM comparison. It starts from the compact, checksummed validation "
        "bundle. It does not read the Measurement Set, invoke CASA, or freeze a "
        "production beam.\n"
        "\n"
        "Full-Jones panels are experimental. SPW 5 is sealed."
    ),
    (
        "## 1. Executive summary\n"
        "\n"
        "CASSBEAM is the reference diagonal C-band beam inside a stated validity "
        "domain. The claim registry below is loaded from the bundle. The notebook "
        "does not invent claim status."
    ),
    (
        "## 2. Why a measured voltage beam is needed\n"
        "\n"
        "An incomplete beam can appear as spatial, spectral, polarised, or temporal "
        "sky structure. The compact-source holography equation is\n"
        "\n"
        "$$\n"
        "V_{mr}=E_m(s,\\nu,\\chi)\\,S(\\nu)\\,E_r(0,\\nu,\\chi)^{\\mathrm H}.\n"
        "$$\n"
        "\n"
        "CASSBEAM is a physical model. Holography decides whether a smooth "
        "correction is justified. The intended model is "
        "$E_{\\mathrm{VLA}}=E_{\\mathrm{CASSBEAM}}+\\Delta E_{\\mathrm{holography}}$, "
        "with $\\Delta E$ introduced only when it improves held-out visibilities."
    ),
    (
        "## 3. THOL0001 observation\n"
        "\n"
        "`POINTING_OFFSET` is an AZELGEO antenna coordinate. `ON_SOURCE` is not a "
        "valid selection field. Version 1 publishes 4.564 GHz only."
    ),
    (
        "## 4. Calibration and source model\n"
        "\n"
        "HOLORASTER field 10 already has `CORRECTED_DATA`. C147-* fields 1–8 were "
        "never applied on disk; those predictions start from `DATA`. Full-pol uses "
        "`parang=True`. The C147-* ring is unused prediction data."
    ),
    (
        "## 5. Convention and software correctness\n"
        "\n"
        "These residuals lock CASA/JAX apply semantics. They are not evidence that "
        "CASSBEAM is physically correct. The 128-member convention search is closed."
    ),
    (
        "## 6–8. Direct HOLORASTER comparison\n"
        "\n"
        "Every calibrated moving–reference visibility is compared with CASSBEAM at "
        "the measured AZELGEO coordinate. No coefficient is fitted. The source is "
        "at the phase centre, so the geometric fringe is identically one.\n"
        "\n"
        "On axis, $V_{mr}\\simeq S$ with $E_m(0)=E_r(0)=1$. Flux calibration cannot "
        "remove an off-axis beam error without spoiling that on-axis agreement. "
        "Away from the origin, $\\Delta V(s)\\simeq S\\,[E_{\\mathrm{measured}}(s)"
        "-E_{\\mathrm{CASSBEAM}}(s)]$. A 1 Jy residual on this 8 Jy calibrator is a "
        "voltage-beam error of about 0.125, not a missing flux-scale factor.\n"
        "\n"
        "The quoted residual-power scores are not flux or amplitude errors. Their "
        "square roots are the RMS visibility errors. The notebook also reports "
        "median $|\\Delta V|/I$. The publication scatter uses the same $V/I_{\\mathrm{model}}$ "
        "main-lobe mask as the claimed metric.\n"
        "\n"
        "Maps are visibility-domain binned means, not recovered $E$ interpolants. "
        "Unsupported cells are masked. Raster-family residuals are a Memo 195 "
        "dense/sparse occupancy proxy, not a scan-id join.\n"
        "\n"
        "The measured raster reaches about $\\pm 51'$ and $71'$ in the corners. "
        "High-resolution CASSBEAM covers essentially that whole field. Outer-beam "
        "CASSBEAM captures substantial coherent structure (complex correlation "
        "about 0.82), but residual power of 32--35% and a median amplitude ratio "
        "near 1.6 mean it is not yet an unqualified bright-source predictor. "
        "The first-null region near 20--30′ is poorly predicted; coherence "
        "partially returns in the next sidelobe. F24 shows the same radial "
        "degradation across the 20 movers.\n"
        "\n"
        "The dB ratio is a diagnostic, not a multiplicative correction. When "
        "CASSBEAM predicts only 0.05--0.15 Jy and the residual floor is about "
        "0.2 Jy, magnitude ratios are biased upward. A correction would need "
        "coherent complex averaging, uncertainty propagation, and transfer "
        "across held-out antennas and cells. Phase means are exploratory and "
        "must not be read as a general outer phase correction.\n"
        "\n"
        "For a common scalar beam, $E_p(s)E_q(s)^{*}=|E(s)|^{2}$, so a shared "
        "voltage phase—and even a shared $\\pi$ lobe sign—cancels in ordinary "
        "Stokes I. Outer-lobe magnitude is the main imaging concern. Phase "
        "matters when antenna beams differ, for R/L differences and cross-polar "
        "terms, and for antenna-dependent Jones beams."
    ),
    (
        "## 9. R/L squint\n"
        "\n"
        "The publication estimator is the 20%-of-peak main-lobe power centroid, "
        "with independent RR and LL flags and weights. The full-raster centroid "
        "is a bias diagnostic only."
    ),
    (
        "## 10. C147-* offset ring\n"
        "\n"
        "Fields 1–8 were unused prediction data. Offsets come from `FIELD.PHASE_DIR`. "
        "The CASA fringe $\\exp(+2\\pi i[ul+vm+w(n-1)])$ is required. The already-written "
        "ring Q/U used all 64 channels and is not a frequency holdout."
    ),
    (
        "## 11. Frequency transfer\n"
        "\n"
        "This section is `not_run`. SPW 5 at 4.692 GHz remains sealed."
    ),
    (
        "## 12. Experimental full Jones\n"
        "\n"
        "The locked mount-frame CASSBEAM full-Jones prediction is labelled "
        "experimental. The diagonal prediction is the null. A correlation near "
        "zero and residual power near one means the template explains almost none "
        "of the RL/LR variance. Because the observed cloud is larger than the "
        "predicted signal, this is a sensitivity-limited non-detection, not a "
        "clean rejection of CASSBEAM leakage physics."
    ),
    (
        "## 13. Conclusions and limitations\n"
        "\n"
        "Use CASSBEAM as the diagonal structural prior. Trust the main beam most "
        "strongly. Treat the mid beam as qualified. Attach substantial model "
        "uncertainty outside it. Keep explicit bright out-of-field source terms, "
        "because CASSBEAM alone can misestimate their apparent flux by factors of "
        "roughly two or more at particular outer locations. Do not infer a general "
        "outer phase correction from the current phase means.\n"
        "\n"
        "The next scientific work is a validation-selected low-order diagonal "
        "correction. Outer-field corrections should be admitted only if they "
        "transfer across movers and spatial holdouts; otherwise bright outer "
        "sources should get source-specific nuisance terms or peeling rather than "
        "a globally corrected beam. Full Jones remains an experimental "
        "non-detection. SPW 5 remains sealed."
    ),
]

CODE = [
    """from pathlib import Path

from sl1mjax.beam_validation_outputs import default_bundle_root, load_bundle
from sl1mjax.beam_validation_plots import write_all_figures
from sl1mjax.beam_validation_statistics import (
    claim_table,
    copolar_correlation,
    copolar_slope,
    crosshand_non_detection,
    frequency_copolar_series,
    offset_ring_diagonal_closure,
    publication_squint_pair,
    residual_power_table,
)

bundle = load_bundle()
figure_dir = Path("docs/assets/vla_c_band_beam_validation")
written = write_all_figures(bundle, figure_dir)
print(bundle.root)
print("publication", bundle.manifest["publication_version"])
print("bundle_sha256", bundle.manifest["bundle_sha256"])
print("schema", bundle.manifest["schema_version"])
print("wrote", len(written), "figure files")
""",
    """rows = claim_table(bundle)
print(f"{'id':<4} {'status':<10} {'class':<28} title")
for row in rows:
    print(f"{row['id']:<4} {row['status']:<10} {str(row.get('support_class','')):<28} {row['title']}")
""",
    """residuals = residual_power_table(bundle.holoraster_channel32)
print("Quoted % residual power is not a flux error.")
print("region            RR power   RR RMS vis   RR med|dV|/I   LL power   LL RMS vis   LL med|dV|/I")
for name in ("main_lobe", "mid", "outer_diagnostic", "all"):
    row = residuals[name]
    print(
        f"{name:<17} {100*row['rr']:7.2f}%  {100*row['rr_rms']:9.2f}%  "
        f"{100*row['rr_median_abs_over_i']:11.2f}%  "
        f"{100*row['ll']:7.2f}%  {100*row['ll_rms']:9.2f}%  "
        f"{100*row['ll_median_abs_over_i']:11.2f}%"
    )
print("RR slope", copolar_slope(bundle.holoraster_channel32, "rr"))
print("LL slope", copolar_slope(bundle.holoraster_channel32, "ll"))
print("RR corr", copolar_correlation(bundle.holoraster_channel32, "rr"))
print("LL corr", copolar_correlation(bundle.holoraster_channel32, "ll"))
print("n_rows", bundle.holoraster_channel32["n_rows"])
print("frequency_hz", bundle.holoraster_channel32["frequency_hz"])
print("source I_model Jy", bundle.residual_strata.get("source_i_jy"))
""",
    """obs = bundle.observation
print(obs["project"], "SB", obs["scheduling_block"], "EB", obs["execution_block"])
print("version-1 frequency_hz", obs["version1_frequency_hz"])
print("SPW 5", obs["spw5_status"], obs["spw5_frequency_hz"])
print("POINTING_OFFSET frame", obs["pointing_offset_frame"])
print("ON_SOURCE is a selection field", obs["on_source_is_selection"])
print("references", obs["reference_antenna_names"])
print("occupancy", obs["occupancy"])
""",
    """cal = bundle.calibration
print("refant", cal["reference_antenna"], "calwt", cal["calwt"])
print("diagonal parang", cal["diagonal_parang"], "fullpol parang", cal["fullpol_parang"])
print("HOLORASTER apply from", cal["apply_holoraster_from"])
print("C147-* apply from", cal["apply_c147_offset_from"])
print("source model", cal["source_model"])
print("3C286", cal["three_c286"])
""",
    """gates = bundle.convention_gates
print("software gates passed", gates["software_gates_passed"])
print("convention", gates["cassbeam_convention"])
print("not physical beam evidence", gates["not_physical_beam_evidence"])
for name, row in (gates.get("cumulative_residuals") or {}).items():
    print(name, row.get("median_rel_l2"))
""",
    """figure_dir = Path("docs/assets/vla_c_band_beam_validation")
written = write_all_figures(bundle, figure_dir)
print("wrote", len(written), "files under", figure_dir)
""",
    """from IPython.display import Image, display

def show(*names):
    root = Path("docs/assets/vla_c_band_beam_validation")
    for name in names:
        display(Image(filename=str(root / name)))

show("01_observation_timeline.png", "02_raster_occupancy.png", "03_antenna_roles.png")
""",
    """show("04_casa_jax_operator_residual.png", "05_onaxis_amplitude.png")
""",
    """show(
    "08_cassbeam_scatter_main_lobe.png",
    "10_spatial_measured_cassbeam_residual.png",
    "12_residual_vs_radius.png",
    "15_residual_strata.png",
)
strata = bundle.residual_strata
print("mask", strata.get("mask"))
print("raster_family", strata.get("raster_family"))
for key in ("mover", "reference", "pass"):
    print(key, [(row.get("name"), row.get("n"), row.get("median_abs")) for row in strata.get(key) or ()][:8])
""",
    """show(
    "20_amplitude_db.png",
    "21_signed_complex_cuts.png",
    "22_masked_phase.png",
    "23_radial_coherence.png",
    "24_antenna_coherence.png",
    "25_bright_source_examples.png",
)
coh = bundle.radial_coherence
print("phase_status", coh.get("phase_status"), "amp_floor_jy", coh.get("amp_floor_jy"))
print("extent", coh.get("raster_extent_arcmin"))
print("map phase >40'", coh.get("map_phase_beyond_40_arcmin"))
for hand in ("rr", "ll"):
    print(hand, [(row["r_mid_arcmin"], row["correlation_abs"], row["circular_phase_deg"]) for row in coh.get(hand) or ()])
print("movers", [row["name"] for row in bundle.antenna_coherence.get("movers") or ()])
print("examples", [row.get("label") for row in bundle.bright_source_examples.get("examples") or ()])
""",
    """squint = publication_squint_pair(bundle.squint)
for name, row in squint.items():
    print(name, "sep", row["separation_arcmin"], "memo", row.get("memo195_separation_arcmin"),
          "full_raster", row["full_raster_separation_arcmin"], "n", row.get("n_rr"), row.get("n_ll"))
show("13_squint_mainlobe_20pct.png", "16_spw4_frequency.png")
""",
    """ring = offset_ring_diagonal_closure(bundle.offset_ring)
print(ring)
print("inner/sealed scores", bundle.offset_ring["scores"])
print("Q/U is frequency holdout", ring["q_u_is_frequency_holdout"])
show("17_c147_offset_ring.png")
""",
    """print("SPW 5 section status: not_run")
show("19_spw5_sealed.png")
""",
    """cross = crosshand_non_detection(bundle.holoraster_channel32)
print(cross)
print("predicted RL median", cross["rl_median_abs_pred"], "observed", cross["rl_median_abs_obs"])
show("18_experimental_crosshand.png")
""",
    """print("next artifact: cassbeam_diagonal_low_order_correction")
print("convention search reopened", bundle.convention_gates.get("cassbeam_convention", {}).get("note"))
for row in claim_table(bundle):
    print(row["id"], row["status"], row["evidence"])
""",
]


def build_notebook() -> nbformat.NotebookNode:
    cells = [
        nbformat.v4.new_markdown_cell(MARKDOWN[0]),
        nbformat.v4.new_code_cell(CODE[0]),
        nbformat.v4.new_markdown_cell(MARKDOWN[1]),
        nbformat.v4.new_code_cell(CODE[1]),
        nbformat.v4.new_code_cell(CODE[2]),
        nbformat.v4.new_markdown_cell(MARKDOWN[2]),
        nbformat.v4.new_markdown_cell(MARKDOWN[3]),
        nbformat.v4.new_code_cell(CODE[3]),
        nbformat.v4.new_code_cell(CODE[7]),
        nbformat.v4.new_markdown_cell(MARKDOWN[4]),
        nbformat.v4.new_code_cell(CODE[4]),
        nbformat.v4.new_markdown_cell(MARKDOWN[5]),
        nbformat.v4.new_code_cell(CODE[5]),
        nbformat.v4.new_code_cell(CODE[8]),
        nbformat.v4.new_markdown_cell(MARKDOWN[6]),
        nbformat.v4.new_code_cell(CODE[9]),
        nbformat.v4.new_code_cell(CODE[10]),
        nbformat.v4.new_markdown_cell(MARKDOWN[7]),
        nbformat.v4.new_code_cell(CODE[11]),
        nbformat.v4.new_markdown_cell(MARKDOWN[8]),
        nbformat.v4.new_code_cell(CODE[12]),
        nbformat.v4.new_markdown_cell(MARKDOWN[9]),
        nbformat.v4.new_code_cell(CODE[13]),
        nbformat.v4.new_markdown_cell(MARKDOWN[10]),
        nbformat.v4.new_code_cell(CODE[14]),
        nbformat.v4.new_markdown_cell(MARKDOWN[11]),
        nbformat.v4.new_code_cell(CODE[15]),
    ]
    notebook = nbformat.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    notebook["cells"] = cells
    return notebook


def main() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(build_notebook(), NOTEBOOK_PATH)
    print(NOTEBOOK_PATH)


if __name__ == "__main__":
    main()
