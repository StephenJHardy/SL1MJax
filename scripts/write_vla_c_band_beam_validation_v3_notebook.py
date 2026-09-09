#!/usr/bin/env python3
"""Write the survey v3 notebook. Does not replace the live v2 notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat

NOTEBOOK_PATH = (
    Path(__file__).resolve().parents[1] / "notebooks" / "vla_c_band_beam_validation_v3.ipynb"
)

MARKDOWN = [
    (
        "# VLA C-band diagonal survey (v3)\n"
        "\n"
        "This notebook is the executable band-survey account. It starts from the "
        "compact checksummed `vla_c_band_beam_validation_v3` bundle. It does not "
        "read a Measurement Set, invoke CASA, follow Bacchus paths, or overwrite "
        "the preserved v2 SPW-4 publication.\n"
        "\n"
        "Full Jones is diagnostic and is not a gate. Residual Jones is not applied. "
        "This is not a production beam freeze."
    ),
    (
        "## What can be used for 3C391 now\n"
        "\n"
        "Use the opt-in selector `evla_c_diagonal_survey_v1` with the catalog root "
        "and digest recorded in the bundle. Exact native frequency only. No "
        "nearest-plane substitution and no Airy fallback. A defensible Stokes-I "
        "trial uses the sampled accepted centres around 4436–5820 MHz, not a "
        "continuous channel range. Pass A is complete; Pass B is limited to "
        "lower-C SPWs 4–6. Numerical qualification is incomplete. Upper-C is "
        "archive-integrity unverified. Imaging uses the fixed 4536/4598/4662 MHz "
        "nodes until more native planes exist."
    ),
    (
        "## What changed since SPW 4\n"
        "\n"
        "The v2 refresh reused a channel-32 residual-Jones plane. This survey does "
        "not. SPW 5 is a declared diagonal frequency-transfer test, not a "
        "full-Jones reopen. The large SPW-4 channel-24 LL and channel-40 RR "
        "excursions survive without residual Jones."
    ),
]

CODE = [
    """from pathlib import Path
import json
from IPython.display import Image, display

root = Path("src/sl1mjax/data/vla_c_band_beam_validation_v3")
manifest = json.loads((root / "manifest.json").read_text())
if manifest["publication_version"] != "vla_c_band_beam_validation_v3":
    raise RuntimeError(f"unexpected publication {manifest['publication_version']}")
if "vla_c_band_beam_validation_v2" in str(root):
    raise RuntimeError("refusing to load the frozen v2 bundle")
suitability = json.loads((root / "suitability_table.json").read_text())
handoff = json.loads((root / "imaging_handoff.json").read_text())
ledger = json.loads((root / "supersession_ledger.json").read_text())
phase5 = json.loads((root / "phase5_diagnostics.json").read_text())
slot_status = json.loads((root / "slot_status.json").read_text())
print(root.resolve())
print("publication", manifest["publication_version"])
print("bundle_sha256", manifest["bundle_sha256"])
print("catalog_digest", manifest.get("catalog_digest"))
print("production_accepted", manifest["production_accepted"])
print("full_jones_is_gate", manifest["full_jones_is_gate"])
print("residual_jones", manifest["residual_jones_policy"])
print("interpolation", manifest["interpolation_policy"])
print("n_scored", suitability["n_scored"])
print("slot_status", slot_status["n_scored"], "of", slot_status["n_slots"], slot_status["counts"])
print("pass_a_survey_complete", slot_status.get("pass_a_survey_complete"))
""",
    """print("usable main-lobe slots")
print("exec     SPW  ch   MHz     RR      LL")
for slot in handoff["usable_main_lobe_slots"]:
    mhz = slot["frequency_hz"] / 1.0e6
    print(
        f"{slot.get('execution') or '':<8} {slot['spectral_window_id']:>3} "
        f"{slot['channel']:>3} {mhz:7.0f} {slot['main_lobe_rr']:.5f} "
        f"{slot['main_lobe_ll']:.5f}"
    )
print()
print("limitation slots")
for slot in handoff["limitation_slots"]:
    mhz = slot["frequency_hz"] / 1.0e6
    print(
        f"{slot.get('execution') or '':<8} {slot['spectral_window_id']:>3} "
        f"{slot['channel']:>3} {mhz:7.0f} {slot['main_lobe_rr']:.5f} "
        f"{slot['main_lobe_ll']:.5f} {slot.get('status')}"
    )
""",
    """print("supersession")
for entry in ledger["entries"]:
    print(f"- {entry['item']}: {entry['status']} ({entry['reason']})")
print()
print("smoke command")
print(handoff.get("smoke_command") or handoff["command"])
print()
print("mosaic command")
print(handoff.get("mosaic_command") or handoff["command"])
""",
    """print("phase5 slots", phase5["n_slots"], "interpolation_validated", phase5["interpolation_validated"])
print("freq_MHz  SPW  coarseRR  modelRR  measuredRR")
for slot in phase5["slots"]:
    coarse = (slot.get("coarse_radial_minima") or slot.get("nulls") or {}).get("RR") or {}
    model = ((slot.get("model_first_minima") or {}).get("RR") or {})
    measured = (slot.get("measured_first_null") or {}).get("RR") or {}
    coarse_txt = (
        f"{coarse['r_lo_arcmin']:.0f}-{coarse['r_hi_arcmin']:.0f}"
        if coarse.get("status") == "coarse_minimum"
        else coarse.get("status")
    )
    model_r = model.get("median_r_arcmin")
    model_txt = f"{float(model_r):.2f}" if model_r is not None and float(model_r) == float(model_r) else "nan"
    print(
        f"{slot['frequency_hz']/1e6:8.0f} {slot.get('spectral_window_id')}  "
        f"{coarse_txt}  {model_txt}  {measured.get('status')}"
    )
""",
    """figures = [
    "frequency_radius_suitability.png",
    "pass_b_mainlobe.png",
    "nulls_vs_frequency.png",
    "spw02_channel32_spatial.png",
    "spw02_channel32_signed_cuts.png",
    "spw03_channel32_spatial.png",
    "spw03_channel32_signed_cuts.png",
    "spw04_channel32_spatial.png",
    "spw04_channel32_signed_cuts.png",
    "spw05_channel32_spatial.png",
    "spw05_channel32_signed_cuts.png",
    "spw06_channel32_spatial.png",
    "spw06_channel32_signed_cuts.png",
    "spw07_channel32_spatial.png",
    "spw07_channel32_signed_cuts.png",
    "spw15_channel32_spatial.png",
    "upper_c_spw00_channel32_spatial.png",
    "upper_c_spw01_channel32_spatial.png",
    "upper_c_spw15_channel32_spatial.png",
]
for name in figures:
    path = root / "figures" / name
    if not path.is_file():
        print("missing", name)
        continue
    print(name, path.stat().st_size)
    display(Image(filename=str(path)))
""",
]


def write_notebook(path: Path = NOTEBOOK_PATH) -> Path:
    notebook = nbformat.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    for text in MARKDOWN:
        notebook.cells.append(nbformat.v4.new_markdown_cell(text))
    for source in CODE:
        notebook.cells.append(nbformat.v4.new_code_cell(source))
    path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, path)
    return path


def main() -> None:
    print(write_notebook())


if __name__ == "__main__":
    main()
