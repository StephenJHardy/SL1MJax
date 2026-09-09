#!/usr/bin/env python3
"""Render the survey summary from the v3 pack. Does not overwrite the v2 page."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "src" / "sl1mjax" / "data" / "vla_c_band_beam_validation_v3"
DOC_PATH = ROOT / "docs" / "evla_c_diagonal_survey" / "SURVEY.md"


def _pct(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def render(bundle: Path = BUNDLE, *, live: bool = False) -> str:
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    if manifest["publication_version"] != "vla_c_band_beam_validation_v3":
        raise RuntimeError("refusing to render a non-v3 bundle as the survey page")
    suitability = json.loads((bundle / "suitability_table.json").read_text(encoding="utf-8"))
    slot_status = {}
    if (bundle / "slot_status.json").is_file():
        slot_status = json.loads((bundle / "slot_status.json").read_text(encoding="utf-8"))
    handoff = json.loads((bundle / "imaging_handoff.json").read_text(encoding="utf-8"))
    ledger = json.loads((bundle / "supersession_ledger.json").read_text(encoding="utf-8"))
    rows = []
    for slot in suitability["slots"]:
        mhz = float(slot["frequency_hz"]) / 1.0e6
        both = "yes" if slot.get("main_lobe_both_hands_accepted") else "no"
        execution = slot.get("execution") or (
            "upper_c" if str(slot.get("file") or "").startswith("upper_c_") else "lower_c"
        )
        rows.append(
            f"| {execution} | {slot.get('spectral_window_id')} | {slot.get('channel')} | "
            f"{mhz:.0f} | {_pct(slot['main_lobe_rr'])} / {_pct(slot['main_lobe_ll'])} | "
            f"{both} | {slot.get('status')} |"
        )
    ledger_rows = "\n".join(
        f"- **{entry['item']}** — {entry['status']}: {entry['reason']}"
        for entry in ledger["entries"]
    )
    return (
        "# EVLA-C diagonal survey summary (v3)\n\n"
        "Rendered from `src/sl1mjax/data/vla_c_band_beam_validation_v3`. "
        + (
            "The frozen v2 SPW-4 page is snapshotted in "
            "`docs/vla_c_band_beam_validation_v2_snapshot/`. "
            if live
            else "This page does not replace `docs/vla_c_band_beam_validation.md` (v2). "
        )
        + "Full Jones is diagnostic and is not a gate.\n\n"
        f"Bundle `{manifest['bundle_sha256'][:12]}…`. Catalog digest "
        f"`{manifest.get('catalog_digest') or 'pending'}`. "
        f"Residual Jones: {manifest['residual_jones_policy']}. "
        "Interpolation: refused.\n\n"
        "## Slot status\n\n"
        + (
            f"{slot_status.get('n_scored', 0)} of {slot_status.get('n_slots', 0)} "
            f"declared science slots are scored. "
            "Pass A is complete; Pass B is limited to lower-C SPWs 4–6. "
            "A `scientifically-qualified` label is the empirical 1% main-lobe "
            "residual-power cut only; numerical qualification is incomplete. "
            "Upper-C claims are archive-integrity unverified. "
            + f"Counts: `{slot_status.get('counts')}`.\n\n"
            if slot_status
            else "Slot-status ledger is not in this bundle yet.\n\n"
        )
        + "## Scored slots\n\n"
        "| exec | SPW | ch | MHz | RR / LL | Both ≤ 1% | status |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        + "\n".join(rows)
        + "\n\n"
        "## Imaging commands\n\n"
        "Smoke (completed on Bacchus):\n\n"
        "```bash\n"
        f"{handoff.get('smoke_command') or handoff['command']}\n"
        "```\n\n"
        "Seven-pointing mosaic (not a long optimized reconstruction):\n\n"
        "```bash\n"
        f"{handoff['command']}\n"
        "```\n\n"
        "## Supersession\n\n"
        f"{ledger_rows}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=BUNDLE)
    parser.add_argument("--output", type=Path, default=DOC_PATH)
    arguments = parser.parse_args()
    if "vla_c_band_beam_validation_v2" in str(arguments.output):
        raise RuntimeError("refusing to overwrite the frozen v2 summary")
    text = render(arguments.bundle)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(text, encoding="utf-8")
    print(arguments.output)


if __name__ == "__main__":
    main()
