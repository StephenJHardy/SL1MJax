#!/usr/bin/env python3
"""Point live v2 entry points at the survey v3 pack after the snapshot exists."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sl1mjax.beam_validation_outputs import PUBLICATION_VERSION_V2

ROOT = Path(__file__).resolve().parents[1]
V2_BUNDLE = ROOT / "src" / "sl1mjax" / "data" / PUBLICATION_VERSION_V2
V2_SHA256 = "fa01cc15f715e75ace7088ed1bc16a7fac75cfafd2b5b68fda071986f024d5d9"
SNAPSHOT = ROOT / "docs" / "vla_c_band_beam_validation_v2_snapshot"
LIVE_NOTEBOOK = ROOT / "notebooks" / "vla_c_band_beam_validation.ipynb"
LIVE_PAGE = ROOT / "docs" / "vla_c_band_beam_validation.md"


def _require_snapshot() -> None:
    for name in (
        "README.md",
        "vla_c_band_beam_validation.md",
        "vla_c_band_beam_validation.ipynb",
    ):
        path = SNAPSHOT / name
        if not path.is_file():
            raise RuntimeError(f"v2 snapshot missing {path}")


def _require_frozen_v2() -> None:
    manifest = json.loads((V2_BUNDLE / "manifest.json").read_text(encoding="utf-8"))
    digest = str(manifest.get("bundle_sha256") or "")
    if digest != V2_SHA256:
        raise RuntimeError(f"frozen v2 checksum moved: {digest}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    arguments = parser.parse_args()
    _require_snapshot()
    _require_frozen_v2()

    from write_vla_c_band_beam_validation_v3_notebook import write_notebook
    from render_vla_c_band_beam_validation_v3 import render

    write_notebook(LIVE_NOTEBOOK)
    page = render(live=True)
    preface = (
        "<!-- Live survey page. The frozen v2 SPW-4 account is snapshotted in "
        "`docs/vla_c_band_beam_validation_v2_snapshot/` and "
        "`src/sl1mjax/data/vla_c_band_beam_validation_v2`. -->\n\n"
    )
    LIVE_PAGE.write_text(preface + page, encoding="utf-8")
    if arguments.execute:
        import nbformat
        from nbclient import NotebookClient

        notebook = nbformat.read(LIVE_NOTEBOOK, as_version=4)
        client = NotebookClient(notebook, timeout=120, kernel_name="python3")
        client.execute()
        nbformat.write(notebook, LIVE_NOTEBOOK)
    print(LIVE_NOTEBOOK)
    print(LIVE_PAGE)


if __name__ == "__main__":
    main()
