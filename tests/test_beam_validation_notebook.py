from __future__ import annotations

from pathlib import Path

import nbformat

from sl1mjax.beam_validation_outputs import (
    ALLOWED_PUBLICATION_VERSIONS,
    default_bundle_root,
    load_bundle,
)

NOTEBOOK = Path("notebooks/vla_c_band_beam_validation.ipynb")
REQUIRED_HEADINGS = (
    "Executive summary",
    "Why a measured voltage beam is needed",
    "THOL0001 observation",
    "Calibration and source model",
    "Convention and software correctness",
    "Corrected coordinate and EVLA-C feed update",
    "Direct HOLORASTER comparison",
    "R/L squint",
    "C147-* offset ring",
    "Frequency transfer",
    "Experimental full Jones",
    "Conclusions and limitations",
)


def test_notebook_has_required_sections() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    text = "\n".join(
        "".join(cell.source) if isinstance(cell.source, list) else cell.source
        for cell in notebook.cells
        if cell.cell_type == "markdown"
    )
    for heading in REQUIRED_HEADINGS:
        assert heading in text
    sources = [
        "".join(cell.source) if isinstance(cell.source, list) else cell.source
        for cell in notebook.cells
        if cell.cell_type == "code"
    ]
    joined = "\n".join(sources)
    assert "0.0064" not in joined
    assert "0.515" not in joined
    assert "load_bundle" in joined
    assert "Measurement Set" not in joined or "does not read" in text


def test_publication_bundle_loads_without_bacchus_if_present() -> None:
    root = default_bundle_root()
    if not (root / "manifest.json").is_file():
        return
    bundle = load_bundle(root)
    assert bundle.manifest["publication_version"] in ALLOWED_PUBLICATION_VERSIONS
    assert "holoraster_channel32.json" in bundle.manifest["files"]


def test_committed_notebook_was_executed() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    code = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert code
    assert all(cell.get("execution_count") for cell in code)
    assert all(cell.get("outputs") for cell in code)


def test_notebook_source_does_not_import_casa_or_ms() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    joined = "\n".join(
        "".join(cell.source) if isinstance(cell.source, list) else cell.source
        for cell in notebook.cells
        if cell.cell_type == "code"
    )
    assert "casacore" not in joined
    assert "tables.table" not in joined
    assert ".ms" not in joined
    assert "/media/stephen" not in joined
