"""Execute the validation notebook from a clean kernel.

The working directory is the repository root. The Measurement Set, CASA, and
Bacchus paths are not required.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient

from sl1mjax.beam_validation_outputs import default_bundle_root, load_bundle

NOTEBOOK_PATH = Path("notebooks/vla_c_band_beam_validation.ipynb")


def execute_notebook(path: Path, *, timeout_s: int = 300) -> Path:
    load_bundle(default_bundle_root())
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=timeout_s,
        kernel_name="python3",
        resources={"metadata": {"path": str(Path.cwd())}},
    )
    client.execute()
    nbformat.write(notebook, path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", type=Path, default=NOTEBOOK_PATH)
    parser.add_argument("--timeout", type=int, default=300)
    arguments = parser.parse_args()
    print(execute_notebook(arguments.notebook, timeout_s=arguments.timeout))


if __name__ == "__main__":
    main()
