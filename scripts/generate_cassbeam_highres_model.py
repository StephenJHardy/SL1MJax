"""Generate the native-channel high-resolution CASSBEAM C-band artifact.

Run this on the pinned Linux host.  It is an offline artifact generator;
the SL1MJax imaging runtime must not invoke CASSBEAM.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def _copy_or_require_equal(source: Path, destination: Path) -> None:
    if destination.exists():
        if source.read_bytes() != destination.read_bytes():
            raise ValueError(f"refusing to replace different file {destination}")
        return
    shutil.copy2(source, destination)


def _expected_rows(gridsize: int) -> int:
    half = gridsize // 2
    if half % 2:
        half += 1
    aperture_n = 2 * half
    crop_size = aperture_n - 2 * (aperture_n // 4) + 1
    return crop_size * crop_size


def _line_count(path: Path) -> int:
    with path.open("rb") as stream:
        return sum(chunk.count(b"\n") for chunk in iter(lambda: stream.read(1024 * 1024), b""))


def generate(
    *,
    binary: Path,
    base_input: Path,
    geometry: Path,
    output_dir: Path,
    frequencies_mhz: list[int],
    reference_frequencies_mhz: set[int],
    gridsize: int,
    pixelsperbeam: int,
) -> None:
    if gridsize < 32 or gridsize % 2:
        raise ValueError("gridsize must be even and at least 32")
    if pixelsperbeam <= 0:
        raise ValueError("pixelsperbeam must be positive")
    reference = output_dir / "reference"
    reference.mkdir(parents=True, exist_ok=True)
    (output_dir / "spw4").mkdir(parents=True, exist_ok=True)
    copied_input = reference / "base.in"
    copied_geometry = reference / "vla_geom"
    _copy_or_require_equal(base_input, copied_input)
    _copy_or_require_equal(geometry, copied_geometry)
    expected_rows = _expected_rows(gridsize)
    for frequency_mhz in frequencies_mhz:
        group = "reference" if frequency_mhz in reference_frequencies_mhz else "spw4"
        prefix = output_dir / group / f"vla-cband-{frequency_mhz}-g{gridsize}-p{pixelsperbeam}"
        data_path = prefix.with_suffix(".jones.dat")
        params_path = prefix.with_suffix(".params")
        if data_path.exists() and params_path.exists():
            if _line_count(data_path) != expected_rows:
                raise ValueError(f"existing output has the wrong size: {data_path}")
            continue
        if data_path.exists() or params_path.exists():
            raise ValueError(f"refusing incomplete existing output {prefix}")
        command = [
            str(binary),
            str(copied_input),
            f"geom={copied_geometry}",
            f"freq={frequency_mhz / 1000.0:.3f}",
            f"gridsize={gridsize}",
            f"pixelsperbeam={pixelsperbeam}",
            f"out={prefix}",
            "compute=jp",
        ]
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
        prefix.with_suffix(".log").write_text(completed.stdout + completed.stderr, encoding="utf-8")
        if completed.returncode:
            raise RuntimeError(f"CASSBEAM failed for {frequency_mhz} MHz")
        if _line_count(data_path) != expected_rows:
            raise ValueError(f"generated output has the wrong size: {data_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=Path("/usr/bin/cassbeam"))
    parser.add_argument("--base-input", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gridsize", type=int, default=1024)
    parser.add_argument("--pixelsperbeam", type=int, default=32)
    arguments = parser.parse_args()
    spw4 = list(range(4500, 4628, 2))
    reference = {4692}
    generate(
        binary=arguments.binary,
        base_input=arguments.base_input,
        geometry=arguments.geometry,
        output_dir=arguments.output_dir,
        frequencies_mhz=[*spw4, *sorted(reference)],
        reference_frequencies_mhz=reference,
        gridsize=arguments.gridsize,
        pixelsperbeam=arguments.pixelsperbeam,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
