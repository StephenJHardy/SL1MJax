# SL1MJax

SL1MJax is a differentiable radio-interferometric sky–instrument model. It uses
pure JAX measurement equations and Optax gradient inference while treating
polarization products, weights, flags, and provenance as first-class data.

The package fits positive regular Stokes-I grids and now includes staged,
diagonal RR/LL calibration (`G`, `K`, and `B`) with structured holdouts,
portable CASA-solution import, flux transfer, and residual/closure diagnostics.
Cross-hand polarization and direction-dependent calibration remain deferred.

MeasurementSet support is an optional, initially VLA-oriented extractor:

```text
MeasurementSet + casacore -> canonical Zarr -> core-only JAX inference
```

The canonical dataset and all scientific modelling are independent of CASA.

## Development

```bash
uv sync
uv run pytest
uv run ruff check .
uv run mypy
```

On a host with system casacore libraries:

```bash
brew tap casacore/tap
brew install casacore
CMAKE_ARGS="-DCASACORE_ROOT_DIR=$(brew --prefix casacore)" uv sync --extra ms
uv run sl1mjax ingest observation.ms observation.zarr
```

The project uses Python 3.14 so its Python/Boost.Python ABI matches the current
Homebrew casacore formula on Apple Silicon.

With the CASA application installed, create a compact known-truth VLA fixture:

```bash
/Applications/CASA.app/Contents/MacOS/casa --nologger --nogui \
  -c scripts/create_vla_fixture.py
uv run sl1mjax ingest \
  outputs/casa_vla_fixture/casa_vla_fixture.vla.d.ms \
  outputs/casa_vla_fixture.zarr --column DATA
```

Large real datasets can be bounded explicitly at extraction time:

```bash
uv run sl1mjax ingest observation.ms selected.zarr \
  --channels 2 --row-stride 100
```

Selections are stored in provenance; they are never silently applied by the
model.

The committed 3C391 fixture runs both CASA-application parity and independent
JAX solve/flux-transfer gates without CASA:

```bash
uv run pytest tests/test_calibration_3c391.py
```

With the prepared tutorial MS available, compare CASA-corrected target imaging
against calibration solved and transferred in JAX:

```bash
uv run scripts/image_3c391_target.py /path/to/3c391_ctm_mosaic_10s_spw0.ms
```

See `docs/3c391_target_imaging.md` for measured results and limitations.

Synthetic end-to-end example:

```bash
uv run sl1mjax simulate synthetic.zarr --basis linear
uv run sl1mjax image synthetic.zarr image.fits --steps 500
```

See `docs/scientific_conventions.md` for phase, polarization, objective, and
precision conventions.
