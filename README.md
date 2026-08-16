# SL1MJax

SL1MJax is a differentiable radio-interferometric sky–instrument model. It uses
pure JAX measurement equations and Optax gradient inference while treating
polarization products, weights, flags, and provenance as first-class data.

The first release fits a positive regular Stokes-I grid with a fixed scalar
instrument response. Its APIs are designed for later joint calibration,
multiscale components, full Stokes/Jones models, and temporal inference.

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

Synthetic end-to-end example:

```bash
uv run sl1mjax simulate synthetic.zarr --basis linear
uv run sl1mjax image synthetic.zarr image.fits --steps 500
```

See `docs/scientific_conventions.md` for phase, polarization, objective, and
precision conventions.
