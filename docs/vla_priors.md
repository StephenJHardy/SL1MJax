# Independent VLA a-priori calibration

SL1MJax represents gain-curve, opacity, and requantizer corrections as fixed,
diagonal antenna Jones terms. `CalibrationChain` composes those terms ahead of
the existing solved G/K/B `CalibrationSolution`; only G/K/B parameters are
optimized. Visibility amplitudes and weights remain separate, and RQ is applied
with CASA's `calwt=False` convention.

## Portable inputs

MeasurementSet extraction preserves:

- complete spectral-window and data-description coordinates;
- antenna ECEF positions and field directions for lazy elevation/airmass;
- WEATHER samples, including temperature and dew point;
- SYSPOWER switched-difference, switched-sum, and requantizer gains;
- CALDEVICE noise temperatures, efficiencies, and load names;
- STATE intent plus SIG/REF/CAL/LOAD flags.

The canonical dataset schema is version 1.2 and remains readable back to 1.0.

## Independent generators

- `generate_vla_gain_curve` selects dated, frequency- and antenna-bounded
  coefficients from the complete NRAO VLA `GainCurves` table pinned at
  `casa-data` commit `65a746f9e666`. Antenna-specific values override the
  nominal antenna-0 curve. Unsupported dates or frequencies are invalid rather
  than silently assigned a curve.
- `estimate_vla_zenith_opacity` implements the VLA seasonal and weather-station
  PWV equations and a portable dry/wet opacity approximation. Its provenance
  records measured and seasonal PWV separately, the requested/used mixture,
  model version, and uncertainty.
- `generate_vla_requantizer` reads the RQ-only voltage gain from
  `SYSPOWER.REQUANTIZER_GAIN` in bounded chunks, requires CALDEVICE as CASA
  does, and emits the stored voltage gain directly, matching the CASA 6.5
  `G EVLASWPOW` RQ tables used by the SRDP corpus. Weight scaling is disabled;
  switched-power compression and Tsys weight calibration are not performed.

`write_calibration_chain` and `read_calibration_chain` serialize terms and
validity masks without requiring CASA.

## CASA references and validation

`import_casa_prior_table` reads EGainCurve, TOpac, and G EVLASWPOW tables only
as comparison oracles. The seven-case fixture can be regenerated with:

```bash
uv run scripts/build_vla_prior_fixtures.py \
  /path/to/extracted/calibration-tables \
  tests/fixtures/vla_priors_srdp_golden.json
```

The corpus analyzer inventories the same prior tables alongside final K/B/G
tables. Once a compact canonical export of a raw SRDP calibrator execution is
available, run the strict coordinate-aligned validation:

```bash
uv run scripts/validate_vla_priors.py /path/to/raw-export \
  --gain-curve-table /path/to/gc.tbl \
  --opacity-table /path/to/opac.tbl \
  --requantizer-table /path/to/rq.tbl \
  --output outputs/vla_prior_validation.json
```

The validator enforces relative RMS below `1e-3` for gain curve and RQ, below
`5e-3` for opacity, below `2e-3` for combined prior-corrected visibilities, and
identical validity flags. Opacity provenance is retained in the report so a
weather/model-source difference is visible rather than absorbed into a fitted
term.

The release gate was run on raw 3C286 scan 2 from
`23B-041.sb44952728.eb45179764.60333.33735652777`, sampled at one channel per
SPW and every 100th row. Across all 16 SPWs, gain-curve and RQ relative RMS were
zero, maximum opacity relative RMS was `4.01e-5`, maximum combined normalized
complex RMS was `8.01e-5`, and there were no validity-flag mismatches. The full
report is `outputs/vla_prior_validation_23b041.json`.
