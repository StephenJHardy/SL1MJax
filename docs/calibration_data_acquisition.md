# Calibration data acquisition

The acquisition ladder starts with public VLA calibration products from the
[NRAO archive](https://data.nrao.edu/portal/). The catalog query does not
require credentials; the archive portal handles authenticated downloads.

## C-band pilot

Build a reproducible 30-product shortlist:

```bash
uv run scripts/build_nrao_calibration_corpus.py \
  --output outputs/nrao_calibration_corpus
```

The default query uses:

- calibrator searches `3C286` and `3C48`;
- public EVLA observations with restorable calibration products;
- C band only, excluding mixed-band observations;
- compact C and D configurations;
- a balanced sample across calibrator and configuration;
- formal SRDP QA products first within each group, then evenly across observing
  date, with restorable products used where no formal SRDP product exists.

The generated files are:

- `candidates.csv`: all matching calibration products;
- `selected.csv` and `selected.json`: the balanced pilot cohort;
- `download_files.txt`: exact calibration archive filenames.

`qa_class` distinguishes formal SRDP products, staff-checked non-SRDP
reprocessing, and legacy restorable products. This matters because the archive
"Cals" filter is broader than the formal SRDP program.

## Login-assisted download

For each selected row, open `product_url` from `selected.csv`, log into the
NRAO archive, click the numbered entry in the **Cals** column, and download the
selected calibration tarball. The expected filename is `calibration_file`.

Store the downloads outside Git, for example:

```text
/Volumes/BagOfWinds/NRAO/srdp-cals/
```

Do not initially download the much larger raw visibility products. The
shortlist retains their sizes and execution-block identifiers so representative
raw data can be selected after inspecting calibration behavior.

## Inventory and extraction

Inventory downloaded archives without extracting them:

```bash
uv run scripts/inspect_srdp_calibration_archive.py \
  /Volumes/BagOfWinds/NRAO/srdp-cals/*.tar \
  --output outputs/srdp_calibration_inventory.json
```

To safely extract the products and inspect CASA calibration tables with
`python-casacore`:

```bash
uv run scripts/inspect_srdp_calibration_archive.py \
  /Volumes/BagOfWinds/NRAO/srdp-cals/*.tar \
  --extract-to /Volumes/BagOfWinds/NRAO/srdp-cals/extracted \
  --output outputs/srdp_calibration_inventory.json
```

The inventory records table roots, G/K/B table metadata where available,
flags, QA/weblog products, time and interval ranges, antennas, spectral
windows, row counts and flagged fractions.

## Subsequent rungs

After the C-band distributions are understood:

1. Repeat the catalog build for L band to stress RFI and ionospheric behavior.
2. Repeat for K and Ka bands to stress atmospheric phase, opacity and pointing.
3. Select 5–10 representative execution blocks per band and download raw or
   calibrated visibilities, then extract only calibrator scans.
4. Add NVAS C/X-band calibrated data for historical distribution-shift tests.
5. Add small CASA regression datasets only as deterministic automated gates.
