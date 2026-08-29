# C-band full-Jones acquisition

Phase 6A of [`vla-beam-model-proposal.md`](vla-beam-model-proposal.md).
This note records what has been searched, what to request, and what must
be pinned before CASSBEAM output can be a convention oracle. It does not
freeze a Jones table. Runtime SL1MJax must not invoke CASA or CASSBEAM.

## Two acquisitions

1. Generate a nominal C-band Jones artifact with CASSBEAM on Bacchus.
2. Seek processed Perley holography only if accepted-mode residuals stay
   significant after the CASA `awp2` and 3C391 tests.

Do not download the Iheanetu L-band model or the plumber S-band
coefficients. They show useful representations in the wrong bands.

## BagOfWinds search (2026-08-29)

`/Volumes/BagOfWinds` is mounted on this machine. Searched with Spotlight
and `find` for `C-BEAM-*`, `CW-BEAM-*`, `CHOLO-*`, `UVHOL`, `cassbeam`,
and `vla-template`.

No compact Memo 195 products were found. The only holography-named hits
are CASA ALMA source files under `/Volumes/BagOfWinds/radio/casa`. Those
are not Perley C-band grids.

Do not download the ~150 GB `CHOLO` FITAB/FITTP databases while looking
for the compact products.

## Holography request

Memo 195 §4.4.1 identifies processed UVHOL products that are much smaller
than the calibrated databases:

- `C-BEAM-ffffpp01`: central oversampled grid;
- `CW-BEAM-ffffpp01`: sparse grid toward the fourth null;
- `pp` is `RR`, `RL`, `LR`, or `LL`.

The memo places the large calibrated databases on the NRAO FTP archive
and describes the compact `C-BEAM` products under an internal
`/users/rperley/...` path. There is no public indexed download for those
compact files.

Request first, at 4564 MHz and 4692 MHz, all four correlations, both the
central and sparse grids:

```
C-BEAM-4564RR01  C-BEAM-4564RL01  C-BEAM-4564LR01  C-BEAM-4564LL01
CW-BEAM-4564RR01 CW-BEAM-4564RL01 CW-BEAM-4564LR01 CW-BEAM-4564LL01
C-BEAM-4692RR01  C-BEAM-4692RL01  C-BEAM-4692LR01  C-BEAM-4692LL01
CW-BEAM-4692RR01 CW-BEAM-4692RL01 CW-BEAM-4692LR01 CW-BEAM-4692LL01
```

Also request the coordinate, polarisation, calibration, normalization,
and antenna-averaging metadata. Do not treat a copied filename as a
frozen reference until those conventions and a checksum are pinned.

A helpdesk or author request can point at EVLA Memo 195 §4.4.1 and ask
that the compact C-band products be exposed or copied. The FITAB/FITTP
`CHOLO` databases are a last resort.

## CASSBEAM generator

Run CASSBEAM in a pinned Linux or container environment, not inside the
imager and not in the SL1MJax virtualenv.

| Item | Pin |
| --- | --- |
| Preferred source | Debian `cassbeam` 1.1 ([1.1-5](https://sources.debian.org/src/cassbeam/1.1-5/), [ratt-ru/cassbeam](https://github.com/ratt-ru/cassbeam)) |
| Avoid | Original NRAO 1.0 at [~wbrisken/src](https://www.aoc.nrao.edu/~wbrisken/src/) (GLib 1.2 / old FFTW) |
| Packaged example | `examples/vla/vla-1500MHz.in` is 1.5 GHz L-band and is not a validated Jansky VLA C-band feed |
| Native sense | Transmit. Reciprocity is required before the receive Jones used here |
| Samples | Only the compact set in `orientation_oracle_sample_spec()` |

Installing the binary is not enough. Before accepting any output, pin:

- the exact 1.1 source or package version;
- VLA geometry and an EVLA C-band feed file;
- requested products and numerical settings;
- circular basis and transmit-to-receive conversion;
- output checksums.

The generic VLA template is a starting geometry only. Jagannathan et al.
2017 already note that L/S/C can need diffraction beyond geometric
optics. A C-band input must be verified against Memo 195 scalar width
and squint separation before it is an orientation oracle.

### Bacchus install (2026-08-29)

Verified over SSH: Ubuntu 24.04 (`noble`) package `cassbeam` **1.1-4build2**
from `noble/universe`, binary `/usr/bin/cassbeam`. It is a general system
install, not inside the SL1MJax virtualenv. The binary has no
`--version` flag; pin the package version. The packaged example is
`/usr/share/doc/cassbeam/examples/vla-1500MHz.in` at **1.5 GHz** with
the `dfeed_x`, `focus`, and `dsub_z` pathologies left on. That file is
not a C-band configuration.

A general OS install is acceptable. Do not invoke `/usr/bin/cassbeam`
from the imager. Do not generate the oracle from `vla-1500MHz.in`.

The next generator step is a declared nominal C-band input: VLA
geometry, C-band frequencies (4.564 and 4.692 GHz first), and an
explicit statement that the feed is the packaged geometric-optics
nominal, not a measured EVLA C-band feed.

## Freeze rule

Freeze checksums and conventions only after comparing CASSBEAM,
holography, and the known Memo 195 scalar and squint properties.
`full_jones_reference_is_frozen()` stays false until that comparison
exists. Do not implement the interpolating 6B backend before the freeze.
