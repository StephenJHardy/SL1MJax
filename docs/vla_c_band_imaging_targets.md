# Interesting VLA C-band imaging targets

**Purpose:** shortlist of public/archival VLA C-band datasets that look useful for testing radio imaging and deconvolution methods, with an emphasis on extended structure, wide bandwidth, mosaics, dynamic range, and polarisation.

**Last researched:** 30 August 2026

> **Archive note.** Search the NRAO Science Data Archive by the project IDs below:
> https://data.nrao.edu/
>
> I could verify project codes, observing setup and scientific use from NRAO documentation and published papers. I could **not reliably obtain archive download byte sizes from the public indexed pages**, so I have not invented them. Modern wideband datasets can become large once restored to Measurement Sets; the archive query is the right place to check the actual SDM/tar size for a particular execution block before downloading.

---

## Executive shortlist

| Rank | Target / field | Project ID(s) | Configuration | C-band setup | Polarisation | Why it is interesting |
|---|---|---|---|---|---|---|
| **1** | **3C 391 SNR** | **TDEM0001** | D | 4.6 & 7.5 GHz, 2 × 128 MHz | **Full pol** | Seven-pointing mosaic; extended shell/filaments; NRAO tutorial dataset; ideal first benchmark |
| **2** | **M87 jet** | **15A-170** | A | **4–8 GHz**, 2 MHz channels | **Full pol** | High-dynamic-range jet; Q/U structure; Faraday/depolarisation science; beautiful compact test |
| **3** | **Sgr B2** | **16A-195, 17B-063, 16B-031** | CnB, D, A | **4–8 GHz**, many SPWs | continuum setup; check correlations per EB | Ten-point mosaic, extreme thermal/nonthermal complexity, dynamic-range limited, multi-configuration |
| **4** | **Orion Nebula Cluster** | **SD630** | BnA→A / A | two 1-GHz basebands at 4.736 & 7.336 GHz | **Full pol** | ~30 h deep field; 556 compact sources plus complex extended emission; wideband primary-beam challenge |
| **5** | **GLOSTAR Galactic plane** | **13A-334, 14A-420, 15B-175, 17A-197, 16A-174** | D/DnC/C and B/BnA | **4–8 GHz**, typically 2 MHz continuum channels | **Full-pol basebands** | Huge mosaics; thousands of sources; HII/SNR/filaments; ideal wide-field statistical benchmark |
| **6** | **Crab Nebula (modern)** | **12A-486 / AB1439** | B | wide C-band, nominally 4–8 GHz resource | **Full pol resource** | ~5 GHz synchrotron nebula; 3 × 5 h epochs; bright structured source and polarisation potential |
| **7** | **Cassiopeia A** | **AR0435** | A+B+C+D | legacy 6-cm/C-band | legacy; verify products | Extremely bright, filamentary, multi-configuration; brutal dynamic-range/multiscale test |
| **8** | **Crab Nebula (legacy multi-config)** | **AH0337, AB0876, AH0625** | A/B/C/D | legacy 6-cm/C-band | varies; verify per project | Existing multi-configuration 5-GHz data used in published fluctuation analysis |

### If I were constructing a deconvolution benchmark

I would start with:

1. **3C 391 / TDEM0001** — controlled extended-source + mosaic + full-pol baseline.
2. **M87 / 15A-170** — high-resolution broadband polarisation and dynamic range.
3. **Orion / SD630** — deep wideband field containing compact and extended structure.
4. **Sgr B2 / 16A-195 + 17B-063** — much harder complex mosaic; strong spatial-scale mixing and dynamic-range errors.
5. **One GLOSTAR 1°–2° strip** — genuine wide-field/mosaic stress test and statistical evaluation.
6. **Cas A / AR0435** — hard mode, especially after combining configurations.

---

# 1. 3C 391 — best controlled starting point

## Archive identifier

**VLA project: `TDEM0001`**

This was an EVLA demonstration-science observation designed specifically as a synthesis-imaging/polarimetry tutorial dataset.

## Observing setup

- Target: supernova remnant **3C 391**
- Array: **D configuration**
- Date: **24 April 2010**
- Total observing span: about **8 hours**
- Science layout: **7-pointing mosaic**
- Receiver: C band
- Two widely separated spectral windows:
  - approximately **4.6 GHz**
  - approximately **7.5 GHz**
- Bandwidth: **128 MHz per spectral window**
- Original integration: **1 s**, with NRAO tutorial versions also provided averaged to 10 s
- **Full polarisation calibration was built into the experiment**
- Calibrators in the tutorial include:
  - 3C286 for flux/bandpass/polarisation angle
  - 3C84 for instrumental polarisation leakage
  - J1822−0938 for phase

The NRAO `listobs` example reports roughly **28,716 s total integration time** and 18.7 million visibility records before the tutorial averaging.

## Why it is a particularly good imaging target

3C 391 is large enough to substantially fill the C-band primary beam, hence the seven-pointing mosaic. Its emission is neither point-like nor a featureless smooth blob: the SNR contains shell/filamentary emission over a range of angular scales.

This gives a useful controlled problem for:

- joint mosaic deconvolution
- multi-scale reconstruction
- point-source CLEAN vs multiscale methods
- primary-beam weighting
- Stokes I/Q/U/V imaging
- polarised intensity and EVPA reconstruction
- spectral reconstruction using separated frequency windows
- sensitivity to short-baseline recovery
- testing whether a new method invents compact structure inside diffuse emission

### Difficulty

**2/5 initially; 3/5 in full polarisation.**

It is complex enough to be meaningful, while being unusually well documented. NRAO calibration and expected images give you an external reference.

### Recommendation

**Download this first.** It is probably the fastest route to establishing that an experimental imaging pipeline is numerically and scientifically sane.

---

# 2. M87 — probably the best polarisation target

## Archive identifier

**VLA project: `15A-170`**

PI-led observations subsequently used by Pasetto et al. for the detailed M87 jet polarisation/Faraday analysis.

## Observing setup

- Date: **8 August 2015**
- Array: **A configuration**
- Bands observed: C, X and Ku
- C band: **4–8 GHz**
- Channel width: **2 MHz**
- Standard wideband continuum mode
- Approximately **1 hour on source per band**
- Two closely separated pointings:
  - M87 core
  - knot A
- **Full polarisation**
- Broad parallactic-angle coverage deliberately obtained for leakage calibration
- 3C286 used for flux, bandpass and EVPA calibration
- Published reduction solved polarisation calibration channel-by-channel
- Only about **0.6% of C-band data** were reported flagged in the published reduction

## What makes the data special

This is not merely “M87 is famous.” The dataset was good enough to recover:

- resolved jet width
- a **double-helix filamentary morphology**
- strong fractional-polarisation structure
- transverse Faraday-depth gradients
- opposite RM signs across parts of the jet
- wavelength-dependent depolarisation

The published work imaged I, Q and U in **128-MHz sub-bands** and used them for QU fitting.

At C band alone you have a 2:1 frequency span (4–8 GHz), which is excellent for:

- MT-MFS
- frequency-dependent uv sampling experiments
- wideband polarised imaging
- RM synthesis / QU fitting
- measuring whether deconvolution errors leak I into Q/U
- checking whether a method preserves fractional polarisation
- checking spatially varying spectral index

## Imaging characteristics

The source combines:

- very bright core
- HST-1
- narrow jet
- knots
- fainter inter-knot emission
- broader/lower-surface-brightness structure

The published full C+X+Ku reduction reached about **0.2 arcsec** with robust weighting and **0.09 arcsec** using uniform weighting. C band alone will of course have coarser resolution, but remains a strong high-resolution structural test.

### Difficulty

**3/5 Stokes I; 4/5 full polarisation.**

### Recommendation

If polarisation fidelity matters to your work, **this is the most compelling target in this list after 3C 391**.

---

# 3. Sagittarius B2 — excellent hard wideband mosaic

Sgr B2 may be the most interesting dataset here from a pure imaging-algorithm perspective.

It is an enormously complex Galactic-centre star-forming region containing bright compact HII regions, diffuse free-free emission, non-thermal structures, crowded source fields and structure over very different angular scales.

## Relevant archive projects

### `16A-195`
- **CnB configuration**
- observations on **3 and 5 May 2016**
- C and X bands
- C band **4–8 GHz**

### `17B-063`
- **D configuration**
- observations on **22 and 23 February 2017**
- C and X bands
- C band **4–8 GHz**

### `16B-031`
- **A configuration**
- observations in **October 2016**
- C band **4–8 GHz**
- ten-pointing mosaic
- high-resolution extension of the same Sgr B2 programme

## Setup of the CnB + D observations

Published reduction of `16A-195` + `17B-063` reports:

- C band: **4–8 GHz**
- X band: 8–12 GHz
- continuum assembled from **64 × 128-MHz spectral windows** across the two bands
- additional high-resolution radio-recombination-line SPWs
- mosaic covering approximately **20′ × 20′**
- **10 C-band pointings**
- C-band primary beam quoted as approximately **7.5′**
- 3C286 flux/bandpass calibrator
- J1820−2528 phase calibrator
- data from CnB and D concatenated for improved uv coverage
- several self-calibration iterations

The published combined C-band image had approximately:

- **2.7″ × 2.5″** synthesised beam
- ~**0.4 mJy/beam** RMS
- dynamic range of only about **2000**, explicitly limited by bright-source/dynamic-range effects

That last point is algorithmically interesting: this is not a thermal-noise-dominated toy problem.

## A-configuration extension: `16B-031`

The later A-array data:

- cover the entire Sgr B2 complex
- use C band **4–8 GHz**
- use **10 mosaic pointings**
- are described as having **64 spectral windows of 128 MHz each**
- were taken at 2-s integration and averaged to 10 s in the published processing
- were combined with lower-resolution data to recover a wider range of spatial scales

Published C-band A-array imaging reached approximately **0.62″ × 0.28″** with robust 0.

## Why it is ideal for imaging research

This one dataset lets you study:

- missing short spacings
- multi-configuration combination
- mosaicking
- spatially varying spectral index
- compact + diffuse decomposition
- dynamic-range limitation
- self-calibration/deconvolution interaction
- bright-source sidelobes
- frequency-dependent PSFs
- wideband primary beams
- super-resolution claims
- whether a method confuses thermal and non-thermal spectral structure

### Difficulty

**4/5 for CnB+D.**
**5/5 if attempting joint A+CnB+D high-dynamic-range wideband imaging.**

### Recommendation

This is probably the dataset I would pick once a method has passed 3C391 and M87.

---

# 4. Orion Nebula Cluster — deep and heterogeneous

## Archive identifier

**VLA project: `SD630`**

## Observing setup

Observations:

- **30 September and 2–5 October 2012**
- roughly **30 hours total**
- individual epochs mostly around **7.5 h**, with shorter ~3 h and ~5 h sessions
- first two epochs during reconfiguration from **BnA to A**
- final three in **A configuration**
- single pointing centred on the Orion Nebula Cluster
- C-band receivers
- **full polarisation**
- two **1-GHz basebands**:
  - centre **4.736 GHz**
  - centre **7.336 GHz**
- each baseband:
  - 8 × 128-MHz subbands
  - 64 × 2-MHz channels per subband
- 3C48 flux calibrator
- J0541−0541 gain calibrator

## Published imaging result

The deep analysis detected **556 compact sources** with nominal RMS around **3 μJy/beam**.

That is only part of the attraction: the field contains complex emission across a wide range of spatial scales and the published authors explicitly discuss the wideband primary-beam response.

## Why it is useful

Orion produces a very different deconvolution problem from an SNR:

- hundreds of compact sources
- extremely deep sensitivity
- bright nebular emission
- resolved and partially resolved sources
- crowded field
- strong primary-beam frequency dependence across separated basebands
- source spectral indices
- nonthermal stellar sources among thermal sources
- time-variable compact emitters across epochs

This makes it particularly attractive for testing:

- source completeness
- false-positive rate near extended emission
- flux recovery vs distance from pointing centre
- in-band spectral-index bias
- primary-beam-aware reconstruction
- joint multi-epoch imaging
- robustness to variable sources
- direction-dependent imaging

### Difficulty

**3–4/5**, depending on how much of the diffuse Orion emission you try to retain.

### Important nuance

The observational design intentionally favoured high angular resolution and spatial filtering to find compact sources. If your primary purpose is faithful recovery of the largest nebular scales, this is not the optimal standalone uv dataset. It is, however, an excellent **mixed compact/extended and deep-field** benchmark.

---

# 5. GLOSTAR — the wide-field benchmark factory

GLOSTAR (“A Global View on Star Formation in the Milky Way”) is not one dataset but a family of large VLA C-band Galactic-plane mosaics.

If the aim is eventually to test an imaging method statistically across many morphologies rather than on one hand-picked object, GLOSTAR is probably the most valuable resource on this list.

## Main project IDs worth searching

- **13A-334**
- **14A-420**
- **15B-175**
- **17A-197**
- **16A-174** (B/BnA portion)
- **11B-168** for some outer-Galactic-plane coverage

## C-band observing concept

The survey uses approximately **4–8 GHz** C band with broad continuum coverage plus selected spectral lines.

Published descriptions include:

- two ~1-GHz continuum basebands in **full polarisation**
- basebands around approximately **4.7 and 6.9 GHz** in parts of the survey
- 128-MHz spectral windows
- typical continuum channel width **2 MHz**
- methanol, formaldehyde and recombination-line windows interspersed with continuum
- mosaics of hundreds of individual VLA pointings

The observing strategy avoids particularly persistent RFI zones around parts of C band rather than simply treating 4–8 GHz as one pristine rectangle.

---

## 5a. GLOSTAR pilot: `13A-334`

A particularly self-contained place to start.

Published pilot-region observations cover:

- approximately **28° < Galactic longitude < 36°**
- |b| < 1°
- roughly **16 square degrees**
- **40 hours** total telescope time for the B-configuration campaign
- approximately **five hours per epoch**
- B-configuration data at approximately **1″** final resolution
- lower-resolution D/DnC data also exist for the survey region

The published high-resolution processing divided the frequency coverage into **nine frequency bins** before deconvolution to deal with:

- spectral index
- frequency-dependent primary beam
- wide fractional bandwidth

That makes this especially relevant to modern wideband imaging work.

---

## 5b. GLOSTAR `14A-420` D-array strips

This project contains many extremely convenient approximately one-degree Galactic longitude strips.

Examples documented in the survey paper include:

- l = 15–16° — 14 Jul 2014 — D
- l = 16–17° — 24 Jul 2014 — D
- l = 17–18° — 5 Aug 2014 — D
- l = 18–19° — 14 Aug 2014 — D
- l = 19–20° — 12 Jul 2014 — D
- l = 20–21° — 23 Jul 2014 — D
- continuing through l ≈ 28°
- additional D-array coverage around l ≈ 36–46°

These are good if you want a **bounded but genuinely wide-field** problem without initially tackling the entire survey.

---

## 5c. Cygnus X GLOSTAR: `14A-420`

Cygnus X is a particularly attractive astronomical field.

Published Cygnus observations:

- cover roughly **7° × 3°**
- used **28 epochs**
- each epoch covered about **1° × 1.5°**
- typically around **532 pointings per epoch**
- two ~11 s visits per pointing
- ~15 s effective on-source integration per pointing
- two 1-GHz-wide continuum basebands
- **full polarisation**
- baseband centres around **4.7 and 6.9 GHz**

A single epoch here would already be a significant wide-field deconvolution test.

## Why GLOSTAR is unusually valuable

Within a modest patch you can encounter:

- compact HII regions
- resolved HII regions
- SNRs
- background AGN
- planetary nebulae
- stellar radio sources
- filaments
- diffuse Galactic emission
- bright off-axis sources
- severe morphology and flux-density heterogeneity

At catalogue level the survey contains many thousands of sources, so an imaging method can be assessed quantitatively on:

- completeness
- reliability
- astrometry
- peak vs integrated flux
- source-size recovery
- spectral-index fidelity
- artefact rate around bright sources

### Difficulty

A single D-array strip: **4/5**.

A large B-array mosaic: **5/5** because of image dimensions, primary-beam effects and sheer source density.

### Recommendation

For an ML/deconvolution project I would **not start with the whole survey**. Pick one 1°–2° strip containing a mixture of compact and extended structures, then scale outward.

---

# 6. Crab Nebula — modern wideband dataset

## Archive identifier

**VLA project: `12A-486`**
Legacy ID: **`AB1439`**

This was a Director's Discretionary/Target-of-Opportunity programme triggered by a gamma-ray flare.

## Proposed/observed setup

NRAO proposal documentation specifies:

- **B configuration**
- **15 h total**
- three **5-hour** sessions
- C-band “wide” resource
- nominal frequency resource: **4–8 GHz**
- WIDAR
- **full polarisation**
- nominal 2-MHz channels in the resource definition

Published later work identifies the relevant C-band VLA observations on **20 and 26 August 2012**, producing a modern ~5.45-GHz Crab image of order **1″** resolution.

## Why it is attractive

The Crab is:

- bright
- extended
- synchrotron dominated
- filamentary
- strongly polarised physically
- morphologically complex

Compared with Cas A it may be a somewhat less pathological dynamic-range problem, while still being much harder than 3C391.

Potential tests:

- multiscale reconstruction
- recovery of fine synchrotron structure
- sensitivity to diffuse background
- comparing independently observed epochs
- polarisation fidelity if the required calibration scans/products are usable

### Caveat

Because this was a transient-trigger programme, verify the exact execution blocks, frequency setup and calibrator scans in the archive before assuming all three proposed epochs executed identically.

### Difficulty

**4/5.**

---

# 7. Cassiopeia A — legacy multi-configuration hard mode

## Archive identifier

**VLA project: `AR0435`**

Published archival analyses identify C-band/6-cm observations in all four main configurations:

- **C array** — 25 Apr 2000
- **D array** — 7 Sep 2000
- **A array** — 9–10 Dec 2000
- **B array** — 25 Mar and 20 Apr 2001

## Why it is algorithmically valuable

Cas A is an archetypal multiscale source:

- extraordinarily bright
- shell/ring morphology
- many compact knots
- filamentary fine structure
- diffuse emission
- wide spatial dynamic range

With A+B+C+D visibility data, you can construct experiments in which uv coverage is progressively added and ask how the reconstruction changes.

This is excellent for:

- multi-resolution reconstruction
- combining configurations
- testing multiscale priors
- high dynamic range
- calibration-error sensitivity
- uv-domain validation

### Important caveat

These are **legacy VLA** observations, not modern 4-GHz-wide JVLA data. The bandwidth and correlator setup are much narrower and less convenient for modern wideband/RM work.

Also, Cas A evolves measurably. Combining data separated in time without considering structural/flux evolution can itself become an error source.

### Difficulty

**5/5.**

This is an “algorithm torture test”, not the dataset on which I would debug basic code.

---

# 8. Crab Nebula — legacy multi-configuration set

A published VLA archival fluctuation study used the following 6-cm data:

- **AH0337**
  - A array
  - 19 Oct 1988
  - 8 Nov 1988
- **AB0876**
  - B array
  - 9 Aug 1998
  - C array
  - 27 Jan 1999
- **AH0625**
  - D array
  - 19 Nov 1997

These are useful if you specifically want historical multi-configuration visibility data, but the modern `12A-486` dataset is more attractive for contemporary wideband work.

---

# A suggested benchmark programme

## Phase 1 — establish correctness

### Dataset A: 3C391 / TDEM0001

Run:

- standard CASA multiscale mosaic as reference
- your reconstruction in Stokes I
- same with identical weighting and uv selection
- compare residuals in visibility space
- compare integrated flux and morphology
- then add Q/U

**Primary question:** can the method reconstruct real extended interferometric data without pathological artefacts?

---

## Phase 2 — spectral and polarisation fidelity

### Dataset B: M87 / 15A-170

Create:

- full 4–8 GHz Stokes-I image
- several 128-MHz or 256-MHz subband images
- Q and U cubes
- reference MT-MFS image

Evaluate:

- jet transverse profile
- knot fluxes
- fractional polarisation
- EVPA
- RM / Faraday-depth gradients
- spectral index
- residual I→Q/U leakage structure

**Primary question:** does improved-looking deconvolution preserve physically meaningful polarisation and spectrum?

---

## Phase 3 — source population + deep-field behaviour

### Dataset C: Orion / SD630

Evaluate against the published compact-source catalogue.

Potential metrics:

- detected-source completeness vs flux
- false detections
- astrometric offset
- recovered peak flux
- recovered integrated flux
- spectral-index error
- performance vs primary-beam radius
- artefact density around bright extended emission

**Primary question:** does the method remain trustworthy across a heterogeneous field rather than merely producing visually attractive images?

---

## Phase 4 — dynamic-range and multiscale stress

### Dataset D: Sgr B2 / 16A-195 + 17B-063

Start with CnB+D only.

Compare:

- CASA tclean reference
- single vs multiscale
- common uv-restricted subband imaging
- wideband reconstruction
- visibility-domain residual statistics
- flux conservation in compact and diffuse regions

Then optionally add A-array `16B-031`.

**Primary question:** can the method improve a field known to be limited by dynamic-range effects rather than thermal noise?

---

## Phase 5 — genuinely wide field

### Dataset E: one GLOSTAR strip

Do **not** start with tens of square degrees.

Pick approximately one longitude strip from `14A-420` or a similarly bounded epoch.

Metrics can be automated using the published GLOSTAR catalogue:

- source completeness
- false-positive density
- size/shape recovery
- flux recovery
- image RMS away from sources
- dynamic range around the N brightest sources
- residual spatial correlation
- spectral-index bias

**Primary question:** does the method scale, both computationally and statistically?

---

# Polarisation-specific ranking

If the main scientific hook is polarisation, I would rank the targets:

1. **M87 / 15A-170**
2. **3C391 / TDEM0001**
3. **GLOSTAR** selected fields / epochs
4. **Crab / 12A-486**
5. **Orion / SD630** — full-pol observing setup is present, although its famous published use is compact-source continuum rather than a showcase polarisation map

M87 has the strongest combination of:

- modern 4–8 GHz bandwidth
- full Stokes
- very good angular resolution
- physically complicated Q/U behaviour
- an excellent published science result against which to compare.

---

# Wide-field ranking

For wide-field imaging:

1. **GLOSTAR**
2. **Sgr B2**
3. **3C391**
4. **Orion**

GLOSTAR is qualitatively different from the others: it can become a benchmark **dataset family**, not just a single benchmark image.

---

# Dynamic-range ranking

From gentler to nastier:

1. 3C391
2. Orion
3. M87
4. GLOSTAR selected strip
5. Sgr B2
6. Crab
7. Cas A

This ranking is approximate because calibration choices and uv weighting strongly affect apparent difficulty.

---

# Archive and processing notes

## Finding the data

Use:

**NRAO Science Data Archive**  
https://data.nrao.edu/

Search the exact project ID, e.g.:

- `TDEM0001`
- `15A-170`
- `SD630`
- `16A-195`
- `17B-063`
- `16B-031`
- `13A-334`
- `14A-420`
- `12A-486`
- `AR0435`

For modern JVLA observations, archive products are generally organised by execution block / scheduling block. Where pipeline calibration products exist, downloading the calibrated products can save substantial time; for algorithm benchmarking, retaining the raw/uncalibrated data as well can be valuable.

## Before downloading a large dataset

For each execution block record:

- archive file size
- configuration
- time range
- target fields
- number of pointings
- spectral windows
- channel count
- integration time
- correlation products
- whether pipeline-calibrated products are available

For polarisation work, explicitly verify:

- cross-hand correlations are present
- suitable leakage calibrator/parallactic-angle coverage exists
- EVPA calibrator is present
- pipeline calibration did not discard cross-hands
- frequency averaging has not destroyed the RM range you care about

## For fair deconvolution comparisons

Save and report:

- exact visibility selection
- exact flag state
- uv range
- weighting / robust parameter
- taper
- image size and cell size
- w-projection / gridder settings
- primary-beam treatment
- number of Taylor terms
- clean mask / stopping rule
- restoring beam
- self-calibration state

A method that changes several of these at once is very hard to compare meaningfully against CLEAN.

---

# Sources

## NRAO documentation

- NRAO 3C391 continuum tutorial:  
  https://science.nrao.edu/facilities/vla/data-processing/end-to-end-recipes/continuum

- NRAO EVLA Demonstration Science table (`TDEM0001`):  
  https://science.nrao.edu/facilities/vla/science/demo

- NRAO VLA data archive:  
  https://data.nrao.edu/

## 3C391

NRAO documentation describes `TDEM0001` as a seven-pointing D-configuration C-band polarimetry mosaic with two 128-MHz subbands, and provides the calibration/imaging recipe.

## M87

Pasetto, A. et al. (2021), **Reading M87's DNA: A Double Helix Revealing a Large-scale Helical Magnetic Field**, ApJL 923 L5.  
ArXiv: https://arxiv.org/abs/2112.06971

The paper identifies NRAO project `15A-170`, A configuration, 4–8 / 8–12 / 12–18 GHz, 2-MHz channels, full polarisation and ~1 h on source per band.

## Orion

Forbrich, J. et al. (2016), **The Population of Compact Radio Sources in the Orion Nebula Cluster**, ApJ 822, 93.  
ArXiv: https://arxiv.org/abs/1603.05666

HEASARC catalogue summary:  
https://heasarc.gsfc.nasa.gov/W3Browse/catalog/vlaonccat.html

These identify project `SD630`, ~30 h, BnA→A/A configurations, full polarisation and two 1-GHz basebands centred at 4.736 and 7.336 GHz.

## GLOSTAR

Brunthaler et al. / GLOSTAR survey series and later catalog papers.

Useful survey/catalogue literature includes the B-configuration GLOSTAR catalogue describing proposal IDs `14A-420/15B-175/16A-174`, and the low-resolution survey tables listing individual D/DnC/C epochs under `13A-334`, `14A-420`, `15B-175`, `17A-197` and `11B-168`.

## Sgr B2

Meng, F. et al. (2019), **The physical and chemical structure of Sagittarius B2 — V. Non-thermal emission in the envelope of Sgr B2**.  
ArXiv: https://arxiv.org/abs/1908.07237

Identifies:
- `16A-195` — CnB
- `17B-063` — D

and describes the 20′×20′ C/X-band mosaic and dynamic-range-limited imaging.

Meng et al. later work on UCHII regions identifies:
- `16B-031` — A configuration, C band, ten-pointing mosaic.

## Crab and Cas A legacy datasets

Roy et al. / Dutta et al. archival fluctuation work tabulates:

- Cas A `AR0435` in A/B/C/D
- Crab `AH0337`, `AB0876`, `AH0625`

at legacy VLA 6 cm.

## Modern Crab

NRAO observing proposal:

- VLA `12A-486`
- legacy `AB1439`
- B configuration
- 15 h requested as 3 × 5 h
- wide C-band/full-polarisation resource

Proposal PDF indexed by NRAO:  
https://www.vla.nrao.edu/astro/prop/rapid/vlarapid/ab1439.pdf

Later Crab work uses the corresponding 2012 C-band imaging.

---

# Bottom line

If the practical goal is **interesting real-data imaging experiments**, rather than assembling a source zoo, my strongest sequence is:

**3C391 → M87 → Orion → Sgr B2 → one GLOSTAR strip → Cas A.**

The first two have unusually good external references. Orion provides deep-field statistics. Sgr B2 exposes serious dynamic-range and spatial-scale problems. GLOSTAR lets the experiment graduate into a repeatable wide-field benchmark. Cas A is where you go when you want to find out what still breaks.
