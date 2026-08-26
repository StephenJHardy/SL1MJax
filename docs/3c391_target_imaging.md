# 3C391 target imaging comparison

## Purpose

This comparison separates real-data imaging from calibration:

1. image CASA `CORRECTED_DATA` with the SL1MJax imager;
2. solve G/K/B and secondary gains from raw calibrator samples in JAX;
3. flux-scale the secondary gains and apply them to raw target `DATA`;
4. image both paths with identical averaging and image settings.

The selected target is the central mosaic pointing, field 2 (`3C391 C1`).
The workflow averages 64 channels into four bins and 10-second integrations
into 60-second baseline bins, producing a `(20542, 4, 2)` RR/LL block.

Run:

```bash
uv run scripts/image_3c391_target.py \
  /path/to/3c391_ctm_mosaic_10s_spw0.ms
```

Products are written under `outputs/3c391_target/`.

## Results

- primary 3C286 train/holdout RMS: `0.0291 / 0.0309`;
- secondary J1822-0938 train/holdout RMS: `0.0495 / 0.0493`;
- transferred J1822-0938 flux: `2.2867 Jy`;
- target visibility difference from CASA calibration: `0.0651` normalized
  complex RMS over 108,672 common active samples;
- dirty-image correlation: `0.9898`;
- dirty-image normalized RMS difference: `0.1762`;
- bounded positive-grid reconstruction correlation: `0.9960`;
- reconstruction normalized RMS difference: `0.0747`;
- reconstruction peaks: `0.03125` and `0.03141 Jy/pixel` for CASA and JAX
  calibration respectively.

The first application exposed an important transfer detail: gains solved on
the secondary calibrator against a unit model contain `sqrt(S_secondary)`.
Those gain amplitudes must be divided by the square root of the transferred
flux before applying them to a target. This operation is now explicit in
`flux_scale_solution` and has a real-data regression test.

## Interpretation and limitations

The close reconstruction morphology, peak and flux scale show that independent
JAX G/K/B solving and secondary-gain transfer are producing a useful target
calibration. The larger dirty-image RMS difference is partly associated with
different accepted flags and weights: the corresponding PSFs differ by
`0.0995` normalized RMS.

This is not yet a publication-quality 3C391 image. The reconstruction uses only
1,000 averaged rows, a coarse 40×40 positive pixel grid, one central mosaic
pointing and no primary-beam or joint-mosaic model. Both calibration paths have
high imaging holdout losses (`0.473` CASA and `0.482` JAX), so their agreement
shows calibration-path consistency rather than complete sky modelling.

The calibration remains partly anchored by imported information: frozen CASA
tutorial flags, the CASA 3C286 sky model and the VLA antenna-position
correction. G, K, B, time gains and flux transfer are solved in JAX.

## Gain-interpolation validation

The original comparison transferred time gains by nearest neighbour. A later
controlled sweep held the seven-pointing consensus hierarchy fixed and
compared nearest transfer with linear interpolation of amplitude and unwrapped
phase. Both candidates used identical post-application flags, weights,
60-second bins, sky predictions, and complete held-out scans.

Linear interpolation reduces held-out fixed-sky residual power from 0.01320 to
0.01035, or 21.6%, and reduces normalized RMS from 0.1149 to 0.1017. It wins in
all four frequency bins and six of seven pointings. Its normalized power
distance from CASA `CORRECTED_DATA` is 0.00166, compared with 0.00417 for
nearest transfer. CASA reaches 0.00908 against the same frozen sky, so a
smaller calibration gap remains. Machine-readable results are under
`outputs/3c391_calibration_interpolation_sweep/`.

## Next boundary

The next imaging tranche should improve the shared sky model rather than tune
calibration against this target:

1. add a scalable gridded/NUFFT imaging operator;
2. model the primary beam;
3. jointly image all seven mosaic pointings;
4. use the full averaged visibility set with a larger multiscale sky model;
5. retain frozen validation visibilities before considering target
   self-calibration.

## Independent CASA CLEAN comparison

CASA 6.7.6 was also run directly on the full-resolution C1
`CORRECTED_DATA`. The controlled single-pointing reference uses a 96×96,
4-arcsecond grid, natural weighting, `wproject`, and classic Högbom CLEAN to a
1 mJy threshold. It reached a `1.51 mJy/beam` residual RMS and
`4.26 mJy/beam` maximum residual.

The naturally weighted dirty-image operators agree well:

- CASA full-resolution wproject versus SL1MJax averaged direct DFT
  correlation: `0.9941`;
- normalized dirty-image RMS difference: `0.1106`;
- peak: `0.2341` versus `0.2325 Jy/beam`.

This validates the direct adjoint DFT geometry and normalization on the real
target. Time/frequency averaging, w-projection and gridding account for the
remaining controlled differences.

The present bounded deconvolution is not yet competitive with CLEAN:

- restored-image correlation: `0.7359`;
- normalized RMS difference: `0.5579`;
- peak: `0.1772` versus `0.0799 Jy/beam`;
- Högbom component flux: `7.95 Jy`;
- SL1MJax positive-grid flux: `5.72 Jy`.

This is not an equal-compute comparison. CASA used all 105,580 C1 rows and 64
channels on a 96×96 grid; SL1MJax used 1,000 rows, four averaged channels and a
40×40 grid because the direct DFT scales with visibility-pixel products. The
SL1MJax model misses smooth interior emission and underestimates restored peak
brightness. The failed gate is therefore scalable, multiscale deconvolution,
not calibration or the direct measurement equation.

An exploratory single-pointing CASA multiscale run with an automatic mask
stopped repeatedly on diverging large-scale components and was not used as the
oracle. The tutorial's publication-style reference instead requires the full
seven-pointing mosaic, primary-beam handling, a curated mask, Briggs weighting
and optional self-calibration. Those effects remain beyond this controlled
single-pointing test.
