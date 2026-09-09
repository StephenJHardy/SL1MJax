# VLA C-band beam validation

This page is rendered from the compact SPW-4 validation bundle and the
shared plotting functions. It is not an independently written narrative.
The executable account is `notebooks/vla_c_band_beam_validation.ipynb`.

Bundle `vla_c_band_beam_validation_v2` checksum
`fa01cc15f715e75ace7088ed1bc16a7fac75cfafd2b5b68fda071986f024d5d9`.

## Executive summary

This refresh tests EVLA-C CASSBEAM at `source_lm_feed` on THOL0001 SPW 4
native channels 4500–4626 MHz
(9 sampled channels). It is not a whole-C-band validation
and does not freeze a production beam. Residual power is not a flux error;
the RMS visibility column is $\sqrt{L}$.

Channel 32 uses 538,489 development rows. Diagonal support:

| Region | Support class | Residual power | RMS visibility | Median $|\Delta V|/I$ |
|---|---|---|---|---|
| Main lobe | **accepted** | RR 0.62%, LL 0.56% | RR 7.85%, LL 7.48% | RR 4.55%, LL 4.68% |
| Mid beam | **qualified** | RR 4.03%, LL 3.77% | RR 20.09%, LL 19.42% | RR 5.41%, LL 5.56% |
| Outer raster | **diagnostic** | RR 29.53%, LL 28.98% | RR 54.34%, LL 53.83% | RR 3.50%, LL 3.43% |

Outer CASSBEAM is coherent but not unqualified:
correlation 0.842 / 0.845.
A common scalar beam cancels voltage phase in Stokes I. Outer phase
maps are exploratory.

Full-Jones outcome: `inconclusive_sensitivity`.
Channel-32 experimental RL/LR residual power 0.993 / 0.994.
Residual Jones is the field-9 channel-32 plane applied to all publication
channels; that frequency limit is explicit.

EVLA-C diagonal slopes are 1.027-0.003i (RR) and
1.027-0.000i (LL), with correlations
0.976 /
0.976.

The C147-* ring at 3.61′ gives RR/LL residual power
0.91% / 0.55% after the geometric fringe.
Those field partitions are historical development diagnostics, not a new sealed holdout.

Publication squint uses the 20%-of-peak main-lobe estimator:
measured 0.526′,
Memo 195 0.526′,
CASSBEAM 0.476′.

## Current EVLA-C / source-in-beam panels

These figures are recomputed from the refresh exports. They are not the
historical generic/commanded comparison.

![On-axis V/I](assets/vla_c_band_beam_validation/05_onaxis_amplitude.png)

![Main-lobe CASSBEAM scatter](assets/vla_c_band_beam_validation/08_cassbeam_scatter_main_lobe.png)

![Spatial residual maps](assets/vla_c_band_beam_validation/10_spatial_measured_cassbeam_residual.png)

![Residual versus radius](assets/vla_c_band_beam_validation/12_residual_vs_radius.png)

![Publication squint](assets/vla_c_band_beam_validation/13_squint_mainlobe_20pct.png)

![Residual strata](assets/vla_c_band_beam_validation/15_residual_strata.png)

![Frequency series](assets/vla_c_band_beam_validation/16_spw4_frequency.png)

![Offset ring](assets/vla_c_band_beam_validation/17_c147_offset_ring.png)

![Experimental cross-hands](assets/vla_c_band_beam_validation/18_experimental_crosshand.png)

![SPW 5 sealed](assets/vla_c_band_beam_validation/19_spw5_sealed.png)

![Amplitude in dB](assets/vla_c_band_beam_validation/20_amplitude_db.png)

![Signed complex cuts](assets/vla_c_band_beam_validation/21_signed_complex_cuts.png)

![Masked phase](assets/vla_c_band_beam_validation/22_masked_phase.png)

![Radial coherence](assets/vla_c_band_beam_validation/23_radial_coherence.png)

![Per-antenna coherence](assets/vla_c_band_beam_validation/24_antenna_coherence.png)

![Bright-source examples](assets/vla_c_band_beam_validation/25_bright_source_examples.png)

## Observation and calibration provenance (F01–F04 reused)

These panels validate the observation, occupancy, antenna roles, and
CASA/JAX apply operator. They do not validate HOLORASTER coordinates or
the EVLA-C beam.

![Observation timeline](assets/vla_c_band_beam_validation/01_observation_timeline.png)

![Raster occupancy](assets/vla_c_band_beam_validation/02_raster_occupancy.png)

![Antenna roles](assets/vla_c_band_beam_validation/03_antenna_roles.png)

![CASA/JAX operator residual](assets/vla_c_band_beam_validation/04_casa_jax_operator_residual.png)

## Appendix: coordinate/feed controls (labelled historical models)

The HOLORASTER metadata gives the commanded antenna displacement. The
beam must be queried at the source relative to that pointing:
$s_{\rm feed}=-\Delta_{\rm commanded}$.
The following paired scores keep generic and EVLA-C models on matched
development rows. They are controls, not a second unseen validation.

| Change | Spatial paired ΔL [95% CI] | Mover paired ΔL [95% CI] |
|---|---:|---:|
| Correct coordinate, generic feed | +0.0012 [-0.0053, +0.0099] | -0.0030 [-0.0035, -0.0024] |
| EVLA-C feed at corrected coordinate | -0.0056 [-0.0081, -0.0038] | -0.0061 [-0.0071, -0.0049] |

The corrected-frame vector plot uses the training-only independent-mask
measured estimate (0.720′)
to compare direction. It does not replace the all-data publication estimate
(0.526′). The EVLA-C plane is
0.516′.

![Coordinate and feed impact](assets/vla_c_band_beam_validation/26_coordinate_feed_impact.png)

![Updated measured-versus-predicted scatter](assets/vla_c_band_beam_validation/27_coordinate_feed_scatter.png)

![Updated residual maps](assets/vla_c_band_beam_validation/28_coordinate_feed_residual_maps.png)

![Corrected-frame squint vectors](assets/vla_c_band_beam_validation/29_coordinate_feed_squint.png)

## Claim registry

| ID | Status | Claim |
|---|---|---|
| C01 | pass | CASSBEAM is the SPW-4 diagonal C-band reference in the main lobe |
| C02 | warn | Middle-beam CASSBEAM is useful but imperfect |
| C03 | warn | Outer-raster CASSBEAM is diagnostic only |
| C04 | pass | The C147-* offset ring independently supports the diagonal beam |
| C05 | warn | EVLA-C full Jones is inconclusive on these SPW-4 observations |
| C06 | not_run | SPW 5 remains sealed |
| C07 | pass | Publication squint uses the 20%-of-peak main-lobe estimator |
| C08 | pass | HOLORASTER beam queries use source-in-feed coordinates |
| C09 | pass | EVLA-C feed parameters improve SPW-4 development holdouts |

## Limitations

- Use EVLA-C CASSBEAM at source-in-beam coordinates as the leading SPW-4 development prior.
- Keep generic VLA at commanded coordinates as a historical ablation.
- Mid beam is qualified; attach substantial model uncertainty outside it.
- Bright out-of-field sources can be misestimated by factors of two or more.
- Do not infer a general outer phase correction from the current phase means.
- Outer-field corrections must transfer across movers and spatial holdouts;
  otherwise use source-specific nuisance terms or peeling.
- Tested SPW-4 frequencies: 4500–4626 MHz; SPW 5 remains sealed.
- Full-Jones outcome is `inconclusive_sensitivity`; this does not accept a production beam.
- Residual Jones has no per-channel axis; the channel-32 plane is applied explicitly.
- C147-* partitions are historical development diagnostics.
- The next bounded full-Jones step is a residual-Jones frequency axis;
  do not retune the current gates. The next diagonal step remains a
  validation-selected low-order correction or a sealed frequency-transfer
  test, not a production full-Jones freeze.
