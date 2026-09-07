# VLA C-band beam validation

This page is rendered from the compact SPW-4 validation bundle and the
shared plotting functions. It is not an independently written narrative.
The executable account is `notebooks/vla_c_band_beam_validation.ipynb`.

Bundle `vla_c_band_beam_validation_v1` checksum
`0571f4a7b38d69e4c9d9b8400f4ff53af2b744944af3780107f4d59e923b680b`.

## Executive summary

CASSBEAM is the reference diagonal C-band beam inside a stated validity
domain. It is not an unqualified high-dynamic-range model of the whole
raster.

| Region | Support class | Channel-32 residual power |
|---|---|---|
| Main lobe | **accepted** | RR 0.64%, LL 0.76% |
| Mid beam | **qualified** | RR 7.70%, LL 8.91% |
| Outer raster | **diagnostic** | RR 32.31%, LL 34.85% |

Complex slopes are 1.032+0.003i (RR) and
1.029+0.004i (LL), with correlations
0.971 /
0.968.

The C147-* ring at 3.61′ gives RR/LL residual power
1.17% / 0.90% after the geometric fringe.
That Q/U result is not a frequency holdout.

The cross-hand result is a sensitivity-limited non-detection:
RL/LR correlation 0.101 / 0.104,
residual power 0.994 / 0.992.

Publication squint uses the 20%-of-peak main-lobe estimator:
measured 0.515′,
Memo 195 0.526′,
CASSBEAM 0.411′.

![Observation timeline](assets/vla_c_band_beam_validation/01_observation_timeline.png)

![Raster occupancy](assets/vla_c_band_beam_validation/02_raster_occupancy.png)

![Main-lobe CASSBEAM scatter](assets/vla_c_band_beam_validation/08_cassbeam_scatter_main_lobe.png)

![Spatial residual maps](assets/vla_c_band_beam_validation/10_spatial_measured_cassbeam_residual.png)

![Residual versus radius](assets/vla_c_band_beam_validation/12_residual_vs_radius.png)

![Publication squint](assets/vla_c_band_beam_validation/13_squint_mainlobe_20pct.png)

![Offset ring](assets/vla_c_band_beam_validation/17_c147_offset_ring.png)

![Experimental cross-hands](assets/vla_c_band_beam_validation/18_experimental_crosshand.png)

## Claim registry

| ID | Status | Claim |
|---|---|---|
| C01 | pass | CASSBEAM is the SPW-4 diagonal C-band reference in the main lobe |
| C02 | warn | Middle-beam CASSBEAM is useful but imperfect |
| C03 | warn | Outer-raster CASSBEAM is diagnostic only |
| C04 | pass | The C147-* offset ring independently supports the diagonal beam |
| C05 | blocked | CASSBEAM full Jones is an experimental non-detection |
| C06 | not_run | SPW 5 remains sealed |
| C07 | pass | Publication squint uses the 20%-of-peak main-lobe estimator |

## Limitations

- One published SPW-4 frequency; SPW 5 remains sealed.
- Outer-raster residual power is too large for an unqualified HDR model.
- Full Jones is an experimental non-detection below the THOL0001 floor.
- The next artifact is a low-order diagonal correction aimed at the 10′ ring.
