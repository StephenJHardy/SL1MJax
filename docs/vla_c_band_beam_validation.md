<!-- Live survey page. The frozen v2 SPW-4 account is snapshotted in `docs/vla_c_band_beam_validation_v2_snapshot/` and `src/sl1mjax/data/vla_c_band_beam_validation_v2`. -->

# EVLA-C diagonal survey summary (v3)

Rendered from `src/sl1mjax/data/vla_c_band_beam_validation_v3`. The frozen v2 SPW-4 page is snapshotted in `docs/vla_c_band_beam_validation_v2_snapshot/`. Full Jones is diagnostic and is not a gate.

Bundle `8cd815eb08e9…`. Catalog digest `828c9ece95d3bc6c95bed8091840a6bceea3214e59b42895530b23224973d843`. Residual Jones: not_applied. Interpolation: refused.

## Slot status

44 of 160 declared science slots are scored. Pass A is complete; Pass B is limited to lower-C SPWs 4–6. A `scientifically-qualified` label is the empirical 1% main-lobe residual-power cut only; numerical qualification is incomplete. Upper-C claims are archive-integrity unverified. Counts: `{'complete': 28, 'pass_a_only': 116, 'scientifically-qualified': 16}`.

## Scored slots

| exec | SPW | ch | MHz | RR / LL | Both ≤ 1% | status |
| --- | --- | --- | --- | --- | --- | --- |
| lower_c | 0 | 32 | 4052 | 6.96% / 10.25% | no | complete |
| lower_c | 1 | 32 | 4180 | 25.15% / 23.44% | no | complete |
| lower_c | 2 | 32 | 4308 | 94.66% / 94.03% | no | complete |
| lower_c | 3 | 32 | 4436 | 0.54% / 0.57% | yes | scientifically-qualified |
| lower_c | 4 | 8 | 4516 | 0.96% / 0.96% | yes | scientifically-qualified |
| lower_c | 4 | 24 | 4548 | 0.58% / 1.40% | no | complete |
| lower_c | 4 | 32 | 4564 | 0.60% / 0.54% | yes | scientifically-qualified |
| lower_c | 4 | 40 | 4580 | 1.64% / 0.62% | no | complete |
| lower_c | 4 | 56 | 4612 | 1.14% / 1.03% | no | complete |
| lower_c | 5 | 8 | 4644 | 1.75% / 0.88% | no | complete |
| lower_c | 5 | 24 | 4676 | 0.60% / 0.65% | yes | scientifically-qualified |
| lower_c | 5 | 32 | 4692 | 0.59% / 0.60% | yes | scientifically-qualified |
| lower_c | 5 | 40 | 4708 | 0.65% / 0.62% | yes | scientifically-qualified |
| lower_c | 5 | 56 | 4740 | 1.05% / 0.87% | no | complete |
| lower_c | 6 | 8 | 4772 | 1.05% / 0.91% | no | complete |
| lower_c | 6 | 24 | 4804 | 0.62% / 0.61% | yes | scientifically-qualified |
| lower_c | 6 | 32 | 4820 | 0.55% / 0.58% | yes | scientifically-qualified |
| lower_c | 6 | 40 | 4836 | 0.61% / 0.61% | yes | scientifically-qualified |
| lower_c | 6 | 56 | 4868 | 1.02% / 0.97% | no | complete |
| lower_c | 7 | 32 | 4948 | 0.65% / 0.58% | yes | scientifically-qualified |
| lower_c | 8 | 32 | 5052 | 0.60% / 0.65% | yes | scientifically-qualified |
| lower_c | 9 | 32 | 5180 | 0.62% / 0.64% | yes | scientifically-qualified |
| lower_c | 10 | 32 | 5308 | 1.46% / 1.47% | no | complete |
| lower_c | 11 | 32 | 5436 | 0.68% / 0.68% | yes | scientifically-qualified |
| lower_c | 12 | 32 | 5564 | 0.68% / 0.68% | yes | scientifically-qualified |
| lower_c | 13 | 32 | 5692 | 0.65% / 0.65% | yes | scientifically-qualified |
| lower_c | 14 | 32 | 5820 | 0.70% / 0.69% | yes | scientifically-qualified |
| lower_c | 15 | 32 | 5948 | 30.01% / 22.95% | no | complete |
| upper_c | 0 | 32 | 6052 | 1.39% / 1.40% | no | complete |
| upper_c | 1 | 32 | 6180 | 95.39% / 98.02% | no | complete |
| upper_c | 2 | 32 | 6308 | 92.77% / 95.10% | no | complete |
| upper_c | 3 | 32 | 6436 | 94.64% / 97.19% | no | complete |
| upper_c | 4 | 32 | 6564 | 1.17% / 1.11% | no | complete |
| upper_c | 5 | 32 | 6692 | 1.22% / 1.15% | no | complete |
| upper_c | 6 | 32 | 6820 | 1.20% / 1.17% | no | complete |
| upper_c | 7 | 32 | 6948 | 1.68% / 1.62% | no | complete |
| upper_c | 8 | 32 | 7052 | 1.41% / 1.38% | no | complete |
| upper_c | 9 | 32 | 7180 | 1.28% / 1.28% | no | complete |
| upper_c | 10 | 32 | 7308 | 1.46% / 1.34% | no | complete |
| upper_c | 11 | 32 | 7436 | 1.64% / 1.40% | no | complete |
| upper_c | 12 | 32 | 7564 | 2.05% / 1.45% | no | complete |
| upper_c | 13 | 32 | 7692 | 1.39% / 1.39% | no | complete |
| upper_c | 14 | 32 | 7820 | 1.41% / 1.41% | no | complete |
| upper_c | 15 | 32 | 7948 | 1.46% / 1.47% | no | complete |

## Imaging commands

Smoke (completed on Bacchus):

```bash
PYTHONPATH=/tmp/sl1mjax-evla-c-survey JAX_PLATFORMS=cpu python -u scripts/run_3c391_survey_smoke.py --native-root outputs/3c391_native_averaging_ablation --survey-catalog-root /media/stephen/astro/vla/beam_models/cassbeam_evla_c_diagonal_survey_v1 --survey-catalog-digest 828c9ece95d3bc6c95bed8091840a6bceea3214e59b42895530b23224973d843 --output outputs/3c391_survey_smoke
```

Seven-pointing mosaic (not a long optimized reconstruction):

```bash
PYTHONPATH=/tmp/sl1mjax-evla-c-survey JAX_PLATFORMS=cpu python -u scripts/run_3c391_phase6_bacchus.py --stage baseline --beams evla_c_diagonal_survey_v1 --native-root outputs/3c391_native_averaging_ablation --survey-catalog-root /media/stephen/astro/vla/beam_models/cassbeam_evla_c_diagonal_survey_v1 --survey-catalog-digest 828c9ece95d3bc6c95bed8091840a6bceea3214e59b42895530b23224973d843 --output outputs/3c391_survey_mosaic
```

## Supersession

- **generic commanded HOLORASTER comparison** — historical: Used generic VLA feed at commanded AZELGEO offsets
- **vla_c_band_beam_validation_v2 SPW-4 refresh** — preserved_checkpoint: Channel-32 residual Jones reused across its series; not band-ready
- **SPW 5** — declared_diagonal_frequency_transfer: Opened under identity residual Jones; not a full-Jones reopen
- **channel-32 residual Jones apply** — not_applied: Survey policy is identity residual Jones
- **width 1.04 / inherited squint / convention ladder** — not_inherited: Unit EVLA-C physical model only
- **full Jones scientific acceptance** — diagnostic_not_gate: Refresh outcome inconclusive_sensitivity; not required for imaging
- **3C391 seven-pointing survey smoke** — complete: Finite train/reload loss on C1–C7 at 4536/4598/4662 MHz
- **Pass B within-SPW extras outside SPWs 4–6** — pass_a_only: Pass A complete; Pass B limited to lower-C SPWs 4–6
- **band-wide numerical convergence** — incomplete: No low/mid/high aperture-resolution and angular-sampling suite
- **upper-C archive integrity** — unverified: Inner ms.tgz CRC error; casacore readability is not a payload checksum
