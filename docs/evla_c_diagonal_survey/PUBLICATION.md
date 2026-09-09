# EVLA-C diagonal survey — compact publication notes

This is the band-survey account. It does not replace
`docs/vla_c_band_beam_validation.md` (v2, SPW-4 refresh). The compact
bundle is `src/sl1mjax/data/vla_c_band_beam_validation_v3`. Full Jones
is diagnostic and is not a gate.

## What can be used for 3C391 now

The opt-in selector is `evla_c_diagonal_survey_v1` with an explicit
catalog root and digest. Exact native frequency only. No nearest-plane
substitution and no Airy fallback.

Native 3C391 SPW0 is 64 channels at 2 MHz from 4536–4662 MHz. Exact
imaging nodes 4536, 4598, and 4662 MHz are generated, plus Pass-B
matches at 4548, 4564, 4580, 4612, and 4644 MHz. Other native channels
remain interpolation-unvalidated. Catalog digest after the SPW-6
Pass-B extras and imaging nodes:

`828c9ece95d3bc6c95bed8091840a6bceea3214e59b42895530b23224973d843`

The seven-pointing smoke completed on those three nodes: train and
reload loss both 0.0380. That is the opt-in adapter proof, not a
scientific mosaic freeze.

Main-lobe 1% residual-power (identity residual Jones) on the scientific
MS, 538,489 development samples:

| SPW | ch | MHz | Both hands ≤ 1% |
| --- | --- | --- | --- |
| 4 | 8 | 4516 | yes |
| 4 | 24 | 4548 | no (LL) |
| 4 | 32 | 4564 | yes |
| 4 | 40 | 4580 | no (RR) |
| 4 | 56 | 4612 | no |
| 5 | 8 | 4644 | no (RR) |
| 5 | 24 | 4676 | yes |
| 5 | 32 | 4692 | yes |
| 5 | 40 | 4708 | yes |
| 5 | 56 | 4740 | no (RR) |
| 6 | 8 | 4772 | no (RR) |
| 6 | 24 | 4804 | yes |
| 6 | 32 | 4820 | yes |
| 6 | 40 | 4836 | yes |
| 6 | 56 | 4868 | no (RR) |
| 0 | 32 | 4052 | no |
| 1 | 32 | 4180 | no |
| 2 | 32 | 4308 | no |
| 3 | 32 | 4436 | yes |
| 7 | 32 | 4948 | yes |
| 8 | 32 | 5052 | yes |
| 9 | 32 | 5180 | yes |
| 10 | 32 | 5308 | no |
| 11 | 32 | 5436 | yes |
| 12 | 32 | 5564 | yes |
| 13 | 32 | 5692 | yes |
| 14 | 32 | 5820 | yes |
| 15 | 32 | 5948 | no |
| upper-C 0 | 32 | 6052 | no |
| upper-C 1 | 32 | 6180 | no |
| upper-C 2 | 32 | 6308 | no |
| upper-C 3 | 32 | 6436 | no |
| upper-C 4 | 32 | 6564 | no |
| upper-C 5 | 32 | 6692 | no |
| upper-C 6 | 32 | 6820 | no |
| upper-C 7 | 32 | 6948 | no |
| upper-C 8 | 32 | 7052 | no |
| upper-C 9 | 32 | 7180 | no |
| upper-C 10 | 32 | 7308 | no |
| upper-C 11 | 32 | 7436 | no |
| upper-C 12 | 32 | 7564 | no |
| upper-C 13 | 32 | 7692 | no |
| upper-C 14 | 32 | 7820 | no |
| upper-C 15 | 32 | 7948 | no |

A defensible Stokes-I trial uses the sampled accepted centres and the
fixed 4536/4598/4662 MHz imaging nodes. That is not continuous channel
acceptance: SPW-4 4548/4580 MHz already miss in one hand. Pass A is
complete; Pass B is limited to lower-C SPWs 4–6. The empirical 1%
main-lobe cut is not numerical qualification. All 16 upper-C Pass-A
centres miss the 1% cut (6180–6436 MHz hard; the rest 1.1–2.1%) and
are archive-integrity unverified. Known Pass-B edge failures sit
inside 4516–4868 MHz. The cliff is between 4308 MHz (95/94%) and
4436 MHz (0.54/0.57%). Mid-beam is qualified; outer remains
diagnostic. First-null alignment is unresolved at HOLORASTER
sampling. SPW 5 is a declared frequency-transfer test, not a
production freeze.

## What changed since SPW 4

Residual Jones is **not applied**. The published ch24 LL and ch40 RR
excursions survive without it. SPW 5 mid-SPW channels pass the same 1%
cut. Coordinate query remains `source_lm_feed` with unit EVLA-C.

## Commands

See `imaging_handoff.json` in the v3 bundle.

Smoke (already run):

```bash
PYTHONPATH=/tmp/sl1mjax-evla-c-survey JAX_PLATFORMS=cpu \
  python -u scripts/run_3c391_survey_smoke.py \
  --native-root outputs/3c391_native_averaging_ablation \
  --survey-catalog-root /media/stephen/astro/vla/beam_models/cassbeam_evla_c_diagonal_survey_v1 \
  --survey-catalog-digest 828c9ece95d3bc6c95bed8091840a6bceea3214e59b42895530b23224973d843 \
  --output outputs/3c391_survey_smoke
```

Seven-pointing Stokes-I mosaic (existing FOV/guard/catalog; not a long
optimized reconstruction):

```bash
PYTHONPATH=/tmp/sl1mjax-evla-c-survey JAX_PLATFORMS=cpu \
  python -u scripts/run_3c391_phase6_bacchus.py \
  --stage baseline --beams evla_c_diagonal_survey_v1 \
  --native-root outputs/3c391_native_averaging_ablation \
  --survey-catalog-root /media/stephen/astro/vla/beam_models/cassbeam_evla_c_diagonal_survey_v1 \
  --survey-catalog-digest 828c9ece95d3bc6c95bed8091840a6bceea3214e59b42895530b23224973d843 \
  --output outputs/3c391_survey_mosaic
```
