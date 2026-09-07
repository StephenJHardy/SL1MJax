# THOL0001 SPW-4 physical squint and common width

This is **SPW-4 development**. Native CASSBEAM squint was never removed
by default. SPW 5 stayed sealed. Full Jones was untouched. No general
C-band beam was frozen.

Product:
`/media/stephen/astro/vla/extracted/commissioning/validation/scientific/holoraster_physical_squint_width_v1`.

The parameterisation is independent common width \(w\), common pointing
\(p=0\) (not fitted), and physical R/L separation \(\delta\). Identity
\(w=1\), \(p=0\), \(\delta=\delta_{\mathrm{native}}\) reproduced the
frozen HOLORASTER comparison on all 538,489 development rows at
numerical zero.

## Map-domain squint

Training cells only, equal cell weight, common R/L support, 20%-of-peak
\(|E|^2\):

| Estimator | Separation |
|---|---|
| Measured, independent hand masks | 0.720′ |
| Mover-cluster bootstrap | 0.692′ to 0.733′ |
| Measured, common mask | 0.449′ |
| Native CASSBEAM on the same cells | 0.422′ |
| Frozen CASSBEAM catalogue centres | 0.411′ |
| Memo 195 | 0.526′ |
| Published holography checkpoint | 0.518′ |

The independent-mask vector is \((\Delta l,\Delta m)=(0.252,-0.675)\)
arcmin. Mover and reference directions are stable. The commanded
empirical \(\delta\) moved the forward-model centroid in the same
direction.

## Visibility-domain scores

Complex visibility residual power. Paired \(\Delta L=L_{\mathrm{cand}}-L_{\mathrm{base}}\).

| Comparison | Spatial \(\Delta L\) (95%) | Mover \(\Delta L\) (95%) | Holdout movers |
|---|---|---|---|
| Native vs no-squint | \(+0.0028\) \([+0.0020,+0.0039]\) | \(+0.0030\) \([+0.0028,+0.0031]\) | 0/5 improve |
| Empirical vs native | \(-0.0046\) \([-0.0065,-0.0033]\) | \(-0.0051\) \([-0.0054,-0.0047]\) | 5/5 improve |
| Width 1.04 vs native \(\delta\) | \(-0.0063\) \([-0.0096,-0.0039]\) | \(-0.0078\) \([-0.0093,-0.0064]\) | 5/5 improve |
| Width 1.04 vs empirical \(\delta\) | \(-0.0063\) \([-0.0096,-0.0040]\) | \(-0.0078\) \([-0.0093,-0.0064]\) | 5/5 improve |

RR and LL do not regress under the empirical \(\delta\). The
squint-sensitive RR−LL residual falls from 1.31 (native) to 0.86
(empirical) on the mover holdout.

The train-row joint surface along the native CASSBEAM axis bottoms at
magnitude scale 0 and width 1.04. The perpendicular profile bottoms on
the grid edge toward the measured map direction.

## Outcome

- Width **1.04** survived: interior, stable under both fixed-\(\delta\)
  choices, paired intervals below zero, all five holdout movers, no
  material main-lobe regression. This reproduces the earlier prior
  without forcing it.
- The predeclared empirical-versus-native paired gates passed, so an
  empirical \(\delta\) may be stored as an SPW-4 development coefficient.
- Native CASSBEAM still loses to the no-squint ablation in visibility
  loss. Keep native squint as the physics prior; do not set it to zero.
- Do not freeze a general C-band beam. Do not open SPW 5. A later
  one-shot transfer is not justified until the native-axis versus
  empirical-direction disagreement is resolved without refitting on
  SPW 5.
