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

The missing direct paired empirical-versus-no-squint interval is
computed in the follow-on spatial-convention product, not inferred
from the two native-referenced point estimates.

## Outcome

The holography measures a real differential R/L displacement. The
current native CASSBEAM orientation is not an adequate production
prior: it loses to no-squint in visibility loss, while the measured
direction beats native. That is a spatial-convention problem, not a
proof that squint is absent.

- Keep the physical role of CASSBEAM squint. Do not set \(\delta=0\)
  as the model.
- Do **not** retain the current CASSBEAM coordinate orientation as
  the production prior. The independent-mask vector
  \((0.252,-0.675)\) arcmin differs from the frozen centres
  \((0.292,+0.290)\) by about \(114^\circ\).
- Width **1.04** survived and is frozen for the discrete convention
  test.
- The 0.692′–0.733′ bootstrap interval is sampling variation of the
  independent-mask estimator only. The common-mask magnitude is
  0.449′. Direction is better determined than magnitude.
- Do not freeze a general C-band beam. Leave SPW 5 sealed until a
  discrete \(l/m\)-sign, axis-swap, and R/L-swap convention is locked
  on the complete diagonal beam. Magnitude adjustment stays separate.
  That discrete test is `thol0001_spw4_spatial_convention.md`.
