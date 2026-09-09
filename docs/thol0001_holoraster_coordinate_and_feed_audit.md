# THOL0001 HOLORASTER coordinate sign and EVLA-C feed

The 16-way spatial-convention product
`holoraster_spatial_convention_v1` finished. Identity matched the frozen
path at numerical zero. It did **not** lock a production orientation.
That search is not to be interpreted until the deterministic coordinate
and feed issues below are corrected. SPW 5 stays sealed. The 128-member
Jones ladder stays closed.

## Findings

1. HOLORASTER commanded offsets were used as CASSBEAM source
   coordinates. In the THOL0001 MS,
   \(\mathrm{DIRECTION}-\mathrm{TARGET}=\mathrm{POINTING\_OFFSET}\).
   3C147 remains at `TARGET`, so its coordinate in the moved beam is
   \(\mathrm{TARGET}-\mathrm{DIRECTION}=-\mathrm{POINTING\_OFFSET}\)
   after the AZELGEO-to-CASSBEAM map. `source_relative_lm_rad` already
   implements that sign. The specialised HOLORASTER path copied the
   commanded offset unchanged.

2. The frozen high-resolution CASSBEAM artifact uses the generic VLA
   template feed (`feed_x = feed_y = -0.6896837`, taper 10 dB), not
   CASA's `EVLA_C` feed. The feed-ring radii match; the feed-ring
   angles differ by about \(30^\circ\). CASA's 4.564 GHz taper is
   12.2115 dB. Keep the generic artifact as an ablation.

3. The old pointing-convention diagnostic compared `POINTING_OFFSET`
   with itself when that column was selected. It now scores
   `DIRECTION-TARGET` against the stored offset.

4. Code squint vectors are \(c_R-c_L\) (`r_minus_l`). Memo 195 uses
   \(c_L-c_R\) (`r_to_l`). Magnitudes match; position angles differ by
   \(180^\circ\).

Jones packing, \(V=J_p C J_q^H\), circular Stokes, and the calibrated
\(P^H E P\) sandwich were not found to be wrong. CASA/JAX calibration
goldens do not depend on HOLORASTER pointing coordinates.

## What is now locked in software

- Named HOLORASTER coordinates: `commanded_offset_azelgeo` and
  `source_lm_feed`. Bare `offset_lm_rad` is refused on the new path.
  The legacy `offset_lm_rad` field remains the commanded displacement
  so frozen comparison products stay an explicit ablation.
- AZELGEO \(\to\) CASSBEAM: no axis swap, \(+\mathrm{az}\to +l\),
  \(+\mathrm{el}\to +m\), locked on a manufactured asymmetric beam and
  one raster track. Memo 195: \(+l\) right looking outward, \(+m\)
  toward local zenith.
- CASA `EVLA_C` input files live beside the generic VLA inputs. Do not
  overwrite the frozen generic high-resolution artifact.

The three-way comparison product is
`holoraster_coordinate_feed_comparison_v1`. The EVLA-C development
plane at 4.564 GHz is
`/media/stephen/astro/vla/beam_models/cassbeam_evla_c_g1024_p32_20260908`.
It does not replace the frozen generic artifact.

Identity of generic CASSBEAM at commanded offsets matched the frozen
HOLORASTER path on all 538,489 development rows.

Generic CASSBEAM at `source_lm_feed` improved the mover holdout
(\(\Delta L=-0.0030\), interval entirely below zero, all five movers)
but not the spatial holdout (interval crosses zero). The coordinate
sign alone is therefore not a production lock.

EVLA-C at `source_lm_feed` beat generic CASSBEAM at the same corrected
coordinates on both holdouts
(spatial \(\Delta L=-0.0056\), mover \(\Delta L=-0.0061\), both
intervals entirely below zero, all five movers). RR−LL residual power
fell from 1.00 to 0.88. That separates the feed-parameter effect from
the coordinate effect.

The EVLA-C plane 20%-of-peak vector is
\((0.119,+0.502)\) arcmin (`r_minus_l`), separation \(0.516'\).
The holography vector in source-in-beam labels is
\((-0.252,+0.675)\) arcmin. Those directions differ by about
\(34^\circ\), versus about \(114^\circ\) for the old generic
orientation. Width 1.04 remained interior in both coordinate frames.

Do not freeze a production beam. Leave SPW 5 sealed.

## Next comparison, in order

1. Current generic artifact with current (commanded) coordinates.
2. Current generic artifact with corrected `source_lm_feed`.
3. A new EVLA-C-parameterised artifact with corrected coordinates.

Then rerun SPW-4 HOLORASTER plots, diagonal holdouts, and width/squint
measurements. Only search a convention that is still unresolved after
those deterministic corrections.
