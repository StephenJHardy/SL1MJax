# THOL0001 SPW-4 discrete spatial-convention test

This is **SPW-4 development**. CASSBEAM squint stays physically present.
Its current coordinate orientation is not treated as the production
prior. Width is frozen at 1.04. Magnitude is not fitted. The 128-member
CASSBEAM convention ladder stays closed. SPW 5 stays sealed.

Product:
`/media/stephen/astro/vla/extracted/commissioning/validation/scientific/holoraster_spatial_convention_v1`.

The previous physical-squint experiment measured a stable R/L
displacement near \((0.252,-0.675)\) arcmin, while the frozen CASSBEAM
centres imply roughly \((0.292,+0.290)\) arcmin. Those directions differ
by about \(114^\circ\). Native CASSBEAM loses to no-squint; the measured
direction beats native. This test asks whether a discrete \(l/m\) sign,
axis swap, or R/L swap of the **complete diagonal beam** accounts for
that mismatch.

## Design

1. Freeze width at 1.04.
2. Enumerate only the 16 sign / swap / R/L candidates.
3. Transform the complete CASSBEAM diagonal beam, not merely its
   centroid: \(x_t=T^{-1}(x)\), then \(q_h=c_h+(x_t-c_h)/w\).
4. Rank on SPW-4 training rows using RR, LL, and especially RR−LL.
5. Compare the selected transform with no-squint and the empirical-vector
   model, and compute the missing paired empirical-versus-no-squint
   interval.
6. Lock only if the selected transform aligns the map-domain vector and
   improves paired spatial and mover scores versus both native and
   no-squint.
7. Leave magnitude adjustment for after the orientation is fixed.
8. Leave SPW 5 sealed.

Identity \(T\) plus \(w=1\) must reproduce the frozen HOLORASTER
comparison. Identity \(T\) plus \(w=1.04\) must reproduce the previous
native-plus-width physical state.

The Bacchus product finished. Identity was exact on all 538,489
development rows. Train ranking selected `l+1_m-1`. The paired lock
gates did **not** pass. Do not interpret that selection as a physical
orientation: the HOLORASTER path was still querying CASSBEAM at the
commanded offset, and the CASSBEAM artifact is still the generic VLA
feed. See `thol0001_holoraster_coordinate_and_feed_audit.md`.

## Policy

Retain the fact and physical role of CASSBEAM squint. Do not retain the
current CASSBEAM coordinate orientation as the production prior until a
discrete convention locks.
