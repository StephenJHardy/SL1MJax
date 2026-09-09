"""Read-only, diagonal visibility test for catalogue contamination of holography.

Geometry is explicit: source LMN is relative to the correlator phase centre;
source-in-beam LM is supplied independently for antenna p and q. The latter
must come from the actual pointing/frame transform, not just negating the
HOLORASTER commanded offset. This module never reads or changes an MS.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from sl1mjax.catalog import RadioCatalogSource

C_M_S = 299_792_458.0
BeamLookup = Callable[[float, NDArray, NDArray], tuple[NDArray, NDArray]]


@dataclass(frozen=True)
class BackgroundGeometry:
    """Single-channel rows, arbitrary number of catalogue sources.

    ``source_lmn``: (row, source, 3), exact phase-centre direction cosines.
    ``beam_[pq]_lmn``: (row, source, 3), feed-frame source-in-beam directions.
    ``uvw_m`` is in the original MS p/q order. Never reverse only the beams.
    """

    uvw_m: NDArray
    frequency_hz: float
    source_lmn: NDArray
    beam_p_lmn: NDArray
    beam_q_lmn: NDArray

    def validate(self, n_sources: int) -> None:
        uvw = np.asarray(self.uvw_m)
        if uvw.ndim != 2 or uvw.shape[1] != 3 or len(uvw) == 0:
            raise ValueError("uvw_m must be nonempty (row,3)")
        shapes = {
            "source_lmn": (len(uvw), n_sources, 3),
            "beam_p_lmn": (len(uvw), n_sources, 3),
            "beam_q_lmn": (len(uvw), n_sources, 3),
        }
        for name, shape in shapes.items():
            arr = np.asarray(getattr(self, name))
            if arr.shape != shape or not np.all(np.isfinite(arr)):
                raise ValueError(f"{name} must be finite with shape {shape}")
            if not np.allclose(np.sum(arr**2, axis=-1), 1, rtol=0, atol=1e-10):
                raise ValueError(f"{name} must contain unit direction cosines")
        if not np.all(np.isfinite(uvw)):
            raise ValueError("nonfinite UVW")
        if not np.isfinite(self.frequency_hz) or self.frequency_hz <= 0:
            raise ValueError("frequency must be finite and positive")
        lmn = np.asarray(self.source_lmn)
        if not np.allclose(np.sum(lmn**2, axis=-1), 1, rtol=0, atol=1e-10):
            raise ValueError("source_lmn must be unit direction cosines")
        if np.any(lmn[..., 2] <= 0):
            raise ValueError("source is outside the phase-centre hemisphere")


def source_fluxes(
    sources: Sequence[RadioCatalogSource],
    frequency_hz: float,
    *,
    default_spectral_index: float | None = None,
) -> NDArray:
    """Fixed catalogue predictions; missing spectral indices need an explicit prior."""
    if not np.isfinite(frequency_hz) or frequency_hz <= 0:
        raise ValueError("frequency must be finite and positive")
    flux = []
    for source in sources:
        alpha = source.spectral_index
        if alpha is None:
            alpha = default_spectral_index
        if alpha is None:
            raise ValueError(f"{source.name}: spectral index required (or explicit default)")
        if not np.isfinite(alpha) or source.reference_frequency_hz <= 0:
            raise ValueError("invalid spectral model")
        value = source.integrated_flux_jy * (frequency_hz / source.reference_frequency_hz) ** alpha
        if not np.isfinite(value) or value < 0:
            raise ValueError("invalid extrapolated source flux")
        flux.append(value)
    return np.asarray(flux)


def predict_background(
    geometry: BackgroundGeometry,
    flux_jy: NDArray,
    lookup: BeamLookup,
    *,
    row_chunk: int = 2048,
    residual_p: NDArray | None = None,
    residual_q: NDArray | None = None,
) -> tuple[NDArray, NDArray, list[dict]]:
    """Return summed RR/LL, all-source support by hand, and source ranking.

    Sources are unresolved, unpolarized, fixed-flux catalogue hypotheses.
    Samples outside either antenna beam support are unknown, never known zero.
    No source-by-row-by-channel prediction cube is retained.
    """
    flux = np.asarray(flux_jy, dtype=float)
    if flux.ndim != 1 or not len(flux) or np.any(~np.isfinite(flux)) or np.any(flux < 0):
        raise ValueError("flux_jy must be nonempty, finite and nonnegative")
    if row_chunk < 1:
        raise ValueError("row_chunk must be positive")
    geometry.validate(len(flux))
    n = len(geometry.uvw_m)
    residuals = []
    for value in (residual_p, residual_q):
        arr = (
            np.broadcast_to(np.eye(2, dtype=complex), (n, 2, 2))
            if value is None
            else np.asarray(value)
        )
        if arr.shape != (n, 2, 2) or not np.all(np.isfinite(arr)):
            raise ValueError("residual Jones must be finite (row,2,2) in sky basis")
        residuals.append(arr)
    rp, rq = residuals
    prediction = np.zeros((n, 2), dtype=complex)
    support = np.ones((n, 2), dtype=bool)
    ranking = []
    for s, intensity in enumerate(flux):
        peak = np.zeros(2)
        sum_power = np.zeros(2)
        counts = np.zeros(2, dtype=int)
        for start in range(0, n, row_chunk):
            sl = slice(start, min(start + row_chunk, n))
            p = geometry.beam_p_lmn[sl, s]
            q = geometry.beam_q_lmn[sl, s]
            ep, vp = lookup(geometry.frequency_hz, p[:, 0], p[:, 1])
            eq, vq = lookup(geometry.frequency_hz, q[:, 0], q[:, 1])
            ep, eq = np.asarray(ep), np.asarray(eq)
            size = len(p)
            if ep.shape != (size, 2) or eq.shape != (size, 2):
                raise ValueError("beam lookup must return (row,2) complex R/L voltage")
            vp, vq = np.asarray(vp, dtype=bool), np.asarray(vq, dtype=bool)
            if vp.shape != ep.shape or vq.shape != eq.shape:
                raise ValueError("beam lookup validity must be (row,2)")
            good = vp & vq & np.isfinite(ep) & np.isfinite(eq)
            good &= (p[:, 2, None] > 0) & (q[:, 2, None] > 0)
            lmn = np.array(geometry.source_lmn[sl, s], copy=True)
            lmn[:, 2] -= 1
            phase = np.exp(
                2j
                * np.pi
                * geometry.frequency_hz
                / C_M_S
                * np.sum(geometry.uvw_m[sl] * lmn, axis=1)
            )
            vis = intensity * ep * np.conjugate(eq) * phase[:, None]
            # Fixed, direction-independent residual factors in the calibrated
            # sky basis: diag(Rp diag(vis_R,vis_L) Rq^H). Never refit them here.
            mixing = rp[sl] * np.conjugate(rq[sl])
            vis = np.einsum("nij,nj->ni", mixing, np.where(good, vis, 0))
            good = np.all(good[:, None, :] | (mixing == 0), axis=-1)
            prediction[sl] += np.where(good, vis, 0)
            support[sl] &= good
            amp = np.where(good, np.abs(vis), 0)
            peak = np.maximum(peak, np.max(amp, axis=0))
            sum_power += np.sum(amp**2, axis=0)
            counts += np.sum(good, axis=0)
        ranking.append(
            {
                "source_index": s,
                "flux_jy": float(intensity),
                "peak_apparent_jy_rr_ll": peak.tolist(),
                "rms_apparent_jy_rr_ll": np.sqrt(
                    np.divide(sum_power, counts, out=np.zeros(2), where=counts > 0)
                ).tolist(),
                "supported_rows_rr_ll": counts.tolist(),
            }
        )
    prediction[~support] = np.nan
    return prediction, support, ranking


def compare_background(
    measured: NDArray,
    calibrator: NDArray,
    background: NDArray,
    weight: NDArray,
    flag: NDArray,
    support: NDArray,
    score_mask: NDArray,
    cluster_id: NDArray,
    *,
    n_boot: int = 400,
    seed: int = 0,
) -> dict:
    """Fixed-model paired RR/LL scoring, cluster bootstrap of reduced sums.

    Report excluded support explicitly. No fitting, trimming, reflagging, or
    model promotion. Clusters should be dwell/scan groups, not individual rows.
    """
    arrays = [np.asarray(x) for x in (measured, calibrator, background, weight, flag, support)]
    measured, calibrator, background, weight, flag, support = arrays
    shape = measured.shape
    if len(shape) != 2 or shape[1] != 2 or any(x.shape != shape for x in arrays):
        raise ValueError("visibility/weight/flag/support arrays must have shape (row,2)")
    if flag.dtype != bool or support.dtype != bool:
        raise ValueError("flag and support must be boolean")
    mask = np.asarray(score_mask)
    clusters = np.asarray(cluster_id)
    if mask.shape != (shape[0],) or mask.dtype != bool or clusters.shape != mask.shape:
        raise ValueError("score_mask must be boolean (row,); cluster_id must be (row,)")
    if n_boot < 2:
        raise ValueError("n_boot must be at least 2")
    result = {"model_fitted": False, "production_accepted": False, "hands": {}}
    rng = np.random.default_rng(seed)
    for hand, i in (("RR", 0), ("LL", 1)):
        eligible = (
            mask
            & ~flag[:, i]
            & np.isfinite(weight[:, i])
            & (weight[:, i] > 0)
            & np.isfinite(measured[:, i])
            & np.isfinite(calibrator[:, i])
        )
        good = eligible & support[:, i] & np.isfinite(background[:, i])
        record = {
            "eligible_rows": int(eligible.sum()),
            "scored_rows": int(good.sum()),
            "unsupported_rows": int(np.sum(eligible & ~good)),
        }
        if not np.any(good):
            record["status"] = "no_supported_samples"
            result["hands"][hand] = record
            continue
        y, a, b, w = (x[good, i] for x in (measured, calibrator, background, weight))
        labels, inverse = np.unique(clusters[good], return_inverse=True)
        base = w * abs(y - a) ** 2
        candidate = w * abs(y - a - b) ** 2
        denom = w * abs(y) ** 2
        reduced = np.stack(
            [
                np.bincount(inverse, weights=x, minlength=len(labels))
                for x in (base, candidate, denom)
            ],
            axis=1,
        )
        sums = reduced.sum(axis=0)
        record["n_clusters"] = len(labels)
        if sums[2] == 0:
            record["status"] = "zero_observed_power"
        else:
            record.update(
                {
                    "status": "scored_development_diagnostic",
                    "calibrator_only_power": float(sums[0] / sums[2]),
                    "plus_catalogue_power": float(sums[1] / sums[2]),
                    "paired_delta": float((sums[1] - sums[0]) / sums[2]),
                }
            )
            draws = []
            if len(labels) >= 2:
                for _ in range(n_boot):
                    draw = reduced[rng.integers(len(labels), size=len(labels))].sum(axis=0)
                    if draw[2] > 0:
                        draws.append((draw[1] - draw[0]) / draw[2])
            record["paired_ci95"] = np.quantile(draws, [0.025, 0.975]).tolist() if draws else None
        result["hands"][hand] = record
    return result
