"""Q/I, U/I, EVPA and V/I on 3C286 after the full-pol apply."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from casacore import tables

from sl1mjax.holography_calibration import write_json

GLOBAL_X = ("ea03", "ea11", "ea13", "ea19", "ea25")


def _stokes(vis: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rr, rl, lr, ll = vis[..., 0], vis[..., 1], vis[..., 2], vis[..., 3]
    stokes_i = 0.5 * (rr + ll)
    stokes_v = 0.5 * (rr - ll)
    stokes_q = 0.5 * (rl + lr)
    stokes_u = 0.5j * (lr - rl)
    return stokes_i, stokes_q, stokes_u, stokes_v


def _ratio(num: np.ndarray, den: np.ndarray) -> float:
    usable = np.isfinite(num) & np.isfinite(den) & (np.abs(den) > 1.0e-6)
    if not np.any(usable):
        return float("nan")
    return float(np.median(np.real(num[usable] / den[usable])))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("measurement_set", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--edge-channels", type=int, default=5)
    arguments = parser.parse_args()
    with tables.table(
        str(arguments.measurement_set / "ANTENNA"), readonly=True, ack=False
    ) as table:
        names = [str(name) for name in table.getcol("NAME")]
    with tables.table(str(arguments.measurement_set), readonly=True, ack=False) as main:
        vis = np.asarray(main.getcol("CORRECTED_DATA"))
        flag = np.asarray(main.getcol("FLAG"), dtype=bool)
        antenna1 = np.asarray(main.getcol("ANTENNA1"), dtype=np.int32)
        antenna2 = np.asarray(main.getcol("ANTENNA2"), dtype=np.int32)
        ddid = np.asarray(main.getcol("DATA_DESC_ID"), dtype=np.int32)
    vis = np.where(flag, np.nan, vis)
    nchan = vis.shape[1]
    edge = arguments.edge_channels
    holdout = np.zeros(nchan, dtype=bool)
    holdout[:edge] = True
    holdout[-edge:] = True
    inner = ~holdout
    i, q, u, v = _stokes(vis)
    evpa = 0.5 * np.rad2deg(np.arctan2(np.real(u), np.real(q)))

    def summary(mask_chan: np.ndarray) -> dict[str, float]:
        return {
            "q_over_i": _ratio(q[:, mask_chan], i[:, mask_chan]),
            "u_over_i": _ratio(u[:, mask_chan], i[:, mask_chan]),
            "v_over_i": _ratio(v[:, mask_chan], i[:, mask_chan]),
            "evpa_deg": float(np.nanmedian(evpa[:, mask_chan])),
        }

    by_spw = {}
    for spw in np.unique(ddid):
        rows = ddid == spw
        by_spw[str(int(spw))] = {
            "inner": {
                "q_over_i": _ratio(q[rows][:, inner], i[rows][:, inner]),
                "u_over_i": _ratio(u[rows][:, inner], i[rows][:, inner]),
                "v_over_i": _ratio(v[rows][:, inner], i[rows][:, inner]),
                "evpa_deg": float(np.nanmedian(evpa[rows][:, inner])),
            },
            "xf_excluded_edges": {
                "q_over_i": _ratio(q[rows][:, holdout], i[rows][:, holdout]),
                "u_over_i": _ratio(u[rows][:, holdout], i[rows][:, holdout]),
                "v_over_i": _ratio(v[rows][:, holdout], i[rows][:, holdout]),
                "evpa_deg": float(np.nanmedian(evpa[rows][:, holdout])),
            },
        }
    global_x_rows = np.zeros(vis.shape[0], dtype=bool)
    for index, name in enumerate(names):
        if name in GLOBAL_X:
            global_x_rows |= (antenna1 == index) | (antenna2 == index)
    report = {
        "independence": "apply_back_on_xf_excluded_edge_channels",
        "inner_channels": summary(inner),
        "xf_excluded_edges": {
            "q_over_i": _ratio(q[:, holdout], i[:, holdout]),
            "u_over_i": _ratio(u[:, holdout], i[:, holdout]),
            "v_over_i": _ratio(v[:, holdout], i[:, holdout]),
            "evpa_deg": float(np.nanmedian(evpa[:, holdout])),
        },
        "by_spectral_window": by_spw,
        "global_x_baselines": {
            "n_rows": int(np.sum(global_x_rows)),
            "xf_excluded_edges": {
                "q_over_i": _ratio(q[global_x_rows][:, holdout], i[global_x_rows][:, holdout]),
                "u_over_i": _ratio(u[global_x_rows][:, holdout], i[global_x_rows][:, holdout]),
                "v_over_i": _ratio(v[global_x_rows][:, holdout], i[global_x_rows][:, holdout]),
            },
        },
        "notes": (
            "Inner channels contributed to Xf; edges are the excluded-channel apply-back",
            "Global-X antennas carry the same Xf as the reference frame",
        ),
    }
    print(
        {
            "inner": report["inner_channels"],
            "xf_excluded_edges": report["xf_excluded_edges"],
            "global_x_rows": report["global_x_baselines"]["n_rows"],
        }
    )
    if arguments.output is not None:
        write_json(report, arguments.output)
        print(arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
