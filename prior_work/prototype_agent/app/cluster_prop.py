"""Multi-scale closed-cluster domain features and Stage-2 cluster score propagation.

Why this helps:
1. Every planted motif scenario (both easy ~5 min and medium ~60-180 min) is a small closed group of
   first-time (novel) payments. While individual edges inside the same scenario receive different Stage-1
   scores (entry/exit hops vs middle hops), propagating Stage-1 scores across 2-hop/3-hop closed novel-edge
   neighbourhoods at multiple time scales (5m, 15m, 30m, 1h, 2h, 4h) enforces scenario-wide consistency:
   - Short windows (5m, 15m) average out single-edge spikes in fast normal twins and lift lagging legs of easy scenarios.
   - Longer windows (30m, 1h, 2h, 4h) use upper quantiles (q75, max) to pull up slow medium laundering scenarios.
2. Closed-cluster domain features capture intermediary pass-through ratio (out_usd / in_usd ~0.96 in laundering
   vs ~0.83 in normal twins), conversion share, and inter-step timing regularity across all scales.
"""
from __future__ import annotations

from collections import defaultdict
import numpy as np
import pandas as pd

PROP_COLS = [
    "s1_score",
    "c_s1_mean_5m", "c_s1_max_5m",
    "c_s1_mean_15m", "c_s1_max_15m",
    "c_s1_q75_30m", "c_s1_max_1h",
    "c_s1_q75_2h", "c_s1_max_2h",
    "c_s1_q75_4h",
]
PROP_WEIGHTS = np.array([0.22, 0.06, 0.10, 0.06, 0.04, 0.06, 0.14, 0.12, 0.08, 0.12])


def _novel_incidences(e: pd.DataFrame):
    pair_n = e.groupby(["sender", "receiver"]).hash.transform("size").to_numpy()
    novel = pair_n == 1
    t = e.t.to_numpy()
    S = e.sender.to_numpy()
    R = e.receiver.to_numpy()
    inc = defaultdict(list)
    for i in np.flatnonzero(novel):
        inc[S[i]].append((t[i], R[i], i))
        inc[R[i]].append((t[i], S[i], i))
    inc = {
        a: (
            np.array([x[0] for x in l]),
            np.array([x[1] for x in l]),
            np.array([x[2] for x in l]),
        )
        for a, l in inc.items()
    }
    return novel, inc


def _win_rows(inc: dict, a: str, lo: int, hi: int):
    if a not in inc:
        return np.array([], dtype=object), np.array([], int)
    ts, oth, rows = inc[a]
    s = slice(np.searchsorted(ts, lo, "left"), np.searchsorted(ts, hi, "right"))
    return oth[s], rows[s]


def cluster_domain_features(e: pd.DataFrame) -> pd.DataFrame:
    """Multi-scale closed-cluster domain features (pass-through ratio, conversion share, step timing)."""
    idx_index = e.hash.to_numpy()
    e = e.reset_index(drop=True)
    novel, inc = _novel_incidences(e)
    t = e.t.to_numpy()
    S = e.sender.to_numpy()
    R = e.receiver.to_numpy()
    v = e.usd.to_numpy()
    conv = (e.n_legs > 1).astype(float).to_numpy()
    cross_c = (e.src_asset != e.dst_asset).astype(float).to_numpy()

    dom_cols = defaultdict(lambda: np.zeros(len(e)))
    for w_sec, w_name in ((300, "5m"), (900, "15m"), (1800, "30m"), (3600, "1h"), (7200, "2h"), (14400, "4h")):
        for i in range(len(e)):
            lo, hi = t[i] - w_sec, t[i] + w_sec
            so, sr = _win_rows(inc, S[i], lo, hi)
            ro, rr = _win_rows(inc, R[i], lo, hi)
            nodes = set(so) | set(ro) | {S[i], R[i]}
            rows = set(sr) | set(rr) | ({i} if novel[i] else set())
            for a in sorted(nodes - {S[i], R[i]})[:30]:
                oo, orows = _win_rows(inc, a, lo, hi)
                for o, r_ in zip(oo, orows):
                    if o in nodes:
                        rows.add(r_)
                    elif w_sec <= 3600 and len(nodes) <= 15:
                        oo2, _ = _win_rows(inc, o, lo, hi)
                        if any(o2 in nodes for o2 in oo2):
                            rows.add(r_)
            if rows:
                idx = list(rows)
                dom_cols[f"c_conv_{w_name}"][i] = conv[idx].mean()
                dom_cols[f"c_crossc_{w_name}"][i] = cross_c[idx].mean()
                dom_cols[f"c_size_{w_name}"][i] = len(idx)
                if len(idx) > 1:
                    ts_c = np.sort(t[idx])
                    dts = np.diff(ts_c)
                    dom_cols[f"c_span_{w_name}"][i] = (ts_c[-1] - ts_c[0]) / 60.0
                    dom_cols[f"c_dt_med_{w_name}"][i] = np.median(dts)
                    dom_cols[f"c_dt_min_{w_name}"][i] = np.min(dts)
                    dom_cols[f"c_dt_cv_{w_name}"][i] = np.std(dts) / (np.mean(dts) + 1e-9)
                    cs, cr, cu = S[idx], R[idx], v[idx]
                    mids = set(cs) & set(cr)
                    if mids:
                        ratios = [cu[cs == m].sum() / max(cu[cr == m].sum(), 1e-9) for m in mids]
                        dom_cols[f"c_pt_med_{w_name}"][i] = np.median(ratios)
                        dom_cols[f"c_pt_err_{w_name}"][i] = np.median(np.abs(np.array(ratios) - 1.0))
                    else:
                        dom_cols[f"c_pt_med_{w_name}"][i] = -1
                        dom_cols[f"c_pt_err_{w_name}"][i] = -1
            else:
                dom_cols[f"c_pt_med_{w_name}"][i] = -1
                dom_cols[f"c_pt_err_{w_name}"][i] = -1
    return pd.DataFrame(dict(dom_cols), index=idx_index)


def propagate_cluster_scores(e: pd.DataFrame, s1_scores: np.ndarray) -> np.ndarray:
    """Multi-scale closed-cluster score propagation on top of Stage-1 rank scores."""
    e = e.reset_index(drop=True)
    novel, inc = _novel_incidences(e)
    t = e.t.to_numpy()
    S = e.sender.to_numpy()
    R = e.receiver.to_numpy()

    cprop = {"s1_score": s1_scores}
    for w_sec, w_name in ((300, "5m"), (900, "15m"), (1800, "30m"), (3600, "1h"), (7200, "2h"), (14400, "4h")):
        c_mean = np.copy(s1_scores)
        c_max = np.copy(s1_scores)
        c_q75 = np.copy(s1_scores)
        for i in range(len(e)):
            lo, hi = t[i] - w_sec, t[i] + w_sec
            so, sr = _win_rows(inc, S[i], lo, hi)
            ro, rr = _win_rows(inc, R[i], lo, hi)
            nodes = set(so) | set(ro) | {S[i], R[i]}
            rows = set(sr) | set(rr) | ({i} if novel[i] else set())
            for a in sorted(nodes - {S[i], R[i]})[:30]:
                oo, orows = _win_rows(inc, a, lo, hi)
                for o, r_ in zip(oo, orows):
                    if o in nodes:
                        rows.add(r_)
                    elif w_sec <= 3600 and len(nodes) <= 15:
                        oo2, _ = _win_rows(inc, o, lo, hi)
                        if any(o2 in nodes for o2 in oo2):
                            rows.add(r_)
            if rows:
                sc = s1_scores[list(rows)]
                c_mean[i] = sc.mean()
                c_max[i] = sc.max()
                c_q75[i] = np.percentile(sc, 75)
        cprop[f"c_s1_mean_{w_name}"] = c_mean
        cprop[f"c_s1_max_{w_name}"] = c_max
        cprop[f"c_s1_q75_{w_name}"] = c_q75

    mat = np.column_stack([cprop[c] for c in PROP_COLS])
    prop = mat @ PROP_WEIGHTS
    rk = lambda x: pd.Series(x).rank(pct=True).to_numpy()
    return 0.35 * rk(s1_scores) + 0.65 * rk(prop)
