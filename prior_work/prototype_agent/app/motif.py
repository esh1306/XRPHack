"""Novel-edge motif features: suspicious scenarios are small clusters of first-time payments between the same
few accounts over a few hours. Normal first-time payments are scattered one-offs."""
from collections import defaultdict
import numpy as np, pandas as pd

HOUR = 3600


def motif_features(e: pd.DataFrame) -> pd.DataFrame:
    pair_n = e.groupby(["sender", "receiver"]).hash.transform("size").to_numpy()
    novel = pair_n == 1
    t = e.t.to_numpy(); S = e.sender.to_numpy(); R = e.receiver.to_numpy(); v = e.usd.to_numpy()
    # per-account novel incidences (time, other, row), time-sorted
    inc = defaultdict(list)
    for i in np.flatnonzero(novel):
        inc[S[i]].append((t[i], R[i], i)); inc[R[i]].append((t[i], S[i], i))
    inc = {a: (np.array([x[0] for x in l]), np.array([x[1] for x in l]), np.array([x[2] for x in l])) for a, l in inc.items()}

    def window(a, lo, hi):
        if a not in inc:
            return np.array([], dtype=object), np.array([], int)
        ts, oth, rows = inc[a]
        s = slice(np.searchsorted(ts, lo, "left"), np.searchsorted(ts, hi, "right"))
        return oth[s], rows[s]

    out = defaultdict(lambda: np.zeros(len(e)))
    for i in range(len(e)):
        for w, name in ((300, "5m"), (600, "10m"), (1200, "20m"), (HOUR, "1h"), (3 * HOUR, "3h"), (6 * HOUR, "6h")):
            lo, hi = t[i] - w, t[i] + w
            so, sr = window(S[i], lo, hi)
            ro, rr = window(R[i], lo, hi)
            out[f"m_s_novel_{name}"][i] = len(sr)
            out[f"m_r_novel_{name}"][i] = len(rr)
            nodes = set(so) | set(ro) | {S[i], R[i]}
            rows = set(sr) | set(rr)
            # second hop: novel edges of neighbours that stay inside the neighbourhood (closure = tight cluster)
            closed, ext = 0, 0
            for a in list(nodes - {S[i], R[i]})[:30] if False else sorted(nodes - {S[i], R[i]})[:30]:
                oo, orows = window(a, lo, hi)
                for o, r_ in zip(oo, orows):
                    if r_ in rows:
                        continue
                    if o in nodes:
                        closed += 1
                    else:
                        ext += 1
                    rows.add(r_)
            out[f"m_nodes_{name}"][i] = len(nodes)
            out[f"m_closed_{name}"][i] = closed
            out[f"m_ext_{name}"][i] = ext
            out[f"m_cluster_edges_{name}"][i] = len(set(sr) | set(rr)) + closed
            out[f"m_density_{name}"][i] = (len(set(sr) | set(rr)) + closed) / max(1, len(nodes))
            if name in ("1h", "3h"):  # how tight in time is the surrounding one-off cluster
                rows_c = list(set(sr) | set(rr))
                if len(rows_c) > 1:
                    tc = t[rows_c]
                    out[f"m_span_min_{name}"][i] = (tc.max() - tc.min()) / 60
                    out[f"m_absdt_med_min_{name}"][i] = np.median(np.abs(tc - t[i])) / 60
                else:
                    out[f"m_span_min_{name}"][i] = -1
                    out[f"m_absdt_med_min_{name}"][i] = -1
            if name == "3h":
                vals = v[list(set(sr) | set(rr))] if len(set(sr) | set(rr)) else np.array([v[i]])
                out["m_amount_cv_3h"][i] = vals.std() / max(vals.mean(), 1e-9)
                out["m_amount_ratio_med_3h"][i] = v[i] / max(np.median(vals), 1e-9)
    f = pd.DataFrame(dict(out))
    f["m_is_novel"] = novel.astype(int)
    for short_w, long_w in (("5m", "1h"), ("10m", "1h"), ("20m", "1h"), ("10m", "3h")):
        f[f"m_tight_{short_w}_{long_w}"] = f[f"m_cluster_edges_{short_w}"] / f[f"m_cluster_edges_{long_w}"].clip(lower=1)
    f.index = e.hash.to_numpy()
    return f
