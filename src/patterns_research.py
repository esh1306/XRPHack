"""patterns_research.py - Solo Validator's own research-based laundering-pattern layer (v11).

Per-transaction pattern features computed from the transaction graph of one split (time-windowed motifs),
each tied to a published source:
  [AMLworld]  Altman et al., "Realistic Synthetic Financial Transactions for Anti-Money Laundering Models",
              NeurIPS 2023 Datasets & Benchmarks. Laundering patterns: fan-out, fan-in, scatter-gather,
              gather-scatter, cycle, bipartite, stack, random chain.
  [Egressy]   Egressy et al., "Provably Powerful Graph Neural Networks for Directed Multigraphs", AAAI 2024.
              Reverse-edge (reverse message passing), port numbering (edge order at a node), ego-ID (cycles
              through the ego node) -> implemented here as explicit features.
  [Elliptic]  Weber et al., "Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks
              for Financial Forensics", KDD 2019 workshop. Local + one-hop neighbourhood aggregate features.
  [FATF]      FATF typologies (e.g. "Money Laundering and Terrorist Financing Red Flag Indicators Associated
              with Virtual Assets", 2020): structuring just below thresholds, rapid pass-through / peel chains.
A small gradient-boosted model is fit on these features only (train labels), giving a pattern score that is
rank-blended into the v9c pipeline. No external services. Fixed seed 42.

Usage:
  python src/patterns_research.py cv     # grouped CV on train (StratifiedGroupKFold(5) by scenario_id, seed 42)
  python src/patterns_research.py test OUT_PARQUET   # fit on train, score test edges
Input: prototype edge tables prototype/cache/edges_{train,test}.parquet (built from data/ by the prototype loader).
"""
import os, sys, numpy as np, pandas as pd
from scipy.stats import rankdata

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
W1, W6 = 3600, 6 * 3600
SEED = 42


def _win_count(key, t, qkey, qt, lo, hi, val=None):
    """For each query (qkey, qt): count (and value sum) of rows with key==qkey and t in (qt+lo, qt+hi]."""
    o = np.lexsort((t, key)); k, tt = key[o], t[o]
    v = None if val is None else np.concatenate([[0.0], np.cumsum(val[o])])
    kk = k.astype(np.int64) * (10 ** 11) + tt  # composite sortable key (t < 1e11)
    a = np.searchsorted(kk, qkey.astype(np.int64) * (10 ** 11) + qt + lo, side="right")
    b = np.searchsorted(kk, qkey.astype(np.int64) * (10 ** 11) + qt + hi, side="right")
    c = (b - a).astype(float)
    return c if v is None else (c, v[b] - v[a])


def features(e):
    e = e.reset_index(drop=True)
    acc = pd.Index(pd.unique(np.concatenate([e.sender.values, e.receiver.values])))
    s = acc.get_indexer(e.sender).astype(np.int64); r = acc.get_indexer(e.receiver).astype(np.int64)
    t = e.t.to_numpy(np.int64); u = e.usd.fillna(0).to_numpy(float); n = len(e)
    F = {}
    for W, tag in ((W1, "1h"), (W6, "6h")):
        # [AMLworld] fan-out / fan-in (sender out-burst, receiver in-burst around t)
        F[f"fanout_{tag}"] = _win_count(s, t, s, t, -W, W)
        F[f"fanin_{tag}"] = _win_count(r, t, r, t, -W, W)
        # [Egressy] reverse edges: sender's incoming activity before t; receiver's outgoing after t
        F[f"snd_in_before_{tag}"], sin_usd = _win_count(r, t, s, t, -W, -1, u)
        F[f"rcv_out_after_{tag}"], rout_usd = _win_count(s, t, r, t, 0, W, u)
        # [FATF] rapid pass-through ratios (money in ~= money out quickly)
        F[f"pass_rcv_{tag}"] = np.clip(rout_usd / np.maximum(u, 1e-9), 0, 5)
        F[f"pass_snd_{tag}"] = np.clip(u / np.maximum(sin_usd, 1e-9), 0, 5) * (sin_usd > 0)
        # [AMLworld] gather-scatter at receiver: in-burst before x out-burst after
        F[f"gather_scatter_{tag}"] = _win_count(r, t, r, t, -W, 0) * F[f"rcv_out_after_{tag}"]
    # [Egressy] port numbering: order of this edge among sender's out-edges and receiver's in-edges
    F["port_out"] = pd.Series(t).groupby(s).rank(method="first").to_numpy()
    F["port_in"] = pd.Series(t).groupby(r).rank(method="first").to_numpy()
    deg_out = np.bincount(s, minlength=len(acc)); deg_in = np.bincount(r, minlength=len(acc))
    # [Elliptic] neighbourhood aggregates
    F["snd_deg_out"], F["snd_deg_in"] = deg_out[s], deg_in[s]
    F["rcv_deg_out"], F["rcv_deg_in"] = deg_out[r], deg_in[r]
    su = pd.Series(u).groupby(s).transform("mean").to_numpy()
    F["usd_vs_snd_mean"] = u / np.maximum(su, 1e-9)
    F["log_usd"] = np.log1p(u)
    pair = s * len(acc) + r
    F["pair_count"] = pd.Series(pair).map(pd.Series(pair).value_counts()).to_numpy()
    # [FATF] structuring: just below round thresholds
    th = np.array([1000, 3000, 5000, 10000, 15000])
    rel = u[:, None] / th[None, :]
    F["just_below_thr"] = ((rel > 0.9) & (rel < 1.0)).any(1).astype(float)
    # --- 2-hop paths within 6h (bounded): edge i = s->m, edge j = m->z with t_j in (t_i, t_i+W6]
    E = pd.DataFrame({"i": np.arange(n), "s": s, "r": r, "t": t, "u": u})
    hub = deg_out[r] <= 50
    A = E[hub].merge(E.rename(columns={"i": "j", "s": "r", "r": "z", "t": "t2", "u": "u2"}), on="r")
    A = A[(A.t2 > A.t) & (A.t2 <= A.t + W6)]
    # [AMLworld/Egressy] 2-cycle (ego-ID): m sends back to s
    F["cycle2"] = np.bincount(A.i[A.z == A.s], minlength=n).astype(float)
    # [AMLworld] scatter-gather: distinct mids from s that reach the same z
    sg = A[A.z != A.s].groupby(["s", "z"]).r.nunique().rename("nm").reset_index()
    A2 = A.merge(sg, on=["s", "z"], how="left")
    F["scatter_gather"] = A2.groupby("i").nm.max().reindex(range(n)).fillna(0).to_numpy()
    # [AMLworld] bipartite: distinct z reached from s via 2-hop paths
    F["bipartite_z"] = A.groupby("i").z.nunique().reindex(range(n)).fillna(0).to_numpy()
    # [AMLworld/FATF] stack / peel chain: follow forward edge carrying 50-105% of the amount, depth <= 4
    P = A[(A.u2 >= 0.5 * A.u) & (A.u2 <= 1.05 * A.u)].sort_values("t2").drop_duplicates("i")
    nxt = np.full(n, -1); nxt[P.i.to_numpy()] = P.j.to_numpy()
    depth = np.zeros(n); cur = np.arange(n)
    for _ in range(4):
        ok = cur >= 0; cur = np.where(ok, nxt[np.maximum(cur, 0)], -1); depth += (cur >= 0)
    F["chain_depth"] = depth
    # [AMLworld/Egressy] 3-cycle: s->m->z->s within 6h (bounded by hub filter)
    B = A[(A.z != A.s) & (deg_out[A.z.to_numpy()] <= 50)][["i", "s", "z", "t", "t2"]]
    C = B.merge(E[["s", "r", "t"]].rename(columns={"s": "z", "r": "s", "t": "t3"}), on=["z", "s"])
    C = C[(C.t3 > C.t2) & (C.t3 <= C.t + W6)]
    F["cycle3"] = np.bincount(C.i, minlength=n).astype(float)
    return pd.DataFrame(F, index=e.hash.values).astype("float32")


def _model():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                          l2_regularization=1.0, random_state=SEED)


def load(split):
    return pd.read_parquet(os.path.join(ROOT, "prototype", "cache", f"edges_{split}.parquet"))


if __name__ == "__main__":
    mode = sys.argv[1]
    e = load("train"); X = features(e)
    L = pd.read_csv(os.path.join(ROOT, "data/train/labels.csv")).set_index("TRANSACTION_HASH").loc[e.hash]
    y = L.is_suspicious.to_numpy()
    if mode == "cv":
        from sklearn.model_selection import StratifiedGroupKFold
        from sklearn.metrics import average_precision_score as AP
        g = L.scenario_id.fillna(pd.Series(L.index, index=L.index)).to_numpy()
        oof = np.zeros(len(y))
        for tr, va in StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(X, y, g):
            oof[va] = _model().fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[va])[:, 1]
        print("pattern-only OOF AP", round(100 * AP(y, oof), 2), flush=True)
        pd.DataFrame({"TRANSACTION_HASH": e.hash.values, "oof": oof}).to_parquet(os.path.join(ROOT, "cache/pat_oof.parquet"))
    else:
        et = load("test"); Xt = features(et)
        p = _model().fit(X, y).predict_proba(Xt)[:, 1]
        pd.DataFrame({"TRANSACTION_HASH": et.hash.values, "score": p}).to_parquet(sys.argv[2])
        print("wrote", sys.argv[2], len(p))
