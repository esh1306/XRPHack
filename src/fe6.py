"""v3 features: fuzzy pattern counts inspired by BlazingAML (arXiv:2604.12241) -- scatter-gather with variable
intermediates, gather-scatter, loosely time-ordered 3-cycles, windowed fan-in/out -- and cheap parts of TCEE
(arXiv:2610.29667): w=8 neighbourhood windows in the sender's/receiver's own history (relative amounts, deltas,
direction, counterparty index/novelty) and PNA-style pair pooling. Own implementation, label-free, per split."""
import sys, warnings, numpy as np, pandas as pd
from collections import defaultdict
warnings.filterwarnings("ignore")
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from fe import load_split


def build(tx):
    S = tx.ACCOUNT.to_numpy(); R = tx.DEST.to_numpy(); T = tx.TIMESTAMP.to_numpy().astype(np.int64)
    V = np.nan_to_num(tx.usd.to_numpy().astype(float)); lv = np.log1p(V); n = len(tx); F = {}
    out_i = defaultdict(list)
    for i in np.argsort(T, kind="stable"): out_i[S[i]].append(i)
    out_i = {k: np.array(v) for k, v in out_i.items()}
    # 2-hop time-respecting paths i (S->M) then j (M->D), t_i <= t_j <= t_i + w
    for w in (3600, 21600):
        P1, P2 = [], []
        for i in range(n):
            ix = out_i.get(R[i])
            if ix is None: continue
            t = T[ix]; lo, hi = np.searchsorted(t, T[i], "left"), np.searchsorted(t, T[i] + w, "right")
            j = ix[lo:hi]; j = j[j != i]
            P1.append(np.full(len(j), i)); P2.append(j)
        P1 = np.concatenate(P1); P2 = np.concatenate(P2)
        p = pd.DataFrame({"i": P1, "j": P2, "S": S[P1], "M": R[P1], "D": R[P2], "day": T[P1] // (w * 4)})
        p = p[(p.D != p.S)]
        sg = p.groupby(["S", "D", "day"]).M.transform("nunique")
        p["sg"] = sg.to_numpy()
        h = w // 3600
        F[f"sg_k_first_{h}h"] = pd.Series(p.sg.to_numpy()).groupby(p.i.to_numpy()).max().reindex(range(n)).fillna(0).to_numpy()
        F[f"sg_k_second_{h}h"] = pd.Series(p.sg.to_numpy()).groupby(p.j.to_numpy()).max().reindex(range(n)).fillna(0).to_numpy()
        F[f"sg_inst_{h}h"] = (pd.Series(p.sg.to_numpy() > 1).groupby(p.i.to_numpy()).sum().reindex(range(n)).fillna(0).to_numpy()
                              + pd.Series(p.sg.to_numpy() > 1).groupby(p.j.to_numpy()).sum().reindex(range(n)).fillna(0).to_numpy())
        F[f"paths2_out_{h}h"] = np.bincount(p.i, minlength=n); F[f"paths2_in_{h}h"] = np.bincount(p.j, minlength=n)
        # gather-scatter: many sources into M then many destinations out of M
        gs_src = p.groupby(["M", "day"]).S.transform("nunique").to_numpy(); gs_dst = p.groupby(["M", "day"]).D.transform("nunique").to_numpy()
        gs = pd.Series(np.minimum(gs_src, gs_dst))
        F[f"gs_first_{h}h"] = gs.groupby(p.i.to_numpy()).max().reindex(range(n)).fillna(0).to_numpy()
        F[f"gs_second_{h}h"] = gs.groupby(p.j.to_numpy()).max().reindex(range(n)).fillna(0).to_numpy()
        # 3-cycles: S->M->D then D->S within w after j
        c3 = np.zeros(n)
        key = defaultdict(list)
        for k in range(n): key[(S[k], R[k])].append(T[k])
        key = {k: np.sort(v) for k, v in key.items()}
        for i, j, s, d in zip(p.i.to_numpy(), p.j.to_numpy(), p.S.to_numpy(), p.D.to_numpy()):
            t = key.get((d, s))
            if t is not None and np.searchsorted(t, T[j] + w, "right") > np.searchsorted(t, T[j], "left"):
                c3[i] += 1; c3[j] += 1
        F[f"cyc3_{h}h"] = c3
        # stack / layered: distinct sources feeding distinct destinations via the same set of mids
        F[f"layer_mids_{h}h"] = pd.Series(p.groupby(["S", "day"]).M.transform("nunique").to_numpy()).groupby(p.i.to_numpy()).max().reindex(range(n)).fillna(0).to_numpy()
    # TCEE neighbourhood windows (w=8) in each account's own history (in + out)
    inc = pd.DataFrame({"a": np.r_[S, R], "i": np.r_[np.arange(n), np.arange(n)], "dir": np.r_[np.ones(n), np.zeros(n)],
                        "cp": np.r_[R, S], "T": np.r_[T, T], "lv": np.r_[lv, lv]}).sort_values(["a", "T", "i"])
    inc["cpidx"] = inc.groupby(["a", "cp"]).cumcount()  # 0 = first time this counterparty appears
    inc["dt"] = inc.groupby("a").T.diff()
    res = {}
    for a, g in inc.groupby("a", sort=False):
        ii = g.i.to_numpy(); d = g.dir.to_numpy(); l = g.lv.to_numpy(); t = g["T"].to_numpy(); ci = g.cpidx.to_numpy(); m = len(g)
        cl = np.r_[0, np.cumsum(l)]; cd = np.r_[0, np.cumsum(d)]; cn = np.r_[0, np.cumsum(ci == 0)]
        k = np.arange(m); lo = np.maximum(k - 8, 0); hi = np.minimum(k + 9, m)
        nb_pre = k - lo; nb_post = hi - k - 1
        pre_mean = np.where(nb_pre > 0, (cl[k] - cl[lo]) / np.maximum(nb_pre, 1), l)
        post_mean = np.where(nb_post > 0, (cl[hi] - cl[k + 1]) / np.maximum(nb_post, 1), l)
        out = dict(rel_pre=l - pre_mean, rel_post=l - post_mean,
                   dir_pre=np.where(nb_pre > 0, (cd[k] - cd[lo]) / np.maximum(nb_pre, 1), 0.5),
                   dir_post=np.where(nb_post > 0, (cd[hi] - cd[k + 1]) / np.maximum(nb_post, 1), 0.5),
                   new_pre=np.where(nb_pre > 0, (cn[k] - cn[lo]) / np.maximum(nb_pre, 1), 1),
                   span_pre=np.log1p(t - t[lo]), span_post=np.log1p(t[hi - 1] - t),
                   cpidx=np.log1p(ci), pos=k / max(m - 1, 1))
        # share of next-8 opposite-direction txs with amount within 5% (forwarded)
        fw = np.zeros(m)
        for q in range(m):
            sl = slice(q + 1, hi[q]); fw[q] = ((d[sl] != d[q]) & (np.abs(l[sl] - l[q]) < 0.05)).mean() if hi[q] > q + 1 else 0
        out["fwd_share"] = fw
        res[a] = (ii, d, out)
    for role, dv in (("s", 1), ("r", 0)):
        cols = None
        for a, (ii, d, out) in res.items():
            m = d == dv
            if cols is None: cols = {k: np.zeros(n) for k in out}
            for k, v in out.items(): cols[k][ii[m]] = v[m]
        for k, v in cols.items(): F[f"nw_{role}_{k}"] = v
    # PNA-style pair pooling
    d = pd.DataFrame({"S": S, "R": R, "lv": lv, "T": T}).sort_values("T")
    d["iat"] = np.log1p(d.groupby(["S", "R"]).T.diff())
    g = d.groupby(["S", "R"])
    agg = pd.concat([g.lv.agg(["mean", "std", "max", "min", "size"]), g.iat.agg(["mean", "std"]).add_prefix("iat_")], axis=1)
    sc = np.log1p(agg["size"])
    for c in ["mean", "std", "max", "min", "iat_mean", "iat_std"]: agg[c + "_sc"] = agg[c] * sc
    j = agg.reindex(pd.MultiIndex.from_arrays([S, R]))
    for c in agg.columns: F[f"pair_pna_{c}"] = j[c].to_numpy()
    F["pair_pna_dev"] = lv - j["mean"].to_numpy()
    f = pd.DataFrame(F); f.insert(0, "TRANSACTION_HASH", tx.TRANSACTION_HASH.values)
    return f


if __name__ == "__main__":
    import time; t0 = time.time()
    tx = load_split(sys.argv[1]); f = build(tx); f.to_parquet(sys.argv[2]); print(f.shape, time.time() - t0)
