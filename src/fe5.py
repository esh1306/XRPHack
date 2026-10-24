"""v3 features inspired by ExSTraQt (Tariq & Hassani 2026, arXiv:2604.02899) -- own implementation:
(a) time-respecting multi-hop flow ('collect' upstream into the sender before the tx, 'dispense' downstream from
the receiver after it, 1-3 hops via iterated window sums), per-account dispense/collect ratios;
(b) Louvain communities on the per-split account graph and on daily subgraphs. Label-free, per split."""
import sys, warnings, numpy as np, pandas as pd, networkx as nx
from collections import defaultdict
warnings.filterwarnings("ignore")
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from fe import load_split


def build(tx):
    S = tx.ACCOUNT.to_numpy(); R = tx.DEST.to_numpy(); T = tx.TIMESTAMP.to_numpy().astype(np.int64)
    V = np.nan_to_num(tx.usd.to_numpy().astype(float)); n = len(tx); F = {}
    out_i, in_i = defaultdict(list), defaultdict(list)
    for i in np.argsort(T, kind="stable"): out_i[S[i]].append(i); in_i[R[i]].append(i)
    out_i = {k: np.array(v) for k, v in out_i.items()}; in_i = {k: np.array(v) for k, v in in_i.items()}

    def wsum(idx_map, key, val, w, fwd):
        res = np.zeros(n); cnt = np.zeros(n)
        for a, ix in idx_map.items():  # windows evaluated for txs whose `key` account == a
            pass
        return res
    # precompute per account sorted times and cumsums of an arbitrary value vector
    def window(val, w, fwd):
        res = np.zeros(n)
        if fwd:  # sum of val over receiver's outflows in (T, T+w]
            for a, ix in out_i.items():
                q = in_i.get(a);
                if q is None: continue
                t = T[ix]; cs = np.r_[0, np.cumsum(val[ix])]
                lo = np.searchsorted(t, T[q], "right"); hi = np.searchsorted(t, T[q] + w, "right")
                res[q] = cs[hi] - cs[lo]
        else:  # sum of val over sender's inflows in [T-w, T)
            for a, ix in in_i.items():
                q = out_i.get(a)
                if q is None: continue
                t = T[ix]; cs = np.r_[0, np.cumsum(val[ix])]
                lo = np.searchsorted(t, T[q] - w, "left"); hi = np.searchsorted(t, T[q], "left")
                res[q] = cs[hi] - cs[lo]
        return res
    lV = np.log1p(V)
    for w in (3600, 21600, 86400):
        c1 = window(V, w, False); c2 = window(c1, w, False); c3 = window(c2, w, False)
        d1 = window(V, w, True); d2 = window(d1, w, True); d3 = window(d2, w, True)
        h = w // 3600
        F[f"collect1_{h}h"] = np.log1p(c1); F[f"collect2_{h}h"] = np.log1p(c2); F[f"collect3_{h}h"] = np.log1p(c3)
        F[f"dispense1_{h}h"] = np.log1p(d1); F[f"dispense2_{h}h"] = np.log1p(d2); F[f"dispense3_{h}h"] = np.log1p(d3)
        F[f"collect_ratio_{h}h"] = np.log1p(c1) - lV; F[f"dispense_ratio_{h}h"] = np.log1p(d1) - lV
        F[f"thru_{h}h"] = np.minimum(np.log1p(c1), np.log1p(d1)) - lV
    # per-account dispense/collect: share of outflow preceded by inflow within 6h
    cov = window(V, 21600, False)
    a = pd.DataFrame({"S": S, "v": V, "cov": np.minimum(cov, V)}).groupby("S").sum()
    acc_cov = (a["cov"] / (a.v + 1e-9))
    tin = pd.Series(V).groupby(R).sum(); tout = pd.Series(V).groupby(S).sum()
    for role, col in (("s", S), ("r", R)):
        F[f"{role}_out_covered"] = acc_cov.reindex(col).fillna(0).to_numpy()
        F[f"{role}_disp_coll"] = np.log1p(tout.reindex(col).fillna(0).to_numpy()) - np.log1p(tin.reindex(col).fillna(0).to_numpy())
    # (b) communities
    def comm_feats(mask, tag, res):
        G = nx.Graph()
        d = pd.DataFrame({"S": S[mask], "R": R[mask], "V": V[mask]}); d = d[d.S != d.R]
        e = d.groupby(["S", "R"]).V.agg(["size", "sum"]).reset_index()
        for s, r, c, v in e.itertuples(index=False):
            if G.has_edge(s, r): G[s][r]["weight"] += c
            else: G.add_edge(s, r, weight=c)
        cm = {}
        for k, c in enumerate(nx.community.louvain_communities(G, weight="weight", resolution=res, seed=42)):
            for x in c: cm[x] = k
        cs = pd.Series(cm).reindex(S).to_numpy(); cr = pd.Series(cm).reindex(R).to_numpy()
        size = pd.Series(cm).value_counts()
        dd = pd.DataFrame({"cs": cs, "cr": cr, "V": V})[mask]
        intra = dd[dd.cs == dd.cr].groupby("cs").V.agg(["size", "sum"])
        outf = dd[dd.cs != dd.cr].groupby("cs").V.sum(); inf = dd[dd.cs != dd.cr].groupby("cr").V.sum()
        allf = dd.groupby("cs").V.sum().add(dd.groupby("cr").V.sum(), fill_value=0)
        sz = size.reindex(cs).to_numpy().astype(float)
        F[f"{tag}_intra"] = np.where(mask, (cs == cr).astype(float), np.nan)
        F[f"{tag}_s_size"] = np.where(mask, sz, np.nan)
        F[f"{tag}_r_size"] = np.where(mask, size.reindex(cr).to_numpy().astype(float), np.nan)
        F[f"{tag}_dens"] = np.where(mask, intra["size"].reindex(cs).fillna(0).to_numpy() / np.maximum(sz * (sz - 1), 1), np.nan)
        F[f"{tag}_int_share"] = np.where(mask, intra["sum"].reindex(cs).fillna(0).to_numpy() / (allf.reindex(cs).fillna(0).to_numpy() + 1e-9), np.nan)
        F[f"{tag}_io_bal"] = np.where(mask, (np.log1p(inf.reindex(cs).fillna(0)) - np.log1p(outf.reindex(cs).fillna(0))).to_numpy(), np.nan)
        F[f"{tag}_n_comm"] = np.where(mask, len(size), np.nan)
    all_m = np.ones(n, bool)
    comm_feats(all_m, "lv1", 1.0); comm_feats(all_m, "lv3", 3.0)
    day = (T - T.min()) // 86400
    parts = {}
    for dval in np.unique(day):
        m = day == dval
        comm_feats(m, "lvd", 1.0); parts[dval] = {k: F[k][m] for k in list(F) if k.startswith("lvd_")}
    for k in [k for k in F if k.startswith("lvd_")]:
        arr = np.full(n, np.nan)
        for dval, p in parts.items(): arr[day == dval] = p[k]
        F[k] = arr
    f = pd.DataFrame(F); f.insert(0, "TRANSACTION_HASH", tx.TRANSACTION_HASH.values)
    return f


if __name__ == "__main__":
    import time; t0 = time.time()
    tx = load_split(sys.argv[1]); f = build(tx); f.to_parquet(sys.argv[2]); print(f.shape, time.time() - t0)
