"""GARG-AML-style smurfing features (Deprez et al., arXiv 2506.04292): for each account v, order nodes as
[v | N1 (1-hop) | N2 (2-hop)] and measure block densities of the adjacency matrix. Smurfing (v -> many mules ->
few collectors) gives a dense N1->N2 block and a sparse N1 block. Directed (out/in) and undirected variants,
whole-week graph plus calendar windows (6h / 24h, two offsets); mapped to sender and receiver of each tx."""
import sys, time, numpy as np, pandas as pd
from common import *

def garg_block(A):
    """A: dense binary (n x n) directed adjacency, no self loops. Returns per-node feature matrix."""
    n = A.shape[0]; F = np.zeros((n, 6))
    for v in range(n):
        n1 = np.flatnonzero(A[v]); 
        if len(n1) == 0: continue
        reach = A[n1].any(0); reach[n1] = False; reach[v] = False; n2 = np.flatnonzero(reach)
        k1, k2 = len(n1), len(n2)
        d11 = A[np.ix_(n1, n1)].sum() / max(k1 * (k1 - 1), 1)
        d12 = A[np.ix_(n1, n2)].sum() / max(k1 * k2, 1) if k2 else 0.0
        # share of N1 nodes that forward into N2 (mule participation)
        part = A[np.ix_(n1, n2)].any(1).mean() if k2 else 0.0
        F[v] = [k1, k2, d11, d12, d12 - d11, part]
    return F

NAMES = ["k1", "k2", "d11", "d12", "score", "part"]

def account_feats(S, R, idx):
    """S,R arrays of account ids (int) for edges in idx; return dict variant -> (n_acc x 6)."""
    n = idx_n
    A = np.zeros((n, n), bool); A[S, R] = True; np.fill_diagonal(A, False)
    U = A | A.T
    return {"out": garg_block(A), "in": garg_block(A.T.copy()), "und": garg_block(U)}

def build(split_dir, hashes):
    global idx_n
    m = meta_of(split_dir, hashes)
    acc = pd.Index(pd.unique(np.concatenate([m.ACCOUNT, m.DEST])))
    S = acc.get_indexer(m.ACCOUNT); R = acc.get_indexer(m.DEST); T = m.TIMESTAMP.to_numpy()
    T = T // 10**9 if T.max() > 1e12 else T
    idx_n = len(acc); out = {}
    def put(tag, rows, V):
        for var, F in V.items():
            for j, nm in enumerate(NAMES):
                for role, col in (("s", S), ("r", R)):
                    key = f"garg_{tag}_{var}_{nm}_{role}"
                    if key not in out: out[key] = np.zeros(len(m))
                    out[key][rows] = F[col[rows], j]
    put("all", np.arange(len(m)), account_feats(S, R, None))
    for hours in (6, 24):
        for off in (0, 0.5):
            w = hours * 3600; b = ((T + off * w) // w).astype(np.int64)
            agg = {}
            for bb in np.unique(b):
                rows = np.flatnonzero(b == bb)
                V = account_feats(S[rows], R[rows], None); put(f"w{hours}o{int(off*2)}", rows, V)
            # merge two offsets later via max
    df = pd.DataFrame(out)
    for hours in (6, 24):
        for var in ("out", "in", "und"):
            for nm in NAMES:
                for role in ("s", "r"):
                    a, c = f"garg_w{hours}o0_{var}_{nm}_{role}", f"garg_w{hours}o1_{var}_{nm}_{role}"
                    df[f"garg_w{hours}_{var}_{nm}_{role}"] = np.maximum(df[a], df[c])
                    df = df.drop(columns=[a, c])
    df.insert(0, "TRANSACTION_HASH", m.TRANSACTION_HASH.to_numpy())
    return df

if __name__ == "__main__":
    t0 = time.time()
    tr, te, y, folds, cols = train_setup()
    print("setup", f"{time.time()-t0:.0f}s", flush=True)
    import os
    for split, d, h in (("train", "data/train", tr.TRANSACTION_HASH), ("test", "data/test", te.TRANSACTION_HASH)):
        f = f"{MEM}/garg_feats_{split}.parquet"
        if not os.path.exists(f): build(f"{ROOT}/{d}", h).to_parquet(f)
        print("built", split, f"{time.time()-t0:.0f}s", flush=True)
    gtr = pd.read_parquet(f"{MEM}/garg_feats_train.parquet").set_index("TRANSACTION_HASH").loc[tr.TRANSACTION_HASH].reset_index(drop=True)
    gte = pd.read_parquet(f"{MEM}/garg_feats_test.parquet").set_index("TRANSACTION_HASH").loc[te.TRANSACTION_HASH].reset_index(drop=True)
    # whole-week graph-level stats shift between weeks: percentile-rank within split (as v3 does)
    for c in gtr.columns:
        if c.startswith("garg_all"): gtr[c] = gtr[c].rank(pct=True); gte[c] = gte[c].rank(pct=True)
    gcols = list(gtr.columns)
    Xtr = pd.concat([tr[cols].reset_index(drop=True), gtr], axis=1); Xte = pd.concat([te[cols].reset_index(drop=True), gte], axis=1)
    res = {}
    base_o, _ = (None, None)
    for name, Xa, Xb in (("garg_only", gtr, gte), ("garg_v3feat_lgb", Xtr, Xte)):
        o, p = cv(Xa, y, folds, "lgb", SEED, Xb)
        save(name, tr.TRANSACTION_HASH, o, te.TRANSACTION_HASH, p)
        res[name] = blend_report(tr, y, o, name); print(f"{time.time()-t0:.0f}s", flush=True)
    import json; json.dump(res, open(f"{MEM}/garg_results.json", "w"), indent=1)
