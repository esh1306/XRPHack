"""v12: post-processing variants of the v9c_w60 cluster-score propagation step (builds on v9c_w60, no retrain).
Base (identical to v9c_w60): s = rank(0.4*rank(ours v7x no-head) + 0.6*rank(Sahid's earlier prototype stage-1)).
Cluster row sets use the exact closed first-time-payment cluster definition from Sahid's earlier prototype
(prior_work/prototype_agent/app/cluster_prop.py, MIT, Copyright (c) 2026 Sahid); they are computed once per window
and cached, then each variant only re-aggregates scores.
Usage:  python src/v12_prop_variants.py clusters SPLIT WIN_SEC     (cache clusters; SPLIT in test/train)
        python src/v12_prop_variants.py write                      (write all variant CSVs)
        python src/v12_prop_variants.py cv                         (fold-42 OOF AP of each variant on train)
Deterministic: no randomness (seed fixed anyway)."""
import os, sys, pickle, numpy as np, pandas as pd
from scipy.stats import rankdata
np.random.seed(42)
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "prior_work", "prototype_agent"))
from app.cluster_prop import _novel_incidences, _win_rows
rk = lambda x: rankdata(x) / len(x)
EDGES = {"test": f"{ROOT}/prototype/cache/edges_test.parquet", "train": f"{ROOT}/prototype/cache/edges_train.parquet"}
WNAME = {300: "5m", 900: "15m", 1800: "30m", 3600: "1h", 7200: "2h", 14400: "4h", 28800: "8h", 86400: "24h"}

def clusters(e, w_sec):
    """Same row-set logic as the prototype's propagate_cluster_scores, for one window."""
    e = e.reset_index(drop=True); novel, inc = _novel_incidences(e)
    t, S, R = e.t.to_numpy(), e.sender.to_numpy(), e.receiver.to_numpy(); out = []
    for i in range(len(e)):
        lo, hi = t[i] - w_sec, t[i] + w_sec
        so, sr = _win_rows(inc, S[i], lo, hi); ro, rr = _win_rows(inc, R[i], lo, hi)
        nodes = set(so) | set(ro) | {S[i], R[i]}
        rows = set(sr) | set(rr) | ({i} if novel[i] else set())
        for a in sorted(nodes - {S[i], R[i]})[:30]:
            oo, orows = _win_rows(inc, a, lo, hi)
            for o, r_ in zip(oo, orows):
                if o in nodes: rows.add(r_)
                elif w_sec <= 3600 and len(nodes) <= 15:
                    oo2, _ = _win_rows(inc, o, lo, hi)
                    if any(o2 in nodes for o2 in oo2): rows.add(r_)
        out.append(np.array(sorted(rows), dtype=np.int32))
    return out

def cpath(split, w): return f"{ROOT}/cache/v12_clusters_{split}_{WNAME[w]}.pkl"
def load_cl(split, w): return pickle.load(open(cpath(split, w), "rb"))

def agg(s, cl):
    n = len(s); mean, mx, q75 = s.copy(), s.copy(), s.copy()
    for i, r in enumerate(cl):
        if len(r):
            sc = s[r]; mean[i] = sc.mean(); mx[i] = sc.max(); q75[i] = np.percentile(sc, 75)
    return {"mean": mean, "max": mx, "q75": q75}

# (column, weight) lists; "s1" = own score. PROTO = the prototype's exact PROP_COLS / PROP_WEIGHTS.
PROTO = [("s1", .22), ("mean_5m", .06), ("max_5m", .10), ("mean_15m", .06), ("max_15m", .04), ("q75_30m", .06),
         ("max_1h", .14), ("q75_2h", .12), ("max_2h", .08), ("q75_4h", .12)]
VARIANTS = {
    # name: (column weights, own-rank weight in final mix, n passes)
    "ref_w60":  (PROTO, 0.35, 1),   # must reproduce v9c_w60 ranking (sanity check, not written)
    "own25":    (PROTO, 0.25, 1),
    "own45":    (PROTO, 0.45, 1),
    "iter2":    (PROTO, 0.35, 2),
    "wide8h":   (PROTO + [("q75_8h", .08), ("max_8h", .04)], 0.35, 1),
    "no5m":     ([c for c in PROTO if not c[0].endswith("_5m")] + [("mean_15m", .06), ("max_15m", .10)], 0.35, 1),
    "maxheavy": ([("s1", .22), ("max_5m", .16), ("max_15m", .10), ("max_30m", .06), ("max_1h", .14),
                  ("max_2h", .20), ("max_4h", .12)], 0.35, 1),
    "meanheavy": ([("s1", .22), ("mean_5m", .16), ("mean_15m", .10), ("q75_30m", .06), ("mean_1h", .14),
                   ("q75_2h", .12), ("mean_2h", .08), ("q75_4h", .12)], 0.35, 1),
}
NEED = {300, 900, 1800, 3600, 7200, 14400, 28800}

def propagate(s, cls, cols, own):
    A = {}
    W = {int(k): v for k, v in cls.items()}
    for w in sorted(W):
        a = agg(s, W[w])
        for k in a: A[f"{k}_{WNAME[w]}"] = a[k]
    A["s1"] = s
    tot = {}
    for c, wt in cols: tot[c] = tot.get(c, 0) + wt
    prop = sum(A[c] * wt for c, wt in tot.items()) / sum(tot.values())
    return own * rk(s) + (1 - own) * rk(prop)

def run_variant(s, cls, name):
    cols, own, npass = VARIANTS[name]; p = s
    for _ in range(npass): p = propagate(rk(p) if _ else p, cls, cols, own)
    return p

def base_scores(split):
    e = pd.read_parquet(EDGES[split])
    if split == "test":
        ft = np.load(f"{ROOT}/prototype/cache/test_stage1.npy")
        ut = e[["hash"]].merge(pd.read_parquet(f"{ROOT}/cache/pred_test_v7x_base.parquet"), left_on="hash", right_on="TRANSACTION_HASH", how="left").score.to_numpy()
    else:
        ft = np.load(f"{ROOT}/prototype/cache/oof42_stage1.npy")
        ut = e[["hash"]].merge(pd.read_parquet(f"{ROOT}/cache/pred_test_v7x_oof.parquet"), left_on="hash", right_on="TRANSACTION_HASH", how="left").oof.to_numpy()
    assert not np.isnan(ut).any() and len(ft) == len(e)
    return e, rk(0.4 * rk(ut) + 0.6 * rk(ft))

if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "clusters":
        split, w = sys.argv[2], int(sys.argv[3])
        cl = clusters(pd.read_parquet(EDGES[split]), w); pickle.dump(cl, open(cpath(split, w), "wb")); print("cached", split, w, len(cl))
    elif mode in ("write", "cv"):
        split = "test" if mode == "write" else "train"
        e, s = base_scores(split)
        cls = {w: load_cl(split, w) for w in sorted(NEED) if os.path.exists(cpath(split, w))}
        if mode == "cv":
            from sklearn.metrics import average_precision_score as AP
            L = pd.read_csv(f"{ROOT}/data/train/labels.csv").set_index("TRANSACTION_HASH").loc[e.hash]
            y, d = L.is_suspicious.to_numpy(), L.difficulty.to_numpy()
            for n in VARIANTS:
                if n == "wide8h" and 28800 not in cls: continue
                p = run_variant(s, cls, n); r = [100 * AP(y, p)] + [100 * AP(y[(d == k) | (y == 0)], p[(d == k) | (y == 0)]) for k in ("easy", "medium", "hard")]
                print("CV", n, np.round(r, 2).tolist(), flush=True)
        else:
            ref = pd.read_csv(f"{ROOT}/outputs/submission_v9c_w60.csv").set_index("TRANSACTION_HASH").score.loc[e.hash].to_numpy()
            from scipy.stats import spearmanr
            for n in VARIANTS:
                if n == "wide8h" and 28800 not in cls: continue
                p = rk(run_variant(s, cls, n)); rho = spearmanr(p, ref).correlation
                top_ref = set(np.argsort(-ref)[:401]); top = set(np.argsort(-p)[:401])
                print(n, "spearman_vs_w60", round(rho, 5), "top401_changed", len(top - top_ref), flush=True)
                if n == "ref_w60": continue
                out = f"{ROOT}/outputs/submission_v12_{n}.csv"
                assert not os.path.exists(out), out
                pd.DataFrame({"TRANSACTION_HASH": e.hash.values, "score": np.round(p, 8)}).to_csv(out, index=False); print("wrote", out)
