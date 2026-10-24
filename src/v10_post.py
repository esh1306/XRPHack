"""v10 post-processing (also used for v9c): ours (env OURS_OOF / OURS_TEST) blended with prototype stage1, prototype cluster propagation, optional our-top-K block.
v9b-style score prop(w*rank(ours) + (1-w)*rank(prototype stage1)) using the prototype's propagate_cluster_scores (prior_work/prototype_agent, MIT).
Run from prototype/:  ../.venv/bin/python ../src/v9c_bestofboth.py [write]"""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, "../prior_work/prototype_agent")
from app.cluster_prop import propagate_cluster_scores as P
from sklearn.metrics import average_precision_score as AP
from scipy.stats import rankdata
rk = lambda x: rankdata(x) / len(x)
e = pd.read_parquet("cache/edges_train.parquet"); et = pd.read_parquet("cache/edges_test.parquet")
L = pd.read_csv("../data/train/labels.csv").set_index("TRANSACTION_HASH").loc[e.hash]; y = L.is_suspicious.to_numpy(); d = L.difficulty.to_numpy()
def rep(sc):
    r = [100 * AP(y, sc)]
    for k in ("easy", "medium", "hard"):
        m = (d == k) | (y == 0); r.append(100 * AP(y[m], sc[m]))
    return np.round(r, 2).tolist()
fo, ft = np.load("cache/oof42_stage1.npy"), np.load("cache/test_stage1.npy")
us = e[["hash"]].merge(pd.read_parquet(os.environ.get("OURS_OOF", "../cache/pred_test_v7x_oof.parquet")), left_on="hash", right_on="TRANSACTION_HASH", how="left").oof.to_numpy()
ut = et[["hash"]].merge((pd.read_parquet(os.environ["OURS_TEST"]) if "OURS_TEST" in os.environ else pd.read_parquet("../cache/pred_test_v7x_base.parquet")), left_on="hash", right_on="TRANSACTION_HASH", how="left").score.to_numpy()
def topk(a, b, K):
    if K == 0: return b
    ra = rk(a); top = ra > 1 - K; s = rk(b).copy(); s[top] = 2 + ra[top]; return s
res = []
FX = tuple(map(float, os.environ['FIXED'].split(','))) if os.environ.get('FIXED') else None
for w in ((FX[0],) if FX else (0.6, 0.7, 0.8, 1.0)):
    B = P(e, rk(w * rk(us) + (1 - w) * rk(fo)))
    for K in ((FX[1],) if FX else (0, 0.01, 0.015, 0.02)):
        for topsrc in ("ours",):
            r = rep(topk(us, B, K)); res.append((w, K, r)); print("w", w, "K", K, r, flush=True)
best = max(res, key=lambda t: t[2][0]); print("BEST", best, flush=True)
if len(sys.argv) > 1 and sys.argv[1] == "write":
    w, K, _ = best
    if os.environ.get("FIXED"): w, K = map(float, os.environ["FIXED"].split(","))  # e.g. FIXED=0.7,0 (v9b-style)
    Bt = P(et, rk(w * rk(ut) + (1 - w) * rk(ft))); s = topk(ut, Bt, K); s = rk(s)
    pd.DataFrame({"TRANSACTION_HASH": et.hash.values, "score": np.round(s, 8)}).assign(reason=lambda d: np.where(d.score > 0.99, "high risk: top-1% after cluster-score propagation", "low risk")).to_csv(os.environ.get("OUT", "../cache/submission_v9c_bestofboth.csv"), index=False)
    print("wrote", os.environ.get("OUT", "../cache/submission_v9c_bestofboth.csv"), w, K)
