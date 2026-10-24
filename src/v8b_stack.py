"""v8b probe: medium/hard-weighted stacker = LightGBM on v6x features + rank(base OOF), all positives kept
(easy weight 0.3, medium/hard/ambiguous weight 3, legit look-alike negatives weight 3). Fold seed 42, base = v6x OOF.
Then base kept for top-K, stacker used below (and plain rank-average variants)."""
import sys, os, time, numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, "src")
from pipeline import load_feats, DROP, StratifiedGroupKFold, AP, rank01
t0 = time.time(); LOOK = {"bipartite","chain","cycle","fan_in","fan_out","fan_out_in","multi_stage","random","stack"}
P = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_child_samples=20, feature_fraction=0.6,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, n_jobs=6, seed=42)
lab = pd.read_csv("data/train/labels.csv"); tr = load_feats("cache/train").merge(lab, on="TRANSACTION_HASH")
y = tr.is_suspicious.to_numpy(); diff = tr.difficulty.to_numpy(); g = tr.scenario_id.fillna(tr.TRANSACTION_HASH).to_numpy()
te = load_feats("cache/test"); cols = [c for c in te.columns if c not in DROP]
base = pd.read_parquet("cache/oof_v6x.parquet").set_index("TRANSACTION_HASH").oof.loc[tr.TRANSACTION_HASH].to_numpy()
X = tr[cols].copy(); X["base_rank"] = rank01(base)
w = np.ones(len(y)); w[(y == 1) & (diff == "easy")] = 0.3; w[(y == 1) & (diff != "easy")] = 3.0
w[(y == 0) & tr.family.isin(LOOK).to_numpy()] = 3.0
def met(s):
    r = [100 * AP(y, s)] + [100 * AP(y[(diff == d) | (y == 0)], s[(diff == d) | (y == 0)]) for d in ("easy", "medium", "hard")]
    return np.round(r, 2).tolist()
folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(tr, y, g))
st = np.zeros(len(y))
for a, b in folds:
    st[b] = lgb.train(P, lgb.Dataset(X.iloc[a], y[a], weight=w[a]), 500).predict(X.iloc[b])
print("baseline", met(base)); print("stacker alone", met(st))
rb = rank01(base)
for K in (0.01, 0.02, 0.03):
    top = rb > 1 - K
    print("tier K", K, met(np.where(top, 2 + rb, rank01(st) * .999)))
for a in (0.2, 0.35, 0.5):
    print("rankavg", a, met((1 - a) * rb + a * rank01(st)))
print(f"{time.time()-t0:.0f}s")
