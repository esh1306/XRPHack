"""v6 look-alike head: LightGBM trained only on positives vs legit look-alike-scenario rows (scenario_id set, label 0),
so it learns what separates real laundering from its decoys. OOF over StratifiedGroupKFold(scenario) for fold seeds
given; test prediction from a full refit (mean best-iter). Usage: lookalike.py OUT_NPZ [fold seeds...] [TEST_PREFIX]"""
import sys, os, numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from pipeline import *
from pipeline import load_f8
lab = pd.read_csv("data/train/labels.csv")
G = ("A", "B", "C", "D")
def _lf(p):  # v3 features (+ fe8 if F8_GROUPS is set) plus every fe8 group A-D, without duplicate columns
    a = load_feats(p); b = load_f8(p, G)
    return a.merge(b[["TRANSACTION_HASH"] + [c for c in b.columns if c not in a.columns]], on="TRANSACTION_HASH")
tr = _lf("cache/train").merge(lab, on="TRANSACTION_HASH")
te = _lf(sys.argv[3] if len(sys.argv) > 3 and not sys.argv[3].isdigit() else "cache/test")
cols = [c for c in te.columns if c not in DROP]
y = tr.is_suspicious.to_numpy(); groups = tr.scenario_id.fillna(tr.TRANSACTION_HASH).to_numpy()
look = (tr.scenario_id.notna() & (y == 0)).to_numpy(); sub = look | (y == 1)
P = dict(objective="binary", metric="average_precision", learning_rate=0.03, num_leaves=15, min_child_samples=40,
         feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1, n_jobs=int(os.environ.get("NJOBS", 1)))
out = {}; its = []
for fs in [int(s) for s in sys.argv[2:] if s.isdigit()] or [42]:
    oof = np.zeros(len(y))
    for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=fs).split(tr, y, groups):
        a = a[sub[a]]; bs = b[sub[b]]
        m = lgb.train({**P, "seed": 42}, lgb.Dataset(tr[cols].iloc[a], y[a]), 2000, valid_sets=[lgb.Dataset(tr[cols].iloc[bs], y[bs])],
                      callbacks=[lgb.early_stopping(150, verbose=False)])
        oof[b] = m.predict(tr[cols].iloc[b]); its.append(m.best_iteration)
    out[f"oof_{fs}"] = oof
    print(fs, "head AP on pos-vs-lookalike subset", round(100 * AP(y[sub], oof[sub]), 2), "all", round(100 * AP(y, oof), 2), flush=True)
m = lgb.train({**P, "seed": 42}, lgb.Dataset(tr[cols][sub], y[sub]), max(50, int(np.mean(its))))
out["test"] = m.predict(te[cols]); out["test_hash"] = te.TRANSACTION_HASH.to_numpy(); out["train_hash"] = tr.TRANSACTION_HASH.to_numpy()
np.savez(sys.argv[1], **out); print("saved", sys.argv[1])
