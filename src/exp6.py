"""v4: level-3 stacker over v3 OOF + medium/hard specialist + component-smoothed scores (grouped CV)."""
import sys, numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, "src")
from pipeline import rank01, AP
from sklearn.model_selection import StratifiedGroupKFold
lab = pd.read_csv("data/train/labels.csv")
o = pd.read_parquet("cache/oof_final.parquet").merge(lab, on="TRANSACTION_HASH")
y = o.is_suspicious.to_numpy(); diff = o.difficulty.to_numpy()
groups = o.scenario_id.fillna(o.TRANSACTION_HASH).to_numpy()
def rep(s):
    r = [round(100 * AP(y, s), 2)]
    for d in ["easy", "medium", "hard"]:
        m = (diff == d) | (y == 0); r.append(round(100 * AP(y[m], s[m]), 2))
    return r
def comp_feats(r0, comp):
    F = {"r0": r0}
    for c in ["cmp120", "cmp900", "flow480", "flow3600", "flow14400"]:
        g = pd.Series(r0).groupby(comp[c].to_numpy())
        F[c + "_mean"] = g.transform("mean").to_numpy(); F[c + "_max"] = g.transform("max").to_numpy()
        F[c + "_n"] = g.transform("size").to_numpy(); F[c + "_rk"] = g.rank(pct=True).to_numpy()
    return pd.DataFrame(F)
r0 = rank01(o.oof.to_numpy())
comp = pd.read_parquet("cache/train_comp.parquet").set_index("TRANSACTION_HASH").loc[o.TRANSACTION_HASH]
X = comp_feats(r0, comp)
hm = pd.read_parquet("cache/train_f4.parquet").set_index("TRANSACTION_HASH").loc[o.TRANSACTION_HASH].index  # order check
spec = (rank01(np.load("cache/spec_all_w3.npy")) + rank01(np.load("cache/spec_all_w3_s7.npy"))) / 2
print("base", rep(r0))
P = dict(objective="binary", learning_rate=0.03, num_leaves=7, min_data_in_leaf=100, feature_fraction=0.8,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=10, verbose=-1, n_jobs=1)
for name, Xs in [("comp", X), ("comp+spec", X.assign(spec=spec))]:
    for fs in (42, 7):
        out = np.zeros(len(y))
        for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=fs).split(Xs, y, groups):
            m = lgb.train({**P, "seed": fs}, lgb.Dataset(Xs.iloc[a], y[a]), 300)
            out[b] = m.predict(Xs.iloc[b])
        s = rank01(out)
        print(name, fs, "stack", rep(s), "avg w/ base", rep(0.5 * r0 + 0.5 * s), flush=True)
