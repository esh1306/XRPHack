import sys, numpy as np, pandas as pd
sys.path.insert(0, "src")
from pipeline import *
lab = pd.read_csv("data/train/labels.csv")
def lf(p, ks):
    a = pd.read_parquet(f"{p}_f.parquet")
    for k in ks: a = a.merge(pd.read_parquet(f"{p}_{k}.parquet"), on="TRANSACTION_HASH")
    return a
tr = lf("cache/train", ["f2", "f3", "f4"]).merge(lab, on="TRANSACTION_HASH")
y = tr.is_suspicious.to_numpy(); groups = tr.scenario_id.fillna(tr.TRANSACTION_HASH).to_numpy()
f4c = [c for c in pd.read_parquet("cache/train_f4.parquet").columns if c != "TRANSACTION_HASH"]
base = [c for c in lf("cache/train", ["f2", "f3"]).columns if c not in DROP]
def rep(o):
    r = {"all": round(100 * AP(y, o), 2)}
    for d in ["easy", "medium", "hard"]:
        m = (tr.difficulty == d) | (y == 0); r[d] = round(100 * AP(y[m], o[m]), 2)
    return r
sets = {"base": base, "base+f4": base + f4c, "base+f4_noep": base + [c for c in f4c if not c.startswith("ep_")]}
for fs in (42, 7):
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=fs).split(tr, y, groups))
    for k, cols in sets.items():
        o, _ = cv(tr[cols], y, folds, "lgb", 42); print(fs, k, len(cols), rep(o), flush=True)
        np.save(f"cache/exp3_{k}_{fs}.npy", o)
