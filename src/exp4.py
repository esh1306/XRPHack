import sys, numpy as np, pandas as pd, time
sys.path.insert(0, "src")
from pipeline import *
lab = pd.read_csv("data/train/labels.csv")
def lf(p, ks):
    a = pd.read_parquet(f"{p}_f.parquet")
    for k in ks: a = a.merge(pd.read_parquet(f"{p}_{k}.parquet"), on="TRANSACTION_HASH")
    return a
tr = lf("cache/train", ["f2", "f3", "f4", "f5", "f6", "f7"]).merge(lab, on="TRANSACTION_HASH")
y = tr.is_suspicious.to_numpy(); groups = tr.scenario_id.fillna(tr.TRANSACTION_HASH).to_numpy()
C = {k: [c for c in pd.read_parquet(f"cache/train_{k}.parquet").columns if c != "TRANSACTION_HASH"] for k in ("f", "f2", "f3", "f4", "f5", "f6", "f7")}
def rep(o):
    r = {"all": round(100 * AP(y, o), 2)}
    for d in ["easy", "medium", "hard"]:
        m = (tr.difficulty == d) | (y == 0); r[d] = round(100 * AP(y[m], o[m]), 2)
    return r
base = [c for c in C["f"] + C["f2"] + C["f3"] if c not in DROP]
sets = {s: base + sum((C[k] for k in s.split("+")), []) for s in sys.argv[1].split(",")}
fs = int(sys.argv[2]) if len(sys.argv) > 2 else 42
folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=fs).split(tr, y, groups))
from joblib import Parallel, delayed
def run(k, cols):
    t0 = time.time(); o, _ = cv(tr[cols], y, folds, "lgb", 42); np.save(f"cache/exp4_{k}_{fs}.npy", o)
    return k, len(cols), rep(o), round(time.time() - t0)
for r in Parallel(n_jobs=len(sets))(delayed(run)(k, c) for k, c in sets.items()): print(fs, *r, flush=True)
