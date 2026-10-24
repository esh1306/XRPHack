"""v6 ablation: stage-1 LightGBM (seed 42) on v3 features + each fe8 group; 3 fold seeds. Usage: exp8.py CONFIGS (e.g. base,A,B,C,D,G,ABCDG)"""
import sys, os, time, json, numpy as np, pandas as pd
sys.path.insert(0, "src")
from pipeline import *
from pipeline import load_f8
lab = pd.read_csv("data/train/labels.csv")
base = load_feats("cache/train").merge(lab, on="TRANSACTION_HASH")
f8 = pd.read_parquet("cache/train_f8.parquet")
tr = base.merge(f8, on="TRANSACTION_HASH")
y = tr.is_suspicious.to_numpy(); groups = tr.scenario_id.fillna(tr.TRANSACTION_HASH).to_numpy()
bcols = [c for c in base.columns if c not in DROP and c not in lab.columns]
from pipeline import F8_SHIFT_DROP
def rep(o):
    r = [round(100 * AP(y, o), 2)]
    for d in ["easy", "medium", "hard"]:
        m = (tr.difficulty == d) | (y == 0); r.append(round(100 * AP(y[m], o[m]), 2))
    return r
def cols_for(k):
    if k == "base": return bcols
    return bcols + [c for c in f8.columns if c != "TRANSACTION_HASH" and c[0] in k and c not in F8_SHIFT_DROP]
from joblib import Parallel, delayed
def run(k, fs):
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=fs).split(tr, y, groups))
    t0 = time.time(); o, _ = cv(tr[cols_for(k)], y, folds, "lgb", 42); np.save(f"cache/exp8_{k}_{fs}.npy", o)
    return k, fs, len(cols_for(k)), rep(o), round(time.time() - t0)
jobs = [(k, fs) for k in sys.argv[1].split(",") for fs in (42, 7, 2026)]
res = Parallel(n_jobs=int(os.environ.get("PAR", 7)))(delayed(run)(*j) for j in jobs)
out = {}
for k, fs, n, r, t in res:
    print(k, fs, n, r, t, flush=True); out.setdefault(k, []).append(r)
for k, v in out.items():
    v = np.array(v); print(f"{k:8s} mean {v.mean(0).round(2).tolist()} sd {v.std(0, ddof=1).round(2).tolist()}")
json.dump({k: v for k, v in out.items()}, open(f"logs/exp8_{sys.argv[1].replace(',', '_')}.json", "w"))
