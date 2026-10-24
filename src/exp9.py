"""v6 + other worker's episode features (cache/fe8/*_fe8.parquet, read-only): stage-1 LGB seed 42, fold seeds 42/7/2026.
EP = all episode cols; EPks = only cols with train/test KS < 0.05 (shift filter)."""
import sys, os, time, json, numpy as np, pandas as pd
from scipy.stats import ks_2samp
sys.path.insert(0, "src")
from pipeline import *
from pipeline import F8_SHIFT_DROP
lab = pd.read_csv("data/train/labels.csv")
base = load_feats("cache/train").merge(lab, on="TRANSACTION_HASH")
f8 = pd.read_parquet("cache/train_f8.parquet")
ep = pd.read_parquet("cache/fe8/train_fe8.parquet"); ept = pd.read_parquet("cache/fe8/test_fe8.parquet")
ep = ep.rename(columns={c: "EP_" + c for c in ep.columns if c != "TRANSACTION_HASH"})
ept = ept.rename(columns={c: "EP_" + c for c in ept.columns if c != "TRANSACTION_HASH"})
tr = base.merge(f8, on="TRANSACTION_HASH").merge(ep, on="TRANSACTION_HASH", how="left")
print("rows", len(tr), "ep missing", tr[ep.columns[1]].isna().mean(), flush=True)
y = tr.is_suspicious.to_numpy(); groups = tr.scenario_id.fillna(tr.TRANSACTION_HASH).to_numpy()
bcols = [c for c in base.columns if c not in DROP and c not in lab.columns]
v6c = bcols + [c for c in f8.columns if c != "TRANSACTION_HASH" and c[0] in "ABCDG" and c not in F8_SHIFT_DROP]
epc = [c for c in ep.columns if c != "TRANSACTION_HASH"]
ks = {c: ks_2samp(tr[c].dropna(), ept[c].dropna()).statistic for c in epc}
epks = [c for c in epc if ks[c] < 0.05]
print("ep cols", len(epc), "ks<0.05", len(epks), flush=True)
def rep(o):
    r = [round(100 * AP(y, o), 2)]
    for d in ["easy", "medium", "hard"]:
        m = (tr.difficulty == d) | (y == 0); r.append(round(100 * AP(y[m], o[m]), 2))
    return r
CF = {"v6": v6c, "v6EP": v6c + epc, "v6EPks": v6c + epks}
from joblib import Parallel, delayed
def run(k, fs):
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=fs).split(tr, y, groups))
    t0 = time.time(); o, _ = cv(tr[CF[k]], y, folds, "lgb", 42); np.save(f"cache/exp9_{k}_{fs}.npy", o)
    return k, fs, len(CF[k]), rep(o), round(time.time() - t0)
jobs = [(k, fs) for k in sys.argv[1].split(",") for fs in (42, 7, 2026)]
res = Parallel(n_jobs=int(os.environ.get("PAR", 3)))(delayed(run)(*j) for j in jobs)
out = {}
for k, fs, n, r, t in res:
    print(k, fs, n, r, t, flush=True); out.setdefault(k, []).append(r)
for k, v in out.items():
    v = np.array(v); print(f"{k:8s} mean {v.mean(0).round(2).tolist()} sd {v.std(0, ddof=1).round(2).tolist()}", flush=True)
