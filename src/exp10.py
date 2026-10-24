"""v6b ablation: stage-1 LGB seed 42 on v6 features + fe9 groups (H1..H4, KS<0.1 only) and/or the 8 chosen
fe8-episode columns (E8, from the other worker's cache/fe8, read-only). Fold seeds 42/7/2026."""
import sys, os, time, json, numpy as np, pandas as pd
from scipy.stats import ks_2samp
sys.path.insert(0, "src")
from pipeline import *
from pipeline import F8_SHIFT_DROP
E8 = ['g2_am15m_pair_days','g2_am15m_recur_days','g2_am1h_recur_days','g2_tw15m_novel_pair','g2_am1h_dwell_med','g2_am1h_ccy_switch','g2_am1h_acc_neps_mean','g2_tw15m_tx_per_acc']
lab = pd.read_csv("data/train/labels.csv")
base = load_feats("cache/train").merge(lab, on="TRANSACTION_HASH")
f8 = pd.read_parquet("cache/train_f8.parquet"); f9 = pd.read_parquet("cache/train_f9.parquet"); f9t = pd.read_parquet("cache/test_f9.parquet")
e8 = pd.read_parquet("cache/fe8/train_fe8.parquet")[["TRANSACTION_HASH"] + E8]
tr = base.merge(f8, on="TRANSACTION_HASH").merge(f9, on="TRANSACTION_HASH").merge(e8, on="TRANSACTION_HASH", how="left")
y = tr.is_suspicious.to_numpy(); groups = tr.scenario_id.fillna(tr.TRANSACTION_HASH).to_numpy()
bcols = [c for c in base.columns if c not in DROP and c not in lab.columns]
v6c = bcols + [c for c in f8.columns if c != "TRANSACTION_HASH" and c[0] in "ABCDG" and c not in F8_SHIFT_DROP]
H = [c for c in f9.columns if c != "TRANSACTION_HASH" and ks_2samp(f9[c], f9t[c]).statistic < 0.1]
print("H kept", len(H), "dropped", [c for c in f9.columns[1:] if c not in H], flush=True)
def cols(k):
    c = list(v6c)
    for g in k.split("+")[1:]:
        c += E8 if g == "E8" else [h for h in H if (g == "H" or h.startswith(g + "_"))]
    return c
def rep(o):
    r = [round(100 * AP(y, o), 2)]
    for d in ["easy", "medium", "hard"]:
        m = (tr.difficulty == d) | (y == 0); r.append(round(100 * AP(y[m], o[m]), 2))
    return r
from joblib import Parallel, delayed
def run(k, fs):
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=fs).split(tr, y, groups))
    t0 = time.time(); o, _ = cv(tr[cols(k)], y, folds, "lgb", 42); np.save(f"cache/exp10_{k}_{fs}.npy", o)
    return k, fs, len(cols(k)), rep(o), round(time.time() - t0)
jobs = [(k, fs) for k in sys.argv[1].split(",") for fs in (42, 7, 2026)]
res = Parallel(n_jobs=int(os.environ.get("PAR", 8)))(delayed(run)(*j) for j in jobs)
out = {}
for k, fs, n, r, t in res:
    print(k, fs, n, r, t, flush=True); out.setdefault(k, []).append(r)
b = np.array(out.get("v6", [[0]*4]))
for k, v in out.items():
    v = np.array(v); d = v[:, 0] - b[:, 0]
    print(f"{k:14s} mean {v.mean(0).round(2).tolist()} sd {v[:,0].std(ddof=1):.2f} | gain vs v6 {d.mean():+.2f} sd(diff) {d.std(ddof=1):.2f}", flush=True)
json.dump(out, open("logs/exp10.json", "w"))
