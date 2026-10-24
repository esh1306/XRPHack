"""v6: blend look-alike head into the top band of a base score. Grid evaluated over 3 fold seeds.
Usage: v6_blend.py BASE_CFG (uses cache/exp8_<cfg>_<fs>.npy and cache/lookalike.npz)"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "src")
from pipeline import AP, rank01
lab = pd.read_csv("data/train/labels.csv")
import os; h = np.load(os.environ.get("LA_NPZ", "cache/lookalike.npz"), allow_pickle=True)
order = pd.Series(range(len(lab)), index=lab.TRANSACTION_HASH)
L = lab.set_index("TRANSACTION_HASH").loc[h["train_hash"]].reset_index()
y = L.is_suspicious.to_numpy()
def rep(o):
    r = [round(100 * AP(y, o), 2)]
    for d in ["easy", "medium", "hard"]:
        m = (L.difficulty == d).to_numpy() | (y == 0); r.append(round(100 * AP(y[m], o[m]), 2))
    return r
def mix(rb, rh, a, q): return rb + a * rh * (rb >= q)
cfg = sys.argv[1]; res = {}
for fs in (42, 7, 2026):
    b = rank01(np.load(f"cache/exp8_{cfg}_{fs}.npy")); hr = rank01(h[f"oof_{fs}"])
    for a in (0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0):
        for q in (0.0, 0.9, 0.95, 0.98):
            res.setdefault((a, q), []).append(rep(mix(b, hr, a, q)))
rows = sorted(((k, np.mean(v, 0), np.std(np.array(v)[:, 0], ddof=1)) for k, v in res.items()), key=lambda x: -x[1][0])
base = np.array(res[(0, 0.0)]); print("base", cfg, base.mean(0).round(2), "sd", base[:, 0].std(ddof=1).round(2))
for k, m, s in rows[:8]:
    d = np.array(res[k])[:, 0] - base[:, 0]
    print(k, m.round(2), "sd", round(s, 2), "gain per seed", d.round(2), "gain mean/sd", round(d.mean(), 2), round(d.std(ddof=1), 2))
