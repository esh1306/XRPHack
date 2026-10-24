"""v6: full-pipeline OOF (per fold seed) +/- look-alike head blend: final = rank(base) + A * rank(head)."""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, "src"); from pipeline import AP, rank01
lab = pd.read_csv("data/train/labels.csv"); import os; h = np.load(os.environ.get("LA_NPZ", "cache/lookalike.npz"), allow_pickle=True)
H = pd.Series(range(len(h["train_hash"])), index=h["train_hash"])
FILES = {"v3": {42: "cache/oof_final_v3.parquet", 7: "cache/oof_v3_fs7.parquet", 2026: "cache/oof_v3_fs2026.parquet"},
         "v6": {42: "cache/oof_final_v6.parquet", 7: "cache/oof_v6_fs7.parquet", 2026: "cache/oof_v6_fs2026.parquet"}}
res = {}
for name, fm in FILES.items():
    for fs, f in fm.items():
        if not os.path.exists(f): continue
        o = pd.read_parquet(f).merge(lab, on="TRANSACTION_HASH"); y = o.is_suspicious.to_numpy()
        hd = rank01(h[f"oof_{fs}"][H.loc[o.TRANSACTION_HASH].to_numpy()]); b = rank01(o.oof.to_numpy())
        for a in (0, 0.1, 0.2, 0.3):
            s = b + a * hd; r = [100 * AP(y, s)]
            for d in ("easy", "medium", "hard"):
                m = (o.difficulty == d).to_numpy() | (y == 0); r.append(100 * AP(y[m], s[m]))
            res.setdefault((name, a), {})[fs] = r
for k, v in res.items():
    a = np.array(list(v.values()))
    print(k, "seeds", list(v), "mean", a.mean(0).round(2).tolist(), "sd", a[:, 0].std(ddof=1).round(2) if len(a) > 1 else "-", "per-seed all", a[:, 0].round(2).tolist())
