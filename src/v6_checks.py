"""v6 checks: account overlap across grouped CV folds (scenario groups)."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "src")
from pipeline import meta_of, StratifiedGroupKFold
lab = pd.read_csv("data/train/labels.csv")
m = meta_of("data/train", lab.TRANSACTION_HASH).merge(lab, on="TRANSACTION_HASH")
y = m.is_suspicious.to_numpy(); g = m.scenario_id.fillna(m.TRANSACTION_HASH).to_numpy()
print("scenario ids:", m.scenario_id.nunique(), "rows with scenario:", m.scenario_id.notna().mean().round(3), "positives w/ scenario:", m.scenario_id[y == 1].notna().mean().round(3))
for fs in (42, 7, 2026):
    for k, (a, b) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=fs).split(m, y, g)):
        if k: continue
        ta = set(m.ACCOUNT.iloc[a]) | set(m.DEST.iloc[a]); vb = m.iloc[b]; pv = vb[vb.is_suspicious == 1]
        acc_v = set(vb.ACCOUNT) | set(vb.DEST); acc_p = set(pv.ACCOUNT) | set(pv.DEST)
        sc_overlap = len(set(m.scenario_id.iloc[a].dropna()) & set(vb.scenario_id.dropna()))
        print(f"seed {fs} fold0: scenario overlap {sc_overlap}; val accounts also in train folds {len(acc_v & ta)}/{len(acc_v)}; positive-tx accounts also in train folds {len(acc_p & ta)}/{len(acc_p)}")
