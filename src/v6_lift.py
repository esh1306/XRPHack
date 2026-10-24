"""v6: tune detector thresholds on TRAIN labels (lift of flagged tx among positives) + train-vs-test KS of f8."""
import numpy as np, pandas as pd
from scipy.stats import ks_2samp
lab = pd.read_csv("data/train/labels.csv")
tr = pd.read_parquet("cache/train_f8.parquet").merge(lab, on="TRANSACTION_HASH"); te = pd.read_parquet("cache/test_f8.parquet")
y = tr.is_suspicious.to_numpy(); base = y.mean()
fam = "family" if "family" in lab else [c for c in lab.columns if "famil" in c or "pattern" in c or "typology" in c][0]
print("label cols", list(lab.columns), "base rate", round(base, 4))
rows = []
for c in [c for c in tr.columns if c.startswith(("C_next", "C_prev"))]:
    f = tr[c] > 0; rows.append((c, int(f.sum()), round(y[f].mean() / base, 2) if f.sum() else 0, int(y[f].sum()), round(y[f].sum() / y.sum(), 3), round(f.mean() and (te[c] > 0).mean() / f.mean(), 2)))
for c, ths in (("D_max_dsend_1h_win", (3, 4, 5, 6, 8, 10)), ("D_dsend_trail_1h", (3, 4, 6, 8)), ("D_dsend_cent_6h", (4, 6, 8, 12)), ("C_chain_len", (2, 3, 4)), ("B_pair_cnt", (2, 3, 5, 8))):
    for t in ths:
        f = tr[c] >= t; rows.append((f"{c}>={t}", int(f.sum()), round(y[f].mean() / base, 2), int(y[f].sum()), round(y[f].sum() / y.sum(), 3), round((te[c] >= t).mean() / max(f.mean(), 1e-9), 2)))
for c in ("B_recurring_lookalike", "B_recurring_loose", "C_returns_origin", "C_tight", "C_flag", "D_flag", "B_daily_cadence", "B_reverse_pair", "B_first_time_pair", "A_seq_ooo_same_sec", "A_seq_backwards", "A_cross_issuer"):
    f = tr[c] > 0; rows.append((c, int(f.sum()), round(y[f].mean() / base, 2), int(y[f].sum()), round(y[f].sum() / y.sum(), 3), round((te[c] > 0).mean() / max(f.mean(), 1e-9), 2)))
r = pd.DataFrame(rows, columns=["detector", "n_flag", "lift", "n_pos", "recall", "test/train_flag_rate"])
print(r.sort_values("lift", ascending=False).to_string(index=False))
print("\nflag x family (positives):")
for c in ("C_flag", "C_tight", "D_flag", "B_recurring_lookalike"):
    print(c, tr[y == 1].groupby([ "difficulty", fam])[c].apply(lambda s: f"{int((s>0).sum())}/{len(s)}").to_dict())
ks = {c: round(ks_2samp(tr[c], te[c]).statistic, 3) for c in te.columns if c != "TRANSACTION_HASH"}
print("\nKS>0.05:", {k: v for k, v in sorted(ks.items(), key=lambda x: -x[1]) if v > 0.05})
