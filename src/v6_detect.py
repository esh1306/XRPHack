"""v6 error analysis: which detector flag (chain / gather / recurring / none) the top-k positives and top-500 FPs carry.
Usage: v6_detect.py OOF_NPY_OR_PARQUET"""
import sys, numpy as np, pandas as pd
p = sys.argv[1]
lab = pd.read_csv("data/train/labels.csv")
f8 = pd.read_parquet("cache/train_f8.parquet")
if p.endswith(".npy"):
    h = np.load("cache/lookalike.npz", allow_pickle=True)["train_hash"]; o = pd.DataFrame({"TRANSACTION_HASH": h, "s": np.load(p)})
else:
    o = pd.read_parquet(p); o.columns = ["TRANSACTION_HASH", "s"]
d = o.merge(lab, on="TRANSACTION_HASH").merge(f8[["TRANSACTION_HASH", "C_flag", "D_flag", "B_recurring_lookalike"]], on="TRANSACTION_HASH")
d["flag"] = np.select([d.C_flag > 0, d.D_flag > 0, d.B_recurring_lookalike > 0], ["chain", "gather", "recurring"], "none")
y = d.is_suspicious == 1; k = int(y.sum())
d = d.sort_values("s", ascending=False).reset_index(drop=True); d["topk"] = d.index < k
pos = d[y.loc[d.index] if False else d.is_suspicious == 1]
t = pos.groupby(["difficulty", "family", "flag"]).topk.agg(["sum", "size"]).unstack("flag").fillna(0).astype(int)
print(f"positives caught in top-k (k={k}) by flag  [caught/total]:")
cols = sorted(set(c[1] for c in t.columns))
for idx, r in t.iterrows():
    print(f"  {idx[0]:6s} {idx[1]:18s} " + "  ".join(f"{c}:{r[('sum', c)]}/{r[('size', c)]}" for c in cols))
print("recall@k by flag:", pos.groupby("flag").topk.mean().round(3).to_dict(), "overall", round(pos.topk.mean(), 3))
fp = d[d.is_suspicious == 0].head(500)
fp = fp.assign(kind=np.where(fp.scenario_id.notna(), "lookalike:" + fp.family.astype(str), "background"))
print("top-500 FPs by flag:", fp.flag.value_counts().to_dict())
print("top-500 FPs by kind:", fp.kind.value_counts().head(12).to_dict())
print("top-500 FPs flag x kind:\n", pd.crosstab(fp.kind, fp.flag).to_string())
