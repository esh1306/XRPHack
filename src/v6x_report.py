"""v6x probe: fold-seed-42 full-pipeline OOF (base and base+0.2*head) vs v6, plus test Spearman vs v6/v3."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "src"); from pipeline import AP, rank01
from scipy.stats import spearmanr
lab = pd.read_csv("data/train/labels.csv"); h = np.load("cache/lookalike.npz", allow_pickle=True)
H = pd.Series(range(len(h["train_hash"])), index=h["train_hash"])
for name, f in (("v6", "cache/oof_final_v6.parquet"), ("v6x", "cache/oof_v6x.parquet")):
    o = pd.read_parquet(f).merge(lab, on="TRANSACTION_HASH"); y = o.is_suspicious.to_numpy()
    hd = rank01(h["oof_42"][H.loc[o.TRANSACTION_HASH].to_numpy()]); b = rank01(o.oof.to_numpy())
    for a in (0, 0.2):
        s = b + a * hd; r = [100 * AP(y, s)]
        for d in ("easy", "medium", "hard"):
            m = (o.difficulty == d).to_numpy() | (y == 0); r.append(100 * AP(y[m], s[m]))
        print(name, "head", a, "all/easy/medium/hard", np.round(r, 2).tolist())
x = pd.read_csv(sys.argv[1]) if len(sys.argv) > 1 else None
if x is not None:
    for ref in ("outputs/submission_v6.csv", "outputs/submission_v3.csv", "outputs/submission_v7.csv"):
        r = pd.read_csv(ref).merge(x, on="TRANSACTION_HASH", suffixes=("_r", "_x"))
        c = [c for c in r.columns if c.endswith("_r") and c != "reason_r"][0]; cx = c[:-2] + "_x"
        rho = spearmanr(r[c], r[cx]).correlation
        t = {k: len(set(r.nlargest(k, c).TRANSACTION_HASH) - set(r.nlargest(k, cx).TRANSACTION_HASH)) for k in (400, 1250)}
        print(ref, "spearman", round(rho, 4), "top-k changed", t)
