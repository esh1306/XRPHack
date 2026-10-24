"""v7x2: fold-42 OOF vs v7x; test Spearman/top-k vs v7x no-head, v7x, v6x."""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, "src"); from pipeline import AP, rank01
from scipy.stats import spearmanr
lab = pd.read_csv("data/train/labels.csv"); h = np.load("cache/lookalike.npz", allow_pickle=True)
H = pd.Series(range(len(h["train_hash"])), index=h["train_hash"])
for name, f in (("v7x", "cache/pred_test_v7x_oof.parquet"), ("v7x2", "cache/pred_test_v7x2_oof.parquet")):
    o = pd.read_parquet(f).merge(lab, on="TRANSACTION_HASH"); y = o.is_suspicious.to_numpy()
    hd = rank01(h["oof_42"][H.loc[o.TRANSACTION_HASH].to_numpy()]); b = rank01(o.oof.to_numpy())
    for a in (0, 0.2):
        s = b + a * hd; r = [100 * AP(y, s)]
        for d in ("easy", "medium", "hard"):
            m = (o.difficulty == d).to_numpy() | (y == 0); r.append(100 * AP(y[m], s[m]))
        print(name, "head", a, "all/easy/medium/hard", np.round(r, 2).tolist())
for xf in sys.argv[1:]:
    x = pd.read_csv(xf) if xf.endswith(".csv") else pd.read_parquet(xf)
    for ref in ("cache/submission_v7x_nohead.csv", "outputs/submission_v7x.csv", "outputs/submission_v6x.csv"):
        r = pd.read_csv(ref)[["TRANSACTION_HASH", "score"]].merge(x[["TRANSACTION_HASH", "score"]], on="TRANSACTION_HASH", suffixes=("_r", "_x"))
        t = {k: len(set(r.nlargest(k, "score_r").TRANSACTION_HASH) - set(r.nlargest(k, "score_x").TRANSACTION_HASH)) for k in (400, 1250)}
        print(xf, "vs", ref, "spearman", round(spearmanr(r.score_r, r.score_x).correlation, 4), "top-k changed", t)
