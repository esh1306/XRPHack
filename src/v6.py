"""v6 final score = rank(v3-pipeline score with fe8 groups A,B,C,D,G) + 0.2 * rank(look-alike head).
The head (src/lookalike.py) is trained only on positives vs legit look-alike-scenario rows, so it pushes decoys down.
Usage: v6.py BASE_PRED_PARQUET LOOKALIKE_NPZ OUT_PARQUET"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from pipeline import rank01
A = 0.2  # chosen on 3 fold seeds (stage-1 proxy: +0.44 +/- 0.04 AP; full pipeline seed 42: +0.70)
b = pd.read_parquet(sys.argv[1]); h = np.load(sys.argv[2], allow_pickle=True)
hd = pd.Series(h["test"], index=h["test_hash"]).loc[b.TRANSACTION_HASH].to_numpy()
s = rank01(b.score.to_numpy()) + A * rank01(hd)
pd.DataFrame({"TRANSACTION_HASH": b.TRANSACTION_HASH, "score": s / s.max()}).to_parquet(sys.argv[3]); print("wrote", sys.argv[3])
