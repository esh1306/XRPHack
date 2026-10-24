"""Final post-step (v9c/v9e/v11): optional rank-blend of our research pattern score into the propagated v9c score.
Usage: final_v11.py RAW_V10POST_CSV OUT_PARQUET [PATTERN_PARQUET PATTERN_WEIGHT]
  The v9c top block (top PATTERNS_PROTECT share by v9c score, default 0.05) is kept fixed in v9c order; below it
  score = (1-PW)*rank(v9c) + PW*rank(pattern score).  PW=0 -> v9c unchanged.
  Fold-42 OOF (logs/v11_patterns_cv*.log): v9c_w60 65.05/84.04/14.52/0.75; protect 5% PW 0.1 65.16/84.00/14.70/0.76;
  PW 0.2 65.20/83.98/14.78/0.77. An unprotected blend hurt (PW 0.1: 61.53), so the protected form is used.
Pattern score: src/patterns_research.py (AMLworld / Egressy et al. / Elliptic / FATF-inspired motifs)."""
import sys, numpy as np, pandas as pd
from scipy.stats import rankdata
rk = lambda x: rankdata(x) / len(x)
d = pd.read_csv(sys.argv[1])[["TRANSACTION_HASH", "score"]]
pw = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
if pw > 0:
    p = \
        d[["TRANSACTION_HASH"]].merge(pd.read_parquet(sys.argv[3]).rename(columns={"score": "pat"}), on="TRANSACTION_HASH", how="left").pat
    assert not p.isna().any(), "pattern scores missing hashes"
    import os
    K = float(os.environ.get("PATTERNS_PROTECT", "0.05"))
    B = rk(d.score.to_numpy()); sc = (1 - pw) * B + pw * rk(p.to_numpy()); top = B > 1 - K; sc[top] = 2 + B[top]
    d["score"] = rk(sc)
d.to_parquet(sys.argv[2]); print("final_v11", sys.argv[2], "pattern weight", pw, len(d))
