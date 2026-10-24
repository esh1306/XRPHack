"""v11 CV variant: keep the v9c top block (top K by v9c score) fixed; blend the pattern score only below it."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "../prior_work/prototype_agent")
from app.cluster_prop import propagate_cluster_scores as P
from sklearn.metrics import average_precision_score as AP
from scipy.stats import rankdata
rk = lambda x: rankdata(x) / len(x)
e = pd.read_parquet("cache/edges_train.parquet")
L = pd.read_csv("../data/train/labels.csv").set_index("TRANSACTION_HASH").loc[e.hash]; y = L.is_suspicious.to_numpy(); d = L.difficulty.to_numpy()
rep = lambda sc: np.round([100 * AP(y, sc)] + [100 * AP(y[(d == k) | (y == 0)], sc[(d == k) | (y == 0)]) for k in ("easy", "medium", "hard")], 2).tolist()
fo = np.load("cache/oof42_stage1.npy")
us = e[["hash"]].merge(pd.read_parquet("../cache/pred_test_v7x_oof.parquet"), left_on="hash", right_on="TRANSACTION_HASH", how="left").oof.to_numpy()
pat = e[["hash"]].merge(pd.read_parquet("../cache/pat_oof.parquet"), left_on="hash", right_on="TRANSACTION_HASH", how="left").oof.to_numpy()
B = rk(P(e, rk(0.4 * rk(us) + 0.6 * rk(fo))))
def protect(B, pat, K, pw):
    s = (1 - pw) * B + pw * rk(pat); top = B > 1 - K; s[top] = 2 + B[top]; return s
for K in (0.02, 0.03, 0.05):
    for pw in (0.05, 0.1, 0.2):
        print("protect top", K, "pw", pw, rep(protect(B.copy(), pat, K, pw)), flush=True)
