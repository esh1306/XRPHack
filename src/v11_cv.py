"""v11 CV: v9c fold-42 OOF (0.4 ours v7x + 0.6 prototype stage-1, then cluster propagation) rank-blended with the
pattern-layer OOF (src/patterns_research.py cv) at weights 0/0.05/0.1/0.2. Run from prototype/."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "../prior_work/prototype_agent")
from app.cluster_prop import propagate_cluster_scores as P
from sklearn.metrics import average_precision_score as AP
from scipy.stats import rankdata
rk = lambda x: rankdata(x) / len(x)
e = pd.read_parquet("cache/edges_train.parquet")
L = pd.read_csv("../data/train/labels.csv").set_index("TRANSACTION_HASH").loc[e.hash]; y = L.is_suspicious.to_numpy(); d = L.difficulty.to_numpy()
def rep(sc):
    r = [100 * AP(y, sc)] + [100 * AP(y[(d == k) | (y == 0)], sc[(d == k) | (y == 0)]) for k in ("easy", "medium", "hard")]
    return np.round(r, 2).tolist()
fo = np.load("cache/oof42_stage1.npy")
us = e[["hash"]].merge(pd.read_parquet("../cache/pred_test_v7x_oof.parquet"), left_on="hash", right_on="TRANSACTION_HASH", how="left").oof.to_numpy()
pat = e[["hash"]].merge(pd.read_parquet("../cache/pat_oof.parquet"), left_on="hash", right_on="TRANSACTION_HASH", how="left").oof.to_numpy()
B = P(e, rk(0.4 * rk(us) + 0.6 * rk(fo)))
print("pattern only", rep(pat))
for pw in (0, 0.05, 0.1, 0.2):
    print("v9c_w60 + patterns", pw, rep(rk((1 - pw) * rk(B) + pw * rk(pat))), flush=True)
    print("  (pattern before propagation)", pw, rep(P(e, rk((1 - pw) * rk(0.4 * rk(us) + 0.6 * rk(fo)) + pw * rk(pat)))), flush=True)
