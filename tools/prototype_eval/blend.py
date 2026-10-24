import sys, numpy as np, pandas as pd
sys.path.insert(0, "agent")
from app.cluster_prop import propagate_cluster_scores
from sklearn.metrics import average_precision_score as AP
from scipy.stats import rankdata
e = pd.read_parquet("cache/edges_train.parquet"); et = pd.read_parquet("cache/edges_test.parquet")
L = pd.read_csv("../data/train/labels.csv").set_index("TRANSACTION_HASH").loc[e.hash]; y=L.is_suspicious.to_numpy(); d=L.difficulty.to_numpy()
def rep(tag, sc):
    r=[100*AP(y,sc)]
    for k in ("easy","medium","hard"):
        m=(d==k)|(y==0); r.append(100*AP(y[m],sc[m]))
    print(tag, np.round(r,2).tolist(), flush=True); return r[0]
rk=lambda x: rankdata(x)/len(x)
fo = np.load("cache/oof42_stage1.npy")
o = pd.read_parquet("../cache/pred_test_v7x_oof.parquet")
us = e[["hash"]].merge(o, left_on="hash", right_on="TRANSACTION_HASH", how="left").oof.to_numpy()
ft = np.load("cache/test_stage1.npy")
x = pd.read_csv("../cache/submission_v7x_nohead.csv"); ut = et[["hash"]].merge(x, left_on="hash", right_on="TRANSACTION_HASH", how="left").score.to_numpy()
best=None
for w in (0.0, 0.3, 0.5, 0.7, 1.0):
    s = w*rk(us)+(1-w)*rk(fo); rep(f"w_ours={w} stage1", s)
    r = rep(f"w_ours={w} +prop", propagate_cluster_scores(e, rk(s)))
    st = w*rk(ut)+(1-w)*rk(ft)
    pd.DataFrame({"TRANSACTION_HASH": et.hash.values, "score": np.round(propagate_cluster_scores(et, rk(st)),6)}).to_csv(f"cache/blend_w{w}_prop_test.csv", index=False)
