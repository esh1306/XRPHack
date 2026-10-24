import sys, time, numpy as np, pandas as pd
sys.path.insert(0, "../prior_work/prototype_agent"); sys.path.insert(0, "../src")
from app.data import load_split
from app.cluster_prop import propagate_cluster_scores
from sklearn.metrics import average_precision_score as AP
from scipy.stats import rankdata
t0=time.time()
lab = pd.read_csv("../data/train/labels.csv")
sp = load_split("../data/train", "train"); e = sp.edges
o = pd.read_parquet("../cache/pred_test_v7x_oof.parquet")[["TRANSACTION_HASH","oof"]]
s = e[["hash"]].merge(o, left_on="hash", right_on="TRANSACTION_HASH", how="left").oof.to_numpy()
assert not np.isnan(s).any()
L = lab.set_index("TRANSACTION_HASH").loc[e.hash]; y = L.is_suspicious.to_numpy(); d = L.difficulty.to_numpy()
def rep(tag, sc):
    r=[100*AP(y,sc)]
    for k in ("easy","medium","hard"):
        m=(d==k)|(y==0); r.append(100*AP(y[m],sc[m]))
    print(tag, np.round(r,2).tolist(), flush=True)
s01 = rankdata(s)/len(s)
rep("v7x_nohead_oof", s)
p = propagate_cluster_scores(e, s01); np.save("cache/prop_ours_oof.npy", p)
rep("v7x+prototype_prop", p)
print("t", time.time()-t0, flush=True)
spt = load_split("../data/test", "test"); et = spt.edges
x = pd.read_csv("../cache/submission_v7x_nohead.csv")
st = et[["hash"]].merge(x, left_on="hash", right_on="TRANSACTION_HASH", how="left").score.to_numpy()
pt = propagate_cluster_scores(et, rankdata(st)/len(st))
pd.DataFrame({"TRANSACTION_HASH": et.hash, "score": np.round(pt,6)}).to_csv("cache/v7x_nohead_prop_test.csv", index=False)
print("done", time.time()-t0)
