import sys, numpy as np, pandas as pd
sys.path.insert(0, "../prior_work/prototype_agent")
from app.cluster_prop import propagate_cluster_scores
from sklearn.metrics import average_precision_score as AP
from scipy.stats import rankdata
e = pd.read_parquet("cache/edges_train.parquet")
L = pd.read_csv("../data/train/labels.csv").set_index("TRANSACTION_HASH").loc[e.hash]; y=L.is_suspicious.to_numpy(); d=L.difficulty.to_numpy()
def rep(tag, sc):
    r=[100*AP(y,sc)]
    for k in ("easy","medium","hard"):
        m=(d==k)|(y==0); r.append(100*AP(y[m],sc[m]))
    print(tag, np.round(r,2).tolist(), flush=True)
for f in ("oof_v6_fs7","oof_v6_fs2026","oof_v6x","oof_final_v6"):
    o = pd.read_parquet(f"../cache/{f}.parquet"); col=[c for c in o.columns if c!="TRANSACTION_HASH"][0]
    s = e[["hash"]].merge(o, left_on="hash", right_on="TRANSACTION_HASH", how="left")[col].to_numpy()
    if np.isnan(s).any(): print(f,"nan"); continue
    rep(f, s); rep(f+"+prop", propagate_cluster_scores(e, rankdata(s)/len(s)))
