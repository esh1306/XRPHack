import sys, time, numpy as np, pandas as pd
sys.path.insert(0, "agent")
from joblib import Parallel
from app.train_e import _ensemble
from app.cluster_prop import propagate_cluster_scores
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import average_precision_score as AP
t0=time.time()
D = pd.read_parquet("cache/D_train.parquet"); T = pd.read_parquet("cache/D_test.parquet").reindex(columns=D.columns)
e = pd.read_parquet("cache/edges_train.parquet"); et = pd.read_parquet("cache/edges_test.parquet")
lab = pd.read_csv("../data/train/labels.csv").set_index("TRANSACTION_HASH").loc[e.hash]
y = lab.is_suspicious.to_numpy(); d = lab.difficulty.to_numpy()
g = lab.scenario_id.fillna(pd.Series(lab.index, index=lab.index)).to_numpy()
def rep(tag, sc):
    r=[100*AP(y,sc)]
    for k in ("easy","medium","hard"):
        m=(d==k)|(y==0); r.append(100*AP(y[m],sc[m]))
    print(tag, np.round(r,2).tolist(), round(time.time()-t0), flush=True)
run = Parallel(n_jobs=4)
# test first (deliverable)
st = _ensemble(D, y, T, run); np.save("cache/test_stage1.npy", st)
pt = propagate_cluster_scores(et, st)
out = pd.DataFrame({"TRANSACTION_HASH": et.hash.values, "score": np.round(pt,6)})
out.to_csv("out_test.csv", index=False); print("test written", round(time.time()-t0), flush=True)
oof = np.zeros(len(y))
for f, c in StratifiedGroupKFold(5, shuffle=True, random_state=42).split(D, y, g):
    oof[c] = _ensemble(D.iloc[f], y[f], D.iloc[c], run)
np.save("cache/oof42_stage1.npy", oof)
rep("prototype_stage1_cv42", oof)
p = propagate_cluster_scores(e, oof); np.save("cache/oof42_prop.npy", p)
rep("prototype_full_cv42", p)
pd.DataFrame({"TRANSACTION_HASH": e.hash.values, "oof_stage1": oof, "oof_prop": p}).to_parquet("cache/oof42.parquet")
