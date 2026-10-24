import sys, glob, numpy as np, pandas as pd
sys.path.insert(0, "src")
from pipeline import AP, load_feats
lab = pd.read_csv("data/train/labels.csv")
tr = pd.read_parquet("cache/train_f8.parquet")[["TRANSACTION_HASH"]].merge(pd.read_parquet("cache/train_f9.parquet")[["TRANSACTION_HASH"]]).merge(load_feats("cache/train")[["TRANSACTION_HASH"]]).merge(lab)
e8 = pd.read_parquet("cache/fe8/train_fe8.parquet")[["TRANSACTION_HASH"]]; tr = tr.merge(e8, how="left")
y = tr.is_suspicious.to_numpy()
def rep(o):
    r = [100 * AP(y, o)]
    for d in ["easy", "medium", "hard"]:
        m = ((tr.difficulty == d) | (y == 0)).to_numpy(); r.append(100 * AP(y[m], o[m]))
    return np.array(r)
ks = sorted({f.split("_")[1] for f in glob.glob("cache/exp10_*_42.npy")})
res = {k: [rep(np.load(f"cache/exp10_{k}_{fs}.npy")) for fs in (42, 7, 2026) if glob.glob(f"cache/exp10_{k}_{fs}.npy")] for k in ks}
b = np.array(res["v6"])
for k, v in res.items():
    v = np.array(v); 
    if len(v) < 3: print(k, "incomplete", len(v)); continue
    d = v[:, 0] - b[:, 0]
    print(f"{k:12s} mean {v.mean(0).round(2).tolist()} sd {v[:,0].std(ddof=1):.2f} gain {d.mean():+.2f} sd(diff) {d.std(ddof=1):.2f} per-seed {v[:,0].round(2).tolist()}")
