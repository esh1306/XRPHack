import sys, glob, numpy as np, pandas as pd
sys.path.insert(0, "src"); from pipeline import AP
lab = pd.read_csv("data/train/labels.csv"); h = np.load("cache/lookalike.npz", allow_pickle=True)
L = lab.set_index("TRANSACTION_HASH").loc[h["train_hash"]].reset_index(); y = L.is_suspicious.to_numpy()
def rep(o):
    r = [100 * AP(y, o)]
    for d in ["easy", "medium", "hard"]:
        m = (L.difficulty == d).to_numpy() | (y == 0); r.append(100 * AP(y[m], o[m]))
    return r
R = {}
for f in sorted(glob.glob("cache/exp8_*_*.npy")):
    k, fs = f[11:-4].rsplit("_", 1); R.setdefault(k, {})[fs] = rep(np.load(f))
b = R["base"]
for k, v in R.items():
    a = np.array(list(v.values())); d = [v[s][0] - b[s][0] for s in v if s in b]
    print(f"{k:6s} n={len(v)} mean {a.mean(0).round(2).tolist()} sd_all {a[:,0].std(ddof=1).round(2) if len(a)>1 else '-'} gain_vs_base {np.round(d,2).tolist()}")
