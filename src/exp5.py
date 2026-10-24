"""v4 experiments on top of the v3 OOF: medium/hard specialist, fusion, episode smoothing."""
import sys, os, time, numpy as np, pandas as pd
sys.path.insert(0, "src")
from pipeline import *
from joblib import Parallel, delayed
lab = pd.read_csv("data/train/labels.csv")
tr = load_feats("cache/train", ("f2", "f3", "f4", "f6")).merge(lab, on="TRANSACTION_HASH")
oof = pd.read_parquet("cache/oof_final.parquet").set_index("TRANSACTION_HASH").loc[tr.TRANSACTION_HASH, "oof"].to_numpy()
y = tr.is_suspicious.to_numpy(); diff = tr.difficulty.to_numpy(); fam = tr.family.to_numpy()
groups = tr.scenario_id.fillna(tr.TRANSACTION_HASH).to_numpy()
f6c = set(pd.read_parquet("cache/train_f6.parquet").columns)
allc = [c for c in tr.columns if c not in DROP and c not in lab.columns]
noc6 = [c for c in allc if c not in f6c]
def rep(o):
    r = [round(100 * AP(y, o), 2)]
    for d in ["easy", "medium", "hard"]:
        m = (diff == d) | (y == 0); r.append(round(100 * AP(y[m], o[m]), 2))
    return r
print("v3 main", rep(oof), flush=True)
mh = (y == 1) & np.isin(diff, ["medium", "hard"])
lookalike = (y == 0) & tr.scenario_id.notna().to_numpy()
def spec(cols, seed, fs, wneg):
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=fs).split(tr, mh.astype(int), groups))
    o = np.zeros(len(y))
    X = tr[cols]
    for a, b in folds:
        a = a[~((y[a] == 1) & ~mh[a])]  # drop easy positives from training
        w = np.where(lookalike[a], wneg, 1.0)
        import lightgbm as lgb
        p = lgb_params(seed); p["n_jobs"] = 1
        # inner early stopping on held-out fold restricted to non-easy rows
        bb = b[~((y[b] == 1) & ~mh[b])]
        m = lgb.train(p, lgb.Dataset(X.iloc[a], mh[a].astype(int), weight=w), 2000,
                      valid_sets=[lgb.Dataset(X.iloc[bb], mh[bb].astype(int))], callbacks=[lgb.early_stopping(150, verbose=False)])
        o[b] = m.predict(X.iloc[b])
    return o
jobs = {"all_w3": (allc, 42, 42, 3.0), "no6_w3": (noc6, 42, 42, 3.0), "all_w1": (allc, 42, 42, 1.0), "all_w3_s7": (allc, 7, 42, 3.0)}
t0 = time.time()
res = dict(zip(jobs, Parallel(n_jobs=4)(delayed(spec)(*v) for v in jobs.values())))
print("spec done", round(time.time() - t0), flush=True)
for k, o in res.items():
    np.save(f"cache/spec_{k}.npy", o)
    m = (y == 0) | mh
    print("spec", k, "AP(med+hard vs neg)", round(100 * AP(y[m], o[m]), 2), "alone", rep(o), flush=True)
r0 = rank01(oof)
for k in ["all_w3", "no6_w3", "all_w1"] + ["avg"]:
    o = (rank01(res["all_w3"]) + rank01(res["all_w3_s7"])) / 2 if k == "avg" else res[k]
    rs = rank01(o)
    for w in (0.1, 0.2, 0.3, 0.5):
        print(k, "add", w, rep(r0 + w * rs))
    for thr in (0.97, 0.95, 0.9):
        for w in (0.3, 0.6):
            s = r0.copy(); low = r0 < thr; s[low] = np.minimum(r0[low] + w * rs[low], thr - 1e-9 + 1e-12 * rs[low])
            print(k, "band", thr, w, rep(s))
    print(k, "max", rep(np.maximum(r0, rs)))
comp = pd.read_parquet("cache/train_comp.parquet").set_index("TRANSACTION_HASH").loc[tr.TRANSACTION_HASH]
for c in ["cmp120", "cmp900", "flow480", "flow3600", "flow14400"]:
    g = pd.Series(r0).groupby(comp[c].to_numpy())
    for a in (0.2, 0.5):
        print("smooth", c, a, rep(r0 + a * g.transform("mean").to_numpy()), rep(r0 + a * g.transform("max").to_numpy()))
