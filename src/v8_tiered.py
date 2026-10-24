"""v8: tiered re-rank. Base scores keep the top-K block; below it, rows are re-ordered by a medium/hard
specialist LightGBM (trained without easy positives; legit look-alike 'twin' negatives upweighted x3).
Usage: python src/v8_tiered.py cv            -> OOF evaluation over fold seeds 42/7/2026
       python src/v8_tiered.py test BASE_CSV OUT_PARQUET K A   -> full-fit specialist + combine with base test scores"""
import sys, os, time, json, numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from pipeline import load_feats, DROP, StratifiedGroupKFold, AP, rank01
NJ = int(os.environ.get("NJOBS", "3")); N_IT = int(os.environ.get("SPEC_IT", "600")); t0 = time.time()
LOOK = {"bipartite", "chain", "cycle", "fan_in", "fan_out", "fan_out_in", "multi_stage", "random", "stack"}
P = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_child_samples=20, feature_fraction=0.6,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, n_jobs=NJ)
lab = pd.read_csv("data/train/labels.csv")
tr = load_feats("cache/train").merge(lab, on="TRANSACTION_HASH")
y = tr.is_suspicious.to_numpy(); diff = tr.difficulty.to_numpy()
groups = tr.scenario_id.fillna(tr.TRANSACTION_HASH).to_numpy()
te = load_feats("cache/test"); cols = [c for c in te.columns if c not in DROP]
keep = ~((y == 1) & (diff == "easy"))                     # specialist never sees easy positives
w = np.where((y == 0) & tr.family.isin(LOOK).to_numpy(), 3.0, 1.0)


def fit(idx, seed=42):
    return lgb.train({**P, "seed": seed}, lgb.Dataset(tr[cols].iloc[idx], y[idx], weight=w[idx]), N_IT)


def combine(base, spec, K, a, soft=False):
    rb = rank01(base); n = len(base); top = rb > 1 - K
    below = rank01(spec) * (rb ** a) if not soft else 0.5 * rb + 0.5 * rank01(spec)
    return np.where(top, 2 + rb, rank01(below) * 0.999)


def metrics(s):
    r = [100 * AP(y, s)]
    for d in ("easy", "medium", "hard"):
        m = (diff == d) | (y == 0); r.append(100 * AP(y[m], s[m]))
    return np.round(r, 2).tolist()


if sys.argv[1] == "cv":
    res = {}
    for fs in (42, 7, 2026):
        folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=fs).split(tr, y, groups))
        spec = np.zeros(len(y))
        for a_, b_ in folds:
            ai = a_[keep[a_]]; spec[b_] = fit(ai).predict(tr[cols].iloc[b_])
        np.save(f"cache/v8_spec_oof_{fs}.npy", spec)
        bases = {"fe10": pd.read_parquet(f"cache/fe10/oof_fe10_{fs}.parquet")}
        if fs == 42: bases["v6x"] = pd.read_parquet("cache/oof_v6x.parquet")
        print(fs, "spec alone", metrics(spec), f"{time.time()-t0:.0f}s", flush=True)
        for bn, bdf in bases.items():
            base = bdf.set_index("TRANSACTION_HASH").oof.loc[tr.TRANSACTION_HASH].to_numpy()
            out = {"baseline": metrics(base)}
            for K in (0.015, 0.02, 0.025, 0.03, 0.04):
                for a in (0, 0.5, 1, 2):
                    out[f"K{K}_a{a}"] = metrics(combine(base, spec, K, a))
                out[f"K{K}_soft"] = metrics(combine(base, spec, K, 0, True))
            for name, v in out.items():
                print(fs, bn, name, v, "d_all", round(v[0] - out["baseline"][0], 2), flush=True)
            res[f"{fs}_{bn}"] = out
    json.dump(res, open("logs/v8_cv.json", "w"), indent=1)
else:
    base_csv, outp, K, a = sys.argv[2], sys.argv[3], float(sys.argv[4]), sys.argv[5]
    m = fit(np.where(keep)[0]); spec = m.predict(te[cols])
    b = pd.read_csv(base_csv).set_index("TRANSACTION_HASH").score.loc[te.TRANSACTION_HASH].to_numpy()
    s = combine(b, spec, K, 0, True) if a == "soft" else combine(b, spec, K, float(a))
    pd.DataFrame({"TRANSACTION_HASH": te.TRANSACTION_HASH, "score": s}).to_parquet(outp)
    pd.DataFrame({"TRANSACTION_HASH": te.TRANSACTION_HASH, "spec": spec}).to_parquet(outp.replace(".parquet", "_spec.parquet"))
    print("done", f"{time.time()-t0:.0f}s")
