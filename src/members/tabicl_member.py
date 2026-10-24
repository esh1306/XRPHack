"""TabICLv2 (Qu et al., github.com/soda-inria/tabicl, ckpt tabicl-classifier-v2-20260212) as an ensemble member.
Per fold: top-K v3 features by a quick in-fold LightGBM gain, context = all fold-train positives + sampled
negatives, in-context prediction on fold-val (OOF) and test (averaged over folds). CPU, seeds fixed."""
import sys, time, json, numpy as np, pandas as pd, torch, lightgbm as lgb
from common import *
torch.set_num_threads(2)
K = int(sys.argv[1]) if len(sys.argv) > 1 else 100; NNEG = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
NEST = int(sys.argv[3]) if len(sys.argv) > 3 else 4; TIMING = len(sys.argv) > 4 and sys.argv[4] == "timing"
from tabicl import TabICLClassifier
t0 = time.time()
tr, te, y, folds, cols = train_setup()
Xa = tr[cols].reset_index(drop=True); Xb = te[cols].reset_index(drop=True)
rng = np.random.RandomState(SEED); oof = np.zeros(len(y)); tep = np.zeros(len(te))
for k, (a, b) in enumerate(folds):
    m = lgb.train(dict(objective="binary", learning_rate=0.05, num_leaves=31, feature_fraction=0.6, seed=SEED, verbose=-1, n_jobs=2),
                  lgb.Dataset(Xa.iloc[a], y[a]), 150)
    top = list(pd.Series(m.feature_importance("gain"), cols).sort_values(ascending=False).index[:K])
    pos = a[y[a] == 1]; neg = rng.choice(a[y[a] == 0], NNEG, replace=False); ctx = np.concatenate([pos, neg])
    clf = TabICLClassifier(n_estimators=NEST, random_state=SEED, n_jobs=2, device="cpu", kv_cache=True)
    clf.fit(Xa.iloc[ctx][top].to_numpy(np.float32), y[ctx])
    print("fit", k, f"{time.time()-t0:.0f}s", flush=True)
    if TIMING:
        t1 = time.time(); clf.predict_proba(Xa.iloc[b[:2000]][top].to_numpy(np.float32)); print("2000 rows", f"{time.time()-t1:.1f}s"); sys.exit()
    XV = Xa.iloc[b][top].to_numpy(np.float32)
    oof[b] = np.concatenate([clf.predict_proba(XV[i:i + 2000])[:, 1] for i in range(0, len(XV), 2000)])
    print("fold", k, round(100 * AP(y[b], oof[b]), 2), f"{time.time()-t0:.0f}s", flush=True)
    np.save(f"{MEM}/tabicl_oof_partial.npy", oof)
# test: one context drawn from the full train set (5x cheaper than per-fold test inference on CPU)
m = lgb.train(dict(objective="binary", learning_rate=0.05, num_leaves=31, feature_fraction=0.6, seed=SEED, verbose=-1, n_jobs=2), lgb.Dataset(Xa, y), 150)
top = list(pd.Series(m.feature_importance("gain"), cols).sort_values(ascending=False).index[:K])
ctx = np.concatenate([np.flatnonzero(y == 1), rng.choice(np.flatnonzero(y == 0), NNEG, replace=False)])
clf = TabICLClassifier(n_estimators=NEST, random_state=SEED, n_jobs=2, device="cpu", kv_cache=True)
clf.fit(Xa.iloc[ctx][top].to_numpy(np.float32), y[ctx]); XB = Xb[top].to_numpy(np.float32)
tep = np.concatenate([clf.predict_proba(XB[i:i + 2000])[:, 1] for i in range(0, len(XB), 2000)])  # chunked: full 40k OOMs
print("test done", f"{time.time()-t0:.0f}s", flush=True)
tag = f"tabicl_k{K}"
save(tag, tr.TRANSACTION_HASH, oof, te.TRANSACTION_HASH, tep)
res = blend_report(tr, y, oof, tag); json.dump(res, open(f"{MEM}/{tag}_results.json", "w"), indent=1)
