"""v7: transductive self-training on the test week (pseudo-labelling), built on the v6 pipeline.
1) pseudo-labels from v6 test scores: top POS_FRAC -> 1, bottom NEG_FRAC -> 0; pseudo-negatives that share a
   cmp900 cluster with a pseudo-positive are dropped (a cluster's labelled rows share one label).
2) stage 1 (lgb42,lgb1,lgb2,xgb42,cat42) and stage 2 (lgb42,lgb1,lgb2,xgb42) retrained on train + pseudo rows (weight 0.5).
   Train OOF (sanity CV): 5 grouped folds of train, every fold also trains on all pseudo rows.
   Test: 5-fold GroupKFold(cmp900 cluster) over the pseudo set; model k trains on all train + pseudo folds != k,
   predicts pseudo fold k (so no pseudo row is scored by a model that saw its own label) and all unlabelled
   test rows (averaged over the 5 models).
3) final = (1-W)*s2 + W*s1 (W picked on train OOF), then + 0.2*rank(look-alike head), as in v6.
Usage: pseudo.py V6_PRED_PARQUET LOOKALIKE_NPZ OUT_PARQUET   env POS_FRAC (0.008) NEG_FRAC (0.6) PW (0.5)"""
import sys, os, time, json, numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from pipeline import *
from sklearn.model_selection import GroupKFold
t0 = time.time()
v6p, la_npz, out = sys.argv[1], sys.argv[2], sys.argv[3]
POS, NEG, PW = float(os.environ.get("POS_FRAC", 0.008)), float(os.environ.get("NEG_FRAC", 0.6)), float(os.environ.get("PW", 0.5))
lab = pd.read_csv("data/train/labels.csv")
tr = load_feats("cache/train").merge(lab, on="TRANSACTION_HASH"); te = load_feats("cache/test")
y = tr.is_suspicious.to_numpy(); groups = tr.scenario_id.fillna(tr.TRANSACTION_HASH).to_numpy()
folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(tr, y, groups))
cols = [c for c in te.columns if c not in DROP]
comp_tr = pd.read_parquet("cache/train_comp.parquet").set_index("TRANSACTION_HASH").loc[tr.TRANSACTION_HASH].reset_index()
comp_te = pd.read_parquet("cache/test_comp.parquet").set_index("TRANSACTION_HASH").loc[te.TRANSACTION_HASH].reset_index()
meta_tr, meta_te = meta_of("data/train", tr.TRANSACTION_HASH), meta_of("data/test", te.TRANSACTION_HASH)
# ---- pseudo-labels
s6 = pd.read_parquet(v6p).set_index("TRANSACTION_HASH").score.loc[te.TRANSACTION_HASH].to_numpy()
r = pd.Series(s6).rank(pct=True, method="first").to_numpy()
pl = np.full(len(te), -1); pl[r > 1 - POS] = 1; pl[r <= NEG] = 0
cl = comp_te.cmp900.to_numpy()
poscl = set(cl[pl == 1]); drop = (pl == 0) & np.isin(cl, list(poscl)); pl[drop] = -1
P = np.flatnonzero(pl >= 0); U = np.flatnonzero(pl < 0)
pfold = np.zeros(len(te), int) - 1
for k, (_, b) in enumerate(GroupKFold(5).split(P, groups=cl[P])): pfold[P[b]] = k
log = {"n_pseudo_pos": int((pl == 1).sum()), "n_pseudo_neg": int((pl == 0).sum()), "neg_dropped_shared_cluster": int(drop.sum()),
       "POS_FRAC": POS, "NEG_FRAC": NEG, "PW": PW}
print(log, flush=True)

def fit(kind, seed, Xa, ya, wa, Xb=None, yb=None, n_iter=None):
    if kind == "lgb":
        d = lgb.Dataset(Xa, ya, weight=wa)
        if n_iter is None:
            m = lgb.train(lgb_params(seed), d, 3000, valid_sets=[lgb.Dataset(Xb, yb)], callbacks=[lgb.early_stopping(200, verbose=False)])
            return m, m.best_iteration
        return lgb.train(lgb_params(seed), d, n_iter), n_iter
    if kind == "xgb":
        import xgboost as xgb
        p = dict(objective="binary:logistic", eval_metric="aucpr", eta=0.03, max_depth=6, subsample=0.8,
                 colsample_bytree=0.6, min_child_weight=2, seed=seed, nthread=NJ, tree_method="hist")
        da = xgb.DMatrix(Xa, ya, weight=wa)
        if n_iter is None:
            m = xgb.train(p, da, 3000, evals=[(xgb.DMatrix(Xb, yb), "v")], early_stopping_rounds=200, verbose_eval=False)
            return m, m.best_iteration + 1
        return xgb.train(p, da, n_iter), n_iter
    from catboost import CatBoostClassifier
    kw = dict(learning_rate=0.05, depth=6, random_seed=seed, verbose=0, thread_count=NJ, eval_metric="PRAUC")
    if n_iter is None:
        m = CatBoostClassifier(iterations=3000, od_type="Iter", od_wait=200, **kw); m.fit(Xa, ya, sample_weight=wa, eval_set=(Xb, yb), use_best_model=True)
        return m, m.get_best_iteration() + 1
    m = CatBoostClassifier(iterations=n_iter, **kw); m.fit(Xa, ya, sample_weight=wa); return m, n_iter

def job(Xtr, Xte, m, part, n_iter=None):
    kind, seed = m[:3], int(m[3:])
    XP, yP, wP = Xte.iloc[P], pl[P], np.full(len(P), PW)
    if part == "cv":   # train OOF with all pseudo rows added to every training fold
        oof = np.zeros(len(y)); its = []
        for a, b in folds:
            Xa = pd.concat([Xtr.iloc[a], XP]); ya = np.r_[y[a], yP]; wa = np.r_[np.ones(len(a)), wP]
            mm, it = fit(kind, seed, Xa, ya, wa, Xtr.iloc[b], y[b]); oof[b] = predict(mm, Xtr.iloc[b]); its.append(it)
        return oof, its
    pred = np.zeros(len(Xte)); nU = 0
    for k in range(5):   # test: pseudo fold k held out
        keep = P[pfold[P] != k]; hold = P[pfold[P] == k]
        Xa = pd.concat([Xtr, Xte.iloc[keep]]); ya = np.r_[y, pl[keep]]; wa = np.r_[np.ones(len(y)), np.full(len(keep), PW)]
        mm, _ = fit(kind, seed, Xa, ya, wa, n_iter=n_iter)
        pred[hold] = predict(mm, Xte.iloc[hold]); pred[U] += predict(mm, Xte.iloc[U]) / 5
    return pred, None

def stage(Xtr, Xte, models, tag):
    from joblib import Parallel, delayed
    par = int(os.environ.get("PAR", 6))
    cvres = Parallel(n_jobs=par)(delayed(job)(Xtr, Xte, m, "cv") for m in models)
    tres = Parallel(n_jobs=par)(delayed(job)(Xtr, Xte, m, "te", max(50, int(np.mean(its) * 1.1))) for m, (_, its) in zip(models, cvres))
    oofs = [rank01(o) for o, _ in cvres]; tes = [rank01(p) for p, _ in tres]
    for m, o in zip(models, oofs): log[f"{tag}_{m}"] = round(100 * AP(y, o), 2); print(tag, m, log[f"{tag}_{m}"], f"{time.time()-t0:.0f}s", flush=True)
    bo, bt = np.mean(oofs, 0), np.mean(tes, 0); log[f"{tag}_blend"] = round(100 * AP(y, bo), 2); print(tag, "blend", log[f"{tag}_blend"], flush=True)
    return bo, bt

M = os.environ.get("MODELS", "lgb42,lgb1,lgb2,xgb42,cat42").split(",")
o1, p1 = stage(tr[cols], te[cols], M, "s1")
X2tr = pd.concat([tr[cols].reset_index(drop=True), propagate(meta_tr, o1, comp_tr)], axis=1)
X2te = pd.concat([te[cols].reset_index(drop=True), propagate(meta_te, p1, comp_te)], axis=1)
o2, p2 = stage(X2tr, X2te, [m for m in M if not m.startswith("cat")], "s2")
for w in (0.0, 0.3, 0.5): log[f"final_w{w}"] = round(100 * AP(y, (1 - w) * o2 + w * o1), 2)
W = max((0.0, 0.3, 0.5), key=lambda w: log[f"final_w{w}"]); oo = (1 - W) * o2 + W * o1
for d in ["easy", "medium", "hard"]:
    m = (tr.difficulty == d) | (y == 0); log[f"oof_{d}"] = round(100 * AP(y[m], oo[m]), 2)
base = (1 - W) * p2 + W * p1
h = np.load(la_npz, allow_pickle=True)
hd = pd.Series(h["test"], index=h["test_hash"]).loc[te.TRANSACTION_HASH].to_numpy()
ho = pd.Series(h["oof_42"], index=h["train_hash"]).loc[tr.TRANSACTION_HASH].to_numpy()
log["oof_with_head0.2"] = round(100 * AP(y, rank01(oo) + 0.2 * rank01(ho)), 2)
for d in ["easy", "medium", "hard"]:
    m = ((tr.difficulty == d) | (y == 0)).to_numpy(); log[f"oof_head_{d}"] = round(100 * AP(y[m], (rank01(oo) + 0.2 * rank01(ho))[m]), 2)
s = rank01(base) + 0.2 * rank01(hd); log["weight_stage1"] = W
pd.DataFrame({"TRANSACTION_HASH": te.TRANSACTION_HASH, "score": s / s.max()}).to_parquet(out)
pd.DataFrame({"TRANSACTION_HASH": te.TRANSACTION_HASH, "score": rank01(base)}).to_parquet(out.replace(".parquet", "_base.parquet"))  # no-head variant
pd.DataFrame({"TRANSACTION_HASH": tr.TRANSACTION_HASH, "oof": oo}).to_parquet(out.replace(".parquet", "_oof.parquet"))
json.dump(log, open(out.replace(".parquet", "_cv.json"), "w"), indent=1)
print(log, f"{time.time()-t0:.0f}s", flush=True)
