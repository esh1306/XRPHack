"""v10: Super-Ensemble + Hard Relay / Medium Closed-Cluster detectors + Multi-Scale Propagation.
Usage: python src/v10.py TEST_DIR OUT_PARQUET
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import average_precision_score as AP
from scipy.stats import rankdata, ks_2samp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "prior_work", "prototype_agent"))

from app.data import load_split, labels_array
from app.cluster_prop import propagate_cluster_scores
from app.cluster import cluster_features
from app.relay import relay_features
from app.funding import funding_features
from app.fingerprint import fingerprint_features
from app.train_e import design as proto_design, _ensemble as proto_ensemble
from fe11_hard_med import build_medium_cluster_features, build_hard_relay_features
import pipeline as pa_pipe


def rk(x):
    return rankdata(x) / len(x)


def ensure_caches(test_dir: str):
    os.makedirs("cache", exist_ok=True)
    for s, d in (("train", "data/train"), ("test", test_dir)):
        if not (os.path.exists(f"cache/proto_D_{s}.parquet") and os.path.exists(f"cache/proto_edges_{s}.parquet")):
            sp, D = proto_design(d, s)
            D.to_parquet(f"cache/proto_D_{s}.parquet")
            sp.edges.to_parquet(f"cache/proto_edges_{s}.parquet")
        if not os.path.exists(f"cache/proto_ext_{s}.parquet"):
            e = pd.read_parquet(f"cache/proto_edges_{s}.parquet")
            ext = pd.concat([cluster_features(e), relay_features(e), funding_features(e), fingerprint_features(e)], axis=1)
            ext.to_parquet(f"cache/proto_ext_{s}.parquet")
        if not os.path.exists(f"cache/fe11_{s}.parquet"):
            e = pd.read_parquet(f"cache/proto_edges_{s}.parquet")
            f11 = pd.concat([build_medium_cluster_features(e), build_hard_relay_features(e)], axis=1)
            f11.to_parquet(f"cache/fe11_{s}.parquet")


def load_super_matrices(e_tr: pd.DataFrame, e_te: pd.DataFrame):
    D_tr = pd.read_parquet("cache/proto_D_train.parquet").loc[e_tr.hash]
    D_te = pd.read_parquet("cache/proto_D_test.parquet").loc[e_te.hash].reindex(columns=D_tr.columns)
    ext_tr = pd.read_parquet("cache/proto_ext_train.parquet").loc[e_tr.hash].rename(columns=lambda c: f"uf_{c}")
    ext_te = pd.read_parquet("cache/proto_ext_test.parquet").loc[e_te.hash].rename(columns=lambda c: f"uf_{c}").reindex(columns=ext_tr.columns)
    f11_tr = pd.read_parquet("cache/fe11_train.parquet").loc[e_tr.hash]
    f11_te = pd.read_parquet("cache/fe11_test.parquet").loc[e_te.hash].reindex(columns=f11_tr.columns)

    os.environ["SHIFT_EXT"] = "0"
    os.environ["F8_GROUPS"] = "A,B,C,D,G"
    os.environ["FEATS"] = "f2,f3,f4,f6"
    os.environ["FE10"] = "1"
    import importlib
    importlib.reload(pa_pipe)
    A_tr = pa_pipe.load_feats("cache/train").set_index("TRANSACTION_HASH").loc[e_tr.hash]
    A_te = pa_pipe.load_feats("cache/test").set_index("TRANSACTION_HASH").loc[e_te.hash]
    a_cols = [c for c in A_te.columns if c not in pa_pipe.DROP]
    A_tr = A_tr[a_cols].rename(columns=lambda c: f"pa_{c}")
    A_te = A_te[a_cols].rename(columns=lambda c: f"pa_{c}").reindex(columns=A_tr.columns)

    for df_tr, df_te in ((ext_tr, ext_te), (f11_tr, f11_te)):
        for c in df_tr.columns:
            ks = ks_2samp(df_tr[c].fillna(-999), df_te[c].fillna(-999)).statistic
            if ks > 0.06 and df_tr[c].nunique() >= 20:
                df_tr[c] = df_tr[c].rank(pct=True)
                df_te[c] = df_te[c].rank(pct=True)

    Xp_tr = pd.concat([D_tr, ext_tr, f11_tr], axis=1).astype(np.float32)
    Xp_te = pd.concat([D_te, ext_te, f11_te], axis=1).reindex(columns=Xp_tr.columns).astype(np.float32)
    Xs_tr = pd.concat([Xp_tr, A_tr.astype(np.float32)], axis=1)
    Xs_te = pd.concat([Xp_te, A_te.astype(np.float32)], axis=1).reindex(columns=Xs_tr.columns)
    return D_tr, D_te, Xp_tr, Xp_te, Xs_tr, Xs_te, f11_tr, f11_te


def fit_gbdt_suite(Xtr: pd.DataFrame, y: np.ndarray, folds: list, Xte: pd.DataFrame, tag: str, t0: float):
    oof_l1, oof_l2 = np.zeros(len(y)), np.zeros(len(y))
    oof_cat, oof_xgb = np.zeros(len(y)), np.zeros(len(y))
    its_l1, its_l2, its_c, its_x = [], [], [], []
    p1 = dict(objective="binary", metric="average_precision", learning_rate=0.04, num_leaves=31,
              min_child_samples=30, feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1,
              lambda_l2=5.0, seed=42, verbose=-1, n_jobs=8)
    p2 = {**p1, "seed": 7, "num_leaves": 23, "min_child_samples": 40, "lambda_l2": 8.0}
    xp = dict(objective="binary:logistic", eval_metric="aucpr", eta=0.04, max_depth=6, subsample=0.8,
              colsample_bytree=0.5, min_child_weight=3, reg_lambda=5.0, seed=42, nthread=8, tree_method="hist")

    for a, b in folds:
        m1 = lgb.train(p1, lgb.Dataset(Xtr.iloc[a], y[a]), 1500, valid_sets=[lgb.Dataset(Xtr.iloc[b], y[b])],
                       callbacks=[lgb.early_stopping(120, verbose=False)])
        oof_l1[b] = m1.predict(Xtr.iloc[b]); its_l1.append(m1.best_iteration)

        m2 = lgb.train(p2, lgb.Dataset(Xtr.iloc[a], y[a]), 1500, valid_sets=[lgb.Dataset(Xtr.iloc[b], y[b])],
                       callbacks=[lgb.early_stopping(120, verbose=False)])
        oof_l2[b] = m2.predict(Xtr.iloc[b]); its_l2.append(m2.best_iteration)

        mc = CatBoostClassifier(iterations=1800, learning_rate=0.05, depth=6, random_seed=42, verbose=0,
                                thread_count=8, eval_metric="PRAUC", od_type="Iter", od_wait=120)
        mc.fit(Xtr.iloc[a], y[a], eval_set=(Xtr.iloc[b], y[b]), use_best_model=True)
        oof_cat[b] = mc.predict_proba(Xtr.iloc[b])[:, 1]; its_c.append(mc.get_best_iteration() + 1)

        mx = xgb.train(xp, xgb.DMatrix(Xtr.iloc[a], y[a]), 1500, evals=[(xgb.DMatrix(Xtr.iloc[b], y[b]), "v")],
                       early_stopping_rounds=120, verbose_eval=False)
        oof_xgb[b] = mx.predict(xgb.DMatrix(Xtr.iloc[b])); its_x.append(mx.best_iteration + 1)

    b_oof = 0.22 * rk(oof_l1) + 0.22 * rk(oof_l2) + 0.36 * rk(oof_cat) + 0.20 * rk(oof_xgb)
    print(f"[{tag}] OOF AP: l1={100*AP(y,oof_l1):.2f} l2={100*AP(y,oof_l2):.2f} cat={100*AP(y,oof_cat):.2f} xgb={100*AP(y,oof_xgb):.2f} -> blend={100*AP(y,b_oof):.2f} ({time.time()-t0:.0f}s)", flush=True)

    # Full refits on all train
    ml1 = lgb.train(p1, lgb.Dataset(Xtr, y), max(80, int(np.mean(its_l1) * 1.1)))
    ml2 = lgb.train(p2, lgb.Dataset(Xtr, y), max(80, int(np.mean(its_l2) * 1.1)))
    mc = CatBoostClassifier(iterations=max(100, int(np.mean(its_c) * 1.1)), learning_rate=0.05, depth=6,
                            random_seed=42, verbose=0, thread_count=8, eval_metric="PRAUC").fit(Xtr, y)
    mx = xgb.train(xp, xgb.DMatrix(Xtr, y), max(80, int(np.mean(its_x) * 1.1)))

    b_te = (0.22 * rk(ml1.predict(Xte)) + 0.22 * rk(ml2.predict(Xte)) +
            0.36 * rk(mc.predict_proba(Xte)[:, 1]) + 0.20 * rk(mx.predict(xgb.DMatrix(Xte))))
    return rk(b_oof), rk(b_te)


def main():
    t0 = time.time()
    test_dir = sys.argv[1] if len(sys.argv) > 1 else "data/test"
    out_pq = sys.argv[2] if len(sys.argv) > 2 else "cache/pred_test_v10.parquet"
    ensure_caches(test_dir)

    e_tr = pd.read_parquet("cache/proto_edges_train.parquet")
    e_te = pd.read_parquet("cache/proto_edges_test.parquet")
    lab = pd.read_csv("data/train/labels.csv").set_index("TRANSACTION_HASH").loc[e_tr.hash]
    y = lab.is_suspicious.to_numpy(); d = lab.difficulty.to_numpy()
    g = lab.scenario_id.fillna(pd.Series(lab.index, index=lab.index)).to_numpy()
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(e_tr, y, g))

    def rep(tag, sc):
        r = [100 * AP(y, sc)]
        for k in ("easy", "medium", "hard"):
            m = (d == k) | (y == 0); r.append(100 * AP(y[m], sc[m]))
        print(f"  {tag:34s}: all={r[0]:5.2f} | easy={r[1]:5.2f} | med={r[2]:5.2f} | hard={r[3]:4.2f}", flush=True)
        return r[0]

    D_tr, D_te, Xp_tr, Xp_te, Xs_tr, Xs_te, f11_tr, f11_te = load_super_matrices(e_tr, e_te)

    # 1. Prototype Stage-1 ensemble (cached if available)
    if os.path.exists("cache/proto_oof42_stage1.npy") and os.path.exists("cache/proto_test_stage1.npy"):
        oof_proto = rk(np.load("cache/proto_oof42_stage1.npy"))
        te_proto = rk(np.load("cache/proto_test_stage1.npy"))
    else:
        from joblib import Parallel
        run = Parallel(n_jobs=6, prefer="processes")
        te_proto = rk(proto_ensemble(D_tr, y, D_te, run))
        oof_p = np.zeros(len(y))
        for f, c in folds:
            oof_p[c] = proto_ensemble(D_tr.iloc[f], y[f], D_tr.iloc[c], run)
        oof_proto = rk(oof_p)
        np.save("cache/proto_oof42_stage1.npy", oof_proto)
        np.save("cache/proto_test_stage1.npy", te_proto)
    rep("1. Proto_S1 (257 cols)", oof_proto)

    # 2. Xp_476 GBDT Suite (Prototype + Ext + FE11 Hard-Relay & Medium-Cluster)
    oof_xp, te_xp = fit_gbdt_suite(Xp_tr, y, folds, Xp_te, "Xp_476", t0)
    rep("2. Xp_476_GBDT (476 cols)", oof_xp)

    # 3. Xs_983 GBDT Suite (All 983 features across both pipelines + FE11)
    oof_xs, te_xs = fit_gbdt_suite(Xs_tr, y, folds, Xs_te, "Xs_983", t0)
    rep("3. Xs_983_GBDT (983 cols)", oof_xs)

    # 4. Pipeline A v7x no-head test predictions (and OOF if present)
    v7x_te_df = pd.read_parquet("cache/pred_test_v7x_base.parquet")
    te_v7x = rk(e_te[["hash"]].merge(v7x_te_df, left_on="hash", right_on="TRANSACTION_HASH", how="left").score.to_numpy())

    # Combine Stage-1 scores and run Multi-Scale Closed-Cluster Propagation
    # On OOF (where we have oof_proto, oof_xp, oof_xs):
    s1_oof = rk(0.35 * oof_proto + 0.35 * oof_xp + 0.30 * oof_xs)
    rep("Combined S1 (Proto+Xp+Xs)", s1_oof)
    prop_oof = propagate_cluster_scores(e_tr, s1_oof)
    rep("Combined + ClusterProp", prop_oof)

    # On Test: blend v7x_nohead (0.35) + Proto_S1 (0.25) + Xp_476_GBDT (0.20) + Xs_983_GBDT (0.20)
    w_v7x = float(os.environ.get("W_V7X", "0.35"))
    w_pro = float(os.environ.get("W_PRO", "0.25"))
    w_xp  = float(os.environ.get("W_XP",  "0.20"))
    w_xs  = float(os.environ.get("W_XS",  "0.20"))
    s1_te = rk(w_v7x * te_v7x + w_pro * te_proto + w_xp * te_xp + w_xs * te_xs)
    prop_te = propagate_cluster_scores(e_te, s1_te)

    # Save components so we can also emit weight variants instantaneously
    np.savez("cache/v10_components.npz",
             hash_tr=e_tr.hash.to_numpy(), hash_te=e_te.hash.to_numpy(),
             oof_proto=oof_proto, oof_xp=oof_xp, oof_xs=oof_xs, prop_oof=prop_oof,
             te_v7x=te_v7x, te_proto=te_proto, te_xp=te_xp, te_xs=te_xs, prop_te=prop_te)

    pd.DataFrame({"TRANSACTION_HASH": e_te.hash.values, "score": prop_te}).to_parquet(out_pq)
    print(f"Saved {out_pq} ({len(prop_te)} rows) in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
