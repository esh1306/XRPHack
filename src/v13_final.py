"""v13_final.py: Reproduces the 71.6 Leaderboard model (submission_v13_protected_hard_med.csv).
Combines:
1. Pipeline A (v7x no-head: transductive self-trained GBDT + Stage-2 graph propagation, weight 0.08)
2. Pipeline B (Prototype Stage-1: 5-seed HistGradientBoosting + ExtraTrees + QuantileLogReg on flow/motif/account-agg/closed-cluster, weight 0.82)
3. Pipeline C (3-seed LightGBM specialist on the 983-feature Super-Matrix including FE11 Hard-Relay & Medium-Cluster features, weight 0.10)
4. Multi-scale closed-cluster score propagation (propagate_cluster_scores over 5m..4h novel-edge clusters)
5. Protected Top-4% re-ranking: locks the top 4% (Easy + top Medium) in exact cluster-propagated order, and below the top 4% blends in the 3-seed Xs_983 LightGBM specialist (0.15) + Verified Distributed-Relay Episode Propagation (0.10).

Usage: python src/v13_final.py TEST_DIR OUT_PARQUET
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import rankdata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "prior_work", "prototype_agent"))

from app.cluster_prop import propagate_cluster_scores
from app.train_e import _ensemble as proto_ensemble
import v10, v10b


def rk(x):
    return rankdata(x) / len(x)


def main():
    t0 = time.time()
    test_dir = sys.argv[1] if len(sys.argv) > 1 else "data/test"
    out_pq = sys.argv[2] if len(sys.argv) > 2 else "cache/pred_test_v13.parquet"
    v10.ensure_caches(test_dir)

    e_tr = pd.read_parquet("cache/proto_edges_train.parquet")
    e_te = pd.read_parquet("cache/proto_edges_test.parquet")
    lab = pd.read_csv("data/train/labels.csv").set_index("TRANSACTION_HASH").loc[e_tr.hash]
    y = lab.is_suspicious.to_numpy()

    D_tr, D_te, _, _, Xs_tr, Xs_te, _, _ = v10.load_super_matrices(e_tr, e_te)

    # 1. Prototype Stage-1 test predictions
    if os.path.exists("cache/proto_test_stage1.npy"):
        te_proto = rk(np.load("cache/proto_test_stage1.npy"))
    else:
        from joblib import Parallel
        run = Parallel(n_jobs=6, prefer="processes")
        te_proto = rk(proto_ensemble(D_tr, y, D_te, run))
        np.save("cache/proto_test_stage1.npy", te_proto)

    # 2. 3-seed LightGBM specialist on the 983-feature Super-Matrix (no CatBoost dilution)
    if os.path.exists("cache/te_xs_lgb.npy"):
        te_xs_lgb = rk(np.load("cache/te_xs_lgb.npy"))
    else:
        te_xs_lgb = np.zeros(len(e_te))
        for seed in (42, 7, 2026):
            p = dict(objective="binary", metric="average_precision", learning_rate=0.03, num_leaves=31,
                     min_child_samples=30, feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1,
                     lambda_l2=5.0, seed=seed, verbose=-1, n_jobs=8)
            m = lgb.train(p, lgb.Dataset(Xs_tr, y), 500)
            te_xs_lgb += rk(m.predict(Xs_te)) / 3.0
        te_xs_lgb = rk(te_xs_lgb)
        np.save("cache/te_xs_lgb.npy", te_xs_lgb)

    # 3. Pipeline A v7x no-head test predictions
    v7x_te_df = pd.read_parquet("cache/pred_test_v7x_base.parquet")
    te_v7x = rk(e_te[["hash"]].merge(v7x_te_df, left_on="hash", right_on="TRANSACTION_HASH", how="left").score.to_numpy())

    # 4. Stage-1 blend + Multi-scale closed-cluster propagation
    base_w90_plus = rk(propagate_cluster_scores(e_te, rk(0.08 * te_v7x + 0.82 * te_proto + 0.10 * te_xs_lgb)))
    rel_prop = rk(v10b.propagate_relay_episodes(e_te, base_w90_plus))

    # 5. Protect top 4% (Easy + Medium), blend specialist + relay episode propagation below top 4%
    K = 0.04
    top = base_w90_plus > (1.0 - K)
    sc = 0.75 * base_w90_plus + 0.15 * te_xs_lgb + 0.10 * rel_prop
    sc[top] = 2.0 + base_w90_plus[top]
    final_v13 = rk(sc)

    pd.DataFrame({"TRANSACTION_HASH": e_te.hash.values, "score": final_v13}).to_parquet(out_pq)
    print(f"Saved {out_pq} ({len(final_v13)} rows) in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
