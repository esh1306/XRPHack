"""v10b: Medium + Hard-Relay Super-Blend (No CatBoost dilution) + 2-Round Multi-Scale Closed-Cluster & Relay Episode Propagation.
Why v10b fixes v10 (71.1 LB):
1. Removes CatBoost from the 983-feature Super-Matrix (CatBoost had 2.49% Medium AP vs LightGBM's 14.24% Medium AP, which crushed Medium recall in v10).
2. Uses 2-round multi-scale closed-cluster propagation (0.40 * Round1 + 0.60 * Round2), which lifts Medium AP from 11.66 -> 13.38 (+1.72 AP) and Hard AP from 0.71 -> 0.75 while boosting Easy AP (79.19 -> 79.45).
3. Propagates scores across verified 2-hop distributed_relay bridges (pair_n in {3,4}, gap >= 20h, strict forward [1.5h, 3.3h] pre-funding and post-payout).

Usage: python src/v10b.py TEST_DIR OUT_PARQUET
"""
from __future__ import annotations
import os, sys, time
from collections import defaultdict
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import average_precision_score as AP
from scipy.stats import rankdata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "prior_work", "prototype_agent"))

from app.cluster_prop import propagate_cluster_scores
from app.train_e import _ensemble as proto_ensemble
import v10


def rk(x):
    return rankdata(x) / len(x)


def propagate_relay_episodes(e: pd.DataFrame, scores: np.ndarray) -> np.ndarray:
    """Propagates scores across verified strict-forward distributed_relay episode chains."""
    t = e.t.to_numpy(); S = e.sender.to_numpy(); R = e.receiver.to_numpy()
    n = len(e)
    pair_n = e.groupby(["sender", "receiver"]).hash.transform("size").to_numpy()
    in_row, out_row = defaultdict(list), defaultdict(list)
    for i in range(n):
        in_row[R[i]].append(i); out_row[S[i]].append(i)
    in_row = {k: np.asarray(x) for k, x in in_row.items()}
    out_row = {k: np.asarray(x) for k, x in out_row.items()}
    pair_groups = defaultdict(list)
    for i in range(n):
        pair_groups[(S[i], R[i])].append(i)

    rel_prop = np.copy(scores)
    for (m1, m2), rows in pair_groups.items():
        if len(rows) not in (3, 4):
            continue
        if np.min(np.diff(t[rows])) < 20 * 3600:
            continue
        in_all, out_all = [], []
        n_fwd, n_rev = 0, 0
        for r_idx in rows:
            tb = t[r_idx]
            ir = in_row.get(m1, np.array([], int))
            ir_f = ir[(t[ir] >= tb - int(3.3 * 3600)) & (t[ir] <= tb - int(1.5 * 3600)) & (pair_n[ir] <= 3)]
            ir_r = ir[(t[ir] >= tb + int(1.5 * 3600)) & (t[ir] <= tb + int(3.3 * 3600)) & (pair_n[ir] <= 3)]
            or_ = out_row.get(m2, np.array([], int))
            or_f = or_[(t[or_] >= tb + int(1.5 * 3600)) & (t[or_] <= tb + int(3.3 * 3600)) & (pair_n[or_] <= 3)]
            or_r = or_[(t[or_] >= tb - int(3.3 * 3600)) & (t[or_] <= tb - int(1.5 * 3600)) & (pair_n[or_] <= 3)]
            if len(ir_f) > 0 and len(or_f) > 0:
                n_fwd += 1
            if len(ir_r) > 0 and len(or_r) > 0:
                n_rev += 1
            in_all.extend(ir_f); out_all.extend(or_f)
        if n_fwd == len(rows) and n_rev < len(rows):
            grp = list(set(rows) | set(in_all) | set(out_all))
            g_sc = 0.5 * scores[grp].mean() + 0.5 * np.percentile(scores[grp], 80)
            for idx_ in grp:
                rel_prop[idx_] = max(rel_prop[idx_], g_sc)
    return rel_prop


def multi_round_propagate(e: pd.DataFrame, s1: np.ndarray) -> np.ndarray:
    s1 = rk(s1)
    p1 = rk(propagate_cluster_scores(e, s1))
    p2 = rk(propagate_cluster_scores(e, p1))
    base_prop = rk(0.40 * p1 + 0.60 * p2)
    rel_prop = rk(propagate_relay_episodes(e, base_prop))
    return rk(0.90 * base_prop + 0.10 * rel_prop)


def main():
    t0 = time.time()
    test_dir = sys.argv[1] if len(sys.argv) > 1 else "data/test"
    out_pq = sys.argv[2] if len(sys.argv) > 2 else "cache/pred_test_v10b.parquet"
    v10.ensure_caches(test_dir)

    e_tr = pd.read_parquet("cache/proto_edges_train.parquet")
    e_te = pd.read_parquet("cache/proto_edges_test.parquet")
    lab = pd.read_csv("data/train/labels.csv").set_index("TRANSACTION_HASH").loc[e_tr.hash]
    y = lab.is_suspicious.to_numpy(); d = lab.difficulty.to_numpy()
    g = lab.scenario_id.fillna(pd.Series(lab.index, index=lab.index)).to_numpy()
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(e_tr, y, g))

    D_tr, D_te, Xp_tr, Xp_te, Xs_tr, Xs_te, _, _ = v10.load_super_matrices(e_tr, e_te)

    # 1. Prototype Stage-1 ensemble (5-seed HistGBM + ExtraTrees + QuantileLogReg)
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

    # 2. Xs_983 LightGBM specialist (14.24% Medium AP — no CatBoost dilution!)
    if os.path.exists("cache/oof_xs_lgb.npy") and os.path.exists("cache/te_xs_lgb.npy"):
        oof_xs_lgb = rk(np.load("cache/oof_xs_lgb.npy"))
        te_xs_lgb = rk(np.load("cache/te_xs_lgb.npy"))
    else:
        oof_xs_lgb = np.zeros(len(y)); te_xs_lgb = np.zeros(len(e_te))
        for seed in (42, 7):
            oof = np.zeros(len(y)); its = []
            p = dict(objective="binary", metric="average_precision", learning_rate=0.03, num_leaves=31,
                     min_child_samples=30, feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1,
                     lambda_l2=5.0, seed=seed, verbose=-1, n_jobs=8)
            for a, b in folds:
                m = lgb.train(p, lgb.Dataset(Xs_tr.iloc[a], y[a]), 1500, valid_sets=[lgb.Dataset(Xs_tr.iloc[b], y[b])],
                              callbacks=[lgb.early_stopping(150, verbose=False)])
                oof[b] = m.predict(Xs_tr.iloc[b]); its.append(m.best_iteration)
            oof_xs_lgb += 0.5 * rk(oof)
            m = lgb.train(p, lgb.Dataset(Xs_tr, y), int(np.mean(its) * 1.1))
            te_xs_lgb += 0.5 * rk(m.predict(Xs_te))
        oof_xs_lgb = rk(oof_xs_lgb); te_xs_lgb = rk(te_xs_lgb)

    # 3. Pipeline A v7x no-head test predictions (transductive self-trained)
    v7x_te_df = pd.read_parquet("cache/pred_test_v7x_base.parquet")
    te_v7x = rk(e_te[["hash"]].merge(v7x_te_df, left_on="hash", right_on="TRANSACTION_HASH", how="left").score.to_numpy())

    # Report CV on OOF (Proto_S1 + Xs_983_LGB + 2-round closed-cluster & relay propagation)
    s1_oof = rk(0.65 * oof_proto + 0.35 * oof_xs_lgb)
    final_oof = multi_round_propagate(e_tr, s1_oof)
    r = [100 * AP(y, final_oof)]
    for k in ("easy", "medium", "hard"):
        m = (d == k) | (y == 0); r.append(100 * AP(y[m], final_oof[m]))
    print(f"v10b OOF AP (all/easy/med/hard): {np.round(r, 2).tolist()} ({time.time()-t0:.1f}s)", flush=True)

    # Final Test Score: 0.42 * v7x_nohead + 0.40 * Proto_S1 + 0.18 * Xs_983_LGB -> 2-Round Cluster & Relay Propagation
    s1_te = rk(0.42 * te_v7x + 0.40 * te_proto + 0.18 * te_xs_lgb)
    final_te = multi_round_propagate(e_te, s1_te)

    pd.DataFrame({"TRANSACTION_HASH": e_te.hash.values, "score": final_te}).to_parquet(out_pq)
    print(f"Saved {out_pq} ({len(final_te)} rows) in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
