"""Training/scoring pipeline: stage-1 GBDTs -> graph score propagation -> stage-2 -> rank blend.
Usage: python src/pipeline.py [TEST_FEATURE_PREFIX OUT_PARQUET]   (CV only when no args)"""
import sys, json, time, numpy as np, pandas as pd, lightgbm as lgb
from collections import defaultdict
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import average_precision_score as AP
from scipy.stats import rankdata
sys.path.insert(0, __file__.rsplit("/", 1)[0])
SEED = 42
import os
NJ = int(os.environ.get("NJOBS", "8"))
DROP = {"TRANSACTION_HASH", "t_rel", "dow"}  # week-specific features dropped for robustness


FEATS = tuple(os.environ.get("FEATS", "f2,f3,f4").split(","))
# adversarial validation (train vs practice split): split-level graph statistics shift strongly between weeks.
# Constant-per-split / giant-component columns are dropped; the remaining shifted ones are percentile-ranked
# within their own split (monotone, so tree CV is unchanged, but test values land on the train scale).
SHIFT_NORM = ["s_core", "r_core", "s_clust", "r_clust", "s_cross_rate", "r_cross_rate", "lv1_dens", "lv1_int_share", "lv3_dens",
              # v4: train-vs-test KS > 0.08 (adversarial check on the real test features)
              "cmp900_cross", "acc_r_mean_r_out_after_total", "s_out_med", "acc_s_mean_s_conserve_2d", "acc_s_mean_s_net_before", "spread"]
if os.environ.get("SHIFT_EXT", "1") != "1": SHIFT_NORM = SHIFT_NORM[:9]  # SHIFT_EXT=0 -> exact v3 normalisation


def is_shift_drop(c):
    return c.startswith("cmp3600_") or c.startswith("ep_cmp3600_") or c.endswith("_n_comm")


# v6 (fe8) feature groups kept via F8_GROUPS (e.g. "C,B"); conversion/leg columns shift train->test (KS ~0.09) -> dropped
F8_GROUPS = tuple(g for g in os.environ.get("F8_GROUPS", "").split(",") if g)
F8_SHIFT_DROP = {"A_sent_over_deliv", "A_path_code", "A_n_transfers", "A_is_conversion", "A_n_distinct_ccy", "A_ccy_change", "A_n_issuers",
                 "A_seq_ooo_same_sec", "A_seq_backwards"}


def load_f8(prefix, groups=F8_GROUPS):
    b = pd.read_parquet(f"{prefix}_f8.parquet")
    keep = [c for c in b.columns if c != "TRANSACTION_HASH" and c[0] in groups and c not in F8_SHIFT_DROP]
    return b[["TRANSACTION_HASH"] + keep]


# v6b (fe9) explorer-pattern groups via F9_GROUPS (e.g. "H1,H2"); columns with train/test KS >= 0.1 dropped
F9_GROUPS = tuple(g for g in os.environ.get("F9_GROUPS", "").split(",") if g)
F9_SHIFT_DROP = {"H1_size_30m", "H1_span_30m", "H1_rate_30m", "H2_pair_cnt_rk", "H2_oneoff_1h", "H3_s_fast_oth", "H3_r_fast_oth",
                 "H3_s_chain_oth", "H3_r_chain_oth", "H3_s_gather_mean", "H3_r_gather_mean", "H4_s_fee_rel", "H4_s_ep_gap_d"}
E8_COLS = ['g2_am15m_pair_days', 'g2_am15m_recur_days', 'g2_am1h_recur_days', 'g2_tw15m_novel_pair', 'g2_am1h_dwell_med',
           'g2_am1h_ccy_switch', 'g2_am1h_acc_neps_mean', 'g2_tw15m_tx_per_acc']


def load_f9(prefix, groups=F9_GROUPS):
    b = pd.read_parquet(f"{prefix}_f9.parquet")
    keep = [c for c in b.columns if c != "TRANSACTION_HASH" and c.split("_")[0] in groups and c not in F9_SHIFT_DROP]
    return b[["TRANSACTION_HASH"] + keep]


def load_feats(prefix, feats=FEATS):
    a = pd.read_parquet(f"{prefix}_f.parquet")
    for k in feats:
        a = a.merge(pd.read_parquet(f"{prefix}_{k}.parquet"), on="TRANSACTION_HASH")
    if F8_GROUPS:
        a = a.merge(load_f8(prefix), on="TRANSACTION_HASH")
    if F9_GROUPS:
        a = a.merge(load_f9(prefix), on="TRANSACTION_HASH")
    if os.environ.get("FE10") == "1":   # 21 look-alike-suppression columns from cache/fe10/<split>_fe10.parquet (other worker, read-only)
        sp = prefix.rsplit("/", 1)[-1]
        e = pd.read_parquet(f"cache/fe10/{sp}_fe10.parquet")
        a = a.merge(e[["TRANSACTION_HASH"] + [c for c in e.columns if c not in a.columns]], on="TRANSACTION_HASH", how="left")
    if os.environ.get("USE_E8") == "1":   # 8 episode columns from cache/fe8/<split>_fe8.parquet (raw)
        sp = prefix.rsplit("/", 1)[-1]
        e = pd.read_parquet(f"cache/fe8/{sp}_fe8.parquet")[["TRANSACTION_HASH"] + E8_COLS]
        a = a.merge(e, on="TRANSACTION_HASH", how="left")
    for c in SHIFT_NORM:
        if c in a: a[c] = a[c].rank(pct=True)
    return a.drop(columns=[c for c in a.columns if is_shift_drop(c)])


def lgb_params(seed):
    return dict(objective="binary", metric="average_precision", learning_rate=0.03, num_leaves=31,
                min_child_samples=20, feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1,
                lambda_l2=1.0, seed=seed, verbose=-1, n_jobs=NJ)


def fit_lgb(Xa, ya, Xb=None, yb=None, seed=SEED, n_iter=None):
    if n_iter is None:
        m = lgb.train(lgb_params(seed), lgb.Dataset(Xa, ya), 3000, valid_sets=[lgb.Dataset(Xb, yb)],
                      callbacks=[lgb.early_stopping(200, verbose=False)])
        return m, m.best_iteration
    return lgb.train(lgb_params(seed), lgb.Dataset(Xa, ya), n_iter), n_iter


def fit_xgb(Xa, ya, Xb=None, yb=None, seed=SEED, n_iter=None):
    import xgboost as xgb
    p = dict(objective="binary:logistic", eval_metric="aucpr", eta=0.03, max_depth=6, subsample=0.8,
             colsample_bytree=0.6, min_child_weight=2, seed=seed, nthread=NJ, tree_method="hist")
    da = xgb.DMatrix(Xa, ya)
    if n_iter is None:
        db = xgb.DMatrix(Xb, yb)
        m = xgb.train(p, da, 3000, evals=[(db, "v")], early_stopping_rounds=200, verbose_eval=False)
        return m, m.best_iteration + 1
    return xgb.train(p, da, n_iter), n_iter


def fit_cat(Xa, ya, Xb=None, yb=None, seed=SEED, n_iter=None):
    from catboost import CatBoostClassifier
    kw = dict(learning_rate=0.05, depth=6, random_seed=seed, verbose=0, thread_count=NJ, eval_metric="PRAUC")
    if n_iter is None:
        m = CatBoostClassifier(iterations=3000, od_type="Iter", od_wait=200, **kw)
        m.fit(Xa, ya, eval_set=(Xb, yb), use_best_model=True)
        return m, m.get_best_iteration() + 1
    m = CatBoostClassifier(iterations=n_iter, **kw); m.fit(Xa, ya); return m, n_iter


def predict(m, X):
    import xgboost as xgb
    if isinstance(m, lgb.Booster): return m.predict(X)
    if isinstance(m, xgb.Booster): return m.predict(xgb.DMatrix(X))
    return m.predict_proba(X)[:, 1]


FITTERS = {"lgb": fit_lgb, "xgb": fit_xgb, "cat": fit_cat}


def cv(X, y, folds, kind="lgb", seed=SEED, Xte=None):
    oof = np.zeros(len(y)); its = []
    for a, b in folds:
        m, it = FITTERS[kind](X.iloc[a], y[a], X.iloc[b], y[b], seed=seed)
        oof[b] = predict(m, X.iloc[b]); its.append(it)
    te = None
    if Xte is not None:
        m, _ = FITTERS[kind](X, y, seed=seed, n_iter=max(50, int(np.mean(its) * 1.1)))
        te = predict(m, Xte)
    return oof, te


def propagate(meta, p, comp=None):
    """Graph stacking features from per-tx scores p (OOF on train, stage-1 preds on test)."""
    S, R, T = meta.ACCOUNT.to_numpy(), meta.DEST.to_numpy(), meta.TIMESTAMP.to_numpy()
    n = len(p); F = {}
    df = pd.DataFrame({"S": S, "R": R, "p": p})
    # account-level stats over all incident txs (excluding self via leave-one-out mean)
    inc = pd.concat([df[["S", "p"]].rename(columns={"S": "a"}), df[["R", "p"]].rename(columns={"R": "a"})])
    g = inc.groupby("a").p
    asum, acnt, amax = g.sum(), g.size(), g.max()
    aq = g.quantile(0.9)
    for role, col in (("s", S), ("r", R)):
        F[f"{role}_acc_loo_mean"] = ((asum.reindex(col).to_numpy() - p) / np.maximum(acnt.reindex(col).to_numpy() - 1, 1))
        F[f"{role}_acc_max"] = amax.reindex(col).to_numpy()
        F[f"{role}_acc_q90"] = aq.reindex(col).to_numpy()
    pr = df.groupby(["S", "R"]).p
    F["pair_loo_mean"] = (pr.transform("sum").to_numpy() - p) / np.maximum(pr.transform("size").to_numpy() - 1, 1)
    # temporal neighbours: max score of sender's inflows in prior 3d, receiver's outflows next 3d
    out_idx, in_idx = defaultdict(list), defaultdict(list)
    for i in range(n): out_idx[S[i]].append(i); in_idx[R[i]].append(i)
    w = 3 * 86400; e = np.array([], int)
    a1 = np.zeros(n); a2 = np.zeros(n); a3 = np.zeros(n); a4 = np.zeros(n)
    out_idx = {k: np.array(v) for k, v in out_idx.items()}; in_idx = {k: np.array(v) for k, v in in_idx.items()}
    for i in range(n):
        si = in_idx.get(S[i], e); m = si[(T[si] >= T[i] - w) & (T[si] < T[i])]
        a1[i] = p[m].max() if len(m) else 0; a3[i] = p[m].mean() if len(m) else 0
        ro = out_idx.get(R[i], e); m = ro[(T[ro] > T[i]) & (T[ro] <= T[i] + w)]
        a2[i] = p[m].max() if len(m) else 0; a4[i] = p[m].mean() if len(m) else 0
    F["s_prev_in_max3d"], F["r_next_out_max3d"], F["s_prev_in_mean3d"], F["r_next_out_mean3d"] = a1, a2, a3, a4
    # 2-hop: neighbours' account max
    nb = defaultdict(set)
    for s, r in zip(S, R): nb[s].add(r); nb[r].add(s)
    am = amax.to_dict()
    hop = {a: np.mean([am[b] for b in v]) for a, v in nb.items()}
    F["s_2hop"] = pd.Series(hop).reindex(S).to_numpy(); F["r_2hop"] = pd.Series(hop).reindex(R).to_numpy()
    F["p_rank_in_s"] = df.groupby("S").p.rank(pct=True).to_numpy()
    F["p_rank_in_r"] = df.groupby("R").p.rank(pct=True).to_numpy()
    if comp is not None:  # candidate-scenario smoothing: leave-one-out mean / max within temporal components
        for c in [c for c in comp.columns if c != "TRANSACTION_HASH"]:
            g = pd.Series(p).groupby(comp[c].to_numpy())
            F[f"{c}_loo"] = (g.transform("sum").to_numpy() - p) / np.maximum(g.transform("size").to_numpy() - 1, 1)
            F[f"{c}_max"] = g.transform("max").to_numpy()
    return pd.DataFrame(F)


def meta_of(split_dir, hashes):
    import json as js
    tx = pd.read_parquet(f"{split_dir}/transactions.parquet", columns=["TRANSACTION_HASH", "ACCOUNT", "TIMESTAMP", "EXTRA_TRANSACTION_FIELDS"])
    tx["DEST"] = tx.EXTRA_TRANSACTION_FIELDS.map(lambda s: js.loads(s)["Destination"])
    tx["TIMESTAMP"] = tx.TIMESTAMP.astype("int64")
    return tx.set_index("TRANSACTION_HASH").loc[hashes, ["ACCOUNT", "DEST", "TIMESTAMP"]].reset_index()


def rank01(x): return rankdata(x) / len(x)
