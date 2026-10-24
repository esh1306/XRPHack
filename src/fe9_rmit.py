"""fe9_rmit: features from Samadi (RMIT) thesis / MPOCryptoML (arXiv 2508.12641), label-free, per split.
 P: multi-source Personalised PageRank seeded on net-source accounts (sender->destination USD graph, top-3% degree
    hubs removed) -> s/r PPR rank and PPR lift over unseeded PageRank.
 T: normalised timestamp score (NTS) |span(out) - span(in)| of the account's events, local windows (+-2h, +-6h) and week.
 W: normalised weight score (NWS) |USD in - USD out| / (in + out) of the account, same windows.
build: python src/fe9_rmit.py build ; eval: python src/fe9_rmit.py eval <cfg> <seed>   (cfg: ref | +P | +T | +W | +PTW)
"""
import sys, os, json, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
C1 = ['g2_am15m_pair_days', 'g2_am15m_recur_days', 'g2_am1h_recur_days', 'g2_tw15m_novel_pair', 'g2_am1h_dwell_med',
      'g2_am1h_ccy_switch', 'g2_am1h_acc_neps_mean', 'g2_tw15m_tx_per_acc']


def ppr_feats(S, R, V):
    import networkx as nx
    accs = pd.Index(np.unique(np.concatenate([S, R])))
    deg = pd.Series(np.concatenate([S, R])).value_counts()
    hubs = set(deg.index[: max(1, int(0.03 * len(deg)))])
    m = np.array([(s not in hubs) and (r not in hubs) for s, r in zip(S, R)])
    e = pd.DataFrame({"s": S[m], "r": R[m], "w": V[m]}).groupby(["s", "r"]).w.sum().reset_index()
    G = nx.DiGraph(); G.add_nodes_from(accs)
    G.add_weighted_edges_from(zip(e.s, e.r, e.w + 1e-6))
    out_s = pd.Series(V).groupby(S).sum().reindex(accs, fill_value=0)
    in_s = pd.Series(V).groupby(R).sum().reindex(accs, fill_value=0)
    net = (out_s - in_s).clip(lower=0)
    indeg0 = (pd.Series(1, index=R).groupby(level=0).sum().reindex(accs, fill_value=0) == 0)
    pers = net + indeg0 * net.max() * 0.0 + 1e-12
    pers = (net / net.sum()).to_dict() if net.sum() > 0 else None
    ppr = pd.Series(nx.pagerank(G, alpha=0.85, personalization=pers, weight="weight"))
    pr = pd.Series(nx.pagerank(G, alpha=0.85, weight="weight"))
    lift = np.log((ppr + 1e-12) / (pr + 1e-12))
    rk = ppr.rank(pct=True); lk = lift.rank(pct=True)
    return {"r9P_s_ppr_rk": rk.reindex(S).to_numpy(), "r9P_r_ppr_rk": rk.reindex(R).to_numpy(),
            "r9P_s_ppr_lift": lk.reindex(S).to_numpy(), "r9P_r_ppr_lift": lk.reindex(R).to_numpy(),
            "r9P_hub_edge": (~m).astype(int)}


def local_feats(S, R, T, V):
    out = {}
    ev = pd.DataFrame({"a": np.concatenate([S, R]), "t": np.concatenate([T, T]), "v": np.concatenate([V, V]),
                       "d": np.concatenate([np.ones(len(S)), np.zeros(len(R))])})  # d=1 out, 0 in
    idx = {}
    for a, g in ev.groupby("a"):
        g = g.sort_values("t")
        to, vo = g.t[g.d == 1].to_numpy(), g.v[g.d == 1].to_numpy()
        ti, vi = g.t[g.d == 0].to_numpy(), g.v[g.d == 0].to_numpy()
        idx[a] = (to, np.concatenate([[0], np.cumsum(vo)]), ti, np.concatenate([[0], np.cumsum(vi)]))
    def span(ts, lo, hi):
        a, b = np.searchsorted(ts, lo, "left"), np.searchsorted(ts, hi, "right")
        return (ts[b - 1] - ts[a] if b - a >= 2 else 0.0), a, b
    for side, A in (("s", S), ("r", R)):
        for W, nm in ((7200, "2h"), (21600, "6h"), (10**9, "wk")):
            nts = np.zeros(len(T)); nws = np.zeros(len(T))
            for k, (a, t) in enumerate(zip(A, T)):
                to, co, ti, ci = idx[a]
                so, a1, b1 = span(to, t - W, t + W); si, a2, b2 = span(ti, t - W, t + W)
                wo, wi = co[b1] - co[a1], ci[b2] - ci[a2]
                nts[k] = abs(so - si) / (min(W, 7 * 86400)); nws[k] = abs(wi - wo) / (wi + wo + 1.0)
            out[f"r9T_{side}_nts_{nm}"] = nts; out[f"r9W_{side}_nws_{nm}"] = nws
    return out


def build(split):
    from fe import load_split
    tx = load_split(f"data/{split}")
    S = tx.ACCOUNT.to_numpy(); R = tx.DEST.to_numpy(); T = tx.TIMESTAMP.to_numpy().astype(np.int64)
    V = np.nan_to_num(tx.usd.to_numpy().astype(float))
    f = {"TRANSACTION_HASH": tx.TRANSACTION_HASH.to_numpy()}
    f.update(ppr_feats(S, R, V)); f.update(local_feats(S, R, T, V))
    return pd.DataFrame(f)


def evaluate(cfg, seed):
    import lightgbm as lgb
    os.environ["SHIFT_EXT"] = "0"; os.environ["FEATS"] = "f2,f3,f4"; os.environ["F8_GROUPS"] = ""; os.environ["NJOBS"] = "1"
    from pipeline import load_feats, DROP
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.metrics import average_precision_score as AP
    lab = pd.read_csv("data/train/labels.csv")
    tr = load_feats("cache/train").merge(lab, on="TRANSACTION_HASH")
    cols = [c for c in tr.columns if c not in DROP and c not in lab.columns]
    f8 = pd.read_parquet("cache/fe8/train_fe8.parquet", columns=["TRANSACTION_HASH"] + C1)
    tr = tr.merge(f8, on="TRANSACTION_HASH"); cols += C1
    if cfg != "ref":
        f9 = pd.read_parquet("cache/fe9/train_fe9.parquet")
        if cfg.startswith("="): g = cfg[1:].split(";")
        else: g = [c for c in f9.columns if c != "TRANSACTION_HASH" and c[2] in cfg[1:]]
        tr = tr.merge(f9[["TRANSACTION_HASH"] + g], on="TRANSACTION_HASH"); cols += g
    y = tr.is_suspicious.to_numpy(); grp = tr.scenario_id.fillna(tr.TRANSACTION_HASH).to_numpy()
    folds = StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(tr, y, grp)
    p = dict(objective="binary", metric="average_precision", learning_rate=0.05, num_leaves=31, min_child_samples=20,
             feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, seed=42, verbose=-1, n_jobs=1)
    oof = np.zeros(len(y)); X = tr[cols].astype("float32"); imp = pd.Series(0.0, index=cols)
    for a, b in folds:
        m = lgb.train(p, lgb.Dataset(X.iloc[a], y[a]), 2000, valid_sets=[lgb.Dataset(X.iloc[b], y[b])],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[b] = m.predict(X.iloc[b], num_iteration=m.best_iteration); imp += m.feature_importance("gain")
    r = {"cfg": cfg, "fold_seed": seed, "n_feat": len(cols), "ap_all": round(100 * AP(y, oof), 2)}
    for dd in ("easy", "medium", "hard"):
        mk = (tr.difficulty == dd) | (y == 0); r[f"ap_{dd}"] = round(100 * AP(y[mk], oof[mk]), 2)
    top = np.argsort(-oof)[:400]; fp = tr.iloc[top][y[top] == 0]
    r["top400_fp"] = int(len(fp)); r["top400_fp_lookalike"] = int((fp.family != "background").sum())
    r["imp_new"] = {c: round(float(imp[c]), 1) for c in cols if c.startswith("r9")}
    print(json.dumps(r), flush=True)
    with open("logs/fe9_rmit_results.jsonl", "a") as fh: fh.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    if sys.argv[1] == "build":
        from scipy.stats import ks_2samp
        t0 = time.time(); a = build("train"); b = build("test"); print("built", a.shape, b.shape, f"{time.time()-t0:.0f}s")
        ks = {c: round(ks_2samp(a[c].dropna(), b[c].dropna()).statistic, 3) for c in a.columns[1:]}
        a.to_parquet("cache/fe9/train_fe9.parquet"); b.to_parquet("cache/fe9/test_fe9.parquet")
        pd.Series(ks).to_csv("logs/fe9_rmit_ks.csv"); print(json.dumps(ks))
    elif sys.argv[1] == "eval":
        evaluate(sys.argv[2], int(sys.argv[3]))
