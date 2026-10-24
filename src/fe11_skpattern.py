"""FE11: scikit-learn 'pattern mapping' (label-free per split except the OOF cluster target encoding done in eval).
1) template repetition density on a per-tx vector (NearestNeighbors / BallTree, standardised within split)
2) episode pattern map: amount-matched 1h link components -> shape vectors -> MiniBatchKMeans fit on TRAIN episodes,
   + HDBSCAN within split; 3) LocalOutlierFactor score on the tx vector.
build: python src/fe11_skpattern.py build  -> cache/fe11/{train,test}_fe11.parquet (+ logs/fe11_ks.csv)
eval:  python src/fe11_skpattern.py eval <base|all|tx|ep|...> <fold_seed>   (base = v6x stage-1 feature set)"""
import sys, os, json, time
from collections import defaultdict
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
H = 3600


def tx_frame(split_dir):
    from fe import load_split
    tx = load_split(split_dir).reset_index(drop=True)
    S = tx.ACCOUNT.to_numpy(); R = tx.DEST.to_numpy(); T = tx.TIMESTAMP.to_numpy().astype(np.int64)
    V = np.nan_to_num(tx.usd.to_numpy().astype(float)); VP = np.nan_to_num(tx.paid_usd.to_numpy().astype(float))
    VP = np.where(VP > 0, VP, V)
    return tx, S, R, T, V, VP


def build(split_dir):
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
    tx, S, R, T, V, VP = tx_frame(split_dir); n = len(tx); F = {}
    xrp = (tx.dst_asset.astype(str) == "XRP").to_numpy().astype(float); cross = tx.cross_ccy.fillna(0).to_numpy().astype(float)
    hr = (T % 86400) / 86400 * 2 * np.pi
    outdeg = pd.Series(S).map(pd.Series(S).value_counts()).rank(pct=True).to_numpy()
    indeg = pd.Series(R).map(pd.Series(R).value_counts()).rank(pct=True).to_numpy()
    # pass-through dwell: time since the sender's last incoming payment (log hours, capped 1 week)
    order = np.argsort(T, kind="stable"); last_in = {}; dwell = np.full(n, 168.0)
    for i in order:
        if S[i] in last_in: dwell[i] = min((T[i] - last_in[S[i]]) / H, 168.0)
        last_in[R[i]] = T[i]
    X = np.column_stack([np.log1p(V), xrp, cross, np.sin(hr), np.cos(hr), outdeg, indeg, np.log1p(dwell)])
    Z = StandardScaler().fit_transform(X)
    # ---- 1. template repetition density
    nn = NearestNeighbors(n_neighbors=21, n_jobs=2).fit(Z)
    d, ix = nn.kneighbors(Z); d, ix = d[:, 1:], ix[:, 1:]
    F["k11_d5"] = d[:, :5].mean(1); F["k11_d20"] = d.mean(1)
    F["k11_nn_same_s"] = (S[ix] == S[:, None]).mean(1); F["k11_nn_same_r"] = (R[ix] == R[:, None]).mean(1)
    F["k11_nn_other"] = ((S[ix] != S[:, None]) & (R[ix] != R[:, None])).mean(1)
    F["k11_nn_nsend"] = np.array([len(set(S[r])) for r in ix]) / 20.0
    for r in (0.25, 0.5, 1.0):
        F[f"k11_rad{r}"] = nn.radius_neighbors(Z, radius=r, return_distance=False)
        F[f"k11_rad{r}"] = np.array([len(a) - 1 for a in F[f"k11_rad{r}"]], dtype=float)
    big = V >= np.quantile(V[V > 0], 0.75); bi = np.where(big)[0]
    for c in ("k11b_d5", "k11b_same_s", "k11b_other", "k11b_rad0.5"): F[c] = np.full(n, np.nan)
    nb = NearestNeighbors(n_neighbors=11, n_jobs=2).fit(Z[bi]); db, ib = nb.kneighbors(Z[bi]); db, ib = db[:, 1:], bi[ib[:, 1:]]
    F["k11b_d5"][bi] = db[:, :5].mean(1); F["k11b_same_s"][bi] = (S[ib] == S[bi, None]).mean(1)
    F["k11b_other"][bi] = ((S[ib] != S[bi, None]) & (R[ib] != R[bi, None])).mean(1)
    F["k11b_rad0.5"][bi] = [len(a) - 1 for a in nb.radius_neighbors(Z[bi], radius=0.5, return_distance=False)]
    # ---- 3. LOF
    F["k11_lof"] = -LocalOutlierFactor(n_neighbors=20, n_jobs=2).fit(Z).negative_outlier_factor_
    # ---- 2. episodes: amount-matched 1h links (non-hub connector), components over tx
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    cp = pd.concat([pd.DataFrame({"a": S, "b": R}), pd.DataFrame({"a": R, "b": S})]).groupby("a").b.nunique()
    hub = set(cp[cp > cp.quantile(0.97)].index)
    outg = defaultdict(list)
    for i in order: outg[S[i]].append(i)
    outg = {k: (np.array(v), T[np.array(v)]) for k, v in outg.items()}
    ea, eb = [], []
    for i in range(n):
        if R[i] in hub or V[i] <= 0: continue
        g = outg.get(R[i])
        if g is None: continue
        idx, tt = g; a = np.searchsorted(tt, T[i], "right"); b = np.searchsorted(tt, T[i] + H, "right")
        c = idx[a:b][:30]; r = VP[c] / V[i]; c = c[(r >= 0.85) & (r <= 1.05)]
        ea += [i] * len(c); eb += list(c)
    m = coo_matrix((np.ones(len(ea)), (ea, eb)), shape=(n, n))
    _, ep = connected_components(m, directed=False)
    day = T // 86400; pair = pd.Series(S) + "|" + pd.Series(R)
    pdays = pair.map(pd.DataFrame({"p": pair, "d": day}).groupby("p").d.nunique()).to_numpy()
    df = pd.DataFrame({"ep": ep, "S": S, "R": R, "T": T, "V": V, "a": tx.dst_asset.astype(str).values, "pd": pdays})
    g = df.groupby("ep")
    shp = pd.DataFrame({"n_tx": g.size(), "span": (g["T"].max() - g["T"].min()) / H,
                        "n_s": g.S.nunique(), "n_r": g.R.nunique(),
                        "fo": df.groupby(["ep", "S"]).size().groupby(level=0).max(),
                        "fi": df.groupby(["ep", "R"]).size().groupby(level=0).max(),
                        "ret": g.V.min() / g.V.max().clip(lower=1e-9), "ncur": g.a.nunique(), "rec": g["pd"].mean(),
                        "lv": np.log1p(g.V.sum())})
    accs = pd.concat([df[["ep", "S"]].rename(columns={"S": "x"}), df[["ep", "R"]].rename(columns={"R": "x"})])
    shp["n_acc"] = accs.groupby("ep").x.nunique()
    E = np.log1p(shp[["n_tx", "span", "n_s", "n_r", "fo", "fi", "ncur", "rec", "n_acc"]].to_numpy())
    E = np.column_stack([E, shp.ret.to_numpy(), shp.lv.to_numpy()])
    print(split_dir, "n", n, "links", len(ea), "episodes", len(shp), "multi", int((shp.n_tx > 1).sum()), flush=True)
    return tx.TRANSACTION_HASH.values, F, ep, shp, E


def finalize_pair(tr, te):
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import MiniBatchKMeans, HDBSCAN
    out = []
    Es = [StandardScaler().fit_transform(x[4]) for x in (tr, te)]  # standardise within split (week shift)
    km = MiniBatchKMeans(n_clusters=40, random_state=42, n_init=5, batch_size=4096).fit(Es[0])
    for (h, F, ep, shp, E), Z in zip((tr, te), Es):
        lab = km.predict(Z); dist = np.linalg.norm(Z - km.cluster_centers_[lab], axis=1)
        share = pd.Series(lab).map(pd.Series(lab).value_counts(normalize=True)).to_numpy()
        hl = np.full(len(Z), -1); multi = np.where(shp.n_tx.to_numpy() > 1)[0]
        if len(multi) > 20:
            hl[multi] = HDBSCAN(min_cluster_size=10, min_samples=5).fit(Z[multi]).labels_
        hsz = pd.Series(hl).map(pd.Series(hl).value_counts()).to_numpy().astype(float)
        hsz[hl == -1] = 0
        epi = pd.Series(ep)
        F = dict(F)
        F["k11_ep_ntx"] = shp.n_tx.to_numpy()[ep].astype(float)
        F["k11_km_cluster"] = lab[ep].astype(float)   # used for OOF target encoding only (not a model input)
        F["k11_km_dist"] = dist[ep]; F["k11_km_share"] = share[ep]
        F["k11_hdb_noise"] = (hl[ep] == -1).astype(float); F["k11_hdb_size"] = hsz[ep] / max(len(multi), 1)
        d = pd.DataFrame(F).astype("float32"); d.insert(0, "TRANSACTION_HASH", h); out.append(d)
    return out


def ks_filter(a, b, thr=0.1):
    from scipy.stats import ks_2samp
    rep = []; keep = ["TRANSACTION_HASH"]
    for c in a.columns[1:]:
        x, y = a[c].dropna(), b[c].dropna(); ks = ks_2samp(x, y).statistic; act = "keep"
        if c == "k11_km_cluster": act = "id(TE only)"
        elif ks > thr:
            if a[c].nunique() >= 50:
                a[c] = a[c].rank(pct=True); b[c] = b[c].rank(pct=True); ks = ks_2samp(a[c].dropna(), b[c].dropna()).statistic; act = "ranked"
            if ks > thr: act = "drop"
        rep.append((c, round(float(ks), 4), act))
        if act != "drop": keep.append(c)
    pd.DataFrame(rep, columns=["col", "ks", "action"]).to_csv("logs/fe11_ks.csv", index=False); print(pd.DataFrame(rep).to_string(), flush=True)
    return a[keep], b[keep]


def te_oof(cl, y, folds, alpha=20.0):
    prior = y.mean(); out = np.zeros(len(y))
    for a, b in folds:
        s = pd.DataFrame({"c": cl[a], "y": y[a]}).groupby("c").y.agg(["sum", "count"])
        te = (s["sum"] + alpha * prior) / (s["count"] + alpha)
        out[b] = pd.Series(cl[b]).map(te).fillna(prior).to_numpy()
    return out


def evaluate(cfg, seed):
    os.environ.update(SHIFT_EXT="0", F8_GROUPS="A,B,C,D,G", FEATS="f2,f3,f4,f6", FE10="1", NJOBS="1")
    import lightgbm as lgb
    from pipeline import load_feats, DROP
    from sklearn.model_selection import StratifiedGroupKFold
    from fe10_medium import score
    lab = pd.read_csv("data/train/labels.csv")
    tr = load_feats("cache/train").merge(lab, on="TRANSACTION_HASH")
    cols = [c for c in tr.columns if c not in DROP and c not in lab.columns]
    f = pd.read_parquet("cache/fe11/train_fe11.parquet"); tr = tr.merge(f, on="TRANSACTION_HASH", how="left")
    y = tr.is_suspicious.to_numpy(); grp = tr.scenario_id.fillna(tr.TRANSACTION_HASH).to_numpy()
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(tr, y, grp))
    if "k11_km_cluster" in tr: tr["k11_km_te"] = te_oof(tr.k11_km_cluster.to_numpy(), y, folds)
    g = [c for c in f.columns if c not in ("TRANSACTION_HASH", "k11_km_cluster")] + ["k11_km_te"]
    TX = [c for c in g if c.startswith("k11_") and not c.startswith(("k11_ep", "k11_km", "k11_hdb"))] + [c for c in g if c.startswith("k11b_")]
    EP = [c for c in g if c.startswith(("k11_ep", "k11_km", "k11_hdb"))]
    add = {"base": [], "all": g, "tx": TX, "ep": EP, "eptop": ["k11_km_te", "k11_km_dist", "k11_hdb_noise"],
           "nn": ["k11_nn_same_s", "k11_nn_same_r", "k11_nn_other", "k11_nn_nsend", "k11b_same_s", "k11b_other"]}[cfg]
    cols = cols + [c for c in add if c in tr]
    p = dict(objective="binary", metric="average_precision", learning_rate=0.05, num_leaves=31, min_child_samples=20,
             feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, seed=42, verbose=-1, n_jobs=1)
    oof = np.zeros(len(y)); X = tr[cols].astype("float32")
    for a, b in folds:
        m = lgb.train(p, lgb.Dataset(X.iloc[a], y[a]), 2000, valid_sets=[lgb.Dataset(X.iloc[b], y[b])],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[b] = m.predict(X.iloc[b], num_iteration=m.best_iteration)
    pd.DataFrame({"TRANSACTION_HASH": tr.TRANSACTION_HASH, "oof": oof}).to_parquet(f"cache/fe11/oof_{cfg}_{seed}.parquet")
    r = score(tr, y, oof); r.update(cfg=cfg, fold_seed=seed, n_feat=len(cols))
    print(json.dumps(r), flush=True)
    with open("logs/fe11_results.jsonl", "a") as fh: fh.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(__file__), ".."))
    if sys.argv[1] == "build":
        t0 = time.time(); a = build("data/train"); b = build("data/test")
        A, B = finalize_pair(a, b); A, B = ks_filter(A, B)
        A.to_parquet("cache/fe11/train_fe11.parquet"); B.to_parquet("cache/fe11/test_fe11.parquet")
        print("cols", A.shape[1] - 1, "secs", round(time.time() - t0), flush=True)
    elif sys.argv[1] == "eval": evaluate(sys.argv[2], int(sys.argv[3]))
