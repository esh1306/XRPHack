"""fe8: episode reconstruction + per-episode statistics (label-free, per split, no ID-derived features).
Groups (column prefixes):
  g1_  episode reconstruction variants: size/duration/depth of (a) time-bucketed sender->destination components
       (15m/1h/6h/24h, high-degree accounts excluded as connectors) and (b) amount-conserving link components
       (out within [0.85,1.05] x in, same account, within 15m/1h/6h/24h); seed-and-grow hop counts fwd/back.
  g2_  rich per-episode stats broadcast to each tx (sources/sinks, burstiness, value retention, currency/issuer
       switches, path share, pass-through share, dwell, round trip, roundness/structuring, split/merge,
       account reuse across episodes, recurrence of the same account set on several days, novelty, account age,
       newly activated accounts).
  g3_  episode shape classes (star-in/out, chain, bipartite + density, diamond, cycle).
  g4_  tx position inside its episode (time rank, hop index, source/sink/middle, value share).
All continuous columns are percentile-ranked within the split (monotone -> tree CV unchanged; removes level shift).
Usage: python src/fe8_episode.py data/train cache/fe8/train_fe8.parquet"""
import sys, time, warnings
from collections import defaultdict
import numpy as np, pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
warnings.filterwarnings("ignore")
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from fe import load_split

WINS = {"15m": 900, "1h": 3600, "6h": 21600, "24h": 86400}
LO, HI = 0.85, 1.05


def _cv(x):
    return x.std() / (x.mean() + 1e-9) if len(x) > 1 else 0.0


def amount_links(S, R, T, Vin, Vout, w, kmax=30):
    """links i->j: R[i]==S[j], 0<=T[j]-T[i]<=w, Vout[j]/Vin[i] in [LO,HI]; also returns candidate count."""
    n = len(S); out_i = defaultdict(list)
    for j in range(n): out_i[S[j]].append(j)
    out_i = {a: np.array(v) for a, v in out_i.items()}
    order_t = {a: T[v] for a, v in out_i.items()}
    src, dst = [], []
    for i in range(n):
        oo = out_i.get(R[i])
        if oo is None or Vin[i] <= 0: continue
        tt = order_t[R[i]]
        a = np.searchsorted(tt, T[i], "left"); b = np.searchsorted(tt, T[i] + w, "right")
        if b <= a: continue
        c = oo[a:min(b, a + kmax)]; c = c[c > i]
        if not len(c): continue
        r = Vout[c] / Vin[i]; c = c[(r >= LO) & (r <= HI)]
        src += [i] * len(c); dst += list(c)
    return np.array(src, int), np.array(dst, int)


def dag_depth(n, src, dst):
    """longest path to each node (from sources) and from each node (to sinks); edges go i<j (time order)."""
    fwd = np.zeros(n, int); bwd = np.zeros(n, int)
    if len(src):
        o = np.argsort(dst, kind="stable")
        for i, j in zip(src[o], dst[o]):
            if fwd[i] + 1 > fwd[j]: fwd[j] = fwd[i] + 1
        o = np.argsort(-src, kind="stable")
        for i, j in zip(src[o], dst[o]):
            if bwd[j] + 1 > bwd[i]: bwd[i] = bwd[j] + 1
    return fwd, bwd


def comps(n, src, dst):
    A = coo_matrix((np.ones(len(src)), (src, dst)), shape=(n, n))
    return connected_components(A, directed=False)[1]


def episode_stats(d, lab, name, src=None, dst=None, acc_info=None):
    """d: per-tx frame; lab: episode id per tx. Returns per-tx feature frame."""
    n = len(d); F = {}
    d = d.assign(c=lab)
    size = d.groupby("c").c.transform("size").to_numpy()
    gb = d.groupby("c")
    tmin = gb.TS.transform("min").to_numpy(); tmax = gb.TS.transform("max").to_numpy()
    # accounts per episode, sources/sinks at account level
    inc = pd.concat([pd.DataFrame({"c": lab, "a": d.S.values, "o": 1, "i": 0}), pd.DataFrame({"c": lab, "a": d.R.values, "o": 0, "i": 1})])
    ac = inc.groupby(["c", "a"]).agg(o=("o", "sum"), i=("i", "sum")).reset_index()
    ac["src"] = (ac.i == 0).astype(int); ac["snk"] = (ac.o == 0).astype(int); ac["pt"] = ((ac.i > 0) & (ac.o > 0)).astype(int)
    if acc_info is not None:
        ac = ac.join(acc_info, on="a")
    agg = ac.groupby("c").agg(n_acc=("a", "size"), n_src=("src", "sum"), n_snk=("snk", "sum"), n_pt=("pt", "sum"),
                              max_out=("o", "max"), max_in=("i", "max"))
    m = lambda col: agg[col].reindex(lab).to_numpy()
    nacc = m("n_acc")
    F[f"g1_{name}_ntx"] = size; F[f"g1_{name}_nacc"] = nacc
    F[f"g1_{name}_dur"] = np.log1p(tmax - tmin)
    # senders / receivers sets
    ns = gb.S.transform("nunique").to_numpy(); nr = gb.R.transform("nunique").to_numpy()
    npair = d.assign(p=d.S + "|" + d.R).groupby("c").p.transform("nunique").to_numpy()
    big = size >= 2
    F[f"g2_{name}_nsrc"] = m("n_src"); F[f"g2_{name}_nsnk"] = m("n_snk"); F[f"g2_{name}_pt_share"] = m("n_pt") / nacc
    F[f"g2_{name}_tx_per_acc"] = size / nacc
    d2 = d.sort_values(["c", "TS"]); gap = d2.groupby("c").TS.diff()
    gcv = gap.groupby(d2.c).agg(lambda x: _cv(x.dropna()) if x.notna().sum() > 1 else 0.0)
    F[f"g2_{name}_burst_cv"] = gcv.reindex(lab).to_numpy()
    F[f"g2_{name}_logusd"] = np.log1p(gb.V.transform("sum").to_numpy())
    F[f"g2_{name}_amt_cv"] = gb.V.transform(lambda x: _cv(x.to_numpy())).to_numpy()
    F[f"g2_{name}_n_assets"] = pd.concat([d[["c"]].assign(x=d.DA), d[["c"]].assign(x=d.SA)]).groupby("c").x.nunique().reindex(lab).to_numpy()
    F[f"g2_{name}_conv_share"] = gb.X.transform("mean").to_numpy()
    F[f"g2_{name}_path_share"] = gb.P.transform("mean").to_numpy()
    F[f"g2_{name}_hub_share"] = gb.HUB.transform("mean").to_numpy()
    F[f"g2_{name}_round"] = gb.RND.transform("mean").to_numpy()
    F[f"g2_{name}_below_thr"] = gb.BT.transform("mean").to_numpy()
    F[f"g2_{name}_novel_pair"] = gb.NOV.transform("mean").to_numpy()
    F[f"g2_{name}_pair_days"] = gb.PDAYS.transform("mean").to_numpy()
    F[f"g2_{name}_pair_reps"] = gb.PREPS.transform("mean").to_numpy()
    # near-identical splits: amount CV among txs of same sender inside episode
    sc = d.groupby(["c", "S"]).V.agg(lambda x: _cv(x.to_numpy()) if len(x) > 1 else np.nan).groupby(level=0).mean()
    F[f"g2_{name}_split_cv"] = sc.reindex(lab).to_numpy()
    # account age at episode start, newly-activated accounts, reuse across episodes
    if acc_info is not None:
        ac["age"] = 0.0
        st = pd.Series(tmin, index=lab).groupby(level=0).first()
        ac["age"] = st.reindex(ac.c).to_numpy() - ac.first_seen.to_numpy()
        a2 = ac.groupby("c").agg(age_min=("age", "min"), age_med=("age", "median"), newact=("new_act", "mean"))
        F[f"g2_{name}_age_min"] = a2.age_min.reindex(lab).to_numpy(); F[f"g2_{name}_age_med"] = a2.age_med.reindex(lab).to_numpy()
        F[f"g2_{name}_newact"] = a2.newact.reindex(lab).to_numpy()
    multi = ac[ac.c.map(pd.Series(size, index=lab).groupby(level=0).first()) >= 2]
    neps = multi.groupby("a").c.nunique()
    F[f"g2_{name}_s_neps"] = neps.reindex(d.S).fillna(0).to_numpy(); F[f"g2_{name}_r_neps"] = neps.reindex(d.R).fillna(0).to_numpy()
    ac["neps"] = neps.reindex(ac.a).fillna(0).to_numpy()
    F[f"g2_{name}_acc_neps_mean"] = ac.groupby("c").neps.mean().reindex(lab).to_numpy()
    # recurrence: same account set seen on several distinct days
    sig = ac.groupby("c").a.agg(lambda x: hash(tuple(sorted(x))))
    day = pd.Series(tmin // 86400, index=lab).groupby(level=0).first()
    sd = pd.DataFrame({"sig": sig, "day": day.reindex(sig.index)})
    rec = sd.groupby("sig").day.nunique()
    F[f"g2_{name}_recur_days"] = sd.sig.map(rec).reindex(lab).to_numpy()
    # round trip: some tx pays an episode source account (account level) -> after the fact a source receives
    src_acc = set(zip(ac.c[ac.src == 1], ac.a[ac.src == 1]))
    first_sender = d.sort_values("TS").groupby("c").S.first()
    rt = (d.R.values == first_sender.reindex(lab).values).astype(int)
    F[f"g2_{name}_roundtrip"] = pd.Series(rt).groupby(lab).transform("max").to_numpy()
    # shape classes
    dens = npair / np.maximum(ns * nr, 1)
    F[f"g3_{name}_bip_dens"] = np.where(big, dens, 0)
    F[f"g3_{name}_ns_nr_ratio"] = np.log((ns + 1) / (nr + 1))
    F[f"g3_{name}_star_in"] = ((nr == 1) & (ns >= 3)).astype(int)
    F[f"g3_{name}_star_out"] = ((ns == 1) & (nr >= 3)).astype(int)
    F[f"g3_{name}_bipartite"] = ((ns >= 2) & (nr >= 2) & (m("n_pt") == 0) & (dens >= 0.6)).astype(int)
    F[f"g3_{name}_chainlike"] = ((m("max_out") <= 1) & (m("max_in") <= 1) & (size >= 2)).astype(int)
    F[f"g3_{name}_maxdeg_out"] = m("max_out"); F[f"g3_{name}_maxdeg_in"] = m("max_in")
    # position
    F[f"g4_{name}_trank"] = (gb.TS.rank(method="first").to_numpy() - 1) / np.maximum(size - 1, 1)
    F[f"g4_{name}_vshare"] = d.V.to_numpy() / (gb.V.transform("sum").to_numpy() + 1e-9)
    acc_role = ac.set_index(["c", "a"])
    so = acc_role.i.reindex(list(zip(lab, d.S))).to_numpy(); ro = acc_role.o.reindex(list(zip(lab, d.R))).to_numpy()
    F[f"g4_{name}_from_source"] = (so == 0).astype(int); F[f"g4_{name}_to_sink"] = (ro == 0).astype(int)
    if src is not None:  # link-based extras (DAG)
        fwd, bwd = dag_depth(n, src, dst)
        dep = pd.Series(fwd + bwd).groupby(lab).transform("max").to_numpy()
        F[f"g1_{name}_depth"] = dep; F[f"g2_{name}_width"] = size / (dep + 1)
        F[f"g4_{name}_hop"] = fwd; F[f"g4_{name}_hops_to_sink"] = bwd; F[f"g4_{name}_hop_rel"] = fwd / np.maximum(dep, 1)
        outdeg = np.bincount(src, minlength=n); indeg = np.bincount(dst, minlength=n)
        isrc = indeg == 0; isnk = outdeg == 0
        vin = pd.Series(np.where(isrc, d.VP.values, 0)).groupby(lab).transform("sum").to_numpy()
        vout = pd.Series(np.where(isnk, d.V.values, 0)).groupby(lab).transform("sum").to_numpy()
        F[f"g2_{name}_retention"] = vout / (vin + 1e-9)
        F[f"g2_{name}_n_split"] = pd.Series((outdeg > 1).astype(int)).groupby(lab).transform("sum").to_numpy()
        F[f"g2_{name}_n_merge"] = pd.Series((indeg > 1).astype(int)).groupby(lab).transform("sum").to_numpy()
        F[f"g3_{name}_diamond"] = ((F[f"g2_{name}_n_split"] > 0) & (F[f"g2_{name}_n_merge"] > 0)).astype(int)
        F[f"g4_{name}_mid"] = ((indeg > 0) & (outdeg > 0)).astype(int)
        F[f"g4_{name}_outdeg"] = outdeg; F[f"g4_{name}_indeg"] = indeg
        if len(src):
            dw = pd.Series(d["TS"].values[dst] - d["TS"].values[src]).groupby(lab[src]).median()
            sw = pd.Series((d.DA.values[src] != d.DA.values[dst]).astype(int)).groupby(lab[src]).mean()
            iss = lambda a: np.array([x.split(":")[0] for x in a])
            isw = pd.Series((iss(d.DA.values[src]) != iss(d.DA.values[dst])).astype(int)).groupby(lab[src]).mean()
            lr = pd.Series(d.VP.values[dst] / np.maximum(d.V.values[src], 1e-9)).groupby(lab[src]).mean()
            F[f"g2_{name}_dwell_med"] = np.log1p(dw.reindex(lab).to_numpy())
            F[f"g2_{name}_ccy_switch"] = sw.reindex(lab).to_numpy(); F[f"g2_{name}_iss_switch"] = isw.reindex(lab).to_numpy()
            F[f"g2_{name}_link_ratio"] = lr.reindex(lab).to_numpy()
    return pd.DataFrame(F)


def build(split_dir):
    tx = load_split(split_dir)
    n = len(tx)
    S = tx.ACCOUNT.to_numpy(); R = tx.DEST.to_numpy(); T = tx.TIMESTAMP.to_numpy().astype(np.int64)
    V = np.nan_to_num(tx.usd.to_numpy().astype(float)); VP = np.nan_to_num(tx.paid_usd.to_numpy().astype(float))
    VP = np.where(VP > 0, VP, V)
    # transfer-leg hop structure: intermediaries (hub accounts) touched by each tx
    trf = pd.read_parquet(f"{split_dir}/transfers.parquet", columns=["TRANSACTION_HASH", "FROM_ADDRESS", "TO_ADDRESS"])
    ends = dict(zip(tx.TRANSACTION_HASH, zip(S, R)))
    inter = trf[[ (f not in ends[h]) or (t not in ends[h]) for h, f, t in zip(trf.TRANSACTION_HASH, trf.FROM_ADDRESS, trf.TO_ADDRESS)]]
    hub_tx = set(inter.TRANSACTION_HASH)
    # roundness / structuring
    ru = tx.dst_units.to_numpy()
    rnd = ((np.abs(ru - np.round(ru, -1)) < 1e-6) | (np.abs(V - np.round(V, -1)) < 0.5)).astype(int)
    bt = np.zeros(n)
    for thr in (1000, 3000, 5000, 10000):
        bt = np.maximum(bt, ((V >= 0.9 * thr) & (V < thr)).astype(int))
    pair = pd.Series(S) + "|" + pd.Series(R)
    d = pd.DataFrame({"S": S, "R": R, "TS": T, "V": V, "VP": VP, "DA": tx.dst_asset.values, "SA": tx.src_asset.values,
                      "X": tx.cross_ccy.values, "P": tx.path_type.fillna(0).values,
                      "HUB": tx.TRANSACTION_HASH.isin(hub_tx).astype(int).values, "RND": rnd, "BT": bt,
                      "NOV": (~pair.duplicated()).astype(int).values,
                      "PDAYS": pair.groupby(pair).transform(lambda x: 0).values})
    d["PDAYS"] = pd.Series(T // 86400).groupby(pair).transform("nunique").values
    d["PREPS"] = pair.groupby(pair).transform("size").values
    # account info: first seen, newly activated (first event is incoming, then an outflow within 1h)
    ev = pd.concat([pd.DataFrame({"a": S, "T": T, "dir": 1}), pd.DataFrame({"a": R, "T": T, "dir": 0})]).sort_values(["a", "T", "dir"])
    fs = ev.groupby("a").T.min(); fdir = ev.groupby("a").dir.first()
    fout = ev[ev.dir == 1].groupby("a").T.min()
    new_act = ((fdir == 0) & ((fout.reindex(fs.index) - fs) <= 3600)).astype(int)
    acc_info = pd.DataFrame({"first_seen": fs, "new_act": new_act})
    parts = [pd.DataFrame({"TRANSACTION_HASH": tx.TRANSACTION_HASH.values})]
    # high-degree accounts (top 3% by distinct counterparties) are not connectors in window components
    cp = pd.concat([pd.DataFrame({"a": S, "b": R}), pd.DataFrame({"a": R, "b": S})]).groupby("a").b.nunique()
    hi = set(cp[cp > cp.quantile(0.97)].index)
    for wn, w in WINS.items():
        b = T // w
        ks = np.where(pd.Series(S).isin(hi), pd.Series(S) + "#" + pd.Series(np.arange(n)).astype(str), pd.Series(S) + "@" + pd.Series(b).astype(str))
        kr = np.where(pd.Series(R).isin(hi), pd.Series(R) + "#" + pd.Series(np.arange(n)).astype(str), pd.Series(R) + "@" + pd.Series(b).astype(str))
        codes, uniq = pd.factorize(np.concatenate([ks, kr]))
        m = len(uniq); A = coo_matrix((np.ones(n), (codes[:n], codes[n:])), shape=(m, m))
        lab = connected_components(A, directed=False)[1][codes[:n]]
        parts.append(episode_stats(d, lab, f"tw{wn}", acc_info=acc_info))
        src, dst = amount_links(S, R, T, V, VP, w)
        lab2 = comps(n, src, dst)
        parts.append(episode_stats(d, lab2, f"am{wn}", src, dst, acc_info=acc_info))
        print(wn, "links", len(src), "eps>=2", (np.bincount(lab2) >= 2).sum(), flush=True)
    f = pd.concat(parts, axis=1)
    # raw values are kept: within-split percentile ranks create split-specific values at tie mass points
    # (KS~1 train vs test for discrete columns). Use rank_shifted() on the continuous, shifted columns instead.
    return f.astype({c: "float32" for c in f.columns[1:]})


def rank_shifted(tr, te, ks_thr=0.08, min_unique=50):
    """percentile-rank, within each split, only columns that are continuous (>=min_unique values) AND shifted (KS>thr)."""
    from scipy.stats import ks_2samp
    done = []
    for c in tr.columns[1:]:
        if tr[c].nunique() >= min_unique and ks_2samp(tr[c].dropna(), te[c].dropna()).statistic > ks_thr:
            tr[c] = tr[c].rank(pct=True); te[c] = te[c].rank(pct=True); done.append(c)
    return done


if __name__ == "__main__" and sys.argv[1] == "finalize":
    a = pd.read_parquet("cache/fe8/train_fe8_raw.parquet"); b = pd.read_parquet("cache/fe8/test_fe8_raw.parquet")
    done = rank_shifted(a, b); print("ranked", len(done), done)
    a.to_parquet("cache/fe8/train_fe8.parquet"); b.to_parquet("cache/fe8/test_fe8.parquet")

if __name__ == "__main__" and sys.argv[1] not in ("eval", "finalize"):
    t0 = time.time()
    f = build(sys.argv[1]); f.to_parquet(sys.argv[2]); print(f.shape, f"{time.time()-t0:.0f}s")


def evaluate(cfg, fold_seed):
    """CV: single-thread LightGBM on cached v3 features (+ fe8 groups). cfg: base | all | -gK | +gK"""
    import os, json, lightgbm as lgb
    os.environ["SHIFT_EXT"] = "0"; os.environ["FEATS"] = "f2,f3,f4"; os.environ["F8_GROUPS"] = ""; os.environ["NJOBS"] = "1"
    from pipeline import load_feats, DROP
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.metrics import average_precision_score as AP
    lab = pd.read_csv("data/train/labels.csv")
    tr = load_feats("cache/train").merge(lab, on="TRANSACTION_HASH")
    cols = [c for c in tr.columns if c not in DROP and c not in lab.columns]
    if cfg != "base":
        f8 = pd.read_parquet("cache/fe8/train_fe8.parquet")
        g = [c for c in f8.columns if c != "TRANSACTION_HASH"]
        if cfg.startswith("-"): g = [c for c in g if not c.startswith(cfg[1:] + "_")]
        elif cfg.startswith("+"): g = [c for c in g if c.startswith(cfg[1:] + "_")]
        elif cfg.startswith("="): g = [c for c in g if c in cfg[1:].split(";")]
        tr = tr.merge(f8[["TRANSACTION_HASH"] + g], on="TRANSACTION_HASH"); cols += g
    y = tr.is_suspicious.to_numpy(); grp = tr.scenario_id.fillna(tr.TRANSACTION_HASH).to_numpy()
    folds = StratifiedGroupKFold(5, shuffle=True, random_state=fold_seed).split(tr, y, grp)
    p = dict(objective="binary", metric="average_precision", learning_rate=0.05, num_leaves=31, min_child_samples=20,
             feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, seed=42, verbose=-1, n_jobs=1)
    oof = np.zeros(len(y)); imp = pd.Series(0.0, index=cols)
    X = tr[cols].astype("float32")
    for a, b in folds:
        m = lgb.train(p, lgb.Dataset(X.iloc[a], y[a]), 2000, valid_sets=[lgb.Dataset(X.iloc[b], y[b])],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[b] = m.predict(X.iloc[b], num_iteration=m.best_iteration); imp += m.feature_importance("gain")
    r = {"cfg": cfg, "fold_seed": fold_seed, "n_feat": len(cols), "ap_all": round(100 * AP(y, oof), 2)}
    for dd in ("easy", "medium", "hard"):
        mk = (tr.difficulty == dd) | (y == 0); r[f"ap_{dd}"] = round(100 * AP(y[mk], oof[mk]), 2)
    top = np.argsort(-oof)[:400]; fp = tr.iloc[top][y[top] == 0]
    r["top400_fp"] = int(len(fp)); r["top400_fp_lookalike"] = int((fp.family != "background").sum())
    r["top400_fp_by_family"] = fp.family.value_counts().to_dict()
    os.makedirs("logs", exist_ok=True)
    tag = cfg.replace(";", "_")[:40]
    if cfg != "base" and fold_seed == 42:
        imp[[c for c in cols if c.startswith("g")]].sort_values(ascending=False).head(60).to_csv(f"logs/fe8_imp_{tag}.csv")
    pd.DataFrame({"TRANSACTION_HASH": tr.TRANSACTION_HASH, "oof": oof}).to_parquet(f"cache/fe8/oof_{tag}_{fold_seed}.parquet")
    print(json.dumps(r), flush=True)
    with open("logs/fe8_results.jsonl", "a") as fh: fh.write(json.dumps(r) + "\n")


if __name__ == "__main__" and sys.argv[1] == "eval":
    evaluate(sys.argv[2], int(sys.argv[3]))
