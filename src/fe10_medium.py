"""FE10: stretched-motif detectors for MEDIUM scenarios (3h tolerance, kept sparse by structure).
Candidate edges = one-off pairs (sender->dest pair seen once in the split) between non-hub accounts
(hub = top 3% by distinct counterparties). Label-free, per split, no ID-derived features.
build:   python src/fe10_medium.py build            -> cache/fe10/{train,test}_fe10.parquet
eval:    python src/fe10_medium.py eval <cfg> <seed> (cfg: base | fe10 | fe10r9 | spec:<feat>)"""
import sys, os, json, time
from collections import defaultdict
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
H = 3600; W = 3 * H; LO, HI = 0.85, 1.05


def _grp_index(keys, T):
    d = defaultdict(list)
    for i, k in enumerate(keys): d[k].append(i)
    out = {}
    for k, v in d.items():
        v = np.array(v); o = np.argsort(T[v], kind="stable"); out[k] = (v[o], T[v][o])
    return out


def build(split_dir):
    from fe import load_split
    tx = load_split(split_dir); n = len(tx)
    S = tx.ACCOUNT.to_numpy(); R = tx.DEST.to_numpy(); T = tx.TIMESTAMP.to_numpy().astype(np.int64)
    V = np.nan_to_num(tx.usd.to_numpy().astype(float)); VP = np.nan_to_num(tx.paid_usd.to_numpy().astype(float))
    VP = np.where(VP > 0, VP, V)
    cp = pd.concat([pd.DataFrame({"a": S, "b": R}), pd.DataFrame({"a": R, "b": S})]).groupby("a").b.nunique()
    hub = set(cp[cp > cp.quantile(0.97)].index)
    pair = pd.Series(S) + "|" + pd.Series(R)
    oneoff = (pair.map(pair.value_counts()) == 1).to_numpy()
    nh = ~(pd.Series(S).isin(hub).to_numpy() | pd.Series(R).isin(hub).to_numpy())
    C = oneoff & nh & (V > 0)
    acc_med = pd.concat([pd.Series(V, index=S), pd.Series(V, index=R)]).groupby(level=0).median()
    ci = np.where(C)[0]
    outg = _grp_index(S[ci], T[ci]); ing = _grp_index(R[ci], T[ci])
    outg = {k: (ci[v], t) for k, (v, t) in outg.items()}; ing = {k: (ci[v], t) for k, (v, t) in ing.items()}
    F = {}
    # ---- 1. amount-conserving forward links (3h), onward to a NEW counterparty
    succ = defaultdict(list); loose = defaultdict(list)
    for i in ci:
        g = outg.get(R[i])
        if g is None: continue
        idx, tt = g
        a = np.searchsorted(tt, T[i], "right"); b = np.searchsorted(tt, T[i] + W, "right")
        c = idx[a:b]
        if not len(c): continue
        c = c[R[c] != S[i]]
        r = VP[c] / V[i]
        succ[i] = list(c[(r >= LO) & (r <= HI)])
        b6 = np.searchsorted(tt, T[i] + 2 * W, "right"); c6 = idx[a:b6]; r6 = VP[c6] / V[i]
        loose[i] = list(c6[(r6 >= 0.7) & (r6 <= 1.1)])  # for cycles (may return to origin)
    pred = defaultdict(list)
    for i, js in succ.items():
        for j in js: pred[j].append(i)
    order = np.argsort(T, kind="stable")
    fl = np.ones(n); fe_t = T.astype(float).copy(); fe_v = V.copy()   # forward longest chain
    for i in order[::-1]:
        for j in succ.get(i, ()):
            if fl[j] + 1 > fl[i]: fl[i] = fl[j] + 1; fe_t[i] = fe_t[j]; fe_v[i] = fe_v[j]
    bl = np.ones(n); bs_t = T.astype(float).copy(); bs_v = V.copy()
    for j in order:
        for i in pred.get(j, ()):
            if bl[i] + 1 > bl[j]: bl[j] = bl[i] + 1; bs_t[j] = bs_t[i]; bs_v[j] = bs_v[i]
    clen = np.where(C, fl + bl - 1, 0)
    cspan = np.where(clen > 1, (fe_t - bs_t) / H, 0)
    F["m10_chain_len"] = clen; F["m10_chain_span_h"] = cspan
    F["m10_chain_retention"] = np.where(clen > 1, fe_v / np.maximum(bs_v, 1e-9), 0)
    F["m10_chain_nsucc"] = np.array([len(succ.get(i, ())) for i in range(n)]); F["m10_chain_npred"] = np.array([len(pred.get(i, ())) for i in range(n)])
    # ---- 2. stretched fan-out / fan-in (+-3h, similar amounts) and gather-scatter
    def fan(groups, key, i):
        g = groups.get(key)
        if g is None: return 0, 0.0, 0.0, 0.0
        idx, tt = g
        a = np.searchsorted(tt, T[i] - W, "left"); b = np.searchsorted(tt, T[i] + W, "right")
        c = idx[a:b]; c = c[np.abs(np.log(V[c] / V[i])) < 0.15]
        if len(c) < 2: return len(c), 0.0, 0.0, 0.0
        v = V[c]; return len(c), (T[c].max() - T[c].min()) / H, v.std() / v.mean(), v.sum() / max(acc_med.get(key, 1.0), 1e-9)
    fo = np.zeros((n, 4)); fi = np.zeros((n, 4)); gsb = np.zeros(n); gsa = np.zeros(n)
    for i in ci:
        fo[i] = fan(outg, S[i], i); fi[i] = fan(ing, R[i], i)
        g = ing.get(S[i])  # one-off inflows to sender in the 3h before (gather before scatter)
        if g is not None:
            a = np.searchsorted(g[1], T[i] - W, "left"); b = np.searchsorted(g[1], T[i], "right"); gsb[i] = b - a
        g = outg.get(R[i])  # one-off outflows from receiver in the 3h after
        if g is not None:
            a = np.searchsorted(g[1], T[i], "right"); b = np.searchsorted(g[1], T[i] + W, "right"); gsa[i] = b - a
    for k, nm in enumerate(["n", "span_h", "cv", "sum_vs_typ"]):
        F[f"m10_fanout_{nm}"] = fo[:, k]; F[f"m10_fanin_{nm}"] = fi[:, k]
    F["m10_gather_before_send"] = gsb; F["m10_scatter_after_recv"] = gsa
    # ---- 3. cycles: return to origin within 6h through one-off non-hub edges (amount 0.7-1.1 per hop), depth<=5
    cyc_len = np.zeros(n); cyc_span = np.zeros(n)
    for i in ci:
        start = S[i]; frontier = [(i, 1)]; seen = {i}; found = 0
        while frontier and not found:
            nxt = []
            for k, d in frontier:
                if d >= 5: continue
                for j in loose.get(k, ()):
                    if j in seen or T[j] - T[i] > 2 * W: continue
                    if R[j] == start: found = (d + 1, (T[j] - T[i]) / H); break
                    seen.add(j); nxt.append((j, d + 1))
                if found: break
            frontier = nxt[:200]
        if found: cyc_len[i], cyc_span[i] = found
    F["m10_cycle_len"] = cyc_len; F["m10_cycle_span_h"] = cyc_span
    # ---- 4. per-tx motif summary
    sizes = np.vstack([clen, fo[:, 0], fi[:, 0], cyc_len]).T
    spans = np.vstack([cspan, fo[:, 1], fi[:, 1], cyc_span]).T
    k = sizes.argmax(1); F["m10_motif_max"] = sizes.max(1); F["m10_motif_type"] = np.where(sizes.max(1) >= 2, k + 1, 0)
    sp = spans[np.arange(n), k]
    F["m10_motif_tight"] = np.where(sizes.max(1) >= 2, sizes.max(1) / (sp + 0.25), 0)
    F["m10_is_cand"] = C.astype(int)
    # propagate: max motif size over accounts' candidate edges (account-week context of the stretched motif)
    ms = pd.Series(F["m10_motif_max"])
    F["m10_sender_motif_max"] = ms.groupby(S).transform("max").to_numpy()
    F["m10_recv_motif_max"] = ms.groupby(R).transform("max").to_numpy()
    out = pd.DataFrame(F).astype("float32"); out.insert(0, "TRANSACTION_HASH", tx.TRANSACTION_HASH.values)
    print(split_dir, "cand", C.sum(), "links", sum(len(v) for v in succ.values()), "chain>=3", (clen >= 3).sum(),
          "cyc", (cyc_len > 0).sum(), "fanout>=3", (fo[:, 0] >= 3).sum(), flush=True)
    return out


def finalize(tr, te, thr=0.1):
    from scipy.stats import ks_2samp
    keep, rep = ["TRANSACTION_HASH"], []
    for c in tr.columns[1:]:
        ks = ks_2samp(tr[c], te[c]).statistic; act = "keep"
        if ks > thr:
            if tr[c].nunique() >= 50:
                tr[c] = tr[c].rank(pct=True); te[c] = te[c].rank(pct=True)
                ks = ks_2samp(tr[c], te[c]).statistic; act = "ranked"
            if ks > thr: act = "drop"
        rep.append((c, round(ks, 4), act))
        if act != "drop": keep.append(c)
    pd.DataFrame(rep, columns=["col", "ks", "action"]).to_csv("logs/fe10_ks.csv", index=False)
    return tr[keep], te[keep]


COMPACT = ["m10_chain_len", "m10_fanout_n", "m10_fanin_n", "m10_gather_before_send", "m10_scatter_after_recv",
           "m10_motif_max", "m10_motif_tight"]
R9 = ["r9T_s_nts_6h", "r9T_s_nts_wk", "r9T_r_nts_wk"]


def evaluate(cfg, seed):
    os.environ["SHIFT_EXT"] = "0"; os.environ["FEATS"] = "f2,f3,f4"; os.environ["F8_GROUPS"] = ""; os.environ["NJOBS"] = "1"
    import lightgbm as lgb
    from pipeline import load_feats, DROP
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.metrics import average_precision_score as AP
    lab = pd.read_csv("data/train/labels.csv")
    tr = load_feats("cache/train").merge(lab, on="TRANSACTION_HASH")
    cols = [c for c in tr.columns if c not in DROP and c not in lab.columns]
    f10 = pd.read_parquet("cache/fe10/train_fe10.parquet"); g10 = [c for c in f10.columns if c != "TRANSACTION_HASH"]
    tr = tr.merge(f10, on="TRANSACTION_HASH", how="left")
    f9 = pd.read_parquet("cache/fe9/train_fe9.parquet")[["TRANSACTION_HASH"] + R9]
    tr = tr.merge(f9, on="TRANSACTION_HASH", how="left")
    spec = cfg.startswith("spec")
    if cfg in ("fe10", "spec10"): cols += g10
    if cfg in ("fe10r9", "spec10r9"): cols += g10 + R9
    if cfg == "fe10c": cols += COMPACT
    y = tr.is_suspicious.to_numpy(); grp = tr.scenario_id.fillna(tr.TRANSACTION_HASH).to_numpy()
    med = (tr.difficulty == "medium").to_numpy()
    p = dict(objective="binary", metric="average_precision", learning_rate=0.05, num_leaves=31, min_child_samples=20,
             feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, seed=42, verbose=-1, n_jobs=1)
    oof = np.zeros(len(y)); X = tr[cols].astype("float32")
    for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(tr, y, grp):
        if spec:  # medium positives + all negatives; ES on medium-vs-negatives in the val fold
            a = a[(y[a] == 0) | med[a]]; bv = b[(y[b] == 0) | med[b]]
        else: bv = b
        m = lgb.train(p, lgb.Dataset(X.iloc[a], y[a]), 2000, valid_sets=[lgb.Dataset(X.iloc[bv], y[bv])],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[b] = m.predict(X.iloc[b], num_iteration=m.best_iteration)
    pd.DataFrame({"TRANSACTION_HASH": tr.TRANSACTION_HASH, "oof": oof}).to_parquet(f"cache/fe10/oof_{cfg}_{seed}.parquet")
    r = score(tr, y, oof); r.update(cfg=cfg, fold_seed=seed, n_feat=len(cols))
    print(json.dumps(r), flush=True)
    with open("logs/fe10_results.jsonl", "a") as fh: fh.write(json.dumps(r) + "\n")


def score(tr, y, oof):
    from sklearn.metrics import average_precision_score as AP
    r = {"ap_all": round(100 * AP(y, oof), 2)}
    for dd in ("easy", "medium", "hard"):
        mk = ((tr.difficulty == dd) | (y == 0)).to_numpy(); r[f"ap_{dd}"] = round(100 * AP(y[mk], oof[mk]), 2)
    top = np.argsort(-oof)[:400]; fp = tr.iloc[top][y[top] == 0]
    r["top400_fp"] = int(len(fp)); r["top400_fp_lookalike"] = int((fp.family != "background").sum())
    return r


def blend(seed):
    lab = pd.read_csv("data/train/labels.csv")
    for base in ("base", "fe10", "fe10r9"):
        for sp in ("spec10", "spec10r9"):
            fb, fs = f"cache/fe10/oof_{base}_{seed}.parquet", f"cache/fe10/oof_{sp}_{seed}.parquet"
            if not (os.path.exists(fb) and os.path.exists(fs)): continue
            d = pd.read_parquet(fb).merge(pd.read_parquet(fs), on="TRANSACTION_HASH", suffixes=("_b", "_s")).merge(lab, on="TRANSACTION_HASH")
            y = d.is_suspicious.to_numpy()
            for w in (0.1, 0.2):
                o = (1 - w) * d.oof_b.rank(pct=True) + w * d.oof_s.rank(pct=True)
                r = score(d, y, o.to_numpy()); r.update(cfg=f"{base}+{sp}@{w}", fold_seed=seed)
                print(json.dumps(r), flush=True)
                with open("logs/fe10_results.jsonl", "a") as fh: fh.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(__file__), ".."))
    if sys.argv[1] == "build":
        t = time.time(); a = build("data/train"); b = build("data/test")
        a, b = finalize(a, b); a.to_parquet("cache/fe10/train_fe10.parquet"); b.to_parquet("cache/fe10/test_fe10.parquet")
        print("cols", a.shape[1] - 1, "secs", round(time.time() - t), flush=True)
    elif sys.argv[1] == "eval": evaluate(sys.argv[2], int(sys.argv[3]))
    elif sys.argv[1] == "blend": blend(int(sys.argv[2]))
