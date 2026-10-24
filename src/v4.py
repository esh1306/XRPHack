"""v4 level-3 stacker.
Inputs: v3 final OOF (cache/oof_final.parquet) and v3 test prediction.
Adds (a) a medium+hard specialist LightGBM (easy positives removed from training, look-alike legit
scenarios up-weighted x3, f2,f3,f4,f6 features, 2 seeds) and (b) component/episode-smoothed versions
of the v3 score (mean/max/size/within-rank over cmp120/cmp900/flow480/flow3600/flow14400, computed
per split). A shallow LightGBM stacker is trained on these with grouped CV; final = 0.5*v3 + 0.5*stack.
Optional extra members: cache/members/<name>_oof.parquet + _test.parquet (TRANSACTION_HASH, score);
each is added greedily only if grouped-CV AP of the stacker improves (env MEMBERS=1 to enable).
Usage: v4.py <train_prefix> <test_prefix> <v3_pred.parquet> <out.parquet>
"""
import sys, os, glob, json, time, numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, os.path.dirname(__file__))
from pipeline import load_feats, lgb_params, rank01, AP, DROP
from sklearn.model_selection import StratifiedGroupKFold
from joblib import Parallel, delayed
trp, tep, v3p, out = sys.argv[1:5]
lab = pd.read_csv("data/train/labels.csv")
oo = pd.read_parquet("cache/oof_final.parquet").merge(lab, on="TRANSACTION_HASH")
H = oo.TRANSACTION_HASH.to_numpy()
y = oo.is_suspicious.to_numpy(); diff = oo.difficulty.to_numpy()
groups = oo.scenario_id.fillna(oo.TRANSACTION_HASH).to_numpy()
def rep(s):
    r = [round(100 * AP(y, s), 2)]
    for d in ["easy", "medium", "hard"]:
        m = (diff == d) | (y == 0); r.append(round(100 * AP(y[m], s[m]), 2))
    return r
log = {}
# ---- (a) specialist
FE = ("f2", "f3", "f4", "f6")
tr = load_feats(trp, FE).set_index("TRANSACTION_HASH").loc[H].reset_index()
te = load_feats(tep, FE)
cols = [c for c in tr.columns if c not in DROP and c in te.columns]
mh = (y == 1) & np.isin(diff, ["medium", "hard"])
look = (y == 0) & oo.scenario_id.notna().to_numpy()
def spec(seed):
    o = np.zeros(len(y)); its = []
    X = tr[cols]
    for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=42).split(tr, mh.astype(int), groups):
        a = a[~((y[a] == 1) & ~mh[a])]; bb = b[~((y[b] == 1) & ~mh[b])]
        p = lgb_params(seed); p["n_jobs"] = 1
        m = lgb.train(p, lgb.Dataset(X.iloc[a], mh[a].astype(int), weight=np.where(look[a], 3.0, 1.0)), 2000,
                      valid_sets=[lgb.Dataset(X.iloc[bb], mh[bb].astype(int))], callbacks=[lgb.early_stopping(150, verbose=False)])
        o[b] = m.predict(X.iloc[b]); its.append(m.best_iteration)
    keep = ~((y == 1) & ~mh)
    p = lgb_params(seed); p["n_jobs"] = 1
    m = lgb.train(p, lgb.Dataset(X[keep], mh[keep].astype(int), weight=np.where(look[keep], 3.0, 1.0)), max(50, int(np.mean(its))))
    return o, m.predict(te[cols]), its
t0 = time.time(); sp = out.replace(".parquet", "_spec.npz")
if os.environ.get("REUSE_SPEC") == "1" and os.path.exists(sp):
    z = np.load(sp); spec_oof, spec_te = z["oof"], z["te"]; print("reused specialist", sp)
else:
    R = Parallel(n_jobs=2)(delayed(spec)(s) for s in (42, 7))
    spec_oof = np.mean([rank01(r[0]) for r in R], 0); spec_te = np.mean([rank01(r[1]) for r in R], 0)
    log["spec_iters"] = [r[2] for r in R]; np.savez(sp, oof=spec_oof, te=spec_te)
    print("specialist done", round(time.time() - t0), "s; iters", log["spec_iters"], flush=True)
# ---- (b) component-smoothed scores
def comp_feats(r0, comp):
    F = {"r0": r0}
    for c in ["cmp120", "cmp900", "flow480", "flow3600", "flow14400"]:
        g = pd.Series(r0).groupby(comp[c].to_numpy())
        F[c + "_mean"] = g.transform("mean").to_numpy(); F[c + "_max"] = g.transform("max").to_numpy()
        F[c + "_n"] = g.transform("size").to_numpy(); F[c + "_rk"] = g.rank(pct=True).to_numpy()
    return pd.DataFrame(F)
r0 = rank01(oo.oof.to_numpy())
v3 = pd.read_parquet(v3p).set_index("TRANSACTION_HASH").loc[te.TRANSACTION_HASH]
t0r = rank01(v3.score.to_numpy())
Xtr = comp_feats(r0, pd.read_parquet(trp + "_comp.parquet").set_index("TRANSACTION_HASH").loc[H]).assign(spec=spec_oof)
Xte = comp_feats(t0r, pd.read_parquet(tep + "_comp.parquet").set_index("TRANSACTION_HASH").loc[te.TRANSACTION_HASH]).assign(spec=spec_te)
# shift robustness: these stacker inputs shift train-vs-test (KS 0.06-0.22), so percentile-rank within split
SHIFT_STACK = ["cmp900_mean", "cmp900_max", "cmp900_n", "flow14400_mean", "flow14400_n", "flow14400_max"]
for c in SHIFT_STACK: Xtr[c] = rank01(Xtr[c].to_numpy()); Xte[c] = rank01(Xte[c].to_numpy())
# optional members
if os.environ.get("MEMBERS", "0") == "1":
    for f in sorted(glob.glob("cache/members/*_oof.parquet")):
        name = os.path.basename(f)[:-12]
        ft = f.replace("_oof.parquet", "_test.parquet")
        if name in ("v3", "v4") or not os.path.exists(ft): continue
        mo = pd.read_parquet(f).set_index("TRANSACTION_HASH"); mt = pd.read_parquet(ft).set_index("TRANSACTION_HASH")
        pick = lambda d: "score" if "score" in d.columns else [c for c in d.columns if c != "TRANSACTION_HASH" and pd.api.types.is_numeric_dtype(d[c])][0]
        sc, sct = pick(mo), pick(mt)
        if not (set(H) <= set(mo.index) and set(te.TRANSACTION_HASH) <= set(mt.index)):
            print("skip member", name, "(hash coverage)"); continue
        Xtr["m_" + name] = rank01(mo.loc[H, sc].to_numpy()); Xte["m_" + name] = rank01(mt.loc[te.TRANSACTION_HASH, sct].to_numpy())
P = dict(objective="binary", learning_rate=0.03, num_leaves=7, min_data_in_leaf=100, feature_fraction=0.8,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=10, verbose=-1, n_jobs=1)
def stack_cv(cs, fs):
    o = np.zeros(len(y))
    for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=fs).split(Xtr, y, groups):
        o[b] = lgb.train({**P, "seed": fs}, lgb.Dataset(Xtr.iloc[a][cs], y[a]), 300).predict(Xtr.iloc[b][cs])
    return 0.5 * r0 + 0.5 * rank01(o)
def score(cs): return np.mean([AP(y, stack_cv(cs, fs)) for fs in (42, 7)])
base_cols = [c for c in Xtr.columns if not c.startswith("m_")]
cs = list(base_cols); best = score(cs)
for c in [c for c in Xtr.columns if c.startswith("m_")]:
    s = score(cs + [c]); print("member", c, "cv", round(100 * s, 2), "vs", round(100 * best, 2), flush=True)
    if s > best + 0.001: cs.append(c); best = s  # require >= +0.1 AP to keep a member
log["cols"] = cs
print("v3 OOF", rep(r0)); log["v3"] = rep(r0)
for fs in (42, 7):
    f = stack_cv(cs, fs); log[f"v4_fs{fs}"] = rep(f); print("v4 stack fold-seed", fs, rep(f), flush=True)
    if fs == 42: v4_oof = f
st = lgb.train({**P, "seed": 42}, lgb.Dataset(Xtr[cs], y), 300).predict(Xte[cs])
final = 0.5 * t0r + 0.5 * rank01(st)
pd.DataFrame({"TRANSACTION_HASH": te.TRANSACTION_HASH.to_numpy(), "score": final}).to_parquet(out, index=False)
os.makedirs("cache/members", exist_ok=True)
if os.environ.get("MEMBERS", "0") != "1":  # only the self-contained v4 is published as a member
    for nm, hh, sc_ in (("v4_oof", H, v4_oof), ("v4_test", te.TRANSACTION_HASH.to_numpy(), final),
                        ("v3_oof", H, r0), ("v3_test", te.TRANSACTION_HASH.to_numpy(), t0r)):
        pd.DataFrame({"TRANSACTION_HASH": hh, "score": sc_}).to_parquet(f"cache/members/{nm}.parquet", index=False)
json.dump(log, open(out.replace(".parquet", "_cv.json"), "w"), indent=1)
print("wrote", out)
