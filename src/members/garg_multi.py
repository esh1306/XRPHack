"""GARG-AML features + v3 features: LightGBM seeds 1,2 and XGBoost seed 42 (seed-42 LightGBM is in garg.py).
Saves each member and a fold-ranked average 'garg_ens' of all four."""
import json, time
from common import *
t0 = time.time()
tr, te, y, folds, cols = train_setup()
gtr = pd.read_parquet(f"{MEM}/garg_feats_train.parquet").set_index("TRANSACTION_HASH").loc[tr.TRANSACTION_HASH].reset_index(drop=True)
gte = pd.read_parquet(f"{MEM}/garg_feats_test.parquet").set_index("TRANSACTION_HASH").loc[te.TRANSACTION_HASH].reset_index(drop=True)
for c in gtr.columns:
    if c.startswith("garg_all"): gtr[c] = gtr[c].rank(pct=True); gte[c] = gte[c].rank(pct=True)
Xtr = pd.concat([tr[cols].reset_index(drop=True), gtr], axis=1); Xte = pd.concat([te[cols].reset_index(drop=True), gte], axis=1)
def foldrank(o):
    r = np.zeros(len(o))
    for a, b in folds: r[b] = rank01(o[b])
    return r
O = [foldrank(pd.read_parquet(f"{MEM}/garg_v3feat_lgb_oof.parquet").oof.to_numpy())]
P = [rank01(pd.read_parquet(f"{MEM}/garg_v3feat_lgb_test.parquet").score.to_numpy())]
for kind, seed in (("lgb", 1), ("lgb", 2), ("xgb", 42)):
    o, p = cv(Xtr, y, folds, kind, seed, Xte); name = f"garg_v3feat_{kind}{seed}"
    save(name, tr.TRANSACTION_HASH, o, te.TRANSACTION_HASH, p); O.append(foldrank(o)); P.append(rank01(p))
    print(name, report(tr, y, foldrank(o)), f"{time.time()-t0:.0f}s", flush=True)
save("garg_ens", tr.TRANSACTION_HASH, np.mean(O, 0), te.TRANSACTION_HASH, np.mean(P, 0))
print("garg_ens", report(tr, y, np.mean(O, 0)), flush=True)
