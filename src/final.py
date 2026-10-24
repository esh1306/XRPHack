"""Final model: CV report + full fit + test scoring.
Stage 1: LightGBM (3 seeds) + XGBoost + CatBoost on intrinsic, account, temporal-motif, graph, flow and
candidate-scenario (session/component) features -> rank blend.
Stage 2: propagate stage-1 scores over accounts, pairs, temporal neighbours and candidate scenarios
(OOF on train, stage-1 predictions on test) -> LightGBM (3 seeds) + XGBoost -> rank blend.
Usage: python src/final.py TEST_PREFIX TEST_DIR OUT_PARQUET   (e.g. cache/test data/test cache/pred_test.parquet)"""
import sys, os, time, json, numpy as np, pandas as pd
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from pipeline import *
t0 = time.time()
test_prefix, test_dir, out = sys.argv[1], sys.argv[2], sys.argv[3]
MODELS = [m for m in (sys.argv[4] if len(sys.argv) > 4 else "lgb42,lgb1,lgb2,xgb42,cat42").split(",")]
lab = pd.read_csv("data/train/labels.csv")
tr = load_feats("cache/train").merge(lab, on="TRANSACTION_HASH")
te = load_feats(test_prefix)
y = tr.is_suspicious.to_numpy(); groups = tr.scenario_id.fillna(tr.TRANSACTION_HASH).to_numpy()
FOLD_SEED = int(os.environ.get("FOLD_SEED", SEED))  # CV fold seed (seed-robustness checks); models keep their own seeds
folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=FOLD_SEED).split(tr, y, groups))
cols = [c for c in te.columns if c not in DROP]
comp_tr = pd.read_parquet("cache/train_comp.parquet").set_index("TRANSACTION_HASH").loc[tr.TRANSACTION_HASH].reset_index()
comp_te = pd.read_parquet(f"{test_prefix}_comp.parquet").set_index("TRANSACTION_HASH").loc[te.TRANSACTION_HASH].reset_index()
meta_tr, meta_te = meta_of("data/train", tr.TRANSACTION_HASH), meta_of(test_dir, te.TRANSACTION_HASH)
log = {}

def _one(Xtr, Xte, m):
    kind, seed = m[:3], int(m[3:])
    return cv(Xtr, y, folds, kind, seed, Xte)


def stage(Xtr, Xte, models, tag):
    # one single-threaded process per model: robust on a shared/oversubscribed CPU (NJOBS=1 recommended)
    from joblib import Parallel, delayed
    res = Parallel(n_jobs=int(os.environ.get("PAR", len(models))))(delayed(_one)(Xtr, Xte, m) for m in models)
    oofs, tes = [], []
    for m, (o, p) in zip(models, res):
        oofs.append(rank01(o)); tes.append(rank01(p))
        log[f"{tag}_{m}"] = round(100 * AP(y, o), 2); print(tag, m, log[f"{tag}_{m}"], f"{time.time()-t0:.0f}s", flush=True)
    bo, bt = np.mean(oofs, 0), np.mean(tes, 0)
    log[f"{tag}_blend"] = round(100 * AP(y, bo), 2); print(tag, "blend", log[f"{tag}_blend"], flush=True)
    np.save(out.replace(".parquet", f"_{tag}_oofs.npy"), np.array(oofs)); np.save(out.replace(".parquet", f"_{tag}_tes.npy"), np.array(tes))
    return bo, bt

o1, p1 = stage(tr[cols], te[cols], MODELS, "s1")
X2tr = pd.concat([tr[cols].reset_index(drop=True), propagate(meta_tr, o1, comp_tr)], axis=1)
X2te = pd.concat([te[cols].reset_index(drop=True), propagate(meta_te, p1, comp_te)], axis=1)
o2, p2 = stage(X2tr, X2te, [m for m in MODELS if not m.startswith("cat")], "s2")
for w in (0.0, 0.3, 0.5):
    log[f"final_w{w}"] = round(100 * AP(y, (1 - w) * o2 + w * o1), 2)
print(log)
W = max((0.0, 0.3, 0.5), key=lambda w: log[f"final_w{w}"])
final = (1 - W) * p2 + W * p1
for d in ["easy", "medium", "hard"]:
    m = (tr.difficulty == d) | (y == 0); log[f"oof_{d}"] = round(100 * AP(y[m], ((1 - W) * o2 + W * o1)[m]), 2)
log["weight_stage1"] = W
pd.DataFrame({"TRANSACTION_HASH": te.TRANSACTION_HASH, "score": final}).to_parquet(out)
pd.DataFrame({"TRANSACTION_HASH": tr.TRANSACTION_HASH, "oof": (1 - W) * o2 + W * o1}).to_parquet(os.environ.get("OOF_OUT", "cache/oof_final.parquet"))
json.dump(log, open(out.replace(".parquet", "_cv.json"), "w"), indent=1)
print(log, f"{time.time()-t0:.0f}s")
