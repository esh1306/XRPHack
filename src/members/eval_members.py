"""Evaluate saved members: alone and rank-blended with the v3 OOF (pooled and fold-ranked variants).
Usage: python eval_members.py name1 [name2 ...]  (multi-name: also tries the equal-weight combo of members)"""
import sys, json
from common import *
lab = pd.read_csv(f"{ROOT}/data/train/labels.csv")
y_all = lab.set_index("TRANSACTION_HASH")
tr = pd.read_parquet(f"{MEM}/v3_oof_snapshot.parquet").merge(lab, on="TRANSACTION_HASH")
y = tr.is_suspicious.to_numpy(); groups = tr.scenario_id.fillna(tr.TRANSACTION_HASH).to_numpy()
# NB folds must match train_setup order: rebuild from the same feature-table row order
tr2, _, y2, folds, _ = train_setup()
tr = tr.set_index("TRANSACTION_HASH").loc[tr2.TRANSACTION_HASH].reset_index(); y = y2
def foldrank(o):
    r = np.zeros(len(o))
    for a, b in folds: r[b] = rank01(o[b])
    return r
base = tr.oof.to_numpy()
print("v3 pooled", report(tr, y, base), "v3 fold-ranked", report(tr, y, foldrank(base)))
s1 = np.load(f"{ROOT}/cache/pred_test_v3_s1_oofs.npy")  # v3 stage-1 member OOFs (lgb42,lgb1,lgb2,xgb42,cat42), same row order
print("v3 s1 lgb42 fold-ranked (no-GARG baseline)", report(tr, y, foldrank(s1[0])))
print("v3 s1 lgb42+lgb1+lgb2+xgb42 fold-ranked mean (no-GARG ensemble baseline)", report(tr, y, np.mean([foldrank(s1[i]) for i in range(4)], 0)))
base = foldrank(base)  # compare like with like: everything fold-ranked
mem = {}
for n in sys.argv[1:]:
    o = pd.read_parquet(f"{MEM}/{n}_oof.parquet").set_index("TRANSACTION_HASH").loc[tr.TRANSACTION_HASH, "oof"].to_numpy()
    mem[n] = foldrank(o)
if len(mem) > 1: mem["+".join(mem)] = np.mean([rank01(v) for v in mem.values()], 0)
out = {}
for n, o in mem.items():
    out[n] = {"alone": report(tr, y, o)}
    for w in (0.1, 0.2, 0.3):
        out[n][f"w{w}"] = report(tr, y, (1 - w) * rank01(base) + w * rank01(o))
    print(n, out[n], flush=True)
json.dump(out, open(f"{MEM}/eval_{'_'.join(sys.argv[1:])[:80]}.json", "w"), indent=1)
