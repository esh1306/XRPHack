"""Candidate blend of v3 + members on fold-ranked OOF; writes cache/members/blend_candidate_{oof,test}.parquet and
cache/members/submission_members_{TAG}.csv. Weights picked on the same OOF (slightly optimistic).
Usage: python blend_candidate.py member1 member2 ...  (v3 always included)"""
import sys, itertools, json
from common import *
tr, te, y, folds, cols = train_setup()
def fr(o):
    r = np.zeros(len(o))
    for a, b in folds: r[b] = rank01(o[b])
    return r
names = ["v3"] + sys.argv[1:]
O = [fr(v3_oof(tr.TRANSACTION_HASH))]
P = [rank01(pd.read_parquet(f"{ROOT}/cache/pred_test_v3.parquet").set_index("TRANSACTION_HASH").loc[te.TRANSACTION_HASH, "score"].to_numpy())]
for n in sys.argv[1:]:
    O.append(fr(pd.read_parquet(f"{MEM}/{n}_oof.parquet").set_index("TRANSACTION_HASH").loc[tr.TRANSACTION_HASH, "oof"].to_numpy()))
    P.append(rank01(pd.read_parquet(f"{MEM}/{n}_test.parquet").set_index("TRANSACTION_HASH").loc[te.TRANSACTION_HASH, "score"].to_numpy()))
O = [rank01(o) for o in O]
grid = [w for w in itertools.product(*[np.arange(0, 1.01, 0.1)] * len(names)) if abs(sum(w) - 1) < 1e-9]
res = sorted(((report(tr, y, sum(wi * o for wi, o in zip(w, O))), w) for w in grid), key=lambda r: -r[0][0])
print("v3 fold-ranked", report(tr, y, O[0]))
for r, w in res[:8]: print(dict(zip(names, np.round(w, 1))), r)
import os
best = res[0][1] if not os.environ.get("W") else tuple(float(x) for x in os.environ["W"].split(","))
TAG = os.environ.get("TAG", "candidate")
oof = sum(wi * o for wi, o in zip(best, O)); pt = sum(wi * p for wi, p in zip(best, P))
pt = rank01(pt + 1e-6 * P[0]) * 0.999 + 0.0005  # strictly inside (0,1); continuous ranks
assert len(pt) == 40101 and te.TRANSACTION_HASH.is_unique and np.isfinite(pt).all()
pd.DataFrame({"TRANSACTION_HASH": tr.TRANSACTION_HASH, "oof": oof}).to_parquet(f"{MEM}/blend_{TAG}_oof.parquet")
pd.DataFrame({"TRANSACTION_HASH": te.TRANSACTION_HASH, "score": pt}).to_parquet(f"{MEM}/blend_{TAG}_test.parquet")
pd.DataFrame({"TRANSACTION_HASH": te.TRANSACTION_HASH, "score": np.round(pt, 8)}).to_csv(f"{MEM}/submission_members_{TAG}.csv", index=False)
json.dump({"weights": dict(zip(names, [float(round(w, 1)) for w in best])), "oof_foldranked": report(tr, y, oof),
           "v3_foldranked": report(tr, y, O[0]), "n_unique_scores": int(pd.Series(np.round(pt, 8)).nunique())},
          open(f"{MEM}/blend_{TAG}.json", "w"), indent=1)
print(json.load(open(f"{MEM}/blend_{TAG}.json")))
