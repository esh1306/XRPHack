"""v9 / v9b: closed-cluster score propagation on top of the v7x no-head test scores.
Usage: v9_prop.py BASE_PRED_PARQUET TEST_DIR OUT_PARQUET [--blend-prototype W_OURS]
  v9  : rank(base) -> propagate_cluster_scores from the earlier prototype (multi-scale closed first-time-payment clusters).
  v9b : W_OURS*rank(base) + (1-W_OURS)*rank(prototype stage-1 ensemble trained on data/train), then propagation (W_OURS=0.7).
Propagation and the prototype stage-1 model come from prior_work/prototype_agent (MIT, Copyright (c) 2026 Sahid; see PRIOR_WORK.md).
CV (StratifiedGroupKFold(5) by scenario_id, seed 42; all/easy/med/hard AP): v7x no-head 63.79/84.58/9.73/0.74,
v9 64.89/85.54/12.65/0.76, v9b 65.64/85.41/13.99/0.76."""
import sys, os, numpy as np, pandas as pd
from scipy.stats import rankdata
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "prior_work", "prototype_agent"))
from app.data import load_split
from app.cluster_prop import propagate_cluster_scores

base, test_dir, out = sys.argv[1], sys.argv[2], sys.argv[3]
w = float(sys.argv[sys.argv.index("--blend-prototype") + 1]) if "--blend-prototype" in sys.argv else None
rk = lambda x: rankdata(x) / len(x)
et = load_split(test_dir, "test").edges
b = pd.read_parquet(base)[["TRANSACTION_HASH", "score"]]
s = et[["hash"]].merge(b, left_on="hash", right_on="TRANSACTION_HASH", how="left").score.to_numpy()
assert not np.isnan(s).any(), "base scores missing hashes"
s = rk(s)
if w is not None:
    from joblib import Parallel
    from app.train_e import design, _ensemble
    from app.data import labels_array
    sp, D = design("data/train", "train"); y = labels_array(sp)
    _, T = design(test_dir, "test"); T = T.reindex(columns=D.columns)
    assert (T.index.to_numpy() == et.hash.to_numpy()).all()
    ft = _ensemble(D, y, T, Parallel(n_jobs=int(os.environ.get("PROTO_JOBS", "4"))))
    s = rk(w * s + (1 - w) * rk(ft))
p = propagate_cluster_scores(et, s)
pd.DataFrame({"TRANSACTION_HASH": et.hash.values, "score": p}).to_parquet(out)
print(out, len(p))
