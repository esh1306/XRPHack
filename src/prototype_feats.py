"""v10: take the prototype's (prior_work/prototype_agent, MIT) design-matrix columns c_*, m_*, acs_*, acr_* (row order = prototype/cache/edges_<split>.parquet),
keep those with train/test KS < 0.1, write cache/f11fr/<split>_fr.parquet keyed by TRANSACTION_HASH. Label-free, per split."""
import pandas as pd, numpy as np
from scipy.stats import ks_2samp
D = {s: pd.read_parquet(f"prototype/cache/D_{s}.parquet") for s in ("train", "test")}
E = {s: pd.read_parquet(f"prototype/cache/edges_{s}.parquet").hash.to_numpy() for s in ("train", "test")}
cols = [c for c in D["train"].columns if c.split("_")[0] in ("c", "m", "acs", "acr")]
keep, rep = [], []
for c in cols:
    a, b = D["train"][c].astype(float).fillna(-1), D["test"][c].astype(float).fillna(-1)
    if a.nunique() < 2: rep.append((c, None, "const")); continue
    ks = ks_2samp(a, b).statistic; rep.append((c, round(ks, 4), "keep" if ks < 0.1 else "drop"))
    if ks < 0.1: keep.append(c)
pd.DataFrame(rep, columns=["col", "ks", "action"]).to_csv("logs/prototype_ks.csv", index=False)
for s in ("train", "test"):
    out = D[s][keep].astype("float32").copy(); out.columns = ["fr_" + c for c in keep]; out.insert(0, "TRANSACTION_HASH", E[s])
    out.to_parquet(f"cache/f11fr/{s}_fr.parquet")
print("kept", len(keep), "of", len(cols), {p: sum(c.split('_')[0] == p for c in keep) for p in ("c", "m", "acs", "acr")})
