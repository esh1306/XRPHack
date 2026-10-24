"""Write a validated submission CSV: TRANSACTION_HASH,score[,reason] (rank-normalised, no ties).
Usage: make_submission.py PRED_PARQUET TEST_DIR OUT_CSV [FEATURE_PREFIX]  -> adds a short 'reason' (dominant pattern)."""
import sys, os, numpy as np, pandas as pd
pred, test_dir, out = sys.argv[1], sys.argv[2], sys.argv[3]
p = pd.read_parquet(pred)
tx = pd.read_parquet(f"{test_dir}/transactions.parquet", columns=["TRANSACTION_HASH"])
assert p.TRANSACTION_HASH.is_unique and set(p.TRANSACTION_HASH) == set(tx.TRANSACTION_HASH), "ID mismatch"
p = p.sort_values(["score", "TRANSACTION_HASH"], ascending=[True, False]).reset_index(drop=True)
p["score"] = (np.arange(len(p)) + 1) / len(p)
cols = ["TRANSACTION_HASH", "score"]
pref = sys.argv[4] if len(sys.argv) > 4 else None
PAT = {"chain_len": "multi-hop relay chain", "sg_k_first_6h": "scatter-gather", "cyc3_6h": "3-cycle",
       "in_match_cnt3600": "fast pass-through (in->out amount match)", "r_sync_cp10m": "fan-in burst",
       "s_sync_cp10m": "fan-out burst", "peel_run": "peel chain", "s_ooc_amt_z": "out-of-character amount",
       "cmp900_n": "dense burst episode", "sync_amt_2m": "synchronised amounts"}
if pref and all(os.path.exists(f"{pref}_{k}.parquet") for k in ("f3", "f4", "f6")):
    f = pd.read_parquet(f"{pref}_f3.parquet", columns=["TRANSACTION_HASH", "cmp900_n"])
    for k in ("f4", "f6"):
        c = ["TRANSACTION_HASH"] + [x for x in PAT if x in pd.read_parquet(f"{pref}_{k}.parquet").columns]
        f = f.merge(pd.read_parquet(f"{pref}_{k}.parquet", columns=c), on="TRANSACTION_HASH")
    r = f.set_index("TRANSACTION_HASH")[list(PAT)].rank(pct=True)
    p["reason"] = r.idxmax(axis=1).map(PAT).reindex(p.TRANSACTION_HASH).to_numpy()
    p.loc[p.score < 0.9, "reason"] = "low risk"
    cols.append("reason")
p[cols].to_csv(out, index=False, float_format="%.8f")
print(out, len(p), "rows; unique scores:", p.score.nunique(), "| size MiB", round(os.path.getsize(out) / 2**20, 2))
