"""Shared helpers for extra ensemble members (read-only use of v3 caches + pipeline helpers)."""
import os, sys, numpy as np, pandas as pd
os.environ.setdefault("NJOBS", "2")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pipeline import load_feats, meta_of, rank01, DROP, SEED, cv  # noqa
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import average_precision_score as AP

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
MEM = os.path.join(ROOT, "cache", "members")
os.makedirs(MEM, exist_ok=True)


def train_setup():
    lab = pd.read_csv(f"{ROOT}/data/train/labels.csv")
    tr = load_feats(f"{ROOT}/cache/train").merge(lab, on="TRANSACTION_HASH")
    te = load_feats(f"{ROOT}/cache/test")
    y = tr.is_suspicious.to_numpy(); groups = tr.scenario_id.fillna(tr.TRANSACTION_HASH).to_numpy()
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(tr, y, groups))  # same folds as v3
    cols = [c for c in te.columns if c not in DROP]
    return tr, te, y, folds, cols


def v3_oof(hashes):
    d = pd.read_parquet(f"{MEM}/v3_oof_snapshot.parquet").set_index("TRANSACTION_HASH")
    return d.loc[hashes, "oof"].to_numpy()


def report(tr, y, p):
    out = [round(100 * AP(y, p), 2)]
    for d in ["easy", "medium", "hard"]:
        m = (tr.difficulty == d).to_numpy() | (y == 0); out.append(round(100 * AP(y[m], p[m]), 2))
    return out  # all / easy / medium / hard


def blend_report(tr, y, member_oof, name):
    base = rank01(v3_oof(tr.TRANSACTION_HASH)); mo = rank01(member_oof)
    res = {"alone": report(tr, y, member_oof)}
    for w in (0.0, 0.1, 0.2, 0.3):
        res[f"blend_w{w}"] = report(tr, y, (1 - w) * base + w * mo)
    print(name, res, flush=True)
    return res


def save(name, tr_hash, oof, te_hash, te_pred):
    pd.DataFrame({"TRANSACTION_HASH": tr_hash, "oof": oof}).to_parquet(f"{MEM}/{name}_oof.parquet")
    pd.DataFrame({"TRANSACTION_HASH": te_hash, "score": te_pred}).to_parquet(f"{MEM}/{name}_test.parquet")
