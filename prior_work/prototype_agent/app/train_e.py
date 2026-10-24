"""Submission J: rank blend of gradient boosting (5 seeds) + ExtraTrees + logistic regression on code + motif
(incl. tightness) + account-level aggregate features.
python -m app.train_e --train TRAIN_DIR --test TEST_DIR --out predictions.csv   (no API keys; ~2-3 min on M3 Pro)
Fits run in parallel processes (joblib) across all cores. Deterministic: fixed seeds 0-4, random_state=0 CV."""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import StratifiedGroupKFold

from .account_agg import account_aggregates
from .cluster_prop import cluster_domain_features, propagate_cluster_scores
from .data import cv_groups, labels_array, load_split
from .features import build_features
from .model import ap, precision_recall_top
from .motif import motif_features

PARAMS = dict(max_iter=800, learning_rate=0.03, max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=5.0,
              max_features=0.5)  # regularised: grouped CV 56.98 vs 56.38
SEEDS = [0, 1, 2, 3, 4]
if os.environ.get("TRAIN_PARAMS"):  # experiment override, e.g. TRAIN_PARAMS='{"min_samples_leaf":20}'
    PARAMS = {**PARAMS, **json.loads(os.environ["TRAIN_PARAMS"])}


def design(directory, name):
    sp = load_split(directory, name)
    X, _, _ = build_features(sp.edges)
    B = pd.concat([X, motif_features(sp.edges).set_axis(X.index)], axis=1)
    C = pd.concat([B, account_aggregates(sp.edges, B), cluster_domain_features(sp.edges)], axis=1)
    return sp, C


BLEND = {"gbm": 0.8, "et": 0.1, "lr": 0.1}  # rank-blend weights chosen by grouped CV (56.88 vs 56.49 GBM alone)


def _model(kind, seed):
    if kind == "gbm":
        return HistGradientBoostingClassifier(**PARAMS, random_state=seed)
    if kind == "et":
        return make_pipeline(SimpleImputer(strategy="median"), ExtraTreesClassifier(
            n_estimators=400, min_samples_leaf=3, max_features=0.3, n_jobs=2, random_state=seed,
            class_weight="balanced_subsample"))
    return make_pipeline(SimpleImputer(strategy="median"), QuantileTransformer(n_quantiles=200, output_distribution="normal"),
                         LogisticRegression(C=0.3, max_iter=2000, class_weight="balanced"))


def _fit_predict(Xa, ya, Xb, seed, kind="gbm"):
    return _model(kind, seed).fit(Xa, ya).predict_proba(Xb)[:, 1]


def _ensemble(Xa, ya, Xb, run):
    """5-seed GBM + ExtraTrees + logistic regression, combined by rank."""
    jobs = [("gbm", s) for s in SEEDS] + [("et", 0), ("lr", 0)]
    preds = run(delayed(_fit_predict)(Xa, ya, Xb, s, k) for k, s in jobs)
    by = {"gbm": np.mean(preds[:len(SEEDS)], axis=0), "et": preds[-2], "lr": preds[-1]}
    rank = lambda x: pd.Series(x).rank(pct=True).to_numpy()
    return sum(w * rank(by[k]) for k, w in BLEND.items())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--out", default="predictions.csv")
    p.add_argument("--cv", action="store_true", help="also report grouped 5-fold CV AP")
    a = p.parse_args()
    sp, D = design(a.train, "train")
    y = labels_array(sp)
    run = Parallel(n_jobs=-1)
    if a.cv:
        folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(D, y, cv_groups(sp)))
        oof = np.zeros(len(y))
        for f, c in folds:
            oof[c] = _ensemble(D.iloc[f], y[f], D.iloc[c], run)
        oof_prop = propagate_cluster_scores(sp.edges, oof)
        print(f"CV AP {ap(y, oof_prop):.2f} (stage1 {ap(y, oof):.2f})  P/R@1% {precision_recall_top(y, oof_prop)}")
    sp_test, T = design(a.test, "test")
    T = T.reindex(columns=D.columns)
    score = propagate_cluster_scores(sp_test.edges, _ensemble(D, y, T, run))
    pd.DataFrame({"TRANSACTION_HASH": T.index, "score": np.round(score, 6)}).to_csv(a.out, index=False)
    print(f"wrote {len(T):,} rows → {a.out}")


if __name__ == "__main__":
    main()
