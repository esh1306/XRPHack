"""Two-stage cascade: code-feature GBM ranks everything, Jev features re-rank the top candidates."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold

PARAMS = dict(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=20,
              l2_regularization=1.0, random_state=0)


def gbm(**overrides) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(**{**PARAMS, **overrides})


def oof_predict(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, folds: int = 5) -> np.ndarray:
    oof = np.zeros(len(y))
    for fit, check in StratifiedGroupKFold(folds, shuffle=True, random_state=0).split(X, y, groups):
        oof[check] = gbm().fit(X.iloc[fit], y[fit]).predict_proba(X.iloc[check])[:, 1]
    return oof


def ap(y, s) -> float:
    return 100 * average_precision_score(y, s)


def precision_recall_top(y, s, frac=0.01) -> tuple[float, float]:
    k = max(1, int(round(len(y) * frac)))
    top = np.argsort(-s, kind="stable")[:k]
    hits = y[top].sum()
    return 100 * hits / k, 100 * hits / max(1, y.sum())


def importances(model, X: pd.DataFrame, y: np.ndarray, n: int = 15) -> list[tuple[str, float]]:
    from sklearn.inspection import permutation_importance
    sample = np.random.default_rng(0).choice(len(y), size=min(len(y), 6000), replace=False)
    r = permutation_importance(model, X.iloc[sample], y[sample], scoring="average_precision",
                               n_repeats=3, random_state=0)
    order = np.argsort(-r.importances_mean)[:n]
    return [(X.columns[i], round(float(r.importances_mean[i]), 4)) for i in order]


def cascade(stage1: np.ndarray, stage2: np.ndarray | None, candidates: np.ndarray, alpha: float) -> np.ndarray:
    """Candidates sit above everyone else; inside, blend stage-2 and stage-1 ranks."""
    s1 = stage1 / max(stage1.max(), 1e-12)
    final = 0.5 * s1
    if stage2 is not None and candidates.any():
        inner = alpha * _rank01(stage2[candidates]) + (1 - alpha) * _rank01(stage1[candidates])
        final[candidates] = 0.5 + 0.5 * inner
    return final


def _rank01(x: np.ndarray) -> np.ndarray:
    if len(x) <= 1:
        return np.ones_like(x, dtype=float)
    return pd.Series(x).rank(method="average").to_numpy() / len(x)


def top_mask(scores: np.ndarray, k: int) -> np.ndarray:
    mask = np.zeros(len(scores), bool)
    mask[np.argsort(-scores, kind="stable")[:k]] = True
    return mask
