"""Adversarial validation: can a model tell train from another split? Prints AUC and top separating features."""
import sys, numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from pipeline import load_feats, DROP
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
ks = sys.argv[2].split(",") if len(sys.argv) > 2 else ("f2", "f3")
def lf(p):
    a = pd.read_parquet(f"{p}_f.parquet")
    for k in ks: a = a.merge(pd.read_parquet(f"{p}_{k}.parquet"), on="TRANSACTION_HASH")
    return a
a, b = lf("cache/train"), lf(sys.argv[1] if len(sys.argv) > 1 else "cache/validation")
cols = [c for c in a.columns if c not in DROP]
X = pd.concat([a[cols], b[cols]], ignore_index=True); y = np.r_[np.zeros(len(a)), np.ones(len(b))]
oof = np.zeros(len(y)); imp = 0
for tr, va in StratifiedKFold(3, shuffle=True, random_state=42).split(X, y):
    m = lgb.train(dict(objective="binary", learning_rate=0.1, num_leaves=31, verbose=-1, n_jobs=4, seed=42), lgb.Dataset(X.iloc[tr], y[tr]), 150)
    oof[va] = m.predict(X.iloc[va]); imp = imp + m.feature_importance("gain")
print("adv AUC", round(roc_auc_score(y, oof), 4))
s = pd.Series(imp, cols).sort_values(ascending=False); print((s / s.sum()).head(25).round(3).to_string())
uni = {c: abs(roc_auc_score(y, X[c].fillna(-999)) - 0.5) for c in cols}
print(pd.Series(uni).sort_values(ascending=False).head(25).round(3).to_string())
