import sys, time, numpy as np, pandas as pd
sys.path.insert(0, "src")
from pipeline import *
lab = pd.read_csv("data/train/labels.csv")
tr = load_feats("cache/train").merge(lab, on="TRANSACTION_HASH")
comp = pd.read_parquet("cache/train_comp.parquet").set_index("TRANSACTION_HASH").loc[tr.TRANSACTION_HASH].reset_index()
y = tr.is_suspicious.to_numpy(); groups = tr.scenario_id.fillna(tr.TRANSACTION_HASH).to_numpy()
FS = [list(StratifiedGroupKFold(5, shuffle=True, random_state=s).split(tr, y, groups)) for s in (42, 7)]
f1 = [c for c in pd.read_parquet("cache/train_f.parquet").columns if c not in DROP]
f2 = [c for c in pd.read_parquet("cache/train_f2.parquet").columns if c != "TRANSACTION_HASH"]
f3 = [c for c in pd.read_parquet("cache/train_f3.parquet").columns if c != "TRANSACTION_HASH"]
def rep(name, os_):
    aps = [100*AP(y,o) for o in os_]; o = os_[0]
    s = f"{name}: AP={np.mean(aps):.2f} ({' / '.join(f'{a:.2f}' for a in aps)})"
    for d in ["easy", "medium", "hard"]:
        m = (tr.difficulty == d) | (y == 0); s += f" {d}={100*AP(y[m],o[m]):.2f}"
    print(s, flush=True)
def run(name, cols, kind="lgb"):
    os_ = [cv(tr[cols], y, f, kind)[0] for f in FS]; rep(name, os_); return os_
meta = meta_of("data/train", tr.TRANSACTION_HASH)
oB = run("B base(no week feats)", f1)
oE = run("E +sessions/components", f1 + f3)
oF = run("F +sessions +flow", f1 + f2 + f3)
best = oE if np.mean([AP(y,o) for o in oE]) >= np.mean([AP(y,o) for o in oF]) else oF
bc = f1 + f3 if best is oE else f1 + f2 + f3
os2 = []
for o, f in zip(best, FS):
    X2 = pd.concat([tr[bc].reset_index(drop=True), propagate(meta, o, comp)], axis=1)
    os2.append(cv(X2, y, f)[0])
rep("G stage2 (+account/pair/temporal/component propagation)", os2)
np.save("cache/oof_best_s1.npy", best[0]); np.save("cache/oof_s2.npy", os2[0])
print("best stage1 cols:", "E" if best is oE else "F")
