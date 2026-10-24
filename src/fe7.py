"""v4 account behaviour-change features (label-free, per split): dormant-then-burst, activity rank in the
account's own history, short-window in/out balance and dwell time from receipt to forward."""
import sys, warnings, numpy as np, pandas as pd
from collections import defaultdict
warnings.filterwarnings("ignore")
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from fe import load_split


def build(tx):
    S = tx.ACCOUNT.to_numpy(); R = tx.DEST.to_numpy(); T = tx.TIMESTAMP.to_numpy().astype(np.int64)
    V = np.nan_to_num(tx.usd.to_numpy().astype(float)); n = len(tx); F = {}
    ev = defaultdict(list); out_i = defaultdict(list); in_i = defaultdict(list)
    for i in range(n):
        ev[S[i]].append(i); ev[R[i]].append(i); out_i[S[i]].append(i); in_i[R[i]].append(i)
    for role, col in (("s", S), ("r", R)):
        gp = np.zeros(n); gr = np.zeros(n); n1 = np.zeros(n); n6 = np.zeros(n); p1 = np.zeros(n)
        rk = np.zeros(n); rkp = np.zeros(n); tot = np.zeros(n)
        for a in np.unique(col):
            ix = np.array(ev[a]); ix = ix[np.argsort(T[ix], kind="stable")]; t = T[ix]
            mine = ix[col[ix] == a]; tm = T[mine]
            g = np.diff(t).astype(float); med = np.median(g) + 1 if len(g) else 1e6
            pos = np.searchsorted(t, tm, "left")
            prev = np.where(pos > 0, tm - t[np.maximum(pos - 1, 0)], 1e7).astype(float)
            gp[mine] = np.log1p(prev); gr[mine] = np.log1p(prev) - np.log(med)
            n1[mine] = np.searchsorted(t, tm + 3600, "right") - pos - 1
            n6[mine] = np.searchsorted(t, tm + 21600, "right") - pos - 1
            p1[mine] = pos - np.searchsorted(t, tm - 3600, "left")
            rk[mine] = np.log1p(pos); rkp[mine] = (pos + 0.5) / len(t); tot[mine] = np.log1p(len(t))
        F[f"{role}_gap_prev"] = gp; F[f"{role}_gap_ratio"] = gr; F[f"{role}_next1h"] = n1; F[f"{role}_next6h"] = n6
        F[f"{role}_prev1h"] = p1; F[f"{role}_hist_rank"] = rk; F[f"{role}_hist_pct"] = rkp; F[f"{role}_ntot"] = tot
        F[f"{role}_dormant_burst"] = ((gr > np.log(10)) & (n6 >= 3)).astype(float)
        F[f"{role}_burst_score"] = np.maximum(gr, 0) * np.log1p(n6)
    # in/out balance within +-6h and dwell (sender: since last receipt; receiver: until next send)
    oi = {k: np.array(v)[np.argsort(T[v], kind="stable")] for k, v in out_i.items()}
    ii = {k: np.array(v)[np.argsort(T[v], kind="stable")] for k, v in in_i.items()}
    E = np.array([], int)
    cin = {k: np.concatenate([[0], np.cumsum(V[v])]) for k, v in ii.items()}
    cout = {k: np.concatenate([[0], np.cumsum(V[v])]) for k, v in oi.items()}
    dw = np.full(n, 16.0); dwr = np.full(n, 16.0); io_s = np.zeros(n); io_r = np.zeros(n); dwamt = np.full(n, 9.0)
    for i in range(n):
        x = ii.get(S[i], E)
        if len(x):
            t = T[x]; p = np.searchsorted(t, T[i], "right")
            if p > 0:
                dw[i] = np.log1p(T[i] - t[p - 1]); dwamt[i] = min(abs(V[x[p - 1]] - V[i]) / max(V[x[p - 1]], 1e-9), 9)
            q = np.searchsorted(t, T[i] - 21600, "left"); inn = cin[S[i]][p] - cin[S[i]][q]
            y = oi[S[i]]; ty = T[y]; a_, b_ = np.searchsorted(ty, T[i], "left"), np.searchsorted(ty, T[i] + 21600, "right")
            out = cout[S[i]][b_] - cout[S[i]][a_]; io_s[i] = np.log1p(inn) - np.log1p(out)
        y = oi.get(R[i], E)
        if len(y):
            ty = T[y]; p = np.searchsorted(ty, T[i], "left")
            if p < len(ty): dwr[i] = np.log1p(ty[p] - T[i])
            b_ = np.searchsorted(ty, T[i] + 21600, "right"); out = cout[R[i]][b_] - cout[R[i]][p]
            x = ii[R[i]]; tx_ = T[x]; q, p2 = np.searchsorted(tx_, T[i] - 21600, "left"), np.searchsorted(tx_, T[i], "right")
            inn = cin[R[i]][p2] - cin[R[i]][q]; io_r[i] = np.log1p(inn) - np.log1p(out)
    F.update(s_dwell=dw, s_dwell_amt=dwamt, r_dwell_next=dwr, s_io6h=io_s, r_io6h=io_r, dwell_sum=np.minimum(dw, 16) + np.minimum(dwr, 16))
    f = pd.DataFrame(F); f.insert(0, "TRANSACTION_HASH", tx.TRANSACTION_HASH.values)
    return f


if __name__ == "__main__":
    import time; t0 = time.time()
    tx = load_split(sys.argv[1]); f = build(tx); f.to_parquet(sys.argv[2]); print(f.shape, time.time() - t0)
