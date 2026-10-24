"""Flow-level features for medium/hard cases: long windows, per-asset flows, running balances,
counterparty novelty, whole-flow conservation. Label-free, per split."""
import sys, warnings
from collections import defaultdict
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from fe import load_split

LW = {"3d": 3 * 86400, "7d": 8 * 86400}


def build_flow(tx):
    n = len(tx)
    S, R = tx.ACCOUNT.to_numpy(), tx.DEST.to_numpy()
    T = tx.TIMESTAMP.to_numpy().astype(np.int64)
    V = tx.usd.fillna(0).to_numpy()
    SA, DA = tx.src_asset.to_numpy(), tx.dst_asset.to_numpy()
    out_idx, in_idx = defaultdict(list), defaultdict(list)
    for i in range(n):
        out_idx[S[i]].append(i); in_idx[R[i]].append(i)
    out_idx = {a: np.array(v) for a, v in out_idx.items()}
    in_idx = {a: np.array(v) for a, v in in_idx.items()}
    empty = np.array([], dtype=int)
    F = defaultdict(lambda: np.full(n, np.nan))
    # first-seen pair / novelty
    seen = set(); s_new = defaultdict(list)
    for i in range(n):
        k = (S[i], R[i]); F["pair_first"][i] = float(k not in seen); seen.add(k)
    for i in range(n):
        s, r, t, v = S[i], R[i], T[i], V[i]
        si = in_idx.get(s, empty); so = out_idx[s]
        ri = in_idx[r]; ro = out_idx.get(r, empty)
        si_t, so_t, ri_t, ro_t = T[si], T[so], T[ri], T[ro]
        # running balance proxies at time t
        cin = V[si[si_t < t]].sum(); cout = V[so[so_t < t]].sum()
        F["s_net_before"][i] = (cin - cout) / (v + 1)
        F["s_cum_in_before"][i] = np.log1p(cin)
        F["s_cum_out_before"][i] = np.log1p(cout)
        rin = V[ri[ri_t < t]].sum(); rout = V[ro[ro_t < t]].sum()
        F["r_net_before"][i] = (rin - rout) / (v + 1)
        F["r_out_after_total"][i] = V[ro[ro_t > t]].sum() / (v + 1)
        for wn, w in LW.items():
            m = (si_t >= t - w) & (si_t < t)
            F[f"s_in_{wn}"][i] = V[si[m]].sum() / (v + 1)
            F[f"s_in_cnt_{wn}"][i] = m.sum()
            F[f"s_in_uniq_{wn}"][i] = len(set(S[si[m]]))
            # same asset received as now spent
            F[f"s_in_same_asset_{wn}"][i] = V[si[m & (DA[si] == SA[i])]].sum() / (v + 1)
            m2 = (ro_t > t) & (ro_t <= t + w)
            F[f"r_out_{wn}"][i] = V[ro[m2]].sum() / (v + 1)
            F[f"r_out_cnt_{wn}"][i] = m2.sum()
            F[f"r_out_uniq_{wn}"][i] = len(set(R[ro[m2]]))
            F[f"r_out_same_asset_{wn}"][i] = V[ro[m2 & (SA[ro] == DA[i])]].sum() / (v + 1)
            mo = (so_t >= t - w) & (so_t <= t + w)
            F[f"s_out_new_cp_{wn}"][i] = len(set(R[so[mo]]))
            mi = (ri_t >= t - w) & (ri_t <= t + w)
            F[f"r_in_uniq_{wn}"][i] = len(set(S[ri[mi]]))
        # sender's outflows in the scenario-ish window vs inflows: conservation around this tx
        m_in = (si_t >= t - 2 * 86400) & (si_t < t); m_out = (so_t >= t) & (so_t <= t + 2 * 86400)
        a, b = V[si[m_in]].sum(), V[so[m_out]].sum()
        F["s_conserve_2d"][i] = min(a, b) / (max(a, b) + 1)
        m_in = (ri_t >= t - 2 * 86400) & (ri_t <= t); m_out = (ro_t > t) & (ro_t <= t + 2 * 86400)
        a, b = V[ri[m_in]].sum(), V[ro[m_out]].sum()
        F["r_conserve_2d"][i] = min(a, b) / (max(a, b) + 1)
        # receiver's share of incoming from this sender, sender's median gap to its in
        F["r_in_from_s_share"][i] = (S[ri] == s).mean()
        F["s_out_to_r_share"][i] = (R[so] == r).mean()
        # asset hopping: distinct assets received by sender in prior 3d vs asset sent
        m = (si_t >= t - 3 * 86400) & (si_t < t)
        F["s_in_assets_3d"][i] = len(set(DA[si[m]]))
        F["asset_hop"][i] = float(m.any() and SA[i] not in set(DA[si[m]]))
    # account-level relay signature: fraction of in-volume that leaves within 24h (over week)
    f = pd.DataFrame({"TRANSACTION_HASH": tx.TRANSACTION_HASH})
    f = pd.concat([f, pd.DataFrame(dict(F))], axis=1)
    # account aggregate of tx-level features (account "role" over whole week)
    tmp = f.assign(S=S, R=R)
    for c in ["s_conserve_2d", "s_net_before"]:
        f["acc_s_mean_" + c] = tmp.groupby("S")[c].transform("mean")
    for c in ["r_conserve_2d", "r_out_after_total"]:
        f["acc_r_mean_" + c] = tmp.groupby("R")[c].transform("mean")
    # relative to receiver's typical relationships: z-score of amount within receiver's inflows
    lv = np.log1p(V)
    g = pd.Series(lv).groupby(R)
    f["r_in_amt_z"] = (lv - g.transform("mean").to_numpy()) / (g.transform("std").fillna(1).to_numpy() + 1e-6)
    g = pd.Series(lv).groupby(S)
    f["s_out_amt_z"] = (lv - g.transform("mean").to_numpy()) / (g.transform("std").fillna(1).to_numpy() + 1e-6)
    # hour-of-day atypicality for sender
    hr = (T // 3600) % 24
    hs = pd.DataFrame({"S": S, "h": hr})
    f["s_hour_freq"] = hs.groupby(["S", "h"]).h.transform("size").to_numpy() / hs.groupby("S").h.transform("size").to_numpy()
    return f


if __name__ == "__main__":
    import time
    t0 = time.time()
    tx = load_split(sys.argv[1])
    f = build_flow(tx)
    f.to_parquet(sys.argv[2]); print(f.shape, time.time() - t0)
