"""v6b explorer patterns (per split, label-free, within-split rank-normalised; no account IDs as features).
H1 burst/episode span+size at 30m/2h/3h/6h (txs linked when they share a non-hub account within the window),
H2 one-off pair signals, H3 account-week context (sender+receiver aggregates of label-free flags),
H4 hard-relay traces (fee steadiness, 12-18 UTC share, episode recurrence, counterparty churn).
Usage: python src/fe9.py SPLIT_DIR F8_PARQUET OUT_PARQUET"""
import sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from fe import load_split

def rk(x): return pd.Series(x).rank(pct=True).to_numpy()

def main(d, f8p, out):
    tx = load_split(d).reset_index(drop=True)
    f8 = pd.read_parquet(f8p).set_index("TRANSACTION_HASH").reindex(tx.TRANSACTION_HASH)
    n = len(tx); S, R = tx.ACCOUNT.to_numpy(), tx.DEST.to_numpy()
    T = tx.TIMESTAMP.to_numpy().astype(np.int64)
    F = pd.DataFrame({"TRANSACTION_HASH": tx.TRANSACTION_HASH})
    # account incidence (sender or receiver)
    inc = pd.DataFrame({"a": np.r_[S, R], "i": np.r_[np.arange(n), np.arange(n)], "t": np.r_[T, T]})
    deg = inc.groupby("a").i.transform("size").to_numpy()
    hubcut = np.quantile(inc.groupby("a").size(), 0.995)
    inc_nh = inc[deg <= max(hubcut, 30)].sort_values(["a", "t"])
    epi = {}
    # connected episodes only at 30m (wider windows percolate into one giant component)
    a, i, t = inc_nh.a.to_numpy(), inc_nh.i.to_numpy(), inc_nh.t.to_numpy()
    same = (a[1:] == a[:-1]) & ((t[1:] - t[:-1]) <= 1800)
    g = coo_matrix((np.ones(same.sum()), (i[:-1][same], i[1:][same])), shape=(n, n))
    _, lab = connected_components(g, directed=False)
    e = pd.DataFrame({"l": lab, "t": T}).groupby("l").t
    size = e.transform("size").to_numpy(); span = (e.transform("max") - e.transform("min")).to_numpy()
    epi["30m"] = (lab, size, span)
    F["H1_size_30m"] = np.log1p(size); F["H1_span_30m"] = np.log1p(span)
    F["H1_rate_30m"] = np.log1p(size / (span / 60 + 1))
    F["H1_fast_30m"] = ((size >= 3) & (span <= 600)).astype(int)
    # local burst around each tx: other txs of its sender/receiver within +-W, their count and time span
    o = inc.sort_values(["a", "t"]).reset_index(drop=True)
    aa, tt, ii = o.a.to_numpy(), o.t.to_numpy(), o.i.to_numpy()
    st = np.r_[0, np.flatnonzero(aa[1:] != aa[:-1]) + 1]; en = np.r_[st[1:], len(aa)]
    for W, nm in ((1800, "30m"), (7200, "2h"), (10800, "3h"), (21600, "6h")):
        cntv = np.zeros(len(o)); spv = np.zeros(len(o))
        for s0, e0 in zip(st, en):
            x = tt[s0:e0]
            lo = np.searchsorted(x, x - W, "left"); hi = np.searchsorted(x, x + W, "right")
            cntv[s0:e0] = hi - lo; spv[s0:e0] = x[hi - 1] - x[lo]
        q = pd.DataFrame({"i": ii, "c": cntv, "s": spv}).groupby("i")
        c_ = q.c.max().reindex(range(n)).to_numpy(); s_ = q.s.max().reindex(range(n)).to_numpy()
        F[f"H1_lcnt_{nm}"] = np.log1p(c_); F[f"H1_lspan_{nm}"] = np.log1p(s_)
        F[f"H1_lrate_{nm}"] = np.log1p(c_ / (s_ / 60 + 1))
    # H2 one-off pairs
    pc = pd.Series(list(zip(S, R))).map(pd.Series(list(zip(S, R))).value_counts()).to_numpy()
    one = (pc == 1).astype(float)
    F["H2_pair_cnt_rk"] = rk(pc); F["H2_oneoff"] = one
    ia = pd.DataFrame({"a": np.r_[S, R], "one": np.r_[one, one], "t": np.r_[T, T], "i": np.r_[np.arange(n), np.arange(n)]})
    sh = ia.groupby("a").one.mean()
    F["H2_s_oneoff_share"] = S_map(S, sh); F["H2_r_oneoff_share"] = S_map(R, sh)
    # one-off txs touching account within +-1h
    o = ia.sort_values(["a", "t"]); cnt = np.zeros(len(o))
    for a_, grp in o.groupby("a", sort=False):
        if len(grp) == 1: continue
        tt = grp.t.to_numpy(); cs = np.r_[0, np.cumsum(grp.one.to_numpy())]
        lo = np.searchsorted(tt, tt - 3600, "left"); hi = np.searchsorted(tt, tt + 3600, "right")
        cnt[o.index.get_indexer(grp.index)] = cs[hi] - cs[lo]
    o["c"] = cnt; c2 = o.groupby("i").c.max().reindex(range(n)).to_numpy()
    F["H2_oneoff_1h"] = rk(c2)
    # H3 account-week context of label-free flags
    flags = {"fast": F["H1_fast_30m"].to_numpy(), "chain": f8.C_flag.fillna(0).to_numpy(),
             "gather": f8.D_flag.fillna(0).to_numpy(), "one": one}
    for k, v in flags.items():
        dd = pd.DataFrame({"a": np.r_[S, R], "v": np.r_[v, v]}).groupby("a").v
        mu, sm = dd.mean(), dd.sum()
        for side, A in (("s", S), ("r", R)):
            F[f"H3_{side}_{k}_mean"] = S_map(A, mu)
            F[f"H3_{side}_{k}_oth"] = rk(S_map(A, sm) - v)   # count over the account's other txs
    # H4 hard-relay traces
    fee = pd.to_numeric(tx.FEE, errors="coerce").fillna(0).to_numpy()
    hr = (T % 86400) // 3600
    df = pd.DataFrame({"S": S, "fee": fee, "aft": ((hr >= 12) & (hr < 18)).astype(float), "R": R})
    gs = df.groupby("S")
    fcv = gs.fee.std() / (gs.fee.mean() + 1e-9); fmode = gs.fee.agg(lambda s: s.value_counts(normalize=True).iloc[0])
    F["H4_s_fee_cv"] = rk(S_map(S, fcv.fillna(0))); F["H4_s_fee_mode"] = S_map(S, fmode)
    F["H4_s_fee_rel"] = rk(fee / (S_map(S, gs.fee.median()) + 1e-9))
    aft = pd.DataFrame({"a": np.r_[S, R], "v": np.r_[df.aft, df.aft]}).groupby("a").v.mean()
    F["H4_s_aft_share"] = S_map(S, aft); F["H4_r_aft_share"] = S_map(R, aft)
    F["H4_hour"] = hr
    lab, size, _ = epi["30m"]
    ep = pd.DataFrame({"S": S, "l": lab, "t": T}).groupby(["S", "l"]).t.min().reset_index().sort_values(["S", "t"])
    ep["gap"] = ep.groupby("S").t.diff() / 86400
    gmed = ep.groupby("S").gap.median(); nep = ep.groupby("S").size()
    F["H4_s_ep_gap_d"] = S_map(S, gmed.fillna(-1)); F["H4_s_neps"] = rk(S_map(S, nep))
    F["H4_s_ep_gap_26"] = (np.abs(F.H4_s_ep_gap_d - 2.6) < 0.6).astype(int)
    churn = gs.R.nunique() / gs.size()
    F["H4_s_cp_churn"] = S_map(S, churn)
    F.to_parquet(out); print(out, F.shape)

def S_map(A, s): return pd.Series(A).map(s).fillna(0).to_numpy().astype(float)

if __name__ == "__main__":
    main(*sys.argv[1:4])
