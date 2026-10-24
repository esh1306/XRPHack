"""v6 feature groups from the owner's test-file pattern spec (per split, transductive, no labels, no account IDs).
A row-level, B pair, C layering chains, D gather bursts, G IsolationForest (fit per split).
End-to-end flow uses the sender->Destination graph (ACCOUNT -> EXTRA_TRANSACTION_FIELDS.Destination);
the transfer graph (dominated by ~9 hub intermediaries) is only used for hop structure (legs / currency path).
USD values use the hourly rate keyed by issuer:currency (see fe.load_split).
Usage: python src/fe8.py SPLIT_DIR OUT_PARQUET"""
import sys, warnings
warnings.filterwarnings("ignore")
from collections import defaultdict
import numpy as np, pandas as pd
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from fe import load_split, XRP

WINS = (60, 300, 900, 3600)
BANDS = {"b88": (0.88, 1.04), "b90": (0.90, 1.00), "b95": (0.95, 1.00)}


def ccy(a):
    return "XRP" if a == "XRP" else a.split(":")[1]


def main(d, out):
    tx = load_split(d)
    tr = pd.read_parquet(f"{d}/transfers.parquet")
    tr["TRANSFER_INDEX"] = pd.to_numeric(tr.TRANSFER_INDEX.astype(str))
    tr = tr.sort_values(["TRANSACTION_HASH", "TRANSFER_INDEX"])
    tr["c"] = np.where(tr.TOKEN_ADDRESS == XRP, "XRP", tr.SYMBOL.astype(str))
    tr["iss"] = np.where(tr.TOKEN_ADDRESS == XRP, "XRP", tr.TOKEN_ADDRESS.astype(str))
    g = tr.groupby("TRANSACTION_HASH")
    path = g.c.agg(lambda s: ">".join(s))
    n_ccy = g.c.nunique(); n_iss = g.iss.nunique()
    n = len(tx); H = tx.TRANSACTION_HASH
    S, R = tx.ACCOUNT.to_numpy(), tx.DEST.to_numpy()
    T = tx.TIMESTAMP.to_numpy().astype(np.int64)
    V = tx.usd.fillna(0).to_numpy()                 # delivered USD
    P = tx.paid_usd.fillna(tx.usd).fillna(0).to_numpy()  # sent USD (first leg actually paid)
    dc = np.array([ccy(a) for a in tx.dst_asset]); sc = np.array([ccy(a) if isinstance(a, str) else "" for a in tx.src_asset])
    F = pd.DataFrame({"TRANSACTION_HASH": H})
    # ---------------- A: row
    F["A_log_usd_deliv"] = np.log1p(V); F["A_log_usd_sent"] = np.log1p(P)
    F["A_sent_over_deliv"] = np.log((P + 1e-6) / (V + 1e-6)).clip(-3, 3)
    pth = path.reindex(H).fillna("").to_numpy()
    PATHS = sorted(set("{}>{}>{}".format(a, b, c) for a in ("XRP", "USD", "EUR", "GBP") for b in ("XRP", "USD", "EUR", "GBP") for c in ("XRP", "USD", "EUR", "GBP")))
    # stable cross-split encoding of the currency-code path (codes only; no issuer/account strings)
    F["A_path_code"] = [hash_path(p) for p in pth]
    F["A_n_transfers"] = tx.n_legs.to_numpy()
    F["A_n_distinct_ccy"] = n_ccy.reindex(H).to_numpy()
    F["A_n_issuers"] = n_iss.reindex(H).to_numpy()
    F["A_cross_issuer"] = ((sc == dc) & (tx.src_asset.to_numpy() != tx.dst_asset.to_numpy())).astype(int)
    F["A_is_conversion"] = tx.has_sendmax.to_numpy()
    F["A_ccy_change"] = (sc != dc).astype(int)
    hr = (T % 86400) / 86400 * 2 * np.pi
    F["A_hour_sin"], F["A_hour_cos"] = np.sin(hr), np.cos(hr)
    df = pd.DataFrame({"S": S, "R": R, "T": T, "V": V, "seq": tx.SEQUENCE.to_numpy(), "li": tx.LEDGER_INDEX.to_numpy(), "ti": tx.TRANSACTION_INDEX.to_numpy()})
    smed = df.groupby("S").V.transform("median")
    F["A_logratio_s_med"] = np.log((V + 1e-6) / (smed + 1e-6)).clip(-6, 6)
    rmed = df.groupby("R").V.transform("median")
    F["A_logratio_r_med"] = np.log((V + 1e-6) / (rmed + 1e-6)).clip(-6, 6)
    o = df.sort_values(["S", "T", "li", "ti"])
    gp = o.groupby("S")
    prev_gap = (o["T"] - gp["T"].shift(1)); next_gap = (gp["T"].shift(-1) - o["T"])
    F["A_log_gap_prev_s"] = np.log1p(prev_gap.reindex(df.index)).fillna(-1).to_numpy()
    F["A_log_gap_next_s"] = np.log1p(next_gap.reindex(df.index)).fillna(-1).to_numpy()
    sg = (o.seq - gp.seq.shift(1) - 1).reindex(df.index)
    F["A_seq_gap_prev"] = np.log1p(sg.clip(lower=0)).fillna(-1).to_numpy()
    F["A_seq_backwards"] = (sg < -1).astype(int).to_numpy()
    same_sec = (prev_gap == 0) | (next_gap == 0)
    F["A_same_second_s"] = same_sec.reindex(df.index).astype(int).to_numpy()
    F["A_seq_ooo_same_sec"] = (same_sec & ((o.seq.diff() < 0) & (prev_gap == 0) | (gp.seq.shift(-1) < o.seq) & (next_gap == 0))).reindex(df.index).astype(int).to_numpy()
    # ---------------- B: pair
    pg = df.groupby(["S", "R"])
    pc = pg.V.transform("size").to_numpy()
    F["B_pair_cnt"] = pc
    F["B_pair_amt_cv"] = (pg.V.transform("std") / (pg.V.transform("mean") + 1e-9)).fillna(-1).to_numpy()
    span = (pg["T"].transform("max") - pg["T"].transform("min")).to_numpy()
    F["B_pair_span_d"] = span / 86400
    F["B_pair_med_gap_h"] = np.where(pc > 1, span / np.maximum(pc - 1, 1) / 3600, -1)
    df["dc"] = dc
    F["B_pair_n_ccy"] = df.groupby(["S", "R"]).dc.transform("nunique").to_numpy()
    srt = df.sort_values("T"); gaps = srt.groupby(["S", "R"])["T"].diff()
    gstd = gaps.groupby([srt.S, srt.R]).transform("std").reindex(df.index)
    gmed = gaps.groupby([srt.S, srt.R]).transform("median").reindex(df.index)
    F["B_pair_gap_cv"] = (gstd / (gmed + 1)).fillna(-1).to_numpy()
    daily = (gmed > 0.8 * 86400) & (gmed < 1.2 * 86400)
    F["B_daily_cadence"] = daily.astype(int).to_numpy()
    rec = (pc >= 5) & (pc <= 7) & daily.to_numpy() & (F.B_pair_n_ccy.to_numpy() == 1) & (F.B_pair_amt_cv.to_numpy().clip(0) < 0.1)
    F["B_recurring_lookalike"] = rec.astype(int)
    F["B_recurring_loose"] = ((pc >= 4) & daily.to_numpy() & (F.B_pair_amt_cv.to_numpy().clip(0) < 0.2)).astype(int)
    pairs = set(zip(S, R))
    F["B_reverse_pair"] = np.array([(r, s) in pairs for s, r in zip(S, R)]).astype(int)
    F["B_first_time_pair"] = (srt.groupby(["S", "R"]).cumcount().reindex(df.index) == 0).astype(int).to_numpy()
    F["B_pair_pos"] = (srt.groupby(["S", "R"]).cumcount().reindex(df.index).to_numpy() / np.maximum(pc - 1, 1))
    # ---------------- C: layering chains (sender->destination graph)
    out_idx = defaultdict(list); in_idx = defaultdict(list)
    order = np.argsort(T, kind="stable")
    for i in order: out_idx[S[i]].append(i); in_idx[R[i]].append(i)
    out_idx = {a: np.array(v) for a, v in out_idx.items()}; in_idx = {a: np.array(v) for a, v in in_idx.items()}
    e = np.array([], int)
    def nxt(i, w):  # txs sent by recipient of i within (0, w]
        c = out_idx.get(R[i], e); tt = T[c]
        a, b = np.searchsorted(tt, T[i], "left"), np.searchsorted(tt, T[i] + w, "right")
        c = c[a:b]; return c[c != i]
    def prv(i, w):
        c = in_idx.get(S[i], e); tt = T[c]
        a, b = np.searchsorted(tt, T[i] - w, "left"), np.searchsorted(tt, T[i], "right")
        c = c[a:b]; return c[c != i]
    for w in WINS:
        nn = {k: np.zeros(n) for k in BANDS}; pp = {k: np.zeros(n) for k in BANDS}
        for i in range(n):
            cj = nxt(i, w)
            if len(cj):
                r = P[cj] / (V[i] + 1e-9)
                for k, (lo, hi) in BANDS.items(): nn[k][i] = ((r >= lo) & (r <= hi)).sum()
            ck = prv(i, w)
            if len(ck):
                r = P[i] / (V[ck] + 1e-9)
                for k, (lo, hi) in BANDS.items(): pp[k][i] = ((r >= lo) & (r <= hi)).sum()
        for k in BANDS:
            F[f"C_next_{w}_{k}"] = nn[k]; F[f"C_prev_{w}_{k}"] = pp[k]
    # chain DP at the spec setting (900s, [0.88,1.04])
    W0, (LO, HI) = 900, BANDS["b88"]
    nxt_l = [None] * n; best_next = -np.ones(n, int); hop_gap = np.full(n, -1.0); retained = np.full(n, -1.0); n_amb = np.zeros(n)
    ccy_hop = np.zeros(n)
    for i in range(n):
        cj = nxt(i, W0); cj = cj[(T[cj] > T[i]) | ((T[cj] == T[i]) & (cj != i))]
        if len(cj):
            r = P[cj] / (V[i] + 1e-9); cj = cj[(r >= LO) & (r <= HI)]
        nxt_l[i] = cj; n_amb[i] = len(cj)
        if len(cj):
            j = cj[np.argmin(np.abs(P[cj] / (V[i] + 1e-9) - 0.97) + (T[cj] - T[i]) / 1e5)]
            best_next[i] = j; hop_gap[i] = T[j] - T[i]; retained[i] = P[j] / (V[i] + 1e-9); ccy_hop[i] = int(dc[i] != dc[j])
    fwd = np.ones(n); back = np.ones(n)
    for i in order[::-1]:
        if len(nxt_l[i]): fwd[i] = 1 + fwd[nxt_l[i]].max()
    for i in order:
        for j in nxt_l[i]:
            if T[j] >= T[i]: back[j] = max(back[j], back[i] + 1)
    F["C_fwd_len"], F["C_back_len"] = fwd, back
    F["C_chain_len"] = fwd + back - 1
    F["C_pos"] = (back - 1) / np.maximum(fwd + back - 2, 1)
    F["C_hop_gap"], F["C_retained"], F["C_n_ambiguous"], F["C_ccy_change_hop"] = hop_gap, retained, n_amb, ccy_hop
    rto = np.zeros(n); vret = np.full(n, -1.0)
    for i in range(n):
        if best_next[i] < 0: continue
        j, seen, k = i, {S[i]}, 0
        while best_next[j] >= 0 and k < 12:
            j = best_next[j]; k += 1
            if R[j] == S[i]: rto[i] = 1; break
            if R[j] in seen: break
            seen.add(R[j])
        vret[i] = V[j] / (V[i] + 1e-9)
    F["C_returns_origin"] = rto; F["C_value_retained_end"] = vret
    F["C_flag"] = ((F["C_next_900_b88"] > 0) | (F["C_prev_900_b88"] > 0)).astype(int)
    F["C_tight"] = ((F["C_next_300_b90"] > 0) | (F["C_prev_300_b90"] > 0)).astype(int)
    # ---------------- D: gather bursts into recipient
    for lab, w in (("1h", 3600), ("6h", 21600)):
        tr_ds = np.zeros(n); ce_ds = np.zeros(n); tr_usd = np.zeros(n); share = np.zeros(n)
        for i in range(n):
            c = in_idx[R[i]]; tt = T[c]; tot = len(c)
            a, b = np.searchsorted(tt, T[i] - w, "left"), np.searchsorted(tt, T[i], "right"); m = c[a:b]
            tr_ds[i] = len(set(S[m])); tr_usd[i] = V[m].sum(); share[i] = len(m) / tot
            a, b = np.searchsorted(tt, T[i] - w / 2, "left"), np.searchsorted(tt, T[i] + w / 2, "right")
            ce_ds[i] = len(set(S[c[a:b]]))
        F[f"D_dsend_trail_{lab}"], F[f"D_dsend_cent_{lab}"] = tr_ds, ce_ds
        F[f"D_usd_pool_{lab}"] = np.log1p(tr_usd); F[f"D_share_in_{lab}"] = share
    fwd_ratio = np.zeros(n); fwd_delay = np.full(n, -1.0); maxwin = np.zeros(n)
    for i in range(n):
        c = in_idx[R[i]]; tt = T[c]
        a, b = np.searchsorted(tt, T[i] - 3600, "left"), np.searchsorted(tt, T[i], "right"); pool = V[c[a:b]].sum()
        oc = out_idx.get(R[i], e); ot = T[oc]
        a2, b2 = np.searchsorted(ot, T[i], "right"), np.searchsorted(ot, T[i] + 21600, "right")
        fwd_ratio[i] = V[oc[a2:b2]].sum() / (pool + 1e-9)
        if a2 < len(oc): fwd_delay[i] = np.log1p(ot[a2] - T[i])
        # any 1h window containing i with >= k distinct senders: max over windows starting in [T-1h, T]
        lo = np.searchsorted(tt, T[i] - 3600, "left"); hi = np.searchsorted(tt, T[i] + 3600, "right")
        best = 0
        for s0 in range(lo, np.searchsorted(tt, T[i], "right")):
            e0 = np.searchsorted(tt, tt[s0] + 3600, "right"); best = max(best, len(set(S[c[s0:e0]])))
            if e0 >= hi: break
        maxwin[i] = best
    F["D_fwd_ratio_6h"] = np.log1p(fwd_ratio).clip(0, 5); F["D_fwd_delay"] = fwd_delay
    F["D_max_dsend_1h_win"] = maxwin
    F["D_flag"] = (maxwin >= 6).astype(int)
    # ---------------- G: IsolationForest on A-D (fit within this split only)
    from sklearn.ensemble import IsolationForest
    X = F.drop(columns=["TRANSACTION_HASH"]).fillna(-1).to_numpy()
    iso = IsolationForest(n_estimators=300, random_state=42, n_jobs=1).fit(X)
    F["G_iso_rank"] = pd.Series(-iso.score_samples(X)).rank(pct=True).to_numpy()
    F.to_parquet(out); print(out, F.shape, {k: int(F[k].sum()) for k in ("C_flag", "C_tight", "D_flag", "B_recurring_lookalike")}, "chains3+:", int((F.C_chain_len >= 3).sum()))


def hash_path(p):
    m = {"XRP": 1, "USD": 2, "EUR": 3, "GBP": 4}
    v = 0
    for t in p.split(">"):
        v = v * 5 + m.get(t, 0)
    return v


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
