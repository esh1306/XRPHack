"""Per-split transductive feature engineering (no labels used).
One row per transaction. All features computed within the split's own 7-day graph."""
import json, sys, warnings
warnings.filterwarnings('ignore')
from collections import defaultdict
from pathlib import Path
import numpy as np, pandas as pd, networkx as nx

XRP = "0x1111222211112222111122221111222211112222"
WINDOWS = {"5m": 300, "1h": 3600, "6h": 21600, "24h": 86400}


def load_split(d):
    d = Path(d)
    tx = pd.read_parquet(d / "transactions.parquet")
    tr = pd.read_parquet(d / "transfers.parquet")
    rates = pd.read_csv(d / "market_rates.csv")
    rate = {(int(a), b): p / 1e6 for a, b, p in zip(rates.TIMESTAMP, rates.ASSET, rates.PRICE_MICRO_USD)}
    tx["TIMESTAMP"] = tx.TIMESTAMP.astype("int64")
    for c in ["LEDGER_INDEX", "TRANSACTION_INDEX", "SOURCE_FEE", "SEQUENCE", "LAST_LEDGER_SEQUENCE"]:
        tx[c] = pd.to_numeric(tx[c].astype(str), errors="coerce")
    ex = tx.EXTRA_TRANSACTION_FIELDS.map(json.loads)
    tx["DEST"] = ex.map(lambda e: e["Destination"])
    tx["has_sendmax"] = ex.map(lambda e: "SendMax" in e).astype(int)
    tx["has_paths"] = ex.map(lambda e: "Paths" in e).astype(int)
    tx["n_paths"] = ex.map(lambda e: len(e.get("Paths", []) or [])).astype(int)
    tx["dest_tag"] = ex.map(lambda e: "DestinationTag" in e).astype(int)
    hour = tx.TIMESTAMP // 3600 * 3600

    def amt(w, ts):
        if isinstance(w, str):
            return "XRP", int(w) / 1e6
        return f"{w['issuer']}:{w['currency']}", float(w["value"])
    dl = [amt(json.loads(w), t) for w, t in zip(tx.DELIVERED_AMOUNT, tx.TIMESTAMP)]
    tx["dst_asset"] = [a for a, _ in dl]
    tx["dst_units"] = [u for _, u in dl]
    tx["usd"] = [u * rate.get((h, a), np.nan) for (a, u), h in zip(dl, hour)]
    sm = [amt(e["SendMax"], 0) if "SendMax" in e else (None, np.nan) for e in ex]
    tx["sendmax_usd"] = [u * rate.get((h, a), np.nan) if a else np.nan for (a, u), h in zip(sm, hour)]
    # transfers: first leg actually paid
    tr["TRANSFER_INDEX"] = pd.to_numeric(tr.TRANSFER_INDEX.astype(str))
    tr = tr.sort_values(["TRANSACTION_HASH", "TRANSFER_INDEX"])
    tr["asset"] = np.where(tr.TOKEN_ADDRESS == XRP, "XRP", tr.TOKEN_ADDRESS + ":" + tr.SYMBOL)
    tr["ts"] = pd.to_numeric(tr.TIMESTAMP.astype(str)) // 3600 * 3600
    tr["usd"] = [v * rate.get((h, a), np.nan) for v, h, a in zip(tr.VALUE, tr.ts, tr.asset)]
    g = tr.groupby("TRANSACTION_HASH")
    first = g.first()
    agg = pd.DataFrame({"n_legs": g.size(), "src_asset": first.asset, "paid_usd": first.usd,
                        "n_assets": g.asset.nunique(), "path_type": (first.TRANSFER_TYPE == "path_payment").astype(int)})
    tx = tx.join(agg, on="TRANSACTION_HASH")
    tx["cross_ccy"] = (tx.src_asset != tx.dst_asset).astype(int)
    return tx.sort_values(["TIMESTAMP", "TRANSACTION_INDEX"]).reset_index(drop=True)


def _win_stats(times, vals, cps, t, lo, hi):
    a, b = np.searchsorted(times, lo, "left"), np.searchsorted(times, hi, "right")
    return a, b


def build_features(tx):
    n = len(tx)
    S, R = tx.ACCOUNT.to_numpy(), tx.DEST.to_numpy()
    T = tx.TIMESTAMP.to_numpy().astype(np.int64)
    V = tx.usd.fillna(0).to_numpy()
    idx = np.arange(n)
    f = pd.DataFrame({"TRANSACTION_HASH": tx.TRANSACTION_HASH})
    # --- per-tx intrinsic
    f["log_usd"] = np.log1p(V)
    f["log_paid"] = np.log1p(tx.paid_usd.fillna(0))
    f["spread"] = (tx.paid_usd / (tx.usd + 1e-9)).clip(0, 5)
    f["sendmax_ratio"] = (tx.sendmax_usd / (tx.usd + 1e-9)).clip(0, 5)
    u = tx.dst_units.to_numpy()
    for k in (1, 10, 100, 1000):
        f[f"round_{k}"] = (np.abs(u / k - np.round(u / k)) < 1e-6).astype(int)
    f["usd_round_100"] = (np.abs(V / 100 - np.round(V / 100)) < 0.005).astype(int)
    frac = u - np.floor(u)
    f["units_frac"] = frac
    f["dst_xrp"] = (tx.dst_asset == "XRP").astype(int)
    f["src_xrp"] = (tx.src_asset == "XRP").astype(int)
    for c in ["n_legs", "n_assets", "path_type", "cross_ccy", "has_sendmax", "has_paths", "n_paths", "dest_tag", "SOURCE_FEE", "TRANSACTION_INDEX"]:
        f[c] = tx[c].astype(float)
    f["lls_gap"] = tx.LAST_LEDGER_SEQUENCE - tx.LEDGER_INDEX
    f["hour"] = (T // 3600) % 24
    f["dow"] = (T // 86400) % 7
    f["t_rel"] = (T - T.min()) / 86400
    f["dst_asset_code"] = tx.dst_asset.astype("category").cat.codes
    f["src_asset_code"] = tx.src_asset.astype("category").cat.codes
    # amount repeats
    key = np.round(V, 2)
    f["amt_repeat_global"] = pd.Series(key).map(pd.Series(key).value_counts()).to_numpy()
    f["amt_repeat_sender"] = tx.assign(k=key).groupby(["ACCOUNT", "k"]).k.transform("size").to_numpy()

    # --- per-account indices
    out_idx, in_idx = defaultdict(list), defaultdict(list)
    for i in idx:
        out_idx[S[i]].append(i); in_idx[R[i]].append(i)
    out_idx = {a: np.array(v) for a, v in out_idx.items()}
    in_idx = {a: np.array(v) for a, v in in_idx.items()}
    empty = np.array([], dtype=int)
    accts = set(S) | set(R)

    # account weekly aggregates
    acc = pd.DataFrame(index=sorted(accts))
    acc["out_cnt"] = pd.Series({a: len(v) for a, v in out_idx.items()})
    acc["in_cnt"] = pd.Series({a: len(v) for a, v in in_idx.items()})
    acc["out_sum"] = pd.Series({a: V[v].sum() for a, v in out_idx.items()})
    acc["in_sum"] = pd.Series({a: V[v].sum() for a, v in in_idx.items()})
    acc["out_uniq"] = pd.Series({a: len(set(R[v])) for a, v in out_idx.items()})
    acc["in_uniq"] = pd.Series({a: len(set(S[v])) for a, v in in_idx.items()})
    acc["out_med"] = pd.Series({a: np.median(V[v]) for a, v in out_idx.items()})
    acc["in_med"] = pd.Series({a: np.median(V[v]) for a, v in in_idx.items()})
    acc["out_iat_med"] = pd.Series({a: np.median(np.diff(T[v])) if len(v) > 1 else np.nan for a, v in out_idx.items()})
    acc["out_iat_cv"] = pd.Series({a: np.std(np.diff(T[v])) / (np.mean(np.diff(T[v])) + 1) if len(v) > 2 else np.nan for a, v in out_idx.items()})
    acc["first_seen"] = pd.Series({a: min(T[out_idx.get(a, empty)].min(initial=10**12), T[in_idx.get(a, empty)].min(initial=10**12)) for a in accts}) - T.min()
    acc["last_seen"] = T.max() - pd.Series({a: max(T[out_idx.get(a, empty)].max(initial=0), T[in_idx.get(a, empty)].max(initial=0)) for a in accts})
    acc["n_assets_out"] = tx.groupby("ACCOUNT").src_asset.nunique()
    acc["n_assets_in"] = tx.groupby("DEST").dst_asset.nunique()
    acc["cross_rate"] = tx.groupby("ACCOUNT").cross_ccy.mean()
    acc = acc.fillna({"out_cnt": 0, "in_cnt": 0, "out_sum": 0, "in_sum": 0, "out_uniq": 0, "in_uniq": 0})
    acc["pass_through"] = (acc.out_sum / (acc.in_sum + 1)).clip(0, 20)
    acc["in_out_ratio_cnt"] = acc.in_cnt / (acc.out_cnt + 1)
    # graph
    G = nx.DiGraph()
    for s, r, v in zip(S, R, V):
        if G.has_edge(s, r):
            G[s][r]["w"] += v; G[s][r]["c"] += 1
        else:
            G.add_edge(s, r, w=v, c=1)
    acc["pagerank"] = pd.Series(nx.pagerank(G, weight="w"))
    acc["pagerank_u"] = pd.Series(nx.pagerank(G, weight=None))
    UG = G.to_undirected()
    acc["clust"] = pd.Series(nx.clustering(UG))
    acc["core"] = pd.Series(nx.core_number(UG))
    acc["reciprocity"] = pd.Series({a: len(set(G.successors(a)) & set(G.predecessors(a))) / (len(set(G.successors(a)) | set(G.predecessors(a))) + 1e-9) for a in G})
    hubs = nx.hits(G, max_iter=500)
    acc["hub"], acc["auth"] = pd.Series(hubs[0]), pd.Series(hubs[1])
    for c in acc.columns:
        f["s_" + c] = acc[c].reindex(S).to_numpy()
        f["r_" + c] = acc[c].reindex(R).to_numpy()

    # pair features
    pair = tx.groupby(["ACCOUNT", "DEST"]).size()
    pk = list(zip(S, R))
    f["pair_cnt"] = pair.reindex(pk).to_numpy()
    f["rev_pair_cnt"] = pair.reindex(list(zip(R, S))).fillna(0).to_numpy()
    f["pair_share_of_s_out"] = f.pair_cnt / f.s_out_cnt
    f["pair_share_of_r_in"] = f.pair_cnt / f.r_in_cnt
    f["pair_rank"] = tx.groupby(["ACCOUNT", "DEST"]).cumcount().to_numpy()
    f["pair_amt_std"] = tx.groupby(["ACCOUNT", "DEST"]).usd.transform("std").to_numpy() / (V + 1)
    # structural motifs (static): intermediaries s->m->r ; 2-hop back r->m->s
    succ = {a: set(G.successors(a)) for a in G}
    pred = {a: set(G.predecessors(a)) for a in G}
    f["n_parallel_paths"] = [len(succ[s] & pred[r]) for s, r in pk]
    f["n_back_2hop"] = [len(succ[r] & pred[s]) for s, r in pk]
    f["common_out"] = [len(succ[s] & succ.get(r, set())) for s, r in pk]
    f["common_in"] = [len(pred.get(s, set()) & pred[r]) for s, r in pk]

    # --- temporal features
    f["amt_vs_s_med"] = np.log1p(V) - np.log1p(f.s_out_med)
    f["amt_vs_r_med"] = np.log1p(V) - np.log1p(f.r_in_med)
    s_pos = np.zeros(n); r_pos = np.zeros(n)
    for a, v in out_idx.items(): s_pos[v] = np.arange(len(v)) / max(len(v) - 1, 1)
    for a, v in in_idx.items(): r_pos[v] = np.arange(len(v)) / max(len(v) - 1, 1)
    f["s_pos"], f["r_pos"] = s_pos, r_pos
    feats = defaultdict(lambda: np.full(n, np.nan))
    for i in idx:
        s, r, t, v = S[i], R[i], T[i], V[i]
        ro = out_idx.get(r, empty); ro_t = T[ro]
        si = in_idx.get(s, empty); si_t = T[si]
        so = out_idx[s]; so_t = T[so]
        ri = in_idx[r]; ri_t = T[ri]
        # receiver forwards after
        a0 = np.searchsorted(ro_t, t, "left")
        feats["r_next_out_dt"][i] = np.log1p(ro_t[a0] - t) if a0 < len(ro) else 15
        b0 = np.searchsorted(si_t, t, "left")
        feats["s_prev_in_dt"][i] = np.log1p(t - si_t[b0 - 1]) if b0 > 0 else 15
        for wn, w in WINDOWS.items():
            b = np.searchsorted(ro_t, t + w, "right")
            later = ro[a0:b]
            lv = V[later]
            feats[f"r_fwd_cnt_{wn}"][i] = len(later)
            feats[f"r_fwd_sum_ratio_{wn}"][i] = lv.sum() / (v + 1e-9)
            feats[f"r_fwd_sim_{wn}"][i] = np.sum(np.abs(lv - v) <= 0.1 * v + 1e-9)
            feats[f"r_fwd_uniq_{wn}"][i] = len(set(R[later]))
            a = np.searchsorted(si_t, t - w, "left")
            earlier = si[a:b0]
            ev = V[earlier]
            feats[f"s_rcv_cnt_{wn}"][i] = len(earlier)
            feats[f"s_rcv_sum_ratio_{wn}"][i] = ev.sum() / (v + 1e-9)
            feats[f"s_rcv_sim_{wn}"][i] = np.sum(np.abs(ev - v) <= 0.1 * v + 1e-9)
            feats[f"s_rcv_uniq_{wn}"][i] = len(set(S[earlier]))
            # bursts (symmetric)
            x0, x1 = np.searchsorted(so_t, t - w, "left"), np.searchsorted(so_t, t + w, "right")
            feats[f"s_burst_{wn}"][i] = x1 - x0
            feats[f"s_burst_uniq_{wn}"][i] = len(set(R[so[x0:x1]]))
            feats[f"s_burst_amt_cv_{wn}"][i] = np.std(V[so[x0:x1]]) / (np.mean(V[so[x0:x1]]) + 1e-9)
            y0, y1 = np.searchsorted(ri_t, t - w, "left"), np.searchsorted(ri_t, t + w, "right")
            feats[f"r_burst_{wn}"][i] = y1 - y0
            feats[f"r_burst_uniq_{wn}"][i] = len(set(S[ri[y0:y1]]))
        # time-respecting 2-hop: r -> x within 24h after, x -> s later (cycle back)
        w = 86400 * 2
        b = np.searchsorted(ro_t, t + w, "right")
        cyc = 0; chain2 = 0
        for j in ro[a0:b]:
            x = R[j]
            if x == s:
                cyc += 1; continue
            xo = out_idx.get(x, empty)
            if len(xo):
                xt = T[xo]; c0 = np.searchsorted(xt, T[j], "left"); c1 = np.searchsorted(xt, T[j] + w, "right")
                nxt = xo[c0:c1]
                chain2 += len(nxt)
                cyc += int(np.sum(R[nxt] == s))
        feats["t_cycle"][i] = cyc
        feats["t_chain2"][i] = chain2
    f = pd.concat([f, pd.DataFrame(dict(feats))], axis=1)
    return f


if __name__ == "__main__":
    import time
    t0 = time.time()
    tx = load_split(sys.argv[1])
    f = build_features(tx)
    f.to_parquet(sys.argv[2])
    print(f.shape, time.time() - t0)
