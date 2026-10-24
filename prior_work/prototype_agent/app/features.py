"""Deterministic flow/graph features. All the arithmetic lives here, never in Jev."""
from __future__ import annotations

from collections import defaultdict

import networkx as nx
import numpy as np
import pandas as pd

MIN, HOUR, DAY = 60, 3600, 86400


class AccountIndex:
    """Per-account time-sorted send/receive arrays for fast window queries."""

    def __init__(self, e: pd.DataFrame):
        self.out_t, self.out_usd, self.out_to, self.out_row = {}, {}, {}, {}
        self.in_t, self.in_usd, self.in_from, self.in_row = {}, {}, {}, {}
        for acct, g in e.groupby("sender", sort=False):
            self.out_t[acct] = g.t.to_numpy()
            self.out_usd[acct] = g.usd.to_numpy()
            self.out_to[acct] = g.receiver.to_numpy()
            self.out_row[acct] = g.index.to_numpy()
        for acct, g in e.groupby("receiver", sort=False):
            self.in_t[acct] = g.t.to_numpy()
            self.in_usd[acct] = g.usd.to_numpy()
            self.in_from[acct] = g.sender.to_numpy()
            self.in_row[acct] = g.index.to_numpy()

    @staticmethod
    def _slice(times, lo, hi, lo_open=False, hi_open=False):
        if times is None:
            return slice(0, 0)
        a = np.searchsorted(times, lo, "right" if lo_open else "left")
        b = np.searchsorted(times, hi, "left" if hi_open else "right")
        return slice(a, b)

    def sends(self, acct, lo, hi, lo_open=False, hi_open=False):
        s = self._slice(self.out_t.get(acct), lo, hi, lo_open, hi_open)
        if acct not in self.out_t:
            return np.array([]), np.array([]), np.array([]), np.array([], int)
        return self.out_t[acct][s], self.out_usd[acct][s], self.out_to[acct][s], self.out_row[acct][s]

    def gets(self, acct, lo, hi, lo_open=False, hi_open=False):
        s = self._slice(self.in_t.get(acct), lo, hi, lo_open, hi_open)
        if acct not in self.in_t:
            return np.array([]), np.array([]), np.array([]), np.array([], int)
        return self.in_t[acct][s], self.in_usd[acct][s], self.in_from[acct][s], self.in_row[acct][s]


def _best_ratio(amounts: np.ndarray, v: float) -> float:
    """Closest amount ratio to 1 (as |ratio-1|); 9 when nothing to compare."""
    if len(amounts) == 0 or v <= 0:
        return 9.0
    return float(np.min(np.abs(amounts / v - 1)))


def account_table(e: pd.DataFrame) -> pd.DataFrame:
    out = e.groupby("sender").agg(out_n=("usd", "size"), out_usd=("usd", "sum"), out_cp=("receiver", "nunique"),
                                  out_first=("t", "min"), out_last=("t", "max"),
                                  out_conv=("n_legs", lambda s: float((s > 1).mean())),
                                  out_logmean=("usd", lambda s: float(np.log1p(s).mean())),
                                  out_logstd=("usd", lambda s: float(np.log1p(s).std(ddof=0))))
    inn = e.groupby("receiver").agg(in_n=("usd", "size"), in_usd=("usd", "sum"), in_cp=("sender", "nunique"),
                                    in_first=("t", "min"), in_last=("t", "max"))
    acc = out.join(inn, how="outer")
    acc[["out_n", "out_usd", "out_cp", "in_n", "in_usd", "in_cp"]] = \
        acc[["out_n", "out_usd", "out_cp", "in_n", "in_usd", "in_cp"]].fillna(0)
    acc["first"] = acc[["out_first", "in_first"]].min(axis=1)
    acc["last"] = acc[["out_last", "in_last"]].max(axis=1)
    acc["span_h"] = (acc["last"] - acc["first"]) / HOUR
    acc["passthrough"] = np.where(acc.in_usd > 0, acc.out_usd / acc.in_usd.clip(lower=1e-9), np.nan)
    acc["n_total"] = acc.out_n + acc.in_n

    g = nx.DiGraph()
    for (s, r), w in e.groupby(["sender", "receiver"]).usd.sum().items():
        g.add_edge(s, r, weight=w)
    pr = nx.pagerank(g, weight="weight")
    und = g.to_undirected()
    clustering = nx.clustering(und)
    core = nx.core_number(und)
    recip = {n: (sum(1 for m in g.successors(n) if g.has_edge(m, n)) / max(1, g.out_degree(n))) for n in g.nodes}
    acc["pagerank"] = acc.index.map(pr).astype(float)
    acc["clustering"] = acc.index.map(clustering).astype(float)
    acc["kcore"] = acc.index.map(core).astype(float)
    acc["reciprocity"] = acc.index.map(recip).astype(float)
    hub_cut = acc.n_total.quantile(0.98)
    acc["is_hub"] = (acc.n_total >= hub_cut).astype(int)
    return acc


def build_features(e: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, AccountIndex]:
    idx = AccountIndex(e)
    acc = account_table(e)
    n = len(e)
    t = e.t.to_numpy()
    v = e.usd.to_numpy()
    S = e.sender.to_numpy()
    R = e.receiver.to_numpy()

    cols = defaultdict(lambda: np.zeros(n))
    pair_counts = e.groupby(["sender", "receiver"]).size()
    pair_first = e.groupby(["sender", "receiver"]).t.min()

    for i in range(n):
        s, r, ti, vi = S[i], R[i], t[i], v[i]

        # Receiver forwarding (layering / relay)
        ft, fu, fto, _ = idx.sends(r, ti, ti + DAY, lo_open=True)
        for w, name in ((10 * MIN, "10m"), (HOUR, "1h"), (6 * HOUR, "6h"), (DAY, "24h")):
            m = ft <= ti + w
            cols[f"r_fwd_n_{name}"][i] = m.sum()
            cols[f"r_fwd_ratio_{name}"][i] = fu[m].sum() / vi if vi > 0 else 0
        cols["r_fwd_best_ratio_err"][i] = _best_ratio(fu, vi)
        cols["r_fwd_delay_min"][i] = (ft[0] - ti) / MIN if len(ft) else 9999
        cols["r_fwd_distinct_24h"][i] = len(set(fto))

        # Sender pre-funding (pass-through)
        pt, pu, pfrom, _ = idx.gets(s, ti - DAY, ti, hi_open=True)
        for w, name in ((HOUR, "1h"), (6 * HOUR, "6h"), (DAY, "24h")):
            m = pt >= ti - w
            cols[f"s_recv_n_{name}"][i] = m.sum()
            cols[f"s_recv_ratio_{name}"][i] = pu[m].sum() / vi if vi > 0 else 0
        cols["s_recv_best_ratio_err"][i] = _best_ratio(pu, vi)
        cols["s_recv_delay_min"][i] = (ti - pt[-1]) / MIN if len(pt) else 9999
        cols["s_recv_distinct_24h"][i] = len(set(pfrom))

        # Bursts (fan-out from sender / fan-in to receiver)
        for w, name in ((10 * MIN, "10m"), (HOUR, "1h"), (6 * HOUR, "6h")):
            bt, bu, bto, _ = idx.sends(s, ti - w, ti + w)
            cols[f"s_burst_n_{name}"][i] = len(bt)
            cols[f"s_burst_distinct_{name}"][i] = len(set(bto))
            if name == "1h" and len(bu) > 1:
                cols["s_burst_amount_cv_1h"][i] = bu.std() / max(bu.mean(), 1e-9)
            gt, gu, gfrom, _ = idx.gets(r, ti - w, ti + w)
            cols[f"r_fanin_n_{name}"][i] = len(gt)
            cols[f"r_fanin_distinct_{name}"][i] = len(set(gfrom))

        # Round trip: money returns to sender within 72h via 1 or 2 intermediaries
        back2 = back3 = 0
        for xt, x in zip(ft[:40], fto[:40]):
            if x == s:
                back2 = 1
                continue
            yt, _, yto, _ = idx.sends(x, xt, ti + 3 * DAY, lo_open=True)
            if s in set(yto):
                back3 = 1
        cols["cycle_2hop"][i] = back2
        cols["cycle_3hop"][i] = back3

        # Convergence: accounts the sender funded in the last 24h that pay the same place the receiver pays
        sent_before = set(idx.sends(s, ti - DAY, ti + HOUR)[2]) - {r}
        targets = set(fto) - {s}
        conv = 0
        if sent_before and targets:
            for y in sorted(targets)[:20]:  # sorted: set order varies with PYTHONHASHSEED
                _, _, yfrom, _ = idx.gets(y, ti - DAY, ti + 2 * DAY)
                conv += len(sent_before & set(yfrom))
        cols["convergence"][i] = conv
        cols["s_sent_distinct_24h"][i] = len(sent_before)

    f = pd.DataFrame(dict(cols))
    f["log_usd"] = np.log1p(v)
    f["n_legs"] = e.n_legs.to_numpy()
    f["is_conversion"] = (e.n_legs > 1).astype(int).to_numpy()
    f["is_bridge"] = (e.n_legs > 2).astype(int).to_numpy()
    f["cross_currency"] = (e.src_asset.str.split(":").str[-1] != e.dst_asset.str.split(":").str[-1]).astype(int).to_numpy()
    f["cross_issuer"] = ((e.src_asset != e.dst_asset) & (e.src_asset != "XRP") & (e.dst_asset != "XRP")).astype(int).to_numpy()
    f["conversion_cost"] = np.where(e.usd > 0, e.src_usd / e.usd.clip(lower=1e-9) - 1, 0)
    f["dst_xrp"] = (e.dst_asset == "XRP").astype(int).to_numpy()
    units = e.dst_units.to_numpy()
    f["round_units_1"] = (np.abs(units - np.round(units)) < 1e-6).astype(int)
    f["round_units_100"] = (np.abs(units / 100 - np.round(units / 100)) < 1e-6).astype(int)
    f["round_usd_100"] = (np.abs(v / 100 - np.round(v / 100)) < 0.02).astype(int)
    f["hour"] = (t % DAY) // HOUR
    f["fee"] = e.fee.to_numpy()

    for prefix, col in (("s", S), ("r", R)):
        a = acc.loc[col]
        for c in ["out_n", "in_n", "out_cp", "in_cp", "out_usd", "in_usd", "span_h", "passthrough",
                  "pagerank", "clustering", "kcore", "reciprocity", "is_hub", "out_conv"]:
            f[f"{prefix}_{c}"] = a[c].to_numpy()
        f[f"{prefix}_age_h"] = (t - a["first"].to_numpy()) / HOUR
        f[f"{prefix}_remaining_h"] = (a["last"].to_numpy() - t) / HOUR
    sa = acc.loc[S]
    f["s_amount_z"] = (np.log1p(v) - sa.out_logmean.to_numpy()) / (sa.out_logstd.to_numpy() + 0.1)

    keys = list(zip(S, R))
    f["pair_n"] = [pair_counts[k] for k in keys]
    f["pair_reverse_n"] = [pair_counts.get((r, s), 0) for s, r in keys]
    f["pair_is_first"] = (t == np.array([pair_first[k] for k in keys])).astype(int)
    f = f.replace([np.inf, -np.inf], np.nan)
    f.index = e.hash.to_numpy()
    return f, acc, idx
