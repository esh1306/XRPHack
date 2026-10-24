"""Scenario reconstruction without labels: link one-off payments (pair seen once this week) that share an account
and happen within LINK seconds of each other into connected clusters, then describe each whole cluster."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd


def _clusters(e: pd.DataFrame, mask: np.ndarray, link: int) -> np.ndarray:
    parent = np.arange(len(e))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    t = e.t.to_numpy()
    by_acct = defaultdict(list)
    for i in np.flatnonzero(mask):
        by_acct[e.sender.iat[i]].append(i)
        by_acct[e.receiver.iat[i]].append(i)
    for rows in by_acct.values():
        rows.sort(key=lambda i: t[i])
        for a, b in zip(rows, rows[1:]):  # chain consecutive incidences within the link window
            if t[b] - t[a] <= link:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[ra] = rb
    return np.array([find(i) for i in range(len(e))])


def cluster_features(e: pd.DataFrame) -> pd.DataFrame:
    e = e.reset_index(drop=True)
    one_off = (e.groupby(["sender", "receiver"]).hash.transform("size") == 1).to_numpy()
    out = {}
    for link, nm in ((600, "10m"), (1200, "20m"), (3600, "1h")):
        cid = np.where(one_off, _clusters(e, one_off, link), -1 - np.arange(len(e)))
        d = e.assign(cid=cid)
        g = d.groupby("cid")
        size = g.hash.transform("size")
        span = (g.t.transform("max") - g.t.transform("min")) / 60
        accts = g.apply(lambda x: len(set(x.sender) | set(x.receiver)), include_groups=False)
        max_out = g.apply(lambda x: x.sender.value_counts().max(), include_groups=False)
        max_in = g.apply(lambda x: x.receiver.value_counts().max(), include_groups=False)
        mid = g.apply(lambda x: len(set(x.sender) & set(x.receiver)), include_groups=False)  # accounts both paid and paying
        cv = g.usd.transform(lambda u: u.std(ddof=0) / max(u.mean(), 1e-9))
        out[f"c_size_{nm}"] = size.to_numpy()
        out[f"c_span_min_{nm}"] = span.to_numpy()
        out[f"c_rate_{nm}"] = (size / (span + 1)).to_numpy()
        out[f"c_accounts_{nm}"] = d.cid.map(accts).to_numpy()
        out[f"c_max_out_{nm}"] = d.cid.map(max_out).to_numpy()
        out[f"c_max_in_{nm}"] = d.cid.map(max_in).to_numpy()
        out[f"c_intermediaries_{nm}"] = d.cid.map(mid).to_numpy()
        out[f"c_amount_cv_{nm}"] = cv.to_numpy()
        out[f"c_edges_per_acct_{nm}"] = out[f"c_size_{nm}"] / np.maximum(out[f"c_accounts_{nm}"], 1)
    return pd.DataFrame(out).set_axis(e.hash.to_numpy())
