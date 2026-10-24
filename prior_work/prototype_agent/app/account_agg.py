"""Account-level aggregates: each account's motif/flow signals across ALL its payments this week, attached to
every payment as sender and receiver context (lets a per-payment model see account-level patterns)."""
from __future__ import annotations

import numpy as np
import pandas as pd

KEYS = ["m_cluster_edges_3h", "m_closed_3h", "m_density_3h", "m_is_novel", "m_cluster_edges_1h", "m_nodes_3h",
        "r_fwd_ratio_1h", "r_fwd_best_ratio_err", "s_recv_best_ratio_err", "s_burst_distinct_1h",
        "r_fanin_distinct_1h", "cycle_3hop", "convergence", "log_usd", "is_conversion", "pair_n"]


def account_aggregates(e: pd.DataFrame, F: pd.DataFrame) -> pd.DataFrame:
    keys = [k for k in KEYS if k in F.columns]
    vals = F[keys].reset_index(drop=True)
    long = pd.concat([vals.assign(acct=e.sender.to_numpy()), vals.assign(acct=e.receiver.to_numpy())])
    agg = long.groupby("acct")[keys].agg(["mean", "max"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    s = agg.reindex(e.sender.to_numpy()).add_prefix("acs_").set_axis(F.index)
    r = agg.reindex(e.receiver.to_numpy()).add_prefix("acr_").set_axis(F.index)
    return pd.concat([s, r], axis=1)
