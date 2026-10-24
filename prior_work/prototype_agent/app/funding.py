"""Funding-vs-payout features (targets fan_out / random twins): how a sender's payout burst compares with the
money it received just before and over the week."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import AccountIndex

MIN, HOUR, DAY = 60, 3600, 86400


def funding_features(e: pd.DataFrame) -> pd.DataFrame:
    idx = AccountIndex(e)
    t, v, S, R = e.t.to_numpy(), e.usd.to_numpy(), e.sender.to_numpy(), e.receiver.to_numpy()
    in_week = e.groupby("receiver").usd.sum()
    out_week = e.groupby("sender").usd.sum()
    n = len(e)
    f = {k: np.zeros(n) for k in ["burst_usd_10m", "burst_usd_1h", "funded_24h", "funded_6h", "burst10_over_funded24",
                                  "burst1h_over_funded24", "pay_over_week_in", "burst10_over_week_in",
                                  "r_recv_usd_10m", "r_recv10_over_week_out"]}
    s_week_in = pd.Series(S).map(in_week).fillna(0).to_numpy()
    r_week_out = pd.Series(R).map(out_week).fillna(0).to_numpy()
    for i in range(n):
        s, r, ti = S[i], R[i], t[i]
        b10 = idx.sends(s, ti - 10 * MIN, ti + 10 * MIN)[1].sum()
        b1h = idx.sends(s, ti - HOUR, ti + HOUR)[1].sum()
        f24 = idx.gets(s, ti - DAY, ti, hi_open=True)[1].sum()
        f6 = idx.gets(s, ti - 6 * HOUR, ti, hi_open=True)[1].sum()
        rr10 = idx.gets(r, ti - 10 * MIN, ti + 10 * MIN)[1].sum()
        f["burst_usd_10m"][i], f["burst_usd_1h"][i], f["funded_24h"][i], f["funded_6h"][i] = b10, b1h, f24, f6
        f["burst10_over_funded24"][i] = b10 / (f24 + 1)
        f["burst1h_over_funded24"][i] = b1h / (f24 + 1)
        f["pay_over_week_in"][i] = v[i] / (s_week_in[i] + 1)
        f["burst10_over_week_in"][i] = b10 / (s_week_in[i] + 1)
        f["r_recv_usd_10m"][i] = rr10
        f["r_recv10_over_week_out"][i] = rr10 / (r_week_out[i] + 1)
    return pd.DataFrame({f"fund_{k}": x for k, x in f.items()}).set_axis(e.hash.to_numpy())
