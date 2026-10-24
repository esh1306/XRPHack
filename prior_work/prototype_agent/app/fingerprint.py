"""Behavioural fingerprints (what forensic tracers use): how 'scripted' an account is — fee consistency and
working-hours concentration across all its payments — attached to each payment for sender and receiver."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _entropy(counts: np.ndarray) -> float:
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum())


def fingerprint_features(e: pd.DataFrame) -> pd.DataFrame:
    e = e.reset_index(drop=True)
    hour = (e.t % 86400) // 3600
    sent = pd.DataFrame({"acct": e.sender, "fee": e.fee, "hour": hour})
    g = sent.groupby("acct")
    acct = pd.DataFrame({
        "fee_mode_share": g.fee.agg(lambda f: f.value_counts(normalize=True).iloc[0]),
        "fee_entropy": g.fee.agg(lambda f: _entropy(f.value_counts().to_numpy())),
        "hour_entropy": g.hour.agg(lambda h: _entropy(h.value_counts().to_numpy())),
        "afternoon_share": g.hour.agg(lambda h: ((h >= 12) & (h < 18)).mean()),
        "night_share": g.hour.agg(lambda h: (h < 6).mean()),
    })
    s = acct.reindex(e.sender).add_prefix("fp_s_").reset_index(drop=True)
    r = acct.reindex(e.receiver).add_prefix("fp_r_").reset_index(drop=True)
    out = pd.concat([s, r], axis=1)
    out["fp_fee_is_sender_mode"] = (e.fee == e.groupby("sender").fee.transform(lambda f: f.mode().iloc[0])).astype(int)
    out["fp_hour_block"] = hour // 6
    return out.set_axis(e.hash.to_numpy())
