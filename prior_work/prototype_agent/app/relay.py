"""Relay-context features: a hub's surrounding traffic, split into the scenario's own thread.
Suspicious relays use new counterparties each episode and receive-then-send; normal relays repeat partners and
often spend first, then get refilled."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

HOUR, DAY = 3600, 86400


def relay_features(e: pd.DataFrame) -> pd.DataFrame:
    t = e.t.to_numpy(); v = e.usd.to_numpy(); S = e.sender.to_numpy(); R = e.receiver.to_numpy()
    n = len(e)
    pair_n = e.groupby(["sender", "receiver"]).hash.transform("size").to_numpy()
    # first time this pair appears (row order is time order)
    seen, first = set(), np.zeros(n, bool)
    for i in range(n):
        k = (S[i], R[i])
        first[i] = k not in seen
        seen.add(k)
    out_t, out_v, out_new, out_rep = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    in_t, in_v, in_new, in_rep = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    for i in range(n):
        out_t[S[i]].append(t[i]); out_v[S[i]].append(v[i]); out_new[S[i]].append(first[i]); out_rep[S[i]].append(pair_n[i] > 1)
        in_t[R[i]].append(t[i]); in_v[R[i]].append(v[i]); in_new[R[i]].append(first[i]); in_rep[R[i]].append(pair_n[i] > 1)
    arr = lambda d: {k: np.asarray(x) for k, x in d.items()}
    out_t, out_v, out_new, out_rep, in_t, in_v, in_new, in_rep = map(arr, (out_t, out_v, out_new, out_rep, in_t, in_v, in_new, in_rep))
    E = np.array([])

    def win(tt, lo, hi, *cols):
        if tt is None:
            return [E] * len(cols)
        a, b = np.searchsorted(tt, lo, "left"), np.searchsorted(tt, hi, "right")
        return [c[a:b] for c in cols]

    f = defaultdict(lambda: np.zeros(n))
    for i in range(n):
        s, r, ti, vi = S[i], R[i], t[i], max(v[i], 1e-9)
        for w, nm in ((3 * HOUR, "3h"), (6 * HOUR, "6h")):
            # receiver: spent before this payment (refill) vs forwards after (relay)
            rb, = win(out_t.get(r), ti - w, ti - 1, out_v.get(r))
            ra, ran = win(out_t.get(r), ti + 1, ti + w, out_v.get(r), out_new.get(r))
            f[f"r_spent_before_{nm}"][i] = rb.sum() / vi
            f[f"r_sent_after_{nm}"][i] = ra.sum() / vi
            f[f"r_order_{nm}"][i] = (ra.sum() - rb.sum()) / vi
            f[f"r_after_new_share_{nm}"][i] = ran.mean() if len(ran) else -1
            # sender: funded before (relay) vs refilled after (spend-then-refill)
            sb, sbn = win(in_t.get(s), ti - w, ti - 1, in_v.get(s), in_new.get(s))
            sa, = win(in_t.get(s), ti + 1, ti + w, in_v.get(s))
            f[f"s_funded_before_{nm}"][i] = sb.sum() / vi
            f[f"s_refilled_after_{nm}"][i] = sa.sum() / vi
            f[f"s_order_{nm}"][i] = (sb.sum() - sa.sum()) / vi
            f[f"s_before_new_share_{nm}"][i] = sbn.mean() if len(sbn) else -1
            # churn: share of first-time partners in the hub's surrounding window
            _, rin_new, rin_rep = win(in_t.get(r), ti - w, ti + w, in_v.get(r), in_new.get(r), in_rep.get(r))
            _, sout_new, sout_rep = win(out_t.get(s), ti - w, ti + w, out_v.get(s), out_new.get(s), out_rep.get(s))
            f[f"r_in_new_share_{nm}"][i] = rin_new.mean() if len(rin_new) else -1
            f[f"r_in_repeat_share_{nm}"][i] = rin_rep.mean() if len(rin_rep) else -1
            f[f"s_out_new_share_{nm}"][i] = sout_new.mean() if len(sout_new) else -1
            f[f"s_out_repeat_share_{nm}"][i] = sout_rep.mean() if len(sout_rep) else -1
    f["pair_first_time"] = first.astype(float)
    return pd.DataFrame(dict(f)).set_axis(e.hash.to_numpy())
