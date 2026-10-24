"""fe11_hard_med.py: Targeted detectors for:
1) Hard relay scenarios (aggregate_relay 1-hop and distributed_relay 2-hop):
   - Exact generator lag bands:
     * aggregate_relay (S -> M -> D): hold lag [3.5h, 6.0h] (12600s..21600s)
     * distributed_relay (S -> M1 -> M2 -> D):
       - M1 -> M2 link has pair_n in {3, 4} with inter-payment gap >= 20h
       - S -> M1 happens [1.5h, 3.5h] (5400s..12600s) BEFORE M1 -> M2
       - M2 -> D happens [1.5h, 3.5h] (5400s..12600s) AFTER M1 -> M2
   - Forward vs Reverse asymmetry:
     * In real relay laundering, M receives in [t-6h, t] and sends in [t, t+6h] (forward flow),
       whereas in 50% of decoys M sends FIRST and gets refilled [3.5h, 6.0h] AFTER.
   - Multi-episode consistency across an account's 3-4 relay episodes in the week:
     * Fraction of episodes that are strictly forward (receive before send)
     * Minimum and median episode conservation ratio (out_usd / in_usd)
     * Running balance non-overdraft signature
2) Medium motif scenarios (stretched 30m - 8h closed novel-edge clusters):
   - Extended 2-hop closed novel-edge clusters at 1h, 2h, 3h, 4h, 6h, 8h (with 2-hop closure enabled!)
   - End-to-end pure-source to pure-sink USD conservation (c_e2e_ratio, c_e2e_err) — fixes bipartite/fan_in/fan_out where mids=0
   - Stack / Bipartite / Multi-stage topology indicators (n_srcs, n_snks, n_mids, edge-to-node density, intermediary pass-through)
   - Step-timing regularity (dt_cv, dt_med, min-to-max dt ratio)
"""
from __future__ import annotations
import sys, time
from collections import defaultdict
import numpy as np
import pandas as pd

HOUR = 3600


def _novel_incidences(e: pd.DataFrame, max_pair: int = 1):
    pair_n = e.groupby(["sender", "receiver"]).hash.transform("size").to_numpy()
    novel = pair_n <= max_pair
    t = e.t.to_numpy()
    S = e.sender.to_numpy()
    R = e.receiver.to_numpy()
    inc = defaultdict(list)
    for i in np.flatnonzero(novel):
        inc[S[i]].append((t[i], R[i], i))
        inc[R[i]].append((t[i], S[i], i))
    inc = {
        a: (
            np.array([x[0] for x in l]),
            np.array([x[1] for x in l]),
            np.array([x[2] for x in l]),
        )
        for a, l in inc.items()
    }
    return novel, inc


def _win_rows(inc: dict, a: str, lo: int, hi: int):
    if a not in inc:
        return np.array([], dtype=object), np.array([], int)
    ts, oth, rows = inc[a]
    s = slice(np.searchsorted(ts, lo, "left"), np.searchsorted(ts, hi, "right"))
    return oth[s], rows[s]


def build_medium_cluster_features(e: pd.DataFrame) -> pd.DataFrame:
    """Extended 2-hop closed-cluster features up to 8h with pure-source->pure-sink conservation and topology."""
    idx_hash = e.hash.to_numpy()
    e = e.reset_index(drop=True)
    novel, inc = _novel_incidences(e, max_pair=1)
    t = e.t.to_numpy()
    S = e.sender.to_numpy()
    R = e.receiver.to_numpy()
    v = e.usd.to_numpy()
    vp = e.src_usd.fillna(e.usd).to_numpy()
    conv = (e.n_legs > 1).astype(float).to_numpy()
    cross_c = (e.src_asset != e.dst_asset).astype(float).to_numpy()

    out = defaultdict(lambda: np.zeros(len(e), dtype=np.float32))
    # Windows specifically covering easy (5m, 15m) + medium (45m, 90m, 3h, 5h, 8h) with 2-hop closure enabled on all!
    for w_sec, w_name in ((900, "15m"), (2700, "45m"), (5400, "90m"), (10800, "3h"), (18000, "5h"), (28800, "8h")):
        for i in range(len(e)):
            lo, hi = t[i] - w_sec, t[i] + w_sec
            so, sr = _win_rows(inc, S[i], lo, hi)
            ro, rr = _win_rows(inc, R[i], lo, hi)
            nodes = set(so) | set(ro) | {S[i], R[i]}
            rows = set(sr) | set(rr) | ({i} if novel[i] else set())
            for a in sorted(nodes - {S[i], R[i]})[:30]:
                oo, orows = _win_rows(inc, a, lo, hi)
                for o, r_ in zip(oo, orows):
                    if o in nodes:
                        rows.add(r_)
                    elif len(nodes) <= 20:  # Enable 2-hop closure across ALL windows up to 8h!
                        oo2, _ = _win_rows(inc, o, lo, hi)
                        if any(o2 in nodes for o2 in oo2):
                            rows.add(r_)
            if not rows:
                out[f"mc_e2e_err_{w_name}"][i] = -1.0
                out[f"mc_pt_err_{w_name}"][i] = -1.0
                continue
            idx = list(rows)
            n_edges = len(idx)
            cs, cr, cu, cp_ = S[idx], R[idx], v[idx], vp[idx]
            s_set, r_set = set(cs), set(cr)
            all_accts = s_set | r_set
            n_accts = len(all_accts)
            mids = s_set & r_set
            srcs = s_set - r_set
            snks = r_set - s_set

            out[f"mc_size_{w_name}"][i] = n_edges
            out[f"mc_accts_{w_name}"][i] = n_accts
            out[f"mc_dens_{w_name}"][i] = n_edges / max(n_accts, 1)
            out[f"mc_mids_{w_name}"][i] = len(mids)
            out[f"mc_srcs_{w_name}"][i] = len(srcs)
            out[f"mc_snks_{w_name}"][i] = len(snks)
            out[f"mc_is_bipartite_{w_name}"][i] = float(len(mids) == 0 and len(srcs) >= 2 and len(snks) >= 2)
            out[f"mc_is_cycle_{w_name}"][i] = float(len(srcs) == 0 and len(snks) == 0 and len(mids) >= 3)
            out[f"mc_is_stack_{w_name}"][i] = float(len(srcs) >= 2 and len(snks) >= 2 and len(mids) >= 2 and n_edges >= 8)

            # End-to-end pure source -> pure sink conservation
            if srcs and snks:
                v_src = cp_[np.isin(cs, list(srcs))].sum()
                v_snk = cu[np.isin(cr, list(snks))].sum()
                e2e_r = v_snk / max(v_src, 1e-9)
                out[f"mc_e2e_ratio_{w_name}"][i] = min(e2e_r, 5.0)
                out[f"mc_e2e_err_{w_name}"][i] = min(abs(e2e_r - 1.0), 5.0)
            else:
                out[f"mc_e2e_ratio_{w_name}"][i] = -1.0
                out[f"mc_e2e_err_{w_name}"][i] = -1.0

            # Intermediary conservation (using src_usd / deliv_usd)
            if mids:
                ratios = [cp_[cs == m].sum() / max(cu[cr == m].sum(), 1e-9) for m in mids]
                out[f"mc_pt_med_{w_name}"][i] = min(float(np.median(ratios)), 5.0)
                out[f"mc_pt_err_{w_name}"][i] = min(float(np.median(np.abs(np.array(ratios) - 1.0))), 5.0)
            else:
                out[f"mc_pt_med_{w_name}"][i] = -1.0
                out[f"mc_pt_err_{w_name}"][i] = -1.0

            if n_edges > 1:
                ts_c = np.sort(t[idx])
                dts = np.diff(ts_c)
                span_m = (ts_c[-1] - ts_c[0]) / 60.0
                out[f"mc_span_m_{w_name}"][i] = span_m
                out[f"mc_dt_med_{w_name}"][i] = float(np.median(dts)) / 60.0
                out[f"mc_dt_max_{w_name}"][i] = float(np.max(dts)) / 60.0
                out[f"mc_dt_cv_{w_name}"][i] = float(np.std(dts) / (np.mean(dts) + 1e-9))
                out[f"mc_usd_cv_{w_name}"][i] = float(np.std(cu) / (np.mean(cu) + 1e-9))
                out[f"mc_usd_med_{w_name}"][i] = float(np.log1p(np.median(cu)))

    return pd.DataFrame(dict(out), index=idx_hash)


def build_hard_relay_features(e: pd.DataFrame) -> pd.DataFrame:
    """Exact lag-band, direction-asymmetry, and 2-hop bridge detectors for aggregate_relay & distributed_relay."""
    idx_hash = e.hash.to_numpy()
    e = e.reset_index(drop=True)
    n = len(e)
    t = e.t.to_numpy()
    S = e.sender.to_numpy()
    R = e.receiver.to_numpy()
    v = e.usd.to_numpy()
    vp = e.src_usd.fillna(e.usd).to_numpy()

    # Build time-sorted per-account inflows and outflows
    in_t, in_v, in_s, in_row = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    out_t, out_v, out_r, out_row = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    for i in range(n):
        s, r, ti, vi, vpi = S[i], R[i], t[i], v[i], vp[i]
        in_t[r].append(ti); in_v[r].append(vi); in_s[r].append(s); in_row[r].append(i)
        out_t[s].append(ti); out_v[s].append(vpi); out_r[s].append(r); out_row[s].append(i)
    arr = lambda d: {k: np.asarray(x) for k, x in d.items()}
    in_t, in_v, in_s, in_row = map(arr, (in_t, in_v, in_s, in_row))
    out_t, out_v, out_r, out_row = map(arr, (out_t, out_v, out_r, out_row))
    E = np.array([], dtype=float)
    EI = np.array([], dtype=int)

    def win_v(tt, vv, lo, hi):
        if tt is None:
            return E
        a, b = np.searchsorted(tt, lo, "left"), np.searchsorted(tt, hi, "right")
        return vv[a:b]

    def win_rows(tt, rr, lo, hi):
        if tt is None:
            return EI
        a, b = np.searchsorted(tt, lo, "left"), np.searchsorted(tt, hi, "right")
        return rr[a:b]

    # Identify candidate M1 -> M2 bridges for distributed_relay:
    # pair_n in {3, 4} and minimum gap between consecutive payments >= 20h (72000s)
    pair_groups = defaultdict(list)
    for i in range(n):
        pair_groups[(S[i], R[i])].append(i)
    is_bridge_edge = np.zeros(n, dtype=np.float32)
    bridge_ep_fwd = np.zeros(n, dtype=np.float32)
    bridge_ep_rev = np.zeros(n, dtype=np.float32)
    bridge_ep_ratio = np.full(n, -1.0, dtype=np.float32)
    bridge_scen_strict = np.zeros(n, dtype=np.float32)
    bridge_scen_min_r = np.full(n, -1.0, dtype=np.float32)

    # Also propagate bridge scenario scores to the S->M1 and M2->D edges in the [1.5h, 3.3h] windows!
    dr_leg_strict = np.zeros(n, dtype=np.float32)
    dr_leg_min_r = np.full(n, -1.0, dtype=np.float32)
    dr_leg_role = np.zeros(n, dtype=np.float32)  # 1=in_leg, 2=bridge, 3=out_leg

    for (m1, m2), rows in pair_groups.items():
        if len(rows) not in (3, 4):
            continue
        ts = t[rows]
        if np.min(np.diff(ts)) < 20 * HOUR:
            continue
        is_bridge_edge[rows] = 1.0
        # For each of the 3-4 episodes of (m1 -> m2) at time tb:
        # Forward hypothesis: S -> m1 in [tb - 3.3h, tb - 1.5h] and m2 -> D in [tb + 1.5h, tb + 3.3h]
        # Reverse hypothesis: m2 -> D in [tb - 3.3h, tb - 1.5h] and S -> m1 in [tb + 1.5h, tb + 3.3h]
        ep_fwds, ep_ratios = [], []
        ep_in_rows, ep_out_rows = [], []
        for r_idx in rows:
            tb = t[r_idx]
            vb = v[r_idx]
            in_fwd = win_v(in_t.get(m1), in_v.get(m1), tb - int(3.3 * HOUR), tb - int(1.5 * HOUR))
            out_fwd = win_v(out_t.get(m2), out_v.get(m2), tb + int(1.5 * HOUR), tb + int(3.3 * HOUR))
            in_rev = win_v(in_t.get(m1), in_v.get(m1), tb + int(1.5 * HOUR), tb + int(3.3 * HOUR))
            out_rev = win_v(out_t.get(m2), out_v.get(m2), tb - int(3.3 * HOUR), tb - int(1.5 * HOUR))

            vf_in, vf_out = in_fwd.sum(), out_fwd.sum()
            vr_in, vr_out = in_rev.sum(), out_rev.sum()
            # Check match of vf_in to vb and vf_out to vb
            fwd_ok = float(vf_in >= 0.5 * vb and vf_out >= 0.4 * vb and vf_in > vr_in)
            rev_ok = float(vr_in >= 0.5 * vb and vr_out >= 0.4 * vb and vr_in > vf_in)
            ratio = min(vf_out / max(vf_in, 1e-9), 5.0) if vf_in > 0 else 0.0
            bridge_ep_fwd[r_idx] = fwd_ok
            bridge_ep_rev[r_idx] = rev_ok
            bridge_ep_ratio[r_idx] = ratio
            ep_fwds.append(fwd_ok)
            ep_ratios.append(ratio)

            ir = win_rows(in_t.get(m1), in_row.get(m1), tb - int(3.3 * HOUR), tb - int(1.5 * HOUR))
            or_ = win_rows(out_t.get(m2), out_row.get(m2), tb + int(1.5 * HOUR), tb + int(3.3 * HOUR))
            ep_in_rows.extend(ir)
            ep_out_rows.extend(or_)

        sc_strict = float(np.mean(ep_fwds))
        sc_min_r = float(np.min(ep_ratios)) if ep_ratios else -1.0
        bridge_scen_strict[rows] = sc_strict
        bridge_scen_min_r[rows] = sc_min_r
        dr_leg_role[rows] = 2.0
        dr_leg_strict[rows] = np.maximum(dr_leg_strict[rows], sc_strict)
        dr_leg_min_r[rows] = np.maximum(dr_leg_min_r[rows], sc_min_r)
        for idx_ in ep_in_rows:
            if dr_leg_role[idx_] == 0: dr_leg_role[idx_] = 1.0
            dr_leg_strict[idx_] = max(dr_leg_strict[idx_], sc_strict)
            dr_leg_min_r[idx_] = max(dr_leg_min_r[idx_], sc_min_r)
        for idx_ in ep_out_rows:
            if dr_leg_role[idx_] == 0: dr_leg_role[idx_] = 3.0
            dr_leg_strict[idx_] = max(dr_leg_strict[idx_], sc_strict)
            dr_leg_min_r[idx_] = max(dr_leg_min_r[idx_], sc_min_r)

    # Now compute 1-hop aggregate_relay exact lag-band features [3.6h, 5.8h] for EVERY transaction:
    # Case 1: i is an inflow S -> M (so R[i] is M).
    #   - Co-inflows to R[i] in [t[i]-1h, t[i]+1h]
    #   - Forward outflows from R[i] in [t[i]+3.6h, t[i]+5.8h]
    #   - Reverse outflows from R[i] in [t[i]-5.8h, t[i]-3.6h] (if R[i] sent first and is now being refilled!)
    # Case 2: i is an outflow M -> D (so S[i] is M).
    #   - Co-outflows from S[i] in [t[i]-1h, t[i]+1h]
    #   - Forward inflows to S[i] in [t[i]-5.8h, t[i]-3.6h] (received 3.6-5.8h before sending!)
    #   - Reverse inflows to S[i] in [t[i]+3.6h, t[i]+5.8h] (refilled 3.6-5.8h after sending!)
    ar_r_pool_1h = np.zeros(n, dtype=np.float32)
    ar_r_fwd_lag = np.zeros(n, dtype=np.float32)
    ar_r_rev_lag = np.zeros(n, dtype=np.float32)
    ar_r_asym = np.zeros(n, dtype=np.float32)
    ar_r_err = np.full(n, 5.0, dtype=np.float32)

    ar_s_pool_1h = np.zeros(n, dtype=np.float32)
    ar_s_fwd_lag = np.zeros(n, dtype=np.float32)
    ar_s_rev_lag = np.zeros(n, dtype=np.float32)
    ar_s_asym = np.zeros(n, dtype=np.float32)
    ar_s_err = np.full(n, 5.0, dtype=np.float32)

    L_LO, L_HI = int(3.6 * HOUR), int(5.8 * HOUR)
    for i in range(n):
        s, r, ti = S[i], R[i], t[i]
        # Receiver R[i] as relay hub M
        r_pool = win_v(in_t.get(r), in_v.get(r), ti - HOUR, ti + HOUR).sum()
        r_fwd = win_v(out_t.get(r), out_v.get(r), ti + L_LO, ti + L_HI).sum()
        r_rev = win_v(out_t.get(r), out_v.get(r), ti - L_HI, ti - L_LO).sum()
        ar_r_pool_1h[i] = np.log1p(r_pool)
        ar_r_fwd_lag[i] = min(r_fwd / max(r_pool, 1e-9), 5.0)
        ar_r_rev_lag[i] = min(r_rev / max(r_pool, 1e-9), 5.0)
        ar_r_asym[i] = (r_fwd - r_rev) / max(r_fwd + r_rev, 1.0)
        if r_pool > 0 and r_fwd > 0:
            ar_r_err[i] = min(abs(r_fwd / r_pool - 0.90), 5.0)

        # Sender S[i] as relay hub M
        s_pool = win_v(out_t.get(s), out_v.get(s), ti - HOUR, ti + HOUR).sum()
        s_fwd = win_v(in_t.get(s), in_v.get(s), ti - L_HI, ti - L_LO).sum()
        s_rev = win_v(in_t.get(s), in_v.get(s), ti + L_LO, ti + L_HI).sum()
        ar_s_pool_1h[i] = np.log1p(s_pool)
        ar_s_fwd_lag[i] = min(s_fwd / max(s_pool, 1e-9), 5.0)
        ar_s_rev_lag[i] = min(s_rev / max(s_pool, 1e-9), 5.0)
        ar_s_asym[i] = (s_fwd - s_rev) / max(s_fwd + s_rev, 1.0)
        if s_fwd > 0 and s_pool > 0:
            ar_s_err[i] = min(abs(s_pool / s_fwd - 0.90), 5.0)

    df = pd.DataFrame({
        "hr_is_bridge": is_bridge_edge,
        "hr_bridge_ep_fwd": bridge_ep_fwd,
        "hr_bridge_ep_rev": bridge_ep_rev,
        "hr_bridge_ep_ratio": bridge_ep_ratio,
        "hr_bridge_scen_strict": bridge_scen_strict,
        "hr_bridge_scen_min_r": bridge_scen_min_r,
        "hr_dr_leg_role": dr_leg_role,
        "hr_dr_leg_strict": dr_leg_strict,
        "hr_dr_leg_min_r": dr_leg_min_r,
        "hr_ar_r_pool_1h": ar_r_pool_1h,
        "hr_ar_r_fwd_lag": ar_r_fwd_lag,
        "hr_ar_r_rev_lag": ar_r_rev_lag,
        "hr_ar_r_asym": ar_r_asym,
        "hr_ar_r_err": ar_r_err,
        "hr_ar_s_pool_1h": ar_s_pool_1h,
        "hr_ar_s_fwd_lag": ar_s_fwd_lag,
        "hr_ar_s_rev_lag": ar_s_rev_lag,
        "hr_ar_s_asym": ar_s_asym,
        "hr_ar_s_err": ar_s_err,
        "hr_ar_best_asym": np.maximum(ar_r_asym, ar_s_asym),
        "hr_ar_best_err": np.minimum(ar_r_err, ar_s_err),
    }, index=idx_hash)

    # Account-level aggregates of the relay signals (so all episodes of the same hub share context)
    for col in ("hr_ar_r_asym", "hr_ar_s_asym", "hr_ar_best_asym", "hr_ar_best_err", "hr_dr_leg_strict"):
        s_mean = df[col].groupby(S).transform("mean").to_numpy()
        r_mean = df[col].groupby(R).transform("mean").to_numpy()
        df[f"acs_{col}_mean"] = s_mean
        df[f"acr_{col}_mean"] = r_mean

    return df


def build_all(split_dir: str, name: str) -> pd.DataFrame:
    sys.path.insert(0, "prior_work/prototype_agent")
    from app.data import load_split
    sp = load_split(split_dir, name)
    e = sp.edges
    f_mc = build_medium_cluster_features(e)
    f_hr = build_hard_relay_features(e)
    return pd.concat([f_mc, f_hr], axis=1)


if __name__ == "__main__":
    t0 = time.time()
    for s in ("train", "test"):
        e = pd.read_parquet(f"cache/proto_edges_{s}.parquet")
        f = pd.concat([build_medium_cluster_features(e), build_hard_relay_features(e)], axis=1)
        f.to_parquet(f"cache/fe11_{s}.parquet")
        print(f"fe11_{s}", f.shape, round(time.time() - t0, 1), flush=True)
