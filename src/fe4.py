"""v3 features (label-free, per split): in->out amount matching after fees/FX (USD), multi-hop chain
following with amount decay / peel chains, out-of-character deviation vs the account's own baseline,
counterparty-of-counterparty novelty, cohort synchrony, and episode-level aggregates broadcast to txs."""
import sys, warnings, numpy as np, pandas as pd
from collections import defaultdict
warnings.filterwarnings("ignore")
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from fe import load_split


def build(tx, comp):
    S = tx.ACCOUNT.to_numpy(); R = tx.DEST.to_numpy(); T = tx.TIMESTAMP.to_numpy().astype(np.int64)
    V = np.nan_to_num(tx.usd.to_numpy().astype(float)); n = len(tx); F = {}
    out_i, in_i = defaultdict(list), defaultdict(list)
    for i in range(n): out_i[S[i]].append(i); in_i[R[i]].append(i)
    out_i = {k: np.array(v) for k, v in out_i.items()}; in_i = {k: np.array(v) for k, v in in_i.items()}
    E = np.array([], int)
    # 1) in->out match
    for w in (3600, 21600):
        bm = np.full(n, 9.0); cm = np.zeros(n); fm = np.full(n, 9.0); fc = np.zeros(n)
        for i in range(n):
            ii = in_i.get(S[i], E); m = ii[(T[ii] <= T[i]) & (T[ii] >= T[i] - w) & (ii != i)]
            if len(m) and V[i] > 0:
                r = np.abs(V[m] - V[i]) / np.maximum(V[m], 1e-9); bm[i] = min(r.min(), 9); cm[i] = (r < 0.05).sum()
                tot = V[m].sum(); fm[i] = min(abs(tot - V[i]) / max(tot, 1e-9), 9)
            oo = out_i.get(R[i], E); m = oo[(T[oo] >= T[i]) & (T[oo] <= T[i] + w) & (oo != i)]
            if len(m) and V[i] > 0:
                r = np.abs(V[m] - V[i]) / V[i]; fc[i] = (r < 0.05).sum()
                F.setdefault(f"fwd_best{w}", np.full(n, 9.0))[i] = min(r.min(), 9)
                F.setdefault(f"fwd_sumratio{w}", np.full(n, 9.0))[i] = min(V[m].sum() / V[i], 9)
        F[f"in_match_best{w}"] = bm; F[f"in_match_cnt{w}"] = cm; F[f"in_sum_match{w}"] = fm; F[f"fwd_cnt{w}"] = fc
        F.setdefault(f"fwd_best{w}", np.full(n, 9.0)); F.setdefault(f"fwd_sumratio{w}", np.full(n, 9.0))
    # 2) chain following (forward/backward, amount within [0.7, 1.02] per hop, within 6h)
    def follow(i, fwd):
        L = 0; cur = i; ratio = 1.0; seen = {S[i], R[i]}
        while L < 8:
            if fwd:
                oo = out_i.get(R[cur], E); m = oo[(T[oo] >= T[cur]) & (T[oo] <= T[cur] + 21600) & (oo != cur)]
                if not len(m) or V[cur] <= 0: break
                r = V[m] / V[cur]
            else:
                ii = in_i.get(S[cur], E); m = ii[(T[ii] <= T[cur]) & (T[ii] >= T[cur] - 21600) & (ii != cur)]
                if not len(m) or V[cur] <= 0: break
                r = V[cur] / np.maximum(V[m], 1e-9)
            ok = (r >= 0.7) & (r <= 1.02)
            if not ok.any(): break
            j = m[ok][np.argmin(np.abs(1 - r[ok]))]
            nxt = R[j] if fwd else S[j]
            if nxt in seen: L += 1; break
            seen.add(nxt); ratio *= (V[j] / V[cur]) if fwd else (V[cur] / max(V[j], 1e-9)); cur = j; L += 1
        return L, ratio, len(seen)
    cf = np.zeros(n); cb = np.zeros(n); rf = np.ones(n); rb = np.ones(n); acc = np.zeros(n)
    for i in range(n):
        a, b, c = follow(i, True); d, e, f = follow(i, False)
        cf[i], rf[i], cb[i], rb[i], acc[i] = a, b, d, e, c + f
    F.update(chain_fwd=cf, chain_bwd=cb, chain_len=cf + cb, chain_decay_fwd=rf, chain_decay_bwd=rb, chain_acc=acc,
             chain_decay_per_hop=np.where(cf + cb > 0, (rf * rb) ** (1 / np.maximum(cf + cb, 1)), 1))
    # peel: sender's consecutive outflows with decreasing amounts
    peel = np.zeros(n)
    for a, ix in out_i.items():
        ix = ix[np.argsort(T[ix], kind="stable")]; run = 0
        for k, i in enumerate(ix):
            run = run + 1 if k and V[i] < V[ix[k - 1]] and T[i] - T[ix[k - 1]] < 86400 else 0; peel[i] = run
    F["peel_run"] = peel
    # 3) out-of-character (leave-one-out)
    lv = np.log1p(V); hr = (T % 86400) / 3600.0
    for role, idx, col in (("s", out_i, S), ("r", in_i, R)):
        z = np.zeros(n); hd = np.zeros(n); gz = np.zeros(n); cnt = np.zeros(n)
        for a, ix in idx.items():
            k = len(ix); cnt[ix] = k
            if k < 3: continue
            x = lv[ix]; s1 = x.sum(); s2 = (x ** 2).sum()
            mu = (s1 - x) / (k - 1); var = np.maximum((s2 - x ** 2) / (k - 1) - mu ** 2, 1e-4)
            z[ix] = (x - mu) / np.sqrt(var)
            ang = hr[ix] / 24 * 2 * np.pi; cs, sn = np.cos(ang).sum() - np.cos(ang), np.sin(ang).sum() - np.sin(ang)
            hd[ix] = 1 - np.cos(ang - np.arctan2(sn, cs))
            o = np.argsort(T[ix], kind="stable"); t = T[ix][o]; gp = np.diff(t, prepend=t[0] - 1e6).astype(float)
            mg = np.median(gp[1:]) + 1; gz[ix[o]] = np.log1p(gp) - np.log(mg)
        F[f"{role}_ooc_amt_z"] = z; F[f"{role}_ooc_hour"] = hd; F[f"{role}_ooc_gap"] = gz
    # 4) counterparty-of-counterparty novelty + 5) cohort synchrony
    order = np.argsort(T, kind="stable"); nb = defaultdict(set); first = {}
    common = np.zeros(n); pair_new = np.zeros(n); r_new = np.zeros(n); s_new = np.zeros(n)
    for i in order:
        s, r = S[i], R[i]; ns, nr = nb[s], nb[r]
        pair_new[i] = r not in ns; common[i] = len(ns & nr) if len(ns) < len(nr) else len(nr & ns)
        s_new[i] = s not in first; r_new[i] = r not in first
        first.setdefault(s, T[i]); first.setdefault(r, T[i]); ns.add(r); nr.add(s)
    F.update(coc_common=common, coc_novel=((pair_new == 1) & (common == 0)).astype(float),
             coc_bridge=((pair_new == 1) & (common > 0)).astype(float), s_first_tx=s_new, r_first_tx=r_new)
    fs = np.array(sorted(first.values()))
    for role, col in (("s", S), ("r", R)):
        ft = np.array([first[a] for a in col])
        F[f"{role}_birth_cohort1h"] = np.searchsorted(fs, ft + 3600, "right") - np.searchsorted(fs, ft - 3600, "left")
        F[f"{role}_age_h"] = np.log1p((T - ft) / 3600)
    ts = T[order]; lvs = lv[order]
    sync = np.zeros(n)
    for k, i in enumerate(order):
        a, b = np.searchsorted(ts, T[i] - 120), np.searchsorted(ts, T[i] + 120, "right")
        sync[i] = (np.abs(lvs[a:b] - lv[i]) < 0.03).sum() - 1
    F["sync_amt_2m"] = sync
    for role, idx, other in (("r", in_i, S), ("s", out_i, R)):
        c = np.zeros(n)
        for a, ix in idx.items():
            o = ix[np.argsort(T[ix])]; t = T[o]
            for k, i in enumerate(o):
                lo, hi = np.searchsorted(t, T[i] - 600), np.searchsorted(t, T[i] + 600, "right")
                c[i] = len(set(other[o[lo:hi]]))
        F[f"{role}_sync_cp10m"] = c
    f = pd.DataFrame(F)
    # 6) episode-level aggregates broadcast to transactions
    base = ["in_match_best3600", "fwd_best3600", "chain_len", "chain_decay_per_hop", "s_ooc_amt_z", "coc_novel",
            "s_first_tx", "r_first_tx", "sync_amt_2m", "peel_run", "s_age_h"]
    for c in ("cmp900", "flow14400", "cmp3600"):
        g = f[base].groupby(comp[c].to_numpy())
        mean = g.transform("mean"); mx = g.transform("max")
        for b in base: f[f"ep_{c}_{b}_mean"] = mean[b].to_numpy()
        for b in ("chain_len", "sync_amt_2m", "peel_run", "s_ooc_amt_z"): f[f"ep_{c}_{b}_max"] = mx[b].to_numpy()
        lab = comp[c].to_numpy()
        d = pd.DataFrame({"c": lab, "S": S, "R": R, "V": V})
        o = d.groupby(["c", "S"]).V.sum(); i_ = d.groupby(["c", "R"]).V.sum()
        bal = pd.concat([o.rename("o"), i_.rename_axis(["c", "S"]).rename("i")], axis=1).fillna(0)
        bal["pt"] = ((bal.o > 0) & (bal.i > 0)).astype(float); bal["net"] = (bal.o - bal.i).abs() / (bal.o + bal.i + 1e-9)
        pb = bal[bal.pt == 1].groupby(level=0).net.mean(); npt = bal.groupby(level=0).pt.sum()
        f[f"ep_{c}_passthru_n"] = pd.Series(lab).map(npt).fillna(0).to_numpy()
        f[f"ep_{c}_passthru_bal"] = pd.Series(lab).map(pb).fillna(1).to_numpy()
    f.insert(0, "TRANSACTION_HASH", tx.TRANSACTION_HASH.values)
    return f


if __name__ == "__main__":
    import time; t0 = time.time()
    tx = load_split(sys.argv[1]); out = sys.argv[2]
    comp = pd.read_parquet(out.replace("_f4", "_comp")).set_index("TRANSACTION_HASH").loc[tx.TRANSACTION_HASH].reset_index()
    f = build(tx, comp); f.to_parquet(out); print(f.shape, time.time() - t0)
