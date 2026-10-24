"""Session / temporal-component features: reconstruct candidate scenarios (bursts of linked payments)
and describe them as wholes (size, accounts, duration, regularity, amount similarity, novelty). Label-free."""
import sys, warnings
import numpy as np, pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
warnings.filterwarnings("ignore")
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from fe import load_split


def _cv(x): return x.std() / (x.mean() + 1e-9)


def session_feats(df, key, g, name):
    d = df.sort_values(key + ["TS"])
    k = d[key].astype(str).agg("|".join, axis=1) if len(key) > 1 else d[key[0]]
    newk = k.ne(k.shift()); gap = d.TS.diff()
    sid = (newk | (gap > g)).cumsum()
    d = d.assign(sid=sid.values, gap=np.where(newk, np.nan, gap))
    gb = d.groupby("sid")
    out = pd.DataFrame(index=d.index)
    out[f"{name}_n"] = gb.TS.transform("size")
    out[f"{name}_dur"] = np.log1p(gb.TS.transform("max") - gb.TS.transform("min"))
    out[f"{name}_amt_cv"] = gb.V.transform(_cv)
    out[f"{name}_logsum"] = np.log1p(gb.V.transform("sum"))
    out[f"{name}_gap_cv"] = gb.gap.transform(_cv)
    out[f"{name}_gap_med"] = np.log1p(gb.gap.transform("median"))
    out[f"{name}_uniq_s"] = gb.S.transform("nunique")
    out[f"{name}_uniq_r"] = gb.R.transform("nunique")
    out[f"{name}_novel"] = gb.novel.transform("mean")
    out[f"{name}_pos"] = gb.cumcount() / out[f"{name}_n"].clip(lower=1)
    return out.reindex(df.index)


def component_feats(df, g, name, flow_only=False):
    """Temporal components: link consecutive txs touching the same account within g seconds."""
    n = len(df); rows, cols = [], []
    inc = pd.concat([pd.DataFrame({"a": df.S, "T": df.TS, "i": df.index, "role": 0}),
                     pd.DataFrame({"a": df.R, "T": df.TS, "i": df.index, "role": 1})])
    if flow_only:  # link tx into an account with the next tx out of that account
        inc = inc.sort_values(["a", "T"])
        for a, grp in inc.groupby("a"):
            t = grp["T"].to_numpy(); ii = grp.i.to_numpy(); ro = grp.role.to_numpy()
            ins = np.where(ro == 1)[0]; outs = np.where(ro == 0)[0]
            if len(ins) == 0 or len(outs) == 0: continue
            j = np.searchsorted(t[outs], t[ins], "right")
            ok = j < len(outs)
            src = ins[ok]; dst = outs[j[ok]]
            m = (t[dst] - t[src]) <= g
            rows += list(ii[src[m]]); cols += list(ii[dst[m]])
    else:
        inc = inc.sort_values(["a", "T"])
        same = inc.a.to_numpy()[1:] == inc.a.to_numpy()[:-1]
        close = np.diff(inc["T"].to_numpy()) <= g
        m = same & close
        ii = inc.i.to_numpy()
        rows, cols = list(ii[:-1][m]), list(ii[1:][m])
    A = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    _, lab = connected_components(A, directed=False)
    d = df.assign(c=lab); gb = d.groupby("c")
    out = pd.DataFrame(index=df.index)
    out[f"{name}_n"] = gb.TS.transform("size")
    out[f"{name}_dur"] = np.log1p(gb.TS.transform("max") - gb.TS.transform("min"))
    acc = pd.concat([d[["c", "S"]].rename(columns={"S": "a"}), d[["c", "R"]].rename(columns={"R": "a"})]).groupby("c").a.nunique()
    out[f"{name}_acc"] = d.c.map(acc)
    out[f"{name}_dens"] = out[f"{name}_n"] / out[f"{name}_acc"]
    out[f"{name}_amt_cv"] = gb.V.transform(_cv)
    out[f"{name}_logsum"] = np.log1p(gb.V.transform("sum"))
    out[f"{name}_novel"] = gb.novel.transform("mean")
    out[f"{name}_assets"] = gb.A.transform("nunique")
    out[f"{name}_cross"] = gb.X.transform("mean")
    return out, lab


def build_sessions(tx):
    df = pd.DataFrame({"S": tx.ACCOUNT.values, "R": tx.DEST.values, "TS": tx.TIMESTAMP.values.astype(np.int64),
                       "V": tx.usd.fillna(0).values, "A": tx.dst_asset.values, "SA": tx.src_asset.values,
                       "X": tx.cross_ccy.values})
    pc = df.groupby(["S", "R"]).TS.transform("size"); df["novel"] = (pc <= 2).astype(float)
    parts = [pd.DataFrame({"TRANSACTION_HASH": tx.TRANSACTION_HASH.values})]
    for g in (300, 1800, 7200):
        parts.append(session_feats(df, ["R", "A"], g, f"rin{g}"))
        parts.append(session_feats(df, ["S", "SA"], g, f"sout{g}"))
    comps = {}
    for g in (120, 900, 3600):
        o, lab = component_feats(df, g, f"cmp{g}"); parts.append(o); comps[f"cmp{g}"] = lab
        o, lab = component_feats(df, g * 4, f"flow{g*4}", flow_only=True); parts.append(o); comps[f"flow{g*4}"] = lab
    f = pd.concat(parts, axis=1)
    c = pd.DataFrame(comps); c.insert(0, "TRANSACTION_HASH", tx.TRANSACTION_HASH.values)
    return f, c


if __name__ == "__main__":
    import time; t0 = time.time()
    tx = load_split(sys.argv[1]); f, c = build_sessions(tx)
    f.to_parquet(sys.argv[2]); c.to_parquet(sys.argv[2].replace("_f3", "_comp")); print(f.shape, time.time() - t0)
