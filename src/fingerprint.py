"""Synthetic-generator fingerprint hunt (finding + limitation: these artifacts will not transfer to the real XRPL).
Builds per-transaction 'boring column' features for a split. Usage: python src/fingerprint.py  -> cache/fp/{train,test}_fp.parquet"""
import json, numpy as np, pandas as pd, hashlib

def ndec(s):
    s = str(s)
    if "." not in s: return 0
    return len(s.split(".")[1].rstrip("0"))

def trailing_zeros(v):
    v = int(round(abs(v)));
    if v == 0: return 0
    n = 0
    while v % 10 == 0: v //= 10; n += 1
    return n

def sigdig(s):
    s = str(s).replace(".", "").replace("-", "").lstrip("0").rstrip("0")
    return len(s)

def amt_parts(a):
    if isinstance(a, str): return "XRP", a, float(a) / 1e6
    return a["issuer"] + ":" + a["currency"], a["value"], float(a["value"])

def build(split):
    D = f"data/{split}"
    tx = pd.read_parquet(f"{D}/transactions.parquet")
    tf = pd.read_parquet(f"{D}/transfers.parquet")
    bl = pd.read_parquet(f"{D}/blocks.parquet")
    mr = pd.read_csv(f"{D}/market_rates.csv")
    f = pd.DataFrame({"TRANSACTION_HASH": tx.TRANSACTION_HASH})
    li = tx.LEDGER_INDEX.astype("int64"); ts = tx.TIMESTAMP.astype("int64")
    seq = tx.SEQUENCE.astype("int64"); lls = tx.LAST_LEDGER_SEQUENCE.astype("int64")
    f["fee_drops"] = tx.SOURCE_FEE.astype("int64")
    f["seq"] = seq
    f["tx_index"] = tx.TRANSACTION_INDEX.astype("int64")
    f["lls_offset"] = lls - li
    f["ledger_rel"] = li - li.min()
    f["ts_sec_of_min"] = ts % 60; f["ts_min_of_hour"] = (ts // 60) % 60; f["ts_mod4"] = ts % 4
    f["ts_hour"] = (ts // 3600) % 24
    f["ledger_mod10"] = li % 10
    # per ledger
    nled = tx.groupby("LEDGER_INDEX").TRANSACTION_HASH.transform("size")
    f["txs_in_ledger"] = nled
    f["tx_index_frac"] = f.tx_index / nled.clip(lower=1)
    # blocks
    b = bl.assign(L=bl.LEDGER_INDEX.astype("int64"), CT=bl.CLOSE_TIME.astype("int64"), PCT=bl.PARENT_CLOSE_TIME.astype("int64"),
                  TC=bl.TOTAL_COINS.astype("float64")).sort_values("L")
    b["close_gap"] = b.CT - b.PCT
    b["coins_burn"] = -b.TC.diff()
    b["acct_hash_c0"] = b.ACCOUNT_HASH.str[0].apply(lambda c: int(c, 16))
    b["ledger_hash_c0"] = b.LEDGER_HASH.str[0].apply(lambda c: int(c, 16))
    has_tx = b.L.isin(set(li))
    b["empty_before"] = b.L - b.L.where(has_tx).shift(1).ffill()
    bm = b.set_index("L")
    for c in ["close_gap", "coins_burn", "acct_hash_c0", "ledger_hash_c0", "empty_before"]:
        f["blk_" + c] = bm[c].reindex(li).to_numpy()
    f["blk_burn_per_tx"] = f.blk_coins_burn / nled
    # per account sequence structure
    o = pd.DataFrame({"a": tx.ACCOUNT, "seq": seq, "ts": ts, "li": li, "ti": f.tx_index}).sort_values(["a", "li", "ti"])
    g = o.groupby("a")
    o["seq_gap_prev"] = g.seq.diff(); o["seq_gap_next"] = -g.seq.diff(-1)
    o["dt_prev"] = g.ts.diff(); o["dt_next"] = -g.ts.diff(-1)
    o["seq_rank"] = g.cumcount(); o["acct_n"] = g.seq.transform("size")
    o["seq_min"] = g.seq.transform("min"); o["seq_span"] = g.seq.transform("max") - o.seq_min
    o["seq_density"] = o.acct_n / (o.seq_span + 1)
    o["seq_minus_min"] = o.seq - o.seq_min
    o["seq_order_viol"] = (o.seq_gap_prev < 0).astype(float)
    for c in ["seq_gap_prev", "seq_gap_next", "dt_prev", "dt_next", "seq_rank", "acct_n", "seq_min", "seq_span", "seq_density", "seq_minus_min", "seq_order_viol"]:
        f["acc_" + c] = o[c].reindex(f.index).to_numpy()
    first_seen = pd.concat([tf[["FROM_ADDRESS", "DATETIME"]].rename(columns={"FROM_ADDRESS": "a"}),
                            tf[["TO_ADDRESS", "DATETIME"]].rename(columns={"TO_ADDRESS": "a"})]).groupby("a").DATETIME.min().rank(pct=True)
    f["acc_first_seen_rank"] = tx.ACCOUNT.map(first_seen).to_numpy()
    # account / hash characters (sim identifiers)
    f["acct_c1"] = tx.ACCOUNT.str[1].map(ord); f["acct_len"] = tx.ACCOUNT.str.len()
    f["hash_c0"] = tx.TRANSACTION_HASH.str[0].apply(lambda c: int(c, 16))
    # JSON
    ex = tx.EXTRA_TRANSACTION_FIELDS.map(json.loads)
    f["has_sendmax"] = ex.map(lambda e: "SendMax" in e).astype(int)
    f["has_paths"] = ex.map(lambda e: "Paths" in e).astype(int)
    f["n_paths"] = ex.map(lambda e: len(e.get("Paths", [])))
    f["path_len"] = ex.map(lambda e: sum(len(p) for p in e.get("Paths", [])))
    f["amount_eq_delivermax"] = ex.map(lambda e: e["Amount"] == e["DeliverMax"]).astype(int)
    f["dest_c1"] = ex.map(lambda e: ord(e["Destination"][1]))
    A = ex.map(lambda e: amt_parts(e["Amount"]))
    f["amt_is_xrp"] = A.map(lambda p: p[0] == "XRP").astype(int)
    f["amt_ndec"] = A.map(lambda p: ndec(p[1]) if p[0] != "XRP" else ndec(float(p[1]) / 1e6))
    f["amt_sigdig"] = A.map(lambda p: sigdig(p[1]))
    f["amt_tz_native"] = A.map(lambda p: trailing_zeros(float(p[1]) if p[0] == "XRP" else float(p[1]) * 1e6))
    f["amt_last_digit"] = A.map(lambda p: int(str(p[1]).replace(".", "")[-1]))
    f["amt_last2"] = A.map(lambda p: int(str(p[1]).replace(".", "")[-2:]))
    f["amt_int_round"] = A.map(lambda p: float(p[2] == round(p[2])))
    S = ex.map(lambda e: amt_parts(e["SendMax"]) if "SendMax" in e else None)
    f["sm_ndec"] = S.map(lambda p: np.nan if p is None else (ndec(p[1]) if p[0] != "XRP" else ndec(float(p[1]) / 1e6)))
    f["sm_sigdig"] = S.map(lambda p: np.nan if p is None else sigdig(p[1]))
    f["sm_same_asset"] = [np.nan if s is None else float(s[0] == a[0]) for s, a in zip(S, A)]
    da = tx.DELIVERED_AMOUNT.map(json.loads).map(amt_parts)
    f["deliv_eq_amount"] = [float(d[1] == a[1]) for d, a in zip(da, A)]
    # transfers
    tf = tf.assign(TI=tf.TRANSFER_INDEX.astype(int)).sort_values(["TRANSACTION_HASH", "TI"])
    gt = tf.groupby("TRANSACTION_HASH")
    f["n_legs"] = f.TRANSACTION_HASH.map(gt.size())
    first = gt.first(); last = gt.last()
    asset = lambda r: "XRP" if r.SYMBOL == "XRP" else r.TOKEN_ADDRESS + ":" + r.SYMBOL
    # USD values
    f["leg1_value"] = f.TRANSACTION_HASH.map(first.VALUE); f["legN_value"] = f.TRANSACTION_HASH.map(last.VALUE)
    f["leg1_ndec"] = f.leg1_value.map(lambda v: ndec(f"{v:.6f}"))
    f["leg1_sigdig"] = f.leg1_value.map(lambda v: sigdig(f"{v:.6f}"))
    f["leg1_src_eq_val"] = f.TRANSACTION_HASH.map((first.SOURCE_VALUE == first.VALUE).astype(float))
    sm_val = S.map(lambda p: np.nan if p is None else p[2])
    f["sendmax_over_leg1"] = sm_val.to_numpy() / f.leg1_value.replace(0, np.nan)
    f["sendmax_over_leg1_m1e4"] = ((f.sendmax_over_leg1 - 1) * 1e4).round(3)
    P = {(int(t), a): v / 1e6 for t, a, v in zip(mr.TIMESTAMP, mr.ASSET, mr.PRICE_MICRO_USD)}
    hr = (ts // 3600 * 3600).to_numpy()
    f["amt_usd"] = [p[2] * P.get((h, p[0]), np.nan) for p, h in zip(A, hr)]
    u = f.amt_usd
    for k in (1, 10, 100, 1000):
        f[f"usd_dist_round{k}"] = (u / k - (u / k).round()).abs()
    f["usd_log10_frac"] = np.log10(u.clip(lower=1e-9)) % 1
    f["native_log10_frac"] = np.log10(A.map(lambda p: p[2]).clip(lower=1e-9)) % 1
    f["usd_cents_zero"] = ((u * 100).round() % 100 == 0).astype(float)
    return f, mr

def usd(f, tx_split, mr):
    return f

if __name__ == "__main__":
    import sys
    for s in sys.argv[1:] or ["train", "test"]:
        f, _ = build(s)
        f.to_parquet(f"cache/fp/{s}_fp.parquet"); print(s, f.shape, flush=True)
