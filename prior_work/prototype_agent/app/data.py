"""Load a split (transactions/transfers/market_rates [+labels]) into one edge per payment."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

XRP_TOKEN = "0x1111222211112222111122221111222211112222"
REQUIRED = ["transactions.parquet", "transfers.parquet", "market_rates.csv"]


@dataclass
class Split:
    name: str
    edges: pd.DataFrame            # one row per transaction, sorted by time
    transfers: pd.DataFrame
    labels: pd.DataFrame | None = None
    extra: dict = field(default_factory=dict)

    @property
    def has_labels(self) -> bool:
        return self.labels is not None


def asset_key(token_address: str, symbol: str) -> str:
    return "XRP" if symbol == "XRP" or token_address == XRP_TOKEN else f"{token_address}:{symbol}"


def _delivered(wire: str) -> tuple[str, float]:
    value = json.loads(wire)
    if isinstance(value, str):
        return "XRP", int(value) / 1_000_000
    return f"{value['issuer']}:{value['currency']}", float(value["value"])


def missing_files(directory: Path) -> list[str]:
    return [f for f in REQUIRED if not (directory / f).exists()]


def load_split(directory: Path, name: str) -> Split:
    directory = Path(directory)
    tx = pd.read_parquet(directory / "transactions.parquet",
                         columns=["TRANSACTION_HASH", "TIMESTAMP", "LEDGER_INDEX", "TRANSACTION_INDEX",
                                  "ACCOUNT", "SOURCE_FEE", "DELIVERED_AMOUNT", "EXTRA_TRANSACTION_FIELDS"])
    tr = pd.read_parquet(directory / "transfers.parquet",
                         columns=["TRANSACTION_HASH", "TIMESTAMP", "TRANSFER_TYPE", "TRANSFER_INDEX",
                                  "FROM_ADDRESS", "TO_ADDRESS", "TOKEN_ADDRESS", "SYMBOL", "VALUE"])
    rates = pd.read_csv(directory / "market_rates.csv")
    price = {(int(r.TIMESTAMP), r.ASSET): r.PRICE_MICRO_USD / 1e6 for r in rates.itertuples()}

    tr["TIMESTAMP"] = tr.TIMESTAMP.astype("int64")
    tr["TRANSFER_INDEX"] = tr.TRANSFER_INDEX.astype(int)
    tr["ASSET"] = [asset_key(a, s) for a, s in zip(tr.TOKEN_ADDRESS, tr.SYMBOL)]
    hours = tr.TIMESTAMP // 3600 * 3600
    tr["USD"] = [v * price[(h, a)] for v, h, a in zip(tr.VALUE, hours, tr.ASSET)]

    e = pd.DataFrame({
        "hash": tx.TRANSACTION_HASH.values,
        "t": tx.TIMESTAMP.astype("int64").values,
        "ledger": tx.LEDGER_INDEX.astype("int64").values,
        "tx_index": tx.TRANSACTION_INDEX.astype(int).values,
        "sender": tx.ACCOUNT.values,
        "fee": tx.SOURCE_FEE.astype(int).values,
    })
    e["receiver"] = [json.loads(x)["Destination"] for x in tx.EXTRA_TRANSACTION_FIELDS]
    delivered = [_delivered(x) for x in tx.DELIVERED_AMOUNT]
    e["dst_asset"] = [d[0] for d in delivered]
    e["dst_units"] = [d[1] for d in delivered]
    e["usd"] = [u * price[(t // 3600 * 3600, a)] for u, t, a in zip(e.dst_units, e.t, e.dst_asset)]

    legs = tr.groupby("TRANSACTION_HASH").agg(n_legs=("TRANSFER_INDEX", "size"))
    first = tr[tr.TRANSFER_INDEX == 0].set_index("TRANSACTION_HASH")
    e = e.join(legs, on="hash")
    e["src_asset"] = e.hash.map(first.ASSET)
    e["src_usd"] = e.hash.map(first.USD)
    e = e.sort_values(["t", "tx_index"]).reset_index(drop=True)

    labels = None
    if (directory / "labels.csv").exists():
        labels = pd.read_csv(directory / "labels.csv")
    return Split(name=name, edges=e, transfers=tr, labels=labels)


def short(address: str) -> str:
    return address[:7]


def symbol(asset: str) -> str:
    return asset.split(":")[-1]


def issuer_tag(asset: str) -> str:
    return "native" if asset == "XRP" else f"issuer {asset[:5]}"


def describe_asset(asset: str) -> str:
    return "XRP" if asset == "XRP" else f"{symbol(asset)} ({issuer_tag(asset)})"


def labels_array(split: Split) -> np.ndarray:
    lab = split.labels.set_index("TRANSACTION_HASH")
    return lab.loc[split.edges.hash, "is_suspicious"].to_numpy()


def cv_groups(split: Split) -> np.ndarray:
    """Group planted scenarios/cohorts together so CV doesn't leak a scenario across folds."""
    lab = split.labels.set_index("TRANSACTION_HASH").loc[split.edges.hash]
    group = lab.cohort_id.fillna(lab.scenario_id).fillna(pd.Series(lab.index, index=lab.index))
    return group.to_numpy()
