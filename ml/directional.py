"""Directional (unidirectional) flow extraction from packet-level records.

A flow is identified by the DIRECTIONAL key
    (src_ip, src_port, dst_ip, dst_port, protocol)
so A->B and B->A are always separate flows. Reverse traffic is never merged.

Packets in the same direction are split into separate flows when the gap
between consecutive packets exceeds `idle_timeout` seconds.

Std of packet length uses ddof=0 (population std) so single-packet flows are
defined (0.0). Inter-arrival-time (IAT) statistics are NaN for single-packet
flows and are never computed across a timeout boundary.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

KEY = ["src_ip", "src_port", "dst_ip", "dst_port", "protocol"]
REQUIRED = KEY + ["timestamp", "length"]


def flow_key(row) -> tuple:
    """Directional key of a packet-like mapping. Reversing src/dst gives a different key."""
    return tuple(row[k] for k in KEY)


def aggregate_packets(packets: pd.DataFrame, idle_timeout: float = 60.0) -> pd.DataFrame:
    missing = [c for c in REQUIRED if c not in packets.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if idle_timeout <= 0:
        raise ValueError("idle_timeout must be positive")

    df = packets[REQUIRED].copy()
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["length"] = pd.to_numeric(df["length"], errors="coerce")
    if df[["timestamp", "length"]].isna().any().any():
        raise ValueError("timestamp and length must be numeric with no missing values")
    if (df["length"] < 0).any():
        raise ValueError("length must be non-negative")

    df = df.sort_values(KEY + ["timestamp"], kind="mergesort").reset_index(drop=True)
    by_key = df.groupby(KEY, sort=False, dropna=False)
    gap = by_key["timestamp"].diff()
    new_segment = gap.gt(idle_timeout)
    df["_seg"] = new_segment.groupby([df[k] for k in KEY], sort=False, dropna=False).cumsum()
    df["_iat"] = gap.where(~new_segment)  # no IAT across a timeout boundary

    out = (
        df.groupby(KEY + ["_seg"], sort=True, dropna=False)
        .agg(
            packet_count=("length", "size"),
            byte_count=("length", "sum"),
            t_start=("timestamp", "min"),
            t_end=("timestamp", "max"),
            mean_pkt_len=("length", "mean"),
            std_pkt_len=("length", lambda x: float(np.std(x, ddof=0))),
            min_pkt_len=("length", "min"),
            max_pkt_len=("length", "max"),
            mean_iat=("_iat", "mean"),
        )
        .reset_index()
        .drop(columns="_seg")
    )
    out["duration"] = out["t_end"] - out["t_start"]
    dur = out["duration"].where(out["duration"] > 0)
    out["pkts_per_sec"] = out["packet_count"] / dur
    out["bytes_per_sec"] = out["byte_count"] / dur
    out.insert(0, "flow_id", [f"flow-{i:06d}" for i in range(len(out))])
    return out
