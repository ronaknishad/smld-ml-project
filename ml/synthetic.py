"""SYNTHETIC directional flow generator for software testing ONLY.

Results obtained on this data are NOT benchmarks and say nothing about
detection of real attacks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SYNTHETIC_NOTICE = "SYNTHETIC DATA - for pipeline testing only; not a benchmark."


def make_synthetic_flows(n: int = 4000, malicious_frac: float = 0.15, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < malicious_frac).astype(int)
    mal = y == 1
    packet_count = np.where(mal, rng.lognormal(3.6, 0.9, n), rng.lognormal(2.8, 1.0, n)).round().clip(1)
    mean_len = np.where(mal, rng.normal(120, 60, n), rng.normal(380, 160, n)).clip(40, 1500)
    duration = np.where(mal, rng.lognormal(0.2, 1.1, n), rng.lognormal(0.9, 1.3, n)).clip(0.01)
    mean_iat = duration / np.maximum(packet_count - 1, 1) * rng.lognormal(0, 0.2, n)
    std_len = mean_len * rng.uniform(0.05, 0.6, n)
    proto = np.where(
        mal, rng.choice(["TCP", "UDP", "ICMP"], n, p=[0.55, 0.25, 0.20]),
        rng.choice(["TCP", "UDP", "ICMP"], n, p=[0.70, 0.27, 0.03]),
    )
    df = pd.DataFrame({
        "flow_id": [f"syn-{i:06d}" for i in range(n)],
        "src_ip": [f"10.0.{rng.integers(0, 4)}.{rng.integers(1, 250)}" for _ in range(n)],
        "src_port": rng.integers(1024, 65535, n),
        "dst_ip": [f"192.168.{rng.integers(0, 4)}.{rng.integers(1, 250)}" for _ in range(n)],
        "dst_port": rng.choice([22, 53, 80, 443, 8080], n),
        "protocol": proto,
        "packet_count": packet_count,
        "byte_count": packet_count * mean_len,
        "duration": duration,
        "mean_pkt_len": mean_len,
        "std_pkt_len": std_len,
        "min_pkt_len": np.maximum(40, mean_len - 2 * std_len),
        "max_pkt_len": np.minimum(1500, mean_len + 2 * std_len),
        "mean_iat": mean_iat,
        "pkts_per_sec": packet_count / duration,
        "bytes_per_sec": packet_count * mean_len / duration,
        "label": np.where(mal, "malicious", "benign"),
    })
    return df
