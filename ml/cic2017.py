"""CIC-IDS2017 (MachineLearningCSV) loader - FORWARD-DIRECTION COMPATIBILITY MODE.

What this is / is not
- The released CSVs are BIDIRECTIONAL flow records with no IP columns and no protocol column.
  This is therefore NOT a strict unidirectional dataset.
- We use only fields that describe the forward direction. Nothing is invented, halved or
  reconstructed for the reverse direction. Backward and whole-flow fields are never used.
- Labels were assigned by the dataset authors to the whole bidirectional flow; this limitation
  stays even though only forward features are used.
- Flow Duration and 'Fwd Packets/s' use whole-flow (both directions) duration, so they are
  NOT used. Forward duration is taken from 'Fwd IAT Total' (approx. first-to-last forward
  packet span) and rates are recomputed from it.
- CICFlowMeter time fields are in microseconds; converted here to seconds.
- Destination Port is excluded from features by default (strong dataset shortcut risk).

Cleaning (every step counted in the returned report, nothing silent):
  non-numeric -> NaN, +/-inf -> NaN, Init_Win_bytes_forward == -1 (sentinel for 'not observed') -> NaN,
  rows with any other negative feature value dropped, exact duplicate rows (features+label) dropped
  by default (duplicates straddling a random split inflate test scores).
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

# raw (stripped) CIC column -> our name. Units noted where converted.
FORWARD_MAP = {
    "Total Fwd Packets": "packet_count",
    "Total Length of Fwd Packets": "byte_count",
    "Fwd IAT Total": "_fwd_iat_total_us",
    "Fwd Packet Length Mean": "mean_pkt_len",
    "Fwd Packet Length Std": "std_pkt_len",
    "Fwd Packet Length Min": "min_pkt_len",
    "Fwd Packet Length Max": "max_pkt_len",
    "Fwd IAT Mean": "_mean_iat_us",
    "Fwd IAT Std": "_std_iat_us",
    "Fwd IAT Max": "_max_iat_us",
    "Fwd Header Length": "fwd_header_len",
    "Fwd PSH Flags": "fwd_psh_flags",
    "Fwd URG Flags": "fwd_urg_flags",
    "Init_Win_bytes_forward": "init_win_bytes_fwd",
    "act_data_pkt_fwd": "act_data_pkt_fwd",
    "min_seg_size_forward": "min_seg_size_fwd",
}
REQUIRED_RAW = list(FORWARD_MAP) + ["Label"]

CIC_NUMERIC = ["packet_count", "byte_count", "duration", "mean_pkt_len", "std_pkt_len", "min_pkt_len",
               "max_pkt_len", "mean_iat", "std_iat", "max_iat", "pkts_per_sec", "bytes_per_sec",
               "fwd_header_len", "fwd_psh_flags", "fwd_urg_flags", "init_win_bytes_fwd",
               "act_data_pkt_fwd", "min_seg_size_fwd"]
CIC_FEATURES = {"numeric": CIC_NUMERIC, "categorical": []}

KNOWN_ATTACKS = {
    "dos hulk", "portscan", "ddos", "dos goldeneye", "ftp-patator", "ssh-patator", "dos slowloris",
    "dos slowhttptest", "bot", "infiltration", "heartbleed",
    "web attack - brute force", "web attack - xss", "web attack - sql injection",
}


def normalise_label(raw: str) -> str:
    """Strip whitespace and replace non-ASCII dash bytes (cp1252 0x96 etc.) with '-'."""
    t = re.sub(r"[^\x20-\x7E]+", "-", str(raw)).strip()
    t = re.sub(r"\s*-\s*", " - ", t) if t.lower().startswith("web attack") else t
    return t


def load_cic_csv(path: str, nrows: int | None = None) -> pd.DataFrame:
    """Read only the needed columns. latin-1 because Thursday labels contain a cp1252 dash byte."""
    header = pd.read_csv(path, nrows=0, encoding="latin-1").columns
    keep = [c for c in header if c.strip() in REQUIRED_RAW]
    found = {c.strip() for c in keep}
    missing = [c for c in REQUIRED_RAW if c not in found]
    if missing:
        raise ValueError(f"Not a supported CIC-IDS2017 MachineLearningCSV file; missing columns: {missing}")
    raw = pd.read_csv(path, usecols=keep, nrows=nrows, encoding="latin-1", low_memory=False)
    raw.columns = raw.columns.str.strip()
    return raw


def to_forward_features(raw: pd.DataFrame, source_file: str = "", drop_duplicates: bool = True):
    """Return (features_df, cleaning_report). Raises on missing columns or unknown labels."""
    missing = [c for c in REQUIRED_RAW if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    report = {"source_file": source_file, "n_raw_rows": int(len(raw))}

    labels = raw["Label"].map(normalise_label)
    unknown = sorted({l for l in labels.unique() if l.lower() != "benign" and l.lower() not in KNOWN_ATTACKS})
    if unknown:
        raise ValueError(f"Unsupported labels (not silently mapped to benign): {unknown}")

    num = raw[list(FORWARD_MAP)].apply(pd.to_numeric, errors="coerce")
    report["n_non_numeric_to_nan"] = int(num.isna().sum().sum() - raw[list(FORWARD_MAP)].isna().sum().sum())
    arr = num.to_numpy(dtype=float)
    report["n_inf_to_nan"] = int(np.isinf(arr).sum())
    num = num.replace([np.inf, -np.inf], np.nan).rename(columns=FORWARD_MAP)

    sentinel = num["init_win_bytes_fwd"] == -1
    report["n_init_win_sentinel_to_nan"] = int(sentinel.sum())
    num.loc[sentinel, "init_win_bytes_fwd"] = np.nan

    neg = (num < 0).any(axis=1)
    report["n_dropped_negative_values"] = int(neg.sum())

    out = pd.DataFrame(index=num.index)
    for c in ["packet_count", "byte_count", "mean_pkt_len", "std_pkt_len", "min_pkt_len", "max_pkt_len",
              "fwd_header_len", "fwd_psh_flags", "fwd_urg_flags", "init_win_bytes_fwd",
              "act_data_pkt_fwd", "min_seg_size_fwd"]:
        out[c] = num[c]
    out["duration"] = num["_fwd_iat_total_us"] / 1e6
    out["mean_iat"] = num["_mean_iat_us"] / 1e6
    out["std_iat"] = num["_std_iat_us"] / 1e6
    out["max_iat"] = num["_max_iat_us"] / 1e6
    dur = out["duration"].where(out["duration"] > 0)  # undefined (NaN), not invented, when 0
    out["pkts_per_sec"] = out["packet_count"] / dur
    out["bytes_per_sec"] = out["byte_count"] / dur
    out = out[CIC_NUMERIC]
    out["label"] = np.where(labels.str.lower() == "benign", "benign", "malicious")
    out["attack_category"] = np.where(labels.str.lower() == "benign", "BENIGN", labels)
    out.insert(0, "row_id", raw.index.to_numpy())
    out.insert(1, "source_file", source_file)

    out = out.loc[~neg]
    if drop_duplicates:
        before = len(out)
        out = out.drop_duplicates(subset=CIC_NUMERIC + ["label"])
        report["n_duplicates_dropped"] = int(before - len(out))
    else:
        report["n_duplicates_dropped"] = 0
    report["n_rows_kept"] = int(len(out))
    return out.reset_index(drop=True), report


def load_forward_dataset(paths, nrows=None, drop_duplicates=True):
    """Load one or more CIC day files. Returns (combined_df, [per-file reports])."""
    frames, reports = [], []
    for p in paths:
        name = str(p).replace("\\", "/").split("/")[-1]
        df, rep = to_forward_features(load_cic_csv(p, nrows), name, drop_duplicates)
        frames.append(df)
        reports.append(rep)
    return pd.concat(frames, ignore_index=True), reports
