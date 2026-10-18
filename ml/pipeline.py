"""Training, threshold selection, evaluation and persistence.

Design rules
- Features are an explicit allow-list; IPs, ports, flow ids and labels are never features.
- All learned preprocessing lives inside the sklearn Pipeline and is fit on training data only.
- Split: stratified train / validation / test. Decision threshold is chosen on VALIDATION
  data (max recall subject to FPR <= max_fpr). The TEST set is used once for final metrics.
- Final model is selected on validation recall (not test).
- Bootstrap CIs assume independent records (a stratified random split is only valid
  when records are not strongly related, e.g. same session/host).
"""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from ml.stats import bootstrap_ci

NUMERIC = ["packet_count", "byte_count", "duration", "mean_pkt_len", "std_pkt_len",
           "min_pkt_len", "max_pkt_len", "mean_iat", "pkts_per_sec", "bytes_per_sec"]
CATEGORICAL = ["protocol"]
FEATURES = NUMERIC + CATEGORICAL
LABEL_COL = "label"
LABEL_MAP = {"benign": 0, "malicious": 1}
MODELS = ("dummy", "logreg", "rf")


def _spec(numeric, categorical):
    return (list(NUMERIC) if numeric is None else list(numeric),
            list(CATEGORICAL) if categorical is None else list(categorical))


def prepare(df: pd.DataFrame, need_label: bool = True, numeric=None, categorical=None):
    """Validate and clean a COPY of the input. Returns (X, y_or_None, report)."""
    numeric, categorical = _spec(numeric, categorical)
    feats = numeric + categorical
    missing = [c for c in feats if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    X = df[feats].copy()
    report = {"n_rows": int(len(df)), "n_duplicates": int(df.duplicated().sum())}
    for c in numeric:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    n_inf = int(np.isinf(X[numeric].to_numpy(dtype=float)).sum())
    X[numeric] = X[numeric].replace([np.inf, -np.inf], np.nan)
    if (X[numeric] < 0).any().any():
        raise ValueError("Negative values found in non-negative traffic features")
    report["n_inf_replaced_with_nan"] = n_inf
    report["missing_per_feature"] = {c: int(v) for c, v in X.isna().sum().items() if v}
    y = None
    if need_label:
        if LABEL_COL not in df.columns:
            raise ValueError(f"Missing label column '{LABEL_COL}'")
        unknown = sorted(set(df[LABEL_COL].astype(str)) - set(LABEL_MAP))
        if unknown:
            raise ValueError(f"Unsupported labels (no silent default to benign): {unknown}")
        y = df[LABEL_COL].astype(str).map(LABEL_MAP).to_numpy()
    return X, y, report


def build_pipeline(model: str, seed: int = 42, numeric=None, categorical=None) -> Pipeline:
    if model not in MODELS:
        raise ValueError(f"Unknown model '{model}'")
    numeric, categorical = _spec(numeric, categorical)
    # add_indicator keeps "was missing" as a signal (e.g. rate undefined for zero-duration flows)
    num_steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if model == "logreg":  # skewed non-negative features: log1p then standardise
        num_steps += [("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                      ("scale", StandardScaler())]
    parts = [("num", Pipeline(num_steps), numeric)]
    if categorical:
        parts.append(("cat", OneHotEncoder(handle_unknown="ignore"), categorical))
    pre = ColumnTransformer(parts)
    clf = {
        "dummy": DummyClassifier(strategy="prior"),
        "logreg": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed),
        "rf": RandomForestClassifier(n_estimators=200, min_samples_leaf=2, class_weight="balanced",
                                     random_state=seed, n_jobs=-1),
    }[model]
    return Pipeline([("pre", pre), ("clf", clf)])


def select_threshold(y_val, proba_val, max_fpr: float = 0.05) -> float:
    """Highest-recall threshold with validation FPR <= max_fpr (ties -> lowest FPR); 0.5 if undefined."""
    if len(set(y_val)) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(y_val, proba_val)
    ok = np.where(fpr <= max_fpr)[0]
    # Among points with the best achievable recall, take the HIGHEST threshold (lowest FPR);
    # do not spend the whole FPR budget when recall has already stopped improving.
    best = tpr[ok].max()
    t = float(thr[ok[np.where(tpr[ok] == best)[0][0]]])
    return min(t, 1.0 + 1e-9) if np.isfinite(t) else 1.0 + 1e-9


def compute_metrics(y_true, proba, threshold: float) -> dict:
    y_true = np.asarray(y_true)
    pred = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    nan = float("nan")
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else nan
    both = len(set(y_true)) == 2
    return {
        "threshold": float(threshold), "n": int(len(y_true)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "accuracy": float((tp + tn) / len(y_true)),
        "precision": float(precision), "recall": float(recall),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "fpr": float(fp / (fp + tn)) if fp + tn else nan,
        "fnr": float(fn / (fn + tp)) if fn + tp else nan,
        "roc_auc": float(roc_auc_score(y_true, proba)) if both else nan,
        "pr_auc": float(average_precision_score(y_true, proba)) if both else nan,
    }


def _recall(y, p):
    return float(((y == 1) & (p == 1)).sum() / max((y == 1).sum(), 1)) if (y == 1).any() else float("nan")


def _fpr(y, p):
    return float(((y == 0) & (p == 1)).sum() / max((y == 0).sum(), 1)) if (y == 0).any() else float("nan")


def dataset_hash(df: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes()).hexdigest()[:16]


def _evaluate_models(X_fit, y_fit, X_val, y_val, X_te, y_te, *, hash_df, dataset_id, models, seed,
                     max_fpr, n_boot, numeric, categorical, split_info, data_report):
    feats = numeric + categorical
    results = []
    for name in models:
        pipe = build_pipeline(name, seed, numeric, categorical)
        t0 = time.perf_counter()
        pipe.fit(X_fit, y_fit)
        train_s = time.perf_counter() - t0
        # Fit is deterministic even when parallel (per-tree seeds), but parallel predict_proba sums
        # tree votes in thread-completion order, so float results differ in the last bits.
        # Single-thread inference makes same-seed runs reproduce exactly.
        if "n_jobs" in pipe.named_steps["clf"].get_params():
            pipe.named_steps["clf"].set_params(n_jobs=1)
        p_val = pipe.predict_proba(X_val)[:, 1]
        thr = select_threshold(y_val, p_val, max_fpr)
        val_m = compute_metrics(y_val, p_val, thr)
        t1 = time.perf_counter()
        p_te = pipe.predict_proba(X_te)[:, 1]
        infer_s = time.perf_counter() - t1
        test_m = compute_metrics(y_te, p_te, thr)
        pred_te = (p_te >= thr).astype(int)
        ci = {"recall": bootstrap_ci(y_te, pred_te, _recall, n_boot, seed=seed),
              "fpr": bootstrap_ci(y_te, pred_te, _fpr, n_boot, seed=seed)}
        metadata = {
            "model_id": f"{name}-{dataset_id}-s{seed}", "algorithm": name, "dataset_id": dataset_id,
            "dataset_hash": dataset_hash(hash_df), "seed": seed, "features": feats,
            "numeric": numeric, "categorical": categorical,
            "label_map": LABEL_MAP, "threshold": thr, "threshold_criterion": f"max recall with val FPR<={max_fpr}",
            "split": split_info, "trained_at": datetime.now(timezone.utc).isoformat(),
            "sklearn_version": sklearn.__version__,
        }
        results.append({"model": name, "pipeline": pipe, "metadata": metadata, "val": val_m,
                        "test": test_m, "test_ci95": ci, "train_seconds": train_s,
                        "inference_seconds": infer_s})

    candidates = [r for r in results if r["model"] != "dummy"]
    selected = max(candidates, key=lambda r: (np.nan_to_num(r["val"]["recall"]), r["val"]["f1"])) if candidates else None
    importance = None
    if selected is not None:
        try:
            pi = permutation_importance(selected["pipeline"], X_te, y_te, scoring="average_precision",
                                        n_repeats=5, random_state=seed)
            importance = sorted(
                [{"feature": f, "mean_drop_pr_auc": float(m), "std": float(s_)}
                 for f, m, s_ in zip(feats, pi.importances_mean, pi.importances_std)],
                key=lambda d: -d["mean_drop_pr_auc"])
        except ValueError:  # e.g. test set has a single class
            importance = None
    return {"data_report": data_report, "split": split_info, "results": results,
            "selected_model": selected["metadata"]["model_id"] if selected else None,
            "selection_rule": "highest validation recall (at validation FPR cap), tie-break F1; dummy excluded",
            "permutation_importance_test_set": importance}


def run_experiment(df: pd.DataFrame, dataset_id: str, models=MODELS, seed: int = 42,
                   test_size: float = 0.2, val_size: float = 0.2, max_fpr: float = 0.05,
                   n_boot: int = 500, numeric=None, categorical=None) -> dict:
    """Random stratified train/val/test split of ONE dataframe."""
    numeric, categorical = _spec(numeric, categorical)
    X, y, data_report = prepare(df, numeric=numeric, categorical=categorical)
    if len(set(y)) < 2:
        raise ValueError("Need both benign and malicious records to train/evaluate")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    X_fit, X_val, y_fit, y_val = train_test_split(X_tr, y_tr, test_size=val_size, stratify=y_tr, random_state=seed)
    split = {"strategy": "random_stratified", "train": len(X_fit), "val": len(X_val), "test": len(X_te)}
    return _evaluate_models(X_fit, y_fit, X_val, y_val, X_te, y_te, hash_df=df, dataset_id=dataset_id,
                            models=models, seed=seed, max_fpr=max_fpr, n_boot=n_boot,
                            numeric=numeric, categorical=categorical, split_info=split, data_report=data_report)


def run_cross_split(train_df: pd.DataFrame, test_df: pd.DataFrame, dataset_id: str, models=MODELS,
                    seed: int = 42, val_size: float = 0.2, max_fpr: float = 0.05, n_boot: int = 500,
                    numeric=None, categorical=None) -> dict:
    """Train/validate on train_df (stratified val split); test on a SEPARATE dataframe
    (e.g. a different capture day). The test set may lack a class; undefined metrics are NaN."""
    numeric, categorical = _spec(numeric, categorical)
    X_tr, y_tr, rep_tr = prepare(train_df, numeric=numeric, categorical=categorical)
    X_te, y_te, rep_te = prepare(test_df, numeric=numeric, categorical=categorical)
    if len(set(y_tr)) < 2:
        raise ValueError("Training data needs both benign and malicious records")
    X_fit, X_val, y_fit, y_val = train_test_split(X_tr, y_tr, test_size=val_size, stratify=y_tr, random_state=seed)
    split = {"strategy": "cross_split", "train": len(X_fit), "val": len(X_val), "test": len(X_te)}
    return _evaluate_models(X_fit, y_fit, X_val, y_val, X_te, y_te, hash_df=train_df, dataset_id=dataset_id,
                            models=models, seed=seed, max_fpr=max_fpr, n_boot=n_boot, numeric=numeric,
                            categorical=categorical, split_info=split,
                            data_report={"train": rep_tr, "test": rep_te})


def save_model(path: str, result: dict) -> None:
    joblib.dump({"pipeline": result["pipeline"], "metadata": result["metadata"]}, path)


def load_model(path: str) -> dict:
    return joblib.load(path)


def predict(df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """Score records with a saved model. Raises on incompatible features."""
    md = bundle["metadata"]
    X, _, _ = prepare(df, need_label=False, numeric=md.get("numeric"), categorical=md.get("categorical"))
    proba = bundle["pipeline"].predict_proba(X)[:, 1]
    thr = bundle["metadata"]["threshold"]
    out = pd.DataFrame({"probability_malicious": proba,
                        "predicted_label": np.where(proba >= thr, "malicious", "benign"),
                        "model_id": bundle["metadata"]["model_id"]}, index=df.index)
    if "flow_id" in df.columns:
        out.insert(0, "flow_id", df["flow_id"].to_numpy())
    return out
