"""NetSentinel service layer: SQLite persistence + dataset/statistics/training/prediction/alert logic.
Framework-independent (no FastAPI import) so it can be unit-tested directly."""
from __future__ import annotations

import csv
import io
import json
import logging
import math
import os
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ml import cic2017 as C
from ml import pipeline as P
from ml import stats as S
from ml.synthetic import SYNTHETIC_NOTICE, make_synthetic_flows

VERSION = "0.2.0"
log = logging.getLogger("netsentinel")
ALERT_STATUSES = ("new", "under_review", "resolved", "dismissed")
HIGH_PROB = 0.9
ALERT_POLICY = (f"An alert is created for each record predicted malicious (probability >= the model's validation-chosen "
                f"threshold), highest probability first, up to max_alerts. Priority is 'high' if probability >= {HIGH_PROB}, "
                f"otherwise 'medium'. Priority is a rule on an uncalibrated model probability; it is not a risk score "
                f"and not confirmation of an attack.")


class ServiceError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status, self.message = status, message


# ---------------------------------------------------------------- paths / db
def _home() -> Path:
    return Path(os.environ.get("NETSENTINEL_HOME") or Path(__file__).resolve().parents[2])


def data_dir() -> Path:
    return Path(os.environ.get("NETSENTINEL_DATA") or (_home() / "data"))


def storage_dir() -> Path:
    p = _home() / "storage"
    (p / "datasets").mkdir(parents=True, exist_ok=True)
    (p / "models").mkdir(parents=True, exist_ok=True)
    return p


SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets(id TEXT PRIMARY KEY, name TEXT, kind TEXT, files TEXT, nrows INTEGER,
  n_rows INTEGER, class_counts TEXT, cleaning_report TEXT, directionality TEXT, path TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS model_runs(id TEXT PRIMARY KEY, dataset_id TEXT, test_dataset_id TEXT, status TEXT,
  config TEXT, evaluation TEXT, selected_model TEXT, error TEXT, created_at TEXT, finished_at TEXT);
CREATE TABLE IF NOT EXISTS models(id TEXT PRIMARY KEY, run_id TEXT, algorithm TEXT, path TEXT, metadata TEXT,
  metrics TEXT, is_active INTEGER DEFAULT 0, created_at TEXT);
CREATE TABLE IF NOT EXISTS analyses(id TEXT PRIMARY KEY, dataset_id TEXT, model_id TEXT, n_records INTEGER,
  n_malicious INTEGER, params TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS predictions(analysis_id TEXT, row_idx INTEGER, probability REAL, predicted TEXT,
  actual TEXT, category TEXT);
CREATE INDEX IF NOT EXISTS ix_pred_a ON predictions(analysis_id, predicted);
CREATE INDEX IF NOT EXISTS ix_pred_p ON predictions(analysis_id, probability);
CREATE TABLE IF NOT EXISTS alerts(id INTEGER PRIMARY KEY AUTOINCREMENT, analysis_id TEXT, row_idx INTEGER,
  model_id TEXT, probability REAL, priority TEXT, status TEXT, note TEXT, created_at TEXT, updated_at TEXT);
CREATE INDEX IF NOT EXISTS ix_alert_a ON alerts(analysis_id, status);
CREATE TABLE IF NOT EXISTS alert_events(id INTEGER PRIMARY KEY AUTOINCREMENT, alert_id INTEGER, at TEXT,
  from_status TEXT, to_status TEXT, note TEXT);
"""


@contextmanager
def db():
    conn = sqlite3.connect(storage_dir() / "netsentinel.db", timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def clean(o):
    """Make values JSON-safe: numpy -> python, NaN/inf -> None."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, pd.Timestamp):
        return str(o)
    return o


def _row(r, json_fields=()):
    d = dict(r)
    for f in json_fields:
        if d.get(f) is not None:
            d[f] = json.loads(d[f])
    return d


# ---------------------------------------------------------------- datasets
_df_cache: dict[str, pd.DataFrame] = {}
_cache_lock = threading.Lock()
_stats_cache: dict[str, dict] = {}


def list_data_files():
    d = data_dir()
    if not d.is_dir():
        return {"data_dir": str(d), "files": []}
    files = [{"name": p.name, "size_mb": round(p.stat().st_size / 1e6, 1)} for p in sorted(d.glob("*.csv"))]
    return {"data_dir": str(d), "files": files}


def _safe_data_file(name) -> Path:
    if (not isinstance(name, str) or name != os.path.basename(name) or "/" in name or "\\" in name
            or not name.lower().endswith(".csv")):
        raise ServiceError(422, "Invalid file name: choose a .csv file from the data folder listing.")
    p = data_dir() / name
    if not p.is_file():
        raise ServiceError(404, f"File not found in data folder: {name}")
    return p


def feature_spec(kind: str) -> dict:
    if kind == "synthetic":
        return {"numeric": list(P.NUMERIC), "categorical": list(P.CATEGORICAL)}
    return {"numeric": list(C.CIC_NUMERIC), "categorical": []}


def register_dataset(name: str, kind: str = "cic2017_forward", files=None, nrows=None, n: int = 5000,
                     seed: int = 0, drop_duplicates: bool = True) -> dict:
    name = (name or "").strip()
    if not name:
        raise ServiceError(422, "Dataset name is required.")
    if kind == "synthetic":
        if not 200 <= n <= 200000:
            raise ServiceError(422, "Synthetic size must be between 200 and 200000.")
        df = make_synthetic_flows(n, seed=seed)
        df["attack_category"] = np.where(df["label"] == "malicious", "SYNTHETIC-MALICIOUS", "BENIGN")
        df["row_id"] = np.arange(len(df))
        df["source_file"] = "synthetic"
        report = {"synthetic": True, "notice": SYNTHETIC_NOTICE, "n_rows": int(len(df)), "seed": seed}
        directionality = "SYNTHETIC directional flow records (software testing only; results are not benchmarks)"
        files = []
    elif kind == "cic2017_forward":
        if not files:
            raise ServiceError(422, "Select at least one CIC-IDS2017 CSV file.")
        paths = [_safe_data_file(f) for f in files]
        try:
            df, reps = C.load_forward_dataset([str(p) for p in paths], nrows, drop_duplicates)
        except ValueError as e:
            raise ServiceError(422, str(e))
        report = {"files": reps, "nrows_limit_per_file": nrows, "drop_duplicates": drop_duplicates}
        directionality = ("Bidirectional flow records; only forward-direction fields used (compatibility mode, "
                          "NOT strict unidirectional data). Labels come from the whole bidirectional flow.")
    else:
        raise ServiceError(422, "kind must be 'cic2017_forward' or 'synthetic'.")
    df = df.reset_index(drop=True)
    did = _id("ds")
    path = storage_dir() / "datasets" / f"{did}.pkl"
    df.to_pickle(path)
    counts = S.class_distribution(df["label"])
    with db() as c:
        c.execute("INSERT INTO datasets VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                  (did, name, kind, json.dumps(files), nrows, len(df), json.dumps(clean(counts)),
                   json.dumps(clean(report)), directionality, str(path), _now()))
    return get_dataset(did)


def get_dataset(dataset_id: str) -> dict:
    with db() as c:
        r = c.execute("SELECT * FROM datasets WHERE id=?", (dataset_id,)).fetchone()
    if not r:
        raise ServiceError(404, "Dataset not found.")
    d = _row(r, ("files", "class_counts", "cleaning_report"))
    d.pop("path", None)
    d["features"] = feature_spec(d["kind"])
    return d


def list_datasets():
    with db() as c:
        ids = [r["id"] for r in c.execute("SELECT id FROM datasets ORDER BY created_at DESC, rowid DESC")]
    return [get_dataset(i) for i in ids]


def _df(dataset_id: str) -> pd.DataFrame:
    with _cache_lock:
        if dataset_id in _df_cache:
            return _df_cache[dataset_id]
    with db() as c:
        r = c.execute("SELECT path FROM datasets WHERE id=?", (dataset_id,)).fetchone()
    if not r:
        raise ServiceError(404, "Dataset not found.")
    p = Path(r["path"])
    if not p.is_file():
        raise ServiceError(410, "Dataset file is missing from storage; register it again.")
    df = pd.read_pickle(p)
    with _cache_lock:
        if len(_df_cache) >= 3:
            _df_cache.pop(next(iter(_df_cache)))
        _df_cache[dataset_id] = df
    return df


def dataset_preview(dataset_id: str, limit: int = 25, offset: int = 0):
    limit = max(1, min(int(limit), 200))
    offset = max(0, int(offset))
    df = _df(dataset_id)
    page = df.iloc[offset: offset + limit]
    return clean({"total": len(df), "offset": offset, "limit": limit, "columns": list(df.columns),
                  "rows": page.to_dict(orient="records")})


# ---------------------------------------------------------------- statistics
def dataset_statistics(dataset_id: str, sample_cap: int = 100000, seed: int = 42) -> dict:
    if dataset_id in _stats_cache:
        return _stats_cache[dataset_id]
    ds = get_dataset(dataset_id)
    df = _df(dataset_id)
    spec = ds["features"]
    feats = [f for f in spec["numeric"] if f in df.columns]
    const = [f for f in feats if df[f].nunique(dropna=True) <= 1]
    tested = [f for f in feats if f not in const]
    s = df if len(df) <= sample_cap else df.sample(sample_cap, random_state=seed)
    desc = S.describe_features(df, feats)
    mal, ben = s[s.label == "malicious"], s[s.label == "benign"]
    tests = []
    if len(mal) and len(ben):
        for f in tested:
            t = S.compare_numeric(mal[f], ben[f])
            row = {"feature": f, "status": t["status"]}
            if t["status"] == "ok":
                row.update({k: t[k] for k in ("n1", "n2", "median1", "median2", "mean_diff", "mean_diff_ci", "cohens_d",
                                              "welch_p", "mannwhitney_p", "rank_biserial", "interpretation")})
            else:
                row["message"] = t["message"]
            tests.append(row)
        ok = [r for r in tests if r["status"] == "ok"]
        for r, adj in zip(ok, S.bh_adjust([r["mannwhitney_p"] for r in ok])):
            r["mannwhitney_p_bh"] = float(adj)
        tests.sort(key=lambda r: -abs(r.get("rank_biserial") or 0))
    corr = S.correlation_matrix(s, tested, "spearman") if tested else pd.DataFrame()
    chi = []
    for cat in spec["categorical"]:
        if cat in df.columns:
            chi.append({"variable": cat, **S.chi_square_association(df[cat], df["label"])})
    out = clean({
        "dataset_id": dataset_id, "n_records": len(df), "directionality": ds["directionality"],
        "class_distribution": S.class_distribution(df["label"]),
        "n_numeric_features": len(feats), "n_categorical_features": len(spec["categorical"]),
        "constant_features_excluded_from_tests": const,
        "descriptive": desc.to_dict(orient="records"),
        "hypothesis_tests": {
            "group_1": "malicious", "group_2": "benign", "tests": "Welch t-test and Mann-Whitney U (two-sided)",
            "null": "the feature distribution does not differ between malicious and benign records",
            "alpha": 0.05, "n_rows_used": int(len(s)), "sampled": bool(len(df) > sample_cap),
            "sample_seed": seed if len(df) > sample_cap else None,
            "bh_family": "all tested numeric features in this table (Mann-Whitney p-values)",
            "rows": tests,
            "caveats": ["Flows from the same attack burst are not independent, so p-values are optimistic.",
                        "With large n, tiny effects are significant; use the effect sizes.",
                        "Mean-based (Cohen's d) and rank-based effects can disagree for heavy-tailed features.",
                        "Statistical association is not causation."]},
        "chi_square": chi,
        "correlation": {"method": "spearman", "features": list(corr.columns), "matrix": corr.values.tolist()},
    })
    _stats_cache[dataset_id] = out
    return out


def feature_distribution(dataset_id: str, feature: str, bins: int = 40, log: bool = False) -> dict:
    ds = get_dataset(dataset_id)
    df = _df(dataset_id)
    if feature not in ds["features"]["numeric"] or feature not in df.columns:
        raise ServiceError(422, "Unknown numeric feature for this dataset.")
    bins = max(5, min(int(bins), 100))
    x = pd.to_numeric(df[feature], errors="coerce").replace([np.inf, -np.inf], np.nan)
    if log:
        if (x.dropna() < 0).any():
            raise ServiceError(422, "log1p transform needs non-negative values.")
        x = np.log1p(x)
    v = x.dropna()
    if v.empty:
        raise ServiceError(422, "Feature has no finite values.")
    lo, hi = np.percentile(v, [1, 99])
    if lo == hi:
        lo, hi = float(v.min()), float(v.max()) if v.max() > v.min() else float(v.min()) + 1.0
    edges = np.linspace(lo, hi, bins + 1)
    groups = {}
    for cls in ("benign", "malicious"):
        vals = x[df["label"] == cls].dropna().clip(lo, hi)
        counts, _ = np.histogram(vals, edges)
        groups[cls] = {"n": int(len(vals)), "density": (counts / max(len(vals), 1)).tolist()}
    return clean({"feature": feature, "transform": "log1p(x)" if log else "none", "edges": edges.tolist(),
                  "groups": groups,
                  "note": "Densities are per-class proportions. Values outside the 1st-99th percentile are clipped "
                          "into the end bins for display only."})


def statistics_csv(dataset_id: str, part: str = "tests") -> str:
    st = dataset_statistics(dataset_id)
    if part == "descriptive":
        rows = st["descriptive"]
    elif part == "tests":
        rows = [{k: (json.dumps(v) if isinstance(v, list) else v) for k, v in r.items() if k != "interpretation"}
                for r in st["hypothesis_tests"]["rows"]]
    else:
        raise ServiceError(422, "part must be 'descriptive' or 'tests'.")
    return _to_csv(rows)


def _csv_cell(v):
    if isinstance(v, str) and v[:1] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + v  # prevent spreadsheet formula injection
    return "" if v is None else v


def _to_csv(rows) -> str:
    if not rows:
        return ""
    cols = list(dict.fromkeys(k for r in rows for k in r))
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(cols)
    for r in rows:
        w.writerow([_csv_cell(r.get(c)) for c in cols])
    return buf.getvalue()


# ---------------------------------------------------------------- training
def create_training_run(cfg: dict) -> dict:
    ds = get_dataset(cfg["dataset_id"])
    test_id = cfg.get("test_dataset_id")
    if test_id:
        tds = get_dataset(test_id)
        if tds["kind"] != ds["kind"]:
            raise ServiceError(422, "Training and test datasets must be the same kind.")
        if test_id == cfg["dataset_id"]:
            raise ServiceError(422, "Test dataset must differ from the training dataset.")
    if len(ds["class_counts"]["counts"]) < 2:
        raise ServiceError(422, "Training dataset needs both benign and malicious records.")
    models = cfg.get("models") or list(P.MODELS)
    bad = [m for m in models if m not in P.MODELS]
    if bad:
        raise ServiceError(422, f"Unknown model(s): {bad}")
    drop = cfg.get("drop_features") or []
    unknown = [f for f in drop if f not in ds["features"]["numeric"]]
    if unknown:
        raise ServiceError(422, f"Unknown feature(s) to drop: {unknown}")
    max_fpr = float(cfg.get("max_fpr", 0.05))
    if not 0 < max_fpr <= 0.5:
        raise ServiceError(422, "max_fpr must be in (0, 0.5].")
    clean_cfg = {"dataset_id": cfg["dataset_id"], "test_dataset_id": test_id, "models": models,
                 "drop_features": drop, "seed": int(cfg.get("seed", 42)), "max_fpr": max_fpr}
    run_id = _id("run")
    with db() as c:
        c.execute("INSERT INTO model_runs(id,dataset_id,test_dataset_id,status,config,created_at) VALUES(?,?,?,?,?,?)",
                  (run_id, cfg["dataset_id"], test_id, "queued", json.dumps(clean_cfg), _now()))
    return {"run_id": run_id, "config": clean_cfg}


def start_training(cfg: dict, background: bool = True) -> dict:
    run = create_training_run(cfg)
    if background:
        threading.Thread(target=_train, args=(run["run_id"], run["config"]), daemon=True).start()
    else:
        _train(run["run_id"], run["config"])
    return get_run(run["run_id"])


def _set_run(run_id, **fields):
    cols = ", ".join(f"{k}=?" for k in fields)
    with db() as c:
        c.execute(f"UPDATE model_runs SET {cols} WHERE id=?", (*fields.values(), run_id))


def _train(run_id: str, cfg: dict) -> None:
    try:
        _set_run(run_id, status="running")
        ds = get_dataset(cfg["dataset_id"])
        df = _df(cfg["dataset_id"])
        spec = ds["features"]
        numeric = [f for f in spec["numeric"] if f not in cfg["drop_features"]]
        const = [f for f in numeric if df[f].nunique(dropna=True) <= 1]
        numeric = [f for f in numeric if f not in const]
        common = dict(models=tuple(cfg["models"]), seed=cfg["seed"], max_fpr=cfg["max_fpr"],
                      numeric=numeric, categorical=spec["categorical"])
        if cfg.get("test_dataset_id"):
            res = P.run_cross_split(df, _df(cfg["test_dataset_id"]), run_id, **common)
        else:
            res = P.run_experiment(df, run_id, **common)
        models_out = []
        with db() as c:
            for r in res["results"]:
                md = r["metadata"]
                md.update({"dataset_id": cfg["dataset_id"], "run_id": run_id,
                           "test_dataset_id": cfg.get("test_dataset_id")})
                path = storage_dir() / "models" / f"{md['model_id']}.joblib"
                P.save_model(str(path), r)
                metrics = {"val": r["val"], "test": r["test"], "test_ci95": r["test_ci95"],
                           "train_seconds": r["train_seconds"], "inference_seconds": r["inference_seconds"]}
                c.execute("INSERT INTO models(id,run_id,algorithm,path,metadata,metrics,is_active,created_at) "
                          "VALUES(?,?,?,?,?,?,0,?)", (md["model_id"], run_id, r["model"], str(path),
                                                      json.dumps(clean(md)), json.dumps(clean(metrics)), _now()))
                models_out.append({"model_id": md["model_id"], "algorithm": r["model"], **metrics})
            evaluation = clean({
                "data_report": res["data_report"], "split": res["split"], "selection_rule": res["selection_rule"],
                "selected_model": res["selected_model"], "models": models_out,
                "permutation_importance_test_set": res["permutation_importance_test_set"],
                "features": {"numeric": numeric, "categorical": spec["categorical"],
                             "dropped_by_request": cfg["drop_features"], "constant_removed": const},
                "limitations": ["Evaluation assumes independent records; related flows make random splits optimistic.",
                                "Feature importance describes model behaviour, not causation.",
                                "Results apply to this dataset only, not to other networks."]})
            c.execute("UPDATE model_runs SET status='completed', evaluation=?, selected_model=?, finished_at=? WHERE id=?",
                      (json.dumps(evaluation), res["selected_model"], _now(), run_id))
            if res["selected_model"] and not c.execute("SELECT 1 FROM models WHERE is_active=1").fetchone():
                c.execute("UPDATE models SET is_active=1 WHERE id=?", (res["selected_model"],))
    except Exception as e:  # noqa: BLE001 - record failure instead of crashing the worker thread
        log.exception("training run %s failed", run_id)
        _set_run(run_id, status="failed", error=str(e)[:500], finished_at=_now())


def get_run(run_id: str) -> dict:
    with db() as c:
        r = c.execute("SELECT * FROM model_runs WHERE id=?", (run_id,)).fetchone()
        if not r:
            raise ServiceError(404, "Training run not found.")
        active = {m["id"] for m in c.execute("SELECT id FROM models WHERE is_active=1 AND run_id=?", (run_id,))}
    d = _row(r, ("config", "evaluation"))
    d["active_model_id"] = next(iter(active), None)
    return clean(d)


def list_runs():
    with db() as c:
        ids = [r["id"] for r in c.execute("SELECT id FROM model_runs ORDER BY created_at DESC, rowid DESC")]
    out = []
    for i in ids:
        r = get_run(i)
        r.pop("evaluation", None)
        out.append(r)
    return out


def list_models():
    with db() as c:
        rows = c.execute("SELECT id,run_id,algorithm,is_active,created_at,metrics FROM models "
                         "ORDER BY created_at DESC, rowid DESC").fetchall()
    return [clean(_row(r, ("metrics",))) for r in rows]


def promote_model(model_id: str) -> dict:
    with db() as c:
        m = c.execute("SELECT algorithm, run_id FROM models WHERE id=?", (model_id,)).fetchone()
        if not m:
            raise ServiceError(404, "Model not found.")
        if m["algorithm"] == "dummy":
            raise ServiceError(409, "The dummy baseline cannot be the active inference model.")
        run = c.execute("SELECT status FROM model_runs WHERE id=?", (m["run_id"],)).fetchone()
        if not run or run["status"] != "completed":
            raise ServiceError(409, "Only models from completed, evaluated runs can be promoted.")
        c.execute("UPDATE models SET is_active=0")
        c.execute("UPDATE models SET is_active=1 WHERE id=?", (model_id,))
    return {"active_model_id": model_id}


def _active_model(c):
    return c.execute("SELECT * FROM models WHERE is_active=1").fetchone()


def evaluation_export(run_id: str) -> dict:
    run = get_run(run_id)
    if run["status"] != "completed":
        raise ServiceError(409, f"Run is {run['status']}; no evaluation to export.")
    ds = get_dataset(run["dataset_id"])
    return {"project": "NetSentinel", "run_id": run_id, "config": run["config"],
            "dataset": {"id": ds["id"], "name": ds["name"], "kind": ds["kind"], "directionality": ds["directionality"],
                        "class_counts": ds["class_counts"], "cleaning_report": ds["cleaning_report"]},
            "evaluation": run["evaluation"], "exported_at": _now()}


# ---------------------------------------------------------------- predictions & alerts
def run_predictions(dataset_id: str, model_id: str | None = None, limit: int | None = None,
                    max_alerts: int = 1000) -> dict:
    with db() as c:
        m = (c.execute("SELECT * FROM models WHERE id=?", (model_id,)).fetchone() if model_id else _active_model(c))
    if not m:
        raise ServiceError(409 if not model_id else 404,
                           "No active model. Train a model and promote one." if not model_id else "Model not found.")
    if not Path(m["path"]).is_file():
        raise ServiceError(410, "Model file is missing from storage.")
    bundle = P.load_model(m["path"])
    md = bundle["metadata"]
    ds = get_dataset(dataset_id)
    df = _df(dataset_id)
    if limit:
        df = df.head(int(limit))
    try:
        out = P.predict(df, bundle)
    except ValueError as e:
        raise ServiceError(422, f"Dataset is incompatible with this model: {e}")
    prob = out["probability_malicious"].to_numpy()
    pred = out["predicted_label"].to_numpy()
    idx = df.index.to_numpy()
    actual = df["label"].to_numpy() if "label" in df.columns else np.array([None] * len(df))
    cat = df["attack_category"].to_numpy() if "attack_category" in df.columns else np.array([None] * len(df))
    in_sample = dataset_id == md.get("dataset_id")
    thr = md["threshold"]
    metrics = None
    if "label" in df.columns and df["label"].nunique() == 2:
        metrics = P.compute_metrics((df["label"] == "malicious").astype(int).to_numpy(), prob, thr)
    mal_idx = np.where(pred == "malicious")[0]
    order = mal_idx[np.argsort(-prob[mal_idx], kind="stable")]
    chosen = order[: max(0, int(max_alerts))]
    aid = _id("an")
    now = _now()
    params = clean({"model_algorithm": md["algorithm"], "threshold": thr, "in_sample": in_sample,
                    "limit": limit, "max_alerts": max_alerts, "n_alerts": len(chosen),
                    "n_alerts_suppressed": int(len(order) - len(chosen)), "metrics_on_this_dataset": metrics,
                    "alert_policy": ALERT_POLICY, "dataset_kind": ds["kind"],
                    "warning": ("This dataset was used to train the model; these metrics are in-sample and optimistic."
                                if in_sample else None)})
    with db() as c:
        c.execute("INSERT INTO analyses VALUES(?,?,?,?,?,?,?)",
                  (aid, dataset_id, m["id"], len(df), int(len(mal_idx)), json.dumps(params), now))
        c.executemany("INSERT INTO predictions VALUES(?,?,?,?,?,?)",
                      [(aid, int(idx[i]), float(prob[i]), str(pred[i]),
                        None if actual[i] is None else str(actual[i]),
                        None if cat[i] is None else str(cat[i])) for i in range(len(df))])
        c.executemany("INSERT INTO alerts(analysis_id,row_idx,model_id,probability,priority,status,note,created_at,updated_at)"
                      " VALUES(?,?,?,?,?,?,?,?,?)",
                      [(aid, int(idx[i]), m["id"], float(prob[i]), "high" if prob[i] >= HIGH_PROB else "medium",
                        "new", "", now, now) for i in chosen])
    return get_analysis(aid)


def get_analysis(aid: str) -> dict:
    with db() as c:
        r = c.execute("SELECT * FROM analyses WHERE id=?", (aid,)).fetchone()
    if not r:
        raise ServiceError(404, "Analysis not found.")
    d = _row(r, ("params",))
    d["n_benign"] = d["n_records"] - d["n_malicious"]
    return clean(d)


def list_analyses():
    with db() as c:
        ids = [r["id"] for r in c.execute("SELECT id FROM analyses ORDER BY created_at DESC, rowid DESC")]
    return [get_analysis(i) for i in ids]


_DISPLAY_COLS = ["flow_id", "source_file", "src_ip", "src_port", "dst_ip", "dst_port", "protocol", "packet_count",
                 "byte_count", "duration"]


def analysis_records(aid: str, predicted=None, actual=None, category=None, min_prob=None, max_prob=None,
                     sort: str = "prob_desc", page: int = 1, page_size: int = 25) -> dict:
    a = get_analysis(aid)
    page, page_size = max(1, int(page)), max(1, min(int(page_size), 200))
    where, args = ["analysis_id=?"], [aid]
    for col, val in (("predicted", predicted), ("actual", actual), ("category", category)):
        if val:
            where.append(f"{col}=?")
            args.append(val)
    if min_prob is not None:
        where.append("probability>=?")
        args.append(float(min_prob))
    if max_prob is not None:
        where.append("probability<=?")
        args.append(float(max_prob))
    order = {"prob_desc": "probability DESC, row_idx", "prob_asc": "probability ASC, row_idx",
             "row": "row_idx"}.get(sort)
    if not order:
        raise ServiceError(422, "sort must be prob_desc, prob_asc or row.")
    w = " AND ".join(where)
    with db() as c:
        total = c.execute(f"SELECT COUNT(*) FROM predictions WHERE {w}", args).fetchone()[0]
        rows = c.execute(f"SELECT row_idx,probability,predicted,actual,category FROM predictions WHERE {w} "
                         f"ORDER BY {order} LIMIT ? OFFSET ?", (*args, page_size, (page - 1) * page_size)).fetchall()
    df = _df(a["dataset_id"])
    cols = [c_ for c_ in _DISPLAY_COLS if c_ in df.columns]
    feats = df.loc[[r["row_idx"] for r in rows], cols] if rows else pd.DataFrame(columns=cols)
    out = []
    for r, (_, f) in zip(rows, feats.iterrows()):
        out.append({**dict(r), **f.to_dict()})
    return clean({"total": total, "page": page, "page_size": page_size, "rows": out,
                  "labels_available": bool(out and out[0]["actual"] is not None)})


def record_detail(aid: str, row_idx: int) -> dict:
    a = get_analysis(aid)
    with db() as c:
        p = c.execute("SELECT * FROM predictions WHERE analysis_id=? AND row_idx=?", (aid, row_idx)).fetchone()
        if not p:
            raise ServiceError(404, "Record not found in this analysis.")
        al = [dict(x) for x in c.execute("SELECT id,priority,status FROM alerts WHERE analysis_id=? AND row_idx=?",
                                         (aid, row_idx))]
        run = c.execute("SELECT m.run_id, r.evaluation FROM models m JOIN model_runs r ON r.id=m.run_id "
                        "WHERE m.id=?", (a["model_id"],)).fetchone()
    df = _df(a["dataset_id"])
    feats = df.loc[row_idx].to_dict()
    imp = json.loads(run["evaluation"]).get("permutation_importance_test_set") if run and run["evaluation"] else None
    return clean({
        "analysis_id": aid, "row_idx": row_idx, "model_id": a["model_id"],
        "prediction": {"predicted_label": p["predicted"], "probability_malicious": p["probability"],
                       "threshold": a["params"]["threshold"],
                       "note": "Class probability from an uncalibrated model; not a risk score or confirmation of an attack."},
        "actual_label": p["actual"], "attack_category": p["category"],
        "record": feats, "alerts": al,
        "provenance": {"dataset_id": a["dataset_id"], "source_file": feats.get("source_file"),
                       "source_row": feats.get("row_id")},
        "global_feature_importance": imp,
        "explanation_note": "Importance shown is global (permutation importance on the evaluation test set), "
                            "not a local explanation of this single prediction, and not evidence of causation."})


def list_alerts(status=None, priority=None, analysis_id=None, page: int = 1, page_size: int = 25) -> dict:
    page, page_size = max(1, int(page)), max(1, min(int(page_size), 200))
    where, args = ["1=1"], []
    if status:
        if status not in ALERT_STATUSES:
            raise ServiceError(422, f"status must be one of {ALERT_STATUSES}")
        where.append("a.status=?")
        args.append(status)
    if priority:
        where.append("a.priority=?")
        args.append(priority)
    if analysis_id:
        where.append("a.analysis_id=?")
        args.append(analysis_id)
    w = " AND ".join(where)
    with db() as c:
        total = c.execute(f"SELECT COUNT(*) FROM alerts a WHERE {w}", args).fetchone()[0]
        rows = c.execute(
            f"SELECT a.*, p.actual AS actual_label, p.category AS attack_category FROM alerts a "
            f"LEFT JOIN predictions p ON p.analysis_id=a.analysis_id AND p.row_idx=a.row_idx WHERE {w} "
            f"ORDER BY a.probability DESC, a.id LIMIT ? OFFSET ?", (*args, page_size, (page - 1) * page_size)).fetchall()
    return clean({"total": total, "page": page, "page_size": page_size, "rows": [dict(r) for r in rows],
                  "policy": ALERT_POLICY})


def get_alert(alert_id: int) -> dict:
    with db() as c:
        r = c.execute("SELECT * FROM alerts WHERE id=?", (alert_id,)).fetchone()
        if not r:
            raise ServiceError(404, "Alert not found.")
        ev = [dict(e) for e in c.execute("SELECT at,from_status,to_status,note FROM alert_events WHERE alert_id=? "
                                         "ORDER BY id", (alert_id,))]
    return clean({**dict(r), "events": ev, "verified_incident": False,
                  "note_about_status": "'resolved' means the analyst closed the review; it does not mean an attack was "
                                       "independently confirmed."})


def update_alert(alert_id: int, status: str | None = None, note: str | None = None) -> dict:
    if status is None and note is None:
        raise ServiceError(422, "Provide a status and/or a note.")
    if status is not None and status not in ALERT_STATUSES:
        raise ServiceError(422, f"status must be one of {ALERT_STATUSES}")
    if note is not None and len(note) > 1000:
        raise ServiceError(422, "Note is limited to 1000 characters.")
    with db() as c:
        r = c.execute("SELECT status FROM alerts WHERE id=?", (alert_id,)).fetchone()
        if not r:
            raise ServiceError(404, "Alert not found.")
        new_status = status or r["status"]
        c.execute("UPDATE alerts SET status=?, note=COALESCE(?, note), updated_at=? WHERE id=?",
                  (new_status, note, _now(), alert_id))
        c.execute("INSERT INTO alert_events(alert_id,at,from_status,to_status,note) VALUES(?,?,?,?,?)",
                  (alert_id, _now(), r["status"], new_status, note))
    return get_alert(alert_id)


def alerts_csv(analysis_id: str | None = None, status: str | None = None) -> str:
    rows = list_alerts(status=status, analysis_id=analysis_id, page=1, page_size=200)["rows"]
    # page through everything
    total, page = list_alerts(status=status, analysis_id=analysis_id, page=1, page_size=1)["total"], 2
    while len(rows) < total:
        rows += list_alerts(status=status, analysis_id=analysis_id, page=page, page_size=200)["rows"]
        page += 1
    # 'id' alone is ambiguous in an exported file
    rows = [{"alert_id": r["id"], **{k: v for k, v in r.items() if k != "id"}} for r in rows]
    return _to_csv(rows)


# ---------------------------------------------------------------- overview / system
def overview() -> dict:
    with db() as c:
        n_ds = c.execute("SELECT COUNT(*) FROM datasets").fetchone()[0]
        n_runs = c.execute("SELECT COUNT(*) FROM model_runs").fetchone()[0]
        open_alerts = c.execute("SELECT COUNT(*) FROM alerts WHERE status IN ('new','under_review')").fetchone()[0]
        last = c.execute("SELECT id FROM analyses ORDER BY created_at DESC, rowid DESC LIMIT 1").fetchone()
        act = _active_model(c)
    latest = get_analysis(last["id"]) if last else None
    ds = get_dataset(latest["dataset_id"]) if latest else None
    return clean({"datasets": n_ds, "training_runs": n_runs, "open_alerts": open_alerts,
                  "active_model": ({"id": act["id"], "algorithm": act["algorithm"]} if act else None),
                  "latest_analysis": latest,
                  "latest_dataset": ({"id": ds["id"], "name": ds["name"], "directionality": ds["directionality"],
                                      "class_counts": ds["class_counts"]} if ds else None)})


def system_info() -> dict:
    with db() as c:
        act = _active_model(c)
        counts = {t: c.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                  for t in ("datasets", "model_runs", "models", "analyses", "alerts")}
    return clean({"application": "NetSentinel", "version": VERSION, "backend": "ok", "database": "ok",
                  "database_path": str(storage_dir() / "netsentinel.db"), "storage_dir": str(storage_dir()),
                  "data_dir": str(data_dir()), "counts": counts,
                  "active_model": act["id"] if act else None,
                  "active_model_file_present": bool(act and Path(act["path"]).is_file()),
                  "supported_input": "CIC-IDS2017 MachineLearningCSV day files (forward-direction compatibility mode) "
                                     "or a labelled synthetic test dataset",
                  "live_packet_capture": "not implemented (offline CSV analysis only)",
                  "alert_policy": ALERT_POLICY,
                  "security_controls": ["input validation", "file access restricted to the data folder (names only)",
                                        "CSV formula-injection escaping on exports",
                                        "no authentication (local single-user academic use)"]})
