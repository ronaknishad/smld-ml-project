"""Statistical analysis helpers. Every function returns computed values only;
interpretation text is produced only when the calculation succeeded."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as st


def describe_features(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        v = s.dropna()
        q1, q3 = (v.quantile(0.25), v.quantile(0.75)) if len(v) else (np.nan, np.nan)
        rows.append({
            "feature": c, "n": int(len(v)), "missing": int(s.isna().sum()),
            "mean": v.mean() if len(v) else np.nan,
            "median": v.median() if len(v) else np.nan,
            "std": v.std(ddof=1) if len(v) > 1 else np.nan,
            "variance": v.var(ddof=1) if len(v) > 1 else np.nan,
            "min": v.min() if len(v) else np.nan, "max": v.max() if len(v) else np.nan,
            "q25": q1, "q75": q3, "iqr": q3 - q1,
            # shape statistics are undefined for constant data; skip instead of emitting numerical noise
            "skewness": st.skew(v, bias=False) if len(v) > 2 and v.nunique() > 1 else np.nan,
            "kurtosis": st.kurtosis(v, bias=False) if len(v) > 3 and v.nunique() > 1 else np.nan,
        })
    return pd.DataFrame(rows)


def class_distribution(y) -> dict:
    counts = pd.Series(y).value_counts()
    total = int(counts.sum())
    ratio = float(counts.max() / counts.min()) if len(counts) > 1 else float("nan")
    return {
        "counts": {str(k): int(v) for k, v in counts.items()},
        "percent": {str(k): float(v / total * 100) for k, v in counts.items()},
        "imbalance_ratio": ratio, "total": total,
    }


def correlation_matrix(df: pd.DataFrame, cols: list[str], method: str = "spearman") -> pd.DataFrame:
    if method not in ("pearson", "spearman"):
        raise ValueError("method must be 'pearson' or 'spearman'")
    X = df[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return X.corr(method=method)


def compare_numeric(a, b, alpha: float = 0.05) -> dict:
    """Welch t-test (mean difference, CI, Cohen's d) and Mann-Whitney U
    (rank-biserial effect size) for two independent groups."""
    a = pd.Series(a, dtype=float).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    b = pd.Series(b, dtype=float).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return {"status": "insufficient_data", "n1": n1, "n2": n2,
                "message": "Each group needs at least 2 observations."}
    v1, v2 = a.var(ddof=1), b.var(ddof=1)
    if v1 == 0 and v2 == 0:
        return {"status": "insufficient_data", "n1": n1, "n2": n2,
                "message": "Both groups are constant; tests are undefined."}
    t = st.ttest_ind(a, b, equal_var=False)
    diff = float(a.mean() - b.mean())
    se = np.sqrt(v1 / n1 + v2 / n2)
    dof = (v1 / n1 + v2 / n2) ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
    tcrit = st.t.ppf(1 - alpha / 2, dof)
    sp = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    d = diff / sp if sp > 0 else float("nan")
    u = st.mannwhitneyu(a, b, alternative="two-sided")
    rank_biserial = float(2 * u.statistic / (n1 * n2) - 1)
    res = {
        "status": "ok", "n1": n1, "n2": n2, "alpha": alpha,
        "median1": float(np.median(a)), "median2": float(np.median(b)),
        "mean1": float(a.mean()), "mean2": float(b.mean()),
        "welch_t": float(t.statistic), "welch_p": float(t.pvalue), "welch_df": float(dof),
        "mean_diff": diff, "mean_diff_ci": (float(diff - tcrit * se), float(diff + tcrit * se)),
        "cohens_d": float(d),
        "mannwhitney_U": float(u.statistic), "mannwhitney_p": float(u.pvalue),
        "rank_biserial": rank_biserial,
    }
    res["interpretation"] = (
        f"Group 1 median {res['median1']:.4g} vs group 2 median {res['median2']:.4g}. "
        f"Welch p={res['welch_p']:.3g} (Cohen's d={d:.2f}); Mann-Whitney p={res['mannwhitney_p']:.3g} "
        f"(rank-biserial={rank_biserial:.2f}). This describes a difference in this sample; "
        f"it does not show causation, and with large n small effects can be significant."
    )
    return res


def chi_square_association(x, y, alpha: float = 0.05) -> dict:
    ct = pd.crosstab(pd.Series(x), pd.Series(y))
    r, c = ct.shape
    if r < 2 or c < 2:
        return {"status": "insufficient_data", "message": "Need at least 2 categories in each variable."}
    n = int(ct.to_numpy().sum())
    chi2, p, dof, expected = st.chi2_contingency(ct, correction=False)
    min_expected = float(expected.min())
    res = {"status": "ok", "n": n, "chi2": float(chi2), "dof": int(dof), "p": float(p),
           "cramers_v": float(np.sqrt(chi2 / (n * (min(r, c) - 1)))),
           "min_expected_count": min_expected, "alpha": alpha, "warnings": []}
    if min_expected < 5:
        res["warnings"].append("Some expected counts < 5; chi-square approximation may be unreliable.")
        if (r, c) == (2, 2):
            res["fisher_exact_p"] = float(st.fisher_exact(ct.to_numpy())[1])
    res["interpretation"] = (f"chi2={chi2:.3g}, p={p:.3g}, Cramer's V={res['cramers_v']:.2f}. "
                             "Association in this sample only; not causal.")
    return res


def bh_adjust(pvalues) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (FDR), returned in input order."""
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    if m == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * m / (np.arange(m) + 1)
    adj_sorted = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.clip(adj_sorted, 0, 1)
    return out


def bootstrap_ci(y_true, y_pred, metric_fn, n_boot: int = 1000, alpha: float = 0.05, seed: int = 42):
    """Percentile bootstrap CI resampling records with replacement.
    Assumes records are independent; document this where used."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        v = metric_fn(y_true[idx], y_pred[idx])
        if not np.isnan(v):
            vals.append(v)
    if len(vals) < 10:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 100 * alpha / 2)), float(np.percentile(vals, 100 * (1 - alpha / 2))))
