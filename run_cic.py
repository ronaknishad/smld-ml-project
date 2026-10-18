"""Run NetSentinel on CIC-IDS2017 MachineLearningCSV day files (forward-direction compatibility mode).

Examples (PowerShell):
  # random stratified split over pooled files (optimistic - related flows can sit on both sides)
  python run_cic.py --files data\\Tuesday-WorkingHours.pcap_ISCX.csv data\\Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
  # cross-day: train on --files, test on a DIFFERENT file (stricter, usually lower scores)
  python run_cic.py --files data\\Tuesday-WorkingHours.pcap_ISCX.csv --test-file data\\Wednesday-workingHours.pcap_ISCX.csv
  # quick trial on the first 100000 rows of each file
  python run_cic.py --files data\\Tuesday-WorkingHours.pcap_ISCX.csv --nrows 100000
"""
import argparse
import os

import pandas as pd

from ml import cic2017 as C
from ml import pipeline as P
from ml import stats

pd.set_option("display.width", 180)
ap = argparse.ArgumentParser()
ap.add_argument("--files", nargs="+", required=True, help="CIC day CSVs used for training/validation")
ap.add_argument("--test-file", nargs="+", default=None, help="CSV(s) from a different day used only for testing")
ap.add_argument("--nrows", type=int, default=None, help="read only first N rows per file (records a partial sample!)")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--stats-sample", type=int, default=200000, help="max rows for hypothesis tests (seeded random sample)")
ap.add_argument("--keep-duplicates", action="store_true")
ap.add_argument("--drop-features", nargs="*", default=[], help="feature names to exclude (ablation), e.g. init_win_bytes_fwd min_seg_size_fwd")
ap.add_argument("--out", default="artifacts")
a = ap.parse_args()

print("MODE: CIC-IDS2017 forward-direction compatibility (bidirectional source records; not strict unidirectional)")
if a.nrows:
    print(f"WARNING: only the first {a.nrows} rows of each file were read; results are NOT on the full files.")

train, reps = C.load_forward_dataset(a.files, a.nrows, drop_duplicates=not a.keep_duplicates)
print("\nCleaning report (per file):")
for r in reps:
    print(" ", r)
print("\nClass distribution:", stats.class_distribution(train["label"]))
print("Attack categories:", train["attack_category"].value_counts().to_dict())

bad = set(a.drop_features) - set(C.CIC_NUMERIC)
if bad:
    raise SystemExit(f"Unknown feature(s) in --drop-features: {sorted(bad)}")
feats = [f for f in C.CIC_NUMERIC if f not in a.drop_features]
const = [f for f in feats if train[f].nunique(dropna=True) <= 1]
if const:
    print("\nConstant features removed (zero variance in training data):", const)
feats = [f for f in feats if f not in const]
if a.drop_features:
    print("Ablation - features dropped by request:", a.drop_features)
print("Features used for statistics and models:", feats)
print("\nDescriptive statistics:")
print(stats.describe_features(train, feats)[["feature", "n", "missing", "mean", "median", "std", "iqr", "skewness"]]
      .round(4).to_string(index=False))

s = train if len(train) <= a.stats_sample else train.sample(a.stats_sample, random_state=a.seed)
mal, ben = s[s.label == "malicious"], s[s.label == "benign"]
rows = []
for f in feats:
    t = stats.compare_numeric(mal[f], ben[f])
    if t["status"] == "ok":
        rows.append({"feature": f, "median_mal": t["median1"], "median_ben": t["median2"],
                     "mw_p": t["mannwhitney_p"], "rank_biserial": t["rank_biserial"], "cohens_d": t["cohens_d"]})
if rows:
    tab = pd.DataFrame(rows)
    tab["mw_p_BH"] = stats.bh_adjust(tab["mw_p"])
    print(f"\nMalicious vs benign, Mann-Whitney U on a seeded sample of {len(s)} rows; BH family = these {len(tab)} features:")
    tab = tab.reindex(tab.rank_biserial.abs().sort_values(ascending=False).index)
    disp = tab.copy()
    for c in ["mw_p", "mw_p_BH"]:
        disp[c] = disp[c].map(lambda v: f"{v:.2e}")
    print(disp.round({"median_mal": 4, "median_ben": 4, "rank_biserial": 3, "cohens_d": 3}).to_string(index=False))
    print("(large n makes tiny effects 'significant' - read effect sizes, not p alone)")

if a.test_file:
    test, treps = C.load_forward_dataset(a.test_file, a.nrows, drop_duplicates=not a.keep_duplicates)
    print("\nTest-day cleaning:", treps)
    res = P.run_cross_split(train, test, "cic2017-" + "+".join(r["source_file"][:3] for r in reps),
                            seed=a.seed, numeric=feats, categorical=[])
else:
    res = P.run_experiment(train, "cic2017-" + "+".join(r["source_file"][:3] for r in reps),
                           seed=a.seed, numeric=feats, categorical=[])

print("\nSplit:", res["split"])
print(f"\n{'model':8s} {'thr':>8s} {'acc':>6s} {'prec':>6s} {'recall':>6s} {'f1':>6s} {'FPR':>7s} {'ROC':>6s} {'PR':>6s}  recall95CI      FPR95CI          train_s")
for r in res["results"]:
    m, ci = r["test"], r["test_ci95"]
    print(f"{r['model']:8s} {m['threshold']:8.4f} {m['accuracy']:6.3f} {m['precision']:6.3f} {m['recall']:6.3f} {m['f1']:6.3f} "
          f"{m['fpr']:7.4f} {m['roc_auc']:6.3f} {m['pr_auc']:6.3f}  [{ci['recall'][0]:.3f},{ci['recall'][1]:.3f}]  "
          f"[{ci['fpr'][0]:.4f},{ci['fpr'][1]:.4f}]  {r['train_seconds']:.1f}")
    print(f"         confusion (test): {m['confusion_matrix']}")
print("\nSelected on VALIDATION:", res["selected_model"], "|", res["selection_rule"])
if res["permutation_importance_test_set"]:
    print("Top permutation importances (drop in PR-AUC on test set):")
    for d in res["permutation_importance_test_set"][:6]:
        print(f"  {d['feature']:20s} {d['mean_drop_pr_auc']:.4f}")

sel = next(r for r in res["results"] if r["metadata"]["model_id"] == res["selected_model"])
os.makedirs(a.out, exist_ok=True)
P.save_model(os.path.join(a.out, "cic_model.joblib"), sel)
print(f"\nSaved selected model to {os.path.join(a.out, 'cic_model.joblib')}")
