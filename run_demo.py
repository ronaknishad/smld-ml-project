"""End-to-end demo on SYNTHETIC data: stats -> train -> evaluate -> save -> predict.
Replace make_synthetic_flows() with a real, validated dataset for real experiments."""
import os

import pandas as pd

from ml import pipeline as P
from ml import stats
from ml.synthetic import SYNTHETIC_NOTICE, make_synthetic_flows

pd.set_option("display.width", 160)
print("=" * 70, "\n", SYNTHETIC_NOTICE, "\n", "=" * 70, sep="")

df = make_synthetic_flows(n=5000, seed=0)
print("\nClass distribution:", stats.class_distribution(df["label"]))
print("\nDescriptive statistics (selected):")
print(stats.describe_features(df, ["packet_count", "mean_pkt_len", "duration"])
      [["feature", "n", "mean", "median", "std", "iqr", "skewness"]].round(3).to_string(index=False))

mal, ben = df[df.label == "malicious"], df[df.label == "benign"]
tests = {c: stats.compare_numeric(mal[c], ben[c]) for c in ["packet_count", "mean_pkt_len", "duration", "mean_iat"]}
adj = stats.bh_adjust([t["mannwhitney_p"] for t in tests.values()])
print("\nMalicious vs benign (Mann-Whitney, BH-adjusted over this 4-test family):")
for (c, t), a in zip(tests.items(), adj):
    print(f"  {c:14s} p={t['mannwhitney_p']:.2e} adj_p={a:.2e} rank-biserial={t['rank_biserial']:+.2f} d={t['cohens_d']:+.2f}")
print("\nProtocol vs label:", {k: v for k, v in stats.chi_square_association(df.protocol, df.label).items() if k != "interpretation"})

res = P.run_experiment(df, dataset_id="synthetic-demo", seed=42)
print("\nSplit:", res["split"], "| data report:", res["data_report"])
print(f"\n{'model':8s} {'thr':>6s} {'acc':>6s} {'prec':>6s} {'recall':>6s} {'f1':>6s} {'FPR':>6s} {'ROC':>6s} {'PR':>6s}  recall95CI        FPR95CI           train_s")
for r in res["results"]:
    m, ci = r["test"], r["test_ci95"]
    print(f"{r['model']:8s} {m['threshold']:6.3f} {m['accuracy']:6.3f} {m['precision']:6.3f} {m['recall']:6.3f} {m['f1']:6.3f} "
          f"{m['fpr']:6.3f} {m['roc_auc']:6.3f} {m['pr_auc']:6.3f}  [{ci['recall'][0]:.2f},{ci['recall'][1]:.2f}]   "
          f"[{ci['fpr'][0]:.3f},{ci['fpr'][1]:.3f}]   {r['train_seconds']:.2f}")
print("\nSelected:", res["selected_model"], "|", res["selection_rule"])
print("Top permutation importances (test set):")
for d in res["permutation_importance_test_set"][:4]:
    print(f"  {d['feature']:14s} {d['mean_drop_pr_auc']:.4f}")

sel = next(r for r in res["results"] if r["metadata"]["model_id"] == res["selected_model"])
os.makedirs("artifacts", exist_ok=True)
P.save_model("artifacts/model.joblib", sel)
out = P.predict(df.head(5), P.load_model("artifacts/model.joblib"))
print("\nReloaded model predictions (first 5):\n", out.round(3).to_string(index=False))
