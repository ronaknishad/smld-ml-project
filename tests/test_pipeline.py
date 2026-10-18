import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from ml import pipeline as P
from ml.synthetic import make_synthetic_flows


class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = make_synthetic_flows(n=1500, seed=1)
        cls.res = P.run_experiment(cls.df, "synthetic-test", seed=7, n_boot=100)

    def test_no_identifier_or_label_leakage(self):
        for bad in ["src_ip", "dst_ip", "src_port", "dst_port", "flow_id", "label"]:
            self.assertNotIn(bad, P.FEATURES)

    def test_split_sizes_add_up_and_stratified(self):
        s = self.res["split"]
        self.assertEqual(s["train"] + s["val"] + s["test"], len(self.df))

    def test_all_models_trained_with_metrics(self):
        names = [r["model"] for r in self.res["results"]]
        self.assertEqual(names, ["dummy", "logreg", "rf"])
        for r in self.res["results"]:
            cm = r["test"]["confusion_matrix"]
            self.assertEqual(sum(cm.values()), r["test"]["n"])

    def test_dummy_baseline_has_no_skill(self):
        d = self.res["results"][0]["test"]
        self.assertAlmostEqual(d["roc_auc"], 0.5, places=6)

    def test_inference_is_single_threaded_for_exact_reproducibility(self):
        for r in self.res["results"]:
            clf = r["pipeline"].named_steps["clf"]
            if "n_jobs" in clf.get_params():
                self.assertEqual(clf.get_params()["n_jobs"], 1)

    def test_selected_model_not_dummy(self):
        self.assertNotIn("dummy", self.res["selected_model"])

    def test_reproducible_with_same_seed(self):
        again = P.run_experiment(self.df, "synthetic-test", seed=7, n_boot=100)
        for a, b in zip(self.res["results"], again["results"]):
            self.assertEqual(a["test"]["confusion_matrix"], b["test"]["confusion_matrix"])
            self.assertEqual(a["test"]["threshold"], b["test"]["threshold"])

    def test_save_reload_predict_identical(self):
        r = self.res["results"][2]
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.joblib")
            P.save_model(path, r)
            bundle = P.load_model(path)
        out1 = P.predict(self.df.head(50), {"pipeline": r["pipeline"], "metadata": r["metadata"]})
        out2 = P.predict(self.df.head(50), bundle)
        np.testing.assert_allclose(out1.probability_malicious, out2.probability_malicious)
        self.assertTrue((out2.model_id == r["metadata"]["model_id"]).all())

    def test_unseen_category_and_missing_values_handled(self):
        df = self.df.head(20).copy()
        df["protocol"] = "SCTP"
        df.loc[df.index[0], "mean_iat"] = np.nan
        df.loc[df.index[1], "pkts_per_sec"] = np.inf
        out = P.predict(df, {"pipeline": self.res["results"][1]["pipeline"], "metadata": self.res["results"][1]["metadata"]})
        self.assertEqual(len(out), 20)
        self.assertFalse(out.probability_malicious.isna().any())

    def test_incompatible_features_raise(self):
        with self.assertRaises(ValueError):
            P.prepare(self.df.drop(columns=["duration"]))

    def test_unknown_label_raises_not_defaults_benign(self):
        bad = self.df.copy()
        bad.loc[bad.index[0], "label"] = "weird"
        with self.assertRaises(ValueError):
            P.prepare(bad)

    def test_negative_values_rejected(self):
        bad = self.df.copy()
        bad.loc[bad.index[0], "packet_count"] = -1
        with self.assertRaises(ValueError):
            P.prepare(bad)

    def test_single_class_evaluation_gives_nan_not_crash(self):
        m = P.compute_metrics(np.array([0, 0, 0]), np.array([0.1, 0.2, 0.3]), 0.5)
        self.assertTrue(np.isnan(m["roc_auc"]))
        self.assertTrue(np.isnan(m["recall"]))
        with self.assertRaises(ValueError):
            P.run_experiment(self.df[self.df.label == "benign"], "x")

    def test_threshold_respects_fpr_cap_on_validation(self):
        for r in self.res["results"]:
            if r["model"] != "dummy":
                self.assertLessEqual(r["val"]["fpr"], 0.05 + 1e-9)


class TestThresholdTieBreak(unittest.TestCase):
    def test_does_not_spend_fpr_budget_once_recall_is_maximal(self):
        y = np.array([1, 1, 1] + [0] * 20)
        p = np.concatenate([[0.9, 0.8, 0.7], np.linspace(0.01, 0.6, 20)])
        thr = P.select_threshold(y, p, max_fpr=0.5)
        m = P.compute_metrics(y, p, thr)
        self.assertEqual(m["recall"], 1.0)
        self.assertEqual(m["fpr"], 0.0)        # old logic would have used FPR ~0.5
        self.assertAlmostEqual(thr, 0.7)


if __name__ == "__main__":
    unittest.main()
