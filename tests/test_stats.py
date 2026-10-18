import math
import unittest

import numpy as np
import pandas as pd

from ml import stats


class TestStats(unittest.TestCase):
    def test_describe_known(self):
        d = stats.describe_features(pd.DataFrame({"x": [1, 2, 3, 4, 5]}), ["x"]).iloc[0]
        self.assertEqual(d.n, 5)
        self.assertAlmostEqual(d["mean"], 3.0)
        self.assertAlmostEqual(d["median"], 3.0)
        self.assertAlmostEqual(d["std"], math.sqrt(2.5))
        self.assertAlmostEqual(d["iqr"], 2.0)

    def test_describe_counts_missing_and_inf(self):
        d = stats.describe_features(pd.DataFrame({"x": [1, np.nan, np.inf, 3]}), ["x"]).iloc[0]
        self.assertEqual(d.n, 2)
        self.assertEqual(d.missing, 2)

    def test_constant_column_shape_stats_are_nan_without_warnings(self):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            d = stats.describe_features(pd.DataFrame({"x": [0.0] * 50}), ["x"]).iloc[0]
        self.assertTrue(np.isnan(d["skewness"]) and np.isnan(d["kurtosis"]))
        self.assertEqual(d["std"], 0.0)

    def test_class_distribution(self):
        d = stats.class_distribution(["a"] * 90 + ["b"] * 10)
        self.assertEqual(d["counts"], {"a": 90, "b": 10})
        self.assertAlmostEqual(d["imbalance_ratio"], 9.0)
        self.assertAlmostEqual(d["percent"]["b"], 10.0)

    def test_bh_known(self):
        adj = stats.bh_adjust([0.01, 0.04, 0.03, 0.005])
        np.testing.assert_allclose(adj, [0.02, 0.04, 0.04, 0.02])

    def test_cohens_d_known(self):
        a = [1, 2, 3, 4, 5]
        b = [3, 4, 5, 6, 7]
        r = stats.compare_numeric(a, b)
        self.assertEqual(r["status"], "ok")
        self.assertAlmostEqual(r["cohens_d"], -2 / math.sqrt(2.5))
        self.assertLess(r["mean_diff_ci"][0], r["mean_diff"])
        self.assertGreater(r["mean_diff_ci"][1], r["mean_diff"])
        self.assertIn("interpretation", r)

    def test_insufficient_data_gives_no_interpretation(self):
        r = stats.compare_numeric([1], [1, 2, 3])
        self.assertEqual(r["status"], "insufficient_data")
        self.assertNotIn("interpretation", r)
        r2 = stats.compare_numeric([5, 5, 5], [5, 5, 5])
        self.assertEqual(r2["status"], "insufficient_data")
        self.assertNotIn("interpretation", r2)

    def test_chi_square(self):
        x = ["a"] * 50 + ["b"] * 50
        y = ["u"] * 40 + ["v"] * 10 + ["u"] * 10 + ["v"] * 40
        r = stats.chi_square_association(x, y)
        self.assertEqual(r["status"], "ok")
        self.assertAlmostEqual(r["chi2"], 36.0)
        self.assertAlmostEqual(r["cramers_v"], 0.6)
        self.assertEqual(stats.chi_square_association(["a", "a"], ["u", "v"])["status"], "insufficient_data")

    def test_chi_square_small_counts_warns(self):
        r = stats.chi_square_association(["a", "a", "b", "b"], ["u", "v", "u", "u"])
        self.assertTrue(r["warnings"])
        self.assertIn("fisher_exact_p", r)

    def test_correlation(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [2, 4, 6, 8], "c": [4, 3, 2, 1]})
        c = stats.correlation_matrix(df, ["a", "b", "c"], "spearman")
        self.assertAlmostEqual(c.loc["a", "b"], 1.0)
        self.assertAlmostEqual(c.loc["a", "c"], -1.0)
        with self.assertRaises(ValueError):
            stats.correlation_matrix(df, ["a"], "kendall-ish")

    def test_bootstrap_ci_reproducible_and_brackets(self):
        y = np.array([1] * 50 + [0] * 50)
        p = np.array([1] * 40 + [0] * 10 + [0] * 50)
        f = lambda a, b: float(((a == 1) & (b == 1)).sum() / (a == 1).sum()) if (a == 1).any() else float("nan")
        lo, hi = stats.bootstrap_ci(y, p, f, n_boot=300, seed=1)
        self.assertEqual((lo, hi), stats.bootstrap_ci(y, p, f, n_boot=300, seed=1))
        self.assertLess(lo, 0.8)
        self.assertGreater(hi, 0.8)


if __name__ == "__main__":
    unittest.main()
