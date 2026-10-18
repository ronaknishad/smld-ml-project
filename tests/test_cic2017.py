import math
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from ml import cic2017 as C
from ml import pipeline as P

HEADER = [" Destination Port", " Flow Duration", " Total Fwd Packets", "Total Length of Fwd Packets",
          " Fwd Packet Length Max", " Fwd Packet Length Min", " Fwd Packet Length Mean", " Fwd Packet Length Std",
          "Fwd IAT Total", " Fwd IAT Mean", " Fwd IAT Std", " Fwd IAT Max", " Fwd Header Length",
          " Fwd Header Length", "Fwd PSH Flags", " Fwd URG Flags", "Init_Win_bytes_forward",
          " act_data_pkt_fwd", " min_seg_size_forward", " Label"]


def row(port, fdur, fpk, flen, mx, mn, mean, std, iat_t, iat_m, iat_s, iat_x, hdr, init, act, seg, label,
        psh=0, urg=0):
    return [port, fdur, fpk, flen, mx, mn, mean, std, iat_t, iat_m, iat_s, iat_x, hdr, hdr, psh, urg,
            init, act, seg, label]


# First three rows are copied from the user's Monday-WorkingHours sample (real dataset values).
# The remaining rows are FIXTURE rows invented only to exercise parsing edge cases.
ROWS = [
    row(49188, 4, 2, 12, 6, 6, 6, 0, 4, 4, 0, 4, 40, 329, 1, 20, "BENIGN"),
    row(88, 609, 7, 484, 233, 0, 69.14285714, 111.9678951, 609, 101.5, 177.0895254, 460, 164, 8192, 5, 20, "BENIGN"),
    row(1034, 6, 1, 6, 6, 6, 6, 0, 0, 0, 0, 0, 20, 2053, 0, 20, "BENIGN"),
    row(80, 5000, 10, 4000, 1200, 0, 400, 300, 4000, 400, 100, 900, 200, -1, 8, 20, "DoS Hulk"),
    row(80, 100, 3, 90, 40, 20, 30, 5, -50, 10, 1, 20, 60, 100, 1, 20, "BENIGN"),  # negative -> dropped
    row(80, 700, 4, 400, 150, 50, 100, 20, 600, 150, 10, 200, 80, 500, 2, 20, "Web Attack \x96 Brute Force"),
]


def write_csv(path, rows, header=HEADER):
    pd.DataFrame(rows, columns=header).to_csv(path, index=False, encoding="latin-1")


class TestCicLoader(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "fixture.csv")
        write_csv(self.path, ROWS)

    def tearDown(self):
        self.tmp.cleanup()

    def load(self, **kw):
        return C.to_forward_features(C.load_cic_csv(self.path), "fixture.csv", **kw)

    def test_columns_stripped_and_duplicate_header_dropped(self):
        raw = C.load_cic_csv(self.path)
        self.assertIn("Total Fwd Packets", raw.columns)
        self.assertEqual(list(raw.columns).count("Fwd Header Length"), 1)
        self.assertNotIn("Fwd Header Length.1", raw.columns)

    def test_missing_required_column_raises(self):
        p = os.path.join(self.tmp.name, "bad.csv")
        pd.DataFrame({"a": [1]}).to_csv(p, index=False)
        with self.assertRaises(ValueError):
            C.load_cic_csv(p)

    def test_units_and_recomputed_rates_forward_only(self):
        df, _ = self.load()
        r0, r1, r2 = df.iloc[0], df.iloc[1], df.iloc[2]
        self.assertAlmostEqual(r0.duration, 4e-6)                  # 4 microseconds -> seconds
        self.assertAlmostEqual(r0.pkts_per_sec, 2 / 4e-6)           # matches CIC Fwd Packets/s = 500000
        self.assertAlmostEqual(r1.pkts_per_sec, 7 / 609e-6, places=3)  # CIC reports 11494.25
        self.assertAlmostEqual(r1.mean_iat, 101.5e-6)
        # zero forward duration: rate undefined (NaN), NOT borrowed from whole-flow duration
        self.assertEqual(r2.duration, 0.0)
        self.assertTrue(math.isnan(r2.pkts_per_sec))

    def test_negative_rows_dropped_and_counted(self):
        df, rep = self.load()
        self.assertEqual(rep["n_dropped_negative_values"], 1)
        self.assertEqual(len(df), 5)
        self.assertEqual(rep["n_raw_rows"], 6)

    def test_init_win_sentinel_becomes_nan_not_negative(self):
        df, rep = self.load()
        self.assertEqual(rep["n_init_win_sentinel_to_nan"], 1)
        self.assertTrue(math.isnan(df[df.attack_category == "DoS Hulk"].iloc[0].init_win_bytes_fwd))
        self.assertTrue((df.init_win_bytes_fwd.dropna() >= 0).all())

    def test_label_mapping_and_encoding_variant(self):
        df, _ = self.load()
        self.assertEqual(set(df.label), {"benign", "malicious"})
        web = df[df.attack_category.str.startswith("Web Attack")]
        self.assertEqual(web.iloc[0].attack_category, "Web Attack - Brute Force")
        self.assertEqual(web.iloc[0].label, "malicious")

    def test_unknown_label_raises(self):
        rows = ROWS[:3] + [row(80, 5, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 20, 1, 0, 20, "TotallyNewAttack")]
        write_csv(self.path, rows)
        with self.assertRaises(ValueError):
            self.load()

    def test_duplicates_reported_and_dropped(self):
        write_csv(self.path, ROWS[:3] + [ROWS[0]])
        df, rep = self.load()
        self.assertEqual(rep["n_duplicates_dropped"], 1)
        self.assertEqual(len(df), 3)
        df2, rep2 = self.load(drop_duplicates=False)
        self.assertEqual(len(df2), 4)

    def test_feature_set_excludes_ports_ids_label_and_backward_fields(self):
        feats = C.CIC_FEATURES["numeric"] + C.CIC_FEATURES["categorical"]
        for bad in ["label", "attack_category", "row_id", "source_file", "destination_port"]:
            self.assertNotIn(bad, feats)
        for f in feats:
            self.assertNotIn("bwd", f)
        for raw_name in C.FORWARD_MAP:
            self.assertNotIn("Bwd", raw_name)
            self.assertNotIn("Flow", raw_name)


def fake_cic_rows(n, seed, attack_label="DoS Hulk", mal_frac=0.3):
    """Larger FIXTURE in CIC format for code-path testing only. NOT real data."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        mal = rng.random() < mal_frac
        pk = int(max(1, rng.lognormal(2.6 if mal else 1.6, 0.7)))
        mean = float(np.clip(rng.normal(150 if mal else 300, 90), 6, 1400))
        dur = float(rng.lognormal(9 if mal else 8, 1.2))
        rows.append(row(80, dur, pk, pk * mean, mean * 2, 0, mean, mean * 0.4, dur, dur / max(pk, 1),
                        dur / 4, dur / 2, 20 * pk, int(rng.choice([8192, 229, 29200])), pk, 20,
                        attack_label if mal else "BENIGN"))
    return rows


class TestCicEndToEnd(unittest.TestCase):
    """Code-path test on a generated CIC-format FIXTURE (not real data, no benchmark meaning)."""

    def test_load_train_cross_split_predict(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = os.path.join(d, "dayA.csv"), os.path.join(d, "dayB.csv")
            write_csv(a, fake_cic_rows(1200, 1))
            write_csv(b, fake_cic_rows(600, 2, attack_label="PortScan"))
            train, reps = C.load_forward_dataset([a])
            test, _ = C.load_forward_dataset([b])
            self.assertEqual(reps[0]["source_file"], "dayA.csv")
            res = P.run_cross_split(train, test, "cic-fixture", seed=3, n_boot=50, **C.CIC_FEATURES)
            self.assertEqual(res["split"]["strategy"], "cross_split")
            for r in res["results"]:
                self.assertEqual(sum(r["test"]["confusion_matrix"].values()), len(test))
            rnd = P.run_experiment(train, "cic-fixture", seed=3, n_boot=50, **C.CIC_FEATURES)
            self.assertEqual(rnd["split"]["strategy"], "random_stratified")
            sel = next(r for r in res["results"] if r["metadata"]["model_id"] == res["selected_model"])
            out = P.predict(test.head(10), {"pipeline": sel["pipeline"], "metadata": sel["metadata"]})
            self.assertEqual(len(out), 10)

    def test_benign_only_test_day_does_not_crash(self):
        with tempfile.TemporaryDirectory() as d:
            a, m = os.path.join(d, "att.csv"), os.path.join(d, "mon.csv")
            write_csv(a, fake_cic_rows(800, 5))
            write_csv(m, fake_cic_rows(300, 6, mal_frac=0.0))
            train, _ = C.load_forward_dataset([a])
            mon, _ = C.load_forward_dataset([m])
            res = P.run_cross_split(train, mon, "cic-fixture", seed=1, n_boot=20, **C.CIC_FEATURES)
            for r in res["results"]:
                self.assertTrue(math.isnan(r["test"]["roc_auc"]))
                self.assertTrue(math.isnan(r["test"]["recall"]))


if __name__ == "__main__":
    unittest.main()
