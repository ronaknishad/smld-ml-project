import csv
import io
import json
import logging
import os
import shutil
import tempfile
import unittest
from unittest import mock

from backend.app import services as svc
from tests.test_cic2017 import fake_cic_rows, write_csv


class ServiceBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logging.getLogger("netsentinel").setLevel(logging.CRITICAL)  # expected-failure tests log tracebacks
        cls.tmp = tempfile.mkdtemp()
        cls._old = os.environ.get("NETSENTINEL_HOME")
        os.environ["NETSENTINEL_HOME"] = cls.tmp
        os.makedirs(os.path.join(cls.tmp, "data"))
        d = os.path.join(cls.tmp, "data")
        write_csv(os.path.join(d, "dayA.csv"), fake_cic_rows(1500, 1))
        write_csv(os.path.join(d, "dayB.csv"), fake_cic_rows(800, 2, attack_label="PortScan"))
        write_csv(os.path.join(d, "benign_only.csv"), fake_cic_rows(300, 3, mal_frac=0.0))

    @classmethod
    def tearDownClass(cls):
        if cls._old is None:
            os.environ.pop("NETSENTINEL_HOME", None)
        else:
            os.environ["NETSENTINEL_HOME"] = cls._old
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def assertErr(self, code, fn, *a, **k):
        with self.assertRaises(svc.ServiceError) as cm:
            fn(*a, **k)
        self.assertEqual(cm.exception.status, code, cm.exception.message)


class TestNoModelYet(ServiceBase):
    def test_prediction_without_active_model_is_409(self):
        a = svc.register_dataset("A", files=["dayA.csv"])
        self.assertErr(409, svc.run_predictions, a["id"])

    def test_empty_overview_and_system(self):
        self.assertIsNone(svc.overview()["latest_analysis"])
        self.assertIn("not implemented", svc.system_info()["live_packet_capture"])


class TestDatasets(ServiceBase):
    def test_file_listing_and_path_safety(self):
        names = [f["name"] for f in svc.list_data_files()["files"]]
        self.assertIn("dayA.csv", names)
        for bad in ["../x.csv", "a/b.csv", "..\\x.csv", "notes.txt", ""]:
            self.assertErr(422, svc.register_dataset, "x", files=[bad])
        self.assertErr(404, svc.register_dataset, "x", files=["missing.csv"])
        self.assertErr(422, svc.register_dataset, "x", files=[])
        self.assertErr(422, svc.register_dataset, "", files=["dayA.csv"])

    def test_register_cic_states_limitation_and_counts(self):
        d = svc.register_dataset("A", files=["dayA.csv"])
        self.assertIn("NOT strict unidirectional", d["directionality"])
        self.assertEqual(sum(d["class_counts"]["counts"].values()), d["n_rows"])
        self.assertEqual(d["cleaning_report"]["files"][0]["source_file"], "dayA.csv")
        self.assertNotIn("path", d)

    def test_synthetic_is_flagged(self):
        d = svc.register_dataset("syn", kind="synthetic", n=600)
        self.assertTrue(d["cleaning_report"]["synthetic"])
        self.assertIn("SYNTHETIC", d["directionality"])
        self.assertErr(422, svc.register_dataset, "s", kind="synthetic", n=10)
        self.assertErr(422, svc.register_dataset, "s", kind="bogus")

    def test_preview_bounds(self):
        d = svc.register_dataset("A", files=["dayA.csv"])
        p = svc.dataset_preview(d["id"], limit=5000, offset=-3)
        self.assertEqual((p["limit"], p["offset"], len(p["rows"])), (200, 0, 200))
        self.assertErr(404, svc.dataset_preview, "ds_nope")

    def test_nrows_limit_recorded(self):
        d = svc.register_dataset("A", files=["dayA.csv"], nrows=200)
        self.assertLessEqual(d["n_rows"], 200)
        self.assertEqual(d["cleaning_report"]["nrows_limit_per_file"], 200)

    def test_single_class_file_rejected_for_training(self):
        d = svc.register_dataset("mon", files=["benign_only.csv"])
        self.assertErr(422, svc.create_training_run, {"dataset_id": d["id"]})


class TestStatistics(ServiceBase):
    def test_statistics_content_and_json_safe(self):
        d = svc.register_dataset("A", files=["dayA.csv"])
        st = svc.dataset_statistics(d["id"])
        json.dumps(st)  # must be JSON serialisable (no NaN objects)
        self.assertEqual(st["n_records"], d["n_rows"])
        rows = st["hypothesis_tests"]["rows"]
        self.assertTrue(rows)
        for r in rows:
            if r["status"] == "ok":
                self.assertGreaterEqual(r["mannwhitney_p_bh"], r["mannwhitney_p"] - 1e-15)
        m = st["correlation"]["matrix"]
        self.assertEqual(len(m), len(st["correlation"]["features"]))
        self.assertTrue(all(len(r) == len(m) for r in m))
        self.assertIn("min_seg_size_fwd", st["constant_features_excluded_from_tests"])  # constant in fixture

    def test_chi_square_for_synthetic_protocol(self):
        d = svc.register_dataset("syn", kind="synthetic", n=2000)
        st = svc.dataset_statistics(d["id"])
        self.assertEqual(st["chi_square"][0]["variable"], "protocol")
        self.assertEqual(st["chi_square"][0]["status"], "ok")

    def test_distribution(self):
        d = svc.register_dataset("A", files=["dayA.csv"])
        r = svc.feature_distribution(d["id"], "packet_count", bins=20, log=True)
        for g in r["groups"].values():
            self.assertAlmostEqual(sum(g["density"]), 1.0, places=9)
        self.assertEqual(len(r["edges"]), 21)
        self.assertErr(422, svc.feature_distribution, d["id"], "label")
        self.assertErr(422, svc.feature_distribution, d["id"], "nope")

    def test_statistics_csv_export(self):
        d = svc.register_dataset("A", files=["dayA.csv"])
        rows = list(csv.reader(io.StringIO(svc.statistics_csv(d["id"], "tests"))))
        self.assertIn("feature", rows[0])
        self.assertGreater(len(rows), 1)
        self.assertErr(422, svc.statistics_csv, d["id"], "bogus")


class TestWorkflow(ServiceBase):
    """Train once, then exercise predictions, records, alerts, exports."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.a = svc.register_dataset("A", files=["dayA.csv"])
        cls.b = svc.register_dataset("B", files=["dayB.csv"])
        cls.train_run = svc.start_training({"dataset_id": cls.a["id"], "seed": 5}, background=False)

    def test_run_completed_and_selected_model_active(self):
        r = self.train_run
        self.assertEqual(r["status"], "completed", r.get("error"))
        ev = r["evaluation"]
        self.assertEqual([m["algorithm"] for m in ev["models"]], ["dummy", "logreg", "rf"])
        self.assertNotIn("dummy", ev["selected_model"])
        self.assertEqual(r["active_model_id"], ev["selected_model"])
        self.assertIn("min_seg_size_fwd", ev["features"]["constant_removed"])
        self.assertTrue(ev["limitations"])
        json.dumps(r)

    def test_training_validation_errors(self):
        self.assertErr(422, svc.create_training_run, {"dataset_id": self.a["id"], "drop_features": ["nope"]})
        self.assertErr(422, svc.create_training_run, {"dataset_id": self.a["id"], "test_dataset_id": self.a["id"]})
        self.assertErr(422, svc.create_training_run, {"dataset_id": self.a["id"], "models": ["svm"]})
        self.assertErr(422, svc.create_training_run, {"dataset_id": self.a["id"], "max_fpr": 0.9})
        self.assertErr(404, svc.create_training_run, {"dataset_id": "ds_none"})
        syn = svc.register_dataset("syn", kind="synthetic", n=500)
        self.assertErr(422, svc.create_training_run, {"dataset_id": self.a["id"], "test_dataset_id": syn["id"]})

    def test_failed_run_is_recorded_not_raised(self):
        with mock.patch("ml.pipeline.run_experiment", side_effect=RuntimeError("boom")):
            r = svc.start_training({"dataset_id": self.a["id"]}, background=False)
        self.assertEqual(r["status"], "failed")
        self.assertIn("boom", r["error"])

    def test_cross_split_run_with_ablation(self):
        r = svc.start_training({"dataset_id": self.a["id"], "test_dataset_id": self.b["id"], "models": ["dummy", "logreg"],
                                "drop_features": ["init_win_bytes_fwd"]}, background=False)
        self.assertEqual(r["status"], "completed", r.get("error"))
        self.assertEqual(r["evaluation"]["split"]["strategy"], "cross_split")
        self.assertIn("init_win_bytes_fwd", r["evaluation"]["features"]["dropped_by_request"])
        self.assertNotIn("init_win_bytes_fwd", r["evaluation"]["features"]["numeric"])

    def test_promotion_rules(self):
        models = svc.list_models()
        dummy = next(m for m in models if m["algorithm"] == "dummy")
        self.assertErr(409, svc.promote_model, dummy["id"])
        self.assertErr(404, svc.promote_model, "nope")

    def test_predictions_in_sample_flag_and_alerts(self):
        an = svc.run_predictions(self.a["id"], max_alerts=50)
        self.assertTrue(an["params"]["in_sample"])
        self.assertIsNotNone(an["params"]["warning"])
        self.assertEqual(an["n_records"], self.a["n_rows"])
        self.assertLessEqual(an["params"]["n_alerts"], 50)
        al = svc.list_alerts(analysis_id=an["id"], page_size=200)
        self.assertEqual(al["total"], an["params"]["n_alerts"])
        self.assertTrue(all(r["priority"] in ("high", "medium") and r["status"] == "new" for r in al["rows"]))
        probs = [r["probability"] for r in al["rows"]]
        self.assertEqual(probs, sorted(probs, reverse=True))
        self.assertEqual(an["params"]["n_alerts"] + an["params"]["n_alerts_suppressed"], an["n_malicious"])

    def test_out_of_sample_and_zero_alerts(self):
        an = svc.run_predictions(self.b["id"], max_alerts=0)
        self.assertFalse(an["params"]["in_sample"])
        self.assertIsNone(an["params"]["warning"])
        self.assertEqual(svc.list_alerts(analysis_id=an["id"])["total"], 0)

    def test_incompatible_dataset_rejected(self):
        syn = svc.register_dataset("syn", kind="synthetic", n=500)
        self.assertErr(422, svc.run_predictions, syn["id"])

    def test_records_filters_pagination_detail(self):
        an = svc.run_predictions(self.b["id"], max_alerts=5)
        allr = svc.analysis_records(an["id"], page_size=10)
        self.assertEqual(allr["total"], an["n_records"])
        self.assertEqual(len(allr["rows"]), 10)
        mal = svc.analysis_records(an["id"], predicted="malicious", page_size=200)
        self.assertEqual(mal["total"], an["n_malicious"])
        self.assertTrue(all(r["predicted"] == "malicious" for r in mal["rows"]))
        hi = svc.analysis_records(an["id"], min_prob=0.95)
        self.assertTrue(all(r["probability"] >= 0.95 for r in hi["rows"]))
        self.assertTrue(allr["labels_available"])
        self.assertErr(422, svc.analysis_records, an["id"], sort="weird")
        det = svc.record_detail(an["id"], allr["rows"][0]["row_idx"])
        self.assertIn("not a local explanation", det["explanation_note"])
        self.assertEqual(det["provenance"]["source_file"], "dayB.csv")
        self.assertIn("uncalibrated", det["prediction"]["note"])
        self.assertErr(404, svc.record_detail, an["id"], 10 ** 9)
        self.assertErr(404, svc.analysis_records, "an_nope")

    def test_alert_updates_events_and_validation(self):
        an = svc.run_predictions(self.b["id"], max_alerts=3)
        aid = svc.list_alerts(analysis_id=an["id"])["rows"][0]["id"]
        u = svc.update_alert(aid, status="under_review", note="checking")
        self.assertEqual((u["status"], u["note"]), ("under_review", "checking"))
        u = svc.update_alert(aid, status="resolved")
        self.assertEqual(u["note"], "checking")  # untouched when note omitted
        self.assertEqual([e["to_status"] for e in u["events"]], ["under_review", "resolved"])
        self.assertFalse(u["verified_incident"])
        self.assertErr(422, svc.update_alert, aid, status="confirmed_attack")
        self.assertErr(422, svc.update_alert, aid)
        self.assertErr(422, svc.update_alert, aid, note="x" * 1001)
        self.assertErr(404, svc.update_alert, 10 ** 9, status="new")
        self.assertEqual(svc.list_alerts(status="resolved", analysis_id=an["id"])["total"], 1)
        self.assertErr(422, svc.list_alerts, status="bogus")

    def test_csv_export_escapes_formulas(self):
        an = svc.run_predictions(self.b["id"], max_alerts=2)
        aid = svc.list_alerts(analysis_id=an["id"])["rows"][0]["id"]
        svc.update_alert(aid, note="=HYPERLINK(\"http://evil\")")
        rows = list(csv.DictReader(io.StringIO(svc.alerts_csv(analysis_id=an["id"]))))
        self.assertEqual(len(rows), 2)
        self.assertIn("alert_id", rows[0])
        self.assertNotIn("id", rows[0])
        notes = [r["note"] for r in rows]
        self.assertTrue(any(n.startswith("'=") for n in notes))
        self.assertFalse(any(n.startswith("=") for n in notes))

    def test_evaluation_export(self):
        ex = svc.evaluation_export(self.train_run["id"])
        self.assertEqual(ex["evaluation"]["selected_model"], self.train_run["evaluation"]["selected_model"])
        self.assertIn("directionality", ex["dataset"])
        with mock.patch("ml.pipeline.run_experiment", side_effect=RuntimeError("x")):
            bad = svc.start_training({"dataset_id": self.a["id"]}, background=False)
        self.assertErr(409, svc.evaluation_export, bad["id"])

    def test_persistence_survives_cache_clear(self):
        an = svc.run_predictions(self.b["id"], max_alerts=1)
        svc._df_cache.clear()
        svc._stats_cache.clear()
        self.assertEqual(svc.get_analysis(an["id"])["n_records"], an["n_records"])
        self.assertTrue(svc.analysis_records(an["id"])["rows"])
        self.assertTrue(svc.list_runs())
        self.assertEqual(svc.overview()["active_model"]["id"], self.train_run["evaluation"]["selected_model"])
        self.assertGreaterEqual(svc.system_info()["counts"]["alerts"], 1)

    def test_missing_model_file_is_410(self):
        with svc.db() as c:
            row = c.execute("SELECT id, path FROM models WHERE algorithm='rf' LIMIT 1").fetchone()
        os.rename(row["path"], row["path"] + ".bak")
        try:
            self.assertErr(410, svc.run_predictions, self.b["id"], model_id=row["id"])
        finally:
            os.rename(row["path"] + ".bak", row["path"])


if __name__ == "__main__":
    unittest.main()
