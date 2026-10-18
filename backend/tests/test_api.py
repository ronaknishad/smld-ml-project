"""HTTP-level tests. Need fastapi + httpx (pip install -r requirements.txt); skipped otherwise."""
import os
import shutil
import tempfile
import unittest

try:
    from fastapi.testclient import TestClient
    HAVE = True
except Exception:  # fastapi/httpx not installed
    HAVE = False

from tests.test_cic2017 import fake_cic_rows, write_csv


@unittest.skipUnless(HAVE, "fastapi/httpx not installed")
class TestApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls._old = os.environ.get("NETSENTINEL_HOME")
        os.environ["NETSENTINEL_HOME"] = cls.tmp
        os.makedirs(os.path.join(cls.tmp, "data"))
        write_csv(os.path.join(cls.tmp, "data", "dayA.csv"), fake_cic_rows(1200, 1))
        from backend.app.main import app
        cls.c = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        if cls._old is None:
            os.environ.pop("NETSENTINEL_HOME", None)
        else:
            os.environ["NETSENTINEL_HOME"] = cls._old
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_01_health_and_index(self):
        self.assertEqual(self.c.get("/api/health").json()["status"], "ok")
        r = self.c.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn("NetSentinel", r.text)
        self.assertEqual(self.c.get("/api/system/info").json()["live_packet_capture"].split()[0], "not")

    def test_02_validation_and_missing_resources(self):
        self.assertEqual(self.c.get("/api/datasets/ds_nope").status_code, 404)
        self.assertEqual(self.c.post("/api/datasets", json={"name": "x", "files": ["../x.csv"]}).status_code, 422)
        self.assertEqual(self.c.post("/api/datasets", json={"name": ""}).status_code, 422)
        self.assertEqual(self.c.post("/api/analyses", json={"dataset_id": "ds_nope"}).status_code, 409)
        self.assertEqual(self.c.patch("/api/alerts/999999", json={"status": "new"}).status_code, 404)
        self.assertEqual(self.c.patch("/api/alerts/1", json={"status": "bogus"}).status_code, 422)
        self.assertEqual(self.c.get("/api/alerts?status=bogus").status_code, 422)

    def test_03_end_to_end_workflow(self):
        files = self.c.get("/api/datasets/files").json()["files"]
        self.assertEqual(files[0]["name"], "dayA.csv")
        ds = self.c.post("/api/datasets", json={"name": "A", "files": ["dayA.csv"]})
        self.assertEqual(ds.status_code, 201)
        did = ds.json()["id"]
        self.assertGreater(len(self.c.get(f"/api/datasets/{did}/preview?limit=5").json()["rows"]), 0)
        st = self.c.get(f"/api/statistics/{did}")
        self.assertEqual(st.status_code, 200)
        self.assertTrue(st.json()["hypothesis_tests"]["rows"])
        self.assertEqual(self.c.get(f"/api/statistics/{did}/distribution?feature=packet_count&log=true").status_code, 200)
        csv_r = self.c.get(f"/api/exports/statistics/{did}?part=tests")
        self.assertIn("text/csv", csv_r.headers["content-type"])

        tr = self.c.post("/api/models/train", json={"dataset_id": did, "seed": 3})
        self.assertEqual(tr.status_code, 202)
        rid = tr.json()["id"]
        import time
        for _ in range(120):  # background thread; poll
            run = self.c.get(f"/api/models/runs/{rid}").json()
            if run["status"] in ("completed", "failed"):
                break
            time.sleep(0.5)
        self.assertEqual(run["status"], "completed", run.get("error"))
        self.assertEqual(self.c.get(f"/api/exports/evaluation/{rid}").status_code, 200)

        an = self.c.post("/api/analyses", json={"dataset_id": did, "max_alerts": 20})
        self.assertEqual(an.status_code, 201)
        aid = an.json()["id"]
        recs = self.c.get(f"/api/analyses/{aid}/records?predicted=malicious&page_size=5").json()
        self.assertTrue(recs["rows"])
        self.assertEqual(self.c.get(f"/api/analyses/{aid}/records/{recs['rows'][0]['row_idx']}").status_code, 200)
        al = self.c.get(f"/api/alerts?analysis_id={aid}").json()
        self.assertGreater(al["total"], 0)
        alert_id = al["rows"][0]["id"]
        p = self.c.patch(f"/api/alerts/{alert_id}", json={"status": "under_review", "note": "looking"})
        self.assertEqual(p.json()["status"], "under_review")
        self.assertIn("alert_id", self.c.get("/api/exports/alerts").text.splitlines()[0])
        self.assertEqual(self.c.get("/api/overview").json()["active_model"]["algorithm"] in ("logreg", "rf"), True)


if __name__ == "__main__":
    unittest.main()
