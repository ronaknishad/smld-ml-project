import json, os, sys, tempfile, logging
OUT = sys.argv[1]
sys.path.insert(0, os.getcwd())
tmp = tempfile.mkdtemp(); os.environ["NETSENTINEL_HOME"] = tmp; os.makedirs(tmp + "/data")
logging.getLogger("netsentinel").setLevel(logging.CRITICAL)
from tests.test_cic2017 import fake_cic_rows, write_csv
write_csv(tmp + "/data/dayA.csv", fake_cic_rows(1500, 1)); write_csv(tmp + "/data/dayB.csv", fake_cic_rows(800, 2, attack_label="PortScan"))
from backend.app import services as s
evil = "<img src=x onerror=alert(1)>"
a = s.register_dataset(evil, files=["dayA.csv"], nrows=1000)
b = s.register_dataset("B", files=["dayB.csv"])
syn = s.register_dataset("syn", kind="synthetic", n=1000)
run = s.start_training({"dataset_id": a["id"], "test_dataset_id": b["id"], "drop_features": ["init_win_bytes_fwd"]}, background=False)
run2 = s.start_training({"dataset_id": syn["id"]}, background=False)
an = s.run_predictions(b["id"], max_alerts=30)
al = s.list_alerts(analysis_id=an["id"], page_size=10)
s.update_alert(al["rows"][0]["id"], status="under_review", note=evil)
al = s.list_alerts(analysis_id=an["id"], page_size=10)
recs = s.analysis_records(an["id"], page_size=10)
P = {
 "overview": s.overview(), "files": s.list_data_files(), "datasets": s.list_datasets(), "dsA": a, "prevA": s.dataset_preview(a["id"], 5),
 "syn": s.get_dataset(syn["id"]), "prevSyn": s.dataset_preview(syn["id"], 5),
 "stats": s.dataset_statistics(a["id"]), "statsSyn": s.dataset_statistics(syn["id"]), "dist": s.feature_distribution(a["id"], "packet_count", 30, True),
 "runs": s.list_runs(), "run": run, "run2": run2, "models": s.list_models(), "analyses": s.list_analyses(), "analysis": s.get_analysis(an["id"]),
 "records": recs, "detail": s.record_detail(an["id"], recs["rows"][0]["row_idx"]), "alerts": al, "system": s.system_info(), "evil": evil,
}
json.dump(P, open(OUT, "w"))
print("payloads written;", len(recs["rows"]), "records,", al["total"], "alerts")
