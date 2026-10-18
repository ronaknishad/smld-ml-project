# NetSentinel (core engine) - SIH26145

Statistical + ML core for detecting potentially malicious network flows.
Status: ML/statistics core, CIC-IDS2017 loader, SQLite-backed FastAPI service, browser UI (single HTML file), alerts and exports.
Not implemented: live packet capture, authentication, PDF reports, multiclass models, anomaly detection.
`run_demo.py` uses **SYNTHETIC data** (not a benchmark). Real results come only from `run_cic.py` on real files.

## Run (Windows PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m unittest discover -s tests -t . -v    # engine tests
python run_demo.py                               # synthetic demo
```

## Run the application
```powershell
pip install -r requirements.txt
python -m unittest discover -s backend\tests -t . -v      # service + API tests
python -m uvicorn backend.app.main:app --port 8000
```
Open http://127.0.0.1:8000 (UI) or http://127.0.0.1:8000/docs (interactive API docs).
Put CIC-IDS2017 CSVs in `data\` first. Suggested flow: Dataset Lab -> Statistical Analysis -> Model Laboratory (train, then "Make active")
-> Traffic Explorer (run predictions) -> Detection Alerts -> Reports. Stop the server with Ctrl+C; data persists in `storage\` (gitignored).
Optional UI render check (needs Node.js): `python scripts\check_frontend.py`.

Alert policy: an alert is created for each record predicted malicious, highest probability first, up to a cap; priority is "high" when probability >= 0.9, else "medium".
Probabilities are uncalibrated model outputs, so priority is not a risk score and an alert is not a confirmed incident.

## Real data: CIC-IDS2017 (forward-direction compatibility mode)
Put the MachineLearningCSV day files in a `data\` folder (not committed), then:
```powershell
# quick trial first (reads only the first 100k rows of each file - say so in your report)
python run_cic.py --files data\Tuesday-WorkingHours.pcap_ISCX.csv --nrows 100000
# pooled attack days, random stratified split (optimistic)
python run_cic.py --files data\Tuesday-WorkingHours.pcap_ISCX.csv data\Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
# cross-day (stricter): train on --files, test on a different day
python run_cic.py --files data\Tuesday-WorkingHours.pcap_ISCX.csv data\Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv --test-file data\Wednesday-workingHours.pcap_ISCX.csv
```
Ablation (measure reliance on specific features): add `--drop-features init_win_bytes_fwd min_seg_size_fwd`. Constant (zero-variance) features are removed automatically and listed.
Monday is benign-only: it cannot train a classifier alone, and as a test file it gives undefined recall/AUC (shown as NaN).

### Dataset limitations (state these in the report)
- Released CSVs are **bidirectional** flow records with no IP or protocol columns. This is a compatibility mode, not a strict unidirectional experiment.
- Only forward-direction fields are used; nothing is halved or reconstructed. Labels still come from whole bidirectional flows.
- `Flow Duration` and `Fwd Packets/s` mix both directions, so they are NOT used; duration = `Fwd IAT Total`, rates are recomputed. Times converted from microseconds to seconds.
- `Destination Port` is excluded from features (shortcut risk). `Init_Win_bytes_forward == -1` is treated as missing. Rows with other negative values and exact duplicates are dropped and counted in the cleaning report.
- Cite: Sharafaldin, Lashkari, Ghorbani, "Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization", ICISSP 2018. Record the download page and date you used.

## Modules
- `ml/directional.py` - packets -> directional flows (reverse traffic never merged; idle timeout).
- `ml/cic2017.py` - CIC-IDS2017 forward-only loader/cleaner.
- `ml/stats.py` - descriptives, class balance, correlation, Welch + Mann-Whitney with effect sizes/CI, chi-square + Cramer's V, Benjamini-Hochberg, bootstrap CI.
- `ml/pipeline.py` - validation, leakage-safe features, Dummy/LogReg/RF, threshold chosen on validation, random or cross-split test, permutation importance, save/load/predict.
- `backend/app/services.py` - SQLite storage, dataset registry, statistics, background training, predictions, alerts, exports.
- `backend/app/main.py` - FastAPI routes (thin layer over services). `frontend/index.html` - UI.
- `ml/synthetic.py` - synthetic fixture (testing only).

## Known limitations
- Bootstrap CIs and the random split assume independent records; flows from the same attack burst are related, so the random split is optimistic. Prefer the cross-day result for honest claims.
- Cross-day tests on unseen attack families are expected to score lower; that is a finding, not a bug.
- No live capture; offline CSV analysis only.


## Dataset

The datasets required to run this project are available on Google Drive.

**[Download the datasets here](https://drive.google.com/drive/folders/1vMKaty1vAChObLwb-veUAGXZ2kFGoXoM)**

Download the dataset files and place them inside the `data/` directory in the project root before running the application.

Note: The datasets are hosted externally because of their size.
