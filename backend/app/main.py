"""NetSentinel FastAPI app: thin HTTP layer over backend.app.services. Run from the project root:
    python -m uvicorn backend.app.main:app --port 8000
Interactive API docs: http://127.0.0.1:8000/docs"""
from pathlib import Path
from typing import List, Literal, Optional

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field

from backend.app import services as svc

app = FastAPI(
    title="NetSentinel API", version=svc.VERSION,
    description="Statistical analysis and ML detection of potentially malicious network flows (offline CSV analysis). "
                "A model prediction is a lead for review, not a confirmed attack.")
FRONTEND = Path(__file__).resolve().parents[2] / "frontend" / "index.html"


@app.exception_handler(svc.ServiceError)
async def service_error_handler(request, exc: svc.ServiceError):
    return JSONResponse(status_code=exc.status, content={"detail": exc.message})


class DatasetCreate(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    kind: Literal["cic2017_forward", "synthetic"] = "cic2017_forward"
    files: List[str] = []
    nrows: Optional[int] = Field(default=None, ge=1)
    n: int = Field(default=5000, ge=200, le=200000)
    seed: int = 0
    drop_duplicates: bool = True


class TrainRequest(BaseModel):
    dataset_id: str
    test_dataset_id: Optional[str] = None
    models: List[Literal["dummy", "logreg", "rf"]] = ["dummy", "logreg", "rf"]
    drop_features: List[str] = []
    seed: int = 42
    max_fpr: float = Field(default=0.05, gt=0, le=0.5)


class PredictRequest(BaseModel):
    dataset_id: str
    model_id: Optional[str] = None
    limit: Optional[int] = Field(default=None, ge=1)
    max_alerts: int = Field(default=1000, ge=0, le=20000)


class AlertPatch(BaseModel):
    status: Optional[Literal["new", "under_review", "resolved", "dismissed"]] = None
    note: Optional[str] = Field(default=None, max_length=1000)


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(FRONTEND)


# ---- system
@app.get("/api/health")
def health():
    return {"status": "ok", "version": svc.VERSION}


@app.get("/api/system/info")
def system_info():
    return svc.system_info()


@app.get("/api/overview")
def overview():
    return svc.overview()


# ---- datasets
@app.get("/api/datasets/files")
def data_files():
    return svc.list_data_files()


@app.get("/api/datasets")
def datasets():
    return svc.list_datasets()


@app.post("/api/datasets", status_code=201)
def create_dataset(body: DatasetCreate):
    return svc.register_dataset(body.name, body.kind, body.files, body.nrows, body.n, body.seed, body.drop_duplicates)


@app.get("/api/datasets/{dataset_id}")
def dataset(dataset_id: str):
    return svc.get_dataset(dataset_id)


@app.get("/api/datasets/{dataset_id}/preview")
def dataset_preview(dataset_id: str, limit: int = 25, offset: int = 0):
    return svc.dataset_preview(dataset_id, limit, offset)


# ---- statistics
@app.get("/api/statistics/{dataset_id}")
def statistics(dataset_id: str):
    return svc.dataset_statistics(dataset_id)


@app.get("/api/statistics/{dataset_id}/distribution")
def distribution(dataset_id: str, feature: str, bins: int = 40, log: bool = False):
    return svc.feature_distribution(dataset_id, feature, bins, log)


@app.get("/api/exports/statistics/{dataset_id}")
def export_statistics(dataset_id: str, part: str = "tests"):
    return Response(svc.statistics_csv(dataset_id, part), media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="statistics_{part}_{dataset_id}.csv"'})


# ---- models
@app.post("/api/models/train", status_code=202)
def train(body: TrainRequest):
    return svc.start_training(body.model_dump(), background=True)


@app.get("/api/models/runs")
def runs():
    return svc.list_runs()


@app.get("/api/models/runs/{run_id}")
def run(run_id: str):
    return svc.get_run(run_id)


@app.get("/api/models")
def models():
    return svc.list_models()


@app.post("/api/models/{model_id}/promote")
def promote(model_id: str):
    return svc.promote_model(model_id)


@app.get("/api/exports/evaluation/{run_id}")
def export_evaluation(run_id: str):
    return JSONResponse(svc.evaluation_export(run_id),
                        headers={"Content-Disposition": f'attachment; filename="evaluation_{run_id}.json"'})


# ---- analyses / records
@app.post("/api/analyses", status_code=201)
def create_analysis(body: PredictRequest):
    return svc.run_predictions(body.dataset_id, body.model_id, body.limit, body.max_alerts)


@app.get("/api/analyses")
def analyses():
    return svc.list_analyses()


@app.get("/api/analyses/{analysis_id}")
def analysis(analysis_id: str):
    return svc.get_analysis(analysis_id)


@app.get("/api/analyses/{analysis_id}/records")
def records(analysis_id: str, predicted: Optional[str] = None, actual: Optional[str] = None,
            category: Optional[str] = None, min_prob: Optional[float] = Query(None, ge=0, le=1),
            max_prob: Optional[float] = Query(None, ge=0, le=1), sort: str = "prob_desc",
            page: int = Query(1, ge=1), page_size: int = Query(25, ge=1, le=200)):
    return svc.analysis_records(analysis_id, predicted, actual, category, min_prob, max_prob, sort, page, page_size)


@app.get("/api/analyses/{analysis_id}/records/{row_idx}")
def record(analysis_id: str, row_idx: int):
    return svc.record_detail(analysis_id, row_idx)


# ---- alerts
@app.get("/api/exports/alerts")
def export_alerts(analysis_id: Optional[str] = None, status: Optional[str] = None):
    return Response(svc.alerts_csv(analysis_id, status), media_type="text/csv",
                    headers={"Content-Disposition": 'attachment; filename="alerts.csv"'})


@app.get("/api/alerts")
def alerts(status: Optional[str] = None, priority: Optional[str] = None, analysis_id: Optional[str] = None,
           page: int = Query(1, ge=1), page_size: int = Query(25, ge=1, le=200)):
    return svc.list_alerts(status, priority, analysis_id, page, page_size)


@app.get("/api/alerts/{alert_id}")
def alert(alert_id: int):
    return svc.get_alert(alert_id)


@app.patch("/api/alerts/{alert_id}")
def patch_alert(alert_id: int, body: AlertPatch):
    return svc.update_alert(alert_id, body.status, body.note)
