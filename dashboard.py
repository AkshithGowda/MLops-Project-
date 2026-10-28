"""Web dashboard: shows each stage of the pipeline, runs it with a live log, and displays forecasts and monitoring.

serve.py mounts this on the same FastAPI app, so  uvicorn serve:app --port 8000  and open http://127.0.0.1:8000/
"""
import json
import os
import sqlite3
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from config import BRONZE, CITIES, EXPERIMENT, FEATURES, GOLD, MODELS, PREDICTIONS, REPORTS, ROOT, SILVER, TARGET

router = APIRouter()
UI_FILE = ROOT / "ui" / "index.html"

# the only commands the dashboard may start (fixed list, no user-supplied arguments)
STEPS = {
    "ingest": ["run_pipeline.py", "ingest"],
    "transform": ["run_pipeline.py", "transform"],
    "train": ["run_pipeline.py", "train"],
    "monitor": ["run_pipeline.py", "monitor"],
    "all": ["run_pipeline.py", "all"],
    "all-auto": ["run_pipeline.py", "all", "--auto-retrain"],
    "demo": ["demo_drift.py"],
}
MAX_LINES = 3000
JOB = {"id": 0, "step": None, "state": "idle", "started": None, "ended": None, "returncode": None, "lines": []}
LOCK = threading.Lock()
_hooks = {"artifacts": None, "on_finish": lambda: None}   # filled in by register()


def _now():
    return datetime.now(timezone.utc).isoformat()


def _run_job(args: list):
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    code = -1
    try:
        proc = subprocess.Popen([sys.executable, *args], cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        for line in proc.stdout:
            with LOCK:
                JOB["lines"].append(line.rstrip("\n"))
                del JOB["lines"][:-MAX_LINES]
        code = proc.wait()
    except Exception as exc:  # the job could not even start
        with LOCK:
            JOB["lines"].append(f"[dashboard] could not start the job: {exc}")
    with LOCK:
        JOB.update(state="done" if code == 0 else "failed", ended=_now(), returncode=code)
    _hooks["on_finish"]()


@router.get("/", include_in_schema=False)
def index():
    return FileResponse(UI_FILE)


@router.post("/api/run/{step}")
def run_step(step: str):
    if step not in STEPS:
        raise HTTPException(status_code=404, detail=f"Unknown step. Choose one of {list(STEPS)}")
    with LOCK:
        if JOB["state"] == "running":
            raise HTTPException(status_code=409, detail="A job is already running.")
        JOB.update(id=JOB["id"] + 1, step=step, state="running", started=_now(), ended=None, returncode=None,
                   lines=[f"$ python {' '.join(STEPS[step])}"])
        job_id = JOB["id"]
    threading.Thread(target=_run_job, args=(STEPS[step],), daemon=True).start()
    return {"job": job_id, "step": step}


@router.get("/api/job")
def job(since: int = 0):
    with LOCK:
        return {**{k: v for k, v in JOB.items() if k != "lines"}, "total_lines": len(JOB["lines"]),
                "lines": JOB["lines"][since:]}


def _read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _bronze():
    files = []
    for p in sorted(BRONZE.glob("*.json")):
        d = _read_json(p) or {}
        files.append({"name": p.name, "city": (d.get("_meta") or {}).get("city"), "kb": round(p.stat().st_size / 1024),
                      "days": len((d.get("daily") or {}).get("time", [])),
                      "ingested_at": (d.get("_meta") or {}).get("ingested_at")})
    return files


def _table(path, cols):
    if not path.exists():
        return None
    df = pd.read_parquet(path, columns=cols)
    return df


def _mlflow_runs(limit=4):
    """Read the experiment straight from the MLflow SQLite file (cheaper than importing mlflow)."""
    db = ROOT / "mlflow.db"
    if not db.exists():
        return {"runs": [], "versions": 0}
    try:
        con = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
        rows = con.execute(
            "SELECT r.run_uuid, r.name, r.start_time FROM runs r JOIN experiments e ON e.experiment_id = r.experiment_id "
            "WHERE e.name = ? AND r.lifecycle_stage = 'active' AND r.status = 'FINISHED' ORDER BY r.start_time DESC LIMIT ?",
            (EXPERIMENT, limit)).fetchall()
        runs = []
        for uid, name, start in reversed(rows):
            m = dict(con.execute("SELECT key, value FROM latest_metrics WHERE run_uuid = ?", (uid,)).fetchall())
            runs.append({"run_id": uid[:8], "name": name, "mae": m.get("mae"), "rmse": m.get("rmse"), "r2": m.get("r2"),
                         "started": datetime.fromtimestamp(start / 1000, timezone.utc).isoformat()})
        versions = con.execute("SELECT COUNT(*) FROM model_versions").fetchone()[0]
        con.close()
        return {"runs": runs, "versions": versions}
    except sqlite3.Error:
        return {"runs": [], "versions": 0}


def _predictions(n=12):
    path = PREDICTIONS / "log.jsonl"
    if not path.exists():
        return {"count": 0, "recent": []}
    lines = path.read_text(encoding="utf-8").splitlines()
    recent = []
    for ln in reversed(lines[-n:]):
        try:
            r = json.loads(ln)
        except ValueError:
            continue
        recent.append({"ts": r.get("ts"), "source": r.get("source"), "city": r.get("city"),
                       "prediction": round(r.get("prediction", 0), 2), "model_version": r.get("model_version")})
    return {"count": len(lines), "recent": recent}


@router.get("/api/overview")
def overview():
    silver = _table(SILVER / "weather.parquet", ["date", "city"])
    gold = _table(GOLD / "features.parquet", ["date", TARGET])
    return {
        "now": _now(),
        "bronze": _bronze(),
        "silver": None if silver is None else {
            "rows": len(silver), "from": str(silver["date"].min().date()), "to": str(silver["date"].max().date()),
            "per_city": silver["city"].value_counts().to_dict()},
        "quality": _read_json(REPORTS / "data_quality.json"),
        "gold": None if gold is None else {
            "rows": int(gold[TARGET].notna().sum()), "features": len(FEATURES), "features_list": FEATURES,
            "latest": str(gold["date"].max().date())},
        "model": _read_json(MODELS / "metadata.json"),
        "mlflow": _mlflow_runs(),
        "monitor": _read_json(REPORTS / "monitoring.json"),
        "predictions": _predictions(),
        "cities": list(CITIES),
    }


@router.get("/api/city/{city}")
def city_series(city: str, days: int = 60):
    """Recent real vs predicted max temperature plus tomorrow's forecast. Does not write to the prediction log."""
    if city not in CITIES:
        raise HTTPException(status_code=404, detail=f"City must be one of {list(CITIES)}")
    path = GOLD / "features.parquet"
    if not path.exists():
        raise HTTPException(status_code=503, detail="No feature table - run the pipeline first.")
    model, meta = _hooks["artifacts"]()
    g = pd.read_parquet(path)
    g = g[g["city"] == city].sort_values("date").tail(max(10, min(days, 365))).reset_index(drop=True)
    g["pred"] = model.predict(g[FEATURES].astype(float))
    history = [{"date": str((d + pd.Timedelta(days=1)).date()), "actual": None if pd.isna(a) else round(float(a), 2),
                "predicted": round(float(p), 2)} for d, a, p in zip(g["date"], g[TARGET], g["pred"])]
    last = g.iloc[-1]
    return {"city": city, "model": meta["model_name"], "based_on_date": str(last["date"].date()),
            "forecast_for": str((last["date"] + pd.Timedelta(days=1)).date()),
            "forecast": round(float(last["pred"]), 2), "today_tmax": round(float(last["tmax"]), 2),
            "history": history[:-1]}


def register(app, artifacts, on_job_finish):
    """Attach the dashboard routes to the main app. `artifacts` loads (model, metadata) or raises HTTP 503."""
    _hooks.update(artifacts=artifacts, on_finish=on_job_finish)
    app.include_router(router)
