"""Stage 4 - serving: FastAPI app. Run with:  uvicorn serve:app --port 8000   (no Docker needed)."""
import json
import os
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import dashboard
from config import CITIES, FEATURES, GOLD, MODELS, PREDICTIONS
from transform import row_features

app = FastAPI(title="Coastal Karnataka next-day max temperature", version="1.0")


@lru_cache(maxsize=4)
def load_artifacts(model_dir: str):
    d = Path(model_dir)
    if not (d / "model.joblib").exists():
        raise FileNotFoundError(f"No trained model in {d} - run `python run_pipeline.py train` first.")
    meta = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
    return joblib.load(d / "model.joblib"), meta


def artifacts():
    try:
        return load_artifacts(os.environ.get("MODEL_DIR", str(MODELS)))
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


class WeatherToday(BaseModel):
    city: Literal[tuple(CITIES)]
    day_of_year: int = Field(ge=1, le=366)
    tmax: float = Field(ge=5, le=50)
    tmin: float = Field(ge=0, le=40)
    precip: float = Field(ge=0, le=600)
    wind: float = Field(ge=0, le=150)
    radiation: float = Field(ge=0, le=40)
    tmax_lag1: float
    tmax_lag2: float
    tmax_lag3: float
    tmin_lag1: float
    precip_lag1: float
    tmax_roll7: float
    precip_roll7: float


def log_prediction(source: str, city: str, features: dict, prediction: float, version):
    """Every served prediction is appended here; the monitor can later join it with the real outcome."""
    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    rec = {"ts": datetime.now(timezone.utc).isoformat(), "source": source, "city": city,
           "prediction": prediction, "model_version": version, "features": features}
    with open(PREDICTIONS / "log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


@app.get("/health")
def health():
    model, meta = artifacts()
    return {"status": "ok", "model": meta["model_name"], "trained_at": meta["trained_at"]}


@app.get("/model-info")
def model_info():
    _, meta = artifacts()
    return meta


@app.post("/predict")
def predict(body: WeatherToday):
    model, meta = artifacts()
    row = row_features(body.model_dump())
    pred = float(model.predict(pd.DataFrame([row])[FEATURES])[0])
    log_prediction("predict", body.city, row, pred, meta.get("registry_version"))
    return {"city": body.city, "predicted_tmax_next_day_c": round(pred, 2), "model": meta["model_name"],
            "model_version": meta.get("registry_version")}


@app.get("/forecast/{city}")
def forecast(city: str):
    """Forecast tomorrow's max temperature from the newest row of the gold feature table."""
    if city not in CITIES:
        raise HTTPException(status_code=404, detail=f"City must be one of {list(CITIES)}")
    path = GOLD / "features.parquet"
    if not path.exists():
        raise HTTPException(status_code=503, detail="No feature table - run the pipeline first.")
    model, meta = artifacts()
    g = pd.read_parquet(path)
    last = g[g["city"] == city].sort_values("date").iloc[-1]
    pred = float(model.predict(last[FEATURES].to_frame().T.astype(float))[0])
    log_prediction("forecast", city, last[FEATURES].astype(float).to_dict(), pred, meta.get("registry_version"))
    target_day = (last["date"] + pd.Timedelta(days=1)).date().isoformat()
    return {"city": city, "based_on_date": str(last["date"].date()), "forecast_for": target_day,
            "predicted_tmax_c": round(pred, 2), "model": meta["model_name"]}


# web dashboard at "/" ; a finished training job must not leave the old model cached
dashboard.register(app, artifacts, load_artifacts.cache_clear)
