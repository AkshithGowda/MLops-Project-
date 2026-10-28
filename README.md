# Coastal Karnataka Weather MLOps Pipeline

Course project for **Data Engineering and MLOps (BAD714C)**, VTU.

An end-to-end pipeline that ingests daily weather for Udupi, Mangaluru and Bengaluru, builds a layered
data store (bronze / silver / gold), trains and tracks models with MLflow, serves the best one through a
FastAPI service, and monitors it for data drift and accuracy loss - with an automatic retrain decision.
**No Docker needed**: everything runs in a plain Python virtual environment.

| Syllabus module | Where it lives |
|---|---|
| 1. Data engineering lifecycle | `ingest.py`, `transform.py` (generation, ingestion, transformation, serving) |
| 2. Data architecture | bronze / silver / gold layers, diagram in the report |
| 3. MLOps challenges, model development | `train.py` (baseline + 3 models, MLflow tracking, registry, time-based split) |
| 4. Preparing and deploying for production | `serve.py` (FastAPI), `.github/workflows/ci.yml` (CI), `requirements.txt` (pinned) |
| 5. Monitoring and feedback loop | `monitor.py`, `demo_drift.py`, `.github/workflows/retrain.yml`, prediction log |

## Setup (Windows / Linux / macOS)

```bash
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python run_pipeline.py all                  # ingest -> transform -> train -> monitor
python run_pipeline.py all --auto-retrain   # also retrain automatically when the monitor says RETRAIN
python demo_drift.py                        # feedback-loop demo: dry-season model meets the monsoon
python -m pytest -q                         # 7 offline tests
uvicorn serve:app --port 8000               # dashboard at http://127.0.0.1:8000/  (API docs at /docs)
mlflow ui --backend-store-uri sqlite:///mlflow.db   # browse experiments and the model registry
```

The **dashboard** (`dashboard.py`, `ui/index.html`) is served by the same app. It shows what is stored at every
pipeline stage, runs any step with a live log, plots the forecast, compares the models, and shows the drift
monitor and the prediction log. Nothing extra to install.

API examples:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/forecast/Udupi
```

Data comes from the free [Open-Meteo historical API](https://open-meteo.com/) (no key). The first run downloads
about 6,000 rows; later runs reuse the cached bronze files.

## Design notes

- **Label** = next day's maximum temperature. Features use only today and earlier days, so there is no leakage
  (covered by a test). The train/test split is chronological (newest 15 % is the test set).
- **Baseline** = "tomorrow equals today". Every model is compared against it; the gain is modest because the
  weather is persistent - the report discusses this honestly.
- **Monitoring** compares the latest window with a *season-matched* slice of the training data (weather is
  seasonal, so comparing a monsoon week with the yearly average would always look like drift). Decision:
  `OK`, `WATCH` (inputs drifted, accuracy holding) or `RETRAIN` (accuracy loss, or drift on a season the model
  never saw).
- `demo_drift.py` simulates a seasonal shift by training only on Dec-May data, then monitoring a June-Sept window.
