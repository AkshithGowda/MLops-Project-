"""Stage 3 - model development: train several models, track them in MLflow, promote the best one."""
import argparse
import json
from datetime import datetime, timezone

import config  # noqa: F401  (must come before mlflow: it silences library warnings)
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import EXPERIMENT, FEATURES, GOLD, MODELS, REGISTERED_MODEL, ROOT, TARGET


def setup_mlflow():
    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=(ROOT / "mlartifacts").as_uri())
    mlflow.set_experiment(EXPERIMENT)


def load_training_data(start=None, end=None, months=None) -> pd.DataFrame:
    df = pd.read_parquet(GOLD / "features.parquet").dropna(subset=[TARGET])
    if start:
        df = df[df["date"] >= start]
    if end:
        df = df[df["date"] <= end]
    if months:
        df = df[df["date"].dt.month.isin(months)]
    if len(df) < 200:
        raise SystemExit(f"Only {len(df)} rows selected - not enough to train.")
    return df.sort_values("date").reset_index(drop=True)


def time_split(df: pd.DataFrame, test_frac=0.15):
    """Chronological split: the newest 15 % of dates is the test set (never shuffle time series)."""
    cutoff = df["date"].sort_values().iloc[int(len(df) * (1 - test_frac))]
    return df[df["date"] < cutoff], df[df["date"] >= cutoff]


def metrics(y, pred) -> dict:
    return {
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "r2": float(r2_score(y, pred)),
    }


def candidates():
    return {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "random_forest": RandomForestRegressor(n_estimators=200, max_depth=12, min_samples_leaf=3, n_jobs=-1, random_state=42),
        "hist_gradient_boosting": HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=42),
    }


def train(start=None, end=None, months=None, register=True) -> dict:
    setup_mlflow()
    df = load_training_data(start, end, months)
    tr, te = time_split(df)
    X_tr, y_tr, X_te, y_te = tr[FEATURES], tr[TARGET], te[FEATURES], te[TARGET]
    window = {
        "train_start": str(tr["date"].min().date()), "train_end": str(tr["date"].max().date()),
        "test_start": str(te["date"].min().date()), "test_end": str(te["date"].max().date()),
        "months": months or "all", "train_rows": len(tr), "test_rows": len(te),
    }
    print(f"[train] {len(tr)} train rows, {len(te)} test rows, window {window['train_start']}..{window['test_end']}")

    # naive baseline: "tomorrow will be as hot as today". Any real model has to beat this.
    base = metrics(y_te, te["tmax"])
    with mlflow.start_run(run_name="baseline_persistence"):
        mlflow.log_params({"model": "persistence", **{k: str(v) for k, v in window.items()}})
        mlflow.log_metrics(base)
    print(f"[train] baseline persistence   MAE={base['mae']:.3f}  RMSE={base['rmse']:.3f}  R2={base['r2']:.3f}")

    results = {}
    for name, model in candidates().items():
        with mlflow.start_run(run_name=name) as run:
            model.fit(X_tr, y_tr)
            m = metrics(y_te, model.predict(X_te))
            mlflow.log_params({"model": name, **{k: str(v) for k, v in window.items()}})
            mlflow.log_params({f"hp_{k}": v for k, v in model.get_params(deep=False).items() if isinstance(v, (int, float, str))})
            mlflow.log_metrics(m)
            info = mlflow.sklearn.log_model(model, name="model", signature=infer_signature(X_te, model.predict(X_te)),
                                           serialization_format="cloudpickle")  # our own models; skops rejects tree types
            results[name] = {"model": model, "metrics": m, "run_id": run.info.run_id, "uri": info.model_uri}
        print(f"[train] {name:<24} MAE={m['mae']:.3f}  RMSE={m['rmse']:.3f}  R2={m['r2']:.3f}")

    best = min(results, key=lambda n: results[n]["metrics"]["mae"])
    chosen = results[best]
    version = None
    if register:
        version = mlflow.register_model(chosen["uri"], REGISTERED_MODEL).version
    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(chosen["model"], MODELS / "model.joblib")
    # reference sample of the training data (with dates, so the monitor can compare like seasons with like)
    tr[FEATURES + ["date"]].to_parquet(MODELS / "reference.parquet", index=False)
    meta = {
        "model_name": best, "registry_version": version, "run_id": chosen["run_id"],
        "trained_at": datetime.now(timezone.utc).isoformat(), "features": FEATURES,
        "test_metrics": chosen["metrics"], "baseline_metrics": base, "window": window,
    }
    (MODELS / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[train] best model: {best} (registry v{version}), MAE {chosen['metrics']['mae']:.3f} vs baseline {base['mae']:.3f}")
    return meta


def parse_months(s):
    return [int(x) for x in s.split(",")] if s else None


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train and register the next-day max-temperature model")
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--months", help="comma-separated months to train on, e.g. 1,2,3,4")
    a = ap.parse_args()
    train(a.start, a.end, parse_months(a.months))
