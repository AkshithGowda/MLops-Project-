"""Offline tests: synthetic data only, no network."""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import monitor  # noqa: E402
import transform  # noqa: E402
from config import BASE, CITIES, FEATURES, TARGET  # noqa: E402


def synthetic_silver(days=400, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=days)
    rows = []
    for i, city in enumerate(CITIES):
        season = np.sin(2 * np.pi * dates.dayofyear / 365.25)
        rows.append(pd.DataFrame({
            "date": dates, "city": city,
            "tmax": 31 + 3 * season + i + rng.normal(0, 0.8, days),
            "tmin": 23 + 2 * season + rng.normal(0, 0.5, days),
            "precip": np.clip(rng.gamma(0.5, 8, days) * (season > 0), 0, None),
            "wind": 10 + rng.normal(0, 2, days).clip(-5, None),
            "radiation": 18 + 4 * season + rng.normal(0, 1, days),
        }))
    return pd.concat(rows, ignore_index=True)


def test_clean_removes_duplicates_and_bad_values():
    s = synthetic_silver(50)
    s["ingested_at"] = "2026-01-01"
    dup = s.iloc[:5].copy()
    dup["ingested_at"] = "2026-02-01"
    s.loc[10, "tmax"] = 999         # impossible temperature (row 10 is not among the duplicated rows)
    out, q = transform.clean(pd.concat([s, dup], ignore_index=True))
    assert q["duplicates_removed"] == 5
    assert q["out_of_range_values_nulled"] == 1
    assert out["tmax"].max() < 50


def test_features_have_no_leakage():
    gold = transform.make_features(synthetic_silver())
    g = gold[gold["city"] == "Udupi"].set_index("date")
    day = g.index[10]
    # the label of day D is the *observed* tmax of day D+1, and lag features only look backwards
    assert g.loc[day, TARGET] == pytest.approx(g.loc[day + pd.Timedelta(days=1), "tmax"])
    assert g.loc[day, "tmax_lag1"] == pytest.approx(g.loc[day - pd.Timedelta(days=1), "tmax"])
    assert not gold[FEATURES].isna().any().any()
    assert (gold[[f"city_{c}" for c in CITIES]].sum(axis=1) == 1).all()


def test_psi_same_vs_shifted():
    rng = np.random.default_rng(1)
    a = rng.normal(30, 2, 3000)
    assert monitor.psi(a, rng.normal(30, 2, 500)) < 0.1
    assert monitor.psi(a, rng.normal(34, 2, 500)) > 0.2


def test_drift_table_flags_shifted_feature():
    gold = transform.make_features(synthetic_silver())
    ref, cur = gold.copy(), gold.copy()
    cur["tmax"] = cur["tmax"] + 6
    t = monitor.drift_table(ref, cur)
    assert t["tmax"]["drifted"] and not t["tmin"]["drifted"]


@pytest.fixture()
def client(tmp_path, monkeypatch):
    gold = transform.make_features(synthetic_silver()).dropna(subset=[TARGET])
    model = Ridge().fit(gold[FEATURES], gold[TARGET])
    joblib.dump(model, tmp_path / "model.joblib")
    meta = {"model_name": "ridge", "registry_version": "test", "trained_at": "now", "features": FEATURES}
    (tmp_path / "metadata.json").write_text(json.dumps(meta))
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    import serve
    monkeypatch.setattr(serve, "PREDICTIONS", tmp_path / "predictions")
    serve.load_artifacts.cache_clear()
    from fastapi.testclient import TestClient
    return TestClient(serve.app)


PAYLOAD = {"city": "Udupi", "day_of_year": 120, "tmax": 33, "tmin": 25, "precip": 0, "wind": 12, "radiation": 22,
           "tmax_lag1": 32.5, "tmax_lag2": 32, "tmax_lag3": 33, "tmin_lag1": 25, "precip_lag1": 0,
           "tmax_roll7": 32.6, "precip_roll7": 0.2}


def test_api_health_and_predict(client):
    assert client.get("/health").json()["status"] == "ok"
    r = client.post("/predict", json=PAYLOAD)
    assert r.status_code == 200
    assert 20 < r.json()["predicted_tmax_next_day_c"] < 45


def test_api_rejects_bad_input(client):
    assert client.post("/predict", json={**PAYLOAD, "city": "Atlantis"}).status_code == 422
    assert client.post("/predict", json={**PAYLOAD, "tmax": 400}).status_code == 422


def test_dashboard_page_and_job_guards(client):
    page = client.get("/")
    assert page.status_code == 200 and "Weather MLOps" in page.text
    assert client.post("/api/run/rm-rf").status_code == 404          # only the fixed list of steps can run
    assert client.get("/api/city/Atlantis").status_code == 404
    assert client.get("/api/job").json()["state"] in ("idle", "done", "failed", "running")
