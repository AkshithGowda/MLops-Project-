"""Stage 2 - transformation: bronze (raw JSON) -> silver (clean table) -> gold (model features)."""
import json
import math

import numpy as np
import pandas as pd

from config import BASE, BRONZE, CITIES, DAILY_VARS, FEATURES, GOLD, LIMITS, REPORTS, SILVER, TARGET


def load_bronze() -> pd.DataFrame:
    frames = []
    for path in sorted(BRONZE.glob("*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        df = pd.DataFrame(raw["daily"]).rename(columns={"time": "date", **DAILY_VARS})
        df["city"] = raw["_meta"]["city"]
        df["ingested_at"] = raw["_meta"]["ingested_at"]
        frames.append(df)
    if not frames:
        raise SystemExit("No bronze files found - run ingest first.")
    return pd.concat(frames, ignore_index=True)


def clean(df: pd.DataFrame):
    """Deduplicate, validate ranges, fill short gaps. Returns (silver, quality_report)."""
    q = {"rows_in": int(len(df))}
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("ingested_at").drop_duplicates(["city", "date"], keep="last")
    q["duplicates_removed"] = q["rows_in"] - int(len(df))

    out_of_range = 0
    for col, (lo, hi) in LIMITS.items():
        bad = df[col].notna() & ~df[col].between(lo, hi)
        out_of_range += int(bad.sum())
        df.loc[bad, col] = np.nan
    q["out_of_range_values_nulled"] = out_of_range

    parts, filled = [], 0
    for city, g in df.groupby("city"):
        g = g.set_index("date").sort_index()
        g = g.reindex(pd.date_range(g.index.min(), g.index.max(), freq="D"))
        g["city"] = city
        before = int(g[BASE].isna().sum().sum())
        g[BASE] = g[BASE].interpolate(limit=2, limit_area="inside")
        filled += before - int(g[BASE].isna().sum().sum())
        parts.append(g.rename_axis("date").reset_index())
    silver = pd.concat(parts, ignore_index=True)
    q["missing_values_interpolated"] = filled
    before = len(silver)
    silver = silver.dropna(subset=BASE).drop(columns=["ingested_at"]).reset_index(drop=True)
    q["rows_dropped_unfixable"] = before - int(len(silver))
    q["rows_out"] = int(len(silver))
    return silver, q


def row_features(payload: dict) -> dict:
    """Calendar + city encoding, shared by the training pipeline and the API."""
    doy = payload["day_of_year"]
    row = {k: payload[k] for k in FEATURES if k in payload}
    row["doy_sin"] = math.sin(2 * math.pi * doy / 365.25)
    row["doy_cos"] = math.cos(2 * math.pi * doy / 365.25)
    for c in CITIES:
        row[f"city_{c}"] = 1.0 if payload["city"] == c else 0.0
    return row


def make_features(silver: pd.DataFrame) -> pd.DataFrame:
    """One row per city-day. The label is tomorrow's max temperature (shifted back, so no leakage)."""
    parts = []
    for city, g in silver.sort_values(["city", "date"]).groupby("city"):
        g = g.copy()
        for k in (1, 2, 3):
            g[f"tmax_lag{k}"] = g["tmax"].shift(k)
        g["tmin_lag1"] = g["tmin"].shift(1)
        g["precip_lag1"] = g["precip"].shift(1)
        g["tmax_roll7"] = g["tmax"].rolling(7).mean()
        g["precip_roll7"] = g["precip"].rolling(7).mean()
        g[TARGET] = g["tmax"].shift(-1)
        parts.append(g)
    gold = pd.concat(parts, ignore_index=True)
    doy = gold["date"].dt.dayofyear
    gold["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    gold["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    for c in CITIES:
        gold[f"city_{c}"] = (gold["city"] == c).astype(float)
    # rows whose features are incomplete are dropped; the newest day keeps a NaN label (used for forecasting)
    gold = gold.dropna(subset=FEATURES).reset_index(drop=True)
    return gold


def run() -> pd.DataFrame:
    SILVER.mkdir(parents=True, exist_ok=True)
    GOLD.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    silver, quality = clean(load_bronze())
    silver.to_parquet(SILVER / "weather.parquet", index=False)
    gold = make_features(silver)
    gold.to_parquet(GOLD / "features.parquet", index=False)
    quality["gold_rows"] = int(len(gold))
    quality["date_range"] = [str(silver["date"].min().date()), str(silver["date"].max().date())]
    (REPORTS / "data_quality.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
    print("[transform] data quality:", json.dumps(quality))
    return gold


if __name__ == "__main__":
    run()
