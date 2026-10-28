"""Stage 1 - ingestion: pull daily weather from Open-Meteo into the bronze layer (raw JSON)."""
import argparse
import json
from datetime import date, datetime, timedelta, timezone

import requests

from config import ARCHIVE_URL, BRONZE, CITIES, DAILY_VARS


def fetch_city(city: str, start: str, end: str) -> dict:
    lat, lon = CITIES[city]
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "daily": ",".join(DAILY_VARS),
        "timezone": "Asia/Kolkata",
    }
    resp = requests.get(ARCHIVE_URL, params=params, timeout=90)
    resp.raise_for_status()
    return resp.json()


def ingest(start: str, end: str, force: bool = False) -> list:
    """Download every city for [start, end]. Existing files are reused unless force=True."""
    BRONZE.mkdir(parents=True, exist_ok=True)
    written = []
    for city in CITIES:
        path = BRONZE / f"{city}_{start}_{end}.json"
        if path.exists() and not force:
            print(f"[ingest] {city}: reusing {path.name}")
        else:
            payload = fetch_city(city, start, end)
            payload["_meta"] = {"city": city, "ingested_at": datetime.now(timezone.utc).isoformat()}
            path.write_text(json.dumps(payload), encoding="utf-8")
            print(f"[ingest] {city}: {len(payload['daily']['time'])} days -> {path.name}")
        written.append(path)
    return written


def default_end() -> str:
    # the archive lags a few days behind real time
    return (date.today() - timedelta(days=6)).isoformat()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Ingest raw weather data (bronze layer)")
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--end", default=default_end())
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    ingest(a.start, a.end, a.force)
