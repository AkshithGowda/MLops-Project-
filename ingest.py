"""Stage 1 - ingestion: pull daily weather from Open-Meteo into the bronze layer (raw JSON)."""
import argparse
import json
from datetime import date, datetime, timedelta, timezone

import requests

from config import ARCHIVE_URL, BRONZE, CITIES, DAILY_VARS, FORECAST_URL


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


def archive_end():
    """Last day covered by the downloaded archive files (live files are ignored), or None."""
    ends = []
    for p in BRONZE.glob("*_*_*.json"):
        if p.stem.endswith("_live"):
            continue
        try:
            ends.append(date.fromisoformat(p.stem.rsplit("_", 1)[1]))
        except ValueError:
            pass
    return max(ends) if ends else None


def fetch_live(city: str, start: str, end: str) -> dict:
    lat, lon = CITIES[city]
    params = {"latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
              "daily": ",".join(DAILY_VARS), "timezone": "Asia/Kolkata"}
    resp = requests.get(FORECAST_URL, params=params, timeout=60)
    resp.raise_for_status()
    return resp.json()


def ingest_live() -> list:
    """Fill the gap between the archive and today from Open-Meteo's forecast API (recent days + today).

    Written to <City>_live.json and overwritten on every call; transform merges it with the archive and, for any
    overlapping day, keeps the most recently ingested value, so the forecast always starts from today's reading.
    """
    BRONZE.mkdir(parents=True, exist_ok=True)
    end_arch = archive_end()
    today = date.today()
    start = (end_arch + timedelta(days=1)) if end_arch else today - timedelta(days=30)
    start = max(start, today - timedelta(days=90))      # the forecast API serves at most ~92 past days
    if start > today:
        print("[ingest] live: archive already covers today")
        return []
    written = []
    for city in CITIES:
        payload = fetch_live(city, start.isoformat(), today.isoformat())
        payload["_meta"] = {"city": city, "ingested_at": datetime.now(timezone.utc).isoformat(), "live": True}
        path = BRONZE / f"{city}_{start.isoformat()}_{today.isoformat()}_live.json"
        for old in BRONZE.glob(f"{city}_*_live.json"):    # keep exactly one live file per city
            old.unlink()
        path.write_text(json.dumps(payload), encoding="utf-8")
        print(f"[ingest] {city}: live {len(payload['daily']['time'])} days {start} .. {today} -> {path.name}")
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
    ingest_live()
