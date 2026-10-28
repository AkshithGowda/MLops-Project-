"""Shared settings for the Coastal Karnataka weather MLOps pipeline."""
import logging
import os
import warnings
from pathlib import Path

# keep console output readable: silence harmless library chatter
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("mlflow.sklearn").setLevel(logging.ERROR)

ROOT = Path(__file__).resolve().parent
BRONZE = ROOT / "data" / "bronze"      # raw API responses, untouched
SILVER = ROOT / "data" / "silver"      # cleaned, validated daily table
GOLD = ROOT / "data" / "gold"          # model-ready features
PREDICTIONS = ROOT / "data" / "predictions"  # log of served predictions (feedback loop)
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
CITIES = {                       # name -> (latitude, longitude)
    "Udupi": (13.34, 74.75),
    "Mangaluru": (12.91, 74.86),
    "Bengaluru": (12.97, 77.59),
}
# Open-Meteo daily variable -> short column name used in the pipeline
DAILY_VARS = {
    "temperature_2m_max": "tmax",
    "temperature_2m_min": "tmin",
    "precipitation_sum": "precip",
    "wind_speed_10m_max": "wind",
    "shortwave_radiation_sum": "radiation",
}
BASE = ["tmax", "tmin", "precip", "wind", "radiation"]

# physical sanity limits used by the data-quality step
LIMITS = {"tmax": (5, 50), "tmin": (0, 40), "precip": (0, 600), "wind": (0, 150), "radiation": (0, 40)}

TARGET = "target_tmax_next"
FEATURES = (
    BASE
    + ["tmax_lag1", "tmax_lag2", "tmax_lag3", "tmin_lag1", "precip_lag1", "tmax_roll7", "precip_roll7"]
    + ["doy_sin", "doy_cos"]
    + [f"city_{c}" for c in CITIES]
)
# continuous features that are checked for drift (dummies / calendar columns are not)
DRIFT_FEATURES = BASE + ["tmax_roll7", "precip_roll7"]

EXPERIMENT = "weather-next-day-tmax"
REGISTERED_MODEL = "temp-forecaster"
