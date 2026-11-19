"""Central configuration. Every value can be overridden with an environment variable
(or a local .env file, loaded automatically if python-dotenv is installed)."""
from __future__ import annotations
 
import os
from pathlib import Path
 
try:  # optional, only for local development
    from dotenv import load_dotenv
 
    load_dotenv()
except Exception:  # pragma: no cover
    pass
 
ROOT = Path(__file__).resolve().parents[1]
 
 
def _f(name: str, default: float) -> float:
    return float(os.getenv(name, default))
 
 
# ----------------------------------------------------------------------------- location
CITY_NAME = os.getenv("CITY_NAME", "Karachi")
LAT = _f("CITY_LAT", 24.8608)
LON = _f("CITY_LON", 67.0104)
 
# ----------------------------------------------------------------------------- secrets
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")
AQICN_TOKEN = os.getenv("AQICN_TOKEN", "")
AQICN_STATION = os.getenv("AQICN_STATION", "")      # e.g. "@11790"; empty = nearest to lat/lon
AQICN_MAX_KM = _f("AQICN_MAX_KM", 50)               # ignore stations farther than this
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_FALLBACK_MODEL = os.getenv("GROQ_FALLBACK_MODEL", "llama-3.1-8b-instant")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY", "")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT", "")
ALERT_WEBHOOK_URL = os.getenv("ALERT_WEBHOOK_URL", "")  # Discord / Slack incoming webhook
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
GITHUB_REPOSITORY = os.getenv("GITHUB_REPOSITORY", "")  # "user/repo" (set automatically in Actions)
TRAIN_WORKFLOW_FILE = os.getenv("TRAIN_WORKFLOW_FILE", "training.yml")

# GROQ_MODEL = os.getenv("GROQ_MODEL", "<model from your list>")
# GROQ_FALLBACK_MODEL = os.getenv("GROQ_FALLBACK_MODEL", "<smaller model from your list>")

# ----------------------------------------------------------------------------- storage
# "auto": use Hopsworks if HOPSWORKS_API_KEY is set, otherwise a local folder (data/local_store).
STORAGE_BACKEND = os.getenv("STORAGE_BACKEND", "auto")
LOCAL_STORE = Path(os.getenv("LOCAL_STORE", ROOT / "data" / "local_store"))
LOCAL_REGISTRY = Path(os.getenv("LOCAL_REGISTRY", ROOT / "data" / "local_registry"))
MODEL_CACHE = Path(os.getenv("MODEL_CACHE", ROOT / "data" / "model_cache"))
 
# ----------------------------------------------------------------------------- feature groups
FG_RAW = "raw_observations"
FG_FEATURES = "weather_features"
FG_LEDGER = "forecast_ledger"
FG_AGENT_STATE = "agent_state"
FG_ALERTS = "alerts_log"
FG_REPORTS = "weather_reports"
FG_LLM_CALLS = "agent_call_log"
FG_FAILURES = "failure_log"
FG_DRIFT = "drift_log"
FG_MODEL_METRICS = "model_metrics"
FG_DRIFT_REF = "drift_reference"
FG_EXPLAIN = "explainability"
FG_VERSION = int(os.getenv("FG_VERSION", 1))
 
# ----------------------------------------------------------------------------- modelling
HORIZON = 24                    # hours ahead
VAL_DAYS = int(os.getenv("VAL_DAYS", 30))       # time-based validation window (most recent days)
EVAL_STRIDE = int(os.getenv("EVAL_STRIDE", 3))  # evaluate every n-th validation origin (speed)
SEQ_LEN = 24                    # LSTM input window (hours)
MIN_TRAIN_ROWS = 24 * 21        # refuse to train on < 3 weeks of data
SEED = 42
 
# ----------------------------------------------------------------------------- drift
PSI_THRESHOLD = _f("PSI_THRESHOLD", 0.25)          # >0.25 = significant shift
MODEL_DRIFT_RATIO = _f("MODEL_DRIFT_RATIO", 1.5)   # recent RMSE / validation RMSE
DRIFT_MIN_SAMPLES = int(os.getenv("DRIFT_MIN_SAMPLES", 12))
DRIFT_LIVE_HOURS = 168          # window of live features used for data drift
DRIFT_ERROR_HOURS = 72          # window of matured forecasts used for model drift
RETRAIN_COOLDOWN_HOURS = int(os.getenv("RETRAIN_COOLDOWN_HOURS", 12))
 
# ----------------------------------------------------------------------------- reports
REPORT_HOUR_UTC = int(os.getenv("REPORT_HOUR_UTC", 1))  # 01:00 UTC = 06:00 in Karachi
LOCAL_UTC_OFFSET_HOURS = int(os.getenv("LOCAL_UTC_OFFSET_HOURS", 5))   # Karachi = UTC+5
 
# Metrics predicted (base column names in the feature table)
METRICS = ["aqi", "temp", "precip_mm", "rain", "humidity", "uv", "wind_speed", "wind_dir_sin", "wind_dir_cos"]
METRIC_LABELS = {
    "aqi": "Air Quality Index (US AQI)",
    "temp": "Temperature (°C)",
    "precip_mm": "Precipitation (mm/h)",
    "rain": "Rain probability",
    "humidity": "Relative humidity (%)",
    "uv": "UV index",
    "wind_speed": "Wind speed (m/s)",
    "wind_dir_sin": "Wind direction (sin)",
    "wind_dir_cos": "Wind direction (cos)",
}
