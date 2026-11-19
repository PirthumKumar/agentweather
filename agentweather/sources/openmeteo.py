"""Open-Meteo (free, no API key): weather + air quality INCLUDING history. Prefix: om_
 
Used (1) as a third live source and (2) as the main source for the historical backfill,
because OpenWeather's free plan has no historical weather and AQICN has no history at all.
"""
from __future__ import annotations
 
import logging
from datetime import date, datetime, timedelta, timezone
 
import pandas as pd
 
from .. import config as C
from .http import get_json
 
log = logging.getLogger(__name__)
W_VARS = "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,wind_direction_10m"
AQ_VARS = "us_aqi,pm2_5,uv_index"
RENAME = {
    "temperature_2m": "om_temp", "relative_humidity_2m": "om_humidity", "precipitation": "om_precip",
    "wind_speed_10m": "om_wind_speed", "wind_direction_10m": "om_wind_dir",
    "us_aqi": "om_aqi", "pm2_5": "om_pm25", "uv_index": "om_uv",
}
 
 
def _frame(j: dict | None) -> pd.DataFrame:
    if not j or "hourly" not in j:
        return pd.DataFrame()
    df = pd.DataFrame(j["hourly"])
    df["ts"] = pd.to_datetime(df.pop("time"))  # timezone=UTC requested -> naive UTC
    return df.rename(columns=RENAME)
 
 
def _weather(url: str, **extra) -> pd.DataFrame:
    p = {"latitude": C.LAT, "longitude": C.LON, "hourly": W_VARS, "wind_speed_unit": "ms", "timezone": "UTC"}
    p.update(extra)
    return _frame(get_json(url, p))
 
 
def _air(**extra) -> pd.DataFrame:
    p = {"latitude": C.LAT, "longitude": C.LON, "hourly": AQ_VARS, "timezone": "UTC"}
    p.update(extra)
    return _frame(get_json("https://air-quality-api.open-meteo.com/v1/air-quality", p))
 
 
def _merge(w: pd.DataFrame, a: pd.DataFrame) -> pd.DataFrame:
    if w.empty and a.empty:
        return pd.DataFrame()
    if w.empty:
        return a
    if a.empty:
        return w
    return w.merge(a, on="ts", how="outer")
 
 
def fetch_recent(past_days: int = 2) -> pd.DataFrame:
    """Last `past_days` of hourly data (analysis values) - for live ingestion."""
    w = _weather("https://api.open-meteo.com/v1/forecast", past_days=past_days, forecast_days=1)
    a = _air(past_days=past_days, forecast_days=1)
    return _merge(w, a)
 
 
def fetch_history(start: date, end: date) -> pd.DataFrame:
    """Hourly history from `start` to `end` (inclusive). Archive API has ~5 day lag, so the
    most recent days are taken from the forecast API's `past_days`."""
    today = datetime.now(timezone.utc).date()
    archive_end = min(end, today - timedelta(days=6))
    parts = []
    if start <= archive_end:
        parts.append(_merge(
            _weather("https://archive-api.open-meteo.com/v1/archive",
                     start_date=start.isoformat(), end_date=archive_end.isoformat()),
            _air(start_date=start.isoformat(), end_date=archive_end.isoformat()),
        ))
    if end > archive_end:
        gap_days = min((today - archive_end).days + 1, 92)
        parts.append(fetch_recent(past_days=gap_days))
    parts = [p for p in parts if not p.empty]
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts).drop_duplicates("ts", keep="last").sort_values("ts")
    now = pd.Timestamp.now(tz="UTC").tz_localize(None).floor("h")
    return df[(df.ts >= pd.Timestamp(start)) & (df.ts <= min(pd.Timestamp(end) + pd.Timedelta(hours=23), now))]
