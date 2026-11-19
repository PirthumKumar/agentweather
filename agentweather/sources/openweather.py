"""OpenWeather (free tier): current weather + air pollution. Prefix: ow_"""
from __future__ import annotations
 
import logging
from datetime import datetime, timezone
 
import pandas as pd
 
from .. import config as C
from .aqi import pm25_to_aqi
from .http import get_json
 
log = logging.getLogger(__name__)
BASE = "https://api.openweathermap.org/data/2.5"
 
 
def fetch_current() -> dict:
    """Return one dict of ow_* values for 'now' ({} if the API is down / key missing)."""
    if not C.OPENWEATHER_API_KEY:
        log.warning("OPENWEATHER_API_KEY not set - skipping OpenWeather")
        return {}
    p = {"lat": C.LAT, "lon": C.LON, "appid": C.OPENWEATHER_API_KEY, "units": "metric"}
    out: dict = {}
    w = get_json(f"{BASE}/weather", p)
    if w:
        out.update(
            ow_temp=w["main"]["temp"], ow_humidity=w["main"]["humidity"],
            ow_wind_speed=w["wind"].get("speed"), ow_wind_dir=w["wind"].get("deg"),
            ow_precip=(w.get("rain", {}).get("1h", 0.0) + w.get("snow", {}).get("1h", 0.0)),
        )
    a = get_json(f"{BASE}/air_pollution", p)
    if a and a.get("list"):
        comp = a["list"][0]["components"]
        out.update(ow_pm25=comp.get("pm2_5"), ow_pm10=comp.get("pm10"), ow_aqi=pm25_to_aqi(comp.get("pm2_5")))
    return out
 
 
def fetch_air_history(start: datetime, end: datetime) -> pd.DataFrame:
    """Hourly air-pollution history (free endpoint) - used by the backfill."""
    if not C.OPENWEATHER_API_KEY:
        return pd.DataFrame()
    p = {"lat": C.LAT, "lon": C.LON, "appid": C.OPENWEATHER_API_KEY,
         "start": int(start.replace(tzinfo=timezone.utc).timestamp()),
         "end": int(end.replace(tzinfo=timezone.utc).timestamp())}
    j = get_json(f"{BASE}/air_pollution/history", p)
    if not j or not j.get("list"):
        return pd.DataFrame()
    rows = [{"ts": pd.to_datetime(i["dt"], unit="s"), "ow_pm25": i["components"].get("pm2_5"),
             "ow_pm10": i["components"].get("pm10")} for i in j["list"]]
    df = pd.DataFrame(rows)
    df["ow_aqi"] = df["ow_pm25"].map(pm25_to_aqi)
    df["ts"] = df["ts"].dt.floor("h")
    return df.groupby("ts", as_index=False).mean()
