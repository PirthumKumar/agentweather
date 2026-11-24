"""Combine several sources into ONE record per hour.
 
Strategy (documented in the report) - SOURCE PRIORITY, to avoid training/serving skew:
  * model inputs come from Open-Meteo in BOTH the historical backfill and the live pipeline
    (it is the only free source with history, so training and live data use the same provider)
  * OpenWeather and AQICN are FALLBACKS, used only when Open-Meteo has no value for that hour
  * every raw source column is kept untouched in `raw_observations` (EDA, cross-checks)
  * set FUSION_MODE=mean to average all available sources instead (for comparison experiments)
"""
from __future__ import annotations
 
import os
 
import numpy as np
import pandas as pd
 
FUSION_MAP = {
    "aqi": ["aq_aqi", "ow_aqi", "om_aqi"],
    "temp": ["ow_temp", "om_temp"],
    "precip_mm": ["ow_precip", "om_precip"],
    "humidity": ["ow_humidity", "om_humidity"],
    "uv": ["om_uv"],
    "wind_speed": ["ow_wind_speed", "om_wind_speed"],
}
WIND_DIR_COLS = ["ow_wind_dir", "om_wind_dir"]
ALL_RAW_COLS = sorted({c for v in FUSION_MAP.values() for c in v} | set(WIND_DIR_COLS)
                      | {"ow_pm25", "ow_pm10", "aq_pm25", "aq_pm10", "om_pm25"})
 
# The raw table must have the SAME columns on every insert (the Feature Store fixes the schema on
# the first insert). Missing sources become NaN floats instead of missing/None columns.
RAW_SCHEMA = ["ow_temp", "ow_humidity", "ow_wind_speed", "ow_wind_dir", "ow_precip", "ow_pm25", "ow_pm10", "ow_aqi",
              "aq_aqi", "aq_pm25", "aq_pm10",
              "om_temp", "om_humidity", "om_precip", "om_wind_speed", "om_wind_dir", "om_aqi", "om_pm25", "om_uv"]


def conform_raw(raw: pd.DataFrame) -> pd.DataFrame:
    """ts + every RAW_SCHEMA column as float64 (NaN when a source did not report)."""
    raw = raw.reset_index(drop=True)
    out = pd.DataFrame({"ts": raw["ts"]})
    for c in RAW_SCHEMA:
        out[c] = pd.to_numeric(raw[c], errors="coerce").astype("float64") if c in raw else np.nan
    return out

def fuse(raw: pd.DataFrame, mode: str | None = None) -> pd.DataFrame:
    """raw: columns ts + any source columns  ->  DataFrame[ts, aqi, temp, ..., wind_dir, n_sources]"""
    mode = mode or os.getenv("FUSION_MODE", "priority")
    raw = raw.copy()
    for c in ALL_RAW_COLS:
        if c not in raw.columns:
            raw[c] = np.nan
    out = pd.DataFrame({"ts": raw["ts"]})
    for name, cols in FUSION_MAP.items():
        mean_all = raw[cols].astype(float).mean(axis=1, skipna=True)
        if mode == "priority":
            om = [c for c in cols if c.startswith("om_")]
            out[name] = raw[om].astype(float).mean(axis=1).fillna(mean_all)   # Open-Meteo first
        else:
            out[name] = mean_all
    rad = np.deg2rad(raw[WIND_DIR_COLS].astype(float))
    s, c = np.sin(rad).mean(axis=1, skipna=True), np.cos(rad).mean(axis=1, skipna=True)
    wd = (np.rad2deg(np.arctan2(s, c)) + 360) % 360          # circular mean of the available sources
    wd = wd.where(raw[WIND_DIR_COLS].notna().any(axis=1))
    if mode == "priority":
        wd = raw["om_wind_dir"].astype(float).fillna(wd)
    out["wind_dir"] = wd
    src_cols = [c for c in ALL_RAW_COLS if c.endswith(("_temp", "_aqi"))]
    out["n_sources"] = raw[src_cols].notna().sum(axis=1)
    return out
