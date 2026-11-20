"""Imputation, time features, derived features and 24h-ahead targets."""
from __future__ import annotations
 
import numpy as np
import pandas as pd
 
from .. import config as C
 
CORE = ["aqi", "temp", "precip_mm", "humidity", "uv", "wind_speed"]
BASE_COLS = CORE + ["wind_dir"]
TIME_FEATS = ["hour_of_day", "day_of_week", "month_of_year", "hour_sin", "hour_cos", "doy_sin", "doy_cos", "is_weekend"]
# compact set fed to the LSTM sequence (current values + clock)
SEQ_COLS = CORE + ["rain", "wind_dir_sin", "wind_dir_cos", "hour_sin", "hour_cos", "doy_sin", "doy_cos"]
 
 
def _derived_names() -> list[str]:
    cols = []
    for v in CORE:
        cols += [f"{v}_lag3", f"{v}_lag24", f"{v}_diff1", f"{v}_diff3", f"{v}_roll6", f"{v}_roll24", f"{v}_std24"]
    cols += ["wind_dir_sin_roll6", "wind_dir_cos_roll6", "rain_roll6", "rain_roll24"]
    return cols
 
 
FEATURE_COLS = CORE + ["rain", "wind_dir_sin", "wind_dir_cos"] + TIME_FEATS + _derived_names() + ["n_imputed"]
TARGET_COLS = [f"target_{m}" for m in C.METRICS]
 
 
# --------------------------------------------------------------------------- imputation
def impute(df: pd.DataFrame) -> pd.DataFrame:
    """df: columns ts + BASE_COLS (+ n_sources). Returns an hourly, gap-free frame.
 
    Documented strategy:
      1. re-index to a strict hourly grid (missing hours become NaN rows)
      2. gaps <= 6 h  -> time-weighted linear interpolation (wind direction is interpolated
         as sin/cos so 350 -> 10 degrees goes through 0, not 180)
      3. longer gaps  -> mean of the same hour-of-day over PAST data only (expanding mean)
      4. anything left -> forward/back fill
      `n_imputed` counts how many of the 7 base values of that hour were imputed.
    """
    d = df.drop_duplicates("ts", keep="last").set_index("ts").sort_index()
    d = d.reindex(pd.date_range(d.index.min(), d.index.max(), freq="h", name="ts"))
    rad = np.deg2rad(d["wind_dir"].astype(float))
    d["wind_dir_sin"], d["wind_dir_cos"] = np.sin(rad), np.cos(rad)
    cols = CORE + ["wind_dir_sin", "wind_dir_cos"]
    d["n_imputed"] = d[cols[:6] + ["wind_dir_sin"]].isna().sum(axis=1)
    for c in cols:
        d[c] = d[c].astype(float).interpolate(method="time", limit=6, limit_area="inside")
        if d[c].isna().any():
            # expanding mean = only PAST observations of that hour-of-day (no future leakage)
            hod_mean = d.groupby(d.index.hour)[c].transform(lambda x: x.expanding().mean())
            d[c] = d[c].fillna(hod_mean)
        d[c] = d[c].ffill().bfill()
    d["wind_dir"] = (np.rad2deg(np.arctan2(d["wind_dir_sin"], d["wind_dir_cos"])) + 360) % 360
    d["precip_mm"] = d["precip_mm"].clip(lower=0)
    d["uv"] = d["uv"].clip(lower=0)
    d["humidity"] = d["humidity"].clip(0, 100)
    d["aqi"] = d["aqi"].clip(lower=0)
    return d.reset_index()
 
 
# --------------------------------------------------------------------------- features
def add_features(d: pd.DataFrame) -> pd.DataFrame:
    d = d.sort_values("ts").reset_index(drop=True).copy()
    d["rain"] = (d["precip_mm"] >= 0.1).astype(float)
    ts = d["ts"]
    d["hour_of_day"] = ts.dt.hour
    d["day_of_week"] = ts.dt.dayofweek
    d["month_of_year"] = ts.dt.month
    d["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)
    d["hour_sin"], d["hour_cos"] = np.sin(2 * np.pi * ts.dt.hour / 24), np.cos(2 * np.pi * ts.dt.hour / 24)
    doy = ts.dt.dayofyear
    d["doy_sin"], d["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    for v in CORE:
        s = d[v]
        d[f"{v}_lag3"], d[f"{v}_lag24"] = s.shift(3), s.shift(24)
        d[f"{v}_diff1"], d[f"{v}_diff3"] = s.diff(1), s.diff(3)          # rate of change
        d[f"{v}_roll6"] = s.rolling(6, min_periods=3).mean()            # rolling averages
        d[f"{v}_roll24"] = s.rolling(24, min_periods=12).mean()
        d[f"{v}_std24"] = s.rolling(24, min_periods=12).std()
    d["wind_dir_sin_roll6"] = d["wind_dir_sin"].rolling(6, min_periods=3).mean()
    d["wind_dir_cos_roll6"] = d["wind_dir_cos"].rolling(6, min_periods=3).mean()
    d["rain_roll6"] = d["rain"].rolling(6, min_periods=3).mean()
    d["rain_roll24"] = d["rain"].rolling(24, min_periods=12).mean()
    return d
 
 
def add_targets(d: pd.DataFrame, horizon: int = C.HORIZON) -> pd.DataFrame:
    """target_X(t) = X(t + 24h). The newest 24 rows get NaN targets -> forecasting only."""
    d = d.sort_values("ts").reset_index(drop=True).copy()
    for m in C.METRICS:
        d[f"target_{m}"] = d[m].shift(-horizon)
    return d
 
 
def build_feature_table(base: pd.DataFrame) -> pd.DataFrame:
    """base: ts + BASE_COLS (+n_sources) -> full feature table with targets."""
    return add_targets(add_features(impute(base)))
 
 
def training_frame(feats: pd.DataFrame) -> pd.DataFrame:
    """Rows with complete features AND targets (i.e. everything except newest 24 h)."""
    return feats.dropna(subset=FEATURE_COLS + TARGET_COLS).reset_index(drop=True)
 
 
def decode_wind_dir(sin_v: float, cos_v: float) -> float:
    return float((np.rad2deg(np.arctan2(sin_v, cos_v)) + 360) % 360)
