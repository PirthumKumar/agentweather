"""Synthetic weather generator - ONLY for development and unit tests (never for the submission!).
 
Lets you run the complete pipeline before API keys exist, and lets the tests run offline.
Produces *raw source columns* (om_*, ow_*, aq_*) so fusion/imputation are exercised too.
"""
from __future__ import annotations
 
import numpy as np
import pandas as pd
 
 
def make_raw(days: int = 60, end: str | None = None, seed: int = 0, drift_after: int | None = None,
             gaps: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    end_ts = pd.Timestamp(end) if end else pd.Timestamp.now(tz="UTC").tz_localize(None).floor("h")
    idx = pd.date_range(end=end_ts, periods=days * 24, freq="h")
    n = len(idx)
    h = idx.hour.values
    temp = 27 + 4 * np.sin(2 * np.pi * (h - 9) / 24) + rng.normal(0, 0.6, n).cumsum() * 0.05 + rng.normal(0, 0.4, n)
    hum = np.clip(80 - 3.2 * (temp - 27) + rng.normal(0, 4, n), 35, 100)
    wet = (rng.random(n // 6 + 1).repeat(6)[:n] < 0.18) & (h > 12) & (h < 23)
    precip = np.where(wet, rng.exponential(2.5, n), 0.0)
    uv = np.clip(9 * np.sin(np.pi * (h - 6) / 12), 0, None) * np.where(wet, 0.4, 1) * rng.uniform(0.85, 1.05, n)
    aqi = np.empty(n)
    aqi[0] = 70
    for i in range(1, n):
        aqi[i] = max(10, 0.96 * aqi[i - 1] + 3 + 6 * np.sin(2 * np.pi * (h[i] - 8) / 24) + rng.normal(0, 4))
    wspd = np.clip(2.5 + 1.2 * np.sin(2 * np.pi * (h - 14) / 24) + rng.gamma(2, 0.6, n) - 1, 0, None)
    wdir = (180 + 70 * np.sin(2 * np.pi * h / 24) + rng.normal(0, 25, n)) % 360
    if drift_after is not None:  # simulate sensor/climate drift in the last `drift_after` hours
        k = drift_after
        temp[-k:] += 4
        aqi[-k:] *= 1.8
    df = pd.DataFrame({"ts": idx, "om_temp": temp, "om_humidity": hum, "om_precip": precip, "om_uv": uv,
                       "om_aqi": aqi, "om_wind_speed": wspd, "om_wind_dir": wdir})
    df["ow_temp"] = temp + rng.normal(0, 0.5, n)
    df["ow_humidity"] = np.clip(hum + rng.normal(0, 2, n), 0, 100)
    df["ow_wind_speed"] = np.clip(wspd + rng.normal(0, 0.3, n), 0, None)
    df["ow_wind_dir"] = (wdir + rng.normal(0, 8, n)) % 360
    df["ow_precip"] = np.clip(precip + rng.normal(0, 0.1, n), 0, None)
    df["aq_aqi"] = aqi + rng.normal(0, 3, n)
    df["ow_aqi"] = aqi + rng.normal(0, 3, n)
    if gaps:  # API downtime
        for _ in range(4):
            s = rng.integers(30, n - 30)
            df.iloc[s: s + rng.integers(1, 5), df.columns.get_indexer(["ow_temp", "ow_humidity", "aq_aqi"])] = np.nan
        s = rng.integers(100, n - 100)
        df = df.drop(df.index[s: s + 10])  # whole hours missing (pipeline did not run)
    return df.reset_index(drop=True)
