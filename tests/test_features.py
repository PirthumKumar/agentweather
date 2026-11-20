import numpy as np
import pandas as pd
 
from agentweather import config as C
from agentweather.features.engineering import (BASE_COLS, FEATURE_COLS, TARGET_COLS, build_feature_table,
                                               impute, training_frame)
from agentweather.features.fusion import fuse
from agentweather.sources.aqi import pm25_to_aqi
from agentweather.synthetic import make_raw
 
 
def _base(days=30):
    return fuse(make_raw(days, end="2026-10-01 00:00"))[["ts"] + BASE_COLS]
 
 
def test_fusion_is_mean_and_survives_missing_source():
    raw = pd.DataFrame({"ts": pd.to_datetime(["2026-01-01 00:00", "2026-01-01 01:00"]),
                        "ow_temp": [30.0, np.nan], "om_temp": [32.0, 28.0]})
    f = fuse(raw, mode="mean")
    assert f.loc[0, "temp"] == 31.0           # mean of two sources
    assert f.loc[1, "temp"] == 28.0           # one source down -> the other one is used
 
 
def test_priority_uses_open_meteo_and_falls_back():
    raw = pd.DataFrame({"ts": pd.to_datetime(["2026-01-01 00:00", "2026-01-01 01:00"]),
                        "ow_temp": [30.0, 30.0], "om_temp": [32.0, np.nan]})
    f = fuse(raw, mode="priority")
    assert f.loc[0, "temp"] == 32.0      # Open-Meteo wins
    assert f.loc[1, "temp"] == 30.0      # Open-Meteo missing -> fallback to OpenWeather
 
 
def test_wind_direction_circular_mean():
    raw = pd.DataFrame({"ts": pd.to_datetime(["2026-01-01"]), "ow_wind_dir": [350.0], "om_wind_dir": [10.0]})
    d = fuse(raw, mode="mean").loc[0, "wind_dir"]
    assert min(d, 360 - d) < 1.0              # ~0/360, NOT 180
 
 
def test_impute_gives_complete_hourly_grid():
    b = _base()
    n = len(b)
    out = impute(b)
    assert out.ts.diff().dropna().eq(pd.Timedelta(hours=1)).all()
    assert len(out) >= n                      # missing hours re-inserted
    assert not out[BASE_COLS].isna().any().any()
    assert out.n_imputed.sum() > 0
 
 
def test_targets_are_shifted_24h_and_newest_24_have_none():
    t = build_feature_table(_base())
    assert t[TARGET_COLS].tail(C.HORIZON).isna().all().all()
    assert t[TARGET_COLS].iloc[:-C.HORIZON].notna().all().all()
    assert np.allclose(t["target_temp"].iloc[:-24].values, t["temp"].iloc[24:].values)
 
 
def test_no_future_leakage_in_features():
    b = _base()
    t1 = build_feature_table(b)
    b2 = b.copy()
    b2.loc[b2.index[-30:], "temp"] += 50      # change the future
    t2 = build_feature_table(b2)
    cut = len(t1) - 60
    pd.testing.assert_frame_equal(t1.loc[:cut, FEATURE_COLS], t2.loc[:cut, FEATURE_COLS])
 
 
def test_training_frame_complete():
    tf = training_frame(build_feature_table(_base()))
    assert not tf[FEATURE_COLS + TARGET_COLS].isna().any().any() and len(tf) > 400
 
 
def test_pm25_to_aqi_breakpoints():
    assert pm25_to_aqi(0) == 0 and pm25_to_aqi(9.0) == 50 and 100 < pm25_to_aqi(40) < 150
