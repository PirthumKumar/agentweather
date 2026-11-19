"""Convert PM2.5 (ug/m3) to the US EPA AQI so that all sources share one scale."""
from __future__ import annotations
 
import math
 
# (conc_low, conc_high, aqi_low, aqi_high)  - EPA 2024 PM2.5 breakpoints
_PM25 = [
    (0.0, 9.0, 0, 50), (9.1, 35.4, 51, 100), (35.5, 55.4, 101, 150),
    (55.5, 125.4, 151, 200), (125.5, 225.4, 201, 300), (225.5, 325.4, 301, 500),
]
 
 
def pm25_to_aqi(c: float | None) -> float:
    if c is None or (isinstance(c, float) and math.isnan(c)) or c < 0:
        return float("nan")
    c = math.floor(c * 10) / 10
    for lo, hi, alo, ahi in _PM25:
        if lo <= c <= hi:
            return round((ahi - alo) / (hi - lo) * (c - lo) + alo)
    return 500.0
