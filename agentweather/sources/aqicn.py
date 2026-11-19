"""AQICN / WAQI: station air quality. Prefix: aq_"""
from __future__ import annotations

import logging
import math

from .. import config as C
from .http import get_json

log = logging.getLogger(__name__)


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _km(lat1, lon1, lat2, lon2) -> float:
    """Distance in km between two coordinates (haversine formula)."""
    p = math.pi / 180
    a = (math.sin((lat2 - lat1) * p / 2) ** 2
         + math.cos(lat1 * p) * math.cos(lat2 * p) * math.sin((lon2 - lon1) * p / 2) ** 2)
    return 12742 * math.asin(math.sqrt(a))


def fetch_current() -> dict:
    if not C.AQICN_TOKEN:
        log.warning("AQICN_TOKEN not set - skipping AQICN")
        return {}
    # use the fixed station if configured, otherwise the station nearest to lat/lon
    target = C.AQICN_STATION or f"geo:{C.LAT};{C.LON}"
    j = get_json(f"https://api.waqi.info/feed/{target}/", {"token": C.AQICN_TOKEN})
    if not j or j.get("status") != "ok":
        return {}
    d = j["data"]
    # safety check: ignore a station that is far from your city
    geo = d.get("city", {}).get("geo")
    if geo and len(geo) == 2:
        km = _km(C.LAT, C.LON, float(geo[0]), float(geo[1]))
        if km > C.AQICN_MAX_KM:
            log.warning("AQICN station '%s' is %.0f km away - ignored",
                        d.get("city", {}).get("name"), km)
            return {}
    iaqi = d.get("iaqi", {})
    return {
        "aq_aqi": _num(d.get("aqi")),
        "aq_pm25": _num(iaqi.get("pm25", {}).get("v")),
        "aq_pm10": _num(iaqi.get("pm10", {}).get("v")),
    }