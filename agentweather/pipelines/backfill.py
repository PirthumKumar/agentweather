# HISTORICAL BACKFILL: run the *same* fuse -> impute -> features -> targets code over past dates.

from __future__ import annotations
 
import argparse
import logging
from datetime import datetime, timedelta, timezone
 
import pandas as pd
 
from .. import config as C
from ..features.engineering import BASE_COLS, build_feature_table
from ..features.fusion import fuse
from ..sources import openmeteo, openweather
from ..storage import get_store
 
log = logging.getLogger("backfill")
 
 
def run(days: int = 730, synthetic: bool = False) -> pd.DataFrame:
    store = get_store()
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=days)
    if synthetic:
        from ..synthetic import make_raw
        raw = make_raw(days=days)
        log.warning("USING SYNTHETIC DATA - for development only")
    else:
        raw = openmeteo.fetch_history(start, end)
        if raw.empty:
            raise RuntimeError("Open-Meteo returned no history - check network / dates")
        ow = openweather.fetch_air_history(datetime.combine(start, datetime.min.time()), datetime.now(timezone.utc).replace(tzinfo=None))
        if not ow.empty:   # extra AQI source aligned on the hour
            raw = raw.merge(ow, on="ts", how="left")
    store.append(C.FG_RAW, raw)
    base = fuse(raw)[["ts"] + BASE_COLS + ["n_sources"]]
    table = build_feature_table(base)
    table["n_sources"] = table["ts"].map(dict(zip(base.ts, base.n_sources))).fillna(0)
    store.append(C.FG_FEATURES, table)
    log.info("Backfilled %d rows: %s -> %s (%d rows with targets, %d forecast-only)", len(table),
             table.ts.min(), table.ts.max(), table.target_aqi.notna().sum(), table.target_aqi.isna().sum())
    return table
 
 
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=730)
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()
    run(a.days, a.synthetic)
