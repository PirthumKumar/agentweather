# HOURLY feature pipeline run by GitHub Actions
from __future__ import annotations
 
import logging
 
import numpy as np
import pandas as pd

 
from .. import config as C
from ..features.engineering import BASE_COLS, build_feature_table
from ..features.fusion import conform_raw, fuse
from ..sources import aqicn, openmeteo, openweather
from ..storage import get_store
 
log = logging.getLogger("feature_pipeline")
HISTORY_HOURS = 24 * 8   # enough context for lag_24 / roll_24 and for filling targets of older rows
 
 
def fetch_live_raw(now: pd.Timestamp) -> pd.DataFrame:
    """One raw row for the current hour; sources that fail are simply left NaN."""
    row: dict = {"ts": now}
    row.update(openweather.fetch_current())
    row.update(aqicn.fetch_current())
    om = openmeteo.fetch_recent(past_days=1)
    if not om.empty:
        cur = om[om.ts == now]
        if not cur.empty:
            row.update({k: v for k, v in cur.iloc[-1].to_dict().items() if k != "ts"})
    return pd.DataFrame([row])
 
 
def run(now: pd.Timestamp | None = None, raw_override: pd.DataFrame | None = None) -> pd.DataFrame:
    store = get_store()
    now = now or pd.Timestamp.now(tz="UTC").tz_localize(None).floor("h")
    raw_new = conform_raw(raw_override if raw_override is not None else fetch_live_raw(now))   
    n_src = int(raw_new.drop(columns="ts").notna().sum(axis=1).iloc[0])
    log.info("Fetched %d raw values for %s", n_src, now)
    if C.STORE_RAW:   # raw table = EDA / source cross-checks only; it must never stop the feature rows
        try:
            store.append(C.FG_RAW, raw_new, retries=1)
        except Exception as exc:
            log.error("Could not store raw observations (continuing with features): %s", str(exc)[:200])
 
    # history = what is already in the feature store base columns only + the new fused row
    hist = store.read(C.FG_FEATURES)
    new_base = fuse(raw_new)[["ts"] + BASE_COLS + ["n_sources"]]
    if hist.empty:
        base = new_base
    else:
        hist = hist.sort_values("ts").tail(HISTORY_HOURS)
        base = pd.concat([hist[["ts"] + BASE_COLS], new_base], ignore_index=True)
        base = base.drop_duplicates("ts", keep="last")
    if new_base[BASE_COLS].isna().all(axis=1).iloc[0]:
        log.error("ALL sources failed this hour - row will be imputed from history")
    table = build_feature_table(base)
    # upsert the last 24+1 rows: the new row AND the older rows whose 24h-ahead target just became known
    out = table.tail(C.HORIZON + 1).copy()
    out["n_sources"] = out["ts"].map(dict(zip(base.ts, base.get("n_sources", np.nan)))).fillna(0).astype("int64")
    store.append(C.FG_FEATURES, out)
    log.info("Upserted %d feature rows (latest %s)", len(out), out.ts.max())
    return out
 
 
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run()
