"""Storage layer.
 
One small interface (`Store.read` / `Store.append`) with two back-ends:
 
* Hopsworks Feature Store  - used automatically when HOPSWORKS_API_KEY is set
* Local parquet folder     - used otherwise (lets you develop & test with no account)
 
Every table in the project (raw data, features, ledger, logs ...) is a Hopsworks
Feature Group, so "store all logs in the Feature Store" is satisfied by construction.
All timestamps are stored as timezone-naive UTC.
"""
from __future__ import annotations
 
import logging
from functools import lru_cache
from typing import Optional
 
import pandas as pd
 
from . import config as C
 
log = logging.getLogger(__name__)
 
# name -> (primary key, event-time column, description)
TABLES: dict[str, tuple[list[str], Optional[str], str]] = {
    C.FG_RAW: (["ts"], "ts", "Raw observations from every API source, one row per hour"),
    C.FG_FEATURES: (["ts"], "ts", "Fused, imputed, engineered features and 24h-ahead targets"),
    C.FG_LEDGER: (["forecast_id"], "issued_at", "Forecast Ledger: every model forecast"),
    C.FG_AGENT_STATE: (["state_id"], "saved_at", "Shared LangGraph state saved between runs"),
    C.FG_ALERTS: (["alert_id"], "issued_at", "Alerts written by the Alert Agent"),
    C.FG_REPORTS: (["report_id"], "generated_at", "News-style weather reports"),
    C.FG_LLM_CALLS: (["call_id"], "logged_at", "Every LLM call: prompt, response, latency, status"),
    C.FG_FAILURES: (["failure_id"], "logged_at", "Failed LLM calls / agent errors"),
    C.FG_DRIFT: (["drift_id"], "logged_at", "Drift events"),
    C.FG_MODEL_METRICS: (["run_id"], "trained_at", "Validation metrics of every candidate model per training run"),
    C.FG_DRIFT_REF: (["ref_id"], "trained_at", "Training-set reference distribution used for data drift"),
    C.FG_EXPLAIN: (["explain_id"], "trained_at", "SHAP / LIME explanations"),
}
 
TIME_COLS = {"ts", "issued_at", "target_ts", "saved_at", "generated_at", "logged_at", "trained_at"}
 
 
def normalise_times(df: pd.DataFrame) -> pd.DataFrame:
    """Make every known time column timezone-naive UTC datetime64[ns]."""
    df = df.copy()
    for c in df.columns:
        if c in TIME_COLS:
            s = pd.to_datetime(df[c], utc=True).dt.tz_localize(None)
            df[c] = s.astype("datetime64[ns]")
    return df
 
 
class LocalStore:
    name = "local"
 
    def __init__(self) -> None:
        C.LOCAL_STORE.mkdir(parents=True, exist_ok=True)
 
    def _path(self, name: str):
        return C.LOCAL_STORE / f"{name}.parquet"
 
    def read(self, name: str) -> pd.DataFrame:
        p = self._path(name)
        return pd.read_parquet(p) if p.exists() else pd.DataFrame()
 
    def append(self, name: str, df: pd.DataFrame) -> None:
        if df is None or df.empty:
            return
        pk = TABLES[name][0]
        df = normalise_times(df)
        old = self.read(name)
        out = pd.concat([old, df], ignore_index=True) if not old.empty else df
        out = out.drop_duplicates(subset=pk, keep="last")
        sort_col = TABLES[name][1]
        if sort_col in out.columns:
            out = out.sort_values(sort_col)
        out.reset_index(drop=True).to_parquet(self._path(name), index=False)
 
 
class HopsworksStore:
    name = "hopsworks"
 
    def __init__(self) -> None:
        import hopsworks  # imported lazily so the dashboard/test envs stay light
 
        kwargs = {"api_key_value": C.HOPSWORKS_API_KEY}
        if C.HOPSWORKS_PROJECT:
            kwargs["project"] = C.HOPSWORKS_PROJECT
        if C.HOPSWORKS_HOST:
            kwargs["host"] = C.HOPSWORKS_HOST
        self.project = hopsworks.login(**kwargs)
        self.fs = self.project.get_feature_store()
        self._fgs: dict = {}
 
    def _fg(self, name: str, create: bool = False):
        if name in self._fgs:
            return self._fgs[name]
        pk, et, desc = TABLES[name]
        if create:
            fg = self.fs.get_or_create_feature_group(
                name=name, version=C.FG_VERSION, primary_key=pk, event_time=et,
                description=desc , time_travel_format="HUDI" , online_enabled=False, 
            )
        else:
            try:
                fg = self.fs.get_feature_group(name, version=C.FG_VERSION)
            except Exception:
                fg = None
        if fg is not None:
            self._fgs[name] = fg
        return fg
 
    def read(self, name: str) -> pd.DataFrame:
        fg = self._fg(name)
        if fg is None:
            return pd.DataFrame()
        try:
            return normalise_times(fg.read())
        except Exception as exc:  # empty FG, schema not materialised yet ...
            log.warning("Could not read %s: %s", name, exc)
            return pd.DataFrame()
 
    def append(self, name: str, df: pd.DataFrame) -> None:
        if df is None or df.empty:
            return
        fg = self._fg(name, create=True)
        df = normalise_times(df)
        try:
            fg.insert(df, wait=True , write_options={"kafka_timeout": 60})  # Hopsworks >= 4.x
        except TypeError:  # older client versions
            fg.insert(df, write_options={"wait_for_job": True})
        self._fgs.pop(name, None)
 
 
@lru_cache(maxsize=1)
def get_store():
    backend = C.STORAGE_BACKEND
    if backend == "hopsworks" or (backend == "auto" and C.HOPSWORKS_API_KEY):
        log.info("Using Hopsworks feature store")
        return HopsworksStore()
    log.info("Using LOCAL parquet store at %s", C.LOCAL_STORE)
    return LocalStore()
