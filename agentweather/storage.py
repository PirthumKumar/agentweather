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
import time
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


def clean_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Hopsworks cannot store a column whose values are all None (dtype 'null') and mixed None/number
    object columns. Turn them into float64 (NaN) before every insert."""
    df = df.copy()
    for c in df.columns:
        if c in TIME_COLS:
            continue
        s = df[c]
        if s.dtype == object:
            if s.isna().all():
                df[c] = s.astype("float64")
            else:
                num = pd.to_numeric(s, errors="coerce")
                if num.notna().sum() == s.notna().sum():
                    df[c] = num.astype("float64")
    return df

def dedupe_latest(df: pd.DataFrame, pk: list[str]) -> pd.DataFrame:
    """Append-only tables (FG_TIME_TRAVEL=NONE) can hold the same key twice (e.g. a forecast-only row and the later
    version with its target). Keep, per key, the most complete row (most non-null values; the newest wins ties)."""
    if df.empty or not set(pk) <= set(df.columns):
        return df
    d = df.copy()
    d["_filled"] = d.notna().sum(axis=1)
    d = d.sort_values("_filled", kind="stable").drop_duplicates(subset=pk, keep="last").drop(columns="_filled")
    return d.sort_values(pk[0]).reset_index(drop=True)

INT_TYPES = {"bigint": "int64", "int": "int32", "smallint": "int16", "tinyint": "int8"}


def align_dtypes(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    """Cast columns to the types the existing Feature Group expects (bigint -> int64, int -> int32,
    double -> float64). The Feature Store fixes the type on the first insert and rejects any other."""
    df = df.copy()
    for col, typ in schema.items():
        if col not in df.columns or col in TIME_COLS:
            continue
        t = str(typ).lower()
        if t in INT_TYPES:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).round().astype(INT_TYPES[t])
        elif t in ("double", "float"):
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
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

def append(self, name: str, df: pd.DataFrame, retries: int = 3) -> None:
        if df is None or df.empty:
            return
        pk = TABLES[name][0]
        df = clean_dtypes(normalise_times(df))
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
                description=desc,
                time_travel_format=C.FG_TIME_TRAVEL,
                online_enabled=False,
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
        last = None
        for attempt in range(1, 4):       # the Hopsworks query service can time out on a cold start
            try:
                return dedupe_latest(normalise_times(fg.read()), TABLES[name][0])
            except Exception as exc:
                last = exc
                log.warning("Reading %s failed (attempt %s/3): %s", name, attempt, str(exc)[:150])
                time.sleep(10 * attempt)
        raise RuntimeError(
            f"Could not read '{name}' from Hopsworks after 3 attempts ({last}). Check your internet/VPN/firewall "
            f"(the Feature Query Service uses gRPC port 5005), try again, or look at the table in the Hopsworks UI.")

    def append(self, name: str, df: pd.DataFrame, retries: int = 3) -> None:
        if df is None or df.empty:
            return
        fg = self._fg(name, create=True)
        df = clean_dtypes(normalise_times(df))
        try:   # match the column types of the table that already exists in Hopsworks
            df = align_dtypes(df, {f.name: f.type for f in (fg.features or [])})
        except Exception as exc:
            log.warning("Could not align dtypes for %s: %s", name, exc)

        for attempt in range(1, retries + 1):
            try:
                try:
                    fg.insert(df, wait=True, write_options={"kafka_timeout": 60})  # Hopsworks >= 4.x
                except TypeError:  # older client versions
                    fg.insert(df, write_options={"wait_for_job": True, "kafka_timeout": 60})
                break
            except Exception as exc:
                # a wrong schema never fixes itself; a failed/cold materialization job often does (upserts are idempotent)
                if "not compatible" in str(exc) or attempt == 3:
                    raise
                log.warning("Insert into %s failed (attempt %s/3): %s - retrying in 30 s", name, attempt, str(exc)[:120])
                time.sleep(30)
        self._fgs.pop(name, None)


@lru_cache(maxsize=1)
def get_store():
    backend = C.STORAGE_BACKEND
    if backend == "hopsworks" or (backend == "auto" and C.HOPSWORKS_API_KEY):
        log.info("Using Hopsworks feature store")
        return HopsworksStore()
    log.info("Using LOCAL parquet store at %s", C.LOCAL_STORE)
    return LocalStore()