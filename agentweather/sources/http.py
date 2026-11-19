"""Tiny HTTP helper with retries - API downtime must never crash the pipeline."""
from __future__ import annotations
 
import logging
import time
from typing import Any, Optional
 
import requests
 
log = logging.getLogger(__name__)
 
 
def get_json(url: str, params: Optional[dict] = None, retries: int = 3, timeout: int = 30) -> Optional[Any]:
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            log.warning("GET %s -> HTTP %s (attempt %s)", url.split("?")[0], r.status_code, attempt)
        except requests.RequestException as exc:
            log.warning("GET %s failed: %s (attempt %s)", url.split("?")[0], exc, attempt)
        time.sleep(2 * attempt)
    return None
