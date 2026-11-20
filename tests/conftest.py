import sys
from pathlib import Path
 
import pytest
 
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
 
 
@pytest.fixture()
def env(tmp_path, monkeypatch):
    """Isolated local store + registry, no API keys."""
    from agentweather import config as C
    from agentweather.storage import get_store
 
    monkeypatch.setattr(C, "LOCAL_STORE", tmp_path / "store")
    monkeypatch.setattr(C, "LOCAL_REGISTRY", tmp_path / "reg")
    monkeypatch.setattr(C, "MODEL_CACHE", tmp_path / "cache")
    monkeypatch.setattr(C, "GROQ_API_KEY", "")
    monkeypatch.setattr(C, "STORAGE_BACKEND", "local")
    get_store.cache_clear()
    yield C
    get_store.cache_clear()
 
 
@pytest.fixture()
def backfilled(env):
    from agentweather.pipelines import backfill
 
    backfill.run(days=45, synthetic=True)
    return env
