"""Model registry with the same interface for Hopsworks and a local folder.
 
A *registered model* = one 24h-ahead target (e.g. ``aw_aqi``). Each version is the
champion at the time it was promoted, together with its validation metrics.
The training pipeline promotes a new version ONLY when it beats the current champion
on the current validation window (see training/train.py).
"""
from __future__ import annotations
 
import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Optional
 
import joblib
 
from . import config as C
 
log = logging.getLogger(__name__)
ARTIFACT = "model.joblib"
 
 
def reg_name(metric: str) -> str:
    return f"aw_{metric}"
 
 
class LocalRegistry:
    name = "local"
 
    def __init__(self) -> None:
        C.LOCAL_REGISTRY.mkdir(parents=True, exist_ok=True)
 
    def _dir(self, metric: str) -> Path:
        return C.LOCAL_REGISTRY / reg_name(metric)
 
    def latest_version(self, metric: str) -> Optional[int]:
        d = self._dir(metric)
        if not d.exists():
            return None
        vs = [int(p.name[1:]) for p in d.glob("v*") if p.is_dir()]
        return max(vs) if vs else None
 
    def get_champion(self, metric: str):
        """Return (model_object, info_dict) or (None, None)."""
        v = self.latest_version(metric)
        if v is None:
            return None, None
        d = self._dir(metric) / f"v{v}"
        info = json.loads((d / "info.json").read_text())
        info["version"] = v
        return joblib.load(d / ARTIFACT), info
 
    def register(self, metric: str, model, info: dict) -> int:
        v = (self.latest_version(metric) or 0) + 1
        d = self._dir(metric) / f"v{v}"
        d.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, d / ARTIFACT, compress=3)
        (d / "info.json").write_text(json.dumps(info, default=str))
        return v
 
 
class HopsworksRegistry:
    name = "hopsworks"
 
    def __init__(self, project) -> None:
        self.mr = project.get_model_registry()
        C.MODEL_CACHE.mkdir(parents=True, exist_ok=True)
 
    def _latest(self, metric: str):
        try:
            models = self.mr.get_models(reg_name(metric))
        except Exception:
            return None
        return max(models, key=lambda m: m.version) if models else None
 
    def get_champion(self, metric: str):
        m = self._latest(metric)
        if m is None:
            return None, None
        cache = C.MODEL_CACHE / reg_name(metric) / f"v{m.version}"
        if not (cache / ARTIFACT).exists():
            src = Path(m.download())
            cache.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src, cache, dirs_exist_ok=True)
        info = dict(m.training_metrics or {})
        info.update(json.loads((cache / "info.json").read_text()) if (cache / "info.json").exists() else {})
        info["version"] = m.version
        return joblib.load(cache / ARTIFACT), info
 
    def register(self, metric: str, model, info: dict) -> int:
        metrics = {k: float(v) for k, v in info.items() if isinstance(v, (int, float)) and v == v}
        with tempfile.TemporaryDirectory() as tmp:
            joblib.dump(model, Path(tmp) / ARTIFACT, compress=3)
            (Path(tmp) / "info.json").write_text(json.dumps(info, default=str))
            m = self.mr.python.create_model(
                name=reg_name(metric), metrics=metrics,
                description=f"Champion 24h-ahead model for {metric}: {info.get('family')}",
            )
            m.save(tmp)
        return m.version
 
 
def get_registry():
    from .storage import HopsworksStore, get_store
 
    store = get_store()
    if isinstance(store, HopsworksStore):
        return HopsworksRegistry(store.project)
    return LocalRegistry()
