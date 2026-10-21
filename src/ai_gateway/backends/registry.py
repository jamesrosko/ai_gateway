import importlib
import os
import tomllib
from pathlib import Path

from .base import Backend


class Registry:
    def __init__(self, models: dict[str, Backend] | None = None):
        self.models: dict[str, Backend] = dict(models or {})

    def register(self, name: str, backend: Backend) -> None:
        self.models[name] = backend

    def resolve(self, capability: str, model: str | None) -> tuple[str, Backend]:
        """Return (name, backend) for a request. Raises KeyError / LookupError on failure."""
        if model is not None:
            return model, self.models[model]
        for name, backend in self.models.items():
            if capability in backend.capabilities:
                return name, backend
        raise LookupError(f"no model registered for '{capability}'")


def _import(path: str):
    module, _, attr = path.partition(":")
    return getattr(importlib.import_module(module), attr)


def load_registry(path: str | os.PathLike | None = None) -> Registry:
    path = Path(path or os.environ.get("AI_GATEWAY_CONFIG", "gateway.toml"))
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    reg = Registry()
    for name, spec in cfg.get("models", {}).items():
        reg.register(name, _import(spec["backend"])(**spec.get("options", {})))
    return reg
