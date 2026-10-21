"""Golden cases: JSON files of {request, expect} pairs checked against the gateway.

Layout: one file per endpoint, e.g. golden/classify.json:

    {"capability": "classify",
     "cases": [{"id": "sentiment-pos",
                "models": ["llm*"],              # fnmatch on model name; default ["*"]
                "request": {...},                # JSON body for POST /v1/classify ("model" is injected)
                "expect": {"top_label_in": ["positive"]}}]}

Because real models are non-deterministic, expectations are assertions about the answer
(membership, containment, thresholds), not exact text. Exact `equals` is available for
deterministic backends.
"""

import json
import math
import re
from collections.abc import Callable
from pathlib import Path

from ai_gateway_client import GatewayClient

from .checks import Check

DEFAULT_DIR = Path("golden")


def _cos(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _generate(r, e):
    text = r.text.lower()
    for sub in e.get("contains", []):
        assert sub.lower() in text, f"expected {sub!r} in {r.text!r}"
    for sub in e.get("not_contains", []):
        assert sub.lower() not in text, f"unexpected {sub!r} in {r.text!r}"
    if "regex" in e:
        assert re.search(e["regex"], r.text, re.S), f"{e['regex']!r} did not match {r.text!r}"
    if "equals" in e:
        assert r.text == e["equals"], f"{r.text!r} != {e['equals']!r}"
    if "max_completion_tokens" in e:
        assert r.usage.completion_tokens <= e["max_completion_tokens"], r.usage


def _embed(r, e):
    if "dim" in e:
        assert all(len(v) == e["dim"] for v in r.embeddings), f"dims {[len(v) for v in r.embeddings]}"
    if "count" in e:
        assert len(r.embeddings) == e["count"]
    if "closer" in e:  # indices into the request's input list
        a, near, far = (e["closer"][k] for k in ("anchor", "near", "far"))
        sn, sf = _cos(r.embeddings[a], r.embeddings[near]), _cos(r.embeddings[a], r.embeddings[far])
        assert sn > sf, f"cos(anchor,near)={sn:.3f} should exceed cos(anchor,far)={sf:.3f}"


def _classify(r, e):
    top = r.labels[0]
    if "top_label_in" in e:
        assert top.label in e["top_label_in"], f"top label {top.label!r} not in {e['top_label_in']}"
    if "min_top_score" in e:
        assert top.score >= e["min_top_score"], f"top score {top.score} < {e['min_top_score']}"
    if "labels_subset_of" in e:
        extra = {x.label for x in r.labels} - set(e["labels_subset_of"])
        assert not extra, f"unexpected labels {extra}"


def _extract(r, e):
    for want in e.get("include", []):
        assert any(
            x.type == want["type"] and x.text == want["text"] for x in r.entities
        ), f"missing {want} in {[(x.type, x.text) for x in r.entities]}"
    for banned in e.get("exclude_types", []):
        assert not any(x.type == banned for x in r.entities), f"unexpected type {banned}"
    if "count" in e:
        assert len(r.entities) == e["count"], f"{len(r.entities)} entities != {e['count']}"


def _decide(r, e):
    if "decision_in" in e:
        assert r.decision in e["decision_in"], f"decision {r.decision!r} not in {e['decision_in']}"
    if "min_score" in e:
        assert r.score >= e["min_score"], f"score {r.score} < {e['min_score']}"
    if "min_reasons" in e:
        assert len(r.reasons) >= e["min_reasons"]


def _predict(r, e):
    got = {o.name: o for o in r.outputs}
    for name, want in e.get("outputs", {}).items():
        assert name in got, f"missing output {name!r}"
        if "shape" in want:
            assert got[name].shape == want["shape"], f"{got[name].shape} != {want['shape']}"
        if "data" in want:
            tol = want.get("tolerance", 1e-6)
            assert len(got[name].data) == len(want["data"])
            for g, w in zip(got[name].data, want["data"]):
                assert abs(g - w) <= tol if isinstance(w, (int, float)) else g == w, f"{g} != {w}"


VERIFIERS: dict[str, Callable] = {
    "generate": _generate,
    "embed": _embed,
    "classify": _classify,
    "extract": _extract,
    "decide": _decide,
    "predict": _predict,
}


def _make_check(capability: str, case: dict) -> Check:
    def fn(client: GatewayClient, model: str) -> None:
        resp = client.call(capability, {**case["request"], "model": model})
        VERIFIERS[capability](resp, case["expect"])

    return Check(
        name=f"golden:{capability}:{case['id']}",
        capability=capability,
        fn=fn,
        models=tuple(case.get("models", ["*"])),
    )


def load_checks(directory: str | Path = DEFAULT_DIR) -> list[Check]:
    checks = []
    for path in sorted(Path(directory).glob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        cap = doc["capability"]
        if cap not in VERIFIERS:
            raise ValueError(f"{path}: unknown capability {cap!r}")
        checks.extend(_make_check(cap, c) for c in doc["cases"])
    return checks
