"""OpenAICompatBackend against a fake upstream (no network / model required)."""

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from ai_gateway.app import create_app
from ai_gateway.backends import Registry
from ai_gateway.backends.openai_compat import OpenAICompatBackend
from ai_gateway_client import GatewayClient, GatewayError


def chat(content: str) -> dict:
    return {
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }


def make_client(handler, **opts) -> GatewayClient:
    http = httpx.AsyncClient(base_url="http://up/v1", transport=httpx.MockTransport(handler))
    backend = OpenAICompatBackend(http_client=http, **opts)
    return GatewayClient(http_client=TestClient(create_app(Registry({"llm": backend}))))


def test_generate_maps_usage_and_text():
    c = make_client(lambda req: httpx.Response(200, json=chat("Paris")))
    r = c.generate("capital of France?", model="llm")
    assert r.text == "Paris" and r.usage.total_tokens == 5


def test_stream_emits_deltas_then_usage():
    events = [
        {"choices": [{"delta": {"content": "Hel"}}]},
        {"choices": [{"delta": {"content": "lo"}, "finish_reason": "stop"}]},
        {"choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}},
    ]
    sse = "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"
    c = make_client(lambda req: httpx.Response(200, text=sse))
    chunks = list(c.generate_stream("hi", model="llm"))
    assert "".join(x.delta for x in chunks) == "Hello"
    assert chunks[-1].finish_reason == "stop" and chunks[-1].usage.total_tokens == 3


def test_capabilities_without_embed_model():
    c = make_client(lambda req: httpx.Response(200, json=chat("x")))
    caps = {m["name"]: m["capabilities"] for m in c.models()}["llm"]
    assert "embed" not in caps and "predict" not in caps and "classify" in caps


def test_embed_orders_by_index():
    body = {"data": [{"index": 1, "embedding": [2.0]}, {"index": 0, "embedding": [1.0]}]}
    c = make_client(lambda req: httpx.Response(200, json=body), embed_model="e")
    assert c.embed(["a", "b"], model="llm").embeddings == [[1.0], [2.0]]


def test_classify_filters_clamps_and_sorts():
    out = {"labels": [{"label": "b", "score": 0.2}, {"label": "zzz", "score": 1}, {"label": "a", "score": 7}]}
    c = make_client(lambda req: httpx.Response(200, json=chat(json.dumps(out))))
    r = c.classify("t", labels=["a", "b"], model="llm")
    assert [(x.label, x.score) for x in r.labels] == [("a", 1.0), ("b", 0.2)]


def test_extract_computes_offsets_and_drops_hallucinations():
    out = {"entities": [{"type": "PERSON", "text": "Bob"}, {"type": "PERSON", "text": "Alice"}]}
    fenced = "```json\n" + json.dumps(out) + "\n```"
    c = make_client(lambda req: httpx.Response(200, json=chat(fenced)))
    r = c.extract("Hi Bob!", entity_types=["PERSON"], model="llm")
    assert [(e.text, e.start, e.end) for e in r.entities] == [("Bob", 3, 6)]


def test_decide_outside_options_is_502():
    out = {"decision": "maybe", "score": 0.5, "reasons": ["x"]}
    c = make_client(lambda req: httpx.Response(200, json=chat(json.dumps(out))))
    with pytest.raises(GatewayError) as e:
        c.decide({"a": 1}, options=["yes", "no"], model="llm")
    assert e.value.status == 502


def test_upstream_failure_is_502():
    c = make_client(lambda req: httpx.Response(500, text="boom"))
    with pytest.raises(GatewayError) as e:
        c.generate("hi", model="llm")
    assert e.value.status == 502
