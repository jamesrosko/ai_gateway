"""Gateway behaviour independent of any particular backend."""

import pytest
from fastapi.testclient import TestClient

from ai_gateway.app import create_app
from ai_gateway.backends import Backend, Registry
from ai_gateway.backends.mock import MockBackend
from ai_gateway_client import GatewayClient, GatewayError


class EmbedOnly(Backend):
    async def embed(self, req):
        from ai_gateway import schemas as s

        return s.EmbedResponse(model="x", embeddings=[[0.0]] * len(req.input))


@pytest.fixture
def client():
    reg = Registry({"mock": MockBackend(), "embed-only": EmbedOnly()})
    with GatewayClient(http_client=TestClient(create_app(reg))) as c:
        yield c


def test_capabilities_detected(client):
    caps = {m["name"]: m["capabilities"] for m in client.models()}
    assert caps["embed-only"] == ["embed"]
    assert "generate" in caps["mock"]


def test_unsupported_capability_is_501(client):
    with pytest.raises(GatewayError) as e:
        client.generate("hi", model="embed-only")
    assert e.value.status == 501


def test_default_model_is_first_capable(client):
    assert client.embed("x").model == "mock"
    assert client.embed("x", model="embed-only").model == "embed-only"


def test_stream_deltas_join_to_unary_text(client):
    streamed = "".join(c.delta for c in client.generate_stream("one two three", model="mock"))
    assert streamed == client.generate("one two three", model="mock").text
