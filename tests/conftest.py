"""Fixtures. By default tests run in-process against the registry in gateway.toml.

Point them at a live gateway instead with:  pytest --gateway-url http://host:8000
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ai_gateway.app import create_app
from ai_gateway.backends import load_registry
from ai_gateway_client import GatewayClient
from ai_gateway_harness import CHECKS, plan
from ai_gateway_harness.golden import load_checks

GOLDEN_DIR = Path(__file__).resolve().parent.parent / "golden"


def pytest_addoption(parser):
    parser.addoption("--gateway-url", default=None, help="test a running gateway, not in-process")
    parser.addoption("--gateway-api-key", default=None)
    parser.addoption("--gateway-model", action="append", default=None, help="limit to model(s)")


@pytest.fixture(scope="session")
def client(request):
    url = request.config.getoption("--gateway-url")
    if url:
        c = GatewayClient(url, api_key=request.config.getoption("--gateway-api-key"))
    else:
        c = GatewayClient(http_client=TestClient(create_app(load_registry())))
    yield c
    c.close()


def pytest_generate_tests(metafunc):
    """Parametrize test_contract over (model, check) pairs the gateway actually supports."""
    if {"model", "chk"} <= set(metafunc.fixturenames):
        url = metafunc.config.getoption("--gateway-url")
        if url:
            c = GatewayClient(url, api_key=metafunc.config.getoption("--gateway-api-key"))
        else:
            c = GatewayClient(http_client=TestClient(create_app(load_registry())))
        pairs = plan(
            c,
            models=metafunc.config.getoption("--gateway-model"),
            checks=[*CHECKS, *load_checks(GOLDEN_DIR)],
        )
        metafunc.parametrize(
            "model,chk", pairs, ids=[f"{m}-{chk.name}" for m, chk in pairs]
        )
