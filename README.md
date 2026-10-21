# ai-gateway

A unified inference gateway, a typed Python client, and a contract-test harness.

| Endpoint | In → Out |
|---|---|
| `POST /v1/generate` | messages → text (SSE streaming with `"stream": true`, token usage) |
| `POST /v1/embed` | texts → vectors |
| `POST /v1/classify` | text or features → `[{label, score}]` |
| `POST /v1/extract` | document → typed entities/spans |
| `POST /v1/decide` | structured facts → `{decision, score, reasons}` |
| `POST /v1/predict` | named tensors (V2-style) → named tensors |
| `GET /v1/models` | registered models and their capabilities |

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

## Run

```powershell
ai-gateway --port 8000        # reads gateway.toml (or $AI_GATEWAY_CONFIG)
```

## Wiring in a backend

Subclass `ai_gateway.backends.Backend` and override only the endpoints the model supports
(see `backends/mock.py` for a full example). Unimplemented endpoints return HTTP 501 and are
skipped by the harness automatically. Register it in `gateway.toml`:

```toml
[models.my-llm]
backend = "my_pkg.backends:MyLLM"
options = { base_url = "http://localhost:11434" }
```

## Client

```python
from ai_gateway_client import GatewayClient

with GatewayClient("http://127.0.0.1:8000") as c:
    print(c.generate("Hello", model="mock").text)
    for chunk in c.generate_stream("Hello", model="mock"):
        print(chunk.delta, end="")
    print(c.embed(["a", "b"]).embeddings)
    print(c.decide({"amount": 120}, options=["approve", "reject"]))
```

## Test harness

Contract checks (`ai_gateway_harness/checks.py`) assert response *shape and invariants*
(sorted scores, span/text consistency, usage accounting, 422/404 handling), not model content,
so they run unchanged against any backend. For each model, only checks matching its
advertised capabilities run.

```powershell
pytest                                            # in-process, models from gateway.toml
pytest --gateway-url http://host:8000             # against a live gateway
pytest --gateway-url ... --gateway-model my-llm   # one model
gateway-harness --url http://host:8000 --capability embed --json   # standalone CLI
```

Add checks with `@check("<capability>")` in `checks.py`, or in your own module.
