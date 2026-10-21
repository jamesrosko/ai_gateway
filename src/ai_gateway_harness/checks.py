"""Backend-agnostic contract checks.

Each check exercises one endpoint through the client and asserts on the *shape* of the
result (types, invariants, usage accounting), never on model-specific content, so the same
checks run against any backend you wire in. Add your own with the @check decorator, or pass
extra `Check`s to `run_checks`.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass

from ai_gateway import schemas as s
from ai_gateway_client import GatewayClient, GatewayError


@dataclass(frozen=True)
class Check:
    name: str
    capability: str
    fn: Callable[[GatewayClient, str], None]
    models: tuple[str, ...] = ("*",)  # fnmatch patterns of model names this check applies to


CHECKS: list[Check] = []


def check(capability: str):
    def deco(fn):
        CHECKS.append(Check(fn.__name__, capability, fn))
        return fn

    return deco


def _usage_ok(u: s.Usage) -> None:
    assert u.total_tokens >= u.prompt_tokens + u.completion_tokens or u.total_tokens == 0, (
        f"inconsistent usage: {u}"
    )


# -- generate -------------------------------------------------------------------


@check("generate")
def generate_basic(c: GatewayClient, model: str):
    r = c.generate("Say hello.", model=model)
    assert r.model == model
    assert isinstance(r.text, str) and r.text
    _usage_ok(r.usage)


@check("generate")
def generate_system_and_history(c: GatewayClient, model: str):
    r = c.generate(
        [
            {"role": "system", "content": "Be terse."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello."},
            {"role": "user", "content": "Bye"},
        ],
        model=model,
    )
    assert r.text


@check("generate")
def generate_stream_matches_shape(c: GatewayClient, model: str):
    chunks = list(c.generate_stream("Count to three.", model=model))
    assert chunks, "stream produced no events"
    assert "".join(ch.delta for ch in chunks), "stream produced no text"
    last = chunks[-1]
    assert last.finish_reason, "final chunk missing finish_reason"
    assert last.usage is not None, "final chunk missing usage"
    _usage_ok(last.usage)


@check("generate")
def generate_rejects_empty_messages(c: GatewayClient, model: str):
    try:
        c.generate([], model=model)
    except GatewayError as e:
        assert e.status == 422
    else:
        raise AssertionError("expected 422 for empty messages")


# -- embed ----------------------------------------------------------------------


@check("embed")
def embed_batch(c: GatewayClient, model: str):
    r = c.embed(["alpha", "beta", "gamma"], model=model)
    assert len(r.embeddings) == 3
    dims = {len(v) for v in r.embeddings}
    assert len(dims) == 1 and dims.pop() > 0, "embeddings must share a non-zero dimension"
    assert all(math.isfinite(x) for v in r.embeddings for x in v)


@check("embed")
def embed_deterministic(c: GatewayClient, model: str):
    a = c.embed("same text", model=model).embeddings[0]
    b = c.embed("same text", model=model).embeddings[0]
    assert len(a) == len(b)
    assert max(abs(x - y) for x, y in zip(a, b)) < 1e-3, "same input gave different vectors"


# -- classify -------------------------------------------------------------------


@check("classify")
def classify_text(c: GatewayClient, model: str):
    r = c.classify("I love this product", model=model)
    assert r.labels, "no labels returned"
    scores = [x.score for x in r.labels]
    assert all(math.isfinite(x) for x in scores)
    assert scores == sorted(scores, reverse=True), "labels must be sorted by score desc"


@check("classify")
def classify_top_k(c: GatewayClient, model: str):
    r = c.classify("some text", top_k=1, model=model)
    assert len(r.labels) == 1


@check("classify")
def classify_requires_input(c: GatewayClient, model: str):
    try:
        c.classify(model=model)
    except GatewayError as e:
        assert e.status == 422
    else:
        raise AssertionError("expected 422 when neither text nor features given")


# -- extract --------------------------------------------------------------------


@check("extract")
def extract_spans_are_consistent(c: GatewayClient, model: str):
    doc = "Contact bob@example.com about invoice 42."
    r = c.extract(doc, model=model)
    for e in r.entities:
        assert 0 <= e.start < e.end <= len(doc), f"bad span {e}"
        assert doc[e.start : e.end] == e.text, f"span/text mismatch {e}"


# -- decide ---------------------------------------------------------------------


@check("decide")
def decide_shape(c: GatewayClient, model: str):
    r = c.decide({"amount": 120, "country": "US"}, model=model)
    assert r.decision
    assert math.isfinite(r.score)
    assert isinstance(r.reasons, list)


@check("decide")
def decide_respects_options(c: GatewayClient, model: str):
    opts = ["approve", "review", "reject"]
    r = c.decide({"amount": 5}, options=opts, model=model)
    assert r.decision in opts, f"{r.decision!r} not in {opts}"


# -- predict --------------------------------------------------------------------


@check("predict")
def predict_tensor_shapes(c: GatewayClient, model: str):
    t = s.Tensor(name="x", shape=[2, 2], datatype="FP32", data=[1.0, 2.0, 3.0, 4.0])
    r = c.predict([t], model=model)
    assert r.outputs, "no outputs"
    for o in r.outputs:
        n = math.prod(o.shape)
        assert len(o.data) == n, f"output {o.name}: shape {o.shape} != {len(o.data)} elements"


# -- unknown model (capability-independent) -------------------------------------


def unknown_model_is_404(c: GatewayClient, model: str):
    try:
        c.embed("x", model="__does_not_exist__")
    except GatewayError as e:
        assert e.status == 404
    else:
        raise AssertionError("expected 404")


CHECKS.append(Check("unknown_model_is_404", "*", unknown_model_is_404))
