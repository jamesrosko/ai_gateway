"""Backend for any OpenAI-compatible server (Ollama, vLLM, LM Studio, OpenAI, ...).

Implements:
  generate (+ true streaming)  -> /chat/completions
  embed                        -> /embeddings          (only if `embed_model` is set)
  classify / extract / decide  -> /chat/completions with a JSON-only prompt, temperature 0

Not implemented: predict (raw tensors have no meaning for a chat model).
Scores from classify/decide are the model's self-reported confidence, not calibrated probabilities.

Options (gateway.toml `options`):
  base_url     default http://localhost:11434/v1 (Ollama)
  model        upstream chat model name, e.g. "llama3.2"
  embed_model  upstream embedding model, e.g. "nomic-embed-text" (omit to disable /v1/embed)
  api_key      optional bearer token
  timeout      seconds, default 120
"""

import json
import re

import httpx

from ai_gateway import schemas as s

from .base import Backend, UpstreamError


class OpenAICompatBackend(Backend):
    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        model: str = "llama3.2",
        embed_model: str | None = None,
        api_key: str | None = None,
        timeout: float = 120.0,
        http_client: httpx.AsyncClient | None = None,  # injectable for tests
    ):
        super().__init__(base_url=base_url, model=model, embed_model=embed_model)
        self.model = model
        self.embed_model = embed_model
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._http = http_client or httpx.AsyncClient(
            base_url=base_url.rstrip("/"), headers=headers, timeout=timeout
        )

    @property
    def capabilities(self) -> set[str]:
        caps = super().capabilities - {"predict"}
        if not self.embed_model:
            caps.discard("embed")
        return caps

    # -- upstream plumbing ------------------------------------------------------

    async def _post(self, path: str, body: dict) -> dict:
        try:
            r = await self._http.post(path, json=body)
        except httpx.HTTPError as e:
            raise UpstreamError(f"cannot reach upstream: {e!r}") from e
        if r.status_code >= 400:
            raise UpstreamError(f"upstream {r.status_code}: {r.text[:300]}")
        return r.json()

    @staticmethod
    def _usage(u: dict | None) -> s.Usage:
        u = u or {}
        p, c = u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
        return s.Usage(prompt_tokens=p, completion_tokens=c, total_tokens=u.get("total_tokens", p + c))

    def _chat_body(self, req: s.GenerateRequest, **extra) -> dict:
        body: dict = {
            "model": self.model,
            "messages": [m.model_dump() for m in req.messages],
            **extra,
        }
        if req.max_tokens is not None:
            body["max_tokens"] = req.max_tokens
        if req.temperature is not None:
            body["temperature"] = req.temperature
        if req.stop:
            body["stop"] = req.stop
        return body

    # -- generate ---------------------------------------------------------------

    async def generate(self, req):
        data = await self._post("/chat/completions", self._chat_body(req))
        try:
            choice = data["choices"][0]
            return s.GenerateResponse(
                model=req.model or self.model,
                text=choice["message"]["content"] or "",
                finish_reason=choice.get("finish_reason") or "stop",
                usage=self._usage(data.get("usage")),
            )
        except (KeyError, IndexError, TypeError) as e:
            raise UpstreamError(f"unexpected upstream response: {data!r:.300}") from e

    async def generate_stream(self, req):
        body = self._chat_body(req, stream=True, stream_options={"include_usage": True})
        finish, usage = "stop", None
        try:
            async with self._http.stream("POST", "/chat/completions", json=body) as r:
                if r.status_code >= 400:
                    await r.aread()
                    raise UpstreamError(f"upstream {r.status_code}: {r.text[:300]}")
                async for line in r.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    ev = json.loads(payload)
                    if ev.get("usage"):
                        usage = self._usage(ev["usage"])
                    for ch in ev.get("choices", []):
                        if ch.get("finish_reason"):
                            finish = ch["finish_reason"]
                        delta = (ch.get("delta") or {}).get("content")
                        if delta:
                            yield s.GenerateChunk(delta=delta)
        except httpx.HTTPError as e:
            raise UpstreamError(f"cannot reach upstream: {e!r}") from e
        yield s.GenerateChunk(finish_reason=finish, usage=usage or s.Usage())

    # -- embed ------------------------------------------------------------------

    async def embed(self, req):
        data = await self._post("/embeddings", {"model": self.embed_model, "input": req.input})
        try:
            items = sorted(data["data"], key=lambda d: d.get("index", 0))
            vecs = [d["embedding"] for d in items]
        except (KeyError, TypeError) as e:
            raise UpstreamError(f"unexpected upstream response: {data!r:.300}") from e
        return s.EmbedResponse(
            model=req.model or self.embed_model, embeddings=vecs, usage=self._usage(data.get("usage"))
        )

    # -- JSON-prompted tasks ----------------------------------------------------

    async def _json_task(self, system: str, user: str) -> dict:
        body = {
            "model": self.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system + " Respond with a single JSON object only."},
                {"role": "user", "content": user},
            ],
        }
        data = await self._post("/chat/completions", body)
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise UpstreamError(f"unexpected upstream response: {data!r:.300}") from e
        return _parse_json(text)

    async def classify(self, req):
        subject = req.text if req.text is not None else json.dumps(req.features)
        label_hint = (
            f"Choose only from these labels: {json.dumps(req.labels)}."
            if req.labels
            else "Pick short, lowercase, single-word labels."
        )
        out = await self._json_task(
            "You are a classifier. " + label_hint + " Score each returned label from 0 to 1."
            ' Format: {"labels": [{"label": str, "score": number}]}.',
            subject,
        )
        scored = []
        for item in out.get("labels", []):
            try:
                label, score = str(item["label"]), float(item["score"])
            except (KeyError, TypeError, ValueError):
                continue
            if req.labels and label not in req.labels:
                continue
            scored.append(s.LabelScore(label=label, score=min(max(score, 0.0), 1.0)))
        if not scored:
            raise UpstreamError(f"model returned no usable labels: {out!r:.300}")
        scored.sort(key=lambda x: -x.score)
        return s.ClassifyResponse(model=req.model or self.model, labels=scored[: req.top_k])

    async def extract(self, req):
        types = req.entity_types or ["PERSON", "ORG", "LOCATION", "DATE", "EMAIL", "NUMBER"]
        schema_hint = f" Also follow this schema for `value`: {json.dumps(req.schema_)}." if req.schema_ else ""
        out = await self._json_task(
            "You extract entities from a document. Only use these types: "
            f"{json.dumps(types)}. Copy each entity's text EXACTLY as it appears in the document."
            + schema_hint
            + ' Format: {"entities": [{"type": str, "text": str, "value": any}]}.',
            req.document,
        )
        ents, cursor = [], 0
        for item in out.get("entities", []):
            text, etype = item.get("text"), item.get("type")
            if not isinstance(text, str) or not text or etype not in types:
                continue
            # The model gives text; we compute offsets so spans are always consistent.
            start = req.document.find(text, cursor)
            if start < 0:
                start = req.document.find(text)
            if start < 0:
                continue  # hallucinated text that is not in the document
            cursor = start + len(text)
            ents.append(
                s.Entity(type=etype, text=text, start=start, end=start + len(text), value=item.get("value"))
            )
        ents.sort(key=lambda e: e.start)
        return s.ExtractResponse(model=req.model or self.model, entities=ents)

    async def decide(self, req):
        options = req.options
        constraint = f" `decision` must be exactly one of {json.dumps(options)}." if options else ""
        out = await self._json_task(
            "You make a decision from structured facts." + constraint
            + " Give `score` as your confidence from 0 to 1 and 1-5 short `reasons`."
            ' Format: {"decision": str, "score": number, "reasons": [str]}.',
            json.dumps(req.facts),
        )
        decision = out.get("decision")
        if not isinstance(decision, str) or (options and decision not in options):
            raise UpstreamError(f"model returned invalid decision: {out!r:.300}")
        try:
            score = min(max(float(out.get("score", 0.0)), 0.0), 1.0)
        except (TypeError, ValueError):
            score = 0.0
        reasons = [str(r) for r in out.get("reasons", []) if r]
        return s.DecideResponse(model=req.model or self.model, decision=decision, score=score, reasons=reasons)


def _parse_json(text: str) -> dict:
    text = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.S)
    if fence:
        text = fence.group(1)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        raise UpstreamError(f"model did not return valid JSON: {text[:200]!r}") from e
    if not isinstance(obj, dict):
        raise UpstreamError(f"model returned non-object JSON: {text[:200]!r}")
    return obj
