"""Deterministic backend implementing every endpoint. Useful for wiring and as a template."""

import hashlib
import re

from ai_gateway import schemas as s

from .base import Backend


def _tokens(text: str) -> int:
    return len(text.split())


class MockBackend(Backend):
    DIM = 8

    async def generate(self, req):
        text = f"echo: {req.messages[-1].content}"
        if req.max_tokens:
            text = " ".join(text.split()[: req.max_tokens])
        p = sum(_tokens(m.content) for m in req.messages)
        c = _tokens(text)
        return s.GenerateResponse(
            model=req.model or "mock",
            text=text,
            usage=s.Usage(prompt_tokens=p, completion_tokens=c, total_tokens=p + c),
        )

    async def generate_stream(self, req):
        res = await self.generate(req)
        words = res.text.split(" ")
        for i, word in enumerate(words):
            yield s.GenerateChunk(delta=word if i == len(words) - 1 else word + " ")
        yield s.GenerateChunk(finish_reason=res.finish_reason, usage=res.usage)

    async def embed(self, req):
        vecs = [[b / 255 for b in hashlib.sha256(t.encode()).digest()[: self.DIM]] for t in req.input]
        n = sum(_tokens(t) for t in req.input)
        return s.EmbedResponse(
            model=req.model or "mock",
            embeddings=vecs,
            usage=s.Usage(prompt_tokens=n, total_tokens=n),
        )

    async def classify(self, req):
        labels = req.labels or ["positive", "negative"]
        text = req.text or str(req.features or "")
        h = int(hashlib.sha256(text.encode()).hexdigest(), 16)
        raw = [(h >> (8 * i)) % 100 + 1 for i in range(len(labels))]
        total = sum(raw)
        scored = sorted(
            (s.LabelScore(label=lb, score=r / total) for lb, r in zip(labels, raw)),
            key=lambda x: -x.score,
        )
        return s.ClassifyResponse(model=req.model or "mock", labels=scored[: req.top_k])

    async def extract(self, req):
        types = req.entity_types or ["EMAIL", "NUMBER"]
        patterns = {"EMAIL": r"[\w.+-]+@[\w-]+\.[\w.]+", "NUMBER": r"\d+(?:\.\d+)?"}
        ents = [
            s.Entity(type=t, text=m.group(), start=m.start(), end=m.end(), score=1.0)
            for t in types
            if t in patterns
            for m in re.finditer(patterns[t], req.document)
        ]
        ents.sort(key=lambda e: e.start)
        return s.ExtractResponse(model=req.model or "mock", entities=ents)

    async def decide(self, req):
        options = req.options or ["approve", "reject"]
        h = int(hashlib.sha256(repr(sorted(req.facts.items())).encode()).hexdigest(), 16)
        return s.DecideResponse(
            model=req.model or "mock",
            decision=options[h % len(options)],
            score=((h >> 8) % 1000) / 1000,
            reasons=[f"{k}={v!r}" for k, v in sorted(req.facts.items())],
        )

    async def predict(self, req):
        # Identity: echo requested tensors (or all of them).
        wanted = set(req.outputs) if req.outputs else None
        outs = [t for t in req.inputs if wanted is None or t.name in wanted]
        return s.PredictResponse(model=req.model or "mock", outputs=outs)
