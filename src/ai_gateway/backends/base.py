"""Backend interface. Subclass and override the capabilities your model supports.

Any method you do not override raises NotSupported, which the gateway maps to HTTP 501.
The set of overridden methods is auto-detected and exposed via `capabilities`.
"""

from collections.abc import AsyncIterator

from ai_gateway import schemas as s


class NotSupported(Exception):
    pass


_CAPABILITIES = ("generate", "embed", "classify", "extract", "decide", "predict")


class Backend:
    def __init__(self, **options):
        self.options = options

    @property
    def capabilities(self) -> set[str]:
        return {c for c in _CAPABILITIES if getattr(type(self), c) is not getattr(Backend, c)}

    async def generate(self, req: s.GenerateRequest) -> s.GenerateResponse:
        raise NotSupported("generate")

    def generate_stream(self, req: s.GenerateRequest) -> AsyncIterator[s.GenerateChunk]:
        """Override for true streaming. Default: run generate() and emit it as one chunk."""
        return self._fallback_stream(req)

    async def _fallback_stream(self, req: s.GenerateRequest):
        res = await self.generate(req)
        yield s.GenerateChunk(delta=res.text)
        yield s.GenerateChunk(finish_reason=res.finish_reason, usage=res.usage)

    async def embed(self, req: s.EmbedRequest) -> s.EmbedResponse:
        raise NotSupported("embed")

    async def classify(self, req: s.ClassifyRequest) -> s.ClassifyResponse:
        raise NotSupported("classify")

    async def extract(self, req: s.ExtractRequest) -> s.ExtractResponse:
        raise NotSupported("extract")

    async def decide(self, req: s.DecideRequest) -> s.DecideResponse:
        raise NotSupported("decide")

    async def predict(self, req: s.PredictRequest) -> s.PredictResponse:
        raise NotSupported("predict")
