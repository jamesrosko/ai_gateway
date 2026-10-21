import json

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from . import schemas as s
from .backends import NotSupported, Registry, load_registry


def create_app(registry: Registry | None = None) -> FastAPI:
    app = FastAPI(title="AI Gateway", version="0.1.0")
    app.state.registry = registry if registry is not None else load_registry()

    def err(status: int, error: str, detail: str | None = None):
        return JSONResponse(s.ErrorBody(error=error, detail=detail).model_dump(), status_code=status)

    def resolve(request: Request, capability: str, model: str | None):
        reg: Registry = request.app.state.registry
        try:
            name, backend = reg.resolve(capability, model)
        except (KeyError, LookupError) as e:
            return None, None, err(404, "model_not_found", str(e))
        if capability not in backend.capabilities:
            return None, None, err(501, "not_supported", f"model '{name}' lacks '{capability}'")
        return name, backend, None

    async def call(request: Request, capability: str, req):
        name, backend, bad = resolve(request, capability, req.model)
        if bad:
            return bad
        try:
            res = await getattr(backend, capability)(req)
        except NotSupported as e:
            return err(501, "not_supported", str(e))
        except ValueError as e:
            return err(422, "invalid_request", str(e))
        res.model = name
        return res

    @app.post("/v1/embed", response_model=s.EmbedResponse)
    async def embed(req: s.EmbedRequest, request: Request):
        return await call(request, "embed", req)

    @app.post("/v1/classify", response_model=s.ClassifyResponse)
    async def classify(req: s.ClassifyRequest, request: Request):
        if req.text is None and req.features is None:
            return err(422, "invalid_request", "one of 'text' or 'features' is required")
        return await call(request, "classify", req)

    @app.post("/v1/extract", response_model=s.ExtractResponse)
    async def extract(req: s.ExtractRequest, request: Request):
        return await call(request, "extract", req)

    @app.post("/v1/decide", response_model=s.DecideResponse)
    async def decide(req: s.DecideRequest, request: Request):
        return await call(request, "decide", req)

    @app.post("/v1/predict", response_model=s.PredictResponse)
    async def predict(req: s.PredictRequest, request: Request):
        return await call(request, "predict", req)

    @app.post("/v1/generate", response_model=s.GenerateResponse)
    async def generate(req: s.GenerateRequest, request: Request):
        if not req.stream:
            return await call(request, "generate", req)
        name, backend, bad = resolve(request, "generate", req.model)
        if bad:
            return bad
        req = req.model_copy(update={"model": name})

        async def events():
            try:
                async for chunk in backend.generate_stream(req):
                    yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
            except Exception as e:  # surface mid-stream failures to the client
                yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.get("/v1/models")
    async def models(request: Request):
        reg: Registry = request.app.state.registry
        return {
            "models": [
                {"name": n, "capabilities": sorted(b.capabilities)} for n, b in reg.models.items()
            ]
        }

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    return app
