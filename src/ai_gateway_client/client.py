from collections.abc import Iterator
from typing import Any

import httpx

from ai_gateway import schemas as s


class GatewayError(Exception):
    def __init__(self, status: int, error: str, detail: str | None = None):
        super().__init__(f"{status} {error}: {detail}" if detail else f"{status} {error}")
        self.status = status
        self.error = error
        self.detail = detail


class GatewayClient:
    """Typed client for the gateway.

    Pass `http_client` (any httpx.Client, e.g. FastAPI's TestClient) to talk to an in-process app.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        *,
        api_key: str | None = None,
        timeout: float = 60.0,
        http_client: httpx.Client | None = None,
    ):
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._http = http_client or httpx.Client(
            base_url=base_url, headers=headers, timeout=timeout
        )

    def close(self) -> None:
        self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- plumbing --------------------------------------------------------------

    @staticmethod
    def _raise_for_error(resp: httpx.Response) -> None:
        if resp.is_success:
            return
        try:
            body = resp.json()
            if "detail" in body and "error" not in body:  # FastAPI validation errors
                raise GatewayError(resp.status_code, "validation_error", str(body["detail"]))
            raise GatewayError(resp.status_code, body.get("error", "error"), body.get("detail"))
        except ValueError:
            raise GatewayError(resp.status_code, "error", resp.text) from None

    def _post(self, path: str, payload: dict[str, Any], out: type):
        resp = self._http.post(path, json=payload)
        self._raise_for_error(resp)
        return out.model_validate(resp.json())

    @staticmethod
    def _body(**kw) -> dict[str, Any]:
        return {k: v for k, v in kw.items() if v is not None}

    # -- endpoints -------------------------------------------------------------

    def models(self) -> list[dict[str, Any]]:
        resp = self._http.get("/v1/models")
        self._raise_for_error(resp)
        return resp.json()["models"]

    def generate(
        self,
        messages: list[dict[str, str]] | str,
        *,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        stop: list[str] | None = None,
    ) -> s.GenerateResponse:
        return self._post(
            "/v1/generate",
            self._body(
                model=model,
                messages=_messages(messages),
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
            ),
            s.GenerateResponse,
        )

    def generate_stream(
        self,
        messages: list[dict[str, str]] | str,
        *,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        stop: list[str] | None = None,
    ) -> Iterator[s.GenerateChunk]:
        """Yield chunks; the last carries finish_reason and usage."""
        payload = self._body(
            model=model,
            messages=_messages(messages),
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            stream=True,
        )
        with self._http.stream("POST", "/v1/generate", json=payload) as resp:
            if not resp.is_success:
                resp.read()
                self._raise_for_error(resp)
            event = "message"
            for line in resp.iter_lines():
                if line.startswith("event:"):
                    event = line[6:].strip()
                elif line.startswith("data:"):
                    data = line[5:].strip()
                    if data == "[DONE]":
                        return
                    if event == "error":
                        raise GatewayError(200, "stream_error", data)
                    yield s.GenerateChunk.model_validate_json(data)
                elif not line:
                    event = "message"

    def embed(self, input: list[str] | str, *, model: str | None = None) -> s.EmbedResponse:
        texts = [input] if isinstance(input, str) else input
        return self._post("/v1/embed", self._body(model=model, input=texts), s.EmbedResponse)

    def classify(
        self,
        text: str | None = None,
        *,
        features: dict[str, Any] | None = None,
        labels: list[str] | None = None,
        top_k: int | None = None,
        model: str | None = None,
    ) -> s.ClassifyResponse:
        return self._post(
            "/v1/classify",
            self._body(model=model, text=text, features=features, labels=labels, top_k=top_k),
            s.ClassifyResponse,
        )

    def extract(
        self,
        document: str,
        *,
        entity_types: list[str] | None = None,
        schema: dict[str, Any] | None = None,
        model: str | None = None,
    ) -> s.ExtractResponse:
        return self._post(
            "/v1/extract",
            self._body(model=model, document=document, entity_types=entity_types, schema=schema),
            s.ExtractResponse,
        )

    def decide(
        self,
        facts: dict[str, Any],
        *,
        options: list[str] | None = None,
        model: str | None = None,
    ) -> s.DecideResponse:
        return self._post(
            "/v1/decide", self._body(model=model, facts=facts, options=options), s.DecideResponse
        )

    def predict(
        self,
        inputs: list[s.Tensor | dict[str, Any]],
        *,
        outputs: list[str] | None = None,
        model: str | None = None,
    ) -> s.PredictResponse:
        tensors = [t.model_dump() if isinstance(t, s.Tensor) else t for t in inputs]
        return self._post(
            "/v1/predict", self._body(model=model, inputs=tensors, outputs=outputs), s.PredictResponse
        )


def _messages(messages: list[dict[str, str]] | str) -> list[dict[str, str]]:
    return [{"role": "user", "content": messages}] if isinstance(messages, str) else messages
