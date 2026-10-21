"""Wire schemas shared by the gateway, client and harness."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


# --- /v1/generate -----------------------------------------------------------


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


class GenerateRequest(BaseModel):
    model: str | None = None
    messages: list[Message] = Field(min_length=1)
    max_tokens: int | None = Field(default=None, gt=0)
    temperature: float | None = Field(default=None, ge=0)
    stop: list[str] | None = None
    stream: bool = False


class GenerateResponse(BaseModel):
    model: str
    text: str
    finish_reason: str = "stop"
    usage: Usage = Field(default_factory=Usage)


class GenerateChunk(BaseModel):
    """One SSE event of a streamed generation. The final event carries finish_reason/usage."""

    delta: str = ""
    finish_reason: str | None = None
    usage: Usage | None = None


# --- /v1/embed --------------------------------------------------------------


class EmbedRequest(BaseModel):
    model: str | None = None
    input: list[str] = Field(min_length=1)


class EmbedResponse(BaseModel):
    model: str
    embeddings: list[list[float]]
    usage: Usage = Field(default_factory=Usage)


# --- /v1/classify -----------------------------------------------------------


class ClassifyRequest(BaseModel):
    model: str | None = None
    text: str | None = None
    features: dict[str, Any] | None = None
    labels: list[str] | None = None  # optional candidate labels (zero-shot)
    top_k: int | None = Field(default=None, gt=0)


class LabelScore(BaseModel):
    label: str
    score: float


class ClassifyResponse(BaseModel):
    model: str
    labels: list[LabelScore]


# --- /v1/extract ------------------------------------------------------------


class ExtractRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    model: str | None = None
    document: str
    entity_types: list[str] | None = None
    schema_: dict[str, Any] | None = Field(default=None, alias="schema")


class Entity(BaseModel):
    type: str
    text: str
    start: int
    end: int
    score: float | None = None
    value: Any = None  # normalised/typed value, if the backend produces one


class ExtractResponse(BaseModel):
    model: str
    entities: list[Entity]


# --- /v1/decide -------------------------------------------------------------


class DecideRequest(BaseModel):
    model: str | None = None
    facts: dict[str, Any]
    options: list[str] | None = None  # allowed decisions, if constrained


class DecideResponse(BaseModel):
    model: str
    decision: str
    score: float
    reasons: list[str] = Field(default_factory=list)


# --- /v1/predict (V2-style tensors) -----------------------------------------


class Tensor(BaseModel):
    name: str
    shape: list[int]
    datatype: str  # e.g. FP32, INT64, BYTES, BOOL
    data: list[Any]  # flattened, row-major


class PredictRequest(BaseModel):
    model: str | None = None
    inputs: list[Tensor] = Field(min_length=1)
    outputs: list[str] | None = None  # names of requested outputs


class PredictResponse(BaseModel):
    model: str
    outputs: list[Tensor]


class ErrorBody(BaseModel):
    error: str
    detail: str | None = None
