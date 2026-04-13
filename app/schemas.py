from pydantic import BaseModel, ConfigDict, Field, model_validator


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1)
    top_k_retrieve: int = Field(default=20, ge=1, le=100)
    top_k_rerank: int = Field(default=5, ge=1, le=20)

    @model_validator(mode="after")
    def normalize_and_validate(self) -> "AskRequest":
        self.question = self.question.strip()
        if not self.question:
            raise ValueError("question must not be blank")
        return self


class SourcePassage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    context: str
    score: float


class AskResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str
    answer: str
    sources: list[SourcePassage]
    retrieval_ms: float
    rerank_ms: float
    generation_ms: float
