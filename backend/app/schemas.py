from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class AnalyzeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000, description="待分析的完整句子或短文本")
    response_language: str = Field(default="zh-CN", max_length=20)

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("text 不能为空白字符串")
        return value


class LabelScore(BaseModel):
    label: str
    label_zh: str
    probability: float = Field(ge=0, le=1)
    threshold: float = Field(ge=0, le=1)
    predicted: bool


class ClassificationResult(BaseModel):
    is_cyberbullying: bool
    primary_type: str | None
    primary_type_zh: str | None
    detected_types: list[str]
    detected_types_zh: list[str]
    scores: list[LabelScore]


class Highlight(BaseModel):
    text: str
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    category: str
    severity: Literal["low", "medium", "high"] = "medium"
    explanation: str = ""


class LLMResult(BaseModel):
    provider: Literal["llm", "fallback", "skipped"]
    highlights: list[Highlight]
    intervention: str


class AnalyzeResponse(BaseModel):
    text: str
    classification: ClassificationResult
    llm: LLMResult
    annotated_html: str


class HealthResponse(BaseModel):
    status: str
    classifier_ready: bool
    llm_configured: bool
    model_dir: str


class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=30, pattern=r"^[A-Za-z0-9_\-]+$")
    email: str = Field(min_length=5, max_length=254)
    password: str = Field(min_length=8, max_length=128)

    @field_validator("email")
    @classmethod
    def validate_email(cls, value: str) -> str:
        value = value.strip().casefold()
        if value.count("@") != 1 or "." not in value.split("@", 1)[1]:
            raise ValueError("邮箱格式不正确")
        return value


class LoginRequest(BaseModel):
    account: str = Field(min_length=3, max_length=254, description="用户名或邮箱")
    password: str = Field(min_length=8, max_length=128)


class UserPublic(BaseModel):
    id: int
    username: str
    email: str
    created_at: str


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserPublic
