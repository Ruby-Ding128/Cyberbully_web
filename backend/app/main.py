"""
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from .classifier import CyberbullyingClassifier
from .auth import UserStore, create_access_token, get_current_user
from .config import ROOT, settings
from .llm_service import LLMService
from .schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    AuthResponse,
    HealthResponse,
    LoginRequest,
    RegisterRequest,
    UserPublic,
)
from .visualization import render_highlighted_html


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.user_store = UserStore(settings.auth_db_path)
    app.state.user_store.initialize()
    app.state.classifier = CyberbullyingClassifier(
        settings.model_dir, max_length=settings.max_model_length
    )
    app.state.llm = LLMService(
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        timeout=settings.llm_timeout_seconds,
    )
    yield


app = FastAPI(
    title="青少年网络欺凌识别与干预 API",
    version="1.0.0",
    description="DistilBERT 多标签识别 + LLM 关键词解释与干预提醒",
    lifespan=lifespan,
)

static_dir = ROOT / "backend" / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index() -> HTMLResponse:
    return HTMLResponse((static_dir / "index.html").read_text(encoding="utf-8"))


@app.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    return HealthResponse(
        status="ok",
        classifier_ready=hasattr(request.app.state, "classifier"),
        llm_configured=request.app.state.llm.configured,
        model_dir=str(settings.model_dir),
    )


@app.post("/api/v1/auth/register", response_model=AuthResponse, status_code=201)
async def register(payload: RegisterRequest, request: Request) -> AuthResponse:
    user = request.app.state.user_store.create_user(
        payload.username, payload.email, payload.password
    )
    token, expires_in = create_access_token(user)
    return AuthResponse(access_token=token, expires_in=expires_in, user=user)


@app.post("/api/v1/auth/login", response_model=AuthResponse)
async def login(payload: LoginRequest, request: Request) -> AuthResponse:
    user = request.app.state.user_store.authenticate(payload.account, payload.password)
    if user is None:
        raise HTTPException(status_code=401, detail="用户名、邮箱或密码错误")
    token, expires_in = create_access_token(user)
    return AuthResponse(access_token=token, expires_in=expires_in, user=user)


@app.get("/api/v1/auth/me", response_model=UserPublic)
async def me(user: UserPublic = Depends(get_current_user)) -> UserPublic:
    return user


@app.post("/api/v1/analyze", response_model=AnalyzeResponse)
async def analyze(
    payload: AnalyzeRequest,
    request: Request,
    _user: UserPublic = Depends(get_current_user),
) -> AnalyzeResponse:
    if len(payload.text) > settings.max_text_length:
        raise HTTPException(status_code=413, detail="输入文本过长")
    classifier: CyberbullyingClassifier = request.app.state.classifier
    llm: LLMService = request.app.state.llm
    classification = classifier.predict(payload.text)
    llm_result = await llm.analyze(
        payload.text, classification, payload.response_language
    )
    return AnalyzeResponse(
        text=payload.text,
        classification=classification,
        llm=llm_result,
        annotated_html=render_highlighted_html(payload.text, llm_result.highlights),
    )
