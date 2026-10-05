"""
OpenAI-compatible FastAPI server for DeepSeek.

Point any OpenAI client at either:
    - http://localhost:8000/v1
    - http://localhost:8000

Example:
    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
    r = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": "Hello!"}],
    )

Endpoints:
    POST /v1/chat/completions or /chat/completions
    GET  /v1/models or /models
    GET  /v1/models/{model} or /models/{model}
    POST /v1/embeddings or /embeddings
    GET  /healthz or /
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool

from deepseek.auth import DEFAULT_SESSION_FILE, LoginRequired, Session
from deepseek.client import DeepSeekClient, UpstreamError
from deepseek.pow import (
    PoWChallengeExpiredError,
    PoWTimeoutError,
    get_pow_queue,
)

from .config import (
    API_KEY,
    DEBUG_REQUESTS,
    MODEL_MAP,
    POW_SOLVE_TIMEOUT,
    POW_WORKERS,
    RATE_LIMIT_PER_MINUTE,
    SERVER_INTERACTIVE_LOGIN,
    UPSTREAM_RETRY_AFTER,
    is_known_model,
    resolve_model_type,
    should_enable_thinking,
)
from .openai_format import (
    completion_response,
    extract_tool_calls,
    messages_to_prompt,
    stream_chunks,
    stream_chunks_with_tools,
    trim_at_role_boundary,
)
from .ratelimit import RateLimiter, install_rate_limit
from .schemas import ChatCompletionRequest, EmbeddingRequest

load_dotenv()

log = logging.getLogger("uvicorn.error")

app = FastAPI(title="DeepSeek OpenAI-compatible API", version="0.3.0")

# Enable CORS for downstream coding agents, IDEs, and browser web-apps
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

install_rate_limit(app, RateLimiter(limit=RATE_LIMIT_PER_MINUTE, window=60.0))

# Shared client and PoW queue
_client: DeepSeekClient | None = None
_client_lock = threading.Lock()
_pow_queue = get_pow_queue(max_workers=POW_WORKERS, timeout=POW_SOLVE_TIMEOUT)


def get_client() -> DeepSeekClient:
    """Build (once) the shared client and its signed-in session."""
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = DeepSeekClient(
                    allow_interactive=SERVER_INTERACTIVE_LOGIN,
                    pow_queue=_pow_queue,
                )
    return _client


def _error(
    message: str,
    status: int = 500,
    err_type: str = "server_error",
    headers: Optional[dict] = None,
):
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": err_type, "code": str(status)}},
        headers=headers or None,
    )


def _handle_upstream_error(exc: UpstreamError):
    """Translate upstream DeepSeek errors into actionable status codes for clients."""
    if exc.is_rate_limit:
        return _error(
            f"DeepSeek upstream rate limit reached: {exc}. Please slow down requests.",
            status=429,
            err_type="rate_limit_exceeded",
            headers={"Retry-After": str(UPSTREAM_RETRY_AFTER)},
        )
    if exc.is_waf_block:
        return _error(
            "DeepSeek AWS WAF / human verification check blocked this request. "
            "Please clear the human-check by running 'python -m deepseek.auth'.",
            status=502,
            err_type="upstream_waf_block",
        )
    return _error(
        f"DeepSeek reported an upstream error: {exc}",
        status=502,
        err_type="upstream_error",
    )


def _handle_pow_error(exc: Exception):
    """Graceful response when PoW solver queue times out or challenge expires."""
    return _error(
        f"Proof-of-work solver queue busy: {exc}. Please retry shortly.",
        status=503,
        err_type="pow_queue_busy",
        headers={"Retry-After": "2"},
    )


def _verify_auth(authorization: Optional[str] = None) -> bool:
    """Optional API Key check if API_KEY is set in environment."""
    if not API_KEY:
        return True
    if not authorization:
        return False
    parts = authorization.split()
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1] == API_KEY
    return authorization == API_KEY


class _Prefetched:
    """A stream with its first delta already pulled.

    The endpoint consumes one delta before returning a response, so an error
    frame -- which DeepSeek sends immediately -- becomes a real status code
    instead of a 200 whose body turns out to be an error mid-flight.
    """

    def __init__(self, stream, iterator, first):
        self._stream, self._iter, self._first = stream, iterator, first

    def __iter__(self):
        if self._first is not None:
            yield self._first
        yield from self._iter

    @property
    def conversation_id(self):
        return getattr(self._stream, "conversation_id", None)


@app.get("/")
@app.get("/healthz")
def healthz():
    """Detailed health check and diagnostics."""
    cached_session = Session.load(DEFAULT_SESSION_FILE)
    session_info = {
        "has_session": cached_session is not None,
        "age_seconds": round(cached_session.age, 1) if cached_session else None,
        "has_token": bool(cached_session and cached_session.token),
    }
    return {
        "status": "ok",
        "service": "deepseek-openai-api",
        "session": session_info,
        "pow_queue": _pow_queue.stats(),
        "models": list(MODEL_MAP.keys()),
    }


@app.get("/models")
@app.get("/v1/models")
def list_models(authorization: Optional[str] = Header(None)):
    if not _verify_auth(authorization):
        return _error("Incorrect API key provided.", status=401, err_type="invalid_api_key")

    created = int(time.time())
    return {
        "object": "list",
        "data": [
            {
                "id": name,
                "object": "model",
                "created": created,
                "owned_by": "deepseek",
                "permission": [],
                "root": name,
                "parent": None,
            }
            for name in MODEL_MAP
        ],
    }


@app.get("/models/{model}")
@app.get("/v1/models/{model}")
def retrieve_model(model: str, authorization: Optional[str] = Header(None)):
    if not _verify_auth(authorization):
        return _error("Incorrect API key provided.", status=401, err_type="invalid_api_key")

    if not is_known_model(model):
        return _error(f"The model `{model}` does not exist.", status=404, err_type="model_not_found")

    return {
        "id": model,
        "object": "model",
        "created": int(time.time()),
        "owned_by": "deepseek",
        "permission": [],
        "root": model,
        "parent": None,
    }


@app.post("/embeddings")
@app.post("/v1/embeddings")
def embeddings(req: EmbeddingRequest, authorization: Optional[str] = Header(None)):
    """Fallback stub for OpenAI embeddings so clients don't crash."""
    if not _verify_auth(authorization):
        return _error("Incorrect API key provided.", status=401, err_type="invalid_api_key")

    inputs = [req.input] if isinstance(req.input, str) else req.input
    return {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "embedding": [0.0] * 8,
                "index": i,
            }
            for i in range(len(inputs))
        ],
        "model": req.model or "text-embedding-ada-002",
        "usage": {"prompt_tokens": len(inputs), "total_tokens": len(inputs)},
    }


@app.post("/chat/completions")
@app.post("/v1/chat/completions")
async def chat_completions(
    req: ChatCompletionRequest,
    authorization: Optional[str] = Header(None),
):
    if not _verify_auth(authorization):
        return _error("Incorrect API key provided.", status=401, err_type="invalid_api_key")

    if not req.messages:
        return _error("`messages` must not be empty", status=400, err_type="invalid_request_error")

    if not is_known_model(req.model):
        return _error(
            f"The model `{req.model}` does not exist. Available models: {', '.join(MODEL_MAP)}",
            status=404,
            err_type="model_not_found",
        )

    # A thread's model is fixed when created; on resume let existing thread stand
    model_type = None if req.conversation_id else resolve_model_type(req.model)
    thinking_enabled = should_enable_thinking(req.model, req.thinking)
    search_enabled = bool(req.search)

    # Tool calling configuration
    tools = None if req.tool_choice == "none" else req.tools
    prompt = messages_to_prompt(req.messages, tools, req.tool_choice)

    if DEBUG_REQUESTS:
        tool_names = [(t.get("function") or t).get("name") for t in (req.tools or [])]
        log.warning(
            "request: model=%s stream=%s thinking=%s search=%s roles=%s tools=%s tool_choice=%s",
            req.model,
            req.stream,
            thinking_enabled,
            search_enabled,
            [m.role for m in req.messages],
            tool_names,
            req.tool_choice,
        )

    try:
        # Off the event loop: get_client() uses Playwright if initializing from session
        client = await run_in_threadpool(get_client)
    except LoginRequired as e:
        return _error(str(e), status=503, err_type="login_required")
    except Exception as e:
        return _error(f"Failed to initialise DeepSeek session: {e}", status=503)

    include_usage = bool(req.stream_options and req.stream_options.get("include_usage"))

    if req.stream:
        try:
            raw = client.stream(
                prompt,
                conversation_id=req.conversation_id,
                model=model_type,
                thinking=thinking_enabled,
                search=search_enabled,
            )
            it = iter(raw)
            # Pull first chunk off the event loop so errors surface as status codes
            first = await run_in_threadpool(lambda: next(it, None))
        except UpstreamError as e:
            return _handle_upstream_error(e)
        except (PoWTimeoutError, PoWChallengeExpiredError) as e:
            return _handle_pow_error(e)
        except LoginRequired as e:
            return _error(str(e), status=503, err_type="login_required")
        except Exception as e:
            return _error(f"DeepSeek request failed: {e}")

        def gen():
            stream = _Prefetched(raw, it, first)
            if tools:
                yield from stream_chunks_with_tools(
                    req.model, stream, tools, include_usage=include_usage, prompt=prompt
                )
            else:
                yield from stream_chunks(
                    req.model, stream, include_usage=include_usage, prompt=prompt
                )

        return StreamingResponse(gen(), media_type="text/event-stream")

    try:
        reply = await run_in_threadpool(
            client.chat,
            prompt,
            req.conversation_id,
            model_type,
            thinking_enabled,
            search_enabled,
        )
    except UpstreamError as e:
        return _handle_upstream_error(e)
    except (PoWTimeoutError, PoWChallengeExpiredError) as e:
        return _handle_pow_error(e)
    except LoginRequired as e:
        return _error(str(e), status=503, err_type="login_required")
    except Exception as e:
        return _error(f"DeepSeek request failed: {e}")

    calls, text = extract_tool_calls(reply.text, tools)
    if not tools:
        text = trim_at_role_boundary(text)

    if DEBUG_REQUESTS:
        log.warning(
            "reply: calls=%s text=%r reasoning=%r",
            [(c["function"]["name"], c["function"]["arguments"]) for c in calls or []],
            (text or "")[:200],
            (reply.reasoning_text or "")[:200],
        )

    return completion_response(
        req.model,
        text,
        prompt,
        reply.conversation_id,
        reasoning_content=reply.reasoning_text,
        tool_calls=calls,
    )
