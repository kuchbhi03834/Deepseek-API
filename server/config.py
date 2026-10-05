"""Server configuration: the OpenAI-facing model names, options, and environment settings."""

import os

# Requests per minute allowed per client IP (override with RATE_LIMIT_PER_MINUTE).
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))

# Optional API key for client Bearer token authentication (off by default for local use).
API_KEY = os.getenv("API_KEY", "").strip()

# When the server has no session, should it pop a visible browser window for
# interactive sign-in (the first request then blocks until you finish logging
# in)? On by default for local single-user use. Set to "0"/"false" for headless
# deployments, where it instead returns a 503 telling the caller to run
# `python -m deepseek.auth`.
SERVER_INTERACTIVE_LOGIN = os.getenv("SERVER_INTERACTIVE_LOGIN", "1").lower() not in (
    "0", "false", "no", "off",
)

# Proof-of-Work queue and worker pool settings
POW_WORKERS = int(os.getenv("POW_WORKERS", "4"))
POW_SOLVE_TIMEOUT = float(os.getenv("POW_SOLVE_TIMEOUT", "25.0"))

# Upstream rate limit backoff seconds advertised to clients in Retry-After header
UPSTREAM_RETRY_AFTER = os.getenv("UPSTREAM_RETRY_AFTER", "30")

# Debug logging for requests and tools
DEBUG_REQUESTS = os.getenv("DEBUG_REQUESTS", "").lower() in ("1", "true", "yes", "on")

# Public model ids the server advertises and accepts, mapped to DeepSeek's `model_type` wire value.
# Also includes common model aliases used by IDE agents like OpenCode, Cursor, and Cline.
MODEL_MAP = {
    "deepseek-chat":     "default",   # Instant — the fast default model
    "deepseek-expert":   "expert",    # Expert  — the stronger, slower model
    "deepseek-reasoner": "expert",    # Reasoning model (auto-enables thinking)
    "deepseek-coder":    "expert",    # Coder alias -> expert
    "gpt-4o":            "expert",    # OpenAI alias -> expert
    "gpt-4":             "expert",    # OpenAI alias -> expert
    "gpt-3.5-turbo":     "default",   # OpenAI alias -> default
}

DEFAULT_MODEL = "deepseek-chat"


def is_known_model(name: str) -> bool:
    """Whether `name` is a model id we accept (used to 404 unknown models)."""
    return name in MODEL_MAP


def resolve_model_type(name: str) -> str:
    """Translate a public model id to DeepSeek's `model_type` wire value.

    Caller must check `is_known_model` first; this raises KeyError otherwise.
    """
    return MODEL_MAP[name]


def should_enable_thinking(model: str, requested_thinking: bool = False) -> bool:
    """Determine if thinking (reasoning) should be enabled for this request.

    Automatically enabled for `deepseek-reasoner` or when explicitly toggled.
    """
    return requested_thinking or model == "deepseek-reasoner"
