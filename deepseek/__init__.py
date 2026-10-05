"""Unofficial OpenAI-compatible client for chat.deepseek.com."""

from .auth import LoginRequired, Session, get_session, login, refresh_session
from .client import Chunk, DeepSeekClient, Reply, UpstreamError
from .pow import (
    DeepSeekPow,
    PoWChallengeExpiredError,
    PoWTimeoutError,
    PowTaskQueue,
    get_pow_queue,
)

__all__ = [
    "Session",
    "get_session",
    "login",
    "refresh_session",
    "LoginRequired",
    "DeepSeekClient",
    "Reply",
    "Chunk",
    "UpstreamError",
    "DeepSeekPow",
    "PowTaskQueue",
    "get_pow_queue",
    "PoWTimeoutError",
    "PoWChallengeExpiredError",
]
