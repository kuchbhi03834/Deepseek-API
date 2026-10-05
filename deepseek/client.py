"""
Pure-HTTP DeepSeek chat client.

Speaks chat.deepseek.com's internal API directly using a captured signed-in
session (see `deepseek.auth`). For each message it:

    1. creates a chat session   (POST /api/v0/chat_session/create)
    2. fetches a PoW challenge   (POST /api/v0/chat/create_pow_challenge)
    3. solves it via the WASM    (deepseek.pow.DeepSeekPow / PowTaskQueue)
    4. POSTs the completion       with the x-ds-pow-response header
    5. parses the SSE stream      into text and reasoning tokens

Automatic session recovery:
If an HTTP 401 or session expiration is encountered, the client automatically
triggers a headless background refresh using the cached Chrome profile in
session/ without crashing the server.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Tuple

import httpx

from .auth import (
    DEFAULT_PROFILE_DIR,
    DEFAULT_SESSION_FILE,
    LoginRequired,
    Session,
    get_session,
    refresh_session,
)
from .pow import (
    PowTaskQueue,
    get_pow_queue,
)

BASE = "https://chat.deepseek.com"
COMPLETION_PATH = "/api/v0/chat/completion"

# DeepSeek's mode pill, sent as `model_type` in the completion body. "default" is
# Instant (the fast model); "expert" is the stronger, slower model. Omitting the
# field lets the backend pick, so we always send one explicitly.
DEFAULT_MODEL_TYPE = "default"

# A conversation_id is an opaque "<chat_session_id>:<last_message_id>" token. It
# carries everything needed to resume a thread, so the client stays stateless.
_CID_SEP = ":"


class UpstreamError(RuntimeError):
    """An error DeepSeek reported inside the completion stream or via HTTP."""

    def __init__(self, message: str, finish_reason: str = "", status_code: int = 502):
        super().__init__(message or "DeepSeek reported an upstream error")
        self.finish_reason = finish_reason or ""
        self.status_code = status_code

    @property
    def is_rate_limit(self) -> bool:
        msg = str(self).lower()
        return (
            "rate_limit" in self.finish_reason.lower()
            or "too frequent" in msg
            or "messages too frequent" in msg
            or "rate limit" in msg
            or self.status_code == 429
        )

    @property
    def is_waf_block(self) -> bool:
        msg = str(self).lower()
        return (
            "waf" in msg
            or "cloudflare" in msg
            or "human check" in msg
            or "verify you're human" in msg
            or "verify you are human" in msg
            or "robot" in msg
            or "challenge" in msg
            or "shield" in msg
            or (self.status_code == 403 and "unauthorized" not in msg)
        )


class Chunk(str):
    """A stream token tagged with chunk_type ('RESPONSE' or 'THINK')."""

    def __new__(cls, value: str, chunk_type: str = "RESPONSE"):
        obj = str.__new__(cls, value)
        obj.chunk_type = chunk_type
        return obj


def _encode_cid(session_id: str, message_id: Optional[int]) -> str:
    if message_id is None:
        return session_id
    return f"{session_id}{_CID_SEP}{message_id}"


def _decode_cid(conversation_id: Optional[str]) -> tuple[Optional[str], Optional[int]]:
    """Split a conversation_id back into (chat_session_id, parent_message_id)."""
    if not conversation_id:
        return None, None
    session_id, _, msg = conversation_id.partition(_CID_SEP)
    parent = int(msg) if msg.isdigit() else None
    return (session_id or None), parent


@dataclass
class Reply:
    """A completed chat reply plus the id to resume the conversation."""

    text: str
    conversation_id: str
    reasoning_text: Optional[str] = None

    def __str__(self) -> str:  # so print(reply) shows the text
        return self.text


def _biz(data: dict) -> dict:
    """Unwrap DeepSeek's `data.biz_data` envelope, raising on API-level errors."""
    if data.get("code") != 0:
        raise RuntimeError(f"DeepSeek API error: {data.get('msg') or data}")
    biz = data.get("data", {}).get("biz_data")
    if biz is None:
        raise RuntimeError(f"Unexpected response shape: {data}")
    return biz


class DeepSeekClient:
    def __init__(
        self,
        session: Optional[Session] = None,
        allow_interactive: bool = True,
        profile_dir: Optional[Path] = None,
        session_file: Optional[Path] = None,
        pow_queue: Optional[PowTaskQueue] = None,
    ):
        self.profile_dir = profile_dir or DEFAULT_PROFILE_DIR
        self.session_file = session_file or DEFAULT_SESSION_FILE
        self.session = session or get_session(
            profile_dir=self.profile_dir,
            session_file=self.session_file,
            allow_interactive=allow_interactive,
        )
        self._pow_queue = pow_queue or get_pow_queue()
        self._refresh_lock = threading.Lock()
        self._http = httpx.Client(
            base_url=BASE,
            headers=self._base_headers(),
            cookies=self.session.cookies,
            timeout=httpx.Timeout(120.0, read=300.0),
        )

    def _base_headers(self) -> dict:
        return {
            "authorization": f"Bearer {self.session.token}",
            "accept": "*/*",
            "content-type": "application/json",
            "user-agent": self.session.user_agent,
            "origin": BASE,
            "referer": f"{BASE}/",
            "x-app-version": "2.0.0",
            "x-client-version": "2.0.0",
            "x-client-platform": "web",
            "x-client-locale": "en_US",
            "x-client-bundle-id": "com.deepseek.chat",
            "x-client-timezone-offset": "19800",
        }

    def _recover_session(self) -> bool:
        """Thread-safe background headless recovery using the cached profile."""
        with self._refresh_lock:
            # 1. Did another concurrent thread already refresh the session?
            cached = Session.load(self.session_file)
            if cached and cached.token != self.session.token:
                self._apply_session(cached)
                return True

            # 2. Attempt headless refresh with Playwright
            try:
                new_session = refresh_session(
                    profile_dir=self.profile_dir,
                    session_file=self.session_file,
                )
                if new_session:
                    self._apply_session(new_session)
                    return True
            except Exception as e:
                print(f"[client] Automated session recovery failed: {e}")

            return False

    def _apply_session(self, session: Session) -> None:
        self.session = session
        self._http.headers.update(self._base_headers())
        self._http.cookies.update(session.cookies)

    def _is_auth_error_data(self, resp: httpx.Response) -> bool:
        try:
            data = resp.json()
            code = data.get("code")
            msg = str(data.get("msg") or "").lower()
            return code in (401, 40001, 40101) or "unauthorized" in msg or "login" in msg
        except Exception:
            return False

    def _request_with_auth_retry(self, method: str, path: str, **kwargs) -> httpx.Response:
        """Execute HTTP request with automated headless recovery on 401."""
        r = self._http.request(method, path, **kwargs)
        if r.status_code == 401 or (r.status_code == 200 and self._is_auth_error_data(r)):
            if self._recover_session():
                headers = kwargs.get("headers", {})
                headers["authorization"] = f"Bearer {self.session.token}"
                kwargs["headers"] = headers
                r = self._http.request(method, path, **kwargs)
            else:
                raise LoginRequired("Session expired and headless recovery failed.")
        if r.status_code == 429:
            raise UpstreamError("Messages too frequent (rate limit reached)", finish_reason="rate_limit_reached", status_code=429)
        if r.status_code == 403:
            raise UpstreamError(f"Access forbidden (WAF / verification): {r.text[:200]}", finish_reason="waf_block", status_code=403)
        r.raise_for_status()
        return r

    # --- protocol steps -----------------------------------------------------

    def create_chat_session(self) -> str:
        r = self._request_with_auth_retry("POST", "/api/v0/chat_session/create", json={})
        return _biz(r.json())["chat_session"]["id"]

    def _pow_header(self, target_path: str = COMPLETION_PATH) -> str:
        r = self._request_with_auth_retry(
            "POST", "/api/v0/chat/create_pow_challenge", json={"target_path": target_path}
        )
        challenge = _biz(r.json())["challenge"]
        return self._pow_queue.solve_challenge(challenge)

    # --- public API ---------------------------------------------------------

    def stream(
        self,
        prompt: str,
        conversation_id: Optional[str] = None,
        model: Optional[str] = None,
        thinking: bool = False,
        search: bool = False,
    ) -> "_Stream":
        """Stream a reply. Iterate it for text chunks; read `.conversation_id`
        afterwards to resume the thread. Pass an existing `conversation_id` to
        continue a previous conversation.

        `model` is DeepSeek's model_type wire value: "default" (Instant) or
        "expert"; it defaults to "default" on a NEW thread. It cannot be combined
        with `conversation_id` — a thread's model is fixed when it's created, so
        resuming keeps the original model. `thinking` enables DeepThink reasoning
        and `search` enables web search; both are independent of the model.
        """
        if conversation_id and model is not None:
            raise ValueError(
                "`model` cannot be set together with `conversation_id`; a thread's "
                "model is fixed when it is created. Pass `model` only on the first turn."
            )
        session_id, parent_id = _decode_cid(conversation_id)
        if session_id is None:
            # New thread: select the model (default when unspecified).
            session_id = self.create_chat_session()
            model_type: Optional[str] = model or DEFAULT_MODEL_TYPE
        else:
            # Resuming: let the existing thread's model stand (send no model_type).
            model_type = None
        return _Stream(self, prompt, session_id, parent_id, model_type, thinking, search)

    def chat(
        self,
        prompt: str,
        conversation_id: Optional[str] = None,
        model: Optional[str] = None,
        thinking: bool = False,
        search: bool = False,
    ) -> Reply:
        """Return the complete reply (`.text`), `.reasoning_text`, plus its `.conversation_id`."""
        s = self.stream(prompt, conversation_id=conversation_id,
                        model=model, thinking=thinking, search=search)
        reasoning_chunks = []
        content_chunks = []
        for chunk in s:
            if getattr(chunk, "chunk_type", "RESPONSE") == "THINK":
                reasoning_chunks.append(chunk)
            else:
                content_chunks.append(chunk)
        text = "".join(content_chunks)
        reasoning_text = "".join(reasoning_chunks) if reasoning_chunks else None
        return Reply(text=text, conversation_id=s.conversation_id, reasoning_text=reasoning_text)

    def close(self) -> None:
        self._http.close()


class _Stream:
    """Iterable of reply-text chunks. After it's consumed, `.conversation_id`
    holds the token for resuming the conversation."""

    def __init__(self, client: "DeepSeekClient", prompt: str, session_id: str,
                 parent_id: Optional[int], model: Optional[str],
                 thinking: bool, search: bool):
        self._client = client
        self._prompt = prompt
        self._session_id = session_id
        self._parent_id = parent_id
        self._model = model
        self._thinking = thinking
        self._search = search
        self._message_id: Optional[int] = None

    def __iter__(self) -> Iterator[Chunk]:
        body = {
            "chat_session_id": self._session_id,
            "parent_message_id": self._parent_id,
            "prompt": self._prompt,
            "ref_file_ids": [],
            "thinking_enabled": self._thinking,
            "search_enabled": self._search,
            "action": None,
            "preempt": False,
        }
        # Only select a model on a new thread; on resume the thread keeps its own.
        if self._model is not None:
            body["model_type"] = self._model

        # PoW challenges are short-lived, so solve right before the request.
        headers = {"x-ds-pow-response": self._client._pow_header()}
        meta: dict = {}

        attempt = 0
        while attempt < 2:
            attempt += 1
            req = self._client._http.build_request(
                "POST", COMPLETION_PATH, json=body, headers=headers
            )
            resp = self._client._http.send(req, stream=True)
            if resp.status_code == 401 and attempt == 1:
                resp.close()
                if self._client._recover_session():
                    headers = {
                        "authorization": f"Bearer {self._client.session.token}",
                        "x-ds-pow-response": self._client._pow_header(),
                    }
                    continue
                else:
                    raise LoginRequired("Session expired and background recovery failed. Please log in.")
            if resp.status_code == 429:
                resp.close()
                raise UpstreamError("Messages too frequent (rate limit reached)", finish_reason="rate_limit_reached", status_code=429)
            if resp.status_code == 403:
                text = resp.read().decode("utf-8", errors="replace")
                resp.close()
                if "waf" in text.lower() or "challenge" in text.lower() or "human" in text.lower():
                    raise UpstreamError("DeepSeek AWS WAF / human verification check triggered.", finish_reason="waf_block", status_code=403)
                raise UpstreamError(f"Upstream returned 403: {text[:200]}", status_code=403)

            resp.raise_for_status()
            try:
                for chunk_type, text in _parse_sse(resp.iter_lines(), meta):
                    yield Chunk(text, chunk_type)
            finally:
                resp.close()
            break

        if meta.get("message_id") is not None:
            self._message_id = meta["message_id"]

    @property
    def conversation_id(self) -> str:
        return _encode_cid(self._session_id, self._message_id)


def _find_fragments(data) -> Iterator[dict]:
    """Recursively yield all fragment dicts containing 'type' and ('id' or 'content')."""
    if isinstance(data, dict):
        if "type" in data and ("id" in data or "content" in data):
            yield data
        else:
            for val in data.values():
                yield from _find_fragments(val)
    elif isinstance(data, list):
        for item in data:
            yield from _find_fragments(item)


def _parse_sse(lines, meta: Optional[dict] = None) -> Iterator[Tuple[str, str]]:
    """Turn DeepSeek's SSE completion stream into (chunk_type, text) tuples.

    The stream sends an initial snapshot frame whose `v` is the full response
    object (with `fragments[].content`), then a series of append frames:
      * {"p":"response/fragments/0/content","o":"APPEND","v":" thinking"}
      * {"p":"response/fragments/1/content","o":"APPEND","v":" what"}
      * {"v":"'s"}
    We track the active append path and emit fragment text for all fragments (THINK, RESPONSE),
    without eating the first token of any fragment!
    """
    active_path: Optional[str] = None
    active_type: str = "RESPONSE"
    processed_fragment_ids = set()

    for line in lines:
        if not line or not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            continue

        if obj.get("type") == "error" or obj.get("finish_reason") == "rate_limit_reached":
            raise UpstreamError(
                str(obj.get("content") or "DeepSeek upstream error").strip(),
                str(obj.get("finish_reason") or ""),
            )

        v = obj.get("v")

        # Snapshot frame: capture assistant message ID
        if isinstance(v, dict) and "response" in v:
            if meta is not None:
                _capture_message_id(meta, v)

        # Path-setting append frame
        if "p" in obj:
            active_path = obj["p"]
            if meta is not None and active_path.endswith("message_id") and isinstance(v, int):
                meta["message_id"] = v

        # Process any new fragments found in the object
        for frag in _find_fragments(obj):
            frag_id = frag.get("id")
            frag_type = frag.get("type")
            content = frag.get("content")
            if frag_id is not None:
                if frag_id not in processed_fragment_ids:
                    processed_fragment_ids.add(frag_id)
                    if frag_type in ("THINK", "THINKING"):
                        active_type = "THINK"
                    elif frag_type == "RESPONSE":
                        active_type = "RESPONSE"
                    if content:
                        yield active_type, content
            elif frag_type:
                if frag_type in ("THINK", "THINKING"):
                    active_type = "THINK"
                elif frag_type == "RESPONSE":
                    active_type = "RESPONSE"
                if content:
                    yield active_type, content

        # Yield any text updates to content
        if isinstance(v, str) and active_path and active_path.endswith("content"):
            yield active_type, v


def _capture_message_id(meta: dict, snapshot: dict) -> None:
    """Best-effort: pull the assistant message_id out of a snapshot frame."""
    for container in (snapshot.get("response"), snapshot):
        if isinstance(container, dict):
            mid = container.get("message_id", container.get("id"))
            if isinstance(mid, int):
                meta["message_id"] = mid
                return
