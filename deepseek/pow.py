"""
Proof-of-work solver for DeepSeek's chat completion endpoint.

DeepSeek gates `POST /api/v0/chat/completion` behind a proof-of-work header
(`x-ds-pow-response`). The PoW algorithm ("DeepSeekHashV1") is shipped as a
WebAssembly module the website loads from its own CDN
(fe-static.deepseek.com/.../sha3_wasm_bg.wasm). Rather than reimplement its
exact float64 hashing logic, we run DeepSeek's own module — the same code the
browser runs — inside the `wasmtime` sandbox (no file/network access).

Public API:
    solver = DeepSeekPow()                      # loads the wasm once
    header = solver.make_header(challenge_dict)  # -> base64 x-ds-pow-response value

`challenge_dict` is the `biz_data.challenge` object returned by
`POST /api/v0/chat/create_pow_challenge`.
"""

from __future__ import annotations

import base64
import json
import os
import queue
import struct
import threading
import time
from pathlib import Path
from typing import Optional

import wasmtime

WASM_PATH = Path(__file__).resolve().parent / "sha3_wasm_bg.wasm"

_SHARED_ENGINE = None
_SHARED_MODULE = None
_MODULE_LOCK = threading.Lock()


def _get_shared_module(wasm_path: Path):
    global _SHARED_ENGINE, _SHARED_MODULE
    if _SHARED_MODULE is None:
        with _MODULE_LOCK:
            if _SHARED_MODULE is None:
                engine = wasmtime.Engine()
                module = wasmtime.Module.from_file(engine, str(wasm_path))
                _SHARED_ENGINE = engine
                _SHARED_MODULE = module
    return _SHARED_ENGINE, _SHARED_MODULE


class DeepSeekPow:
    def __init__(self, wasm_path: Path = WASM_PATH):
        engine, module = _get_shared_module(wasm_path)
        self._store = wasmtime.Store(engine)
        self._inst = wasmtime.Instance(self._store, module, [])
        exp = self._inst.exports(self._store)
        self._memory: wasmtime.Memory = exp["memory"]
        self._solve = exp["wasm_solve"]
        self._malloc = exp["__wbindgen_export_0"]            # malloc(size, align)
        self._add_to_stack = exp["__wbindgen_add_to_stack_pointer"]

    def _write_str(self, text: str) -> tuple[int, int]:
        """malloc + copy a UTF-8 string into wasm memory; return (ptr, len)."""
        data = text.encode("utf-8")
        ptr = self._malloc(self._store, len(data), 1)
        base = self._memory.data_ptr(self._store)
        for i, b in enumerate(data):
            base[ptr + i] = b
        return ptr, len(data)

    def solve(self, challenge: str, prefix: str, difficulty: float) -> Optional[int]:
        """Return the integer PoW answer, or None if the module reports failure.

        Mirrors the website's wasm-bindgen call:
            wasm_solve(retptr, challenge_ptr, challenge_len,
                       prefix_ptr, prefix_len, difficulty)
        with a 16-byte return slot reserved on the shadow stack. The slot holds
        an i32 status flag at +0 and an f64 answer at +8.
        """
        retptr = self._add_to_stack(self._store, -16)
        try:
            c_ptr, c_len = self._write_str(challenge)
            p_ptr, p_len = self._write_str(prefix)
            self._solve(self._store, retptr, c_ptr, c_len, p_ptr, p_len, float(difficulty))

            mem = self._memory.data_ptr(self._store)
            status = struct.unpack("<i", bytes(mem[retptr:retptr + 4]))[0]
            value = struct.unpack("<d", bytes(mem[retptr + 8:retptr + 16]))[0]
        finally:
            self._add_to_stack(self._store, 16)

        if status == 0:
            return None
        return int(value)

    def make_header(self, challenge: dict) -> str:
        """Build the base64 `x-ds-pow-response` header value from a challenge dict."""
        prefix = f"{challenge['salt']}_{challenge['expire_at']}_"
        answer = self.solve(challenge["challenge"], prefix, challenge["difficulty"])
        if answer is None:
            raise RuntimeError("PoW solver returned no answer (challenge expired?)")
        payload = {
            "algorithm": challenge["algorithm"],
            "challenge": challenge["challenge"],
            "salt": challenge["salt"],
            "answer": answer,
            "signature": challenge["signature"],
            "target_path": challenge["target_path"],
        }
        raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        return base64.b64encode(raw).decode("utf-8")


class PoWTimeoutError(RuntimeError):
    """Raised when waiting in the PoW queue times out."""
    pass


class PoWChallengeExpiredError(RuntimeError):
    """Raised when a challenge has already expired before or during solving."""
    pass


class PowTaskQueue:
    """Non-blocking, thread-safe queue and worker pool for DeepSeek Proof-of-Work challenges.

    Replaces the previous global single lock with a pool of independent wasm solver
    instances, non-blocking queueing, graceful timeouts, and challenge expiration detection.
    """

    def __init__(self, max_workers: int = 4, timeout: float = 25.0, wasm_path: Path = WASM_PATH):
        self.max_workers = max(1, max_workers)
        self.timeout = timeout
        self.wasm_path = wasm_path
        self._pool: queue.Queue[DeepSeekPow] = queue.Queue()
        self._created_count = 0
        self._active_count = 0
        self._lock = threading.Lock()

    def _get_solver(self, timeout: float) -> DeepSeekPow:
        """Acquire a solver from the pool or create a new one if below max_workers."""
        try:
            return self._pool.get_nowait()
        except queue.Empty:
            pass

        with self._lock:
            if self._created_count < self.max_workers:
                solver = DeepSeekPow(self.wasm_path)
                self._created_count += 1
                return solver

        try:
            return self._pool.get(timeout=timeout)
        except queue.Empty:
            raise PoWTimeoutError(
                f"PoW queue timed out after {timeout:.1f}s. "
                "The server is experiencing high concurrency; please retry."
            )

    def _release_solver(self, solver: DeepSeekPow):
        self._pool.put(solver)

    def solve_challenge(self, challenge: dict, timeout: Optional[float] = None) -> str:
        """Solve a challenge dictionary and return the base64 x-ds-pow-response header.

        Safe against expired challenges, concurrent contention, and hung threads.
        """
        effective_timeout = timeout if timeout is not None else self.timeout
        start_time = time.time()

        expire_at = challenge.get("expire_at")
        if expire_at and time.time() >= expire_at:
            raise PoWChallengeExpiredError(f"PoW challenge already expired at {expire_at}")

        solver = self._get_solver(timeout=effective_timeout)
        with self._lock:
            self._active_count += 1
        try:
            if expire_at and (time.time() + 1.0) >= expire_at:
                raise PoWChallengeExpiredError("PoW challenge expired while waiting in queue")
            return solver.make_header(challenge)
        finally:
            with self._lock:
                self._active_count -= 1
            self._release_solver(solver)

    def stats(self) -> dict:
        """Diagnostic stats for health checks."""
        with self._lock:
            return {
                "max_workers": self.max_workers,
                "created_solvers": self._created_count,
                "active_solves": self._active_count,
                "idle_solvers": self._pool.qsize(),
            }


_GLOBAL_POW_QUEUE: Optional[PowTaskQueue] = None
_GLOBAL_POW_LOCK = threading.Lock()


def get_pow_queue(max_workers: int = 4, timeout: float = 25.0) -> PowTaskQueue:
    global _GLOBAL_POW_QUEUE
    if _GLOBAL_POW_QUEUE is None:
        with _GLOBAL_POW_LOCK:
            if _GLOBAL_POW_QUEUE is None:
                _GLOBAL_POW_QUEUE = PowTaskQueue(max_workers=max_workers, timeout=timeout)
    return _GLOBAL_POW_QUEUE

