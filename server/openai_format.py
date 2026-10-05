"""Translate between OpenAI's chat-completions shapes and our DeepSeek client.

DeepSeek's protocol has no system/role channel — just a single `prompt` string.
So we flatten the OpenAI `messages` array into one prompt, and wrap DeepSeek's
text output back into OpenAI response/stream objects.

Tool calling is EMULATED. The web chat has no native function-calling channel,
so when a request carries `tools` we:
  1. Describe them in the prompt with a strict JSON output contract.
  2. Parse the model's reply (JSON, ReAct prose, XML, or call syntax) back into
     OpenAI `tool_calls`.
  3. Support multi-turn agent loops by serialising past tool calls and tool results.
"""

from __future__ import annotations

import ast
import json
import logging
import os
import re
import time
import uuid
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from .config import DEBUG_REQUESTS
from .schemas import ChatMessage

log = logging.getLogger("uvicorn.error")

_ROLE_LABELS = {"system": "System", "user": "User", "assistant": "Assistant"}

# The contract we ask the model to follow when tools are available. Kept blunt
# and example-led: the failure mode we're fixing is the model *describing* a
# call instead of emitting one.
_TOOL_PROTOCOL = """\
# Tool calling

You have access to the tools listed below. They are the ONLY tools that exist.

To call one or more tools, reply with a single JSON object and NOTHING else —
no prose before it, no explanation after it, in exactly this shape:

```json
{"tool_calls": [{"name": "<tool name>", "arguments": {"<arg>": "<value>"}}]}
```

Rules, all of them mandatory:
- Use ONLY tool names from the list. Never invent a tool that is not listed.
- `arguments` must be a JSON object matching that tool's parameter schema.
- NEVER write a call as prose or as code. Lines like "Action: some_tool",
  "Action Input: {...}", `some_tool({"arg": "value"})`, or
  `<some_tool arg="value">` are not tool calls and do nothing. Only the JSON
  object above is a tool call.
- NEVER write out, guess, or imagine a tool's result. Stop after the JSON; the
  real result comes back to you in the next turn.
- NEVER announce a call in words. "I'll read the file now", "Let me check
  that", "First I need to open it" — these do nothing, and the turn ends there.
  If you intend to use a tool, the JSON object IS your entire reply, right now.
- When you are done using tools and want to answer the user, reply with normal
  text and no JSON block.

## Available tools

"""


def _text_of(content) -> str:
    """Extract plain text from a message's content (string or list-of-parts)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for p in content:
        if isinstance(p, dict) and p.get("type") == "text":
            parts.append(p.get("text", ""))
    return "\n".join(parts)


def _function_of(tool: dict) -> dict:
    """The function spec of a tool entry, tolerating the flat (pre-`tools`) shape."""
    if isinstance(tool, dict) and isinstance(tool.get("function"), dict):
        return tool["function"]
    return tool if isinstance(tool, dict) else {}


def tool_names(tools) -> List[str]:
    names = []
    for t in tools or []:
        n = _function_of(t).get("name")
        if n:
            names.append(n)
    return names


def tools_preamble(tools, tool_choice=None) -> str:
    """Render the tool schemas plus the output contract as prompt text."""
    blocks = []
    for t in tools or []:
        fn = _function_of(t)
        name = fn.get("name")
        if not name:
            continue
        spec = {
            "name": name,
            "description": fn.get("description", ""),
            "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
        }
        blocks.append(json.dumps(spec, ensure_ascii=False))
    if not blocks:
        return ""

    text = _TOOL_PROTOCOL + "\n\n".join(blocks)

    # tool_choice steers whether a call is optional, mandatory, or pinned.
    if isinstance(tool_choice, dict):
        forced = _function_of(tool_choice).get("name")
        if forced:
            text += (f"\n\nFor this turn you MUST call `{forced}` and no other tool. "
                     "Reply with the JSON object only.")
    elif tool_choice == "required":
        text += ("\n\nFor this turn you MUST call one of the tools above. "
                 "Reply with the JSON object only.")
    elif tool_choice == "none":
        text += ("\n\nFor this turn do NOT call any tool. Answer the user in "
                 "plain text.")
    return text


def _render_assistant_tool_calls(calls) -> str:
    """Replay a past assistant tool call in the same format we ask the model for."""
    rendered = []
    for c in calls or []:
        fn = c.get("function", {}) if isinstance(c, dict) else {}
        args = fn.get("arguments", "{}")
        if isinstance(args, str):
            try:
                args = json.loads(args or "{}")
            except (ValueError, TypeError):
                args = {"_raw": args}
        rendered.append({"name": fn.get("name") or c.get("name"), "arguments": args})
    body = json.dumps({"tool_calls": rendered}, ensure_ascii=False)
    return f"```json\n{body}\n```"


def messages_to_prompt(messages: List[ChatMessage], tools=None,
                       tool_choice=None) -> str:
    """Flatten a chat history into a single prompt DeepSeek can answer.

    A lone user message with no tools is sent verbatim. Anything else — system
    prompts, multi-turn, tool results — is serialised with role labels and a
    trailing 'Assistant:' cue so the model continues in the right voice.
    """
    preamble = tools_preamble(tools, tool_choice)

    if len(messages) == 1 and messages[0].role == "user" and not preamble:
        return _text_of(messages[0].content)

    lines = []
    for m in messages:
        if m.role in ("tool", "function"):
            # The answer to a call we made. Label it with the tool's name or id when
            # the caller gave us one, so the model can match it to its request.
            who = m.name or m.tool_call_id or "tool"
            lines.append(f"Tool result ({who}): {_text_of(m.content)}")
            continue

        label = _ROLE_LABELS.get(m.role, m.role.capitalize())
        body = _text_of(m.content)
        if m.role == "assistant" and m.tool_calls:
            calls = _render_assistant_tool_calls(m.tool_calls)
            body = f"{body}\n{calls}".strip() if body else calls
        lines.append(f"{label}: {body}")

    # The contract goes LAST, immediately before the generation point.
    if preamble:
        lines.append(preamble)

    lines.append("Assistant:")
    return "\n\n".join(lines)


# --- parsing the model's reply back into tool calls -------------------------

_FENCE_RE = re.compile(r"```(?:json)?\s*(.+?)\s*```", re.DOTALL)

# Salvage path 1: ReAct prose ("Action: some_tool  Action Input: {...}").
_REACT_RE = re.compile(
    r"Action\s*:\s*[`\"']?([\w.\-]+)[`\"']?\s*[\r\n]*"
    r"Action\s*Input\s*:\s*(\{.*?\})\s*(?:$|[\r\n])",
    re.DOTALL | re.IGNORECASE,
)

# Salvage path 2: `tool_name({"arg": "value"})` call syntax.
_CALL_SYNTAX_RE = re.compile(r"([A-Za-z_][\w.\-]*)\s*\(\s*(?=[{)])")

# Salvage path 3: XML tags e.g. `<read_file path="a.txt">` or `<tool_call name="...">`.
_XML_TAG_RE = re.compile(r"<\s*([A-Za-z_][\w.\-]*)\s*((?:[\w.\-]+\s*=\s*\"[^\"]*\"\s*)*)/?>")
_XML_ATTR_RE = re.compile(r"([\w.\-]+)\s*=\s*\"([^\"]*)\"")


def _resolve_name(name: str, allowed: List[str]) -> Optional[str]:
    """Map a model-written tool name onto an offered one, or None.

    Exact match first, then a unique suffix match — models routinely drop a
    namespace prefix (`mcp__server__do_thing` -> `do_thing`) or keep it when the
    caller offered the short form.
    """
    if name in allowed:
        return name
    hits = [a for a in allowed if a.endswith(name) or name.endswith(a)]
    return hits[0] if len(hits) == 1 else None


def _loads(fragment: str):
    """Parse a JSON object, falling back to Python literal syntax."""
    if not fragment:
        return None
    try:
        return json.loads(fragment)
    except (ValueError, TypeError):
        pass
    try:
        return ast.literal_eval(fragment)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return None


def _scan_object(text: str, start: int) -> Tuple[Optional[str], Optional[str]]:
    """Scan one JSON object starting at `text[start]` == '{'.

    Returns (complete, partial).
    """
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1], None
    return None, text[start:]


def _repair_truncated(fragment: str) -> Optional[str]:
    """Close a cut-off JSON object so it can be parsed, or None."""
    if '"tool_calls"' not in fragment and '"name"' not in fragment:
        return None

    stack = []
    in_string = False
    escaped = False
    for ch in fragment:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]" and stack:
            stack.pop()
    if not stack:
        return None

    repaired = fragment
    if in_string:
        opening = repaired.rfind('"')
        repaired = repaired[:opening]
        repaired = re.sub(r',?\s*"[^"]*"\s*:\s*$', "", repaired)
    else:
        repaired = re.sub(r',?\s*"[^"]*"\s*:\s*(?:-?\d+(?:\.\d*)?(?:[eE][-+]?\d*)?'
                          r'|t(?:r(?:u(?:e)?)?)?|f(?:a(?:l(?:s(?:e)?)?)?)?'
                          r'|n(?:u(?:l(?:l)?)?)?)?$', "", repaired)
    repaired = re.sub(r"[\s`]*$", "", repaired)
    repaired = re.sub(r",\s*$", "", repaired)
    for opener in reversed(stack):
        repaired += "}" if opener == "{" else "]"
    return repaired


def _iter_json_candidates(text: str) -> Iterable[str]:
    """Yield substrings of `text` that might be the tool-call JSON object."""
    for m in _FENCE_RE.finditer(text):
        body = m.group(1)
        yield body
        repaired = _repair_truncated(body)
        if repaired:
            yield repaired
    partials = []
    i = 0
    while i < len(text):
        if text[i] == "{":
            complete, partial = _scan_object(text, i)
            if complete:
                yield complete
                i += len(complete)
                continue
            if partial:
                partials.append(partial)
            break
        i += 1
    for partial in partials:
        repaired = _repair_truncated(partial)
        if repaired:
            yield repaired


_WRAPPER_NAMES = ("tool_call", "tool", "function", "function_call", "invoke", "call")


def _unwrap_call(fn: dict, allowed: List[str], depth: int = 0,
                 strict: bool = True) -> Optional[dict]:
    """Peel generic wrappers off a call entry until the real one is found."""
    if depth > 3 or not isinstance(fn, dict):
        return None
    name = fn.get("name")
    if not isinstance(name, str):
        return None
    args = fn.get("arguments", fn.get("parameters", {}))
    if isinstance(args, str):
        try:
            args = json.loads(args or "{}")
        except (ValueError, TypeError):
            args = {}
    if not isinstance(args, dict):
        return None

    resolved = _resolve_name(name, allowed) if allowed else name
    if isinstance(args.get("name"), str) and (not resolved or name.lower() in _WRAPPER_NAMES):
        inner = _unwrap_call(args, allowed, depth + 1, strict)
        if inner:
            return inner
    if not resolved:
        if strict or name.lower() in _WRAPPER_NAMES:
            return None
        resolved = name
    return {"name": resolved, "arguments": args}


def _normalise_calls(obj, allowed: List[str], strict: bool = True) -> Optional[List[dict]]:
    """Pull a list of {name, arguments} out of a parsed JSON object, or None."""
    wrapped = isinstance(obj, dict) and ("tool_calls" in obj or "tool_call" in obj)
    if isinstance(obj, dict):
        raw = obj.get("tool_calls") or obj.get("tool_call") or obj
    else:
        raw = obj
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        return None

    calls = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        fn = item.get("function") if isinstance(item.get("function"), dict) else item
        if not wrapped and not isinstance(
                fn.get("arguments", fn.get("parameters")), dict):
            continue
        unwrapped = _unwrap_call(fn, allowed, strict=strict)
        if not unwrapped:
            continue
        name, args = unwrapped["name"], unwrapped["arguments"]
        calls.append({
            "id": "call_" + uuid.uuid4().hex[:24],
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(args, ensure_ascii=False),
            },
        })
    return calls or None


def _extract_inner_calls(text: str, allowed: List[str]) -> Tuple[Optional[List[dict]], str]:
    """Pick individual call objects out of a malformed envelope."""
    calls, spans, i = [], [], 0
    while i < len(text):
        if text[i] != "{":
            i += 1
            continue
        complete, partial = _scan_object(text, i)
        frag = complete or _repair_truncated(partial or "") or ""
        obj = _loads(frag) if frag else None
        if (isinstance(obj, dict) and isinstance(obj.get("name"), str)
                and isinstance(obj.get("arguments", obj.get("parameters")), dict)):
            one = _normalise_calls([obj], allowed, strict=False)
            if one:
                calls.extend(one)
                spans.append((i, i + len(frag)))
                i += len(frag)
                continue
        i += 1
    if not calls:
        return None, text
    leftover = text
    for start, stop in reversed(spans):
        leftover = leftover[:start] + leftover[stop:]
    leftover = re.sub(r"```(?:json)?", "", leftover)
    leftover = re.sub(r'[\s,\[\]{}]*$', "", leftover).strip()
    return calls, leftover


_ROLE_BOUNDARY_RE = re.compile(
    r"^\s*(?:Tool result\s*\(|User\s*:|System\s*:|Assistant\s*:|Human\s*:)",
    re.MULTILINE,
)


def trim_at_role_boundary(text: str) -> str:
    """Drop any continuation past the model's own turn."""
    if not text:
        return text
    m = _ROLE_BOUNDARY_RE.search(text)
    if not m or m.start() == 0:
        return text
    return text[:m.start()].rstrip()


def _extract_react_calls(text: str, allowed: List[str]) -> Tuple[Optional[List[dict]], str]:
    """Parse ReAct-style `Action:` / `Action Input:` prose into tool calls."""
    calls, spans = [], []
    for m in _REACT_RE.finditer(text):
        name = _resolve_name(m.group(1), allowed)
        if not name:
            continue
        try:
            args = json.loads(m.group(2))
        except (ValueError, TypeError):
            continue
        if not isinstance(args, dict):
            continue
        calls.append({
            "id": "call_" + uuid.uuid4().hex[:24],
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
        })
        spans.append(m.span())
    if not calls:
        return None, text
    leftover = text
    for start, end in reversed(spans):
        leftover = leftover[:start] + leftover[end:]
    return calls, leftover.strip()


def _extract_call_syntax(text: str, allowed: List[str]) -> Tuple[Optional[List[dict]], str]:
    """Parse `tool_name({...})` / `tool_name()` invocations into tool calls."""
    calls, spans = [], []
    for m in _CALL_SYNTAX_RE.finditer(text):
        name = _resolve_name(m.group(1), allowed)
        if not name:
            continue
        end = m.end()
        if text[end:end + 1] == ")":          # no-argument call
            args = {}
            end += 1
        else:
            obj, partial = _scan_object(text, end)
            if not obj:
                obj = _repair_truncated(partial or "") or ""
            args = _loads(obj)
            if not isinstance(args, dict):
                continue
            end += len(obj)
            if text[end:end + 1] == ")":
                end += 1
        calls.append({
            "id": "call_" + uuid.uuid4().hex[:24],
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
        })
        spans.append((m.start(), end))
    if not calls:
        return None, text
    leftover = text
    for start, stop in reversed(spans):
        leftover = leftover[:start] + leftover[stop:]
    return calls, leftover.strip()


def _coerce(value: str):
    """Turn an XML attribute string into a JSON scalar where it clearly is one."""
    try:
        parsed = json.loads(value)
    except (ValueError, TypeError):
        return value
    return parsed if isinstance(parsed, (int, float, bool, list, dict)) else value


def _extract_xml_calls(text: str, allowed: List[str]) -> Tuple[Optional[List[dict]], str]:
    """Parse `<tool_name attr="value">` and `<tool_call name="...">`."""
    calls, spans = [], []
    # Check for parameter-style <tool_call name="..."> <parameter name="...">val</parameter> </tool_call>
    xml_tc_pattern = r'<tool_call(?:\s+name=["\']([^"\']+)["\'])?\s*>(.*?)</tool_call>'
    for m in list(re.finditer(xml_tc_pattern, text, re.DOTALL)):
        raw_name = m.group(1)
        inner = m.group(2)
        if not raw_name:
            nm = re.search(r'<name>(.*?)</name>', inner)
            if nm:
                raw_name = nm.group(1).strip()
        name = _resolve_name(raw_name, allowed) if raw_name else None
        if name:
            args = {}
            for p in re.findall(r'<parameter\s+name=["\']([^"\']+)["\'][^>]*>(.*?)</parameter>', inner, re.DOTALL):
                args[p[0]] = _coerce(p[1].strip())
            calls.append({
                "id": "call_" + uuid.uuid4().hex[:24],
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
            })
            spans.append(m.span())

    for m in _XML_TAG_RE.finditer(text):
        tag, attrs = m.group(1), m.group(2)
        name = _resolve_name(tag, allowed)
        end = m.end()
        if not name:
            if tag.lower() not in ("tool_call", "tool", "function_call", "invoke"):
                continue
            rest = text[m.end():]
            inner = re.match(r"\s*([\w.\-]+)", rest)
            if not inner:
                continue
            name = _resolve_name(inner.group(1), allowed)
            if not name:
                continue
            end = m.end() + inner.end()
            attrs = ""
        args = {k: _coerce(v) for k, v in _XML_ATTR_RE.findall(attrs)}
        calls.append({
            "id": "call_" + uuid.uuid4().hex[:24],
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
        })
        spans.append((m.start(), end))

    if not calls:
        return None, text
    leftover = text
    for start, stop in reversed(spans):
        leftover = leftover[:start] + leftover[stop:]
    leftover = re.sub(r"</\s*[\w.\-]+\s*>", "", leftover).strip()
    return calls, leftover


def extract_tool_calls(text: str, tools=None) -> Tuple[Optional[List[dict]], str]:
    """Split a reply into (tool_calls, leftover_text).

    Returns (None, text) when the model answered in normal prose.
    """
    if not text:
        return None, text
    allowed = tool_names(tools) if tools else []
    text = trim_at_role_boundary(text)
    for candidate in _iter_json_candidates(text):
        obj = _loads(candidate)
        if obj is None:
            continue
        calls = _normalise_calls(obj, allowed, strict=bool(allowed))
        if calls:
            leftover = text.replace(candidate, "", 1)
            leftover = _FENCE_RE.sub("", leftover).strip()
            return calls, leftover
    if allowed:
        calls, leftover = _extract_inner_calls(text, allowed)
        if calls:
            return calls, leftover
        calls, leftover = _extract_react_calls(text, allowed)
        if calls:
            return calls, leftover
        calls, leftover = _extract_call_syntax(text, allowed)
        if calls:
            return calls, leftover
        return _extract_xml_calls(text, allowed)
    else:
        calls, leftover = _extract_inner_calls(text, allowed=[])
        if calls:
            return calls, leftover
        return _extract_xml_calls(text, allowed=[])


def _now() -> int:
    return int(time.time())


def _id() -> str:
    return "chatcmpl-" + uuid.uuid4().hex


def _est_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token)."""
    return max(1, len(text or "") // 4)


def completion_response(
    model: str,
    content: Optional[str],
    prompt: str,
    conversation_id: Optional[str] = None,
    reasoning_content: Optional[str] = None,
    tool_calls: Optional[List[dict]] = None,
    tools: Optional[List[dict]] = None,
) -> dict:
    """A full (non-streaming) OpenAI chat.completion object."""
    if tool_calls is None and content:
        extracted_calls, leftover = extract_tool_calls(content, tools)
        if extracted_calls:
            tool_calls = extracted_calls
            content = leftover

    pt, ct = _est_tokens(prompt), _est_tokens(content or "")

    message: Dict[str, Any] = {"role": "assistant", "content": content or None}
    finish_reason = "stop"
    if tool_calls:
        message["tool_calls"] = tool_calls
        finish_reason = "tool_calls"
    if reasoning_content:
        message["reasoning_content"] = reasoning_content

    return {
        "id": _id(),
        "object": "chat.completion",
        "created": _now(),
        "model": model,
        "conversation_id": conversation_id,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "total_tokens": pt + ct,
        },
    }


def _frame(cid: str, created: int, model: str, delta: dict,
           finish=None, extra: dict = None) -> str:
    obj = {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    if extra:
        obj.update(extra)
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


def stream_chunks(model: str, stream: Iterable[Any], include_usage: bool = False,
                  prompt: str = "") -> Iterable[str]:
    """Yield OpenAI SSE lines (`data: {...}\\n\\n`) for a streamed completion."""
    cid, created = _id(), _now()
    pt = _est_tokens(prompt)
    ct = 0

    yield _frame(cid, created, model, {"role": "assistant", "content": ""})

    for d in stream:
        if d:
            chunk_type = getattr(d, "chunk_type", "RESPONSE")
            text = str(d)
            ct += _est_tokens(text)
            if chunk_type == "THINK":
                yield _frame(cid, created, model, {"reasoning_content": text})
            else:
                yield _frame(cid, created, model, {"content": text})

    conversation_id = getattr(stream, "conversation_id", None)
    yield _frame(cid, created, model, {}, finish="stop",
                 extra={"conversation_id": conversation_id})

    if include_usage:
        usage_obj = {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [],
            "usage": {
                "prompt_tokens": pt,
                "completion_tokens": ct,
                "total_tokens": pt + ct,
            },
        }
        yield f"data: {json.dumps(usage_obj, ensure_ascii=False)}\n\n"

    yield "data: [DONE]\n\n"


def stream_chunks_with_tools(model: str, stream: Iterable[Any], tools,
                             include_usage: bool = False, prompt: str = "") -> Iterable[str]:
    """Streaming variant for tool-enabled requests.

    Streams `THINK` chunks in real-time as `reasoning_content`, and buffers `RESPONSE`
    text until the tool call JSON is complete.
    """
    cid, created = _id(), _now()
    pt = _est_tokens(prompt)
    ct = 0

    yield _frame(cid, created, model, {"role": "assistant", "content": ""})

    buf_parts = []
    for d in stream:
        if d:
            chunk_type = getattr(d, "chunk_type", "RESPONSE")
            text = str(d)
            ct += _est_tokens(text)
            if chunk_type == "THINK":
                yield _frame(cid, created, model, {"reasoning_content": text})
            else:
                buf_parts.append(text)

    buf = "".join(buf_parts)
    conversation_id = getattr(stream, "conversation_id", None)
    calls, leftover = extract_tool_calls(buf, tools)

    if DEBUG_REQUESTS:
        log.warning(
            "reply(stream): calls=%s text=%r",
            [(c["function"]["name"], c["function"]["arguments"]) for c in calls or []],
            (leftover or buf)[:200])

    if calls:
        delta = {"tool_calls": [
            {
                "index": i,
                "id": c["id"],
                "type": "function",
                "function": c["function"],
            }
            for i, c in enumerate(calls)
        ]}
        if leftover:
            delta["content"] = leftover
        yield _frame(cid, created, model, delta)
        finish = "tool_calls"
    else:
        if buf:
            yield _frame(cid, created, model, {"content": buf})
        finish = "stop"

    yield _frame(cid, created, model, {}, finish=finish,
                 extra={"conversation_id": conversation_id})

    if include_usage:
        usage_obj = {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [],
            "usage": {
                "prompt_tokens": pt,
                "completion_tokens": ct,
                "total_tokens": pt + ct,
            },
        }
        yield f"data: {json.dumps(usage_obj, ensure_ascii=False)}\n\n"

    yield "data: [DONE]\n\n"
