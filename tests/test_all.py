"""Comprehensive test suite for DeepSeek-API refactored server.

Tests:
- Tool calling prompt formatting and multi-turn thread handling
- Tool calling parsing across multiple formats (JSON, XML, ReAct, Call syntax)
- Streaming chunks with reasoning_content and content separation
- Non-blocking Proof-of-Work task queue concurrency & timeouts
- Upstream error handling (rate limit, WAF block, session expiry)
- FastAPI endpoints compatibility with OpenAI clients
"""

import json
import time
import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

import server.api as api
from deepseek.client import Chunk, Reply, UpstreamError
from deepseek.pow import (
    PoWChallengeExpiredError,
    PoWTimeoutError,
    PowTaskQueue,
)
from server.openai_format import (
    completion_response,
    extract_tool_calls,
    messages_to_prompt,
    stream_chunks,
    stream_chunks_with_tools,
    trim_at_role_boundary,
)
from server.schemas import ChatCompletionRequest, ChatMessage


class TestToolCallingAndPrompt(unittest.TestCase):
    def setUp(self):
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read file contents",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write file contents",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                        },
                        "required": ["path", "content"],
                    },
                },
            },
        ]

    def test_single_user_message_without_tools(self):
        msgs = [ChatMessage(role="user", content="Hello world")]
        prompt = messages_to_prompt(msgs)
        self.assertEqual(prompt, "Hello world")

    def test_messages_to_prompt_with_tools(self):
        msgs = [ChatMessage(role="user", content="Read test.txt")]
        prompt = messages_to_prompt(msgs, tools=self.tools)
        self.assertIn("# Tool calling", prompt)
        self.assertIn("read_file", prompt)
        self.assertIn("User: Read test.txt", prompt)
        self.assertTrue(prompt.endswith("Assistant:"))

    def test_multi_turn_with_assistant_tool_calls_and_results(self):
        msgs = [
            ChatMessage(role="user", content="Read test.txt"),
            ChatMessage(
                role="assistant",
                content="I will read the file.",
                tool_calls=[
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path": "test.txt"}',
                        },
                    }
                ],
            ),
            ChatMessage(
                role="tool",
                tool_call_id="call_123",
                name="read_file",
                content="file content: hello",
            ),
            ChatMessage(role="user", content="What did it say?"),
        ]
        prompt = messages_to_prompt(msgs, tools=self.tools)
        self.assertIn('Assistant: I will read the file.', prompt)
        self.assertIn('"name": "read_file"', prompt)
        self.assertIn('"path": "test.txt"', prompt)
        self.assertIn('Tool result (read_file): file content: hello', prompt)
        self.assertIn('User: What did it say?', prompt)

    def test_extract_fenced_json_tool_calls(self):
        raw = (
            "Here is the tool call:\n"
            "```json\n"
            '{"tool_calls": [{"name": "read_file", "arguments": {"path": "main.py"}}]}\n'
            "```"
        )
        calls, text = extract_tool_calls(raw, self.tools)
        self.assertIsNotNone(calls)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "read_file")
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"path": "main.py"})
        self.assertNotIn("```json", text)

    def test_extract_react_style_tool_calls(self):
        raw = (
            "I need to read the file.\n"
            "Action: read_file\n"
            'Action Input: {"path": "config.json"}\n'
        )
        calls, text = extract_tool_calls(raw, self.tools)
        self.assertIsNotNone(calls)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "read_file")
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"path": "config.json"})

    def test_extract_call_syntax(self):
        raw = 'Let me inspect read_file({"path": "sample.py"}) right now.'
        calls, text = extract_tool_calls(raw, self.tools)
        self.assertIsNotNone(calls)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "read_file")
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"path": "sample.py"})

    def test_extract_xml_style(self):
        raw = (
            '<tool_call name="read_file">\n'
            '<parameter name="path">index.js</parameter>\n'
            '</tool_call>'
        )
        calls, text = extract_tool_calls(raw, self.tools)
        self.assertIsNotNone(calls)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "read_file")
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"path": "index.js"})

    def test_reject_unoffered_tool_hallucination(self):
        raw = '```json\n{"tool_calls": [{"name": "format_hard_drive", "arguments": {}}]}\n```'
        # format_hard_drive is not in self.tools, but in strict=False for explicit envelope it passes to client
        # In prose / call syntax without envelope, it MUST be rejected:
        raw_prose = 'Action: format_hard_drive\nAction Input: {}\n'
        calls, text = extract_tool_calls(raw_prose, self.tools)
        self.assertIsNone(calls)
        self.assertIn("format_hard_drive", text)

    def test_trim_at_role_boundary(self):
        raw = (
            "I will check that.\n"
            "Tool result (read_file): fabricated result\n"
            "User: what next?"
        )
        trimmed = trim_at_role_boundary(raw)
        self.assertEqual(trimmed, "I will check that.")


class TestStreamingAndReasoning(unittest.TestCase):
    def test_stream_chunks_thinking_and_response(self):
        stream = [
            Chunk("Thinking process step 1...", "THINK"),
            Chunk(" step 2.", "THINK"),
            Chunk("Hello, ", "RESPONSE"),
            Chunk("how can I help?", "RESPONSE"),
        ]
        lines = list(stream_chunks("deepseek-reasoner", stream))
        full_output = "".join(lines)
        self.assertIn('"reasoning_content": "Thinking process step 1..."', full_output)
        self.assertIn('"reasoning_content": " step 2."', full_output)
        self.assertIn('"content": "Hello, "', full_output)
        self.assertIn('"content": "how can I help?"', full_output)
        self.assertIn("data: [DONE]", full_output)

    def test_stream_chunks_with_tools_and_reasoning(self):
        tools = [{"type": "function", "function": {"name": "calc", "parameters": {}}}]
        stream = [
            Chunk("Thinking...", "THINK"),
            Chunk('```json\n{"tool_calls": [{"name": "calc", "arguments": {"x": 1}}]}\n```', "RESPONSE"),
        ]
        lines = list(stream_chunks_with_tools("deepseek-reasoner", stream, tools=tools))
        full_output = "".join(lines)
        self.assertIn('"reasoning_content": "Thinking..."', full_output)
        self.assertIn('"tool_calls"', full_output)
        self.assertIn('"calc"', full_output)
        self.assertIn('"finish_reason": "tool_calls"', full_output)

    def test_completion_response_with_reasoning_and_tools(self):
        raw = '```json\n{"tool_calls": [{"name": "calc", "arguments": {"x": 5}}]}\n```'
        res = completion_response(
            "deepseek-reasoner",
            raw,
            "calculate 5",
            reasoning_content="I should use the calc tool.",
        )
        msg = res["choices"][0]["message"]
        self.assertEqual(res["choices"][0]["finish_reason"], "tool_calls")
        self.assertEqual(msg["reasoning_content"], "I should use the calc tool.")
        self.assertEqual(len(msg["tool_calls"]), 1)
        self.assertEqual(msg["tool_calls"][0]["function"]["name"], "calc")


class TestPowTaskQueue(unittest.TestCase):
    def test_queue_expiration_detection(self):
        queue = PowTaskQueue(max_workers=2, timeout=5.0)
        expired_challenge = {
            "algorithm": "DeepSeekHashV1",
            "challenge": "test",
            "salt": "salt",
            "difficulty": 1000,
            "expire_at": time.time() - 10,  # Already expired
            "signature": "sig",
            "target_path": "/api/v0/chat/completion",
        }
        with self.assertRaises(PoWChallengeExpiredError):
            queue.solve_challenge(expired_challenge)

    def test_queue_timeout_graceful(self):
        queue = PowTaskQueue(max_workers=1, timeout=0.2)
        # Mock solver acquisition to simulate all workers busy
        with patch.object(queue, "_get_solver", side_effect=PoWTimeoutError("Timed out")):
            challenge = {
                "algorithm": "DeepSeekHashV1",
                "challenge": "test",
                "salt": "salt",
                "difficulty": 1000,
                "expire_at": time.time() + 60,
                "signature": "sig",
                "target_path": "/api/v0/chat/completion",
            }
            with self.assertRaises(PoWTimeoutError):
                queue.solve_challenge(challenge)


class TestServerEndpoints(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(api.app)

    def test_healthz(self):
        res = self.client.get("/healthz")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("session", data)
        self.assertIn("pow_queue", data)
        self.assertIn("models", data)

    def test_root_health(self):
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "ok")

    def test_list_models_dual_endpoints(self):
        r1 = self.client.get("/v1/models")
        r2 = self.client.get("/models")
        self.assertEqual(r1.status_code, 200)
        self.assertEqual(r2.status_code, 200)
        ids = [m["id"] for m in r1.json()["data"]]
        self.assertIn("deepseek-chat", ids)
        self.assertIn("deepseek-expert", ids)
        self.assertIn("deepseek-reasoner", ids)
        self.assertIn("gpt-4o", ids)

    def test_model_retrieve(self):
        res = self.client.get("/v1/models/deepseek-reasoner")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["id"], "deepseek-reasoner")

    def test_unknown_model_404(self):
        res = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "non-existent-model",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        self.assertEqual(res.status_code, 404)
        self.assertIn("error", res.json())

    def test_empty_messages_400(self):
        res = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "deepseek-chat",
                "messages": [],
            },
        )
        self.assertEqual(res.status_code, 400)

    def test_chat_completions_with_mock_client(self):
        mock_client = MagicMock()
        mock_reply = Reply(
            text='```json\n{"tool_calls": [{"name": "search", "arguments": {"q": "python"}}]}\n```',
            conversation_id="session123:1",
            reasoning_text="I will search for python.",
        )
        mock_client.chat.return_value = mock_reply

        with patch("server.api.get_client", return_value=mock_client):
            res = self.client.post(
                "/v1/chat/completions",
                json={
                    "model": "deepseek-reasoner",
                    "messages": [{"role": "user", "content": "search python"}],
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "search",
                                "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
                            },
                        }
                    ],
                },
            )
            self.assertEqual(res.status_code, 200)
            data = res.json()
            choice = data["choices"][0]
            self.assertEqual(choice["finish_reason"], "tool_calls")
            self.assertEqual(choice["message"]["reasoning_content"], "I will search for python.")
            self.assertEqual(choice["message"]["tool_calls"][0]["function"]["name"], "search")

    def test_upstream_rate_limit_error_mapping(self):
        mock_client = MagicMock()
        mock_client.chat.side_effect = UpstreamError(
            "Messages too frequent. Try again later.",
            finish_reason="rate_limit_reached",
            status_code=429,
        )

        with patch("server.api.get_client", return_value=mock_client):
            res = self.client.post(
                "/v1/chat/completions",
                json={
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
            self.assertEqual(res.status_code, 429)
            self.assertIn("rate_limit_exceeded", res.json()["error"]["type"])
            self.assertIn("Retry-After", res.headers)

    def test_upstream_waf_block_mapping(self):
        mock_client = MagicMock()
        mock_client.chat.side_effect = UpstreamError(
            "DeepSeek AWS WAF / human verification check triggered.",
            finish_reason="waf_block",
            status_code=403,
        )

        with patch("server.api.get_client", return_value=mock_client):
            res = self.client.post(
                "/v1/chat/completions",
                json={
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
            self.assertEqual(res.status_code, 502)
            self.assertIn("upstream_waf_block", res.json()["error"]["type"])

    def test_embeddings_stub(self):
        res = self.client.post("/v1/embeddings", json={"input": "test text"})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(len(res.json()["data"]), 1)
        self.assertEqual(len(res.json()["data"][0]["embedding"]), 8)


if __name__ == "__main__":
    unittest.main()
