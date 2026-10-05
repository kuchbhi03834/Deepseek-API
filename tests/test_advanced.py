"""Advanced stress tests for OpenCode multi-turn loops, PoW concurrent worker pool, and session recovery."""

import json
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

import server.api as api
from deepseek.auth import Session
from deepseek.client import Chunk, DeepSeekClient, Reply, UpstreamError
from deepseek.pow import DeepSeekPow, PowTaskQueue
from server.openai_format import messages_to_prompt
from server.schemas import ChatCompletionRequest, ChatMessage


class TestOpenCodeMultiTurnToolLoop(unittest.TestCase):
    def test_full_opencode_multi_step_agent_interaction(self):
        """Simulate OpenCode driving a 3-step loop:
        1. User asks to investigate an issue
        2. Assistant calls bash command
        3. Tool returns bash output
        4. Assistant calls read_file
        5. Tool returns file content
        6. Assistant gives final answer
        """
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "bash",
                    "description": "Run bash command",
                    "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read file",
                    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
                },
            },
        ]

        messages = [
            ChatMessage(role="user", content="Fix the bug in src/main.py"),
            ChatMessage(
                role="assistant",
                content="Let me check git status first.",
                tool_calls=[{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "bash", "arguments": '{"cmd": "git status"}'},
                }],
            ),
            ChatMessage(
                role="tool",
                tool_call_id="call_1",
                name="bash",
                content="modified: src/main.py",
            ),
            ChatMessage(
                role="assistant",
                content="Now reading src/main.py.",
                tool_calls=[{
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path": "src/main.py"}'},
                }],
            ),
            ChatMessage(
                role="tool",
                tool_call_id="call_2",
                name="read_file",
                content="def main(): return 1/0",
            ),
            ChatMessage(role="user", content="Did you find the bug?"),
        ]

        prompt = messages_to_prompt(messages, tools=tools)

        # Assert all turns are preserved with role labels and tool results
        self.assertIn("User: Fix the bug in src/main.py", prompt)
        self.assertIn("Assistant: Let me check git status first.", prompt)
        self.assertIn('"name": "bash"', prompt)
        self.assertIn("Tool result (bash): modified: src/main.py", prompt)
        self.assertIn('"name": "read_file"', prompt)
        self.assertIn("Tool result (read_file): def main(): return 1/0", prompt)
        self.assertIn("User: Did you find the bug?", prompt)
        self.assertTrue(prompt.endswith("Assistant:"))


class TestConcurrentPoWTaskQueue(unittest.TestCase):
    def test_concurrent_threads_solving(self):
        """Ensure multiple threads can solve challenges concurrently without deadlocks."""
        queue = PowTaskQueue(max_workers=3, timeout=5.0)

        # Mock solver instances
        def mock_make_header(challenge):
            time.sleep(0.02)  # Simulate small solve time
            return f"header_for_{challenge['challenge']}"

        with patch.object(DeepSeekPow, "make_header", side_effect=mock_make_header):
            results = []
            errors = []

            def worker(idx):
                challenge = {
                    "algorithm": "DeepSeekHashV1",
                    "challenge": f"chal_{idx}",
                    "salt": "s",
                    "difficulty": 100,
                    "expire_at": time.time() + 30,
                    "signature": "sig",
                    "target_path": "/api/v0/chat/completion",
                }
                try:
                    res = queue.solve_challenge(challenge)
                    results.append(res)
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=3.0)

            self.assertEqual(len(errors), 0, f"Encountered errors: {errors}")
            self.assertEqual(len(results), 8)
            stats = queue.stats()
            self.assertLessEqual(stats["created_solvers"], 3)
            self.assertEqual(stats["active_solves"], 0)


class TestSessionRecovery(unittest.TestCase):
    @patch("deepseek.client.refresh_session")
    @patch("deepseek.client.get_session")
    def test_client_automated_recovery_on_401(self, mock_get_session, mock_refresh_session):
        fake_session_1 = Session(token="token_old", cookies={}, user_agent="test_ua", captured_at=time.time())
        fake_session_2 = Session(token="token_new", cookies={}, user_agent="test_ua", captured_at=time.time())

        mock_get_session.return_value = fake_session_1
        mock_refresh_session.return_value = fake_session_2

        client = DeepSeekClient(session=fake_session_1, allow_interactive=False)

        # Mock http request: first returns 401, second returns 200 with chat session id
        mock_resp_401 = MagicMock()
        mock_resp_401.status_code = 401

        mock_resp_200 = MagicMock()
        mock_resp_200.status_code = 200
        mock_resp_200.json.return_value = {
            "code": 0,
            "data": {"biz_data": {"chat_session": {"id": "new_session_id"}}},
        }

        with patch.object(client._http, "request", side_effect=[mock_resp_401, mock_resp_200]):
            session_id = client.create_chat_session()
            self.assertEqual(session_id, "new_session_id")
            # Verify that session was refreshed to fake_session_2
            self.assertEqual(client.session.token, "token_new")
            mock_refresh_session.assert_called_once()


if __name__ == "__main__":
    unittest.main()
