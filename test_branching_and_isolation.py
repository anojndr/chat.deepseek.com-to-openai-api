"""Test conversation branching and isolation for new chats."""

import asyncio
import tempfile
import time
import unittest
from pathlib import Path
from fastapi.testclient import TestClient

from app.storage import Storage
from app.accounts import AccountPool
from app.conversations import ConversationManager
from app.turn import prepare_turn
import app.main as main_mod


class FakeDeepSeekClient:
    def __init__(self, token: str) -> None:
        self.token = token
        self.created_sessions: list[str] = []
        self.recorded_calls: list[dict] = []

    async def create_session(self) -> str:
        sid = f"sess_{len(self.created_sessions) + 1}"
        self.created_sessions.append(sid)
        return sid

    async def upload_file(self, *a, **k):
        return "file_123"

    async def stream_completion(self, *, prompt: str, chat_session_id: str, parent_message_id: int | None = None, **kwargs):
        self.recorded_calls.append({
            "prompt": prompt,
            "chat_session_id": chat_session_id,
            "parent_message_id": parent_message_id,
        })
        rid = (parent_message_id or 100) + 1
        ans = f"Reply to {prompt[:20]}"

        yield {"event": "ready", "data": {"response_message_id": rid}}
        yield {
            "event": None,
            "data": {"p": "response/fragments", "o": "APPEND", "v": {"type": "RESPONSE", "content": ""}},
        }
        yield {"event": None, "data": {"v": ans}}

    async def aclose(self):
        pass


class TestBranchingAndIsolation(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmpdir.name) / "test_data.sqlite"
        self.accounts_path = Path(self.tmpdir.name) / "accounts.txt"
        self.accounts_path.write_text('account 1\n{"userToken": "test_tok_1"}')

        self.storage = Storage(self.db_path)
        self.pool = AccountPool(self.accounts_path)

        class DummySolver:
            pass

        self.manager = ConversationManager(self.pool, DummySolver(), storage=self.storage)
        self.fake_client = FakeDeepSeekClient("test_tok_1")
        self.manager._clients["test_tok_1"] = self.fake_client

        main_mod._storage = self.storage
        main_mod._pool = self.pool
        main_mod._solver = DummySolver()
        main_mod._manager = self.manager
        self.client = TestClient(main_mod.app)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_new_chat_isolation(self):
        # First chat
        r1 = self.client.post(
            "/v1/chat/completions",
            json={"model": "deepseek-chat", "messages": [{"role": "user", "content": "My name is Petrig."}]},
        )
        self.assertEqual(r1.status_code, 200)
        self.assertEqual(len(self.fake_client.created_sessions), 1)
        sess1 = self.fake_client.created_sessions[0]
        self.assertEqual(self.fake_client.recorded_calls[-1]["chat_session_id"], sess1)
        self.assertIsNone(self.fake_client.recorded_calls[-1]["parent_message_id"])

        # Second chat (completely different initial question without history)
        r2 = self.client.post(
            "/v1/chat/completions",
            json={"model": "deepseek-chat", "messages": [{"role": "user", "content": "what is my name again?"}]},
        )
        self.assertEqual(r2.status_code, 200)
        # Should create a new session or not have parent_message_id
        self.assertEqual(len(self.fake_client.created_sessions), 2)
        sess2 = self.fake_client.created_sessions[1]
        self.assertEqual(self.fake_client.recorded_calls[-1]["chat_session_id"], sess2)
        self.assertIsNone(self.fake_client.recorded_calls[-1]["parent_message_id"])

    def test_branching_conversation(self):
        # Start turn 1
        messages = [{"role": "user", "content": "My name is Petrig."}]
        r1 = self.client.post(
            "/v1/chat/completions",
            json={"model": "deepseek-chat", "messages": messages},
        )
        self.assertEqual(r1.status_code, 200)
        ans1 = r1.json()["choices"][0]["message"]["content"]
        sess1 = self.fake_client.created_sessions[0]
        self.assertEqual(self.fake_client.recorded_calls[-1]["parent_message_id"], None)

        # Turn 2
        messages.append({"role": "assistant", "content": ans1})
        messages.append({"role": "user", "content": "what is my name again?"})
        r2 = self.client.post(
            "/v1/chat/completions",
            json={"model": "deepseek-chat", "messages": messages},
        )
        self.assertEqual(r2.status_code, 200)
        ans2 = r2.json()["choices"][0]["message"]["content"]
        self.assertEqual(self.fake_client.recorded_calls[-1]["parent_message_id"], 101)

        # Branch A: Add message A
        branch_a_msgs = list(messages)
        branch_a_msgs.append({"role": "assistant", "content": ans2})
        branch_a_msgs.append({"role": "user", "content": "remember the string ABC123XYZ"})
        r3_a = self.client.post(
            "/v1/chat/completions",
            json={"model": "deepseek-chat", "messages": branch_a_msgs},
        )
        self.assertEqual(r3_a.status_code, 200)
        self.assertEqual(self.fake_client.recorded_calls[-1]["parent_message_id"], 102)

        # Branch B: Alternate turn from Turn 2 (not Turn 3A)
        branch_b_msgs = list(messages)
        branch_b_msgs.append({"role": "assistant", "content": ans2})
        branch_b_msgs.append({"role": "user", "content": "what string did i ask you to remember?"})
        r3_b = self.client.post(
            "/v1/chat/completions",
            json={"model": "deepseek-chat", "messages": branch_b_msgs},
        )
        self.assertEqual(r3_b.status_code, 200)
        # Should parent to Turn 2 (102), not Turn 3A (103)
        self.assertEqual(self.fake_client.recorded_calls[-1]["parent_message_id"], 102)
        self.assertEqual(self.fake_client.recorded_calls[-1]["chat_session_id"], sess1)


if __name__ == "__main__":
    unittest.main()
