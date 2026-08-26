"""Regression tests: Verify tree branching and new chat isolation matching the reported user issue."""

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
        # Simulate DeepSeek server message id assignment
        # If parent_message_id is None, it's root (msg 101, reply 102)
        # If parent_message_id is 102, reply is 104, etc.
        rid = (parent_message_id or 100) + 1

        if "My name is Petrig" in prompt:
            ans = "Your name is Petrig."
        elif "remember the string `*h#n3XBe8Y$SjJ92y4FX`" in prompt:
            ans = "understood"
        elif "what is my name again?" in prompt:
            if parent_message_id:
                ans = "Your name is Petrig."
            else:
                ans = "I don't know your name."
        elif "what string did i ask you to remember again?" in prompt:
            if parent_message_id == 103:
                ans = "You asked me to remember: `*h#n3XBe8Y$SjJ92y4FX`"
            else:
                ans = "You haven't asked me to remember any string."
        else:
            ans = f"Reply to {prompt[:20]}"

        yield {"event": "ready", "data": {"response_message_id": rid}}
        yield {
            "event": None,
            "data": {"p": "response/fragments", "o": "APPEND", "v": {"type": "RESPONSE", "content": ""}},
        }
        yield {"event": None, "data": {"v": ans}}

    async def aclose(self):
        pass


class TestUserScenario(unittest.TestCase):
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

    def test_user_branching_and_new_chat_scenario(self):
        # 1. Turn 1: My name is Petrig. Answer in one sentence only.
        msgs = [{"role": "user", "content": "My name is Petrig. Answer in one sentence only."}]
        r1 = self.client.post("/v1/chat/completions", json={"model": "deepseek-chat", "messages": msgs})
        self.assertEqual(r1.status_code, 200)
        ans1 = r1.json()["choices"][0]["message"]["content"]
        self.assertEqual(ans1, "Your name is Petrig.")
        self.assertIsNone(self.fake_client.recorded_calls[-1]["parent_message_id"])

        # 2. Turn 2: what is my name again? Answer in one sentence only.
        msgs.append({"role": "assistant", "content": ans1})
        msgs.append({"role": "user", "content": "what is my name again? Answer in one sentence only."})
        r2 = self.client.post("/v1/chat/completions", json={"model": "deepseek-chat", "messages": msgs})
        self.assertEqual(r2.status_code, 200)
        ans2 = r2.json()["choices"][0]["message"]["content"]
        self.assertEqual(ans2, "Your name is Petrig.")
        self.assertEqual(self.fake_client.recorded_calls[-1]["parent_message_id"], 101)

        # 3. Branch A: remember the string `*h#n3XBe8Y$SjJ92y4FX`. reply with "understood" only.
        branch_a = list(msgs)
        branch_a.append({"role": "assistant", "content": ans2})
        branch_a.append({"role": "user", "content": 'remember the string `*h#n3XBe8Y$SjJ92y4FX`. reply with "understood" only.'})
        r3_a = self.client.post("/v1/chat/completions", json={"model": "deepseek-chat", "messages": branch_a})
        self.assertEqual(r3_a.status_code, 200)
        ans3_a = r3_a.json()["choices"][0]["message"]["content"]
        self.assertEqual(ans3_a, "understood")
        self.assertEqual(self.fake_client.recorded_calls[-1]["parent_message_id"], 102)

        # 4. Branch B: replied at output "Your name is Petrig." (after Turn 2, BEFORE Turn 3)
        branch_b = list(msgs)
        branch_b.append({"role": "assistant", "content": ans2})
        branch_b.append({"role": "user", "content": "what string did i ask you to remember again?"})
        r3_b = self.client.post("/v1/chat/completions", json={"model": "deepseek-chat", "messages": branch_b})
        self.assertEqual(r3_b.status_code, 200)
        ans3_b = r3_b.json()["choices"][0]["message"]["content"]
        # Must branch from Turn 2 (parent_message_id == 102), NOT from Branch A (parent_message_id == 103)
        self.assertEqual(self.fake_client.recorded_calls[-1]["parent_message_id"], 102)
        self.assertEqual(ans3_b, "You haven't asked me to remember any string.")

        # 5. New chat 1: "what is my name again? Answer in one sentence only." on its own
        r_new_1 = self.client.post(
            "/v1/chat/completions",
            json={"model": "deepseek-chat", "messages": [{"role": "user", "content": "what is my name again? Answer in one sentence only."}]},
        )
        self.assertEqual(r_new_1.status_code, 200)
        ans_new_1 = r_new_1.json()["choices"][0]["message"]["content"]
        # Isolated new chat has no parent message id
        self.assertIsNone(self.fake_client.recorded_calls[-1]["parent_message_id"])
        self.assertEqual(ans_new_1, "I don't know your name.")

        # 6. New chat 2: "what string did i ask you to remember again?" on its own
        r_new_2 = self.client.post(
            "/v1/chat/completions",
            json={"model": "deepseek-chat", "messages": [{"role": "user", "content": "what string did i ask you to remember again?"}]},
        )
        self.assertEqual(r_new_2.status_code, 200)
        ans_new_2 = r_new_2.json()["choices"][0]["message"]["content"]
        self.assertIsNone(self.fake_client.recorded_calls[-1]["parent_message_id"])
        self.assertEqual(ans_new_2, "You haven't asked me to remember any string.")


if __name__ == "__main__":
    unittest.main()
