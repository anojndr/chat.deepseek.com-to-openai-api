"""Integration test for full FastAPI app with SQLite storage across restarts."""

import os
import tempfile
import time
from pathlib import Path
from fastapi.testclient import TestClient

from app.storage import Storage
from app.accounts import AccountPool
from app.pow_solver import PowSolver
from app.conversations import ConversationManager
import app.main as main_mod


def test_app_sqlite_integration():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_data.sqlite"
        accounts_path = Path(tmpdir) / "accounts.txt"
        accounts_path.write_text('account 1\n{"userToken": "test_tok_1"}')

        # Setup storage and app components
        storage = Storage(db_path)
        pool = AccountPool(accounts_path)

        class DummySolver:
            pass

        manager = ConversationManager(pool, DummySolver(), storage=storage)

        # Monkeypatch main globals
        main_mod._storage = storage
        main_mod._pool = pool
        main_mod._solver = DummySolver()
        main_mod._manager = manager

        client = TestClient(main_mod.app)

        # 1. Health check
        resp = client.get("/health")
        assert resp.status_code == 200

        # 2. Store response link and a conversation
        storage.save_conversation(
            "conv_session_1",
            account_index=0,
            account_token="test_tok_1",
            deepseek_session_id="sess_123",
            parent_message_id=99,
            history=[
                {"role": "user", "content": "What is Python?"},
                {"role": "assistant", "content": "Python is a programming language."},
            ],
            created_at=time.time(),
            last_used_at=time.time(),
        )
        main_mod._store_response_link("resp_test_1", "conv_session_1", "deepseek-chat")

        # 3. Retrieve response via GET /v1/responses/resp_test_1
        res = client.get("/v1/responses/resp_test_1")
        assert res.status_code == 200
        data = res.json()
        assert data["id"] == "resp_test_1"
        assert data["model"] == "deepseek-chat"
        assert data["output"][0]["content"][0]["text"] == "Python is a programming language."

        # 4. Simulate complete server restart: clear in-memory maps and re-instantiate
        main_mod._response_links.clear()
        new_storage = Storage(db_path)
        new_manager = ConversationManager(pool, DummySolver(), storage=new_storage)
        main_mod._storage = new_storage
        main_mod._manager = new_manager

        # 5. Check response retrieval after restart (ensures _response_links loaded from sqlite)
        res_after = client.get("/v1/responses/resp_test_1")
        assert res_after.status_code == 200
        data_after = res_after.json()
        assert data_after["id"] == "resp_test_1"
        assert data_after["output"][0]["content"][0]["text"] == "Python is a programming language."

        # 6. Delete session endpoint
        del_res = client.delete("/v1/sessions/conv_session_1")
        assert del_res.status_code == 200
        assert new_storage.get_conversation("conv_session_1") is None

        print("All API integration tests passed successfully!")


if __name__ == "__main__":
    test_app_sqlite_integration()
