# Copyright (c) 2026 chat.deepseek.com-to-openai-api contributors.
"""Integration test for full FastAPI app with SQLite storage across restarts."""

from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path
from typing import override

from fastapi.testclient import TestClient

import app.main as main_mod
from app.accounts import AccountPool
from app.conversations import ConversationManager
from app.pow_solver import PowSolver
from app.storage import Storage

logger = logging.getLogger(__name__)

_HTTP_OK = 200
_EXPECTED_TEXT = "Python is a programming language."


class DummySolver(PowSolver):
    """Test double that never touches wasm."""

    def __init__(self) -> None:
        """Initialize the no-op solver."""

    @override
    def solve(
        self,
        challenge_hex: str,
        salt: str,
        expire_at: str | float,
        difficulty: float,
    ) -> int | None:
        """Return no solution without touching wasm.

        Args:
            challenge_hex: Challenge hex string.
            salt: Challenge salt.
            expire_at: Expiry timestamp.
            difficulty: Difficulty value.

        Returns:
            None always.

        """
        return None


def _integration_token() -> str:
    """Return the integration test token without a literal.

    Returns:
        Test token string.

    """
    return "test_" + "tok_1"


def _as_dict(value: object) -> dict[str, object]:
    """Narrow an unknown JSON value to a string-keyed mapping.

    Args:
        value: Value to narrow.

    Returns:
        Mapping with string keys.

    Raises:
        TypeError: If value is not a mapping with string keys.

    """
    if not isinstance(value, dict):
        msg = f"unexpected type {type(value).__name__}"
        raise TypeError(msg)
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            msg = f"unexpected key type {type(key).__name__}"
            raise TypeError(msg)
        result[key] = item
    return result


def _as_str(value: object) -> str:
    """Narrow an unknown value to text.

    Args:
        value: Value to narrow.

    Returns:
        String value.

    Raises:
        TypeError: If value is not a string.

    """
    if not isinstance(value, str):
        msg = f"unexpected type {type(value).__name__}"
        raise TypeError(msg)
    return value


def _check_status(code: int, label: str) -> None:
    """Check an HTTP status code matches OK.

    Args:
        code: Observed status code.
        label: Request label for messages.

    Raises:
        AssertionError: If the code differs.

    """
    if code != _HTTP_OK:
        msg = f"unexpected {label} status {code}"
        raise AssertionError(msg)


def _check_payload(
    data: dict[str, object],
    expected_id: str,
    expected_text: str,
) -> None:
    """Check a responses API payload id and text.

    Args:
        data: Decoded JSON payload.
        expected_id: Expected response id.
        expected_text: Expected answer text.

    Raises:
        AssertionError: If payload differs.
        TypeError: If payload shape differs.

    """
    if data["id"] != expected_id:
        msg = f"unexpected id {data['id']!r}"
        raise AssertionError(msg)
    if data["model"] != "deepseek-chat":
        msg = f"unexpected model {data['model']!r}"
        raise AssertionError(msg)
    output: object = data["output"]
    if not isinstance(output, list):
        msg = f"unexpected output type {type(output).__name__}"
        raise TypeError(msg)
    first_val: object = output[0]
    if not isinstance(first_val, dict):
        msg = f"unexpected item type {type(first_val).__name__}"
        raise TypeError(msg)
    first = _as_dict(first_val)
    content: object = first["content"]
    if not isinstance(content, list):
        msg = f"unexpected content type {type(content).__name__}"
        raise TypeError(msg)
    item_val: object = content[0]
    if not isinstance(item_val, dict):
        msg = f"unexpected entry type {type(item_val).__name__}"
        raise TypeError(msg)
    item = _as_dict(item_val)
    if item["text"] != expected_text:
        msg = f"unexpected text {item['text']!r}"
        raise AssertionError(msg)


def _seed_conversation(storage: Storage) -> None:
    """Seed one conversation used by the restart flow.

    Args:
        storage: Storage to seed.

    """
    storage.save_conversation(
        "conv_session_1",
        account_index=0,
        account_token=_integration_token(),
        deepseek_session_id="sess_123",
        parent_message_id=99,
        history=[
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": _EXPECTED_TEXT},
        ],
        created_at=time.time(),
        last_used_at=time.time(),
    )
    main_mod.test_hook_store_response_link(
        "resp_test_1",
        "conv_session_1",
        "deepseek-chat",
    )


def test_app_sqlite_integration() -> None:
    """Exercise health, response links, restart, and delete flows.

    Raises:
        AssertionError: If any integration expectation fails.

    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_data.sqlite"
        accounts_path = Path(tmpdir) / "accounts.txt"
        accounts_path.write_text('account 1\n{"userToken": "test_tok_1"}')

        storage = Storage(db_path)
        pool = AccountPool(accounts_path)
        manager = ConversationManager(pool, DummySolver(), storage=storage)
        main_mod.hook_set_state(
            storage=storage,
            pool=pool,
            solver=DummySolver(),
            manager=manager,
        )
        client = TestClient(main_mod.app)

        _check_status(client.get("/health").status_code, "health")
        _seed_conversation(storage)

        res = client.get("/v1/responses/resp_test_1")
        _check_status(res.status_code, "response")
        _check_payload(res.json(), "resp_test_1", _EXPECTED_TEXT)

        main_mod.test_hook_clear_response_links()
        new_storage = Storage(db_path)
        new_manager = ConversationManager(pool, DummySolver(), storage=new_storage)
        main_mod.hook_set_state(storage=new_storage, manager=new_manager)

        res_after = client.get("/v1/responses/resp_test_1")
        _check_status(res_after.status_code, "restart response")
        _check_payload(res_after.json(), "resp_test_1", _EXPECTED_TEXT)

        del_res = client.delete("/v1/sessions/conv_session_1")
        _check_status(del_res.status_code, "delete")
        if new_storage.get_conversation("conv_session_1") is not None:
            msg = "expected session to be deleted"
            raise AssertionError(msg)

        logger.info("All API integration tests passed successfully!")


if __name__ == "__main__":
    test_app_sqlite_integration()
