# Copyright (c) 2026 chat.deepseek.com-to-openai-api contributors.
"""Check vision uploads tolerate VISION model kind."""

from __future__ import annotations

import unittest
from typing import Any, override
from unittest.mock import AsyncMock, MagicMock, patch

from app.deepseek import DeepSeekClient
from app.pow_solver import PowSolver

_FILE_ID = "file-abc"
_VISION_KIND = "VISION"
_DEFAULT_KIND = "DEFAULT"
_POW_HEADER_VALUE = "x"
_FETCH_SUCCESS = "SUCCESS"
_TEST_FILENAME = "a.png"
_TEST_MIME = "image/png"


class DummySolver(PowSolver):
    """Refuse to solve PoW in tests."""

    def __init__(self) -> None:
        """Initialize without wasm."""

    @override
    def solve(
        self,
        challenge_hex: str,
        salt: str,
        expire_at: str | float,
        difficulty: float,
    ) -> int | None:
        """Return no solution.

        Returns:
            None: Never solves.

        """
        return None


def _upload_ok(
    file_id: str = _FILE_ID,
    model_kind: str = _VISION_KIND,
) -> dict[str, Any]:
    """Build successful upload payload.

    Returns:
        dict[str, Any]: Upload response payload.

    """
    return {
        "code": 0,
        "msg": "",
        "data": {
            "biz_code": 0,
            "biz_msg": "",
            "biz_data": {"id": file_id, "model_kind": model_kind},
        },
    }


def _fork_satisfied() -> dict[str, Any]:
    """Build fork satisfied payload.

    Returns:
        dict[str, Any]: Fork response payload.

    """
    return {
        "code": 0,
        "msg": "",
        "data": {
            "biz_code": 2,
            "biz_msg": "model kind satisfied",
            "biz_data": None,
        },
    }


def _make_client(
    upload_payload: dict[str, Any],
    fork_payload: dict[str, Any] | None = None,
    fork_raises: BaseException | None = None,
) -> tuple[DeepSeekClient, AsyncMock]:
    """Build client with stubbed transport.

    Returns:
        tuple[DeepSeekClient, AsyncMock]: Client and request mock.

    """
    client = DeepSeekClient("tok", DummySolver())
    pow_mock = AsyncMock(return_value=({"x-ds-pow-response": _POW_HEADER_VALUE}, {}))
    patch.object(client, "_get_pow", pow_mock).start()
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = upload_payload
    http = client.__dict__["_http"]
    patch.object(http, "post", new=AsyncMock(return_value=resp)).start()
    fetch_mock = AsyncMock(return_value=_FETCH_SUCCESS)
    patch.object(client, "_fetch_file_status", fetch_mock).start()
    if fork_raises is not None:
        request_mock = AsyncMock(side_effect=fork_raises)
    else:
        request_mock = AsyncMock(return_value=fork_payload)
    patch.object(client, "_request_json", request_mock).start()
    return client, request_mock


class VisionForkToleranceTest(unittest.IsolatedAsyncioTestCase):
    """Verify VISION uploads skip redundant fork."""

    @staticmethod
    async def test_already_vision_skips_fork() -> None:
        """Check VISION upload skips fork.

        Raises:
            AssertionError: If file id mismatches or fork runs.

        """
        client, request_mock = _make_client(_upload_ok(_FILE_ID, _VISION_KIND))
        try:
            fid = await client.upload_file(
                _TEST_FILENAME,
                b"raw",
                _TEST_MIME,
                vision=True,
            )
        finally:
            await client.aclose()
            patch.stopall()
        if fid != _FILE_ID:
            msg = f"{fid!r} != {_FILE_ID!r}"
            raise AssertionError(msg)
        request_mock.assert_not_awaited()

    @staticmethod
    async def test_fork_satisfied_tolerated() -> None:
        """Check satisfied fork still uploads.

        Raises:
            AssertionError: If file id mismatches.

        """
        client, _request_mock = _make_client(
            _upload_ok(_FILE_ID, _DEFAULT_KIND),
            fork_payload=_fork_satisfied(),
        )
        try:
            fid = await client.upload_file(
                _TEST_FILENAME,
                b"raw",
                _TEST_MIME,
                vision=True,
            )
        finally:
            await client.aclose()
            patch.stopall()
        if fid != _FILE_ID:
            msg = f"{fid!r} != {_FILE_ID!r}"
            raise AssertionError(msg)


if __name__ == "__main__":
    unittest.main()
