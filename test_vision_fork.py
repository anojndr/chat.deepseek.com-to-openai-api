"""Regression: vision uploads must tolerate DeepSeek returning VISION directly.

Upstream now creates image files as model_kind VISION, so fork_file_task
fails with biz_code 2 ("model kind satisfied"). All image turns failed
across all accounts until this was tolerated.
"""

import unittest
from typing import override
from unittest.mock import AsyncMock, MagicMock

from app.deepseek import DeepSeekClient, DeepSeekError
from app.pow_solver import PowSolver


class DummySolver(PowSolver):
    def __init__(self) -> None:
        pass

    @override
    def solve(self, *args, **kwargs):
        return "0"


def _upload_ok(file_id="file-abc", model_kind="VISION"):
    return {
        "code": 0,
        "msg": "",
        "data": {
            "biz_code": 0,
            "biz_msg": "",
            "biz_data": {"id": file_id, "model_kind": model_kind},
        },
    }


def _fork_satisfied():
    return {
        "code": 0,
        "msg": "",
        "data": {"biz_code": 2, "biz_msg": "model kind satisfied", "biz_data": None},
    }


class VisionForkToleranceTest(unittest.IsolatedAsyncioTestCase):
    async def _client_with(self, upload_payload, fork_payload=None, fork_raises=None):
        client = DeepSeekClient("tok", DummySolver())
        client._get_pow = AsyncMock(return_value=({"x-ds-pow-response": "x"}, {}))  # type: ignore[method-assign]
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = upload_payload
        client._http.post = AsyncMock(return_value=resp)  # type: ignore[method-assign]
        client._fetch_file_status = AsyncMock(return_value="SUCCESS")  # type: ignore[method-assign]
        if fork_raises is not None:
            client._request_json = AsyncMock(side_effect=fork_raises)  # type: ignore[method-assign]
        else:
            client._request_json = AsyncMock(return_value=fork_payload)  # type: ignore[method-assign]
        return client

    async def test_already_vision_skips_fork(self):
        client = await self._client_with(_upload_ok("file-abc", "VISION"))
        try:
            fid = await client.upload_file("a.png", b"raw", "image/png", vision=True)
        finally:
            await client.aclose()
        self.assertEqual(fid, "file-abc")
        client._request_json.assert_not_awaited()

    async def test_fork_satisfied_tolerated(self):
        # Older upload shape without model_kind, fork says already vision.
        client = await self._client_with(
            _upload_ok("file-abc", "DEFAULT"), fork_payload=_fork_satisfied()
        )
        try:
            fid = await client.upload_file("a.png", b"raw", "image/png", vision=True)
        finally:
            await client.aclose()
        self.assertEqual(fid, "file-abc")


if __name__ == "__main__":
    unittest.main()
