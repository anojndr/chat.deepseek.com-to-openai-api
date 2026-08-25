"""Async DeepSeek web client: sessions, PoW, SSE completion, file upload."""

from __future__ import annotations

import asyncio
import base64
import json
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

from .pow_solver import PowSolver

BASE_URL = "https://chat.deepseek.com"
TARGET_COMPLETION = "/api/v0/chat/completion"
TARGET_UPLOAD = "/api/v0/file/upload_file"

_CLIENT_HEADERS = {
    "x-client-platform": "web",
    "x-client-version": "2.4.0",
    "x-client-locale": "en_US",
    "x-client-bundle-id": "com.deepseek.chat",
    "referer": f"{BASE_URL}/a/chat/",
    "origin": BASE_URL,
}


class DeepSeekError(RuntimeError):
    def __init__(self, message: str, status: int | None = None, biz_code: int | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.biz_code = biz_code


class DeepSeekClient:
    """One client per account token; safe for concurrent use."""

    def __init__(self, token: str, pow_solver: PowSolver, timeout: float = 120.0) -> None:
        self.token = token
        self._pow = pow_solver
        self._http = httpx.AsyncClient(
            base_url=BASE_URL,
            timeout=httpx.Timeout(timeout, connect=15.0),
            headers={**_CLIENT_HEADERS, "authorization": f"Bearer {token}"},
            follow_redirects=True,
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    # -- low-level helpers -------------------------------------------------

    async def _request_json(
        self, method: str, path: str, *, json_body: Any = None
    ) -> dict[str, Any]:
        """POST/GET JSON with HTTP errors wrapped into DeepSeekError."""
        resp = await self._http.request(method, path, json=json_body)
        if resp.status_code >= 400:
            raise DeepSeekError(
                f"{path} returned HTTP {resp.status_code}: {resp.text[:200]}",
                status=resp.status_code,
            )
        try:
            return resp.json()
        except ValueError as exc:
            raise DeepSeekError(
                f"{path} returned non-JSON response", status=resp.status_code
            ) from exc

    @staticmethod
    def _check_biz(payload: dict[str, Any], context: str) -> dict[str, Any]:
        if payload.get("code") not in (0, None):
            raise DeepSeekError(
                f"{context}: {payload.get('msg') or payload}", biz_code=payload.get("code")
            )
        data = payload.get("data") or {}
        biz_code = data.get("biz_code")
        if biz_code not in (0, None):
            raise DeepSeekError(f"{context}: {data.get('biz_msg') or data}", biz_code=biz_code)
        biz_data = data.get("biz_data")
        if not isinstance(biz_data, dict):
            raise DeepSeekError(f"{context}: response missing biz_data object")
        return biz_data

    @staticmethod
    def _require(biz: dict[str, Any], key: str, context: str) -> Any:
        if key not in biz:
            raise DeepSeekError(f"{context}: missing '{key}' in response")
        return biz[key]

    async def _get_pow(self, target_path: str) -> tuple[dict, dict]:
        """Fetch + solve a PoW challenge; one fresh challenge per call."""
        payload = await self._request_json(
            "POST", "/api/v0/chat/create_pow_challenge",
            json_body={"target_path": target_path},
        )
        challenge = self._require(
            self._check_biz(payload, "create_pow_challenge"),
            "challenge", "create_pow_challenge",
        )
        answer = await asyncio.to_thread(
            self._pow.solve,
            challenge["challenge"],
            challenge["salt"],
            challenge["expire_at"],
            challenge.get("difficulty", 144000),
        )
        if answer is None:
            raise DeepSeekError("PoW solver found no solution")
        header_value = base64.b64encode(
            json.dumps(
                {
                    "algorithm": challenge.get("algorithm", "DeepSeekHashV1"),
                    "challenge": challenge["challenge"],
                    "salt": challenge["salt"],
                    "answer": answer,
                    "signature": challenge["signature"],
                    "target_path": target_path,
                }
            ).encode()
        ).decode()
        return {"x-ds-pow-response": header_value}, challenge

    # -- API surface -------------------------------------------------------

    async def create_session(self) -> str:
        payload = await self._request_json("POST", "/api/v0/chat_session/create", json_body={})
        biz = self._check_biz(payload, "create_chat_session")
        session = self._require(biz, "chat_session", "create_chat_session")
        return self._require(session, "id", "create_chat_session")

    async def delete_session(self, session_id: str) -> None:
        try:
            resp = await self._http.post(
                "/api/v0/chat_session/delete", json={"id": session_id}
            )
            resp.raise_for_status()
        except httpx.HTTPError:
            pass  # best-effort cleanup

    async def _fetch_file_status(self, fid: str) -> str | None:
        check = await self._http.get(f"/api/v0/file/fetch_files?file_ids={fid}")
        if check.status_code != 200:
            return None
        try:
            payload = check.json()
        except ValueError:
            return None
        data = (payload.get("data") or {}).get("biz_data") or {}
        items = data.get("files") or []
        info = next((f for f in items if f.get("id") == fid), None)
        return (info or {}) .get("status")

    async def upload_file(
        self, filename: str, content: bytes, mime: str | None = None, *, vision: bool = False
    ) -> str:
        """Upload a file; when `vision`, fork it so images get parsed for vision."""
        headers, _ = await self._get_pow(TARGET_UPLOAD)
        files = {"file": (filename, content, mime or "application/octet-stream")}
        resp = await self._http.post("/api/v0/file/upload_file", files=files, headers=headers)
        if resp.status_code >= 400:
            raise DeepSeekError(
                f"upload_file returned HTTP {resp.status_code}: {resp.text[:200]}",
                status=resp.status_code,
            )
        biz = self._check_biz(resp.json(), "upload_file")
        file_id = self._require(biz, "id", "upload_file")

        terminal = ("SUCCESS", "CONTENT_EMPTY", "ERROR", "REJECTED")

        async def wait_terminal(fid: str, seconds: float) -> str | None:
            """Wait until the file leaves PENDING/PARSING; return last status."""
            deadline = time.monotonic() + seconds
            status: str | None = None
            while time.monotonic() < deadline:
                status = await self._fetch_file_status(fid)
                if status in terminal:
                    return status
                await asyncio.sleep(0.4)
            return status  # None or still-parsing

        async def wait_success(fid: str, seconds: float) -> None:
            status = await wait_terminal(fid, seconds)
            if status != "SUCCESS":
                raise DeepSeekError(
                    f"upload processing failed for {fid} (status={status})"
                )

        if vision:
            # default-model parse of an image ends CONTENT_EMPTY (no text);
            # that is expected — the vision fork below re-parses it. Wait for
            # the parse to reach a TERMINAL state before forking.
            st = await wait_terminal(file_id, 30.0)
            if st not in ("SUCCESS", "CONTENT_EMPTY"):
                raise DeepSeekError(f"initial upload parse failed ({st or 'timed out'})")

        if vision:
            payload = await self._request_json(
                "POST", "/api/v0/file/fork_file_task",
                json_body={"file_id": file_id, "to_model_type": "vision"},
            )
            forked = self._check_biz(payload, "fork_file_task")
            new_file = forked.get("file")
            new_id = (
                forked.get("id")
                or (new_file.get("id") if isinstance(new_file, dict) else None)
            )
            if not new_id:
                raise DeepSeekError("fork_file_task response missing new file id")
            file_id = str(new_id)
            # non-vision images are useless without text extraction
        elif not vision and mime and mime.startswith("image/"):
            pass  # handled by generic wait below

        if vision:
            await wait_success(file_id, 40.0)
        else:
            await wait_success(file_id, 30.0)
        return file_id

    async def stream_completion(
        self,
        *,
        prompt: str,
        chat_session_id: str,
        parent_message_id: int | None = None,
        ref_file_ids: list[str] | None = None,
        thinking_enabled: bool = False,
        search_enabled: bool = True,
        model_type: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield parsed SSE events from /chat/completion."""
        headers, _ = await self._get_pow(TARGET_COMPLETION)
        body: dict[str, Any] = {
            "chat_session_id": chat_session_id,
            "parent_message_id": parent_message_id,
            "model_type": model_type,
            "prompt": prompt,
            "ref_file_ids": ref_file_ids or [],
            "thinking_enabled": thinking_enabled,
            "search_enabled": search_enabled,
            "action": None,
            "preempt": False,
        }
        req = self._http.build_request(
            "POST", TARGET_COMPLETION, json=body, headers=headers
        )
        response = await self._http.send(req, stream=True)
        try:
            if response.status_code != 200:
                text = (await response.aread()).decode(errors="replace")
                raise DeepSeekError(
                    f"completion failed: HTTP {response.status_code}: {text[:300]}",
                    status=response.status_code,
                )
            event_name: str | None = None
            data_lines: list[str] = []
            async for line in response.aiter_lines():
                if line == "":
                    if data_lines:
                        raw = "\n".join(data_lines)
                        try:
                            payload = json.loads(raw) if raw and raw != "[DONE]" else None
                        except json.JSONDecodeError:
                            payload = {"v": raw}
                        yield {"event": event_name, "data": payload}
                    event_name = None
                    data_lines = []
                    continue
                if line.startswith("event:"):
                    event_name = line[6:].strip()
                elif line.startswith("data:"):
                    data_lines.append(line[5:].strip())
            if data_lines:
                # stream closed without a trailing blank line — don't lose it
                raw_last = "\n".join(data_lines)
                try:
                    payload = json.loads(raw_last) if raw_last != "[DONE]" else None
                except json.JSONDecodeError:
                    payload = {"v": raw_last}
                yield {"event": event_name, "data": payload}
        finally:
            await response.aclose()
