# Copyright (c) 2026 chat.deepseek.com-to-openai-api contributors.
"""Async DeepSeek web client: sessions, PoW, SSE completion, file upload."""

from __future__ import annotations

import asyncio
import base64
import json
import time
from typing import TYPE_CHECKING, Any, NotRequired, TypedDict, Unpack

import httpx

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from .pow_solver import PowSolver

BASE_URL = "https://chat.deepseek.com"
TARGET_COMPLETION = "/api/v0/chat/completion"
TARGET_UPLOAD = "/api/v0/file/upload_file"

# Stall watchdog: longest we'll go without usable upstream SSE output, from
# stream start to first event or between yielded events. httpx's read timeout
# only trips on total silence and SSE keepalives reset it, so without this a
# wedged upstream worker holds the turn open forever with zero deltas and the
# client never gets a terminal event (seen live: a turn parked 13+ min while
# the UI showed "streaming"). Exceeded -> DeepSeekError, which the caller
# turns into a terminal failed event / account rotation.
STALL_TIMEOUT = 90.0

# HTTP status boundary for request failures.
_HTTP_BAD_REQUEST = 400
# Expected success status for polling and streaming endpoints.
_HTTP_OK = 200
# Backend biz_code meaning the file already satisfies the vision model kind.
_BIZ_CODE_MODEL_KIND_SATISFIED = 2
# Poll interval while waiting for file parse to leave PENDING/PARSING.
_FILE_POLL_INTERVAL_S = 0.4
# Timeouts for file-parse waits.
_FILE_WAIT_INITIAL_S = 30.0
_FILE_WAIT_VISION_S = 40.0
# Default PoW difficulty when the challenge omits it.
_POW_DIFFICULTY_DEFAULT = 144000
# Default PoW algorithm identifier.
_POW_ALGORITHM_DEFAULT = "DeepSeekHashV1"
# Preview length for HTTP error snippets.
_ERROR_SNIPPET_LEN = 200
# Preview length for completion HTTP error snippets.
_COMPLETION_SNIPPET_LEN = 300

_CLIENT_HEADERS = {
    "x-client-platform": "web",
    "x-client-version": "2.4.0",
    "x-client-locale": "en_US",
    "x-client-bundle-id": "com.deepseek.chat",
    "referer": f"{BASE_URL}/a/chat/",
    "origin": BASE_URL,
}

# File statuses that end the parse wait.
_TERMINAL_FILE_STATUSES = frozenset({"SUCCESS", "CONTENT_EMPTY", "ERROR", "REJECTED"})
# File statuses that allow the vision fast path to continue.
_VISION_OK_STATUSES = frozenset({"SUCCESS", "CONTENT_EMPTY"})
# Success codes for the outer and inner biz envelopes.
_OK_BIZ_CODES = frozenset({0, None})


class CompletionOptions(TypedDict):
    """Optional arguments for a completion stream."""

    parent_message_id: NotRequired[int | None]
    ref_file_ids: NotRequired[list[str] | None]
    thinking_enabled: NotRequired[bool]
    search_enabled: NotRequired[bool]
    model_type: NotRequired[str | None]


class PowHeaderOptions(TypedDict):
    """Fields for encoding a PoW response header."""

    algorithm: str
    challenge_hex: str
    salt: str
    answer: float
    signature: str
    target_path: str


class CompletionBodyOptions(TypedDict):
    """Fields for building a completion request body."""

    prompt: str
    chat_session_id: str
    parent_message_id: int | None
    ref_file_ids: list[str]
    thinking_enabled: bool
    search_enabled: bool
    model_type: str | None


class DeepSeekError(RuntimeError):
    """Signal an upstream DeepSeek failure."""

    def __init__(
        self,
        message: str,
        status: int | None = None,
        biz_code: int | None = None,
    ) -> None:
        """Store message with HTTP and backend codes."""
        super().__init__(message)
        self.status = status
        self.biz_code = biz_code


class DeepSeekHttpError(DeepSeekError):
    """Raised when a JSON request returns an HTTP error status."""

    def __init__(self, path: str, status: int, snippet: str) -> None:
        """Build message from path, status, and body preview."""
        super().__init__(
            f"{path} returned HTTP {status}: {snippet[:_ERROR_SNIPPET_LEN]}",
            status=status,
        )
        self.path = path
        self.snippet = snippet


class DeepSeekNonJsonError(DeepSeekError):
    """Raised when a JSON request returns a non-JSON body."""

    def __init__(self, path: str, status: int) -> None:
        """Build message from path and status."""
        super().__init__(f"{path} returned non-JSON response", status=status)
        self.path = path


class DeepSeekBizCodeError(DeepSeekError):
    """Raised when the backend biz envelope carries an error code."""

    def __init__(self, context: str, code: object, detail: object) -> None:
        """Build message from context, code, and detail."""
        code_int = code if isinstance(code, int) else None
        super().__init__(f"{context}: {detail}", biz_code=code_int)
        self.context = context
        self.detail = detail


class DeepSeekBizDataMissingError(DeepSeekError):
    """Raised when the biz envelope lacks a biz_data object."""

    def __init__(self, context: str) -> None:
        """Build message from context."""
        super().__init__(f"{context}: response missing biz_data object")
        self.context = context


class DeepSeekFieldMissingError(DeepSeekError):
    """Raised when a required response field is absent."""

    def __init__(self, context: str, key: str) -> None:
        """Build message from context and missing key."""
        super().__init__(f"{context}: missing '{key}' in response")
        self.context = context
        self.key = key


class PowChallengeNotObjectError(DeepSeekError):
    """Raised when the PoW challenge is not an object."""

    def __init__(self) -> None:
        """Build fixed message."""
        super().__init__("create_pow_challenge: 'challenge' is not an object")


class PowChallengeStringError(DeepSeekError):
    """Raised when the PoW challenge hex is not a string."""

    def __init__(self) -> None:
        """Build fixed message."""
        super().__init__("create_pow_challenge: 'challenge' is not a string")


class PowSaltStringError(DeepSeekError):
    """Raised when the PoW salt is not a string."""

    def __init__(self) -> None:
        """Build fixed message."""
        super().__init__("create_pow_challenge: 'salt' is not a string")


class PowExpireMissingError(DeepSeekError):
    """Raised when the PoW expire_at value is missing."""

    def __init__(self) -> None:
        """Build fixed message."""
        super().__init__("create_pow_challenge: 'expire_at' is missing")


class PowDifficultyNumberError(DeepSeekError):
    """Raised when the PoW difficulty is not a number."""

    def __init__(self) -> None:
        """Build fixed message."""
        super().__init__("create_pow_challenge: 'difficulty' is not a number")


class PowNoSolutionError(DeepSeekError):
    """Raised when the PoW solver finds no solution."""

    def __init__(self) -> None:
        """Build fixed message."""
        super().__init__("PoW solver found no solution")


class PowSignatureStringError(DeepSeekError):
    """Raised when the PoW signature is not a string."""

    def __init__(self) -> None:
        """Build fixed message."""
        super().__init__("create_pow_challenge: 'signature' is not a string")


class ChatSessionNotObjectError(DeepSeekError):
    """Raised when chat_session is not an object."""

    def __init__(self) -> None:
        """Build fixed message."""
        super().__init__("create_chat_session: 'chat_session' is not an object")


class ChatSessionIdError(DeepSeekError):
    """Raised when the chat session id is not a string."""

    def __init__(self) -> None:
        """Build fixed message."""
        super().__init__("create_chat_session: 'id' is not a string")


class UploadHttpError(DeepSeekError):
    """Raised when file upload returns an HTTP error status."""

    def __init__(self, status: int, snippet: str) -> None:
        """Build message from status and body preview."""
        super().__init__(
            f"upload_file returned HTTP {status}: {snippet[:_ERROR_SNIPPET_LEN]}",
            status=status,
        )
        self.snippet = snippet


class UploadIdTypeError(DeepSeekError):
    """Raised when the upload file id is not a string."""

    def __init__(self) -> None:
        """Build fixed message."""
        super().__init__("upload_file: 'id' is not a string")


class UploadProcessingError(DeepSeekError):
    """Raised when uploaded file processing does not succeed."""

    def __init__(self, fid: str, status: str | None) -> None:
        """Build message from file id and terminal status."""
        super().__init__(f"upload processing failed for {fid} (status={status})")
        self.fid = fid
        self.file_status = status


class UploadInitialParseError(DeepSeekError):
    """Raised when the initial upload parse fails."""

    def __init__(self, status: str | None) -> None:
        """Build message from parse status."""
        super().__init__(f"initial upload parse failed ({status or 'timed out'})")
        self.file_status = status


class ForkMissingIdError(DeepSeekError):
    """Raised when fork_file_task omits the new file id."""

    def __init__(self) -> None:
        """Build fixed message."""
        super().__init__("fork_file_task response missing new file id")


class CompletionHttpError(DeepSeekError):
    """Raised when completion returns an HTTP error status."""

    def __init__(self, status: int, snippet: str) -> None:
        """Build message from status and body preview."""
        super().__init__(
            f"completion failed: HTTP {status}: {snippet[:_COMPLETION_SNIPPET_LEN]}",
            status=status,
        )
        self.snippet = snippet


class CompletionStalledDataError(DeepSeekError):
    """Raised when completion yields no data within the stall bound."""

    def __init__(self, timeout: float, status: int | None) -> None:
        """Build message from timeout and status."""
        super().__init__(
            f"completion stalled: no data from upstream for {timeout:g}s",
            status=status,
        )
        self.timeout = timeout


class CompletionStalledEventsError(DeepSeekError):
    """Raised when completion yields no usable events within the bound."""

    def __init__(self, timeout: float, status: int | None) -> None:
        """Build message from timeout and status."""
        super().__init__(
            f"completion stalled: no usable events from upstream for {timeout:g}s",
            status=status,
        )
        self.timeout = timeout


class DeepSeekClient:
    """Store one authenticated HTTP client per account token."""

    def __init__(
        self,
        token: str,
        pow_solver: PowSolver,
        timeout: float = 120.0,
    ) -> None:
        """Store token, solver, and HTTP client."""
        self.token = token
        self._pow = pow_solver
        self._http = httpx.AsyncClient(
            base_url=BASE_URL,
            timeout=httpx.Timeout(timeout, connect=15.0),
            headers={**_CLIENT_HEADERS, "authorization": f"Bearer {token}"},
            follow_redirects=True,
        )

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        await self._http.aclose()

    async def _request_json(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, object] | None = None,
    ) -> dict[str, Any]:
        """Send a JSON request and return the decoded payload.

        Returns:
            The decoded JSON object.

        Raises:
            DeepSeekHttpError: If the status indicates failure.
            DeepSeekNonJsonError: If the body is not JSON.

        """
        resp = await self._http.request(method, path, json=json_body)
        if resp.status_code >= _HTTP_BAD_REQUEST:
            raise DeepSeekHttpError(path, resp.status_code, resp.text)
        try:
            decoded: dict[str, Any] = resp.json()
        except ValueError as exc:
            raise DeepSeekNonJsonError(path, resp.status_code) from exc
        else:
            return decoded

    @staticmethod
    def _check_biz(payload: dict[str, Any], context: str) -> dict[str, Any]:
        """Validate biz envelopes and return biz_data.

        Returns:
            The biz_data object.

        Raises:
            DeepSeekBizCodeError: If outer or inner codes signal failure.
            DeepSeekBizDataMissingError: If biz_data is absent.

        """
        if payload.get("code") not in _OK_BIZ_CODES:
            raise DeepSeekBizCodeError(
                context,
                payload.get("code"),
                payload.get("msg") or payload,
            )
        data = payload.get("data") or {}
        biz_code = data.get("biz_code") if isinstance(data, dict) else None
        if biz_code not in _OK_BIZ_CODES:
            detail = data.get("biz_msg") or data if isinstance(data, dict) else data
            raise DeepSeekBizCodeError(context, biz_code, detail)
        biz_data = data.get("biz_data") if isinstance(data, dict) else None
        if not isinstance(biz_data, dict):
            raise DeepSeekBizDataMissingError(context)
        return biz_data

    @staticmethod
    def _require(biz: dict[str, Any], key: str, context: str) -> object:
        """Fetch a required key from a biz object.

        Returns:
            The stored value.

        Raises:
            DeepSeekFieldMissingError: If the key is absent.

        """
        if key not in biz:
            raise DeepSeekFieldMissingError(context, key)
        return biz[key]

    def _extract_challenge(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Extract and validate the PoW challenge object.

        Returns:
            The challenge mapping.

        Raises:
            PowChallengeNotObjectError: If challenge is not a dict.

        """
        raw_challenge = self._require(
            self._check_biz(payload, "create_pow_challenge"),
            "challenge",
            "create_pow_challenge",
        )
        if not isinstance(raw_challenge, dict):
            raise PowChallengeNotObjectError
        return {str(key): val for key, val in raw_challenge.items()}

    @staticmethod
    def _challenge_parts(
        challenge: dict[str, Any],
    ) -> tuple[str, str, str | int | float]:
        """Split challenge hex, salt, and expiry from the mapping.

        Returns:
            Tuple of challenge hex, salt, and expiry value.

        Raises:
            PowChallengeStringError: If challenge hex is invalid.
            PowSaltStringError: If salt is invalid.
            PowExpireMissingError: If expiry is missing.

        """
        challenge_hex = challenge.get("challenge")
        salt = challenge.get("salt")
        expire_at = challenge.get("expire_at")
        if not isinstance(challenge_hex, str) or not challenge_hex:
            raise PowChallengeStringError
        if not isinstance(salt, str) or not salt:
            raise PowSaltStringError
        if isinstance(expire_at, bool) or not isinstance(expire_at, str | int | float):
            raise PowExpireMissingError
        return challenge_hex, salt, expire_at

    @staticmethod
    def _challenge_difficulty(challenge: dict[str, Any]) -> float:
        """Read difficulty with a default fallback.

        Returns:
            The numeric difficulty.

        Raises:
            PowDifficultyNumberError: If difficulty is not numeric.

        """
        difficulty = challenge.get("difficulty", _POW_DIFFICULTY_DEFAULT)
        if isinstance(difficulty, bool) or not isinstance(difficulty, (int, float)):
            raise PowDifficultyNumberError
        return difficulty

    @staticmethod
    def _challenge_meta(challenge: dict[str, Any]) -> tuple[str, str]:
        """Read algorithm and signature with validation.

        Returns:
            Tuple of algorithm and signature.

        Raises:
            PowSignatureStringError: If signature is invalid.

        """
        algorithm = challenge.get("algorithm", _POW_ALGORITHM_DEFAULT)
        if not isinstance(algorithm, str) or not algorithm:
            algorithm = _POW_ALGORITHM_DEFAULT
        signature = challenge.get("signature")
        if not isinstance(signature, str) or not signature:
            raise PowSignatureStringError
        return algorithm, signature

    @staticmethod
    def _pow_header(**options: Unpack[PowHeaderOptions]) -> dict[str, str]:
        """Encode the solved PoW response header.

        Returns:
            Header mapping with the encoded response.

        """
        header_value = base64.b64encode(
            json.dumps(
                {
                    "algorithm": options["algorithm"],
                    "challenge": options["challenge_hex"],
                    "salt": options["salt"],
                    "answer": options["answer"],
                    "signature": options["signature"],
                    "target_path": options["target_path"],
                },
            ).encode(),
        ).decode()
        return {"x-ds-pow-response": header_value}

    async def _get_pow(
        self,
        target_path: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Fetch and solve a PoW challenge for one call.

        Returns:
            Tuple of headers and the raw challenge.

        Raises:
            PowNoSolutionError: If the solver finds nothing.

        """
        payload = await self._request_json(
            "POST",
            "/api/v0/chat/create_pow_challenge",
            json_body={"target_path": target_path},
        )
        challenge = self._extract_challenge(payload)
        challenge_hex, salt, expire_at = self._challenge_parts(challenge)
        difficulty = self._challenge_difficulty(challenge)
        answer = await asyncio.to_thread(
            self._pow.solve,
            challenge_hex,
            salt,
            expire_at,
            difficulty,
        )
        if answer is None:
            raise PowNoSolutionError
        algorithm, signature = self._challenge_meta(challenge)
        return (
            self._pow_header(
                algorithm=algorithm,
                challenge_hex=challenge_hex,
                salt=salt,
                answer=answer,
                signature=signature,
                target_path=target_path,
            ),
            challenge,
        )

    async def create_session(self) -> str:
        """Create a chat session and return its id.

        Returns:
            The new session identifier.

        Raises:
            ChatSessionNotObjectError: If chat_session is not a dict.
            ChatSessionIdError: If the id is not a string.

        """
        payload = await self._request_json(
            "POST",
            "/api/v0/chat_session/create",
            json_body={},
        )
        biz = self._check_biz(payload, "create_chat_session")
        session = self._require(biz, "chat_session", "create_chat_session")
        if not isinstance(session, dict):
            raise ChatSessionNotObjectError
        session_dict: dict[str, Any] = {str(key): val for key, val in session.items()}
        session_id = self._require(session_dict, "id", "create_chat_session")
        if not isinstance(session_id, str) or not session_id:
            raise ChatSessionIdError
        return session_id

    async def delete_session(self, session_id: str) -> None:
        """Delete a chat session with best-effort cleanup."""
        try:
            resp = await self._http.post(
                "/api/v0/chat_session/delete",
                json={"id": session_id},
            )
            resp.raise_for_status()
        except httpx.HTTPError:
            pass  # best-effort cleanup

    @staticmethod
    def _safe_json(check: httpx.Response) -> dict[str, Any] | None:
        """Decode a file-status body or return None.

        Returns:
            The decoded mapping or None when unusable.

        """
        try:
            payload = check.json()
        except ValueError:
            return None
        return payload if isinstance(payload, dict) else None

    @staticmethod
    def _extract_file_status(payload: dict[str, Any], fid: str) -> str | None:
        """Find the status for one file id in a fetch payload.

        Returns:
            The status string or None when absent.

        """
        data_obj = payload.get("data") or {}
        if not isinstance(data_obj, dict):
            return None
        biz_data = data_obj.get("biz_data") or {}
        if not isinstance(biz_data, dict):
            return None
        files_obj = biz_data.get("files") or []
        if not isinstance(files_obj, list):
            return None
        for entry in files_obj:
            if isinstance(entry, dict) and entry.get("id") == fid:
                status = entry.get("status")
                return status if isinstance(status, str) else None
        return None

    async def _fetch_file_status(self, fid: str) -> str | None:
        """Fetch the parse status for one uploaded file.

        Returns:
            The status string or None when unavailable.

        """
        check = await self._http.get(f"/api/v0/file/fetch_files?file_ids={fid}")
        if check.status_code != _HTTP_OK:
            return None
        payload = self._safe_json(check)
        if payload is None:
            return None
        return self._extract_file_status(payload, fid)

    async def _wait_file_terminal(self, fid: str, seconds: float) -> str | None:
        """Wait until a file leaves PENDING/PARSING.

        Returns:
            The last observed status.

        """
        deadline = time.monotonic() + seconds
        status: str | None = None
        while time.monotonic() < deadline:
            status = await self._fetch_file_status(fid)
            if status in _TERMINAL_FILE_STATUSES:
                return status
            await asyncio.sleep(_FILE_POLL_INTERVAL_S)
        return status

    async def _wait_file_success(self, fid: str, seconds: float) -> None:
        """Wait until a file reports SUCCESS.

        Raises:
            UploadProcessingError: If the terminal status is not SUCCESS.

        """
        status = await self._wait_file_terminal(fid, seconds)
        if status != "SUCCESS":
            raise UploadProcessingError(fid, status)

    async def _post_upload_file(
        self,
        filename: str,
        content: bytes,
        mime: str | None,
        headers: dict[str, Any],
    ) -> dict[str, Any]:
        """Post file bytes and return the biz object.

        Returns:
            The upload biz_data mapping.

        Raises:
            UploadHttpError: If the status indicates failure.

        """
        files = {"file": (filename, content, mime or "application/octet-stream")}
        resp = await self._http.post(
            "/api/v0/file/upload_file",
            files=files,
            headers=headers,
        )
        if resp.status_code >= _HTTP_BAD_REQUEST:
            raise UploadHttpError(resp.status_code, resp.text)
        return self._check_biz(resp.json(), "upload_file")

    @staticmethod
    def _upload_file_id(biz: dict[str, Any]) -> str:
        """Extract the file id from an upload biz object.

        Returns:
            The file identifier.

        Raises:
            UploadIdTypeError: If the id is not a string.

        """
        raw_file_id = DeepSeekClient._require(biz, "id", "upload_file")
        if not isinstance(raw_file_id, str) or not raw_file_id:
            raise UploadIdTypeError
        return raw_file_id

    @staticmethod
    def _is_vision_kind(biz: dict[str, Any]) -> bool:
        """Check whether an upload already has vision model kind.

        Returns:
            True when model_kind is vision.

        """
        model_kind = biz.get("model_kind")
        if not isinstance(model_kind, str):
            return False
        return bool(model_kind.lower() == "vision")

    @staticmethod
    def _fork_file_id(forked: dict[str, Any]) -> str:
        """Extract the forked file id from a fork response.

        Returns:
            The new file identifier.

        Raises:
            ForkMissingIdError: If no new id is present.

        """
        new_file = forked.get("file")
        new_id = forked.get("id") or (
            new_file.get("id") if isinstance(new_file, dict) else None
        )
        if not new_id:
            raise ForkMissingIdError
        return str(new_id)

    @staticmethod
    def _is_model_kind_satisfied(exc: DeepSeekError) -> bool:
        """Check whether an error signals an already-vision file.

        Returns:
            True when the file already satisfies vision kind.

        """
        return (
            exc.biz_code == _BIZ_CODE_MODEL_KIND_SATISFIED
            or "model kind satisfied" in str(exc)
        )

    async def _fork_for_vision(self, file_id: str) -> str:
        """Fork a file to vision kind, tolerating already-vision errors.

        Returns:
            The vision file identifier.

        Raises:
            DeepSeekError: If forking fails without tolerance.

        """
        try:
            payload = await self._request_json(
                "POST",
                "/api/v0/file/fork_file_task",
                json_body={"file_id": file_id, "to_model_type": "vision"},
            )
        except DeepSeekError as exc:
            if self._is_model_kind_satisfied(exc):
                await self._wait_file_success(file_id, _FILE_WAIT_VISION_S)
                return file_id
            raise
        try:
            forked = self._check_biz(payload, "fork_file_task")
        except DeepSeekError as exc:
            if self._is_model_kind_satisfied(exc):
                await self._wait_file_success(file_id, _FILE_WAIT_VISION_S)
                return file_id
            raise
        return self._fork_file_id(forked)

    async def _prepare_vision_upload(
        self,
        file_id: str,
        biz: dict[str, Any],
    ) -> str | None:
        """Run the vision fast path before forking.

        Returns:
            The file id when already vision, else None to continue.

        Raises:
            UploadInitialParseError: If the initial parse fails.

        """
        st = await self._wait_file_terminal(file_id, _FILE_WAIT_INITIAL_S)
        if st not in _VISION_OK_STATUSES:
            raise UploadInitialParseError(st)
        if self._is_vision_kind(biz):
            await self._wait_file_success(file_id, _FILE_WAIT_VISION_S)
            return file_id
        return None

    async def upload_file(
        self,
        filename: str,
        content: bytes,
        mime: str | None = None,
        *,
        vision: bool = False,
    ) -> str:
        """Upload a file and wait until parsing succeeds.

        Since the Instant/Expert/Vision unification the backend parses image
        uploads directly as model_kind VISION, so the fork below is normally
        skipped (or fails soft with biz_code 2 "model kind satisfied", which
        is tolerated). It stays as a fallback for backends that still return
        a non-vision kind for images.

        Returns:
            The terminal file identifier.

        """
        headers, _ = await self._get_pow(TARGET_UPLOAD)
        biz = await self._post_upload_file(filename, content, mime, headers)
        file_id = self._upload_file_id(biz)
        if vision:
            fast_path = await self._prepare_vision_upload(file_id, biz)
            if fast_path is not None:
                return fast_path
            file_id = await self._fork_for_vision(file_id)
        if vision:
            await self._wait_file_success(file_id, _FILE_WAIT_VISION_S)
        else:
            await self._wait_file_success(file_id, _FILE_WAIT_INITIAL_S)
        return file_id

    @staticmethod
    def _completion_body(**options: Unpack[CompletionBodyOptions]) -> dict[str, Any]:
        """Build the JSON body for a completion request.

        Returns:
            The request body mapping.

        """
        return {
            "chat_session_id": options["chat_session_id"],
            "parent_message_id": options["parent_message_id"],
            "model_type": options["model_type"],
            "prompt": options["prompt"],
            "ref_file_ids": options["ref_file_ids"],
            "thinking_enabled": options["thinking_enabled"],
            "search_enabled": options["search_enabled"],
            "action": None,
            "preempt": False,
        }

    @staticmethod
    def _parse_sse_data(raw: str) -> object:
        """Parse one SSE data block with a raw fallback.

        Returns:
            The decoded payload or raw wrapper.

        """
        if not raw or raw == "[DONE]":
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"v": raw}

    @staticmethod
    async def _read_sse_line(
        lines: AsyncIterator[str],
        status: int | None,
    ) -> str | None:
        """Read one SSE line with stall protection.

        Returns:
            The next line or None at stream end.

        Raises:
            CompletionStalledDataError: If no data arrives in time.

        """
        try:
            async with asyncio.timeout(STALL_TIMEOUT):
                return await anext(lines)
        except StopAsyncIteration:
            return None
        except TimeoutError as exc:
            raise CompletionStalledDataError(STALL_TIMEOUT, status) from exc

    @staticmethod
    def _check_usable_stall(last_progress: float, status: int | None) -> None:
        """Enforce progress between yielded SSE events.

        Raises:
            CompletionStalledEventsError: If no usable events arrive in time.

        """
        if time.monotonic() - last_progress > STALL_TIMEOUT:
            raise CompletionStalledEventsError(STALL_TIMEOUT, status)

    def _flush_sse_buffer(
        self,
        *,
        event_name: str | None,
        data_lines: list[str],
    ) -> dict[str, object] | None:
        """Flush buffered SSE lines into one event object.

        Returns:
            The flushed event or None when empty.

        """
        if not data_lines:
            return None
        raw = "\n".join(data_lines)
        return {"event": event_name, "data": self._parse_sse_data(raw)}

    async def _iter_sse_events(
        self,
        response: httpx.Response,
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield parsed SSE events with stall protection.

        Yields:
            Parsed event mappings.

        """
        event_name: str | None = None
        data_lines: list[str] = []
        lines = response.aiter_lines()
        last_progress = time.monotonic()
        while True:
            line = await self._read_sse_line(lines, response.status_code)
            if line is None:
                break
            self._check_usable_stall(last_progress, response.status_code)
            if not line:
                flushed = self._flush_sse_buffer(
                    event_name=event_name,
                    data_lines=data_lines,
                )
                if flushed is not None:
                    last_progress = time.monotonic()
                    yield flushed
                event_name = None
                data_lines = []
                continue
            if line.startswith("event:"):
                event_name = line[6:].strip()
            elif line.startswith("data:"):
                data_lines.append(line[5:].strip())
        flushed = self._flush_sse_buffer(event_name=event_name, data_lines=data_lines)
        if flushed is not None:
            yield flushed

    async def stream_completion(
        self,
        *,
        prompt: str,
        chat_session_id: str,
        **options: Unpack[CompletionOptions],
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield parsed SSE events from chat completion.

        Yields:
            Parsed event mappings.

        Raises:
            CompletionHttpError: If the status indicates failure.

        """
        parent_message_id = options.get("parent_message_id")
        ref_file_ids = options.get("ref_file_ids") or []
        thinking_enabled = options.get("thinking_enabled", False)
        search_enabled = options.get("search_enabled", True)
        model_type = options.get("model_type")
        headers, _ = await self._get_pow(TARGET_COMPLETION)
        body = self._completion_body(
            prompt=prompt,
            chat_session_id=chat_session_id,
            parent_message_id=parent_message_id,
            ref_file_ids=list(ref_file_ids),
            thinking_enabled=bool(thinking_enabled),
            search_enabled=bool(search_enabled),
            model_type=model_type,
        )
        req = self._http.build_request(
            "POST",
            TARGET_COMPLETION,
            json=body,
            headers=headers,
        )
        response = await self._http.send(req, stream=True)
        try:
            if response.status_code != _HTTP_OK:
                text = (await response.aread()).decode(errors="replace")
                raise CompletionHttpError(response.status_code, text)
            async for event in self._iter_sse_events(response):
                yield event
        finally:
            await response.aclose()
