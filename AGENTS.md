# Repository Guidelines

## Project Overview

Browserless FastAPI proxy exposing `chat.deepseek.com` as OpenAI-compatible API (`POST /v1/chat/completions`, `POST /v1/responses`, streaming + non-streaming). Replicates DeepSeek web PoW (`DeepSeekHashV1` via wasmtime) + SSE patch protocol using user tokens from `accounts.txt`. Unofficial, ToS-risky — use expendable accounts.

## Architecture & Data Flow

- Entry: `server.py:main()` → `uvicorn.run("app.main:app", HOST, PORT)`. ASGI app is `app/main.py:app` (`lifespan` builds `PowSolver` + `ConversationManager`; shutdown calls `manager().aclose()`).
- Orchestration: `app/conversations.py:ConversationManager` pins one DeepSeek `chat_session_id` per proxy conversation key, multiplexes `AccountPool`. `run_turn()` / `stream_turn()` → `_collect()` / `_stream_events()`.
- Upstream: `app/deepseek.py:DeepSeekClient(token, pow_solver)` — `create_session` / `upload_file` / `stream_completion` with `x-ds-pow-response` header, 90s stall watchdog (`STALL_TIMEOUT`).
- Translation: `app/turn.py:prepare_turn()` + `app/models.py:parse_model()` (OpenAI messages/files → DeepSeek prompt); `app/aggregator.py:FragmentAggregator` (THINK/RESPONSE/search fragments → deltas) + `app/citations.py:CitationRewriter` (`[citation:N]` → markdown links + Sources appendix).
- Infra: `app/accounts.py:AccountPool` (round-robin + exp backoff, cap 900s), `app/storage.py:Storage` (WAL SQLite), `app/pow_solver.py:PowSolver` (`threading.Lock`-serialized wasmtime over `app/vendor/sha3_wasm_bg.wasm`).
- Request lifecycle:
  1. `app/main.py:chat_completions()/responses_api()` validates Pydantic body → `parse_model()` → resolve key (`X-Session-Id`/`X-Conversation-Id` else `auto:{ip}:{user}`; Responses uses `previous_response_id` link) → `compute_history_hashes()` → `prepare_turn()` (first turn = full labeled history; follow-up = latest user + `[system reminder]`; only `data:` URLs decoded).
  2. `ConversationManager` takes per-key `asyncio.Lock` → `_ensure_session()` (reuse or `pool.acquire()` + `create_session`) → `upload_file()` (images = vision fast-path/fork) → `stream_completion()` → aggregate or yield `StreamEvent(reasoning|content|search|sources|references|meta)`.
  3. `app/main.py` formats OpenAI SSE/JSON (usage estimated `len//4`), persists response links (in-memory LRU 10k + SQLite). Multi-turn via DeepSeek `parent_message_id`; failover replays `conv.history` into fresh session. Trim `MAX_HISTORY_CHARS=400k`; sweep sessions `SESSION_TTL=6h` every `900s`.

## Key Directories

- `app/` — all source (no `src/`). `main.py`, `conversations.py`, `deepseek.py`, `turn.py`, `models.py`, `aggregator.py`, `citations.py`, `accounts.py`, `storage.py`, `pow_solver.py`.
- `app/vendor/sha3_wasm_bg.wasm` — PoW WASM bytes (do not edit).
- `vendor/wasmtime-48.0.0-py3-none-manylinux1_x86_64.whl` — vendored x86_64 install source.
- Root `test_*.py` (9 files) + `test_suite.py` — entire suite; no `tests/`, `scripts/`, `docs/`, `.github/`, `Dockerfile`.

## Development Commands

```bash
python3 server.py                                   # bootstrap + serve 127.0.0.1:34868
HOST=127.0.0.1 PORT=34868 API_KEY=secret python3 server.py
./restart.sh                                        # kill old, detached start, /health gate → server.log
./restart.sh -f                                     # same + tail -f (Ctrl-C leaves daemon)
uvicorn app.main:app --host 127.0.0.1 --port 34868 # manual (skips setup.py bootstrap)

uv sync                                             # reproduce env from uv.lock
pip install vendor/wasmtime-48.0.0-py3-none-manylinux1_x86_64.whl  # manual x86_64 PoW dep
python3 -m unittest test_suite -v                   # canonical test run
python3 test_suite.py                               # same via __main__
python3 -m unittest discover -p 'test_*.py' -v     # discovery (no wrapper)
python3 -m unittest test_storage test_turn_recovery -v  # subset
ruff check . && ruff format --check .               # lint (config only, no script)
ty check                                            # typecheck (all=error)
curl -s localhost:34868/health
curl -s localhost:34868/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-chat","messages":[{"role":"user","content":"hi"}],"stream":false}'
```

No `package.json`/`Makefile`/`justfile` scripts; no `uv run` console entry — bless `server.py`.

## Code Conventions & Common Patterns

- Style: `snake_case` funcs/vars, `CapWords` classes, `UPPER_SNAKE` consts, `_private` helpers; `from __future__ import annotations` + exhaustive hints (`str|None`, `TypedDict(NotRequired)`, `Unpack[...]`); Google-style docstrings (`Args/Returns/Raises/Yields`) on every def; `ruff select=ALL, preview=true, py312`.
- Value objects as `@dataclass` (e.g. `PreparedTurn`, `TurnResult`, `_ChatStreamContext`); option bags as `TypedDict + Unpack` (`CompletionOptions`, `RunTurnOptions`).
- Model routing: `app/models.py:MODEL_ALIASES` + `parse_model("deepseek-reasoner-think")` → `ModelSpec(deepthink=True, model_type=None)`; `-think|-thinking|-deepthink` suffix (case-insensitive) toggles thinking.
- Async: `async def` handlers + `AsyncIterator` SSE via `StreamingResponse`; `httpx.AsyncClient` per token cached in manager; per-key `asyncio.Lock`; `asyncio.timeout(STALL_TIMEOUT)` per SSE line; `asyncio.to_thread` for PoW; `threading.Lock` in `AccountPool`/`PowSolver`; thread-local SQLite conns.
- Errors: narrow hierarchy, never bare `except`. `DeepSeekError(status, biz_code)` (~20 subclasses) → `app/main.py:_http_error()` maps biz `40002/40003→401`, `40029→429`, else passthrough or `502`, always `{"error":{"message,type,code}}`. Mid-stream failure after emit → `StreamInterruptedError` (chat: `{"error":…}` chunk, no `[DONE]`; Responses: `response.failed`); pre-emit → rotate accounts `max(pool.size,2)` tries then `TurnExhaustedError`. Empty upstream → `EmptyCompletionError` (no poison); failures → `mark_failure` + `_clear_session` + `delete_session_refs`.
- State/DI: constructor injection `ConversationManager(pool, solver, storage)`; module-global lazy singletons in `app/main.py` (`_pool/_storage/_solver/_manager`, `_active_*()` shims). Tests inject via `hook_set_state()` or `patch.object(main_mod, "_manager")`. Env via `os.environ` only (no dotenv): `HOST`, `PORT`, `API_KEY` (unset=open), `DEEPSEEK_INCLUDE_SOURCES|INCLUDE_SOURCES` (`1/true/yes/on`).
- Files: only `data:` URLs supported — `decode_data_url()`, `ensure_extension()`, `guess_mime()` in `turn.py`/`models.py`.
- Tooling: Always use codebase-memory-mcp.

## Important Files

| Path | Purpose |
|---|---|
| `server.py` | Entry: `setup.bootstrap()` + `uvicorn app.main:app` on `$HOST:$PORT` (defaults `127.0.0.1:34868`) |
| `app/main.py` | Routes (`/`, `/health`, `/v1/models`, `/v1/chat/completions`, `/v1/responses`, `/v1/responses/{id}`, `/accounts/reload`, `/v1/sessions/{key}`), translation, SSE, error map, singletons |
| `app/conversations.py` | `ConversationManager`, `Conversation`, `TurnResult`, `StreamEvent`, failover/replay |
| `app/deepseek.py` | `DeepSeekClient`, `DeepSeekError` tree, `BASE_URL`, `TARGET_COMPLETION/UPLOAD` |
| `app/turn.py`, `app/models.py` | Prompt/file/session-key shaping, `parse_model()`, request models |
| `app/aggregator.py`, `app/citations.py` | Patch-stream → deltas; citation rewrite + appendix |
| `app/accounts.py`, `app/storage.py`, `app/pow_solver.py` | Token pool (`parse_accounts()`), SQLite (`conversations/prefixes/response_links/response_snapshots`), PoW solver |
| `pyproject.toml` | Sole config: deps (`fastapi, httpx, pydantic, uvicorn, wasmtime==48.0.0`), hatchling (`packages=["app"]`), ruff/ty |
| `setup.py` | `bootstrap()`: vendored wasmtime install + `REQUIRED` dep gate |
| `restart.sh` | Ops: `pkill` + `fuser -k 34868/tcp` + `setsid nohup` + `/health` poll |
| `accounts.txt` | Secret (gitignored): `account N` blocks, only `userToken.value` required; reload via `POST /accounts/reload` |
| `README.md` | Sole doc: setup, env, endpoints, models, `accounts.txt` format |

Never commit `accounts.txt`, `*.token`, `.env*`, `server.log`, `data.sqlite*`, `.venv/`.

## Runtime/Tooling Preferences

- Runtime: **CPython 3.12 only** (`.python-version`, `requires-python>=3.12`). No Node/Bun/Go/TS/Docker/CI.
- Package manager: **`uv`** (`uv.lock` authoritative) with `pip` fallback via `setup.py` self-bootstrap (`--break-system-packages` on pip≥23; non-x86_64 falls back to PyPI `wasmtime`). Build: `hatchling` wheel (`uv build`).
- Lint/type policy (mandatory): Always use https://docs.astral.sh/ruff/ with everything enabled and https://docs.astral.sh/ty/ with everything enabled, then fix all of the issues. Make sure to actually fix all of the issues instead of suppressing them — never add `noqa`, `type: ignore`, or `per-file-ignores`/rule relaxations to silence; fix source. Config is already maximal in `pyproject.toml` (`[tool.ruff.lint] select=["ALL"]`, `[tool.ty.rules] all="error"` + strict equality/generics, `error-on-warning`).
- Gotchas: `ACCOUNTS_PATH` has absolute-dev-path fallback before `ROOT/accounts.txt`; README says `0.0.0.0:34868` but `server.py` defaults `127.0.0.1`; in-memory + SQLite duality (TTL vs `delete_stale_conversations`, LRU 10k vs SQL prune).

## Testing & QA

- Framework: **stdlib `unittest` only** (no pytest dep; `IsolatedAsyncioTestCase` for async). No `conftest.py`, no coverage config/thresholds, no CI.
- Layout: flat root — `test_branching_and_isolation.py`, `test_include_sources.py`, `test_integration.py`, `test_storage.py`, `test_stream_stall.py`, `test_turn_recovery.py`, `test_unified_model.py`, `test_user_scenario.py`, `test_vision_fork.py`, aggregated by `test_suite.py` (`unittest.main()` + `asyncio.run` bridges for bare `test_*` funcs in `test_storage`/`test_integration`/`test_turn_recovery`).
- Patterns: per-file helpers (`_make_manager/_make_request/_make_client`, `_check_equal/_check_in/...`), `DummySolver` (refuses real WASM PoW), `FakeDeepSeekClient` script-queue, `_FakeResponse/_FakeHttp` httpx doubles, `fastapi.testclient.TestClient` + `hook_set_state()`, `tempfile.TemporaryDirectory` for SQLite/`accounts.txt`. Naming: `test_<topic>.py`, `Test*` classes, `test_*` methods.
- Examples:
  ```python
  # test_turn_recovery.py — script failures then success through manager
  mgr = _make_manager(scripts=[[DeepSeekBizCodeError(...)], ["ok"]])
  res = await mgr.run_turn(key, prepared)


  # test_stream_stall.py — IsolatedAsyncioTestCase + httpx doubles
  class TestStreamStall(unittest.IsolatedAsyncioTestCase): ...
  ```
- Hygiene (must follow): use temp dirs, never repo-root `data.sqlite`/`accounts.txt`/`server.log`; keep PoW stubbed (`DummySolver`); don't parallelize `TestClient` tests (mutate `app.main` globals); close `patch.object(...).start()` with `addCleanup`/`tearDown` (existing `test_user_scenario`/`test_vision_fork` leak); `.pytest_cache/lastfailed` showed 16 historical `turn_recovery`/`storage` failures — re-run before assuming green.
