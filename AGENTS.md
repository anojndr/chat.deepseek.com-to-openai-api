# Repository Guidelines

## Project Overview

FastAPI proxy exposing `chat.deepseek.com` as OpenAI-compatible API (`/v1/chat/completions`, `/v1/responses`, `/v1/models`).
Browserless: reuses web `accounts.txt` tokens + WASM PoW (`DeepSeekHashV1` via `wasmtime`) over `httpx`.

## Architecture & Data Flow

Layered: `app/main.py` (routes/SSE) → `app/models.py` + `app/turn.py` (translation) → `app/conversations.py` (`ConversationManager`) → `app/accounts.py` (`AccountPool`) + `app/deepseek.py` (`DeepSeekClient`) → `chat.deepseek.com`.
Normalization: `app/aggregator.py` (`FragmentAggregator`) → `app/citations.py` (`CitationRewriter`); PoW: `app/pow_solver.py`; state: `app/storage.py` (SQLite WAL) + in-memory maps.

Request path:
`POST /v1/chat/completions|/v1/responses` → `parse_model` + `compute_history_hashes` → session key (`X-Session-Id` header else `find_prefix` longest-prefix lookup, fork-on-mismatch) → `prepare_turn` (first turn: full transcript; later: latest user msg + `parent_message_id`) → `_ensure_session` (reuse pinned session else `AccountPool.acquire` + `create_session`) → `_upload_files` + vision fork → `stream_completion` (fresh PoW per call via `asyncio.to_thread(PowSolver.solve)`, `x-ds-pow-response` header) → `FragmentAggregator.apply` → `CitationRewriter.feed` → `StreamEvent` → SSE (`_chat_stream`/`_responses_stream`) or `TurnResult`.
Failover: `DeepSeekError` clears pinned session + `mark_failure` (except `EmptyCompletion`, never poisons health), retry across accounts; mid-stream failure → SSE error payload. TTL sweeper (6h) prunes idle sessions; `delete_session` best-effort.

## Key Directories

- `app/`: all source, one concern per module (no `src/`). See Important Files.
- `app/vendor/sha3_wasm_bg.wasm`: PoW WASM blob (same bytes as web client).
- `vendor/wasmtime-48.0.0-py3-none-manylinux1_x86_64.whl`: vendored x86_64 wheel for `setup.py`.
- Root `test_*.py`: flat tests, no `tests/` dir.
- No `scripts/`, `docs/`, `examples/`, `Dockerfile`, `.github/`.

## Development Commands

```bash
python3 server.py              # foreground dev (HOST=127.0.0.1 PORT=34868 default)
./restart.sh                   # detached restart + /health gate, logs server.log
./restart.sh -f|--foreground   # same + tail log
python3 setup.py               # bootstrap only (wasmtime install + dep check)
pip install fastapi httpx uvicorn pydantic  # or: uv sync (uv.lock pinned)
curl localhost:34868/health
curl localhost:34868/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"deepseek-chat","messages":[{"role":"user","content":"hi"}]}'
python3 test_suite.py          # full suite
python3 -m unittest test_suite -v
python3 -m unittest test_storage test_integration test_turn_recovery test_user_scenario test_branching_and_isolation test_include_sources -v
```

Env (process env only, no `.env` file): `HOST`, `PORT`, `API_KEY` (unset=open; set=enforce `Bearer`/`x-api-key` on generation+admin, `/health` + `/v1/models` stay open), `DEEPSEEK_INCLUDE_SOURCES|INCLUDE_SOURCES` (default `0`) + per-request `include_sources` flag. Secrets in gitignored `accounts.txt` (localStorage JSON dumps, only `userToken.value` used); hot-reload via `POST /accounts/reload`.

## Code Conventions & Common Patterns

- Singletons at module scope + `lifespan` init: `_pool/_solver/_storage/_manager` in `app/main.py`; patch in tests via `main_mod._storage/_pool/_manager`.
- Types: `dataclasses` for domain (`Conversation`, `TurnResult`, `StreamEvent`, `PreparedTurn`, `ConvRef`); `Pydantic` for requests (`ChatCompletionRequest`, `ResponsesRequest`, `extra=ignore`).
- Async throughout: `AsyncIterator[StreamEvent]`, `run_turn`/`stream_turn`, per-key `asyncio.Lock` dict to serialize turns; `threading.Lock` only around `wasmtime Store` and `AccountPool` state.
- SSE patch protocol: handle `snapshot`/`patch`/`BATCH` in `FragmentAggregator`; buffer out-of-order deltas; route `THINK`→reasoning, `RESPONSE`→content, `TOOL_SEARCH`→search/sources.
- Citations: stream-safe `[citation:N]` → markdown link rewrite; append `source_appendix` when `include_sources` set.
- Errors: map to OpenAI envelope; `biz_code` `40002/40003`→401, `40029`→429. Cleanup best-effort (swallow `delete_session`, sweeper never raises).
- Tokens: estimated `len//4`, no tokenizer dep.
- Naming: `app/<concern>.py` (`main/deepseek/conversations/turn/aggregator/citations/accounts/storage/models/pow_solver`); `test_<feature>.py` mirrors feature; `UPPER_SNAKE` env; routes OpenAI-shaped (`/v1/chat/completions`, `/v1/responses`, `/health`, `DELETE /v1/sessions/{key}`).
- Tests use `tempfile.TemporaryDirectory` for sqlite/accounts (never touch real `data.sqlite`/`accounts.txt`); `FakeDeepSeekClient` with call-recording for failover/branching tests.

## Important Files

- `server.py`: entry — `sys.path` bootstrap + `uvicorn.run("app.main:app")`.
- `setup.py`: `bootstrap()`/`ensure_wasmtime()` — vendored wheel on x86_64 else PyPI, fail-fast on missing `fastapi/httpx/uvicorn/pydantic`.
- `app/main.py`: FastAPI app, all routes, SSE emitters, response-link LRU.
- `app/conversations.py`: orchestration — pinning, failover, replay, history, TTL sweeper.
- `app/deepseek.py`: `httpx` client — PoW header, session create/delete, upload/vision fork, SSE parser.
- `app/accounts.py`: `AccountPool` round-robin + exp cooldown (cap 900s), `parse_accounts`.
- `app/models.py`: `parse_model` (deepthink suffixes/aliases/vision), `decode_data_url`, content-part flattening.
- `app/turn.py`: `prepare_turn`, `compute_history_hashes`/`item_hash` rolling prefix hashes.
- `app/aggregator.py`: JSON-patch aggregation.
- `app/citations.py`: citation rewrite + appendix builder.
- `app/storage.py`: SQLite WAL (`conversations`, `prefixes` 20k cap/24h prune, `response_links/snapshots`), thread-local conns.
- `app/pow_solver.py`: `wasmtime` runner for `sha3_wasm_bg.wasm`.
- `pyproject.toml`, `uv.lock`, `.python-version`: metadata/deps/pins + `ty` config.
- `restart.sh`, `README.md`, `test_suite.py`: ops, sole doc (endpoints/models/accounts format), unittest aggregator.

## Tooling Mandate

Always use `https://docs.astral.sh/uv/` with everything enabled and `https://docs.astral.sh/ty/` with everything enabled, then fix all of the issues. Make sure to actually fix all of the issues instead of suppressing them.

- `ty` (docs: `https://docs.astral.sh/ty/`): `[tool.ty.rules] all = "error"`, `[tool.ty.analysis] strict-equality-semantics + strict-generic-narrowing + respect-type-ignore-comments = false`, `[tool.ty.environment] python-version = "3.12" + python-platform = "all"`, `terminal.error-on-warning = true`. No `type: ignore` / `ty: ignore` / `noqa` / `[[tool.ty.overrides]]` — fix root cause instead (e.g. function-scope imports after `sys.path` insert in `server.py:_bootstrap`, `importlib.util.find_spec("wasmtime")` in `setup.py:ensure_wasmtime`).
- `uv` (docs: `https://docs.astral.sh/uv/`): `[tool.uv] preview-features = true`; keep `uv.lock` in sync. Gate: `uv check`, `uv lock --check`, `uv sync --locked`, `uv pip check`, `uv audit`, `uv format --check`, `uv build` must all pass. Prefer `uv run <cmd>` so `ty` resolves `.venv`.
- Verify: `ty check --error all` + all `uv` cmds above + `python3 test_suite.py` (29 tests) before submit.

## Runtime/Tooling Preferences

- Required: CPython `3.12` (`.python-version` + `requires-python>=3.12`). NOT Node/Bun — no `package.json`/`tsconfig`/`bunfig.toml`.
- Package manager: `uv` (`uv.lock`) with `pip` fallback (`setup.py` uses `pip install <wheel>`). No `[project.scripts]`; hatchling `packages=["app"]` only for `uv build`.
- No Dockerfile, CI, Makefile, lint/format config. Static checks: `ty` max-strict (see Tooling Mandate) + `uv format --check`. `.ruff_cache/` is ad-hoc local runs only.

## Testing & QA

- Framework: stdlib `unittest` only (`TestCase` + `IsolatedAsyncioTestCase`; helpers driven via `asyncio.run` in `test_suite.py`). No pytest/jest/vitest/playwright, no coverage gate, no thresholds.
- Layout (~30 methods): `test_storage.py` (sqlite/manager persistence), `test_integration.py` (FastAPI `TestClient` + sqlite restart), `test_turn_recovery.py` (stale `parent_message_id`/dead-stream), `test_user_scenario.py` (citations + branching, 2 tests), `test_branching_and_isolation.py` (isolation/tree-branching, 2 tests), `test_include_sources.py` (~20 appendix/flag/SSE tests), `test_suite.py` (`TestSQLitePersistence` aggregator). Each file runnable via `__main__ unittest.main()`.
- Examples: `test_storage_basic` (save/get/update/delete + response-link round-trip in temp sqlite); `test_turn_recovery` + `FakeDeepSeekClient` (replay on stale parent); `ChatCompletionsAppendixEndpointTest` (`include_sources` appendix contract).
- QA: keep tests isolated (temp dirs, `AsyncMock`/`patch`), full suite via `python3 test_suite.py` before submit.
