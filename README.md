# chat.deepseek.com → OpenAI-Compatible API

A FastAPI proxy that exposes **chat.deepseek.com** as an OpenAI-compatible API
(Chat Completions + Responses), browserless, using your own accounts from
`accounts.txt`.

## Quick start

```bash
python3 server.py          # serves on http://0.0.0.0:34868
```

First run installs the vendored `wasmtime` wheel automatically (needed for the
proof-of-work solver). Requires `fastapi`, `httpx`, `uvicorn`, `pydantic`.
### restart.sh

```bash
./restart.sh              # stop old instance, start new one in the background, exit
./restart.sh -f           # same, but tail server.log in this shell (Ctrl-C leaves the daemon running)
```

Logs land in `server.log`.

```bash
curl http://127.0.0.1:34868/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-chat","messages":[{"role":"user","content":"Hello"}]}'
```

## Configuration & security

| Env var | Default | Purpose |
|---|---|---|
| `HOST` | `127.0.0.1` | Bind address. The proxy holds your DeepSeek session tokens — only bind to a public interface if you know what you're doing. |
| `PORT` | `34868` | Listen port |
| `API_KEY` | unset | When set, all generation/admin routes require `Authorization: Bearer <key>` (or `x-api-key`). `/health` and `/v1/models` stay open. |
| `DEEPSEEK_INCLUDE_SOURCES` | `0` | Optional: append a "Sources" + "Search Queries" appendix to answers with web search results for llmcord-go's **Show Sources** button (`1`/`true`/`yes`/`on`); can also be set per request via `include_sources: true` (or `INCLUDE_SOURCES=1`). |

## Disclaimer

This tool drives **chat.deepseek.com** browserlessly using session tokens
extracted from a logged-in browser and replicates its internal
proof-of-work protocol. This is an unofficial client: it likely violates
DeepSeek's Terms of Service, may break when their web client changes, and
can get your accounts suspended or banned. Use at your own risk with
accounts you are willing to lose.

## Endpoints

| Endpoint | Notes |
|---|---|
| `POST /v1/chat/completions` | Full Chat Completions contract, streaming + non-streaming |
| `POST /v1/responses` | Responses API: string or item-list `input`, `instructions`, `previous_response_id`, typed SSE events |
| `GET /v1/responses/{response_id}` | Retrieve a stored response (LRU-bounded, in-process) |
| `GET /v1/models` | Model list |
| `GET /health` | Account pool status |
| `POST /accounts/reload` | Re-read `accounts.txt` without restart (add accounts live) |
| `DELETE /v1/sessions/{key}` | Forget a conversation |

## Models

chat.deepseek.com now serves one unified model behind its **Instant**,
**Expert**, and **Vision** entries — everyday chat, image understanding, and
complex problem-solving share the same upgraded backend. This proxy routes
everything through that unified backend:

- `deepseek-chat` — default unified entry (Instant). Handles text, files, and
  images natively; no special model needed for attachments.
- `deepseek-chat-deepthink` — same, with thinking enabled.
- `deepseek-reasoner` — alias; thinks by default (`deepseek-r1` too).
- `deepseek-expert` / `expert` — explicit Expert route to the unified backend.
- `deepseek-vision` / `vision` — explicit Vision route; still accepted, but no
  longer required for images (the default entry sees them natively).
- `deepseek-instant` / `instant` — spelling of the default entry.
- any model id also accepts `-think`/`-thinking` as the thinking suffix

Any model id ending in `-deepthink` (or `-think`) enables DeepSeek's thinking
mode; reasoning streams as `reasoning_content` deltas (Chat) /
`response.reasoning_text.delta` events (Responses).

## Behavior

- **Search is always on** (`search_enabled: true` on every completion).
- **Search citations are linked**: `[citation:N]` markers are rewritten to Markdown
  links using the corresponding DeepSeek search-result URL, in both streaming and
  non-streaming responses. Adjacent markers get a space between them so links
  don't run together (`[citation:2](url) [citation:10](url)`).
- **Sources appendix for llmcord-go**: To feed search citations into `llmcord-go`'s
  **Show Sources** button, enable globally with `DEEPSEEK_INCLUDE_SOURCES=1` (or
  `INCLUDE_SOURCES=1`) in your environment, or per request with `"include_sources": true`
  in the JSON request body (e.g. `extra_body: {include_sources: true}` in `llmcord-go`).
  When enabled, an appendix formatted as ``\n\nSources\n1. [Title](url) (domain) via `query`\n\nSearch Queries\n1. `query```
  is appended after the answer text.
- **Multi-turn**: pass a stable `X-Session-Id` header to pin one DeepSeek
  chat_session per conversation. Turn 1 sends the whole conversation; later
  turns send only the latest user message and continue via
  `parent_message_id` natively inside DeepSeek. Without the header the key
  derives from client IP + `user`.
- **Files**: attach via content parts (`image_url`, `file`, `input_file`) with
  `data:` URLs. Images upload as Vision-kind files and are understood by the
  default unified entry — no `deepseek-vision` needed. An explicit `vision`
  model still works; the legacy fork-parse is kept only as a tolerated
  fallback. Text/code/JSON/PDF etc. use normal parsing. Any extension in
  DeepSeek's supported list works.
- **Load balancing**: round-robin across every account in `accounts.txt`;
  add as many blocks as you like. Unhealthy accounts (muted, rate-limited,
  network failure, empty stream) get exponential cooldown and traffic rotates
  to healthy ones; they rejoin automatically.
- **Proof-of-work**: each completion/upload fetches a fresh challenge and
  solves `DeepSeekHashV1` by running DeepSeek's own `sha3_wasm_bg.wasm`
  through wasmtime (~50 ms typical).

## accounts.txt format

```
account 1
{ ...localStorage JSON with userToken.value ... }

account 2:
{ ... }
```

Only `userToken.value` is required; any other keys are ignored. After adding
an account, POST `/accounts/reload`.

## API surface notes

- Errors follow OpenAI's shape: `{"error":{"message","type","code"}}`.
- `stream_options.include_usage` supported (Chat); final usage chunk before
  `[DONE]`. Usage figures are estimates (DeepSeek doesn't return per-request
  token counts).
- Responses streaming emits the documented event chain:
  `response.created` → `response.in_progress` → `output_item.added` →
  `content_part.added` → `output_text.delta*` → `output_text.done` →
  `content_part.done` → `output_item.done` → `response.completed`.
