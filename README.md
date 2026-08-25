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

## Endpoints

| Endpoint | Notes |
|---|---|
| `POST /v1/chat/completions` | Full Chat Completions contract, streaming + non-streaming |
| `POST /v1/responses` | Responses API: string or item-list `input`, `instructions`, `previous_response_id`, typed SSE events |
| `GET /v1/models` | Model list |
| `GET /health` | Account pool status |
| `POST /accounts/reload` | Re-read `accounts.txt` without restart (add accounts live) |
| `DELETE /v1/sessions/{key}` | Forget a conversation |

## Models

- `deepseek-chat` — default (Instant)
- `deepseek-chat-deepthink` — thinking enabled
- `deepseek-reasoner[-deepthink]` — aliases
- `vision[-deepthink]` — DeepSeek Vision; forces `model_type:"vision"` even without attachments

Any model id ending in `-deepthink` (or `-think`) enables DeepSeek's thinking
mode; reasoning streams as `reasoning_content` deltas (Chat) /
`response.reasoning_text.delta` events (Responses).

## Behavior

- **Search is always on** (`search_enabled: true` on every completion).
- **Multi-turn**: pass a stable `X-Session-Id` header to pin one DeepSeek
  chat_session per conversation. Turn 1 sends the whole conversation; later
  turns send only the latest user message and continue via
  `parent_message_id` natively inside DeepSeek. Without the header the key
  derives from client IP + `user`.
- **Files**: attach via content parts (`image_url`, `file`, `input_file`) with
  `data:` URLs. Images are fork-parsed for Vision (`model_type:"vision"`);
  text/code/JSON/PDF etc. use normal parsing. Any extension in DeepSeek's
  supported list works.
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
