#!/usr/bin/env bash
# Restart the chat.deepseek.com -> OpenAI-compatible API proxy (port 34868).
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT=34868
# Background by default. Pass -f/--foreground to tail the logs in this shell.
FOREGROUND=0
for arg in "$@"; do
    case "$arg" in
        -f|--foreground) FOREGROUND=1 ;;
        *) echo "usage: $0 [-f|--foreground]" >&2; exit 1 ;;
    esac
done

# 1. Stop any running instance
pkill -f "python3 server.py" 2>/dev/null || true
for _ in $(seq 1 20); do
    pgrep -f "python3 server.py" >/dev/null 2>&1 || break
    sleep 0.5
done
# free the port if something else is squatting on it
if command -v fuser >/dev/null 2>&1; then
    fuser -k "${PORT}/tcp" 2>/dev/null || true
    sleep 1
fi

# 2. Start fresh, logging to file
cd "$DIR"
setsid nohup python3 server.py > "$DIR/server.log" 2>&1 < /dev/null &
disown 2>/dev/null || true
echo "[restart] started (pid $!)"

# 3. Wait until the health endpoint responds
for _ in $(seq 1 30); do
    if curl -sf -m 2 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        echo "[restart] healthy: http://127.0.0.1:${PORT}"
        break
    fi
    sleep 1
done

BASE_URL="http://127.0.0.1:${PORT}/v1"
LOG_PATH="$DIR/server.log"

cat <<EOF

─────────────────────────────────────────────────────
Base URL (copy & paste):
  ${BASE_URL}

Follow logs (copy & paste):
  tail -f ${LOG_PATH}
─────────────────────────────────────────────────────
EOF

# 4. Default: exit and leave the server in the background.
#    With -f/--foreground: follow the log in this shell (Ctrl-C stops the
#    tail only — the detached server keeps running).
if [ "$FOREGROUND" -eq 1 ]; then
    exec tail -f "$LOG_PATH"
fi
