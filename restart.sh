#!/usr/bin/env bash
# Restart the chat.deepseek.com -> OpenAI-compatible API proxy (port 34868).
#
#   ./restart.sh        stop old instance, start new one in background, exit
#   ./restart.sh -f     same, then tail server.log (Ctrl-C stops tail only --
#                       the detached server keeps running)
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT=34868
SERVER="$DIR/server.py"
LOG_PATH="$DIR/server.log"

FOREGROUND=0
for arg in "$@"; do
    case "$arg" in
        -f|--foreground) FOREGROUND=1 ;;
        *) echo "usage: $0 [-f|--foreground]" >&2; exit 1 ;;
    esac
done

# 1. Stop any running instance (pattern anchored to THIS repo's script path)
pkill -f "python3 $SERVER" 2>/dev/null || true
for _ in $(seq 1 20); do
    pgrep -f "python3 $SERVER" >/dev/null 2>&1 || break
    sleep 0.5
done
if pgrep -f "python3 $SERVER" >/dev/null 2>&1; then
    echo "[restart] ERROR: old instance still running after 10s; not starting a duplicate." >&2
    exit 1
fi
# free the port if something else is squatting on it
if command -v fuser >/dev/null 2>&1; then
    fuser -k "${PORT}/tcp" 2>/dev/null || true
    sleep 1
fi

# 2. Start fresh, detached, logging to file
cd "$DIR"
setsid nohup python3 "$SERVER" > "$LOG_PATH" 2>&1 < /dev/null &
disown 2>/dev/null || true
echo "[restart] started (pid $!)"

# 3. Wait until the health endpoint responds; report failure honestly
HEALTHY=0
for _ in $(seq 1 30); do
    if curl -sf -m 2 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        HEALTHY=1
        break
    fi
    sleep 1
done

if [ "$HEALTHY" -ne 1 ]; then
    echo "[restart] ERROR: server did not become healthy within 30s." >&2
    echo "[restart] Check $LOG_PATH for the cause. Common issues:" >&2
    echo "[restart]   - accounts.txt missing or malformed" >&2
    echo "[restart]   - port $PORT already in use" >&2
    exit 1
fi

BASE_URL="http://127.0.0.1:${PORT}/v1"

cat <<INFO

-----------------------------------------------------
Base URL (copy & paste):
  ${BASE_URL}

Follow logs (copy & paste):
  tail -f ${LOG_PATH}
-----------------------------------------------------
INFO

# 4. Foreground mode follows the log; default mode exits and leaves the
#    detached server running.
if [ "$FOREGROUND" -eq 1 ]; then
    exec tail -f "$LOG_PATH"
fi
