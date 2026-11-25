#!/usr/bin/env bash
# Launch BookCast Studio on Android through Termux.
#
#   ./run_termux.sh
#   PORT=7861 ./run_termux.sh
#   BOOKCAST_USER=bookcast BOOKCAST_PASSWORD='...' ./run_termux.sh
#
# The app binds to 0.0.0.0 so another device on the same Wi-Fi can open it.
set -euo pipefail

cd "$(dirname "$0")"

PORT="${PORT:-7860}"
HOST="${HOST:-0.0.0.0}"
OPEN="${OPEN:-1}"

PY=".venv/bin/python"
if [ ! -x "$PY" ]; then
  PY="$(command -v python3 || command -v python || true)"
fi
if [ -z "$PY" ]; then
  echo "Python is missing. Run ./deploy/termux-setup.sh first." >&2
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg is missing. Run: pkg install ffmpeg" >&2
  exit 1
fi

if ! "$PY" -c 'import gradio' >/dev/null 2>&1; then
  echo "BookCast dependencies are missing. Run ./deploy/termux-setup.sh first." >&2
  exit 1
fi

lan_ip=""
if command -v ip >/dev/null 2>&1; then
  lan_ip="$(ip -4 addr show 2>/dev/null | awk '/inet / && $2 !~ /^127\./ {split($2, a, "/"); print a[1]; exit}')"
fi
if [ -z "$lan_ip" ] && command -v ifconfig >/dev/null 2>&1; then
  lan_ip="$(ifconfig 2>/dev/null | awk '/inet / && $2 !~ /^127\./ {print $2; exit}')"
fi

wake_locked=0
if command -v termux-wake-lock >/dev/null 2>&1; then
  termux-wake-lock || true
  wake_locked=1
fi

cleanup() {
  if [ "$wake_locked" = "1" ] && command -v termux-wake-unlock >/dev/null 2>&1; then
    termux-wake-unlock || true
  fi
}
trap cleanup EXIT INT TERM

echo
echo "BookCast Studio"
echo "  This phone:  http://127.0.0.1:${PORT}"
if [ -n "$lan_ip" ]; then
  echo "  Same Wi-Fi:  http://${lan_ip}:${PORT}"
else
  echo "  Same Wi-Fi:  http://<phone-ip>:${PORT}"
fi
if [ -n "${BOOKCAST_USER:-}" ] && [ -n "${BOOKCAST_PASSWORD:-}" ]; then
  echo "  Basic auth:  on"
else
  echo "  Basic auth:  off — only use this on a trusted Wi-Fi network."
fi
echo
echo "Keep Termux open. Press Ctrl+C to stop."
echo

if [ "$OPEN" = "1" ] && command -v termux-open-url >/dev/null 2>&1; then
  (
    for _ in $(seq 1 60); do
      if "$PY" - "$PORT" <<'PY'
import socket
import sys

sock = socket.socket()
sock.settimeout(0.5)
try:
    sock.connect(("127.0.0.1", int(sys.argv[1])))
except OSError:
    raise SystemExit(1)
finally:
    sock.close()
PY
      then
        termux-open-url "http://127.0.0.1:${PORT}" >/dev/null 2>&1 || true
        break
      fi
      sleep 1
    done
  ) &
fi

./run_ui.sh --host "$HOST" --port "$PORT" --no-browser
