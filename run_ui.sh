#!/usr/bin/env bash
# Launch the Gradio UI.
#   ./run_ui.sh                 local browser on 127.0.0.1:7860
#   ./run_ui.sh --share         public link (Colab)
#   ./run_ui.sh --port 7861 --no-browser
#   BOOKCAST_USER=... BOOKCAST_PASSWORD=... ./run_ui.sh --host 0.0.0.0
set -euo pipefail
cd "$(dirname "$0")"

PY=".venv/bin/python"
if [ ! -x "$PY" ]; then
  PY="$(command -v python3)"
fi

exec "$PY" -m webui.app "$@"
