#!/usr/bin/env bash
# Install BookCast dependencies in native Termux.
#
# If this fails while building pydantic-core, orjson, or another Rust/C
# package, use the proot-distro Ubuntu route in deploy/TERMUX.md instead.
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v pkg >/dev/null 2>&1; then
  echo "This script must run inside Termux." >&2
  exit 1
fi

echo "Updating Termux packages..."
pkg update -y
pkg upgrade -y

echo "Installing system tools and prebuilt numeric packages..."
pkg install -y python ffmpeg git qrencode
pkg install -y termux-api || true
pkg install -y python-numpy python-pandas || true

echo "Installing Python dependencies..."
python -m pip install --upgrade pip
if ! python -m pip install -r requirements-termux.txt; then
  cat >&2 <<'EOF'

Native Termux install failed.

Some Gradio dependencies (usually pydantic-core or orjson) need a Rust/C
toolchain on Android. You can either install the build tools:

    pkg install -y rust clang make binutils

and retry, or use the more reliable proot-distro Ubuntu route in
deploy/TERMUX.md.
EOF
  exit 1
fi

cat <<'EOF'

BookCast is installed.

Start it with:

    ./run_termux.sh

For basic auth on a shared Wi-Fi:

    BOOKCAST_USER=bookcast BOOKCAST_PASSWORD='choose-a-password' ./run_termux.sh
EOF
