#!/usr/bin/env bash
# One-time VPS setup helper. Run from the repository root.
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is not installed. Install Docker Engine and the Compose plugin first." >&2
  exit 1
fi

if ! docker compose version >/dev/null 2>&1; then
  echo "The Docker Compose plugin is missing (docker compose)." >&2
  exit 1
fi

if [ ! -f .env ]; then
  cp .env.example .env
  echo
  echo "Created .env from .env.example."
  echo "Edit DOMAIN and BASIC_AUTH_USER, then generate BASIC_AUTH_HASH with:"
  echo "  docker run --rm caddy:2-alpine caddy hash-password --plaintext 'your-password'"
  echo
  echo "Run this script again after editing .env."
  exit 1
fi

if grep -q "replace-with-a-real-caddy-hash" .env; then
  echo "Set BASIC_AUTH_HASH in .env before starting." >&2
  exit 1
fi

if grep -q "^DOMAIN=app.example.com" .env; then
  echo "Set DOMAIN in .env to your real hostname before starting." >&2
  exit 1
fi

mkdir -p data/books data/runs data/models

# The app container runs as UID/GID 10001.
if [ "$(id -u)" -eq 0 ]; then
  chown -R 10001:10001 data
else
  sudo chown -R 10001:10001 data
fi

docker compose up -d --build
docker compose ps
