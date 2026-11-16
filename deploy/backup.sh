#!/usr/bin/env bash
# Back up books, casts, progress, and rendered audio.
set -euo pipefail

cd "$(dirname "$0")/.."

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p backups
tar -czf "backups/bookcast-data-$stamp.tar.gz" data
echo "Wrote backups/bookcast-data-$stamp.tar.gz"
