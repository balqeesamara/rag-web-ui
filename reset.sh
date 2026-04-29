#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Stopping containers..."
docker compose -f "$SCRIPT_DIR/docker-compose.dev.yml" down

echo "Removing data directories..."
rm -rf \
  "$SCRIPT_DIR/docker-data/mysql" \
  "$SCRIPT_DIR/docker-data/qdrant" \
  "$SCRIPT_DIR/uploads"

echo "Recreating empty upload dir..."
mkdir -p "$SCRIPT_DIR/uploads"

echo "Starting fresh..."
docker compose -f "$SCRIPT_DIR/docker-compose.dev.yml" up -d

echo "Waiting for backend..."
until curl -s http://localhost:8000/api/health > /dev/null 2>&1; do
  sleep 1
done

echo "Done. System is reset and ready."
