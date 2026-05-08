#!/usr/bin/env bash
# reset.sh — full environment reset: tear down containers, wipe data, start fresh.
#
# What it does:
#   1. Stops and removes all containers in the compose stack
#   2. Deletes docker-data/* (MySQL, Qdrant, Neo4j persistent storage)
#   3. Clears uploads/ but keeps the directory structure
#   4. Brings everything back up with --wait for health checks
#
# What it does NOT do:
#   - Touch model caches in assets/fastembed or assets/reranker
#   - Modify config files (.env, docker-compose.yml)
#
# Usage:
#   ./reset.sh              # default: asks for confirmation
#   ./reset.sh --yes        # skip confirmation prompt

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE="docker compose -f $SCRIPT_DIR/docker-compose.dev.yml"
DATA_DIR="$SCRIPT_DIR/docker-data"
UPLOADS_DIR="$SCRIPT_DIR/uploads"

# ── Confirmation ───────────────────────────────────────────────────────────────
if [[ "${1:-}" != "--yes" ]]; then
  echo ""
  echo "  This will PERMANENTLY:"
  echo "    • Stop and remove all containers"
  echo "    • Delete $DATA_DIR (MySQL, Qdrant, Neo4j storage)"
  echo "    • Delete all files under $UPLOADS_DIR"
  echo "    • Restart all containers fresh"
  echo ""
  read -r -p "  Type 'yes' to continue: " CONFIRM
  if [[ "$CONFIRM" != "yes" ]]; then
    echo "Aborted."
    exit 0
  fi
fi

# ── 1. Tear down containers ────────────────────────────────────────────────────
echo ""
echo "[1/3] Stopping and removing containers..."
$COMPOSE down --remove-orphans
echo "      Done."

# ── 2. Delete data directories ─────────────────────────────────────────────────
echo ""
echo "[2/3] Deleting data directories..."

for dir in "$DATA_DIR/mysql" "$DATA_DIR/qdrant" "$DATA_DIR/neo4j"; do
  if [[ -d "$dir" ]]; then
    rm -rf "$dir"
    echo "      deleted: $dir"
  else
    echo "      skipped (not found): $dir"
  fi
done

rm -rf "$UPLOADS_DIR"
mkdir -p "$UPLOADS_DIR/temp"
echo "      cleared: $UPLOADS_DIR"
echo "      Done."

# ── 3. Bring everything back up ────────────────────────────────────────────────
echo ""
echo "[3/3] Starting containers (waiting for health checks)..."
$COMPOSE up -d --wait
echo "      Done."

echo ""
echo "Reset complete. All containers are up and healthy."
