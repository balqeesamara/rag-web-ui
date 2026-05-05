#!/usr/bin/env bash
# reset.sh — wipe all data without stopping containers
#
# What it does:
#   1. Truncates every table in the MySQL ragwebui database (schema kept intact)
#   2. Deletes all Qdrant collections (kb_* and any others)
#   3. Clears uploads/ and uploads/temp/ but keeps the directories
#
# What it does NOT do:
#   - Stop or restart any container
#   - Drop/recreate the database schema
#   - Touch model caches or config files
#
# Usage:
#   ./reset.sh              # default: asks for confirmation
#   ./reset.sh --yes        # skip confirmation prompt

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/.env"

# ── Load .env ──────────────────────────────────────────────────────────────────
if [[ ! -f "$ENV_FILE" ]]; then
  echo "ERROR: .env not found at $SCRIPT_DIR/.env" >&2
  exit 1
fi

# Export only the vars we need (skip comments, blanks, keys with special chars)
while IFS='=' read -r key value; do
  [[ "$key" =~ ^#.*$ || -z "$key" ]] && continue
  # Strip surrounding quotes from value if present
  value="${value%\"}"
  value="${value#\"}"
  value="${value%\'}"
  value="${value#\'}"
  export "$key=$value"
done < <(grep -E '^(MYSQL_|QDRANT_)' "$ENV_FILE")

DB_HOST="${MYSQL_SERVER:-db}"
DB_PORT="${MYSQL_PORT:-3306}"
DB_USER="${MYSQL_USER:-ragwebui}"
DB_PASS="${MYSQL_PASSWORD}"
DB_NAME="${MYSQL_DATABASE:-ragwebui}"
QDRANT_HOST="${QDRANT_HOST:-qdrant}"
QDRANT_PORT="${QDRANT_PORT:-6333}"
UPLOADS_DIR="$SCRIPT_DIR/uploads"

# ── Confirmation ───────────────────────────────────────────────────────────────
if [[ "${1:-}" != "--yes" ]]; then
  echo ""
  echo "  This will PERMANENTLY delete:"
  echo "    • All rows in every MySQL table in '$DB_NAME'"
  echo "    • All Qdrant collections"
  echo "    • All files under $UPLOADS_DIR"
  echo ""
  read -r -p "  Type 'yes' to continue: " CONFIRM
  if [[ "$CONFIRM" != "yes" ]]; then
    echo "Aborted."
    exit 0
  fi
fi

# ── Helper: run MySQL inside the db container ──────────────────────────────────
mysql_exec() {
  docker compose -f "$SCRIPT_DIR/docker-compose.dev.yml" exec -T db \
    mysql -u"$DB_USER" -p"$DB_PASS" "$DB_NAME" -e "$1" 2>/dev/null
}

# ── 1. MySQL: truncate all tables ─────────────────────────────────────────────
echo ""
echo "[1/3] Truncating MySQL tables in '$DB_NAME'..."

# Gather table list
TABLES=$(mysql_exec "SHOW TABLES;" | tail -n +2)

if [[ -z "$TABLES" ]]; then
  echo "      No tables found (database might be empty)."
else
  # Disable FK checks, truncate all, re-enable
  TRUNCATE_SQL="SET FOREIGN_KEY_CHECKS=0;"
  while IFS= read -r table; do
    [[ -z "$table" ]] && continue
    TRUNCATE_SQL+=" TRUNCATE TABLE \`$table\`;"
    echo "      truncating: $table"
  done <<< "$TABLES"
  TRUNCATE_SQL+=" SET FOREIGN_KEY_CHECKS=1;"
  mysql_exec "$TRUNCATE_SQL"
  echo "      Done."
fi

# ── 2. Qdrant: delete all collections ─────────────────────────────────────────
echo ""
echo "[2/3] Deleting Qdrant collections..."

# Qdrant is on the docker bridge; call via its published port on the host
QDRANT_URL="http://localhost:$QDRANT_PORT"

COLLECTIONS=$(curl -sf "$QDRANT_URL/collections" | \
  python3 -c "import sys,json; data=json.load(sys.stdin); print('\n'.join(c['name'] for c in data['result']['collections']))" 2>/dev/null || true)

if [[ -z "$COLLECTIONS" ]]; then
  echo "      No collections found."
else
  while IFS= read -r col; do
    [[ -z "$col" ]] && continue
    STATUS=$(curl -sf -X DELETE "$QDRANT_URL/collections/$col" | \
      python3 -c "import sys,json; print(json.load(sys.stdin).get('status','unknown'))" 2>/dev/null || echo "error")
    echo "      deleted collection '$col': $STATUS"
  done <<< "$COLLECTIONS"
  echo "      Done."
fi

# ── 3. Uploads: clear files, keep directories ─────────────────────────────────
echo ""
echo "[3/3] Clearing uploads..."

# Recreate uploads and uploads/temp as empty dirs
rm -rf "$UPLOADS_DIR"
mkdir -p "$UPLOADS_DIR/temp"

echo "      $UPLOADS_DIR cleared."
echo ""
echo "Reset complete. Containers are still running."
