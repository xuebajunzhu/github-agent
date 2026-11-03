#!/usr/bin/env bash
# ============================================================
#  GitHub Agent - one-command fresh install (Linux/macOS)
#    scripts/setup_linux.sh [--with-systemd]
#  Creates the venv, installs dependencies, seeds .env with
#  RANDOM SECRET_KEY/ADMIN_TOKEN, initializes the DB, and
#  optionally installs a systemd user service (self-healing).
# ============================================================
set -euo pipefail
cd "$(dirname "$0")/.."

WITH_SYSTEMD=false
[[ "${1:-}" == "--with-systemd" ]] && WITH_SYSTEMD=true

echo "[1/6] Checking Python 3.10+ ..."
PY=python3
command -v "$PY" >/dev/null || { echo "[ERROR] python3 not found"; exit 1; }
"$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' || {
    echo "[ERROR] Python >= 3.10 required"; exit 1; }

if [[ -x ".venv/bin/python" ]]; then
    echo "[2/6] venv already exists - reusing."
else
    echo "[2/6] Creating venv ..."
    "$PY" -m venv .venv
fi

echo "[3/6] Installing dependencies ..."
.venv/bin/python -m pip install --quiet --upgrade pip
.venv/bin/python -m pip install --quiet -r requirements.txt -r requirements-dev.txt

if [[ -f ".env" ]]; then
    echo "[4/6] .env already exists - keeping it."
else
    echo "[4/6] Seeding .env with RANDOM secrets ..."
    .venv/bin/python - <<'EOF'
import secrets
from pathlib import Path
t = Path(".env.example").read_text(encoding="utf-8")
t = t.replace("please-change-me-to-a-random-string", secrets.token_urlsafe(48))
t = t.replace("change-me-admin-token", secrets.token_urlsafe(24))
Path(".env").write_text(t, encoding="utf-8")
print("       .env created (SECRET_KEY/ADMIN_TOKEN randomized)")
EOF
fi

echo "[5/6] Initializing database ..."
.venv/bin/python -c "
from app.database import create_db_engine, init_db
from app.config import Settings
init_db(create_db_engine(Settings().database_url))
print('       database ready')"

echo "[6/6] Smoke test ..."
.venv/bin/python -m pytest -q tests/test_api.py tests/test_mcp.py

if [[ "$WITH_SYSTEMD" == true ]]; then
    echo "[extra] Installing systemd user service ..."
    mkdir -p ~/.config/systemd/user
    ROOT="$(pwd)"
    sed "s|@ROOT@|$ROOT|g" deploy/systemd/github-agent.service \
        > ~/.config/systemd/user/github-agent.service
    systemctl --user daemon-reload
    systemctl --user enable --now github-agent
    echo "       systemd unit installed: systemctl --user status github-agent"
    echo "       (linger keeps it alive without login: sudo loginctl enable-linger $USER)"
else
    echo "[extra] Start manually with: scripts/manage.sh start"
fi

echo
echo "============================================================"
echo " Install complete. Next steps:"
echo "   start (systemd):   scripts/manage.sh start"
echo "   update in place:   scripts/manage.sh deploy"
echo "   full verification: scripts/manage.sh verify"
echo "   UI: http://127.0.0.1:8788/   MCP: POST /mcp"
echo "============================================================"
