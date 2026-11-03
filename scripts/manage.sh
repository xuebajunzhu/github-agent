#!/usr/bin/env bash
# ============================================================
#  GitHub Agent - deployment control plane (Linux/macOS).
#  Same verbs as scripts/manage.ps1 on Windows.
#  Usage: scripts/manage.sh <status|start|stop|restart|ensure|verify|deploy|preload>
#
#  Process model: if the systemd user unit (github-agent) is installed,
#  systemctl drives the process (Restart=always gives self-healing);
#  otherwise a nohup process with a pidfile under logs/ is used.
# ============================================================
set -euo pipefail
cd "$(dirname "$0")/.."

PORT=8788
BASE="http://127.0.0.1:$PORT"
PY=.venv/bin/python
PIDFILE=logs/api.pid
UNIT=github-agent
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

have_systemd_unit() { systemctl --user list-unit-files "$UNIT.service" 2>/dev/null | grep -q "$UNIT"; }
healthy() { curl -sf -m 4 "$BASE/health" >/dev/null 2>&1; }
wait_healthy() { # wait_healthy <seconds>
    local deadline=$(( $(date +%s) + $1 ))
    while (( $(date +%s) < deadline )); do
        healthy && return 0
        sleep 3
    done
    return 1
}

start_raw() {
    if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
        echo "raw process already running (pid $(cat "$PIDFILE"))"; return 0
    fi
    mkdir -p logs
    TRUST_PROXY_HEADERS=true nohup "$PY" -m uvicorn app.main:app \
        --host 0.0.0.0 --port "$PORT" >> logs/agent.log 2>&1 &
    echo $! > "$PIDFILE"
    echo "API process launched (pid $(cat "$PIDFILE"), loading models ~60s) ..."
}

stop_raw() {
    if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
        echo "Stopping API process (pid $(cat "$PIDFILE")) ..."
        kill "$(cat "$PIDFILE")" 2>/dev/null || true
        sleep 2
        kill -9 "$(cat "$PIDFILE")" 2>/dev/null || true
    fi
    rm -f "$PIDFILE"
    # last resort: anything still bound to the port
    pkill -f "uvicorn app.main:app" 2>/dev/null || true
}

action="${1:-status}"
case "$action" in
    status)
        if healthy; then
            echo "API: RUNNING"
            curl -s -m 4 "$BASE/info" | head -c 120; echo
        else
            echo "API: STOPPED or unhealthy"
        fi
        ;;
    start)
        if have_systemd_unit; then
            systemctl --user start "$UNIT"
        else
            start_raw
        fi
        wait_healthy 150 || { echo "[ERROR] not healthy within 150s"; exit 1; }
        echo "API is healthy on $BASE"
        ;;
    stop)
        if have_systemd_unit; then systemctl --user stop "$UNIT"; else stop_raw; fi
        echo "API stopped."
        ;;
    restart)
        if have_systemd_unit; then systemctl --user restart "$UNIT"; else stop_raw; start_raw; fi
        wait_healthy 150 || { echo "[ERROR] not healthy after restart"; exit 1; }
        echo "API restarted and healthy"
        ;;
    ensure)
        if healthy; then echo "ensure: API already healthy."
        else
            if have_systemd_unit; then systemctl --user restart "$UNIT"; else stop_raw; start_raw; fi
            wait_healthy 150 || { echo "[ERROR] ensure could not recover the API"; exit 1; }
            echo "ensure: API recovered."
        fi
        ;;
    verify)
        fail=0
        healthy && echo "[PASS] /health" || { echo "[FAIL] /health"; fail=1; }
        curl -sf -m 5 "$BASE/info" >/dev/null && echo "[PASS] /info" || { echo "[FAIL] /info"; fail=1; }
        curl -sf -m 60 -X POST "$BASE/api/recommend" -H 'Content-Type: application/json' \
            -d '{"query": "terminal file manager", "top_k": 2}' >/dev/null \
            && echo "[PASS] /api/recommend" || { echo "[FAIL] /api/recommend"; fail=1; }
        curl -sf -m 10 -X POST "$BASE/mcp" -H 'Content-Type: application/json' \
            -d '{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}' >/dev/null \
            && echo "[PASS] /mcp initialize" || { echo "[FAIL] /mcp"; fail=1; }
        (( fail == 0 )) && echo "VERIFY PASSED" || { echo "VERIFY FAILED"; exit 1; }
        ;;
    deploy)
        git pull --ff-only
        "$PY" -m pip install --quiet -r requirements.txt
        "$PY" -m pytest -q || { echo "[ERROR] test gate failed - NOT restarting"; exit 1; }
        if have_systemd_unit; then systemctl --user restart "$UNIT"; else stop_raw; start_raw; fi
        wait_healthy 150 || { echo "[ERROR] not healthy after deploy"; exit 1; }
        "$0" verify
        ;;
    preload)
        "$PY" -c "from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-m3', allow_patterns=['*.json','*.txt','*.safetensors','*.model','*.pt','pytorch_model.bin']); print('bge-m3 cached')"
        "$PY" -c "from laya import Router; Router(preload=True); print('laya checkpoints cached')"
        ;;
    *)
        echo "Usage: scripts/manage.sh <status|start|stop|restart|ensure|verify|deploy|preload>"
        exit 1
        ;;
esac
