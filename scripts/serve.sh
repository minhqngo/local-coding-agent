#!/usr/bin/env bash
# Phase 1, step 2 — launch one llama-server per role, each on its own port.
#
# Requires llama.cpp's `llama-server` on PATH (or set LLAMA_SERVER=/path/to/it).
# Logs + PIDs go to ./runs/. Stop everything with:  ./scripts/serve.sh stop
#
#   ./scripts/serve.sh start            # start all roles
#   ./scripts/serve.sh start coder      # start one role
#   ./scripts/serve.sh stop
#
#   HOST=0.0.0.0 ./scripts/serve.sh start   # accept LAN connections (remote
#   agent). Default binds 127.0.0.1 (local only). llama-server has NO auth —
#   only expose on networks you trust, or use an SSH tunnel instead.
set -euo pipefail

cd "$(dirname "$0")/.."
MODELS_DIR="${MODELS_DIR:-./models}"
RUNS_DIR="./runs"
LLAMA_SERVER="${LLAMA_SERVER:-llama-server}"
NGL="${NGL:-99}"          # GPU layers; set NGL=0 for CPU-only
HOST="${HOST:-127.0.0.1}" # bind address; 0.0.0.0 to serve a remote agent
mkdir -p "$RUNS_DIR"

cmd="${1:-start}"; shift || true

stop_all() {
  for pidfile in "$RUNS_DIR"/*.pid; do
    [ -e "$pidfile" ] || continue
    pid="$(cat "$pidfile")"
    if kill -0 "$pid" 2>/dev/null; then
      echo "stopping $(basename "$pidfile" .pid) (pid $pid)"
      kill "$pid" 2>/dev/null || true
    fi
    rm -f "$pidfile"
  done
}

start_role() {
  local role="$1"
  local port gguf draft ctx
  port="$(python3 scripts/_config.py field "$role" port)"
  gguf="$(python3 scripts/_config.py field "$role" gguf)"
  draft="$(python3 scripts/_config.py field "$role" draft_gguf)"
  ctx="$(python3 scripts/_config.py field "$role" ctx)"
  [ -z "$gguf" ] && { echo "skip $role: no gguf in config"; return 0; }

  # Don't spawn a duplicate if this role's server is already running.
  local pidfile="$RUNS_DIR/$role.pid"
  if [ -e "$pidfile" ] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
    echo "skip $role: already running (pid $(cat "$pidfile"))"; return 0
  fi

  local model_path="$MODELS_DIR/$gguf"
  if [ ! -f "$model_path" ]; then
    echo "skip $role: $model_path not found (run download_models.sh)"; return 0
  fi

  local args=(--model "$model_path" --port "$port" --host "$HOST"
              --alias "$(python3 scripts/_config.py field "$role" model)"
              -ngl "$NGL")
  # Per-role context window (a long-context planner served at the default ctx
  # wastes its whole advantage).
  [ -n "$ctx" ] && args+=(-c "$ctx")
  # Wire the draft model for speculative decoding when present (coder).
  if [ -n "$draft" ] && [ -f "$MODELS_DIR/$draft" ]; then
    args+=(--model-draft "$MODELS_DIR/$draft" --draft-max 16 --draft-min 1)
  fi

  echo "starting $role on $HOST:$port ($gguf${ctx:+, ctx=$ctx})"
  nohup "$LLAMA_SERVER" "${args[@]}" >"$RUNS_DIR/$role.log" 2>&1 &
  echo $! >"$pidfile"
}

# Portable replacement for `mapfile` (bash 4+ only; breaks on macOS bash 3.2).
read_roles() {
  python3 scripts/_config.py roles | while IFS= read -r r; do printf '%s\n' "$r"; done
}

case "$cmd" in
  stop) stop_all ;;
  start)
    if [ "$#" -eq 0 ]; then
      while IFS= read -r role; do [ -n "$role" ] && start_role "$role"; done < <(read_roles)
    else
      for role in "$@"; do start_role "$role"; done
    fi
    echo "servers launching; tail logs in $RUNS_DIR/. Verify with scripts/healthcheck.sh"
    ;;
  *) echo "usage: $0 {start|stop} [roles...]" >&2; exit 1 ;;
esac
