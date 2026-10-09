#!/usr/bin/env bash
# Start/stop the MLflow tracking server as a detached daemon.
#
# The server is a plain read/write UI over the same tracking store the training
# runs write to (`sqlite:///mlflow.db` by default), so it can be restarted at
# any time without disturbing training.
#
# Usage: scripts/mlflow_server.sh {start|stop|restart|status|logs}
#
# Overridable via environment: MLFLOW_HOST, MLFLOW_PORT, MLFLOW_WORKERS,
# MLFLOW_BACKEND_STORE, MLFLOW_ALLOWED_HOSTS, MLFLOW_CORS_ALLOWED_ORIGINS,
# MLFLOW_LOG_FILE, MLFLOW_PID_FILE. All of them can also be set persistently in
# scripts/mlflow_server.env, which this script sources when present (the file is
# local configuration; real environment variables still win).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# shellcheck source=/dev/null
[[ -f scripts/mlflow_server.env ]] && . scripts/mlflow_server.env

HOST="${MLFLOW_HOST:-0.0.0.0}"
PORT="${MLFLOW_PORT:-5000}"
WORKERS="${MLFLOW_WORKERS:-1}"
BACKEND_STORE="${MLFLOW_BACKEND_STORE:-sqlite:///mlflow.db}"

# Host validation and CORS are left to MLflow's defaults unless explicitly
# overridden. The defaults already accept `localhost:<port>`, `127.0.0.1:<port>`
# and RFC-1918 private ranges, and default CORS to localhost origins on any
# port. Do not "widen" these to '*':
#   - `--allowed-hosts '*'` turns off DNS-rebinding protection outright.
#   - `--cors-allowed-origins '*'` makes the server disable credentialed CORS
#     ("Wildcard origins with credentials are not allowed"), which breaks the
#     UI's `credentials: include` fetches.
# Hosts are compared including the port, so an entry must be `name:5000`, not
# `name`. Override only if you browse through a tunnel under another name.
ALLOWED_HOSTS="${MLFLOW_ALLOWED_HOSTS:-}"
CORS_ALLOWED_ORIGINS="${MLFLOW_CORS_ALLOWED_ORIGINS:-}"
LOG_FILE="${MLFLOW_LOG_FILE:-logs/mlflow-server.log}"
PID_FILE="${MLFLOW_PID_FILE:-logs/mlflow-server.pid}"

# `uv run --no-sync` uses the current environment as-is. A plain
# `uv run --extra cpu ...` would re-sync and swap the CUDA torch build that
# live training processes are running against.
MLFLOW=(uv run --no-sync mlflow server
  --host "$HOST"
  --port "$PORT"
  --workers "$WORKERS"
  --backend-store-uri "$BACKEND_STORE")

# Passed only when explicitly set, so the empty default keeps MLflow's own
# localhost/private-range defaults instead of an empty allowlist.
[[ -n "$ALLOWED_HOSTS" ]] && MLFLOW+=(--allowed-hosts "$ALLOWED_HOSTS")
[[ -n "$CORS_ALLOWED_ORIGINS" ]] && MLFLOW+=(--cors-allowed-origins "$CORS_ALLOWED_ORIGINS")

HEALTH_URL="http://127.0.0.1:${PORT}/health"

server_pid() {
  [[ -f "$PID_FILE" ]] || return 1
  local pid
  pid="$(cat "$PID_FILE" 2>/dev/null)" || return 1
  [[ -n "$pid" ]] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  printf '%s' "$pid"
}

is_healthy() {
  curl -fsS -m 5 "$HEALTH_URL" >/dev/null 2>&1
}

wait_healthy() {
  local attempts="${1:-60}"
  for ((i = 0; i < attempts; i++)); do
    if is_healthy; then
      return 0
    fi
    sleep 0.5
  done
  return 1
}

start() {
  if pid="$(server_pid)"; then
    echo "already running (pid ${pid}), $(curl -fsS -m 5 "$HEALTH_URL" 2>/dev/null || echo 'unhealthy')"
    echo "log: ${LOG_FILE}"
    return 0
  fi
  if is_healthy; then
    echo "port ${PORT} already serves MLflow but ${PID_FILE} has no live pid." >&2
    echo "stop the other process first, or set MLFLOW_PID_FILE/MLFLOW_PORT." >&2
    return 1
  fi

  mkdir -p "$(dirname "$LOG_FILE")" "$(dirname "$PID_FILE")"
  rm -f "$PID_FILE"

  # setsid detaches into its own session, so the server outlives this shell and
  # any tmux pane it was started from. The inner shell records its own PID
  # before `exec`, so the pidfile holds the server's PID even when setsid forks.
  setsid nohup bash -c 'echo $$ >"$1"; shift; exec "$@"' _ "$PID_FILE" "${MLFLOW[@]}" \
    >>"$LOG_FILE" 2>&1 &
  disown || true

  if ! wait_healthy 60; then
    echo "server failed to become healthy; last log lines:" >&2
    tail -n 20 "$LOG_FILE" >&2 || true
    return 1
  fi

  echo "started (pid $(cat "$PID_FILE"))"
  echo "url:  http://127.0.0.1:${PORT}/  (bound to ${HOST}:${PORT})"
  echo "store: ${BACKEND_STORE}"
  echo "log:  ${LOG_FILE}"
}

stop() {
  if ! pid="$(server_pid)"; then
    rm -f "$PID_FILE"
    echo "not running (no live pid in ${PID_FILE})"
    return 0
  fi

  # Negative PID signals the whole process group: `mlflow server` spawns uvicorn
  # workers, a job runner, and huey consumers that must go down with it.
  kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
  for _ in {1..40}; do
    kill -0 "$pid" 2>/dev/null || break
    sleep 0.25
  done
  if kill -0 "$pid" 2>/dev/null; then
    kill -KILL -- "-${pid}" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
  fi

  rm -f "$PID_FILE"
  if is_healthy; then
    echo "sent SIGTERM/SIGKILL to pid ${pid}, but ${HEALTH_URL} still answers;"
    echo "another MLflow server may have taken over the port."
    return 1
  fi
  echo "stopped (pid ${pid})"
}

status() {
  if pid="$(server_pid)"; then
    echo "pid:  ${pid} (running)"
  else
    echo "pid:  none (not managed by ${PID_FILE})"
  fi
  if is_healthy; then
    echo "http: $(curl -fsS -m 5 "$HEALTH_URL")  -> healthy"
  else
    echo "http: ${HEALTH_URL} unreachable"
  fi
  echo "store: ${BACKEND_STORE}"
  echo "bind:  ${HOST}:${PORT}"
  echo "log:   ${LOG_FILE}$([[ -f "$LOG_FILE" ]] && echo " ($(wc -l <"$LOG_FILE") lines)")"
}

case "${1:-status}" in
  start) start ;;
  stop) stop ;;
  restart)
    stop
    start
    ;;
  status) status ;;
  logs) tail -n 50 -f "$LOG_FILE" ;;
  *)
    echo "usage: $0 {start|stop|restart|status|logs}" >&2
    exit 2
    ;;
esac
