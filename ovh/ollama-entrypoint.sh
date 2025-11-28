#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${OLLAMA_MODELS:-/var/lib/ollama}"
PRELOAD="${OLLAMA_PRELOAD_MODELS:-}"
HEALTH_FILE="${DATA_DIR}/.preload-complete"
LOG_PREFIX="[ollama-entrypoint]"

log() {
  echo "${LOG_PREFIX} $1"
}

ensure_data_dir() {
  if [ ! -d "$DATA_DIR" ]; then
    log "Creating model directory at $DATA_DIR"
    mkdir -p "$DATA_DIR"
    chmod 700 "$DATA_DIR"
  fi
}

start_temp_daemon() {
  log "Starting temporary Ollama daemon for preload"
  /usr/local/bin/ollama serve >/tmp/ollama-preload.log 2>&1 &
  echo $!
}

preload_models() {
  local daemon_pid="$1"
  IFS=', ' read -ra MODELS <<< "$PRELOAD"
  for model in "${MODELS[@]}"; do
    if [ -n "$model" ]; then
      log "Preloading $model"
      /usr/local/bin/ollama pull "$model"
    fi
  done
  touch "$HEALTH_FILE"
  log "Model preload complete"
  kill "$daemon_pid"
  wait "$daemon_pid" || true
}

main() {
  ensure_data_dir

  if [ -n "$PRELOAD" ] && [ ! -f "$HEALTH_FILE" ]; then
    daemon_pid=$(start_temp_daemon)
    trap 'kill "$daemon_pid" >/dev/null 2>&1 || true' EXIT
    sleep 4
    preload_models "$daemon_pid"
    trap - EXIT
  else
    log "Skipping preload (either disabled or already complete)"
  fi

  log "Starting Ollama server"
  exec /usr/local/bin/ollama serve
}

main "$@"

