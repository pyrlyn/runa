#!/usr/bin/env bash
# Copyright (c) 2026 Ivan Tugay
# SPDX-License-Identifier: GPL-3.0-or-later
# Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

# P6.1: load-test `runa serve` with 8 concurrent streams (oha or curl fallback).
set -euo pipefail

MODEL="${1:-tests/fixtures/qwen2-0_5b-instruct-q4_0.gguf}"
HOST="${RUNA_SERVE_HOST:-127.0.0.1}"
PORT="${RUNA_SERVE_PORT:-0}"
PARALLEL="${RUNA_SERVE_PARALLEL:-8}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BIN="${CARGO_BIN:-target/debug/runa}"
if [[ ! -x "$BIN" ]]; then
  cargo build -p runa
  BIN=target/debug/runa
fi

"$BIN" serve --mode cpu --host "$HOST" --port "$PORT" --ctx 512 --parallel "$PARALLEL" "$MODEL" &
PID=$!
trap 'kill $PID 2>/dev/null || true' EXIT

BASE=""
for _ in $(seq 1 120); do
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "serve exited early" >&2
    wait "$PID" || true
    exit 1
  fi
  # Ephemeral port: parse from process or use lsof when PORT=0
  if [[ "$PORT" == "0" ]]; then
    PORT_ACTUAL=$(lsof -Pan -p "$PID" -iTCP -sTCP:LISTEN 2>/dev/null | awk '{print $9}' | head -1 | sed 's/.*://')
  else
    PORT_ACTUAL="$PORT"
  fi
  if [[ -n "${PORT_ACTUAL:-}" ]]; then
    BASE="http://${HOST}:${PORT_ACTUAL}"
    if curl -sf "${BASE}/health" >/dev/null 2>&1; then
      break
    fi
  fi
  sleep 1
done

if [[ -z "$BASE" ]]; then
  echo "timed out waiting for serve" >&2
  exit 1
fi

echo "load test against $BASE (${PARALLEL} streams)"

if command -v oha >/dev/null 2>&1; then
  oha -n "$PARALLEL" -c "$PARALLEL" -m POST \
    -H 'content-type: application/json' \
    -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":4}' \
    "${BASE}/v1/chat/completions"
else
  echo "oha not installed; using ${PARALLEL} parallel curl requests" >&2
  BODY='{"messages":[{"role":"user","content":"hi"}],"max_tokens":4}'
  for i in $(seq 1 "$PARALLEL"); do
    curl -sf -H 'content-type: application/json' -d "$BODY" "${BASE}/v1/chat/completions" >/dev/null &
  done
  wait
  echo "ok: ${PARALLEL} requests completed"
fi
