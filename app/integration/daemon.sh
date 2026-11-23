#!/usr/bin/env bash
# Integration test of the daemon (PLAN.md, phase 5): runs the real executable against a real model
# in a home of its own, so it neither sees nor disturbs a daemon the user runs.
#
#   app/integration/daemon.sh <runa executable> <models directory> <model>
set -euo pipefail

runa="$1"
models="$2"
model="$3"

home="$(mktemp -d)"
export RUNA_HOME="$home"
ln -s "$models" "$home/models"

pid_of_daemon() { "$runa" status --json | sed -n 's/.*"pid":\([0-9][0-9]*\).*/\1/p'; }

fail() {
  echo "FAILED: $1" >&2
  echo "--- daemon.log" >&2
  cat "$home/logs/daemon.log" >&2 2>/dev/null || true
  exit 1
}

cleanup() {
  pid="$(pid_of_daemon || true)"
  if [ -n "$pid" ]; then kill -9 "$pid" 2>/dev/null || true; fi
  rm -rf "$home"
}
trap cleanup EXIT

# Waits until the discovery file is gone: the daemon removes it as the last thing it does.
wait_for_stop() {
  for _ in $(seq 50); do
    [ -e "$home/run/daemon.json" ] || return 0
    sleep 0.1
  done
  return 1
}

echo "status never starts the daemon"
"$runa" status | grep -q "not running" || fail "status reports a daemon that was never started"
"$runa" status --json | grep -q '"running":false' || fail "status --json reports a running daemon"
"$runa" daemon status >/dev/null
"$runa" unload --all 2>/dev/null
[ ! -e "$home/run" ] || fail "status or unload left files in $home/run"

echo "daemon up, twice"
"$runa" daemon up >/dev/null
first="$(pid_of_daemon)"
[ -n "$first" ] || fail "daemon up did not start a daemon"
"$runa" daemon up >/dev/null
[ "$(pid_of_daemon)" = "$first" ] || fail "a second daemon up started another daemon"

echo "load, ps, unload"
id="$("$runa" load "$model")"
"$runa" ps --json | grep -qF "\"id\":\"$id\"" || fail "ps --json does not list $id"
"$runa" status --json | grep -q '"loaded_instances":1' || fail "status does not count the loaded model"
"$runa" load "$model" --identifier second -c 1024 >/dev/null
"$runa" ps | grep -q "^second " || fail "ps does not list the second instance"
[ "$("$runa" unload second)" = "second" ] || fail "unload did not name what it unloaded"
"$runa" unload "$id" >/dev/null
[ "$("$runa" ps --json)" = "[]" ] || fail "ps --json lists a model after unload"

echo "a model is unloaded after its ttl"
"$runa" load "$model" --ttl 2 >/dev/null
"$runa" ps --json | grep -q '"ttl_seconds":2' || fail "ps --json does not show the ttl"
sleep 4
[ "$("$runa" ps --json)" = "[]" ] || fail "the model outlived its 2 s ttl"

echo "a killed daemon is replaced"
kill -9 "$first"
[ -e "$home/run/daemon.json" ] || fail "a killed daemon cannot have removed its discovery file"
"$runa" status | grep -q "not running" || fail "status takes the killed daemon for a running one"
[ ! -e "$home/run/daemon.json" ] || fail "the stale discovery file was not removed"
"$runa" ps >/dev/null 2>&1
second="$(pid_of_daemon)"
[ -n "$second" ] && [ "$second" != "$first" ] || fail "no new daemon was started after the kill"

echo "SIGTERM stops it in order"
kill -TERM "$second"
wait_for_stop || fail "the daemon did not stop on SIGTERM"
grep -q "Stopped" "$home/logs/daemon.log" || fail "the daemon did not log its stop"

echo "daemon down"
"$runa" load "$model" >/dev/null 2>&1
third="$(pid_of_daemon)"
"$runa" daemon down
[ ! -e "$home/run/daemon.json" ] || fail "daemon down left the discovery file"
if kill -0 "$third" 2>/dev/null; then fail "the daemon process outlived daemon down"; fi
"$runa" daemon down 2>&1 | grep -q "not running" || fail "a second daemon down is not a no-op"

echo "ok"
