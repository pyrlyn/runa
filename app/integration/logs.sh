#!/usr/bin/env bash
# Integration test of `runa log stream` and of the daemon's log file (PLAN.md, phase 7).
#
#   app/integration/logs.sh <runa executable> <models directory> <model>
set -euo pipefail

# shellcheck source=app/integration/lib.sh
source "$(dirname "$0")/lib.sh"
integration_setup "$1" "$2"
model="$3"
log="$home/logs/daemon.log"
question="Which colour is the sky?"

# Waits until the file $2 has a line that matches $1.
wait_for() {
  for _ in $(seq 100); do
    if grep -q "$1" "$2" 2>/dev/null; then return 0; fi
    sleep 0.1
  done
  return 1
}

# Waits until the process $1 is gone.
wait_for_exit() {
  for _ in $(seq 50); do
    kill -0 "$1" 2>/dev/null || return 0
    sleep 0.1
  done
  return 1
}

# Starts `log stream` with the given flags; its events go to $home/$1.out and what it says to $home/$1.err.
listen() {
  name="$1"
  shift
  "$runa" log stream "$@" >"$home/$name.out" 2>"$home/$name.err" &
  wait_for "^Streaming the " "$home/$name.err" || fail "log stream $* never listened: $(cat "$home/$name.err")"
}

echo "a chat, as whoever listens to the models sees it"
listen model --json
listener=$!
grep -q "^Starting the runa daemon$" "$home/model.err" || fail "log stream did not start the daemon"
"$runa" chat "$model" -p "$question" >/dev/null 2>&1 || fail "chat failed"
wait_for '"type":"output"' "$home/model.out" || fail "no output event: $(cat "$home/model.out")"
[ "$(wc -l <"$home/model.out" | tr -d ' ')" = 2 ] || fail "not one input and one output: $(cat "$home/model.out")"
sed -n 1p "$home/model.out" | grep -q "\"source\":\"model\",\"type\":\"input\",\"text\":\".*$question.*\"endpoint\":\"chat\"}\$" ||
  fail "the first event is not what the model read: $(cat "$home/model.out")"
sed -n 2p "$home/model.out" | grep -q '"source":"model","type":"output","text":"..*","model":"..*","endpoint":"chat"}$' ||
  fail "the second event is not what the model wrote: $(cat "$home/model.out")"

echo "the log file has the generation and not the conversation"
grep -q " INFO  chat '.*': [0-9]* prompt tokens, [0-9]* generated" "$log" || fail "the log file has no line for the generation"
if grep -q "colour" "$log"; then fail "the log file has the prompt, and nobody asked for that"; fi
"$runa" daemon status | grep -q "^Log: .*logs/daemon.log$" || fail "daemon status does not say where the log is"

echo "the server's log and the engine's"
listen server --source server
listen runtime --source runtime --json
"$runa" unload --all >/dev/null 2>&1
"$runa" load "$model" >/dev/null 2>&1 || fail "load failed"
wait_for " DEBUG POST /api/v1/models/load 200 [0-9]* ms$" "$home/server.out" || fail "no request in the server's log: $(cat "$home/server.out")"
grep -q " INFO  Loaded " "$home/server.out" || fail "the server's log does not say that the model was loaded"
wait_for '"source":"runtime","type":"log","text":".*","level":"' "$home/runtime.out" || fail "the engine logged nothing while it loaded a model"
if grep -q '"source":"server"' "$home/runtime.out"; then fail "the engine's log has the server's lines"; fi

echo "the streams end when the daemon stops"
"$runa" daemon down >/dev/null 2>&1
wait_for_stop || fail "the daemon did not stop"
wait_for_exit "$listener" || fail "log stream outlived the daemon"
grep -q "^The daemon ended the stream$" "$home/model.err" || fail "log stream did not say why it ended: $(cat "$home/model.err")"

echo "a full log file is put aside, and --log-prompts writes the conversation into the new one"
# The daemon counts what its log file holds already, so a file of the full 10 MB is put aside with the first line.
head -c 10485760 /dev/zero | tr '\0' '\n' >"$log"
"$runa" --log-prompts daemon up >/dev/null 2>&1 || fail "the daemon did not start"
"$runa" chat "$model" -p "$question" >/dev/null 2>&1 || fail "chat failed"
[ -e "$log.1" ] || fail "the full log file was not put aside"
[ "$(wc -c <"$log.1" | tr -d ' ')" -ge 10485760 ] || fail "the file that was put aside is not the full one"
[ "$(wc -c <"$log" | tr -d ' ')" -lt 1000000 ] || fail "the log did not go on in a new file"
grep -q " INFO  chat '.*' input: \".*$question" "$log" || fail "the log file does not have what the model read"
grep -q " INFO  chat '.*' output: \"" "$log" || fail "the log file does not have what the model wrote"

echo "ok"
