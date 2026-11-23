#!/usr/bin/env bash
# Integration test of the OpenAI-compatible server (PLAN.md, phase 6), with curl as the client.
#
#   app/integration/server.sh <runa executable> <models directory> <model>
#
# RUNA_TEST_PORT names the port to serve on when 17862 is taken.
set -euo pipefail

# shellcheck source=app/integration/lib.sh
source "$(dirname "$0")/lib.sh"
integration_setup "$1" "$2"
model="$3"
port="${RUNA_TEST_PORT:-17862}"
base="http://127.0.0.1:$port"

api() { curl -sS --max-time 120 -H 'Content-Type: application/json' "$@"; }
status_of() { api -o /dev/null -w '%{http_code}' "$@"; }
# Whether a command's output has a line that matches. Not `command | grep -q`: grep would leave at the
# first match, and a command that is still writing would die of it and fail the pipeline.
says() {
  local pattern="$1"
  shift
  grep -q -- "$pattern" <<<"$("$@" 2>&1)"
}
chat() { printf '{"model":"%s","messages":[{"role":"user","content":"%s"}],"max_tokens":%s%s}' "$model" "$1" "$2" "${3:-}"; }

echo "server status and stop never start the daemon"
says "not running" "$runa" server status || fail "server status reports a server that was never started"
"$runa" server stop 2>/dev/null
[ ! -e "$home/run" ] || fail "server status or stop left files in $home/run"

echo "server start, twice"
"$runa" server start -p "$port" >/dev/null
says "runs already" "$runa" server start -p "$port" || fail "a second server start did not see the first"
says "\"port\":$port" "$runa" server status --json || fail "server status --json does not show the port"
says "Server: $base" "$runa" status || fail "status does not show the server"
daemon="$(pid_of_daemon)"

echo "models"
grep -q '"object":"list"' <<<"$(api "$base/v1/models")" || fail "/v1/models is not a list"

echo "a chat completion loads the model"
answer="$(api "$base/v1/chat/completions" -d "$(chat "Say hello." 8)")"
grep -q '"object":"chat.completion".*"role":"assistant".*"finish_reason":"' <<<"$answer" || fail "not a chat completion: $answer"
says '"id":' "$runa" ps --json || fail "the request did not load the model"

echo "a streamed chat completion"
stream="$(api -N "$base/v1/chat/completions" -d "$(chat "Count to three." 8 ',"stream":true,"stream_options":{"include_usage":true}')")"
head -1 <<<"$stream" | grep -q '^data: {"id":"chatcmpl-.*"delta":{"role":"assistant","content":""},"finish_reason":null' ||
  fail "the stream does not open with the assistant role: $stream"
grep -q '"choices":\[\],"usage":{"prompt_tokens":' <<<"$stream" || fail "the stream has no usage chunk"
[ "$(grep -v '^$' <<<"$stream" | tail -1)" = "data: [DONE]" ] || fail "the stream does not end with [DONE]"

echo "a text completion"
answer="$(api "$base/v1/completions" -d "$(printf '{"model":"%s","prompt":"The capital of France is","max_tokens":4}' "$model")")"
grep -q '"object":"text_completion".*"text":"' <<<"$answer" || fail "not a text completion: $answer"

echo "a tool call"
tools=',"temperature":0,"tool_choice":"required","tools":[{"type":"function","function":{"name":"get_weather","description":"The weather in a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]'
answer="$(api "$base/v1/chat/completions" -d "$(chat "What is the weather in Paris?" 64 "$tools")")"
grep -q '"content":null,"tool_calls":\[{"id":"call_[^"]*","type":"function","function":{"name":"get_weather","arguments":"{.*city.*}"}}\]},"finish_reason":"tool_calls"' <<<"$answer" ||
  fail "not a tool call: $answer"

echo "a streamed tool call"
stream="$(api -N "$base/v1/chat/completions" -d "$(chat "What is the weather in Paris?" 64 "$tools,\"stream\":true")")"
grep -q '"delta":{"tool_calls":\[{"index":0,"id":"call_[^"]*","type":"function","function":{"name":"get_weather","arguments":""}}\]}' <<<"$stream" ||
  fail "the stream does not announce the call: $stream"
grep -q '"delta":{"tool_calls":\[{"index":0,"function":{"arguments":"[^"]' <<<"$stream" || fail "the stream has no arguments: $stream"
grep -q '"delta":{},"finish_reason":"tool_calls"' <<<"$stream" || fail "the stream does not end with tool_calls: $stream"

echo "an answer of a JSON schema"
schema=',"temperature":0,"response_format":{"type":"json_schema","json_schema":{"name":"city","schema":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"],"additionalProperties":false}}}'
answer="$(api "$base/v1/chat/completions" -d "$(chat "Name a city." 48 "$schema")")"
grep -q '"content":"{\\"city\\": *\\"[^"]*\\"}"},"finish_reason":"stop"' <<<"$answer" || fail "not an answer of the schema: $answer"

echo "mistakes get the status codes of OpenAI"
[ "$(status_of "$base/v1/chat/completions" -d '{"model":"no-such-model","messages":[{"role":"user","content":"Hi"}]}')" = 404 ] ||
  fail "an unknown model is not a 404"
[ "$(status_of "$base/v1/embeddings" -d "$(printf '{"model":"%s","input":"Hi"}' "$model")")" = 400 ] ||
  fail "embeddings from a chat model are not a 400"
[ "$(status_of "$base/api/v1/models")" = 401 ] || fail "the native API is open on the public port"

echo "a client that goes away frees the model"
api -N --max-time 1 -o /dev/null "$base/v1/chat/completions" \
  -d "$(chat "Write a very long story about the sea." 4000 ',"stream":true')" 2>/dev/null || true
says '"finish_reason"' api --max-time 20 "$base/v1/chat/completions" -d "$(chat "Hi" 4)" ||
  fail "the model stayed busy with a client that left"

echo "server stop leaves the daemon and its models"
says "Stopped the server on $base" "$runa" server stop || fail "server stop did not say what it stopped"
if api -o /dev/null "$base/v1/models" 2>/dev/null; then fail "the port is still open after server stop"; fi
says '"id":' "$runa" ps --json || fail "server stop unloaded the model"

echo "a port that is taken is the user's mistake"
taken="$("$runa" status --json | sed -n 's/.*"address":"[^"]*:\([0-9][0-9]*\)".*/\1/p')"
if "$runa" server start -p "$taken" >/dev/null 2>"$home/taken.txt"; then fail "server start took the daemon's own port"; fi
grep -q "Cannot listen on 127.0.0.1:$taken" "$home/taken.txt" || fail "no reason given: $(cat "$home/taken.txt")"
[ "$(pid_of_daemon)" = "$daemon" ] || fail "the daemon did not survive a port that was taken"

echo "an API key"
says "API key required" env RUNA_API_KEY=integration-key "$runa" server start -p "$port" || fail "server start ignored RUNA_API_KEY"
[ "$(status_of "$base/v1/models")" = 401 ] || fail "/v1/models is open without the key"
[ "$(status_of -H 'Authorization: Bearer integration-key' "$base/v1/models")" = 200 ] || fail "the key does not open /v1/models"
[ "$(status_of -H 'Authorization: Bearer integration-key' "$base/api/v1/models")" = 401 ] || fail "the key opens the native API"

echo "SIGTERM stops the daemon and its server"
kill -TERM "$daemon"
wait_for_stop || fail "the daemon did not stop on SIGTERM"
for _ in $(seq 50); do
  kill -0 "$daemon" 2>/dev/null || break
  sleep 0.1
done
if kill -0 "$daemon" 2>/dev/null; then fail "the daemon process outlived SIGTERM"; fi
if api -o /dev/null "$base/v1/models" 2>/dev/null; then fail "the port is still open after the daemon stopped"; fi

echo "ok"
