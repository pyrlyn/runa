#!/usr/bin/env bash
# Integration test of `runa chat` and of the conversations behind it (PLAN.md, phase 6).
#
#   app/integration/chat.sh <runa executable> <models directory> <model>
set -euo pipefail

# shellcheck source=app/integration/lib.sh
source "$(dirname "$0")/lib.sh"
integration_setup "$1" "$2"
model="$3"

field() { sed -n "s/.*\"$1\":\"\{0,1\}\([^\",}]*\).*/\1/p"; }

echo "one question in a pipe: the answer and nothing else"
answer="$("$runa" chat "$model" -p "Say hello." 2>"$home/notes.txt" | cat)"
[ -n "$answer" ] || fail "chat -p printed nothing"
if grep -q "$(printf '\033')" <<<"$answer"; then fail "chat -p wrote escape sequences into a pipe: $answer"; fi
grep -q "^Loaded " "$home/notes.txt" || fail "chat did not load the model: $(cat "$home/notes.txt")"
if grep -q "$(printf '\033')" "$home/notes.txt"; then fail "chat drew a progress bar without a terminal"; fi

echo "the question on standard input, for the model that is loaded"
answer="$(echo "What is 2+2? Answer with one number." | "$runa" chat --stats 2>"$home/notes.txt")"
[ -n "$answer" ] || fail "chat printed nothing for a question on standard input"
grep -q "^prompt: [0-9]* tokens, answer: [0-9]* tokens" "$home/notes.txt" || fail "no statistics: $(cat "$home/notes.txt")"
if grep -q "^Loaded " "$home/notes.txt"; then fail "chat loaded a model that was loaded"; fi
[ "$("$runa" ps --json | grep -o '"id":' | wc -l | tr -d ' ')" = 1 ] || fail "not exactly one model is loaded"

echo "one question, answered as JSON"
answer="$("$runa" chat "$model" -p "Say OK." --json)"
grep -q '"output":\[{"type":"message","content":".*"stop_reason":"' <<<"$answer" || fail "not a chat response: $answer"

echo "a mistake in the name of the model"
if "$runa" chat no-such-model -p "Hi" >/dev/null 2>"$home/notes.txt"; then fail "chat took a model nobody has"; fi
grep -q "No model matches 'no-such-model'" "$home/notes.txt" || fail "no reason given: $(cat "$home/notes.txt")"

# The conversation itself needs a terminal in `runa chat`, so here the API is asked what the command asks it.
port="$(field port <"$home/run/daemon.json")"
token="$(field token <"$home/run/daemon.json")"
api() { curl -sS --max-time 120 -H "Authorization: Bearer $token" -H 'Content-Type: application/json' "http://127.0.0.1:$port/api/v1/chat" "$@"; }
turn() { printf '{"model":"%s","input":"%s","max_output_tokens":8%s}' "$model" "$1" "${2:-}"; }

echo "a second turn continues the first"
first="$(api -d "$(turn "The secret number is 42. Reply with OK.")")"
id="$(field response_id <<<"$first")"
[ -n "$id" ] || fail "the first turn has no response id: $first"
second="$(api -d "$(turn "Which number was it?" ",\"previous_response_id\":\"$id\"")")"
[ "$(field input_tokens <<<"$second")" -gt "$(field input_tokens <<<"$first")" ] ||
  fail "the second turn did not see the first: $second"

echo "a turn that was never stored cannot be continued"
[ "$(api -o /dev/null -w '%{http_code}' -d "$(turn "Hi" ',"previous_response_id":"resp_none"')")" = 400 ] ||
  fail "an unknown response id is not a 400"

echo "a streamed turn"
stream="$(api -N -d "$(turn "Count to three." ',"stream":true')")"
[ "$(head -1 <<<"$stream")" = "event: chat.start" ] || fail "the stream does not open with chat.start: $stream"
grep -q '^event: message.delta$' <<<"$stream" || fail "the stream has no text"
grep -q '^data: {"type":"chat.end","result":{"model_instance_id":' <<<"$stream" || fail "the stream does not end with the result"

echo "ok"
