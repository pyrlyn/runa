#!/usr/bin/env bash
# Smoke test of the OpenAI-compatible server with the official `openai` Python client (PLAN.md,
# phase 6): runs the example of docs/openai.md against a real server.
#
#   app/integration/openai.sh <runa executable> <models directory> <model>
#
# Installs the client from PyPI into a virtual environment that is removed afterwards.
# RUNA_TEST_PORT names the port to serve on when 17863 is taken.
set -euo pipefail

# shellcheck source=app/integration/lib.sh
source "$(dirname "$0")/lib.sh"
integration_setup "$1" "$2"
model="$3"
port="${RUNA_TEST_PORT:-17863}"
examples="$(dirname "$0")/../../docs/examples"

echo "the client"
python3 -m venv "$home/venv"
"$home/venv/bin/pip" install --quiet --disable-pip-version-check -r "$examples/requirements.txt"

echo "the example"
"$runa" server start -p "$port" >/dev/null
RUNA_BASE_URL="http://127.0.0.1:$port/v1" "$home/venv/bin/python" "$examples/openai_client.py" "$model" ||
  fail "the example of docs/openai.md failed"

echo "ok"
