#!/usr/bin/env bash
# Copyright (c) 2026 Ivan Tugay
# SPDX-License-Identifier: GPL-3.0-or-later
# Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

# Run the test suite, then clean downloaded fixtures only on a full pass.
#
#   scripts/test-with-fixture-cleanup.sh [-- <cargo test args...>]
#
# Cleanup removes git-ignored weights from tests/fixtures (keeps hand-made
# files: synthetic-*.gguf, audio/, video/, api/). Set RUNA_KEEP_FIXTURES=1
# (or KEEP=1) to skip cleanup locally and avoid re-downloading weights.
#
# A pass also runs dunnage (`ketch install dunnage`) over ./target: lossless
# compression and dedupe, never deletes; skipped when dunnage is missing.
set -euo pipefail

cd "$(dirname "$0")/.."

cargo test "$@"
status=$?

if [ "$status" -ne 0 ]; then
  echo "tests failed (exit $status); keeping fixtures for debugging" >&2
  exit "$status"
fi

if ! command -v dunnage >/dev/null; then
  echo "dunnage not found; install it with: ketch install dunnage" >&2
elif [ -d target ]; then
  echo "tests passed; compacting target/ with dunnage"
  dunnage run target || [ $? -eq 2 ]
fi

if [ "${RUNA_KEEP_FIXTURES:-${KEEP:-0}}" = "1" ]; then
  echo "tests passed; RUNA_KEEP_FIXTURES=1, keeping fixtures"
  exit 0
fi

echo "tests passed; cleaning downloaded fixtures"
python3 scripts/clean-fixtures.py
