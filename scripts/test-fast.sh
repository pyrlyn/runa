#!/usr/bin/env bash
# Copyright (c) 2026 Ivan Tugay
# SPDX-License-Identifier: GPL-3.0-or-later
# Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

# Fast tests for local development and PR checks (P14.3).
#
# Library unit tests, plus integration tests that stay deterministic
# without a weight download or a GPU. This does not replace
# `moon run :test` (per-crate `cargo test`, integration tests included)
# or `moon run root:test-with-cleanup` (full `cargo test --workspace`,
# then fixture cleanup).
#
# Left on the full gate — they load a model or require the git-ignored
# qwen2 GGUF (~340MB):
#   runa --test e2e, bench, pull
#   runa-engine --test generate, generate_qwen2, load, vision_smolvlm, rpc
#   runa-fit --test remote
# CLI harnesses that link the full binary without loading weights
# (doctor, trycmd, media, secrets, security, cloud) stay there too.
# `cargo test -p runa --bin runa` stays there as well: the package has
# no library target, and `second_daemon_does_not_unlink_a_live_socket`
# identifies the socket by inode, which overlayfs can reuse.
set -euo pipefail

cd "$(dirname "$0")/.."

cargo test --workspace --lib
cargo test -p runa-fit --test gguf --test fuzz_regressions
cargo test -p runa-cloud --test openai
cargo test -p runa-pool --test embed
