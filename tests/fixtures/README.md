# tests/fixtures — fixtures for P0.7

This directory is populated by `scripts/generate-p07-fixtures.py` (task P0.7).

## Synthetic header-only GGUFs (each arch family)
Header-only (no tensor payload) — header + tensor infos padded to 32 bytes, `data_start` past header.

| File | Arch | block_count | n_embd | n_head | ctx | vocab | Weight bytes (header) | License |
|------|------|-------------|--------|--------|-----|-------|-----------------------|---------|
| synthetic-llama.gguf | llama | 32 | 4096 | 32 | 4096 | 32000 | ~ few KB header | MIT (synthetic) |
| synthetic-qwen3.gguf | qwen3 | 28 | 2048 | 16 | 32768 | 151936 | ~ few KB header | MIT (synthetic) |
| synthetic-qwen3moe.gguf | qwen3moe | 48 | 2048 | 16 | 4096 | 151936 | ~ few KB header | MIT (synthetic) |
| synthetic-gemma3.gguf | gemma3 | 34 | 2560 | 32 | 8192 | 262144 | ~ few KB header | MIT (synthetic) |
| synthetic-deepseek2.gguf | deepseek2 | 60 | 2048 | 32 | 4096 | 102400 | ~ few KB header | MIT (synthetic) |
| synthetic-gpt-oss.gguf | gpt-oss | 36 | 2880 | 20 | 8192 | 201088 | ~ few KB header | MIT (synthetic) |
| synthetic-granitehybrid.gguf | granitehybrid | 24 | 2048 | 16 | 8192 | 100352 | ~ few KB header | MIT (synthetic) |

All synthetic files are generated, not derived from upstream weights.

## Real models (≤0.6B + reference large)
| File | Size | Source | License |
|------|------|--------|---------|
| qwen2-0_5b-instruct-q4_0.gguf | 337M | Qwen/Qwen2-0.5B-Instruct-GGUF (TheBloke) | Qwen |
| Qwen3-8B-Q4_K_M.gguf | 4.7G | unsloth/Qwen3-8B-GGUF | Qwen |
| Qwen3-30B-A3B-Q4_K_M.gguf | 11G | unsloth/Qwen3-30B-A3B-GGUF | Qwen |
| gpt-oss-20b-MXFP4.gguf | 11G | ggml-org/gpt-oss-20b-GGUF | Apache-2.0 |
| SmolVLM-256M-Instruct-Q4_K_M.gguf | ~119M | pierretokns/SmolVLM-256M-Instruct-GGUF (from HuggingFaceTB/SmolVLM-256M-Instruct) | Apache-2.0 |
| mmproj-SmolVLM-256M-Instruct-f16.gguf | ~181M | pierretokns/SmolVLM-256M-Instruct-GGUF | Apache-2.0 |
| shapes.png | 2.3K | synthetic | MIT |

## Audio clips (10, 1 sec each, 16kHz mono WAV, sine tones)
`audio/clip-01-220hz.wav` … `audio/clip-10-490hz.wav` — synthetic, MIT. Decodable via `symphonia`/`hound`, resampled via `rubato`.

## Video clips (3, 5 sec placeholder MP4)
`video/clip-01-5s.mp4` … `video/clip-03-5s.mp4` — placeholder ftyp headers (real decode tested via `ffmpeg-sidecar` when ffmpeg is present; add ffmpeg pin to `mise.toml` per K2).

## Recorded API responses
`api/openai-chat-completion.json`, `api/openai-chat-completion-stream.sse`, `api/anthropic-message.json`, `api/anthropic-message-stream.sse` — minimal fixtures for `wiremock` tests (P3.5/P3.6), live smoke behind `RUNA_LIVE=1`.

## Sizes and licenses
Synthetic GGUFs: MIT, header-only, ~2KB each. Audio/video synthetic: MIT. Large models retain upstream licenses (see table). API fixtures: MIT (synthetic).

## Fixture budget: 3 GiB max, downloaded weights are ephemeral

- No single file in this directory may exceed 3 GiB
  (`python3 scripts/check-fixture-size.py`, also enforced in CI;
  override locally with `RUNA_FIXTURE_MAX_BYTES` in bytes).
- `runa pull` refuses fresh downloads over 3 GiB
  (`MAX_MODEL_BYTES` in `crates/runa/src/pull.rs`; override with
  `RUNA_MAX_MODEL_BYTES` in bytes). Already-cached files keep working.
- Downloaded weights (`*.gguf` except `synthetic-*.gguf`) are git-ignored
  and deleted after a full green test run: `moon run root:test-with-cleanup`
  (or `scripts/test-with-fixture-cleanup.sh`, or `python3 scripts/clean-fixtures.py`).
  Cleanup runs only on success so failures keep their weights for debugging;
  set `RUNA_KEEP_FIXTURES=1` to keep them locally and skip re-downloading.
- Hand-made fixtures are never deleted: `synthetic-*.gguf`, `audio/`,
  `video/`, `api/`, `shapes.png`, `mcp-echo.py`.
- Never commit weights: only code, scripts and docs go into git.
