# Audio, video and vision

Three routes, never mixed silently (D12):

| Route | Local | OpenAI | Anthropic |
|-------|-------|--------|-----------|
| **native** | mtmd + audio/vision mmproj | `input_audio` on audio-capable models | not available |
| **asr** | whisper.cpp (`whisper-rs`) → text in the prompt | transcript in the prompt | transcript in the prompt |
| **auto** | native if mmproj (audio) / VL mmproj (vision); else ASR | `input_audio` if capable, else transcript | transcript |

## Audio (`--audio`, `--audio-route`)

Preference: CLI `--audio-route` > `RUNA_AUDIO_ROUTE` > `[audio] route` > `auto`.

- **native** without an audio mmproj (or on Anthropic / non-audio OpenAI) errors.
- **asr** always transcribes (`runa media transcribe` uses the same engine).
- PCM is 16 kHz mono f32 (`runa-media`). OpenAI native wraps WAV base64.

Sibling `*mmproj*.gguf` is picked only when native audio or `--image`/`--video`
needs it. `--mmproj` wins if set.

## Vision (`--image`, `--video`)

Local only, `mtmd` feature + vision mmproj:

- `--image PATH` (repeatable) — stills, media markers, no timestamps.
- `--video PATH` — `runa-media` samples ≤ 32 frames; each is an mtmd bitmap
  with a `[t=12.0s]` marker in the prompt.

Cloud `--image`/`--video` on `runa run openai:…` is an explicit error (use a
local VL model, or the P4.7 adapters from `serve`).

## Video sampling

Uniform + scene-change (histogram L1), cap 32, resize. ffmpeg on PATH or
`ffmpeg-sidecar`. See P4.5.

## Fit

Frames × tokens-per-frame + audio seconds count as context; mmproj bytes sit
on GPU. Overflow → `NO FIT` exit 2 (`docs/fit.md`).
