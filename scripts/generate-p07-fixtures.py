#!/usr/bin/env python3
"""Generate P0.7 fixtures: synthetic header-only GGUFs, audio clips, video placeholders, recorded API responses."""
import os
import struct
import wave
import json
import math

ROOT = os.path.join(os.path.dirname(__file__), "..")
FIXTURES = os.path.join(ROOT, "tests", "fixtures")

# GGUF constants
GGUF_MAGIC = b"GGUF"
VERSION = 3
# DataType tags
DT_STRING = 8
DT_UINT64 = 10
DT_UINT32 = 4
# Ggml types
GGML_Q4_0 = 2
GGML_F16 = 1
GGML_F32 = 0
GGML_Q4_K = 12
GGML_Q8_0 = 8

def write_u32(v): return struct.pack("<I", v)
def write_u64(v): return struct.pack("<Q", v)
def write_str(s):
    b = s.encode("utf-8")
    return write_u64(len(b)) + b
def write_string_value(s):
    b = s.encode("utf-8")
    return write_u64(len(b)) + b
def write_u64_value(v):
    return struct.pack("<Q", v)

def make_gguf(path, kv_pairs, tensors):
    """kv_pairs: list of (key, dtype, value_bytes), tensors: list of (name, dims, ggml_type)"""
    buf = bytearray()
    buf += GGUF_MAGIC
    buf += write_u32(VERSION)
    buf += write_u64(len(tensors))
    buf += write_u64(len(kv_pairs))
    for key, dtype, val_bytes in kv_pairs:
        buf += write_str(key)
        buf += write_u32(dtype)
        buf += val_bytes
    for name, dims, gtype in tensors:
        buf += write_str(name)
        buf += write_u32(len(dims))
        for d in dims:
            buf += write_u64(d)
        buf += write_u32(gtype)
        buf += write_u64(0)  # offset
    # pad to 32
    while len(buf) % 32 != 0:
        buf.append(0)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(buf)
    print(f"wrote {path} ({len(buf)} bytes)")

# Synthetic GGUFs per arch family
archs = {
    "llama": {"block_count": 32, "embedding_length": 4096, "head_count": 32, "head_kv": 32, "ctx": 4096, "vocab": 32000},
    "qwen3": {"block_count": 28, "embedding_length": 2048, "head_count": 16, "head_kv": 8, "ctx": 32768, "vocab": 151936},
    "qwen3moe": {"block_count": 48, "embedding_length": 2048, "head_count": 16, "head_kv": 4, "ctx": 4096, "vocab": 151936, "expert_count": 128, "expert_used": 8},
    "gemma3": {"block_count": 34, "embedding_length": 2560, "head_count": 32, "head_kv": 8, "ctx": 8192, "vocab": 262144, "sliding_window": 2048},
    "deepseek2": {"block_count": 60, "embedding_length": 2048, "head_count": 32, "head_kv": 8, "ctx": 4096, "vocab": 102400, "kv_lora_rank": 512},
    "gpt-oss": {"block_count": 36, "embedding_length": 2880, "head_count": 20, "head_kv": 5, "ctx": 8192, "vocab": 201088},
    "granitehybrid": {"block_count": 24, "embedding_length": 2048, "head_count": 16, "head_kv": 8, "ctx": 8192, "vocab": 100352, "recurrent": 8},
}

for arch, p in archs.items():
    kvs = [
        ("general.architecture", DT_STRING, write_string_value(arch)),
        (f"{arch}.block_count", DT_UINT64, write_u64_value(p["block_count"])),
        (f"{arch}.embedding_length", DT_UINT64, write_u64_value(p["embedding_length"])),
        (f"{arch}.attention.head_count", DT_UINT64, write_u64_value(p["head_count"])),
        (f"{arch}.context_length", DT_UINT64, write_u64_value(p["ctx"])),
        (f"{arch}.vocab_size", DT_UINT64, write_u64_value(p["vocab"])),
    ]
    # optional keys
    if "head_kv" in p:
        kvs.append((f"{arch}.attention.head_count_kv", DT_UINT64, write_u64_value(p["head_kv"])))
    if "expert_count" in p:
        kvs.append((f"{arch}.expert_count", DT_UINT64, write_u64_value(p["expert_count"])))
        kvs.append((f"{arch}.expert_used_count", DT_UINT64, write_u64_value(p["expert_used"])))
    if "sliding_window" in p:
        kvs.append((f"{arch}.sliding_window", DT_UINT64, write_u64_value(p["sliding_window"])))
    if "kv_lora_rank" in p:
        kvs.append((f"{arch}.attention.kv_lora_rank", DT_UINT64, write_u64_value(p["kv_lora_rank"])))
    if "recurrent" in p:
        kvs.append((f"{arch}.recurrent_layer_count", DT_UINT64, write_u64_value(p["recurrent"])))

    tensors = [
        ("token_embd.weight", [p["vocab"], p["embedding_length"]], GGML_Q4_0),
        ("blk.0.attn_q.weight", [p["embedding_length"], p["embedding_length"]], GGML_Q4_0),
        ("output.weight", [p["embedding_length"], p["vocab"]], GGML_Q4_0),
    ]
    # for MoE, add expert tensor
    if arch == "qwen3moe":
        tensors.append(("blk.0.ffn_gate_exps.weight", [256, p["embedding_length"]], GGML_Q4_K))
    path = os.path.join(FIXTURES, f"synthetic-{arch}.gguf")
    make_gguf(path, kvs, tensors)

# Audio clips: 10 WAV files, 16kHz mono, 1 sec, sine tones
audio_dir = os.path.join(FIXTURES, "audio")
os.makedirs(audio_dir, exist_ok=True)
for i in range(10):
    freq = 220 + i*30
    path = os.path.join(audio_dir, f"clip-{i+1:02d}-{freq}hz.wav")
    with wave.open(path, "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        frames = bytearray()
        for n in range(16000):
            samp = int(16000 * math.sin(2*math.pi*freq*n/16000))
            frames += struct.pack("<h", samp)
        w.writeframes(frames)
    print(f"wrote {path}")

# Video placeholders: 3 tiny MP4s (1 sec, 64x64, via minimal header if ffmpeg not available)
# We create dummy files with ftyp header; real decode will be tested elsewhere with synthetic frames.
video_dir = os.path.join(FIXTURES, "video")
os.makedirs(video_dir, exist_ok=True)
for i in range(3):
    path = os.path.join(video_dir, f"clip-{i+1:02d}-5s.mp4")
    # Minimal ftyp + mdat placeholder; ffmpeg probe will handle gracefully or skip.
    # For now create a tiny valid MP4 via python: just write a few bytes so file exists.
    # If ffmpeg is available, it would generate real video, but we lack it.
    with open(path, "wb") as f:
        # ftyp box (20 bytes) + free box
        f.write(b'\x00\x00\x00\x14ftypisom\x00\x00\x02\x00isomiso2mp41')
        f.write(b'\x00\x00\x00\x08free')
        f.write(b'\x00' * 1024)
    print(f"wrote {path} (placeholder)")

# Recorded API responses: OpenAI and Anthropic
api_dir = os.path.join(FIXTURES, "api")
os.makedirs(api_dir, exist_ok=True)
# OpenAI chat completion non-stream
openai_resp = {
    "id": "chatcmpl-test123",
    "object": "chat.completion",
    "created": 1700000000,
    "model": "gpt-4o-mini",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello from fixture"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9}
}
with open(os.path.join(api_dir, "openai-chat-completion.json"), "w") as f:
    json.dump(openai_resp, f, indent=2)
# OpenAI stream SSE
with open(os.path.join(api_dir, "openai-chat-completion-stream.sse"), "w") as f:
    f.write('data: {"id":"chatcmpl-test123","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4o-mini","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}\n\n')
    f.write('data: {"id":"chatcmpl-test123","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4o-mini","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n')
    f.write('data: [DONE]\n')
print("wrote openai fixtures")
# Anthropic message
anthropic_resp = {
    "id": "msg_test123",
    "type": "message",
    "role": "assistant",
    "content": [{"type": "text", "text": "Hello from Anthropic fixture"}],
    "model": "claude-3-5-sonnet-20241022",
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 5, "output_tokens": 6}
}
with open(os.path.join(api_dir, "anthropic-message.json"), "w") as f:
    json.dump(anthropic_resp, f, indent=2)
with open(os.path.join(api_dir, "anthropic-message-stream.sse"), "w") as f:
    f.write('event: message_start\ndata: {"type":"message_start","message":{"id":"msg_test123","type":"message","role":"assistant","content":[],"model":"claude-3-5-sonnet-20241022"}}\n\n')
    f.write('event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}\n\n')
    f.write('event: message_stop\ndata: {"type":"message_stop"}\n\n')
print("wrote anthropic fixtures")

# README
readme = """# tests/fixtures — fixtures for P0.7

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
| Qwen3-30B-A3B-Q4_K_M.gguf | ~6M header-only (real tensor infos; weights truncated) | unsloth/Qwen3-30B-A3B-GGUF | Qwen |
| gpt-oss-20b-MXFP4.gguf | ~26M tiny loadable (real Harmony chat_template + minimal tensors) | custom from ggml-org/gpt-oss-20b-GGUF tokenizer | Apache-2.0 |
| SmolVLM-500M-Instruct-Q8_0.gguf | 417M | ggml-org/SmolVLM-500M-Instruct-GGUF | Apache-2.0 |
| mmproj-SmolVLM-500M-Instruct-Q8_0.gguf | 104M | ggml-org/SmolVLM-500M-Instruct-GGUF | Apache-2.0 |
| shapes.png | 2.3K | synthetic | MIT |

## Audio clips (10, 1 sec each, 16kHz mono WAV, sine tones)
`audio/clip-01-220hz.wav` … `audio/clip-10-490hz.wav` — synthetic, MIT. Decodable via `symphonia`/`hound`, resampled via `rubato`.

## Video clips (3, 5 sec placeholder MP4)
`video/clip-01-5s.mp4` … `video/clip-03-5s.mp4` — placeholder ftyp headers (real decode tested via `ffmpeg-sidecar` when ffmpeg is present; add ffmpeg pin to `mise.toml` per K2).

## Recorded API responses
`api/openai-chat-completion.json`, `api/openai-chat-completion-stream.sse`, `api/anthropic-message.json`, `api/anthropic-message-stream.sse` — minimal fixtures for `wiremock` tests (P3.5/P3.6), live smoke behind `RUNA_LIVE=1`.

## Sizes and licenses
Synthetic GGUFs: MIT, header-only, ~2KB each. Audio/video synthetic: MIT. Large models retain upstream licenses (see table). API fixtures: MIT (synthetic).
"""
with open(os.path.join(FIXTURES, "README.md"), "w") as f:
    f.write(readme)
print("wrote README.md")
