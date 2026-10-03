#!/usr/bin/env python3
"""Regenerate the fuzz seed corpora in fuzz/seeds/ (small, committed).

Run from anywhere: `python3 fuzz/seeds/make_seeds.py`.
"""
import json
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def write(target, name, data):
    d = ROOT / target
    d.mkdir(parents=True, exist_ok=True)
    if isinstance(data, str):
        data = data.encode()
    (d / name).write_bytes(data)


# --- gguf-header: minimal GGUF v3 headers (llama dense + MoE) ----------
def gstr(s):
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def kv(key, ty, payload):
    return gstr(key) + struct.pack("<I", ty) + payload


U32, F32, BOOL, STR, ARR, U64 = 4, 6, 7, 8, 9, 10


def gguf(arch, extra, tensors):
    kvs = [
        kv("general.architecture", STR, gstr(arch)),
        kv("general.name", STR, gstr("seed")),
        kv(f"{arch}.block_count", U32, struct.pack("<I", 2)),
        kv(f"{arch}.embedding_length", U32, struct.pack("<I", 64)),
        kv(f"{arch}.attention.head_count", U32, struct.pack("<I", 4)),
        kv(f"{arch}.attention.head_count_kv", U32, struct.pack("<I", 2)),
        kv(f"{arch}.context_length", U64, struct.pack("<Q", 4096)),
        kv(f"{arch}.rope.freq_base", F32, struct.pack("<f", 10000.0)),
        kv("tokenizer.ggml.add_bos_token", BOOL, b"\x01"),
        kv("tokenizer.ggml.tokens", ARR, struct.pack("<IQ", STR, 2) + gstr("a") + gstr("b")),
    ] + extra
    body = b"".join(kvs)
    for name, dims, ty in tensors:
        body += gstr(name) + struct.pack("<I", len(dims))
        body += b"".join(struct.pack("<Q", d) for d in dims)
        body += struct.pack("<IQ", ty, 0)
    return b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kvs)) + body


dense = [("token_embd.weight", [64, 2], 1), ("blk.0.attn_q.weight", [64, 64], 8)]
write("gguf-header", "llama-dense.gguf", gguf("llama", [], dense))
moe = [
    kv("qwen3moe.expert_count", U32, struct.pack("<I", 8)),
    kv("qwen3moe.expert_used_count", U32, struct.pack("<I", 2)),
]
moe_t = dense + [("blk.0.ffn_gate_exps.weight", [64, 128, 8], 12)]
write("gguf-header", "qwen3moe.gguf", gguf("qwen3moe", moe, moe_t))

# --- reasoning-stream: arbitrary-encoded, so seed with raw tag text ------
for i, s in enumerate([
    "<think>plan</think>answer",
    "<|channel|>analysis<|message|>hm<|end|><|channel|>final<|message|>ok",
    "<start_of_thought>x<end_of_thought>y",
    "<|channel|>thought t <|channel|>response r",
]):
    write("reasoning-stream", f"tags-{i}", s)

# --- cloud-responses -----------------------------------------------------
write("cloud-responses", "anthropic-sse.txt", "\n".join([
    "event: message_start",
    'data: {"type":"message_start","message":{"usage":{"input_tokens":3}}}',
    "",
    "event: content_block_start",
    'data: {"type":"content_block_start","index":0,'
    '"content_block":{"type":"tool_use","id":"t1","name":"f","input":{}}}',
    "",
    "event: content_block_delta",
    'data: {"type":"content_block_delta","index":0,'
    '"delta":{"type":"input_json_delta","partial_json":"{\\"a\\":1}"}}',
    "",
    "event: content_block_stop",
    'data: {"type":"content_block_stop","index":0}',
    "",
    "event: message_delta",
    'data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},'
    '"usage":{"output_tokens":5}}',
    "",
    "event: message_stop",
    'data: {"type":"message_stop"}',
    "",
]))
write("cloud-responses", "anthropic-message.json", json.dumps({
    "content": [
        {"type": "thinking", "thinking": "hm", "signature": "s"},
        {"type": "text", "text": "hi"},
        {"type": "tool_use", "id": "t", "name": "f", "input": {"a": 1}},
    ],
    "stop_reason": "tool_use",
    "usage": {"input_tokens": 1, "output_tokens": 2},
}))
write("cloud-responses", "openai-tools.json", json.dumps([
    {"type": "function", "function": {"name": "f", "description": "d",
                                      "parameters": {"type": "object"}}},
]))

# --- model-refs: arbitrary-encoded; raw strings still steer it -----------
for i, s in enumerate(["hf:org/repo:Q4_K_M", "cloud:openai:gpt-5", "hf:org/repo:safetensors",
                       "low", "4096"]):
    write("model-refs", f"ref-{i}", s)

# --- cli-argv: NUL-separated argv (after the implicit `runa`) ------------
for i, argv in enumerate([
    ["run", "m.gguf", "hi", "--mode", "hybrid", "--kv", "q8_0", "--lora", "a.gguf:0.5",
     "--device", "CUDA0,CUDA1", "--tensor-split", "3,1", "--think", "on", "--effort", "high",
     "--on-unfit", "cloud:openai:gpt-5", "--mcp", "python3 'my dir/s.py'"],
    ["chat", "hf:org/repo:Q4_K_M", "--tui", "--rpc", "127.0.0.1:50052", "--think-budget", "64"],
    ["serve", "--models", "a.gguf,b.gguf", "--port", "8080", "--tensor-split", "1,1"],
    ["bench", "m.gguf", "--kv-k", "f16", "--kv-v", "q4_0", "--json"],
    ["fit", "hf:org/repo:Q4_K_M", "--json"],
    ["daemon", "--models", "a,b", "--install"],
    ["media", "video", "v.mp4", "--fps", "0.5"],
    ["tasks", "claim", "P1", "--agent", "x"],
    ["--help"],
]):
    write("cli-argv", f"argv-{i}", b"\0".join(a.encode() for a in argv))

# --- cli-config: runa.toml text -----------------------------------------
write("cli-config", "runa.toml", """on_unfit = "cloud:openai:gpt-5"

[defaults]
threads = 8

[system]
max_load_percent = 80

[think]
mode = "budget"
budget = 512
show = true

[memory]
idle_secs = 60

[audio]
route = "asr"

[model]
lora = ["base.gguf:0.5"]

[models.q]
source = "hf:org/repo:Q4_K_M"
lora = "q.gguf"

[mcp.servers.fs]
command = "npx"
args = ["-y", "fs"]
env = { A = "1" }
""")
write("cli-config", "slash", "/think budget 128")

# --- serve-body: HTTP JSON bodies ----------------------------------------
write("serve-body", "chat.json", json.dumps({
    "model": "m", "stream": True, "max_tokens": 16, "reasoning_effort": "low",
    "messages": [
        {"role": "system", "content": "s"},
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c", "function": {"name": "f", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c", "content": "42"},
    ],
    "response_format": {"type": "json_schema", "json_schema": {"schema": {
        "type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}}},
    "tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}],
    "tool_choice": {"type": "function", "function": {"name": "f"}},
}))
write("serve-body", "messages.json", json.dumps({
    "model": "m", "max_tokens": 16, "system": "s",
    "thinking": {"type": "enabled", "budget_tokens": 64},
    "messages": [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "t", "name": "f", "input": {"a": 1}}]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t", "content": "ok"}]},
    ],
    "tools": [{"name": "f", "input_schema": {"type": "object"}}],
    "tool_choice": {"type": "tool", "name": "f"},
}))

# --- daemon-proto: NDJSON lines ------------------------------------------
write("daemon-proto", "request.json", json.dumps({
    "v": 1, "model": "m", "request": {
        "messages": [{"role": "user", "content": "hi", "tool_calls": [
            {"id": "c", "name": "f", "arguments": "{}"}], "tool_call_id": None}],
        "sampling": {"temperature": 0.8, "top_k": 40, "top_p": 0.95, "min_p": 0.05,
                     "repeat_penalty": 1.0, "repeat_last_n": 64, "seed": 42},
        "max_tokens": 16, "stop": ["</s>"], "add_generation_prompt": True,
        "think": {"mode": {"kind": "effort", "effort": "high"}, "show": True},
        "json_schema": None, "grammar": None, "tools": "[]", "tool_choice": "auto",
    }}))
write("daemon-proto", "event-usage.json", json.dumps(
    {"type": "usage", "prompt_tokens": 1, "generated_tokens": 2, "reasoning_tokens": 0}))
write("daemon-proto", "event-tools.json", json.dumps(
    {"type": "tool_calls", "calls": [{"id": "c", "name": "f", "arguments": "{}"}]}))
write("daemon-proto", "event-done.json", json.dumps({"type": "done", "stop": "eos"}))
