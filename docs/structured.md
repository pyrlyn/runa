# Structured output

`runa` can force a model's answer into a JSON Schema or a GBNF grammar.
Local models use llama.cpp grammar sampling, so the output always matches.
OpenAI models get the schema as `response_format` (`strict: true`).

## CLI

```sh
runa run qwen "Capital of France and its population?" \
  --json-schema '{"type":"object","properties":{"city":{"type":"string"},"population":{"type":"integer"}},"required":["city","population"]}'
runa run qwen "Is Paris in France?" --grammar yes-no.gbnf
```

| Flag | Value | Notes |
| --- | --- | --- |
| `--json-schema` | file path, or inline JSON starting with `{` | Local, OpenAI, and Anthropic (forced tool, see below). |
| `--grammar` | GBNF file | Local only. Conflicts with `--json-schema`. |

A constrained request turns thinking off, and it turns off the Zig kernel
sampler and n-gram speculation. The grammar must see every token.

## `runa serve`

`POST /v1/chat/completions` accepts the OpenAI `response_format`:

| `type` | Effect |
| --- | --- |
| `text` | No constraint |
| `json_object` | Any JSON object (`{"type":"object"}`) |
| `json_schema` | `json_schema.schema` is required; the answer matches it |

An invalid schema returns `400`.

## Tool calling

Both `POST /v1/chat/completions` and `POST /v1/messages` accept tools and
return calls in the API's own shape:

| | OpenAI | Anthropic |
| --- | --- | --- |
| Tools | `tools` (`type: function`) | `tools` (`name`, `input_schema`) |
| Choice | `auto`, `required`, `none`, `{"function":{"name":…}}` | `auto`, `any`, `none`, `{"type":"tool","name":…}` |
| Calls out | `message.tool_calls`, `finish_reason: tool_calls` | `tool_use` blocks, `stop_reason: tool_use` |
| Results in | `role: tool` + `tool_call_id` | `tool_result` blocks |

Streaming sends each call once, when generation ends: one `tool_calls`
delta for OpenAI, and `content_block_start` / `input_json_delta` /
`content_block_stop` per call for Anthropic.

The model's chat template decides the call format (Hermes `<tool_call>`
for Qwen3, Harmony `<|channel|>commentary to=functions.…` for gpt-oss
— covered by `harmony_tool_reply_parses` on the real 20B template —
generic JSON when the template has no tool support). `auto`
uses a lazy grammar that starts at the format's trigger, so the model can
still answer in plain text or think first. `required` and a named tool use
an eager grammar. A model without a chat template rejects tools with `400`.
Tools cannot be combined with image or audio input.

## MCP tool loop (`run` / `chat`)

```sh
runa run qwen3 "What is the weather in Paris?" --mcp 'python3 weather_server.py'
runa chat qwen3 --mcp 'npx -y @modelcontextprotocol/server-filesystem .'
```

`--mcp '<command args>'` (repeatable) and `[mcp.servers.<name>]` in config
(`docs/config.md`) start stdio MCP servers. Their tools go to the model. When
the model calls one, runa runs it on the server that listed it, sends the
result back as a `tool` message, and generates again. This repeats until the
model answers without a call, or `--max-tool-rounds` (default 8) is reached,
which is an error. Each call is logged to stderr as
`[tool] name(args) -> N bytes`. A failed call goes back to the model as
`error: …` text.

| Backend | How tools travel |
| --- | --- |
| Local | Chat template, as in `runa serve` above |
| OpenAI | `tools`, assistant `tool_calls`, `role: tool` messages |
| Anthropic | `tools`, `tool_use` blocks (sent back whole, thinking included), `tool_result` blocks |

`--mcp` splits with shell quoting (P10.8): `'...'` is literal, `"..."`
allows backslash escapes, and a backslash outside quotes escapes the next
character — `--mcp "python3 'my dir/s.py'"` works. Unterminated quotes
are an error. Anything fancier (env vars, per-server `env`) goes in
`[mcp.servers]`. Two servers that offer the same tool name are an error.
Chat keeps history across turns (P10.9): every turn sends the transcript so far, tool rounds included. `/reset` and `/model` clear it.

Anthropic structured output (`--json-schema`) uses the same mechanism: one
forced `answer` tool whose `input_schema` is the schema. Its input is the
answer. Forcing a tool turns thinking off, and it cannot be combined with
`--mcp`.

## How it renders

Constrained requests render through llama.cpp's Jinja chat handler
(`apply_chat_template_oaicompat`). The handler knows the model's chat
format. It returns the prompt, the grammar, lazy-grammar triggers, and extra
stop strings. Models without a chat template fall back to the plain prompt
plus an eager grammar. Schemas are converted with llama.cpp's
`json_schema_to_grammar` (`runa_engine::schema_to_grammar`).
