# The OpenAI-compatible API

`runa server start` serves the models of this machine to programs that speak the OpenAI API.

```sh
runa get bartowski/Qwen2.5-0.5B-Instruct-GGUF@Q4_K_M
runa server start            # http://127.0.0.1:7862; -p for another port
```

The base URL is `http://127.0.0.1:7862/v1`. A model is named as `runa ls` shows it, and the first
request that names it loads it. Without `--api-key` (or `RUNA_API_KEY`) the server takes any key
or none; with one, every request needs `Authorization: Bearer <key>`.

| Endpoint | What it does |
|---|---|
| `GET /v1/models` | The models that can be asked for |
| `POST /v1/chat/completions` | An answer in a conversation; `stream`, `tools`, `response_format` |
| `POST /v1/completions` | A continuation of a text, without the chat template |
| `POST /v1/embeddings` | The vectors of texts, from an embedding model |

Mistakes are answered as OpenAI answers them: `{"error": {"message", "type", ...}}` with `400`
for a request the server cannot take, `401` for a missing key and `404` for an unknown model.

## The `openai` Python client

[`examples/openai_client.py`](examples/openai_client.py) asks for an answer, a stream, a tool call
and JSON of a schema, and checks each; CI runs it against a real server.

```sh
pip install -r docs/examples/requirements.txt
python docs/examples/openai_client.py bartowski/Qwen2.5-0.5B-Instruct-GGUF@Q4_K_M
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:7862/v1", api_key="none")
answer = client.chat.completions.create(
    model="bartowski/Qwen2.5-0.5B-Instruct-GGUF@Q4_K_M",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
)
print(answer.choices[0].message.content)
```

## curl

The same requests by hand; `app/integration/server.sh` sends them in CI.

```sh
MODEL=bartowski/Qwen2.5-0.5B-Instruct-GGUF@Q4_K_M
URL=http://127.0.0.1:7862/v1
```

The models:

```sh
curl "$URL/models"
```

An answer:

```sh
curl "$URL/chat/completions" -H 'Content-Type: application/json' -d '{
  "model": "'"$MODEL"'",
  "messages": [{"role": "user", "content": "What is the capital of France?"}],
  "max_tokens": 32
}'
```

The answer as it is written, as server-sent events that end with `data: [DONE]`;
`stream_options` adds a last chunk with the token counts:

```sh
curl -N "$URL/chat/completions" -H 'Content-Type: application/json' -d '{
  "model": "'"$MODEL"'",
  "messages": [{"role": "user", "content": "Count from one to five."}],
  "stream": true,
  "stream_options": {"include_usage": true}
}'
```

### Tools

The model answers with `tool_calls` and the finish reason `tool_calls` instead of text:

```sh
curl "$URL/chat/completions" -H 'Content-Type: application/json' -d '{
  "model": "'"$MODEL"'",
  "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
  "tools": [{
    "type": "function",
    "function": {
      "name": "get_weather",
      "description": "The weather in a city",
      "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"]
      }
    }
  }]
}'
```

```json
{"role": "assistant", "content": null, "tool_calls": [{"id": "call_RX12xVtYOAhYnAeYczhun16U",
  "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]}
```

Run the tool, then send the conversation again with the model's message and a `tool` message that
carries the result under the call's `id`:

```sh
curl "$URL/chat/completions" -H 'Content-Type: application/json' -d '{
  "model": "'"$MODEL"'",
  "messages": [
    {"role": "user", "content": "What is the weather in Paris?"},
    {"role": "assistant", "content": null, "tool_calls": [{"id": "call_1", "type": "function",
      "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]},
    {"role": "tool", "tool_call_id": "call_1", "content": "18 degrees, sunny"}
  ]
}'
```

`tool_choice` is `auto` (the default: the model decides), `required` (the answer has a call),
`none` (the tools are kept from the model) or `{"type": "function", "function": {"name": ...}}`
for one tool. `parallel_tool_calls` says whether one answer may have several calls. How well a
model calls tools is the model's doing: the calls are written and read in the format of its chat
template, and the arguments are held to the tool's schema.

### JSON answers

`response_format` holds the answer to a JSON schema: the model cannot write anything else.

```sh
curl "$URL/chat/completions" -H 'Content-Type: application/json' -d '{
  "model": "'"$MODEL"'",
  "messages": [{"role": "user", "content": "Name a city and the country it is in."}],
  "response_format": {"type": "json_schema", "json_schema": {"name": "city", "schema": {
    "type": "object",
    "properties": {"city": {"type": "string"}, "country": {"type": "string"}},
    "required": ["city", "country"],
    "additionalProperties": false
  }}}
}'
```

`{"type": "json_object"}` asks for any JSON object. Give the answer room: one that `max_tokens`
cuts short is not complete JSON, and its finish reason is `length`.

### Text completions and embeddings

```sh
curl "$URL/completions" -H 'Content-Type: application/json' -d '{
  "model": "'"$MODEL"'", "prompt": "The capital of France is", "max_tokens": 8
}'
```

```sh
curl "$URL/embeddings" -H 'Content-Type: application/json' -d '{
  "model": "<an embedding model>", "input": ["first text", "second text"]
}'
```

## What is not there

- `functions`, the predecessor of `tools`, content that is not text (images, audio) and more than
  one choice (`n`): the request is refused with `400`.
- `logprobs`, and a `response_format` for `/v1/completions`: ignored.
- Reasoning apart from the answer: what a model thinks aloud stays in `content`.
