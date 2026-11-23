"""The official OpenAI client talking to `runa server`.

    runa server start
    pip install -r docs/examples/requirements.txt
    python docs/examples/openai_client.py <model>

RUNA_BASE_URL names another address than http://127.0.0.1:7862/v1, RUNA_API_KEY the key the
server was started with. Every step checks its answer, so this is also the smoke test CI runs
(app/integration/openai.sh).
"""

import json
import os
import sys

from openai import OpenAI

model = sys.argv[1]
client = OpenAI(
    base_url=os.environ.get("RUNA_BASE_URL", "http://127.0.0.1:7862/v1"),
    # The client wants a key even when the server asks for none.
    api_key=os.environ.get("RUNA_API_KEY", "none"),
)

models = [listed.id for listed in client.models.list()]
print("models:", models)
assert models

# An answer in one piece.
answer = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": "You answer in one short sentence."},
        {"role": "user", "content": "What is the capital of France?"},
    ],
    max_tokens=32,
)
print("answer:", answer.choices[0].message.content)
assert answer.choices[0].message.content
assert answer.usage.completion_tokens > 0

# The same, piece by piece.
stream = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Count from one to five."}],
    max_tokens=32,
    stream=True,
)
pieces = [chunk.choices[0].delta.content or "" for chunk in stream if chunk.choices]
print("stream:", "".join(pieces))
assert "".join(pieces)

# A tool call, and the answer once the tool's result is known.
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "The weather in a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]
question = {"role": "user", "content": "What is the weather in Paris?"}
called = client.chat.completions.create(
    model=model,
    messages=[question],
    tools=tools,
    tool_choice="required",
    temperature=0,
    max_tokens=64,
)
call = called.choices[0].message.tool_calls[0]
print("tool call:", call.function.name, call.function.arguments)
assert called.choices[0].finish_reason == "tool_calls"
assert call.function.name == "get_weather"
assert "city" in json.loads(call.function.arguments)

told = client.chat.completions.create(
    model=model,
    messages=[
        question,
        called.choices[0].message,
        {"role": "tool", "tool_call_id": call.id, "content": "18 degrees, sunny"},
    ],
    tools=tools,
    temperature=0,
    max_tokens=48,
)
print("after the tool:", told.choices[0].message.content)
assert "18" in told.choices[0].message.content

# An answer that is JSON of a given schema.
shaped = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Name a city and the country it is in."}],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "city",
            "schema": {
                "type": "object",
                "properties": {"city": {"type": "string"}, "country": {"type": "string"}},
                "required": ["city", "country"],
                "additionalProperties": False,
            },
        },
    },
    temperature=0,
    max_tokens=64,
)
city = json.loads(shaped.choices[0].message.content)
print("json:", city)
assert set(city) == {"city", "country"}

# A text completion, without the chat template.
completion = client.completions.create(model=model, prompt="The capital of France is", max_tokens=8)
print("completion:", completion.choices[0].text)
assert completion.choices[0].text

print("ok")
