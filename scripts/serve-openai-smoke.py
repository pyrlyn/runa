#!/usr/bin/env python3
# Copyright (c) 2026 Ivan Tugay
# SPDX-License-Identifier: GPL-3.0-or-later
# Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

"""P3.9: OpenAI Python SDK against a running `runa serve`.

Usage: serve-openai-smoke.py http://127.0.0.1:PORT
"""

from __future__ import annotations

import json
import sys

from openai import OpenAI


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: serve-openai-smoke.py http://127.0.0.1:PORT", file=sys.stderr)
        return 2
    base = sys.argv[1].rstrip("/")
    client = OpenAI(base_url=f"{base}/v1", api_key="runa")
    model = client.models.list().data[0].id
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Say hi"}],
        max_tokens=8,
        extra_body={"reasoning_effort": "low"},
    )
    content = r.choices[0].message.content
    if not content:
        print("empty non-stream content", file=sys.stderr)
        return 1
    n = 0
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Say hi"}],
        max_tokens=8,
        stream=True,
        extra_body={"reasoning_budget_tokens": 64},
    )
    for _chunk in stream:
        n += 1
    if n == 0:
        print("no stream chunks", file=sys.stderr)
        return 1
    call = tool_round_trip(client, model)
    print(f"ok content={content!r} chunks={n} tool={call}")
    return 0


WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


def tool_round_trip(client: OpenAI, model: str) -> str:
    """P8.2: forced tool call (non-stream + stream), then answer the result."""
    messages = [{"role": "user", "content": "What is the weather in Paris?"}]
    r = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=[WEATHER],
        tool_choice="required",
        max_tokens=96,
        temperature=0,
    )
    choice = r.choices[0]
    calls = choice.message.tool_calls or []
    if choice.finish_reason != "tool_calls" or not calls:
        raise SystemExit(f"no tool call: {choice!r}")
    call = calls[0]
    if call.function.name != "get_weather":
        raise SystemExit(f"wrong tool: {call!r}")
    json.loads(call.function.arguments)
    streamed = []
    for chunk in client.chat.completions.create(
        model=model,
        messages=messages,
        tools=[WEATHER],
        tool_choice={"type": "function", "function": {"name": "get_weather"}},
        max_tokens=96,
        temperature=0,
        stream=True,
    ):
        streamed.extend(chunk.choices[0].delta.tool_calls or [])
    if not streamed or streamed[0].function.name != "get_weather":
        raise SystemExit(f"no streamed tool call: {streamed!r}")
    messages += [
        {"role": "assistant", "content": None, "tool_calls": [call.model_dump()]},
        {"role": "tool", "tool_call_id": call.id, "content": "sunny, 21 C"},
    ]
    client.chat.completions.create(
        model=model, messages=messages, tools=[WEATHER], max_tokens=32, temperature=0
    )
    return call.function.arguments


if __name__ == "__main__":
    raise SystemExit(main())
