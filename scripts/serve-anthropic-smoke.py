#!/usr/bin/env python3
# Copyright (c) 2026 Ivan Tugay
# SPDX-License-Identifier: GPL-3.0-or-later
# Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

"""P6.2: Anthropic Python SDK against a running `runa serve`.

Usage: serve-anthropic-smoke.py http://127.0.0.1:PORT
"""

from __future__ import annotations

import json
import sys
import urllib.request

from anthropic import Anthropic


def discover_model(base: str) -> str:
    """First model id from the server's OpenAI-style /v1/models listing.

    The server names CLI-passed models by file stem, so the id is not a
    fixed string (P3.9/P6.2).
    """
    with urllib.request.urlopen(f"{base}/v1/models") as resp:
        data = json.load(resp)
    return str(data["data"][0]["id"])


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: serve-anthropic-smoke.py http://127.0.0.1:PORT", file=sys.stderr)
        return 2
    base = sys.argv[1].rstrip("/")
    client = Anthropic(base_url=base, api_key="runa")
    model = discover_model(base)
    r = client.messages.create(
        model=model,
        max_tokens=8,
        messages=[{"role": "user", "content": "Say hi"}],
    )
    texts = [b.text for b in r.content if getattr(b, "type", "") == "text"]
    if not texts or not texts[0]:
        print(f"empty message content: {r.content!r}", file=sys.stderr)
        return 1
    n = 0
    stream = client.messages.create(
        model=model,
        max_tokens=8,
        messages=[{"role": "user", "content": "Say hi"}],
        stream=True,
    )
    for _event in stream:
        n += 1
    if n == 0:
        print("no anthropic stream events", file=sys.stderr)
        return 1
    tool = tool_round_trip(client, model)
    print(f"ok text={texts[0]!r} events={n} tool={tool}")
    return 0


WEATHER = {
    "name": "get_weather",
    "description": "Current weather for a city",
    "input_schema": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
}


def tool_round_trip(client: Anthropic, model: str) -> dict:
    """P8.2: forced tool_use (non-stream + accumulated stream), then answer
    the tool_result."""
    messages = [{"role": "user", "content": "What is the weather in Paris?"}]
    r = client.messages.create(
        model=model,
        max_tokens=96,
        temperature=0,
        messages=messages,
        tools=[WEATHER],
        tool_choice={"type": "any"},
    )
    uses = [b for b in r.content if b.type == "tool_use"]
    if r.stop_reason != "tool_use" or not uses or uses[0].name != "get_weather":
        raise SystemExit(f"no tool_use: {r!r}")
    with client.messages.stream(
        model=model,
        max_tokens=96,
        temperature=0,
        messages=messages,
        tools=[WEATHER],
        tool_choice={"type": "tool", "name": "get_weather"},
    ) as stream:
        final = stream.get_final_message()
    streamed = [b for b in final.content if b.type == "tool_use"]
    if not streamed or streamed[0].name != "get_weather":
        raise SystemExit(f"no streamed tool_use: {final!r}")
    messages += [
        {"role": "assistant", "content": [b.model_dump() for b in r.content]},
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": uses[0].id, "content": "sunny, 21 C"}
            ],
        },
    ]
    client.messages.create(
        model=model, max_tokens=32, temperature=0, messages=messages, tools=[WEATHER]
    )
    return uses[0].input


if __name__ == "__main__":
    raise SystemExit(main())
