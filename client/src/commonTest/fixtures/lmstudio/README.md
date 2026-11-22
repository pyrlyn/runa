# Recorded LM Studio responses

What `LmStudioBackendTest` answers with instead of a running LM Studio.

The answers and the events are the examples of LM Studio's REST API documentation, read on
2026-11-22 for LM Studio 0.4.x, unchanged but for whitespace:

| File | Source |
| --- | --- |
| `models.json` | https://lmstudio.ai/docs/developer/rest/list |
| `load.json` | https://lmstudio.ai/docs/developer/rest/load |
| `unload.json` | https://lmstudio.ai/docs/developer/rest/unload |
| `chat.json` | https://lmstudio.ai/docs/developer/rest/chat |
| `chat-stream.txt`, `chat-stream-error.txt` | https://lmstudio.ai/docs/developer/rest/streaming-events, one event of each documented kind, framed as the page says: `event: <type>`, `data: <json>` |
| `download.json` | https://lmstudio.ai/docs/developer/rest/download |
| `download-status.json` | https://lmstudio.ai/docs/developer/rest/download-status |

The documentation shows no failed request, so the `error-*.json` files are what LM Studio
0.4.28 answered on 2026-11-22 to: unloading an instance that is not loaded (404), loading a model
it does not have (404), a load request with a key it does not know (400), and a path it does not
serve (404).
