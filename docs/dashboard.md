# Serve live dashboard

`runa serve` hosts a local dashboard on the same host and port as the API.
Open `http://<host>:<port>/dashboard`. The page is the built React app
(`dashboard/`, see `dashboard/README.md`). Numbers stay on this machine:
the server does not send them anywhere else.

| Route | What |
|-------|------|
| `GET /dashboard` and `GET /dashboard/` | The page (`text/html`) |
| `GET /dashboard/dashboard.js` | The page script |
| `GET /dashboard/dashboard.css` | The page stylesheet |
| `GET /dashboard/snapshot` | One JSON snapshot |
| `GET /dashboard/ws` | WebSocket. A text frame with the same JSON about every 1s, and again when a request starts, waits, finishes, or fails |

The UI loads `/dashboard/snapshot` once (TanStack Query), then applies each
websocket frame on top of that cache. `--api-key` covers `/v1` only.
`/dashboard` stays open on the same bind so a browser can load the page.

## Quotas

There is no quota config key. Two documented defaults, plus the `--parallel`
semaphore, decide `quotas.hitting_ceiling`:

- `default_tokens_per_minute` is `100000` (prompt + generated tokens finished
  in the last 60 seconds).
- `default_requests_per_second` is `16` (request starts in the last 60
  seconds, divided by 60).
- The flag is also true when every parallel slot is in use
  (`parallel_in_use == parallel_limit`).

Embeddings count as prompt tokens in that window. The token-speed graph
counts generated tokens only.

## What serve can see

- **Daemon.** `daemon.status` is `not_this_process` and `daemon.alive` is
  `null`. This process is `runa serve`, not `runa daemon`, and it does not
  probe the daemon socket.
- **MCP.** `mcp_tools` is an empty array. Serve does not host MCP servers
  (those run under `runa run` / `runa chat`).
- **Models.** `models.loaded[].memory_bytes` is the on-disk weight size from
  `ModelPool::loaded_weight_bytes`, not process RSS. Process RSS and CPU are
  `system_load` (this process only; CPU may exceed 100 when several cores
  are busy).
- **Reasoning.** `reasoning.tokens_spent` is the cumulative reasoning-token
  count from finished requests. `reasoning.active_budget_tokens` is the sum
  of `ThinkMode::Budget` caps on requests still in flight, or `null` when
  none of them set a budget.
- **Latency.** `latency.p50_ms`, `p95_ms`, and `p99_ms` are nearest-rank
  percentiles over the last 512 completed requests (`ceil(p * n) - 1`).
  `samples` is 0 until the first completion, and the three percentiles are
  then 0.
- **Errors.** `errors.count` is the total since process start.
  `errors.recent` keeps the newest 16, newest first, each with `message`,
  `traceback` (a short panic or backtrace text), and `at_unix_ms`.
- **Queue.** `queue.depth` is how many requests are waiting on the parallel
  semaphore. `in_flight` is the set that holds a slot.

## Snapshot

```json
{
  "token_speed": {
    "toks_per_s": 0.0,
    "decode_toks_per_s": 0.0,
    "history": [0]
  },
  "in_flight": [
    {"id": 1, "route": "/v1/chat/completions", "model": "qwen", "elapsed_ms": 12}
  ],
  "system_load": {"process_cpu_percent": 0.0, "process_rss_mib": 0},
  "latency": {"p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "samples": 0},
  "errors": {
    "count": 0,
    "recent": [{"message": "…", "traceback": "…", "at_unix_ms": 0}]
  },
  "quotas": {
    "parallel_limit": 1,
    "parallel_in_use": 0,
    "tokens_per_minute": 0,
    "requests_per_second": 0.0,
    "requests_in_window": 0,
    "window_secs": 60,
    "default_tokens_per_minute": 100000,
    "default_requests_per_second": 16.0,
    "hitting_ceiling": false
  },
  "daemon": {
    "status": "not_this_process",
    "alive": null,
    "detail": "runa serve does not host the warm daemon (that is `runa daemon`)."
  },
  "models": {
    "configured": 1,
    "loaded_count": 0,
    "loaded": [{"id": "qwen", "memory_bytes": 0}]
  },
  "mcp_tools": [],
  "queue": {
    "depth": 0,
    "waiting": [{"id": 1, "route": "/v1/chat/completions", "wait_ms": 3}]
  },
  "reasoning": {"tokens_spent": 0, "active_budget_tokens": null}
}
```

`token_speed.history` always has 60 numbers, oldest first (the example above shows a single `0` as a stand-in). Each bucket is generated
tokens in that one second (so the height is tok/s for that second).
`toks_per_s` is generated tokens over the whole 60s window, divided by 60.
`decode_toks_per_s` is the latest non-zero engine decode rate.
