# Usage stats

`run`, `chat`, and `serve` append one line to a local journal when a
generation finishes. `runa stats` prints that journal. `runa web` shows
the same numbers in a browser and updates them as new lines land.

The journal stays on this machine. Nothing in either command sends it
anywhere else.

## Journal

Default path: `$XDG_CACHE_HOME/runa/usage.jsonl`, or
`~/.cache/runa/usage.jsonl` when `XDG_CACHE_HOME` is unset. On Windows
without `HOME`, `%LOCALAPPDATA%/runa/usage.jsonl`.

`RUNA_USAGE_LOG` replaces that path. There is no `runa.toml` key.

Each line is one JSON object:

```json
{"ts":1700000000,"source":"run","model":"qwen","prompt_tokens":15,"generated_tokens":48,"reasoning_tokens":35,"pp_toks_per_s":77.1,"tg_toks_per_s":15.5}
```

`source` is `run`, `chat`, or `serve` (including embeddings). A line that
is not valid JSON is skipped. Decode speed is a token-weighted average:
rows with `tg_toks_per_s` of 0 (the daemon socket does not report a rate)
do not pull the average down.

## `runa stats`

```sh
runa stats
runa stats --since 7d
runa stats --since 24h --json
```

`--since` takes `Nh` or `Nd` (`24h`, `7d`, `60d`). The text table lists
totals, sources, models, days, and the 20 newest rows. `--json` prints
the same snapshot the web page loads.

## `runa web`

```sh
runa web
runa web --port 3939
```

Binds `127.0.0.1:3939` unless you pass `--host` and `--port`. `--host`
must be a loopback address (`127.0.0.0/8` or `::1`). The process prints
`usage on http://127.0.0.1:3939` and serves:

| Route | What |
|-------|------|
| `GET /` | The page |
| `GET /stats.js` and `GET /stats.css` | The page assets |
| `GET /api/snapshot` | One JSON snapshot (same shape as `runa stats --json`) |
| `GET /ws` | WebSocket. A text frame with that JSON when the journal changes, and about once a second |
| `GET /health` | `{"ok":true}` |

The page is the built React app in `web/` (Vite, TanStack Query). It
loads `/api/snapshot` once, then applies each websocket frame on top of
that cache. Colours follow the runa brand: warm ash, charcoal, ember.

`runa serve` still has its own live process dashboard at `/dashboard`
([`dashboard.md`](dashboard.md)). That page is the server's in-memory
rings. `runa web` is the journal across `run`, `chat`, and `serve`.
