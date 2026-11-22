# Logs

The daemon tells whoever listens what it does while it does it, and keeps a log file.

## `runa log stream`

```sh
runa log stream                            # what the models read and what they write
runa log stream --filter output --stats    # only the answers, each with its token counts and speed
runa log stream --source server            # the daemon's own log and every request it answered
runa log stream --source runtime           # the log of llama.cpp
runa log stream --json | jq -r .text       # one event a line, for another program
```

The command starts the daemon when it does not run. It ends with Ctrl-C, or when the daemon stops.
It shows what happens while it is open: nothing is kept for a listener that comes later.

| Source | What it has |
|---|---|
| `model` (the default) | The prompt as a model read it, with the chat template applied, and what the model wrote: its text and the tools it called. `--filter input` or `--filter output` keeps one of the two |
| `server` | The lines of the daemon's log at every level, and one line for each request: method, path, status and time |
| `runtime` | What llama.cpp logs, which is most when a model is loaded |

With `--json` every event is one line:

| Field | In | Content |
|---|---|---|
| `time` | every event | When it happened, as an ISO 8601 instant |
| `source` | every event | `model`, `server` or `runtime` |
| `type` | every event | `input` or `output` of a model, a `request` the server answered, or a `log` line |
| `text` | every event | The prompt, the answer, or the line |
| `level` | `log`, `request` | `debug`, `info`, `warn` or `error` |
| `model`, `endpoint` | `input`, `output` | The loaded instance that was asked, and by which endpoint: `chat`, `chat.completions` or `completions` |
| `tool_calls` | `output` | The tools the model called, each with its `name` and `arguments` |
| `stats` | `output`, with `--stats` | Token counts, time to first token, tokens per second and why the answer ended |

A field that does not apply is left out.

## The log file

`runa daemon status` says where the file is: `logs/daemon.log` in runa's directory.

- `--log-level` (or `RUNA_LOG_LEVEL`, or `runa config set log_level`) decides what is written:
  `info` by default. At `debug` the file also has every request and the log of llama.cpp.
- A file that has reached 10 MB becomes `daemon.log.1`, and the log goes on in a new file. The
  three files before the current one are kept.
- What the models read and write is **not** written to the file. It is somebody's conversation, and
  `runa log stream` shows it to whoever may ask the daemon. To have it in the file as well, start
  the daemon with `--log-prompts` (or `RUNA_LOG_PROMPTS=true`, or
  `runa config set log_prompts true`):

  ```sh
  runa daemon down
  runa --log-prompts daemon up
  ```

Both settings are read when the daemon starts; a daemon that runs already keeps what it was
started with.

## Over HTTP

`runa log stream` reads `GET /api/v1/logs/stream` of the daemon, which needs the daemon's token
(`run/daemon.json` in runa's directory). `source` and `filter` are query parameters; without
`source` the stream has every source. The answer is a stream of server-sent events: one
`listening` event, after which no event is missed, and then a `log` event for each event of the
log, with the JSON above as its data. A comment line every 15 seconds keeps a quiet stream open.
