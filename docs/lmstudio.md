# LM Studio

runa runs models itself and does not need LM Studio. With `--backend lmstudio` the model commands
talk to a running LM Studio instead of runa's own daemon, over LM Studio's v1 REST API, so its
models are listed, loaded and chatted with from the same command line.

```sh
runa --backend lmstudio ls                       # the models LM Studio has
runa --backend lmstudio load gemma-4-26b-a4b     # into LM Studio's memory
runa --backend lmstudio ps                       # what it has loaded
runa --backend lmstudio chat -p "Why is the sky blue?"
runa --backend lmstudio unload
runa --backend lmstudio get ibm/granite-4-micro  # LM Studio downloads, into its own folder

export RUNA_BACKEND=lmstudio                     # or `runa config set backend lmstudio`
runa ls
```

runa never starts LM Studio: its server must run already (LM Studio 0.4.0 or newer, in the app
or with `lms server start`). Nothing but its documented REST API is used.

## Where LM Studio is

| | Default | Set with |
|---|---|---|
| Host | `127.0.0.1` | `--host`, `RUNA_HOST`, `host` in `config.json` |
| Port | `1234` | `--port`, `RUNA_PORT`, `port` in `config.json` |
| API token | none | `--token`, or `LM_API_TOKEN` as in LM Studio's own examples |

LM Studio asks for a token only when authentication was turned on in its settings. The token of
runa's daemon (`RUNA_TOKEN`) is never sent to LM Studio.

## The commands

| Command | With `--backend lmstudio` |
|---|---|
| `ls` | The models LM Studio lists, one row each. `--llm`, `--embedding` and a name filter them; `--variants` is refused, because LM Studio lists every model once |
| `ps` | The loaded instances. The `GPU`, `IDLE` and `TTL` columns are left out: LM Studio does not report them |
| `load` | Loads the model into LM Studio. `--context-length` is passed on; `--gpu`, `--parallel`, `--ttl`, `--identifier`, `--estimate-only` and `--auto` are refused, because LM Studio's REST API has nothing for them. There is no progress bar: LM Studio reports no progress |
| `unload` | Unloads an instance; `--all` unloads every instance LM Studio lists, one request each |
| `chat` | Loads the model when it is not loaded, then chats. LM Studio keeps the conversation. Only the answer is printed: the reasoning of a model that thinks first is in `--json`, as `output` items of type `reasoning` |
| `get` | Asks LM Studio to download a model from its catalog (`ibm/granite-4-micro`), or from Hugging Face (`owner/repo@quantization` or a URL), and waits for it. The download belongs to LM Studio and goes on when the command is interrupted. `--select` is refused |

A model is named by its key (`google/gemma-4-26b-a4b`) or by the part after the publisher.

Every other command that needs a daemon is runa's own and says so instead of doing something else:

```console
$ runa --backend lmstudio log stream
Error: `runa log stream` is not supported by the lmstudio backend; `--backend runa` runs it with runa itself
```

That is `run`, `import`, `status`, `daemon up|down|status`, `server start|stop|status` and
`log stream`. `config`, `version` and `completion` work as always.

## JSON output

`--json` prints the same types as with runa's own daemon, so one script reads both. What LM
Studio does not tell is `null`:

| Output | `null` with LM Studio |
|---|---|
| `ls --json` | `params` when the label is not one number (`26B-A4B`); `architecture` of an embedding model |
| `ps --json`, and `instance` of `load --json` | `gpu`, `ttl_seconds`, `idle_seconds`; `parallel` when LM Studio does not say it. `size_bytes` is the size of the model's files, not of what the instance takes in memory |
| `load --json` | `estimate` |
| `chat --json` | `stats.stop_reason`; `response_id` for one question, which is not stored |

`ls --json` with LM Studio is the list of `GET /api/v1/models` in runa's types, not the list of
files that `runa ls --json` prints for runa's own store. `get --json` is the last state of LM
Studio's download job.

## Using LM Studio's models with runa itself

```sh
runa import --from-lmstudio            # every GGUF model of ~/.lmstudio/models
runa import --from-lmstudio=/other/models --dry-run
```

This needs no running LM Studio and no `--backend`: it adds the GGUF files LM Studio downloaded
to runa's store as symlinks, under the `publisher/model` names they have there, so nothing is
downloaded twice. `--copy` and `--hard-link` make copies or hard links instead. Models that are
not GGUF (MLX) are passed over, as are projector files (`mmproj`), and a model the store already
has is left alone.

## Sources

Read on 2026-11-22 for LM Studio 0.4.x, and tried against LM Studio 0.4.28:

- The REST API: <https://lmstudio.ai/docs/developer/rest> and its pages `list`, `load`, `unload`,
  `chat`, `streaming-events`, `download` and `download-status`.
- The API token: <https://lmstudio.ai/docs/developer/core/authentication>.
- The models folder: <https://lmstudio.ai/docs/app/advanced/import-model>.
