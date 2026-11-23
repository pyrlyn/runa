# Commands

<!-- Written by CommandsReferenceTest from the help of every command. Do not edit it:
     change the help, then run RUNA_UPDATE_GOLDEN=1 ./gradlew :cli:allTests. -->

Every command of `runa` with the help it prints for `--help`. The options of `runa`
itself go before the command (`runa --yes get <model>`), the options of a command after
it. What the commands are for, in the order somebody new needs them, is in
[getting-started.md](getting-started.md).

A command ends with one of these exit codes:

| Code | Meaning |
| --- | --- |
| 0 | Done |
| 2 | A mistake in what was typed or asked for: an unknown option, a model that is not there |
| 3 | The daemon, the hub or LM Studio did not answer or refused |
| 1 | An error runa did not expect, with a stack trace: a bug to report |
| 130 | Stopped with Ctrl-C |

## `runa`

```text
Usage: runa [<options>] <command> [<args>]...

  Local models from the terminal.

Options:
  --json               Print JSON instead of tables and text
  --log-level=<level>  debug, info, warn or error (env: RUNA_LOG_LEVEL)
  --log-prompts        Write what the models read and write into the daemon's
                       log file, for a daemon this command starts (env:
                       RUNA_LOG_PROMPTS)
  --quiet              Print errors only
  --verbose            Print debug detail
  --host=<host>        Host of the daemon, or of LM Studio (env: RUNA_HOST)
  --port=<port>        Port of the daemon, or of LM Studio instead of 1234
                       (env: RUNA_PORT)
  --token=<token>      API token of a daemon on another machine (env:
                       RUNA_TOKEN), or of LM Studio (env: LM_API_TOKEN)
  --backend=<name>     runa, or lmstudio to list, load and chat with the models
                       of a running LM Studio (env: RUNA_BACKEND)
  -y, --yes            Take the default answer for every prompt
  -v, --version        Show the version and exit
  -h, --help           Show this message and exit

Local models:
  get     Download a GGUF model from Hugging Face
  ls      List the models on this machine
  import  Add a GGUF file to the model store
  run     Ask a model one question
  chat    Talk to a model
  load    Load a model into the daemon
  unload  Unload a model and free its memory
  ps      List the loaded models

Serve:
  status  Show whether the daemon runs and what it has loaded
  server  Serve the OpenAI-compatible API to other programs
  log     Watch what the daemon and its models do
  daemon  Start and stop the background process that keeps models loaded

Runtime:
  runtime  Show what this machine and this build run models with

Commands:
  version     Print the runa version
  config      Show and change settings (config.json)
  completion  Print a completion script for bash, zsh or fish
```

## `runa get`

```text
Usage: runa get [<options>] <model>

  Download a GGUF model from Hugging Face

  Without @quantization, the largest quantized variant that fits this machine's
  memory is taken. Gated and private models need an access token in HF_TOKEN.
  An interrupted download continues where it stopped when the command is run
  again.

  With --backend lmstudio a running LM Studio downloads the model into its own
  folder: a name from its catalog, or owner/repo@quantization and URLs as
  above. It goes on when this command is interrupted.

Options:
  --select    Choose the variant from a list instead of taking the default
  --json      Print JSON
  -h, --help  Show this message and exit

Arguments:
  <model>  owner/repo, owner/repo@quantization, or a https://huggingface.co/
           URL
```

## `runa ls`

```text
Usage: runa ls [<options>] [<key>]

  List the models on this machine

  With --backend lmstudio: the models of a running LM Studio, one row each, as
  its API lists them.

Options:
  --llm        Show language models only
  --embedding  Show embedding models only
  --variants   Show one row per file instead of one per model
  --json       Print JSON
  -h, --help   Show this message and exit

Arguments:
  <key>  Show the variants of one model, for example publisher/repo
```

## `runa import`

```text
Usage: runa import [<options>] [<file>]

  Add a GGUF file to the model store

  With --from-lmstudio the models that LM Studio downloaded are added under the
  names they have there, as symlinks unless --copy or --hard-link says
  otherwise: nothing is downloaded twice, and LM Studio keeps its files. Models
  that are already in the store are left alone.

Options:
  --from-lmstudio[=<dir>]       Instead of one file: symlink every GGUF model
                                of LM Studio's models folder (default:
                                ~/.lmstudio/models)
  --user-repo=<publisher/repo>  Name the model instead of deriving it from the
                                file
  --copy, --hard-link, --symlink
                                Keep the original file and copy, hard link or
                                symlink it instead (default: move)
  --dry-run                     Show what would be done and change nothing
  --json                        Print JSON
  -h, --help                    Show this message and exit

Arguments:
  <file>  A .gguf file; any part of a split model brings all of its parts
```

## `runa run`

```text
Usage: runa run [<options>] <model> <prompt>

  Ask a model one question

  The model is loaded in this process, answers and is unloaded again: no daemon
  is needed. The answer is printed as it is generated; a longer context than
  the model has is lowered to it.

Options:
  -s, --system=<text>       System prompt
  --max-tokens=<n>          Stop after this many tokens
  --temp=<t>                Sampling temperature; 0 always takes the most
                            likely token (default: 0.8)
  --seed=<n>                Random seed; the same seed gives the same answer
  --gpu=off|max|0..1        How much of the model runs on the GPU (default: gpu
                            in config.json)
  -c, --context-length=<n>  Tokens of context (default: context_length in
                            config.json, else 4096)
  --stats                   Print token counts, time to first token and tokens
                            per second
  --json                    Print one JSON object when the answer is complete
  -h, --help                Show this message and exit

Arguments:
  <model>   A model from `runa ls`, or the path of a .gguf file
  <prompt>  What to ask the model
```

## `runa chat`

```text
Usage: runa chat [<options>] [<model>]

  Talk to a model

  The daemon is started when it does not run, and the model is loaded when it
  is not loaded yet; it stays loaded afterwards, see --ttl. The answers are
  printed as they are generated, and the model remembers the conversation until
  /clear. /help lists what else can be typed. Ctrl-C ends the chat, also in the
  middle of an answer.

  With -p, or with the question on standard input, the one answer goes to
  standard output and nothing else does.

  With --backend lmstudio the model is one of a running LM Studio, which then
  also keeps the conversation.

Options:
  -p, --prompt=<text>  Ask this one question, print the answer and exit
  -s, --system=<text>  System prompt
  --stats              Print token counts, time to first token and tokens per
                       second after every answer
  --ttl=<seconds>      When this command loads the model: unload it after this
                       long without a request; 0 for never (default: ttl in
                       config.json)
  --json               With one question: print one JSON object when the answer
                       is complete
  -h, --help           Show this message and exit

Arguments:
  <model>  A loaded model from `runa ps` or a model from `runa ls`; without
           one, the loaded model or a list to choose from
```

## `runa load`

```text
Usage: runa load [<options>] [<model>]

  Load a model into the daemon

  The model stays in memory until `runa unload`, its --ttl or `runa daemon
  down`. The daemon is started when it does not run. A model can be loaded more
  than once: the second copy is called `key:2`, or what --identifier says. On
  success the identifier is printed.

  With --auto the model goes to the GPU as a whole when it fits there, and gets
  the longest context that still fits, up to 32768 tokens and to what it was
  trained with. A model too large for the GPU is split between the GPU and the
  CPU, or runs on the CPU. A fifth of the memory stays free, and what the
  loaded models hold counts. With --estimate-only it loads nothing and tells
  what it would choose.

  With --backend lmstudio the model is loaded into LM Studio, which takes
  --context-length and none of the other settings.

Options:
  --identifier=<id>         The name to address the loaded model by (default:
                            its key)
  -c, --context-length=<n>  Tokens of context (default: context_length in
                            config.json, else 4096)
  --gpu=off|max|0..1        How much of the model runs on the GPU (default: gpu
                            in config.json)
  --parallel=<n>            How many requests the model serves at once
                            (default: 1)
  --ttl=<seconds>           Unload the model after this long without a request;
                            0 for never (default: ttl in config.json)
  --auto                    Let the daemon choose --gpu and --context-length by
                            the memory of its machine; one that is given is
                            kept
  --estimate-only           Print the memory the model would take and load
                            nothing
  --json                    Print JSON
  -h, --help                Show this message and exit

Arguments:
  <model>  A model from `runa ls`; without one, a list to choose from is shown
```

## `runa unload`

```text
Usage: runa unload [<options>] [<id>]

  Unload a model and free its memory

Options:
  --all       Unload every model
  --json      Print JSON
  -h, --help  Show this message and exit

Arguments:
  <id>  The identifier `runa ps` shows; not needed when one model is loaded
```

## `runa ps`

```text
Usage: runa ps [<options>]

  List the loaded models

Options:
  --json      Print JSON
  -h, --help  Show this message and exit
```

## `runa status`

```text
Usage: runa status [<options>]

  Show whether the daemon runs and what it has loaded

Options:
  --json      Print JSON
  -h, --help  Show this message and exit
```

## `runa server`

```text
Usage: runa server [<options>] <command> [<args>]...

  Serve the OpenAI-compatible API to other programs

  `server start` opens the daemon's API on an address and port of your
  choosing, by default 127.0.0.1:7862. A program that speaks the OpenAI API
  takes `http://127.0.0.1:7862/v1` as its base URL; a model is loaded when a
  request first names it. `server stop` closes the port again and leaves the
  daemon and its models alone.

Options:
  -h, --help  Show this message and exit

Commands:
  start   Open the API on a port for other programs, unless it is open already
  stop    Close the port; the daemon keeps running and keeps its models
  status  Show whether the server runs and where
```

## `runa server start`

```text
Usage: runa server start [<options>]

  Open the API on a port for other programs, unless it is open already

Options:
  -p, --port=<port>   Port to listen on (config: server.port)
  --bind=<address>    Address to listen on; 0.0.0.0 is every network interface
                      (config: server.bind)
  --cors / --no-cors  Let pages in a browser call the API (config: server.cors)
  --api-key=<key>     Ask the callers of /v1 for this bearer key (env:
                      RUNA_API_KEY)
  --json              Print JSON
  -h, --help          Show this message and exit
```

## `runa server stop`

```text
Usage: runa server stop [<options>]

  Close the port; the daemon keeps running and keeps its models

Options:
  -h, --help  Show this message and exit
```

## `runa server status`

```text
Usage: runa server status [<options>]

  Show whether the server runs and where

Options:
  --json      Print JSON
  -h, --help  Show this message and exit
```

## `runa log`

```text
Usage: runa log [<options>] <command> [<args>]...

  Watch what the daemon and its models do

Options:
  -h, --help  Show this message and exit

Commands:
  stream  Print the daemon's log as it is written, until Ctrl-C or until the
          daemon stops
```

## `runa log stream`

```text
Usage: runa log stream [<options>]

  Print the daemon's log as it is written, until Ctrl-C or until the daemon
  stops

  Nothing of what the models read and write is kept: the stream shows what
  happens while it is open. It reaches the daemon's log file only when the
  daemon was started with `--log-prompts`.

Options:
  --source=(model|server|runtime)
                           model: what the models read and write; server: the
                           daemon's log and its requests; runtime: the engine's
                           log (default: model)
  --filter=(input|output)  Of the model source, only what the models read, or
                           only what they write
  --stats                  Show token counts, time to first token and tokens
                           per second of every answer
  --json                   Print every event as one line of JSON
  -h, --help               Show this message and exit
```

## `runa daemon`

```text
Usage: runa daemon [<options>] <command> [<args>]...

  Start and stop the background process that keeps models loaded

  `load`, `unload`, `ps`, `chat`, `server` and `log stream` talk to the daemon;
  `load`, `ps`, `chat`, `server start` and `log stream` start it when it does
  not run. It listens on a loopback port of its own choosing. `daemon status`
  says where its log is; a log file of 10 MB is put aside, and the last three
  of those are kept.

Options:
  -h, --help  Show this message and exit

Commands:
  up      Start the daemon, unless it runs already
  down    Stop the daemon; the models it holds are unloaded
  status  Show whether the daemon runs; the same as `runa status`
```

## `runa daemon up`

```text
Usage: runa daemon up [<options>]

  Start the daemon, unless it runs already

Options:
  --json      Print JSON
  -h, --help  Show this message and exit
```

## `runa daemon down`

```text
Usage: runa daemon down [<options>]

  Stop the daemon; the models it holds are unloaded

Options:
  -h, --help  Show this message and exit
```

## `runa daemon status`

```text
Usage: runa daemon status [<options>]

  Show whether the daemon runs; the same as `runa status`

Options:
  --json      Print JSON
  -h, --help  Show this message and exit
```

## `runa runtime`

```text
Usage: runa runtime [<options>] <command> [<args>]...

  Show what this machine and this build run models with

Options:
  -h, --help  Show this message and exit

Commands:
  survey  Show the CPU, the memory, the GPUs and what the engine computes on
```

## `runa runtime survey`

```text
Usage: runa runtime survey [<options>]

  Show the CPU, the memory, the GPUs and what the engine computes on

  It is the machine this command runs on, and it asks no daemon. A GPU counts
  for `runa load` only when the engine was built to compute on it: the survey
  says so when it was not. `runa load --auto` chooses by this survey, taken by
  the daemon on its own machine.

Options:
  --json      Print JSON
  -h, --help  Show this message and exit
```

## `runa version`

```text
Usage: runa version [<options>]

  Print the runa version

Options:
  --json      Print JSON
  -h, --help  Show this message and exit
```

## `runa config`

```text
Usage: runa config [<options>] <command> [<args>]...

  Show and change settings (config.json)

Options:
  -h, --help  Show this message and exit

Commands:
  get   Print one setting, or all of them, after flags and environment are
        applied
  set   Save a setting to config.json
  path  Print where config.json lives
```

## `runa config get`

```text
Usage: runa config get [<options>] [<key>]

  Print one setting, or all of them, after flags and environment are applied

Options:
  --json      Print JSON
  -h, --help  Show this message and exit

Arguments:
  <key>  A setting name, for example server.port
```

## `runa config set`

```text
Usage: runa config set [<options>] <key> <value>

  Save a setting to config.json

Options:
  -h, --help  Show this message and exit

Arguments:
  <key>    A setting name, for example server.port
  <value>
```

## `runa config path`

```text
Usage: runa config path [<options>]

  Print where config.json lives

Options:
  --json      Print JSON
  -h, --help  Show this message and exit
```

## `runa completion`

```text
Usage: runa completion [<options>] <shell>

  Print a completion script for bash, zsh or fish

Options:
  -h, --help  Show this message and exit
```
