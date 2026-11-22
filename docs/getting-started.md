# Getting started

runa downloads GGUF models, runs them through llama.cpp and serves them to other programs. This
page goes from nothing to a model that answers, in the order the commands are needed. Every
command with all of its options is in [commands.md](commands.md).

## Install

See [install.md](install.md), or build from the source as the [README](../README.md#build) says.

```sh
runa --version
```

## Get a model

A model is named as on Hugging Face, `owner/repo`, with `@quantization` to pick one file of the
repository. Without it runa takes the largest quantized variant that fits the memory of this
machine.

```sh
runa get bartowski/Qwen2.5-0.5B-Instruct-GGUF@Q4_K_M
runa ls
```

```text
MODEL                                  TYPE  PARAMS  ARCH   VARIANTS SIZE
bartowski/Qwen2.5-0.5B-Instruct-GGUF   llm   494M    qwen2  q4_k_m   398 MB
```

The download is checked against the SHA-256 the hub gives, and an interrupted one continues when
the command is run again. A `.gguf` file that is on the disk already is added with
`runa import <file>`, and the models of LM Studio with `runa import --from-lmstudio`.

In the commands below `<model>` is a name from `runa ls`, here
`bartowski/Qwen2.5-0.5B-Instruct-GGUF@Q4_K_M`.

## Ask one question

```sh
runa run <model> "Say hello." --temp 0
```

`runa run` loads the model in its own process, prints the answer as it is generated and unloads
the model again. Nothing keeps running. `--stats` adds the token counts and the speed, `--json`
prints one JSON object instead.

## Chat

```sh
runa chat <model>
```

`runa chat` talks to the daemon, the background process that keeps models loaded. It starts the
daemon when it does not run and loads the model when it is not loaded, so the second question is
answered at once. `/help` lists what can be typed besides a message, and Ctrl-C ends the chat.

With `-p`, or with the question on standard input, there is one answer and no prompt, for
scripts:

```sh
runa chat <model> -p "Hello" --stats
```

```text
Hello! How may I assist you today? If you have any questions or need any information, feel free to ask.
prompt: 30 tokens, answer: 24 tokens (stop), time to first token: 0.02 s, 310.8 tokens/s
```

## Load, see and unload

```sh
runa load <model>            # --context-length, --gpu, --ttl, or --auto to let runa choose
runa ps
runa unload                  # the one loaded model; with several, its ID, or --all
```

```text
Starting the runa daemon
Loaded bartowski/Qwen2.5-0.5B-Instruct-GGUF@q4_k_m in 0.12 s: context 4096, gpu max, about 448 MB (weights 398 MB, context 50.3 MB)
```

`runa load --estimate-only` prints the memory a model would take and loads nothing. How
`--auto` chooses, and what `runa runtime survey` tells about the machine, is in
[hardware.md](hardware.md).

## Serve other programs

```sh
runa server start            # http://127.0.0.1:7862
runa server status
runa server stop
```

The server speaks the OpenAI API, so a program that has a base URL setting works with
`http://127.0.0.1:7862/v1`. The endpoints, `curl` examples and the `openai` Python client are in
[openai.md](openai.md).

## The daemon

```sh
runa status
runa log stream              # what the daemon and its models do, until Ctrl-C
runa daemon down
```

```text
The runa daemon is running on 127.0.0.1:49286 (pid 28799, runa 0.1.0-dev, up 0s)
Engine: llama.cpp 1.0.0 (cpu, mtl, blas)
Loaded models: 1
Server: not running
```

`runa daemon down` unloads every model and stops the process; the next command that needs the
daemon starts it again. The log and its file are described in [logs.md](logs.md).

## Settings

```sh
runa config get              # every setting, its value and where the value comes from
runa config set gpu off
runa config path
```

A setting comes from a flag, else from its environment variable, else from `config.json`, else
from the default; `runa config get` names the source of each. Everything runa keeps is under
`~/.runa`: the models in `models`, `config.json`, and the daemon's log in `logs`. `RUNA_HOME`
moves all of it.

## With LM Studio

`--backend lmstudio` turns the same commands to a running LM Studio: `runa --backend lmstudio ls`
lists its models, `load`, `chat` and `ps` work on them. See [lmstudio.md](lmstudio.md).

## Completion and scripts

```sh
runa completion zsh > ~/.runa-completion.zsh     # or bash, fish
echo 'source ~/.runa-completion.zsh' >> ~/.zshrc
```

Every command that prints a table prints JSON with `--json`, and the exit code tells a mistake in
the command (2) from a failure of the daemon, the engine or the hub (3); the table is in
[commands.md](commands.md).
