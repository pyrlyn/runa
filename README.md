# runa

Local models from the terminal: download GGUF models, load them in a background daemon, serve
them over an OpenAI-compatible API and chat with them, in one native binary.

runa is being rewritten in **Kotlin/Native**. The command set follows
[`lms`](https://github.com/lmstudio-ai/lms), the LM Studio CLI, but runa runs models itself
through llama.cpp and does not need LM Studio.

The implementation plan, phases and acceptance criteria are in [`PLAN.md`](PLAN.md).

How to use the OpenAI-compatible API, with `curl` and the `openai` Python client, is in
[`docs/openai.md`](docs/openai.md). How to watch what the daemon and its models do, and what its
log file holds, is in [`docs/logs.md`](docs/logs.md). How to list, load and chat with the models
of a running LM Studio through the same commands is in [`docs/lmstudio.md`](docs/lmstudio.md).
What `runa runtime survey` tells about a machine, and how `runa load --auto` chooses by it, is in
[`docs/hardware.md`](docs/hardware.md).

## Build

```sh
mise install               # JDK 21 and the lint tools from mise.toml
./gradlew build            # every module, tests on the host target, ktlint
./gradlew :app:linkReleaseExecutableMacosArm64   # or LinuxX64 / LinuxArm64
app/build/bin/macosArm64/releaseExecutable/runa.kexe --version
```

## Licence

GPL-3.0 ([`LICENSE`](LICENSE)); for a commercial licence see [`PRICING.md`](PRICING.md).
