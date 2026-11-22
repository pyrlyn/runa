# runa

Local models from the terminal: download GGUF models, load them in a background daemon, serve
them over an OpenAI-compatible API and chat with them, in one native binary.

runa is being rewritten in **Kotlin/Native**. The command set follows
[`lms`](https://github.com/lmstudio-ai/lms), the LM Studio CLI, but runa runs models itself
through llama.cpp and does not need LM Studio.

The implementation plan, phases and acceptance criteria are in [`PLAN.md`](PLAN.md).

## Build

```sh
mise install               # JDK 21 and the lint tools from mise.toml
./gradlew build            # every module, tests on the host target, ktlint
./gradlew :app:linkReleaseExecutableMacosArm64   # or LinuxX64 / LinuxArm64
app/build/bin/macosArm64/releaseExecutable/runa.kexe --version
```

## Licence

GPL-3.0 ([`LICENSE`](LICENSE)); for a commercial licence see [`PRICING.md`](PRICING.md).
