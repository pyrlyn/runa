# runa

Local models from the terminal: download GGUF models, load them in a background daemon, serve
them over an OpenAI-compatible API and chat with them, in one native binary.

runa is being rewritten in **Kotlin/Native**. The command set follows
[`lms`](https://github.com/lmstudio-ai/lms), the LM Studio CLI, but runa runs models itself
through llama.cpp and does not need LM Studio.

There is no code yet. The implementation plan, phases and acceptance criteria are in
[`PLAN.md`](PLAN.md).

## Licence

GPL-3.0 ([`LICENSE`](LICENSE)); for a commercial licence see [`PRICING.md`](PRICING.md).
