# docs/ — documentation index

Everything a reader needs beyond the root [`README.md`](../README.md): user
guides, reference pages, engineering notes, and the task-claim registry.
All prose in this repository is English, except the translations in `docs/ru/` and
`docs/uk/` (see [`../CONTRIBUTING.md`](../CONTRIBUTING.md)).

| Audience | Start with |
|----------|------------|
| New user | [`getting-started.md`](getting-started.md) → [`guide.md`](guide.md) |
| Looking up a flag or key | [`config.md`](config.md), [`fit.md`](fit.md), [`thinking.md`](thinking.md), [`media.md`](media.md), [`structured.md`](structured.md) |
| Contributor | [`../AGENTS.md`](../AGENTS.md), [`../CONTRIBUTING.md`](../CONTRIBUTING.md), [`versions.md`](versions.md) |
| Agent taking a task | [`tasks.md`](tasks.md), [`../plan.md`](../plan.md), [`../AGENTS.md`](../AGENTS.md) |

## User guides

| File | What |
|------|------|
| [`getting-started.md`](getting-started.md) | Install (release installer, Homebrew, host-tuned build), first commands, where to go next |
| [`guide.md`](guide.md) | Capability-by-capability user guide with copy-pasteable examples and real outputs (calibration, threads, `--lang`, reasoning token counts, idle shrink, offline `--recommend`, MCP quoting, chat history, split-GGUF entries, harmony tool calls, `--max-load-percent`) |

## Reference

| File | What |
|------|------|
| [`config.md`](config.md) | Every `runa.toml` key and `RUNA_*` env var, plus the search order (`~/.config/runa/config.toml`, `./runa.toml`) and the flags > env > files layering |
| [`fit.md`](fit.md) | `runa fit` internals: GGUF header path (local / HTTP range), estimator formulas, compute-buffer and speed model, calibration |
| [`thinking.md`](thinking.md) | `ThinkConfig` modes, budget and effort levels, show/hide, the cloud mapping |
| [`media.md`](media.md) | The three audio/vision routes (`native`, `asr`, cloud), video sampling, `mtmd` limits |
| [`structured.md`](structured.md) | JSON Schema and GBNF grammars, `serve` `response_format`, tool calling, MCP tool loop |
| [`dashboard.md`](dashboard.md) | Live `/dashboard` for `runa serve`: snapshot schema, websocket, quotas |
| [`profiles.md`](profiles.md) | Profiling results and the polyglot escape-hatch evaluation (K3, K4, P5.1); figure in `profiles-p49.svg` |
| [`prices.toml`](prices.toml) | User-editable cloud price table, USD per 1M tokens |
| [`versions.md`](versions.md) | What each pin maps to upstream (D16), cargo feature flags, release artifacts and install variants, and the root `ketch.toml` package manifest |
| [`runa.1`](runa.1), [`runa-run.1`](runa-run.1) | Man pages (`man ./docs/runa.1`) |
| [`site.md`](site.md) | Product card for the project site: tagline, features, install, usage examples, links |

## Engineering

| File | What |
|------|------|
| [`baselines.md`](baselines.md) | `llama-bench` reference numbers (pp512 / tg128) per model × mode × machine |
| [`perf-nightly.md`](perf-nightly.md) | The nightly perf workflow and its gate: `scripts/perf-regress.py` fails on a drop > 3 % |
| [`perf-baseline.json`](perf-baseline.json) | Machine-readable baseline that `scripts/perf-regress.py` reads and refreshes |
| [`kernels.md`](kernels.md) | `runa-kernels` benchmark results and the adoption gates (D1, D14, D23) |
| [`release-1.0.md`](release-1.0.md) | The v1.0 metric checklist with per-metric evidence |
| [`release.md`](release.md) | How a release runs: `bump.yml` (version PR, rebase merge, tag), `scripts/release.sh`, artifacts |
| [`adr/`](adr/) | One ADR per plan decision: `d01`–`d18`, `d23`. D19–D22 are recorded in [`../plan.md`](../plan.md) §1 only |
| [`notes/`](notes/) | Spike write-ups; currently `p0.5-mtmd-spike.md` (is `mtmd` reachable through the pinned `llama-cpp-2`?) |

## Coordination

| File | What |
|------|------|
| [`tasks.md`](tasks.md) | Live claim registry: `free` or `in progress` (+ agent, `started_at`). Finished work is not listed here |
| [`memory.md`](memory.md) | Public methods of `MemoryManager` and `TaskRegistry` (required for every new public method, AGENTS.md §5) |
| [`../plan.md`](../plan.md) | Decisions (D1–D23), the active-task table, and the task cards |
| [`../done.md`](../done.md) | Finished tasks, moved here whole (id, title, description) |
| [`../roadmap.md`](../roadmap.md) | Approved work that is not yet in the active plan |
| [`../todo.md`](../todo.md) | Working list in front of the plan |
| [`../ideas.md`](../ideas.md) | Unapproved ideas |
| [`../research.md`](../research.md) / [`../report.html`](../report.html) | Research analysis, analogs, formulas, fact-check ledger |
| [`../tests/fixtures/README.md`](../tests/fixtures/README.md) | Fixture inventory with sizes and licenses, the 3 GiB budget, and the cleanup rules |

## Adding or changing docs

1. English prose; update the matching `docs/ru/` and `docs/uk/` translations in the
   same change. Document what the tree actually ships — no invented APIs,
   flags, or config keys.
2. One topic per file, with a short H1 title — no `docs/<file> —` prefix and
   no plan codes; the H1 is the page title on the project site.
3. Add the new file to this index **and** to the Docs table in
   [`../README.md`](../README.md) in the same change.
4. A new or changed public method also needs a [`memory.md`](memory.md) entry
   (AGENTS.md §5); the same rule applies to any other documented surface.
5. Keep [`tasks.md`](tasks.md) valid — `moon run root:lint-tasks`
   (`python3 scripts/lint-tasks.py docs/tasks.md`) must stay green.
6. ADRs are append-only: never rewrite an accepted decision; supersede it with
   a new ADR that states what it replaces.
