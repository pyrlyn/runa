# Thinking: `ThinkConfig`

Thinking is a first-class stream, not mixed into answer text. Local and cloud
backends share `runa_core::ThinkConfig`.

## Modes

| Mode | Meaning |
|------|---------|
| `off` | No reasoning channel |
| `on` | Model default thinking |
| `budget { tokens, grace }` | Hard cap on reasoning tokens, then `grace` extra |
| `effort` (`low`/`medium`/`high`/`max`) | Mapped to a budget (P3.4 table) or a cloud `reasoning.effort` |

`--think-budget` and `--effort` override `--think on`. They cannot be combined
with each other or with `--think off`.

## Show vs hide

`ThinkConfig.show` (CLI `--show-reasoning`, config `[think] show`, env
`RUNA_SHOW_REASONING`) prints `GenEvent::Reasoning` / cloud reasoning deltas on
stderr. `--no-show-reasoning` forces hide. Answer tokens stay on stdout.

## Surfaces

- CLI: `--think`, `--think-budget`, `--effort`, `--show-reasoning`
- REPL: `/think on`, `/think budget 2048 show`, `/think effort high`, `/think hide`
- Config: `[think]` in `docs/config.md`
- Cloud: OpenAI `reasoning.effort` / Responses `reasoning`; Anthropic adaptive
  vs `enabled` + budget (`docs/adr/d07-thinkconfig.md`)
