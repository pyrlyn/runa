---
lang: ru
---

# docs/thinking.md — `ThinkConfig` (D07)

Рассуждение — это полноценный отдельный поток, а не часть текста ответа. Локальные и облачные
бэкенды используют общий `runa_core::ThinkConfig`.

## Режимы

| Режим | Значение |
|------|---------|
| `off` | Канала рассуждения нет |
| `on` | Рассуждение по умолчанию для модели |
| `budget { tokens, grace }` | Жёсткий лимит токенов рассуждения, затем ещё `grace` сверху |
| `effort` (`low`/`medium`/`high`/`max`) | Сопоставляется с бюджетом (таблица P3.4) или с облачным `reasoning.effort` |

`--think-budget` и `--effort` переопределяют `--think on`. Их нельзя сочетать
ни друг с другом, ни с `--think off`.

## Показ и скрытие

`ThinkConfig.show` (CLI `--show-reasoning`, конфигурация `[think] show`, переменная окружения
`RUNA_SHOW_REASONING`) выводит `GenEvent::Reasoning` / облачные дельты рассуждения в
stderr. `--no-show-reasoning` принудительно скрывает их. Токены ответа остаются в stdout.

## Где настраивается

- CLI: `--think`, `--think-budget`, `--effort`, `--show-reasoning`
- REPL: `/think on`, `/think budget 2048 show`, `/think effort high`, `/think hide`
- Конфигурация: `[think]` в `docs/config.md`
- Облако: OpenAI `reasoning.effort` / Responses `reasoning`; Anthropic adaptive
  или `enabled` + бюджет (`docs/adr/d07-thinkconfig.md`)
