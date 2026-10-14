---
lang: uk
---

# Міркування: `ThinkConfig`

Міркування — це повноцінний окремий потік, а не домішка до тексту відповіді. Локальні та хмарні
бекенди спільно використовують `runa_core::ThinkConfig`.

## Режими

| Режим | Значення |
|------|---------|
| `off` | Без каналу міркування |
| `on` | Типове міркування моделі |
| `budget { tokens, grace }` | Жорсткий ліміт токенів міркування, потім ще `grace` додаткових |
| `effort` (`low`/`medium`/`high`/`max`) | Відображається на бюджет (таблиця P3.4) або на хмарний `reasoning.effort` |

`--think-budget` і `--effort` перевизначають `--think on`. Їх не можна поєднувати
одне з одним або з `--think off`.

## М’яке закриття локального бюджету

Ближче до кінця локального бюджету міркування рушій додає `CLOSE_LOGIT_BIAS`
(`10`) до токена закриття міркування на копії рядка логітів цього кроку і
семплює вже з копії. Буфер контексту при цьому не змінюється.

## Показ чи приховування

`ThinkConfig.show` (CLI `--show-reasoning`, конфігурація `[think] show`, змінна середовища
`RUNA_SHOW_REASONING`) друкує `GenEvent::Reasoning` / хмарні дельти міркування в
stderr. `--no-show-reasoning` примусово приховує їх. Токени відповіді лишаються в stdout.

## Інтерфейси

- CLI: `--think`, `--think-budget`, `--effort`, `--show-reasoning`
- REPL: `/think on`, `/think budget 2048 show`, `/think effort high`, `/think hide`
- Конфігурація: `[think]` у `docs/config.md`
- Хмара: OpenAI `reasoning.effort` / Responses `reasoning`; Anthropic — adaptive
  або `enabled` + бюджет (`docs/adr/d07-thinkconfig.md`)
