---
lang: ru
---

# Конфигурация: `runa.toml` и переменные окружения

Порядок поиска (более поздние файлы побеждают): `~/.config/runa/config.toml`, затем `./runa.toml`.
Флаги CLI важнее переменных окружения `RUNA_*`, а те важнее файлов. Ключи API, вписанные прямо в TOML,
отклоняются (P3.8).

## Верхний уровень

| Ключ | Значения | Переменная окружения | CLI |
|-----|--------|-----|-----|
| `on_unfit` | `error` (по умолчанию) \| `cpu` \| `cloud:<backend>:<model>` | `RUNA_ON_UNFIT` | `--on-unfit` |

## `[models.<alias>]`

| Ключ | Значения |
|-----|--------|
| `source` | Локальный путь или `hf:org/repo:quant` (разрешается командой `runa pull`, `run` никогда его не скачивает) |
| `lora` | Строка или массив LoRA-адаптеров `path[:scale]` для этого псевдонима (P8.5; добавляются после `[model] lora`) |

## `[model]`

Значения по умолчанию для любой запускаемой модели (P8.5).

| Ключ | Значения | CLI |
|-----|--------|-----|
| `lora` | Строка или массив LoRA-адаптеров `path[:scale]` | `--lora` |

```toml
[model]
lora = ["adapter.gguf:0.5", "other.gguf"]
```

## `[defaults]`

Значения по умолчанию для всего запуска команд инференса (P10.2).

| Ключ | Значения | Переменная окружения | CLI |
|-----|--------|-----|-----|
| `threads` | рабочие потоки, целое число >= 1 (по умолчанию: P-ядра на macOS, иначе все логические CPU) | `RUNA_THREADS` | `--threads` |

```toml
[defaults]
threads = 8
```

## `[system]`

Ограничение доли общих ресурсов системы, которую может использовать это приложение, в процентах
от общего объёма (по умолчанию: 80). Применяется к бюджету RAM (потребность модели должна
укладываться в `total RAM × max_load_percent / 100`) и к доле потоков CPU
(`--threads` должно укладываться в `CPUs × max_load_percent / 100`). Каждая команда,
которая загружает модель (`run`, `chat`, `bench`, `serve`, `daemon`, `fit`),
проверяет ограничение при запуске и выводит `warning:` в stderr, если системная
RAM уже превышает ограничение или модель/потоки не укладываются; в каждом
предупреждении указано значение, которое нужно задать, чтобы запуск уложился. Предупреждения никогда не
прерывают запуск; решает пользователь.

| Ключ | Значения | Переменная окружения | CLI |
|-----|--------|-----|-----|
| `max_load_percent` | целое число 1..=100 (по умолчанию: 80) | `RUNA_MAX_LOAD_PERCENT` | `--max-load-percent` |

```toml
[system]
max_load_percent = 80
```

## `[mcp.servers.<name>]`

Stdio-серверы MCP, инструменты которых `runa run` / `runa chat` предлагают модели
(P8.3). `--mcp '<command args>'` добавляет ещё серверы на один запуск. См. `docs/structured.md`.

| Ключ | Значения |
|-----|--------|
| `command` | Запускаемая программа (обязательно) |
| `args` | Массив строк-аргументов |
| `env` | Таблица дополнительных переменных окружения |

```toml
[mcp.servers.fs]
command = "npx"
args = ["-y", "@modelcontextprotocol/server-filesystem", "."]
```

## `[think]`

| Ключ | Значения | Переменная окружения | CLI |
|-----|--------|-----|-----|
| `mode` | `on` \| `off` \| `budget` \| `effort` | `RUNA_THINK` | `--think` |
| `budget` | лимит токенов рассуждения (обязателен при `mode = "budget"`) | `RUNA_THINK_BUDGET` | `--think-budget` |
| `grace` | дополнительные токены после бюджета (по умолчанию 0) | `RUNA_THINK_GRACE` | (slash-команда `/think grace`) |
| `effort` | `low` \| `medium` \| `high` \| `max` | `RUNA_EFFORT` | `--effort` |
| `show` | bool — выводить рассуждение | `RUNA_SHOW_REASONING` | `--show-reasoning` / `--no-show-reasoning` |

`mode = "budget"` требует `budget`. `--think off` нельзя сочетать с budget/effort.

## `[audio]`

| Ключ | Значения | Переменная окружения | CLI |
|-----|--------|-----|-----|
| `route` | `auto` (по умолчанию) \| `native` \| `asr` | `RUNA_AUDIO_ROUTE` | `--audio-route` |

См. `docs/media.md`.

## `[memory]`

| Ключ | Значения | Переменная окружения |
|-----|--------|-----|
| `idle_timeout_s` | секунды до сжатия в простое | `RUNA_MEMORY_IDLE_TIMEOUT_S` |
| `floor_mib` | нижняя граница в простое | `RUNA_MEMORY_FLOOR_MIB` |
| `max_growth_mib` | ограничение для `grow_for` | `RUNA_MEMORY_MAX_GROWTH_MIB` |

См. `docs/memory.md`.

## Облачные переменные окружения (не TOML)

| Переменная окружения | Назначение |
|-----|-----|
| `OPENAI_API_KEY` | OpenAI |
| `ANTHROPIC_API_KEY` | Anthropic |
| `OPENAI_BASE_URL` | Базовый URL для OpenAI-совместимого API |
