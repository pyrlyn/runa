---
lang: uk
---

# Конфігурація: ключі `runa.toml` і змінні середовища

Порядок пошуку (пізніші файли перемагають): `~/.config/runa/config.toml`, потім `./runa.toml`.
Домашній каталог — `$HOME`, а якщо його не задано у Windows — `$USERPROFILE`. Прапорці CLI
мають перевагу над змінними середовища `RUNA_*`, а ті — над файлами. Вбудовані в TOML
API-ключі відхиляються (P3.8). Пізніші файли `[memory]` накладають лише ключі, які в них
задано; раніше задані ключі лишаються. Ціни хмарних моделей ідуть за тим самим правилом:
спочатку `docs/prices.toml` (коли поточний каталог процесу — репозиторій), потім
`~/.config/runa/prices.toml`.

## Верхній рівень

| Ключ | Значення | Env | CLI |
|-----|--------|-----|-----|
| `on_unfit` | `error` (типово) \| `cpu` \| `cloud:<backend>:<model>` | `RUNA_ON_UNFIT` | `--on-unfit` |

## `[models.<alias>]`

| Ключ | Значення |
|-----|--------|
| `source` | Локальний шлях або `hf:org/repo:quant` (розв'язується командою `runa pull`, `run` ніколи нічого не завантажує) |
| `lora` | Рядок або масив LoRA-адаптерів `path[:scale]` для цього аліасу (P8.5; додаються після `[model] lora`) |

## `[model]`

Типові значення для моделі, що запускається (P8.5).

| Ключ | Значення | CLI |
|-----|--------|-----|
| `lora` | Рядок або масив LoRA-адаптерів `path[:scale]` | `--lora` |

```toml
[model]
lora = ["adapter.gguf:0.5", "other.gguf"]
```

## `[defaults]`

Типові значення для всіх команд інференсу (P10.2).

| Ключ | Значення | Env | CLI |
|-----|--------|-----|-----|
| `threads` | робочі потоки, ціле число >= 1 (типово: P-ядра на macOS, інакше всі логічні CPU) | `RUNA_THREADS` | `--threads` |

```toml
[defaults]
threads = 8
```

## `[system]`

Обмеження частки загальних системних ресурсів, яку може використовувати цей застосунок, у відсотках
від загального обсягу (типово: 80). Застосовується до бюджету RAM (потреба моделі має
вміщатися в `total RAM × max_load_percent / 100`) і до частки потоків CPU
(`--threads` має вміщатися в `CPUs × max_load_percent / 100`). Кожна команда,
що завантажує модель (`run`, `chat`, `bench`, `serve`, `daemon`, `fit`),
перевіряє обмеження під час запуску й друкує `warning:` у stderr, коли системна
RAM уже перевищує обмеження або модель/потоки не вміщаються — кожне
попередження називає значення, яке треба встановити, щоб запуск умістився. Попередження ніколи не зривають
запуск; рішення ухвалює користувач.

| Ключ | Значення | Env | CLI |
|-----|--------|-----|-----|
| `max_load_percent` | ціле число 1..=100 (типово: 80) | `RUNA_MAX_LOAD_PERCENT` | `--max-load-percent` |

```toml
[system]
max_load_percent = 80
```

## `[mcp.servers.<name>]`

Stdio-сервери MCP, інструменти яких `runa run` / `runa chat` пропонують моделі
(P8.3). `--mcp '<command args>'` додає ще сервери на один запуск. Див. `docs/structured.md`.

| Ключ | Значення |
|-----|--------|
| `command` | Програма для запуску (обов'язково) |
| `args` | Масив рядків-аргументів |
| `env` | Таблиця додаткових змінних середовища |

```toml
[mcp.servers.fs]
command = "npx"
args = ["-y", "@modelcontextprotocol/server-filesystem", "."]
```

## `[think]`

| Ключ | Значення | Env | CLI |
|-----|--------|-----|-----|
| `mode` | `on` \| `off` \| `budget` \| `effort` | `RUNA_THINK` | `--think` |
| `budget` | ліміт токенів міркування (обов'язковий, коли `mode = "budget"`) | `RUNA_THINK_BUDGET` | `--think-budget` |
| `grace` | додаткові токени після бюджету (типово 0) | `RUNA_THINK_GRACE` | (slash-команда `/think grace`) |
| `effort` | `low` \| `medium` \| `high` \| `max` | `RUNA_EFFORT` | `--effort` |
| `show` | bool — друкувати міркування | `RUNA_SHOW_REASONING` | `--show-reasoning` / `--no-show-reasoning` |

`mode = "budget"` потребує `budget`. `--think off` не можна поєднувати з budget/effort.

## `[audio]`

| Ключ | Значення | Env | CLI |
|-----|--------|-----|-----|
| `route` | `auto` (типово) \| `native` \| `asr` | `RUNA_AUDIO_ROUTE` | `--audio-route` |

Див. `docs/media.md`.

## `[memory]`

Пізніші файли накладають лише ключі, які в них задано (проєктний `floor_mib` не
скидає `idle_timeout_s` із файлу користувача).

| Ключ | Значення | Env |
|-----|--------|-----|
| `idle_timeout_s` | секунди до стискання під час простою | `RUNA_MEMORY_IDLE_TIMEOUT_S` |
| `floor_mib` | нижня межа під час простою | `RUNA_MEMORY_FLOOR_MIB` |
| `max_growth_mib` | обмеження для `grow_for` | `RUNA_MEMORY_MAX_GROWTH_MIB` |

Див. `docs/memory.md`.

## Serve

`runa serve` типово слухає `127.0.0.1`. `--host` має бути loopback-адресою
(`127.0.0.0/8` або `::1`), якщо не задано `--api-key KEY`. З ключем кожен маршрут
`/v1` вимагає `Authorization: Bearer <key>`; `/health` лишається відкритим. Ключа
`runa.toml` для цього токена немає (вбудовані в TOML API-ключі відхиляються).

## Змінні середовища для хмари (не TOML)

| Env | Призначення |
|-----|-----|
| `OPENAI_API_KEY` | OpenAI |
| `ANTHROPIC_API_KEY` | Anthropic |
| `OPENAI_BASE_URL` | Базовий URL, сумісний з OpenAI |
