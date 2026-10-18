---
lang: uk
---

# Статистика використання

`run`, `chat` і `serve` дописують рядок до локального журналу, коли
генерація завершується. `runa stats` друкує цей журнал. `runa web`
показує ті самі цифри в браузері й оновлює їх, коли з'являються нові рядки.

Журнал лишається на цій машині. Жодна з команд нікуди його не надсилає.

## Журнал

Типовий шлях: `$XDG_CACHE_HOME/runa/usage.jsonl` або
`~/.cache/runa/usage.jsonl`, якщо `XDG_CACHE_HOME` не задано. У Windows без
`HOME` — `%LOCALAPPDATA%/runa/usage.jsonl`.

`RUNA_USAGE_LOG` замінює цей шлях. Ключа в `runa.toml` немає.

Кожен рядок — один JSON-об'єкт:

```json
{"ts":1700000000,"source":"run","model":"qwen","prompt_tokens":15,"generated_tokens":48,"reasoning_tokens":35,"pp_toks_per_s":77.1,"tg_toks_per_s":15.5}
```

`source` — це `run`, `chat` або `serve` (включно з embeddings). Рядок, який
не є коректним JSON, пропускається. Швидкість декодування — середнє,
зважене за токенами: рядки з `tg_toks_per_s` рівним 0 (сокет демона швидкість
не повідомляє) середнє не занижують.

## `runa stats`

```sh
runa stats
runa stats --since 7d
runa stats --since 24h --json
```

`--since` приймає `Nh` або `Nd` (`24h`, `7d`, `60d`). Текстова таблиця
показує підсумки, джерела, моделі, дні та 20 останніх рядків. `--json`
друкує той самий знімок, який завантажує вебсторінка.

## `runa web`

```sh
runa web
runa web --port 3939
```

Слухає `127.0.0.1:3939`, якщо не передано `--host` і `--port`. `--host`
має бути loopback-адресою (`127.0.0.0/8` або `::1`). Процес друкує
`usage on http://127.0.0.1:3939` і віддає:

| Маршрут | Що |
|---------|-----|
| `GET /` | Сторінка |
| `GET /stats.js` і `GET /stats.css` | Ресурси сторінки |
| `GET /api/snapshot` | Один JSON-знімок (та сама форма, що в `runa stats --json`) |
| `GET /ws` | WebSocket. Текстовий кадр із цим JSON, коли журнал змінюється, і приблизно раз на секунду |
| `GET /health` | `{"ok":true}` |

Сторінка — зібраний React-застосунок із `web/` (Vite, TanStack Query).
Він один раз завантажує `/api/snapshot`, потім накладає кожен кадр
websocket на цей кеш. Кольори — бренд runa: теплий попіл, вугілля, жарина.

У `runa serve` і далі своя жива панель процесу на `/dashboard`
([`dashboard.md`](dashboard.md)). Це кільця в пам'яті сервера. `runa web` —
журнал по `run`, `chat` і `serve`.
