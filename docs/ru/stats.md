---
lang: ru
---

# Статистика использования

`run`, `chat` и `serve` дописывают строку в локальный журнал, когда
генерация завершается. `runa stats` печатает этот журнал. `runa web`
показывает те же цифры в браузере и обновляет их, когда появляются новые
строки.

Журнал остаётся на этой машине. Ни одна из команд никуда его не отправляет.

## Журнал

Путь по умолчанию: `$XDG_CACHE_HOME/runa/usage.jsonl` или
`~/.cache/runa/usage.jsonl`, если `XDG_CACHE_HOME` не задан. В Windows без
`HOME` — `%LOCALAPPDATA%/runa/usage.jsonl`.

`RUNA_USAGE_LOG` заменяет этот путь. Ключа в `runa.toml` нет.

Каждая строка — один JSON-объект:

```json
{"ts":1700000000,"source":"run","model":"qwen","prompt_tokens":15,"generated_tokens":48,"reasoning_tokens":35,"pp_toks_per_s":77.1,"tg_toks_per_s":15.5}
```

`source` — это `run`, `chat` или `serve` (включая embeddings). Строка,
которая не является корректным JSON, пропускается. Скорость декодирования —
среднее, взвешенное по токенам: строки с `tg_toks_per_s` равным 0 (сокет
демона скорость не сообщает) среднее не занижают.

## `runa stats`

```sh
runa stats
runa stats --since 7d
runa stats --since 24h --json
```

`--since` принимает `Nh` или `Nd` (`24h`, `7d`, `60d`). Текстовая таблица
показывает итоги, источники, модели, дни и 20 последних строк. `--json`
печатает тот же снимок, который загружает веб-страница.

## `runa web`

```sh
runa web
runa web --port 3939
```

Слушает `127.0.0.1:3939`, если не переданы `--host` и `--port`. `--host`
должен быть loopback-адресом (`127.0.0.0/8` или `::1`). Процесс печатает
`usage on http://127.0.0.1:3939` и отдаёт:

| Маршрут | Что |
|---------|-----|
| `GET /` | Страница |
| `GET /stats.js` и `GET /stats.css` | Ресурсы страницы |
| `GET /api/snapshot` | Один JSON-снимок (та же форма, что у `runa stats --json`) |
| `GET /ws` | WebSocket. Текстовый кадр с этим JSON, когда журнал меняется, и примерно раз в секунду |
| `GET /health` | `{"ok":true}` |

Страница — собранное React-приложение из `web/` (Vite, TanStack Query).
Оно один раз загружает `/api/snapshot`, затем накладывает каждый кадр
websocket на этот кэш. Цвета — бренд runa: тёплый пепел, уголь, уголёк.

У `runa serve` по-прежнему своя живая панель процесса на `/dashboard`
([`dashboard.md`](dashboard.md)). Это кольца в памяти сервера. `runa web` —
журнал по `run`, `chat` и `serve`.
