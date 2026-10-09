---
lang: ru
---

# Живая панель serve

`runa serve` поднимает локальную панель на том же хосте и порту, что и API.
Откройте `http://<host>:<port>/dashboard`. Страница — собранное React-приложение
(`dashboard/`, см. `dashboard/README.md`). Цифры остаются на этой машине:
сервер никуда их не отправляет.

| Маршрут | Что |
|-------|------|
| `GET /dashboard` и `GET /dashboard/` | Страница (`text/html`) |
| `GET /dashboard/dashboard.js` | Скрипт страницы |
| `GET /dashboard/dashboard.css` | Таблица стилей |
| `GET /dashboard/snapshot` | Один JSON-снимок |
| `GET /dashboard/ws` | WebSocket. Текстовый кадр с тем же JSON примерно раз в секунду и ещё раз, когда запрос начинается, ждёт, завершается или падает |

Интерфейс один раз загружает `/dashboard/snapshot` (TanStack Query), затем
накладывает каждый кадр websocket на этот кэш. `--api-key` закрывает только `/v1`.
`/dashboard` остаётся открытым на том же адресе, чтобы браузер мог загрузить страницу.

## Квоты

Ключа конфигурации для квот нет. Два задокументированных значения по умолчанию
и семафор `--parallel` решают `quotas.hitting_ceiling`:

- `default_tokens_per_minute` равен `100000` (токены промпта и сгенерированные
  токены, завершённые за последние 60 секунд).
- `default_requests_per_second` равен `16` (старты запросов за последние 60
  секунд, делённые на 60).
- Флаг также истинен, когда заняты все параллельные слоты
  (`parallel_in_use == parallel_limit`).

Эмбеддинги в этом окне считаются токенами промпта. График скорости считает
только сгенерированные токены.

## Что видит serve

- **Демон.** `daemon.status` равен `not_this_process`, `daemon.alive` равен
  `null`. Этот процесс — `runa serve`, а не `runa daemon`, и сокет демона он
  не проверяет.
- **MCP.** `mcp_tools` — пустой массив. Serve не поднимает MCP-серверы
  (они работают в `runa run` / `runa chat`).
- **Модели.** `models.loaded[].memory_bytes` — размер весов на диске из
  `ModelPool::loaded_weight_bytes`, не RSS процесса. RSS и CPU процесса —
  это `system_load` (только этот процесс; CPU может превысить 100, когда
  занято несколько ядер).
- **Рассуждение.** `reasoning.tokens_spent` — накопленное число токенов
  рассуждения по завершённым запросам. `reasoning.active_budget_tokens` —
  сумма потолков `ThinkMode::Budget` у запросов, которые ещё в полёте, или
  `null`, если ни у одного бюджет не задан.
- **Задержка.** `latency.p50_ms`, `p95_ms` и `p99_ms` — перцентили ближайшего
  ранга по последним 512 завершённым запросам (`ceil(p * n) - 1`).
  `samples` равен 0 до первого завершения, и тогда три перцентиля тоже 0.
- **Ошибки.** `errors.count` — всего с момента старта процесса.
  `errors.recent` хранит 16 самых новых, сначала новые; у каждой есть
  `message`, `traceback` (короткий текст паники или бэктрейса) и `at_unix_ms`.
- **Очередь.** `queue.depth` — сколько запросов ждут семафор параллелизма.
  `in_flight` — те, что уже держат слот.

## Снимок

```json
{
  "token_speed": {
    "toks_per_s": 0.0,
    "decode_toks_per_s": 0.0,
    "history": [0]
  },
  "in_flight": [
    {"id": 1, "route": "/v1/chat/completions", "model": "qwen", "elapsed_ms": 12}
  ],
  "system_load": {"process_cpu_percent": 0.0, "process_rss_mib": 0},
  "latency": {"p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "samples": 0},
  "errors": {
    "count": 0,
    "recent": [{"message": "…", "traceback": "…", "at_unix_ms": 0}]
  },
  "quotas": {
    "parallel_limit": 1,
    "parallel_in_use": 0,
    "tokens_per_minute": 0,
    "requests_per_second": 0.0,
    "requests_in_window": 0,
    "window_secs": 60,
    "default_tokens_per_minute": 100000,
    "default_requests_per_second": 16.0,
    "hitting_ceiling": false
  },
  "daemon": {
    "status": "not_this_process",
    "alive": null,
    "detail": "runa serve does not host the warm daemon (that is `runa daemon`)."
  },
  "models": {
    "configured": 1,
    "loaded_count": 0,
    "loaded": [{"id": "qwen", "memory_bytes": 0}]
  },
  "mcp_tools": [],
  "queue": {
    "depth": 0,
    "waiting": [{"id": 1, "route": "/v1/chat/completions", "wait_ms": 3}]
  },
  "reasoning": {"tokens_spent": 0, "active_budget_tokens": null}
}
```

В `token_speed.history` всегда 60 чисел, сначала самые старые (в примере выше
одно `0` стоит вместо полного ряда). Каждый бакет — сгенерированные токены
за эту секунду (высота и есть tok/s за секунду). `toks_per_s` — сгенерированные
токены за всё окно в 60 с, делённые на 60. `decode_toks_per_s` — последняя
ненулевая скорость декодирования движка.
