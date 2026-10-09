---
lang: uk
---

# Жива панель serve

`runa serve` піднімає локальну панель на тому самому хості й порту, що й API.
Відкрийте `http://<host>:<port>/dashboard`. Сторінка — зібраний React-застосунок
(`dashboard/`, див. `dashboard/README.md`). Цифри лишаються на цій машині:
сервер нікуди їх не надсилає.

| Маршрут | Що |
|-------|------|
| `GET /dashboard` і `GET /dashboard/` | Сторінка (`text/html`) |
| `GET /dashboard/dashboard.js` | Скрипт сторінки |
| `GET /dashboard/dashboard.css` | Таблиця стилів |
| `GET /dashboard/snapshot` | Один JSON-знімок |
| `GET /dashboard/ws` | WebSocket. Текстовий кадр із тим самим JSON приблизно раз на секунду і ще раз, коли запит починається, чекає, завершується або падає |

Інтерфейс один раз завантажує `/dashboard/snapshot` (TanStack Query), потім
накладає кожен кадр websocket на цей кеш. `--api-key` закриває лише `/v1`.
`/dashboard` лишається відкритим на тій самій адресі, щоб браузер міг завантажити сторінку.

## Квоти

Ключа конфігурації для квот немає. Два задокументовані типові значення
і семафор `--parallel` вирішують `quotas.hitting_ceiling`:

- `default_tokens_per_minute` дорівнює `100000` (токени промпта і згенеровані
  токени, завершені за останні 60 секунд).
- `default_requests_per_second` дорівнює `16` (старти запитів за останні 60
  секунд, поділені на 60).
- Прапорець також істинний, коли зайняті всі паралельні слоти
  (`parallel_in_use == parallel_limit`).

Ембединги в цьому вікні рахуються токенами промпта. Графік швидкості рахує
лише згенеровані токени.

## Що бачить serve

- **Демон.** `daemon.status` дорівнює `not_this_process`, `daemon.alive` дорівнює
  `null`. Цей процес — `runa serve`, а не `runa daemon`, і сокет демона він
  не перевіряє.
- **MCP.** `mcp_tools` — порожній масив. Serve не піднімає MCP-сервери
  (вони працюють у `runa run` / `runa chat`).
- **Моделі.** `models.loaded[].memory_bytes` — розмір ваг на диску з
  `ModelPool::loaded_weight_bytes`, не RSS процесу. RSS і CPU процесу —
  це `system_load` (лише цей процес; CPU може перевищити 100, коли
  зайнято кілька ядер).
- **Міркування.** `reasoning.tokens_spent` — накопичена кількість токенів
  міркування за завершеними запитами. `reasoning.active_budget_tokens` —
  сума стель `ThinkMode::Budget` у запитів, які ще в польоті, або
  `null`, якщо жоден не задав бюджет.
- **Затримка.** `latency.p50_ms`, `p95_ms` і `p99_ms` — перцентилі найближчого
  рангу за останніми 512 завершеними запитами (`ceil(p * n) - 1`).
  `samples` дорівнює 0 до першого завершення, і тоді три перцентилі теж 0.
- **Помилки.** `errors.count` — усього з моменту старту процесу.
  `errors.recent` тримає 16 найновіших, спочатку нові; у кожної є
  `message`, `traceback` (короткий текст паніки або бектрейсу) і `at_unix_ms`.
- **Черга.** `queue.depth` — скільки запитів чекають на семафор паралелізму.
  `in_flight` — ті, що вже тримають слот.

## Знімок

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

У `token_speed.history` завжди 60 чисел, спочатку найстаріші (у прикладі вище
одне `0` стоїть замість повного ряду). Кожен бакет — згенеровані токени
за цю секунду (висота і є tok/s за секунду). `toks_per_s` — згенеровані
токени за все вікно в 60 с, поділені на 60. `decode_toks_per_s` — остання
ненульова швидкість декодування рушія.
