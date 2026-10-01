---
lang: uk
---

# Початок роботи з runa

`runa` — це один бінарник CLI, що запускає моделі ШІ локально (GGUF через ggml / llama.cpp)
або через API OpenAI та Anthropic. Керування міркуванням, інструменти та
`runa serve` працюють однаково з обох боків.

**Репозиторій:** https://github.com/pyrlyn/runa

## Встановлення

З GitHub Release (портабельні CPU-архіви; Metal / Vulkan / CUDA — додаткові
артефакти):

```sh
curl --proto '=https' --tlsv1.2 -LsSf \
  https://github.com/listepo/runa/releases/latest/download/runa-installer.sh | sh
```

Або через Homebrew з формули в тому самому релізі: `brew install ./runa.rb`.

Локальна збірка, налаштована під хост:

```sh
RUSTFLAGS='-C target-cpu=native' cargo build --release --features native
```

Подробиці про функції та артефакти — у [`versions.md`](versions.md).

## Перші команди

Перевірте, чи модель уміститься, ще до завантаження, а потім run / serve / chat:

```sh
runa fit hf:unsloth/Qwen3-30B-A3B-GGUF:Q4_K_M --ctx 16384 --kv q8_0
runa fit --recommend --use code   # найкращі моделі з каталогу для цієї машини
runa run qwen "explain KV-cache quantization in one paragraph"
runa serve --port 8080            # HTTP-сервер, сумісний з OpenAI
runa chat qwen                    # REPL; додайте --tui для повноекранного інтерфейсу
```

`runa fit` каже, чи запуститься модель на цій машині і наскільки швидко, ще до того, як ви
її завантажите. Після кількох запусків `runa bench` прогнози калібруються під ваш
пристрій (див. [`guide.md`](guide.md)).

## Корисні наступні кроки

**Калібрування прогнозів швидкості** — кожен запуск `runa bench` дописує дані до бази
калібрування; подальші `fit` / `--recommend` масштабують прогноз за виміряною історією:

```sh
runa bench qwen2.gguf --mode cpu --pp 512 --tg 128
runa fit qwen2.gguf --json
```

**Офлайнові рекомендації** — ранжування каталогу без мережі (враховуються KV і обчислювальний буфер,
а не лише ваги):

```sh
runa fit --recommend --offline --top 5
```

**Бюджет міркування** — жорсткий ліміт токенів міркування і розподіл у JSON:

```sh
runa run qwen3-8b.gguf "What is 2+2?" --think-budget 32 --max-tokens 48 --json
```

**Структурований вивід** — JSON Schema або GBNF локально / в OpenAI / в Anthropic:

```sh
runa run qwen "Capital of France and its population?" \
  --json-schema '{"type":"object","properties":{"city":{"type":"string"},"population":{"type":"integer"}},"required":["city","population"]}'
```

**Інструменти MCP** — запустіть stdio-сервери MCP; модель може викликати їх у циклі:

```sh
runa run qwen3 "What is the weather in Paris?" --mcp 'python3 weather_server.py'
runa chat qwen3 --mcp 'npx -y @modelcontextprotocol/server-filesystem .'
```

**Транскрибування аудіо** — ASR на whisper.cpp (`--lang auto` спершу визначає мову, потім декодує):

```sh
runa media transcribe --model base --lang auto clip.wav
```

## Конфігурація

Порядок пошуку (пізніший перемагає): `~/.config/runa/config.toml`, потім `./runa.toml`.
Прапорці CLI мають перевагу над змінними середовища `RUNA_*`, а ті — над файлами. Повний список ключів:
[`config.md`](config.md).

Приклад налаштувань у `runa.toml`:

```toml
[defaults]
threads = 8

[think]
show = true

[audio]
route = "auto"
```

## Що читати далі

| Документ | Зміст |
| --- | --- |
| [guide.md](guide.md) | Інструкції за можливостями з прикладами |
| [fit.md](fit.md) | Формули fit, `--recommend`, медіаконтекст |
| [thinking.md](thinking.md) | Режими глибокого міркування та відображення на хмару |
| [structured.md](structured.md) | JSON Schema / GBNF, інструменти, MCP |
| [media.md](media.md) | Аудіо, зір, ASR |
| [memory.md](memory.md) | API адаптивної пам'яті та реєстру задач |
| [README.md](README.md) | Повний покажчик документації |

Цільова ліцензія: MIT OR Apache-2.0. Без телеметрії.
