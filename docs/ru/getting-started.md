---
lang: ru
---

# Начало работы с runa

`runa` — это один бинарный файл CLI, который запускает модели ИИ локально (GGUF через ggml / llama.cpp)
или через API OpenAI и Anthropic. Управление рассуждением, инструменты и
`runa serve` работают одинаково в обоих случаях.

**Репозиторий:** https://github.com/pyrlyn/runa

## Установка

Из GitHub Release (переносимые архивы для CPU; Metal / Vulkan / CUDA — отдельные
артефакты):

```sh
curl --proto '=https' --tlsv1.2 -LsSf \
  https://github.com/pyrlyn/runa/releases/latest/download/runa-installer.sh | sh
```

Или через Homebrew из формулы в том же релизе: `brew install ./runa.rb`.

Локальная сборка под конкретный хост:

```sh
RUSTFLAGS='-C target-cpu=native' cargo build --release --features native
```

Подробности о фичах и артефактах — в [`versions.md`](versions.md).

## Первые команды

Проверьте, поместится ли модель, ещё до скачивания, затем run / serve / chat:

```sh
runa fit hf:unsloth/Qwen3-30B-A3B-GGUF:Q4_K_M --ctx 16384 --kv q8_0
runa fit --recommend --use code   # лучшие модели каталога для этой машины
runa run qwen "explain KV-cache quantization in one paragraph"
runa serve --port 8080            # OpenAI-совместимый HTTP-сервер (loopback)
runa chat qwen                    # REPL; добавьте --tui для полноэкранного интерфейса
```

`runa fit` сообщает, запустится ли модель на этой машине и насколько быстро, ещё до того,
как вы её скачаете. После нескольких запусков `runa bench` прогнозы калибруются под ваше
устройство (см. [`guide.md`](guide.md)). `runa serve` слушает `127.0.0.1`,
если вместе с не-loopback `--host` не передан `--api-key`.

## Полезные следующие шаги

**Калибровка прогнозов скорости** — каждый `runa bench` дописывает результат в базу
калибровки; последующие `fit` / `--recommend` масштабируются по истории измерений:

```sh
runa bench qwen2.gguf --mode cpu --pp 512 --tg 128
runa fit qwen2.gguf --json
```

**Офлайн-рекомендации** — ранжирование каталога без сети (учитываются KV и вычислительный буфер,
а не только веса):

```sh
runa fit --recommend --offline --top 5
```

**Бюджет рассуждения** — жёсткий лимит токенов рассуждения и разбивка в JSON:

```sh
runa run qwen3-8b.gguf "What is 2+2?" --think-budget 32 --max-tokens 48 --json
```

**Структурированный вывод** — JSON Schema или GBNF для локальных моделей / OpenAI / Anthropic:

```sh
runa run qwen "Capital of France and its population?" \
  --json-schema '{"type":"object","properties":{"city":{"type":"string"},"population":{"type":"integer"}},"required":["city","population"]}'
```

**Инструменты MCP** — запуск stdio-серверов MCP; модель может вызывать их в цикле:

```sh
runa run qwen3 "What is the weather in Paris?" --mcp 'python3 weather_server.py'
runa chat qwen3 --mcp 'npx -y @modelcontextprotocol/server-filesystem .'
```

**Транскрибация аудио** — ASR на whisper.cpp (`--lang auto` сначала определяет язык, затем декодирует):

```sh
runa media transcribe --model base --lang auto clip.wav
```

## Конфигурация

Порядок поиска (более поздний побеждает): `~/.config/runa/config.toml`, затем `./runa.toml`.
Флаги CLI важнее переменных окружения `RUNA_*`, а те важнее файлов. Полный список ключей:
[`config.md`](config.md).

Пример настроек в `runa.toml`:

```toml
[defaults]
threads = 8

[think]
show = true

[audio]
route = "auto"
```

## Что читать дальше

| Документ | Содержание |
| --- | --- |
| [guide.md](guide.md) | Практическое руководство по возможностям с примерами |
| [fit.md](fit.md) | Формулы fit, `--recommend`, медиа в контексте |
| [thinking.md](thinking.md) | Режимы глубокого рассуждения и сопоставление для облака |
| [structured.md](structured.md) | JSON Schema / GBNF, инструменты, MCP |
| [media.md](media.md) | Аудио, зрение, ASR |
| [memory.md](memory.md) | API адаптивной памяти и реестра задач |
| [README.md](README.md) | Полный указатель документации |

Лицензия: на ваш выбор GPL-3.0-or-later, royalty-free или коммерческая (см. README). Поле `license` в Cargo называет только `GPL-3.0-or-later` — единственный из трёх вариантов с идентификатором SPDX; crates.io отклоняет имена `LicenseRef`.
Никакой телеметрии.
