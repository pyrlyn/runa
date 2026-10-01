---
title: runa
tagline: CLI для ИИ, локальный в первую очередь — проверьте GGUF до скачивания, запускайте локально через ggml / llama.cpp или обращайтесь к OpenAI и Anthropic.
repo: https://github.com/pyrlyn/runa
install: "curl --proto '=https' --tlsv1.2 -LsSf https://github.com/listepo/runa/releases/latest/download/runa-installer.sh | sh"
install_alternatives:
  - 'brew install ./runa.rb'
  - "RUSTFLAGS='-C target-cpu=native' cargo build --release --features native"
version: "0.1.0"
accent: "#F07050"
accent2: "#FFB088"
accentLight: "#E85A3C"
order: 4
lang: ru
---

<!-- Website copy for the listepo project site. The sync-docs workflow copies this file to
pyrlyn/landing (main) as content/projects/runa.md on every change to main and on every v*
tag; front matter follows CONTENT_CONTRACT.md in that repository.
Sources (checked 2026-10-01): README.md, docs/getting-started.md and docs/guide.md; version
from the latest tag (v0.1.0); accent, accent2 and accentLight are the dark-theme --runa-accent,
the dark-theme --runa-ember and the light-theme --runa-accent in docs/brand/tokens.css. -->

## Обзор

`runa` — это один бинарный файл CLI, который запускает модели ИИ локально (GGUF через ggml /
llama.cpp) или через API OpenAI и Anthropic. Управление рассуждением, структурированный вывод,
инструменты и `runa serve` работают одинаково в обоих случаях.

Ещё до скачивания модели `runa fit` скажет, запустится ли она на этой машине и с какой скоростью, а
прогноз подстраивается под ваше устройство по мере записи запусков `runa bench`. Бинарники релиза —
переносимые сборки для CPU под macOS arm64, Linux x86_64 и Windows x86_64; сборки для Metal, Vulkan и
CUDA — дополнительные артефакты. Без телеметрии.

## Возможности

- **Проверка до загрузки.** `runa fit` даёт прогноз памяти и скорости до любой загрузки,
  откалиброванный реальными запусками `runa bench`. `runa fit --recommend` ранжирует курируемый
  каталог под ваше железо, в том числе офлайн.
- **Локально и в облаке — одни и те же настройки.** Глубокое рассуждение (`off` / `on` / бюджет /
  усилие), структурированный вывод и инструменты работают одинаково для моделей GGUF и для API
  OpenAI и Anthropic.
- **Структурированный вывод.** Ответ можно заставить соответствовать JSON Schema или грамматике GBNF;
  локальные модели используют сэмплирование по грамматике llama.cpp, поэтому вывод всегда соответствует.
- **Инструменты MCP.** `--mcp` запускает stdio-серверы MCP, и модель вызывает их в цикле в `run` и
  `chat`; `runa serve` поддерживает вызов инструментов.
- **Медиа в цикле.** Аудио и видео через нативный mmproj, ASR (whisper.cpp) или облако;
  `runa media probe|video|transcribe`.
- **Сервер и тёплые модели.** HTTP-сервер, совместимый с OpenAI (`runa serve`), и сокет локального
  демона для быстрых `run` / `chat`; режимы вычислений `cpu` / `gpu` / `hybrid` плюс размещение `auto`.
- **Адаптивная память.** В простое память сжимается к нижней границе, а для тяжёлой задачи заранее
  растёт в заданных пределах.

## Установка

Из релиза на GitHub (переносимые архивы для CPU; Metal / Vulkan / CUDA — дополнительные артефакты):

```sh
curl --proto '=https' --tlsv1.2 -LsSf https://github.com/listepo/runa/releases/latest/download/runa-installer.sh | sh
```

Или через Homebrew из формулы в том же релизе:

```sh
brew install ./runa.rb
```

Локальная сборка, настроенная под хост:

```sh
RUSTFLAGS='-C target-cpu=native' cargo build --release --features native
```

## Примеры использования

Проверьте модель до скачивания и найдите лучшие модели каталога для этой машины:

```sh
runa fit hf:unsloth/Qwen3-30B-A3B-GGUF:Q4_K_M --ctx 16384 --kv q8_0
runa fit --recommend --use code
runa fit --recommend --offline --top 5
```

Запуск, чат или HTTP API, совместимый с OpenAI:

```sh
runa run qwen "explain KV-cache quantization in one paragraph"
runa chat qwen --tui
runa serve --port 8080
```

Откалибруйте прогнозы скорости измеренным запуском:

```sh
runa bench qwen2.gguf --mode cpu --pp 512 --tg 128
runa fit qwen2.gguf --json
```

Ограничьте токены рассуждения или дайте модели вызывать инструменты MCP:

```sh
runa run qwen3-8b.gguf "What is 2+2?" --think-budget 32 --max-tokens 48 --json
runa run qwen3 "What is the weather in Paris?" --mcp 'python3 weather_server.py'
```

Транскрибируйте аудио с помощью whisper.cpp:

```sh
runa media transcribe --model base --lang auto clip.wav
```

## Ссылки

- Репозиторий: <https://github.com/pyrlyn/runa>
- Документация: <https://github.com/pyrlyn/runa/tree/main/docs>
- Начало работы: <https://github.com/pyrlyn/runa/blob/main/docs/getting-started.md>
- Справочник по конфигурации: <https://github.com/pyrlyn/runa/blob/main/docs/config.md>
- Релизы: <https://github.com/pyrlyn/runa/releases>
- Лицензия: на ваш выбор GNU GPLv3, бесплатная (royalty-free) лицензия для проприетарных настольных,
  мобильных и веб-приложений (с указанием авторства) или коммерческая лицензия
  (см. <https://github.com/pyrlyn/runa#license>)
