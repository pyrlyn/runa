---
title: runa
tagline: CLI для ШІ, локальний насамперед — перевірте GGUF до завантаження, запускайте локально через ggml / llama.cpp або звертайтеся до OpenAI та Anthropic.
repo: https://github.com/pyrlyn/runa
install: "curl --proto '=https' --tlsv1.2 -LsSf https://github.com/pyrlyn/runa/releases/latest/download/runa-installer.sh | sh"
install_alternatives:
  - 'brew install ./runa.rb'
  - "RUSTFLAGS='-C target-cpu=native' cargo build --release --features native"
version: "0.1.0"
accent: "#F07050"
accent2: "#FFB088"
accentLight: "#E85A3C"
order: 4
lang: uk
---

<!-- Website copy for the listepo project site. The sync-docs workflow copies this file to
pyrlyn/landing (main) as content/projects/runa.md on every change to main and on every v*
tag; front matter follows CONTENT_CONTRACT.md in that repository.
Sources (checked 2026-10-01): README.md, docs/getting-started.md and docs/guide.md; version
from the latest tag (v0.1.0); accent, accent2 and accentLight are the dark-theme --runa-accent,
the dark-theme --runa-ember and the light-theme --runa-accent in docs/brand/tokens.css. -->

## Огляд

`runa` — це один бінарник CLI, що запускає моделі ШІ локально (GGUF через ggml / llama.cpp) або
через API OpenAI та Anthropic. Керування міркуванням, структурований вивід, інструменти та
`runa serve` працюють однаково з обох боків.

Ще до завантаження моделі `runa fit` скаже, чи запуститься вона на цій машині і з якою швидкістю, а
прогноз калібрується під ваш пристрій у міру запису запусків `runa bench`. Бінарники релізу —
переносні збірки для CPU під macOS arm64, Linux x86_64 і Windows x86_64; збірки для Metal, Vulkan і
CUDA — додаткові артефакти. Без телеметрії.

## Можливості

- **Перевірка до завантаження.** `runa fit` дає прогноз пам'яті та швидкості до будь-якого
  завантаження, відкалібрований реальними запусками `runa bench`. `runa fit --recommend` ранжує
  кураторський каталог під ваше залізо, зокрема офлайн.
- **Локально й у хмарі — ті самі налаштування.** Глибоке міркування (`off` / `on` / бюджет /
  зусилля), структурований вивід та інструменти працюють однаково для моделей GGUF і для API
  OpenAI та Anthropic.
- **Структурований вивід.** Відповідь можна примусити відповідати JSON Schema або граматиці GBNF;
  локальні моделі використовують семплінг за граматикою llama.cpp, тож вивід завжди відповідає.
- **Інструменти MCP.** `--mcp` запускає stdio-сервери MCP, і модель викликає їх у циклі в `run` і
  `chat`; `runa serve` підтримує виклик інструментів.
- **Медіа в циклі.** Аудіо й відео через нативний mmproj, ASR (whisper.cpp) або хмару;
  `runa media probe|video|transcribe`.
- **Сервер і теплі моделі.** HTTP-сервер, сумісний з OpenAI (`runa serve`), і сокет локального
  демона для швидких `run` / `chat`; режими обчислень `cpu` / `gpu` / `hybrid` плюс розміщення `auto`.
- **Адаптивна пам'ять.** У простої пам'ять стискається до нижньої межі, а для важкої задачі заздалегідь
  зростає в обмежених межах.

## Встановлення

З релізу на GitHub (переносні архіви для CPU; Metal / Vulkan / CUDA — додаткові артефакти):

```sh
curl --proto '=https' --tlsv1.2 -LsSf https://github.com/pyrlyn/runa/releases/latest/download/runa-installer.sh | sh
```

Або через Homebrew з формули в тому самому релізі:

```sh
brew install ./runa.rb
```

Локальна збірка, налаштована під хост:

```sh
RUSTFLAGS='-C target-cpu=native' cargo build --release --features native
```

## Приклади використання

Перевірте модель до завантаження і знайдіть найкращі моделі каталогу для цієї машини:

```sh
runa fit hf:unsloth/Qwen3-30B-A3B-GGUF:Q4_K_M --ctx 16384 --kv q8_0
runa fit --recommend --use code
runa fit --recommend --offline --top 5
```

Запуск, чат або HTTP API, сумісний з OpenAI:

```sh
runa run qwen "explain KV-cache quantization in one paragraph"
runa chat qwen --tui
runa serve --port 8080
```

Відкалібруйте прогнози швидкості виміряним запуском:

```sh
runa bench qwen2.gguf --mode cpu --pp 512 --tg 128
runa fit qwen2.gguf --json
```

Обмежте токени міркування або дайте моделі викликати інструменти MCP:

```sh
runa run qwen3-8b.gguf "What is 2+2?" --think-budget 32 --max-tokens 48 --json
runa run qwen3 "What is the weather in Paris?" --mcp 'python3 weather_server.py'
```

Транскрибуйте аудіо за допомогою whisper.cpp:

```sh
runa media transcribe --model base --lang auto clip.wav
```

## Посилання

- Репозиторій: <https://github.com/pyrlyn/runa>
- Документація: <https://github.com/pyrlyn/runa/tree/main/docs>
- Початок роботи: <https://github.com/pyrlyn/runa/blob/main/docs/getting-started.md>
- Довідник із конфігурації: <https://github.com/pyrlyn/runa/blob/main/docs/config.md>
- Релізи: <https://github.com/pyrlyn/runa/releases>
- Ліцензія: на ваш вибір GNU GPLv3, безоплатна (royalty-free) ліцензія для пропрієтарних настільних,
  мобільних і веб-застосунків (із зазначенням авторства) або комерційна ліцензія
  (див. <https://github.com/pyrlyn/runa#license>)
