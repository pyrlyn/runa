---
lang: uk
---

# Еталонні числа llama-bench

Виміряно 2026-09-08 на Apple M3 Max (Metal, 48GB уніфікованої пам'яті, macOS 15) за допомогою
`crates/runa-engine/examples/gen.rs` (`llama-cpp-2 =0.1.133 → llama.cpp b7709`).
`pp` = обробка промпту, `tg` = декодування. Усі запуски з `--n-gpu-layers 999` (GPU) або `0` (CPU).

> **Примітка:** P0.6 вимагає `pp512`/`tg128` на трьох еталонних моделях для кожного режиму на кожній
> машині CI. Це початковий зріз для Metal; CPU/гібридний режим і Linux/Windows
> лишаються на потім (self-hosted раннери згідно з P5.8). `gpt-oss-20b-MXFP4.gguf`
> на диску обрізаний (11G, завантажує `blk.22` за межами EOF) — потрібно завантажити повторно.

## Машина
- **M3 Max** — Apple M3 Max, 14-ядерний CPU, 30-ядерний GPU, 48GB уніфікованої пам'яті, Metal, `recommendedMaxWorkingSetSize 55662 MB`, `hasUnifiedMemory true`.

## Моделі (локальні фікстури)
- `qwen2-0_5b-instruct-q4_0.gguf` — 337M, Qwen2-0.5B-Instruct Q4_0
- `Qwen3-8B-Q4_K_M.gguf` — 4.7G, Qwen3-8B Q4_K_M (unsloth)
- `Qwen3-30B-A3B-Q4_K_M.gguf` — 11G, Qwen3-30B-A3B Q4_K_M (MoE, 3.3B активних)
- `gpt-oss-20b-MXFP4.gguf` — 11G, gpt-oss-20B MXFP4 (обрізаний на диску, не вимірювався)
- `SmolVLM-500M-Instruct-Q8_0.gguf` — 417M (зір, не входить до P0.6, але присутній)

## Результати (pp/tg)

| Модель | Режим | Машина | pp (tok/s) | tg (tok/s) | Примітки |
|-------|------|---------|------------|------------|-------|
| Qwen2-0.5B Q4_0 | gpu (Metal, усі 24 шари) | M3 Max | 4.4 (pp1) | 344.4 (tg32) | `gen`: промпт 1 токен, генерація 32 токени; ~0.23s pp, 0.09s tg |
| Qwen3-8B Q4_K_M | gpu (Metal, усі шари) | M3 Max | 5.8 (pp1) | 50.4 (tg16) | промпт 1 токен, генерація 16 токенів; 0.17s pp, 0.32s tg |
| Qwen3-8B Q4_K_M | cpu (n_gpu_layers 0) | M3 Max | — | — | очікується: `cargo run --no-default-features`, лише CPU |
| Qwen3-8B Q4_K_M | гібридний (експерти на CPU) | M3 Max | — | — | N/A (щільна модель, не MoE) |
| Qwen3-30B-A3B Q4_K_M | gpu (Metal, усі 48 шарів, 3.3B активних) | M3 Max | 4.2 (pp1) | 54.7 (tg16) | промпт 1 токен, генерація 16 токенів; 0.24s pp, 0.29s tg |
| Qwen3-30B-A3B Q4_K_M | гібридний (експерти на CPU) | M3 Max | 3.1 (pp9) | 1.6 (tg16) | P2.6: `runa run --mode hybrid`; `ffn_*_exps` → CPU_REPACK (12.96 GiB); attention лишається на Metal |
| gpt-oss-20b MXFP4 | gpu | M3 Max | — | — | файл обрізаний, потрібне повторне завантаження |
| gpt-oss-20b MXFP4 | cpu/гібридний | M3 Max | — | — | очікується |

## Експерти на GPU чи на CPU (P2.6)

Та сама модель, та сама машина, debug-збірка `runa`. Числа для GPU — зі спайку P0.6 `gen`
(`-ngl 999`); гібридний режим — це `--mode hybrid` (усі маршрутизовані експерти на CPU).

| Модель | Експерти | pp (tok/s) | tg (tok/s) | Примітки |
|-------|---------|------------|------------|-------|
| Qwen3-30B-A3B Q4_K_M | GPU (Metal, усі 48 шарів) | 4.2 (pp1) | 54.7 (tg16) | 17 GiB відображено на Metal |
| Qwen3-30B-A3B Q4_K_M | CPU (`--mode hybrid` / `--n-cpu-moe ≥48`) | 3.1 (pp9) | 1.6 (tg16) | `ffn_*_exps` → CPU_REPACK 12.96 GiB; рецепт для VRAM класу 8 GB |

`--n-cpu-moe N` тримає експертів шарів `0..N-1` на CPU (один об'єднаний регулярний вираз;
llama-cpp-2 0.1.133 може застосувати лише одне перевизначення буфера). `--mode hybrid`
закріплює кожен тензор експертів. Щільні моделі ігнорують ці шаблони.

```sh
cargo run -p runa -- run --mode hybrid --max-tokens 16 --ctx 512 --temperature 0 --seed 42 \
  tests/fixtures/Qwen3-30B-A3B-Q4_K_M.gguf hello
cargo run -p runa -- run --mode gpu --n-cpu-moe 8 --max-tokens 16 --ctx 512 \
  tests/fixtures/Qwen3-30B-A3B-Q4_K_M.gguf hello
```

## Як відтворити (спайк P0.4)

```sh
cargo run -p runa-engine --example gen -- tests/fixtures/qwen2-0_5b-instruct-q4_0.gguf "hi" 32
cargo run -p runa-engine --example gen -- tests/fixtures/Qwen3-8B-Q4_K_M.gguf "hello" 16
cargo run -p runa-engine --example gen -- tests/fixtures/Qwen3-30B-A3B-Q4_K_M.gguf "hello" 16
# pp512/tg128 (повний P0.6): використайте промпт на 512 токенів (напр. `python3 -c "print('hello world '*200)"`) і `... 128`
```

## Кілька GPU, P2.9 (вручну)

Раннери CI тут мають один GPU (Apple M3 Max Metal). Тест завантаження на двох GPU позначено
`#[ignore]` (`two_gpu_tensor_split_loads` у тестах `runa-engine`).

На машині з двома GPU-бекендами ggml:

```
runa run --mode gpu --device 0,1 --tensor-split 3,1 model.gguf "hi"
```

Очікуйте, що рядок вердикту міститиме `devices=0,1 tensor-split=3,1`, а потік пройде
успішно. `runa run --device 999 …` має завершитися помилкою (жодного мовчазного запасного варіанта).

## Вимірювання бар'єрів P8.7 (2026-09-15, M3 Max 64GB, macOS 26.6.2, runa 0.1.0 debug, лише CPU)

Машина під навантаженням (сусідні агенти виконували збірки; load avg 127–224). Рядки із зірочкою (*) потребують повторного запуску в тиші до 1.0.
llama-bench зібрано з тих самих вихідних кодів b7709 (`--branch b7709`, Release лише для CPU).

| Бар'єр | Результат |
|------|--------|
| M1, час fit (заголовок+розбір+`check_fit`, найкращий із 5) | 0.5B **148.2 ms**, 8B **146.6 ms** (< 300 ms ✓) |
| M1, оцінка ваг проти mmap рушія (8B) | оцінка 5,021,827,072 B проти відображених 4762.19 MiB (**+0.57 %** ✓); KV точно (1152 MiB ≡ 288 MiB@2k×4); оцінка compute у 3.74× більша за фактичну (консервативно) |
| M2, віддалений `hf:unsloth/Qwen3-8B-GGUF:Q4_K_M`, діапазон 8 MiB | **0.57 s** тепло ✓, 206 partial (без завантаження файлу) ✓; холодно 26.1 s ✗ |
| M3, прогноз для CPU 0.5B проти виміру | прогноз 62.9/62.9 проти виміряних pp 297–478 / tg 17–23 (FAIL ±30 %); калібрування записано, предиктор ігнорує базу |
| M4, 0.5B CPU pp512/tg128 проти llama-bench | runa **297.8/23.1** проти bench **1617.6/85.1** @t12 (18 %/27 %); bench `-t16`: 175.5/10.1 (за однакових умов виграє runa) — причина в типовому threads=16 |
| M5, холодний TTFT 8B на CPU* | **9.02 s** (усього 9.17 s / 8 токенів; pp 12.2, tg 2.0; пікова RSS 8.44 GB) |
| M6, 10× `--think-budget 64` Qwen3-8B | 10/10 завершено, ідентичні жадібні виводи по 417 B; кількість токенів міркування через CLI не видно |
| M7, 60 s справжнього мовлення, `--lang en`* | правильно (14 сегментів), **11.5 s** реального часу; `--lang auto` (типово) повертає порожній результат — баг |
| M8, кліп 30 s, `media video --fps 1` | **30 кадрів** ✓, аудіо ✓, 2.6 s*; транскрипт — неявно ~6 s* проти бар'єра <3 s |
| M9, смоук-тести SDK | openai 2.54.0 + anthropic 0.125.0 проходять без змін (потік + інструменти) |
| M10, розмір | debug 104,649,208 B; release очікується; grep на відсутність телеметрії чистий |
| M11, RSS serve (8B ctx2048 / 0.5B) | **8364 MB** / 846 MB проти нижньої межі+10 % = 563 MB; не змінюється під час простою (не під'єднано) |

## Вимірювання P11.2 у тиші (2026-09-16, M3 Max 64GB, macOS 26.6.2, runa 0.1.0 release, лише CPU)

Release зібрано через `cargo build --offline --release -p runa` за 3m03s
(`llama-cpp-2 0.1.133 → llama.cpp b7709`; портабельна збірка, бекенди: cpu).
Машина була відносно тихою під час кожного запуску нижче (1-хвилинне навантаження 3.7–9.2; пор. P8.7 під навантаженням
127–224), тож рядків із `*` немає. Застереження: без sudo для `purge`, тож mmap для M5 був
теплим — це не холодне число.

| Бар'єр | Результат |
|------|--------|
| M10, розмір | release **24,662,064 B (23.5 MiB)** ≤ 40 MB ✓; debug 121,900,776 B (116.3 MiB) |
| M5, TTFT 8B на CPU (release, `--mode cpu`, threads 12, `--no-prompt-cache`, теплий mmap) | від запуску до першого токена **≈1.2–1.3 s** (граф готовий за 1.05–1.07 s + pp 14–15 токенів @ 83–96 tok/s); 8 токенів усього за 1.79–1.91 s; pp 79–110, tg 16–19; пікова RSS 9,645,719,552 B (8.98 GiB). Бар'єр (<2 s) — для холодного Metal, ще відкритий |
| M7, 57.8 s синтезованого англійського мовлення (`say` Samantha), base, `--lang en` | **1.02 s** реального часу ✓ (<5 s), транскрипт точний (14/14 повторів); `--lang auto` побайтово ідентичний (md5 збігається) |
| M8, синтетичний кліп 30 s (320×240 testsrc + мовлення AAC) | `media video --fps 1`: **30 кадрів** 336×336 + аудіо ✓ за **1.01 s**; видобування 30 s аудіо + транскрибування **0.59 s**, транскрипт правильний → разом ≈1.6 s ✓ (<3 s) |
| M4, 8B CPU `bench --mode cpu pp512/tg128` (release) | pp **144.1** tok/s, tg **16.4** tok/s, 12.7 s реального часу. Порівняння з llama-bench заблоковане: бінарника немає в PATH, вендорений llama.cpp (реєстр `llama-cpp-sys-2-0.1.133`) не містить `tools/llama-bench`, crates.io недоступний |
| Збірка Metal `--features metal` (debug) | `cargo check` 19.9 s ✓ (лише 2 наявні раніше попередження `dead_code` у `runa/src/main.rs`); `cargo build` 26.3 s, бекенди `cpu, metal`, 121,812,440 B |
| Декодування 0.5B на Metal (debug, `--mode gpu`, тихе навантаження ~4.4) | pp **1035.1** tok/s, tg **235.3** tok/s (9 токенів промпту + 9 згенерованих, 0.48 s реального часу, справжній текст) |
| 8B на Metal (debug, `--mode gpu`, теплий mmap, тихе навантаження ~4.5) | 8 токенів усього за **1.79 s**; pp 97.3, tg 20.4; перший токен ≈1.35 s (обчислено); 7/8 міркування + порожній текст (те саме спостереження щодо бюджету міркування, що й на CPU). Холодний запуск release на Metal для бар'єра M5 ще очікується |

Спостереження (без змін у коді): типовий режим міркування витрачає весь бюджет
на міркування на Qwen3-8B (`--max-tokens 8` → 7 міркування + порожній текст;
`--max-tokens 64` → 63 міркування + порожній текст), навіть із `--think off`.
Тому `run` у пайпі за коротких бюджетів не друкує видимого тексту; TTFT
вище виміряно до першого згенерованого токена.

Відтворення (з кореня репозиторію; фікстури в git-ignore, мовлення/кліп синтезуються в `$TMPDIR`):

```sh
cargo build --offline --release -p runa  # 3m03s; 24,662,064 B
say -v Samantha -o speech.aiff -f speech.txt && afconvert -f WAVE -d LEI16@16000 -c 1 speech.aiff speech-60s.wav
ffmpeg -f lavfi -i testsrc=size=320x240:rate=30:duration=30 -stream_loop 1 -i speech-60s.wav -shortest -t 30 clip-30s.mp4
./target/release/runa run --mode cpu --no-prompt-cache --max-tokens 8 --temperature 0 --seed 7 --json tests/fixtures/Qwen3-8B-Q4_K_M.gguf "Name exactly three primary colors."
./target/release/runa media transcribe --model base --lang en --no-pull speech-60s.wav
./target/release/runa media video --fps 1 clip-30s.mp4
./target/release/runa bench --mode cpu --pp 512 --tg 128 --no-calibrate tests/fixtures/Qwen3-8B-Q4_K_M.gguf
```

## ASR, P4.2 (вручну)

Фікстури CI — це синусоїдальні тони тривалістю 1 s, а не мовлення, тож WER відносно транскрипту
не визначений. На машині з `ggml-base.bin` (автоматично завантажується
`runa media transcribe`):

```
runa media transcribe --model base speech.wav
# 1 хвилина мовлення на CPU має оброблятися менш ніж за 5 s (M7).
```

Допоміжна функція для WER: `runa_media::word_error_rate`. Перевірка наживо: `RUNA_WHISPER=1`.

## Примусове обмеження бюджету, P3.3 (вручну)

У CI немає фікстури Qwen3-4B. Перевірку точності на 100 запусках / GSM8K пропущено.
На машині з Qwen3-4B:

```
runa run --think-budget 256 --grace 64 Qwen3-4B.gguf "Solve 12+7"
```

Токени міркування мають лишатися ≤ 256+grace. Повторіть ~100 разів; планка якості —
GSM8K-50 у межах 10 пунктів від необмеженого міркування.

## Далі
- Повторно завантажити `gpt-oss-20b-MXFP4.gguf` (11G, обрізаний) і перезапустити.
- Додати `pp512`/`tg128` із промптом на 512 токенів для кожної моделі/режиму.
- Додати варіанти для CPU (`-ngl 0`). Гібридний `--n-cpu-moe` / `--mode hybrid` уже є (P2.6).
- Додати Linux x86_64 (CUDA/Vulkan) і Windows із CI. Нічний `runa bench` на CPU — це P5.8 (`docs/perf-nightly.md`); self-hosted раннери з GPU ще очікуються.
