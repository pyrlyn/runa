---
lang: ru
---

# Эталонные числа llama-bench

Измерено 2026-09-08 на Apple M3 Max (Metal, 48GB объединённой памяти, macOS 15) с помощью
`crates/runa-engine/examples/gen.rs` (`llama-cpp-2 =0.1.133 → llama.cpp b7709`).
`pp` = обработка промпта, `tg` = декодирование. Все запуски с `--n-gpu-layers 999` (GPU) или `0` (CPU).

> **Примечание:** P0.6 требует `pp512`/`tg128` на трёх эталонных моделях для каждого режима на каждой CI-
> машине. Это первый срез для Metal; CPU/гибрид и Linux/Windows
> остаются на потом (self-hosted раннеры согласно P5.8). `gpt-oss-20b-MXFP4.gguf`
> на диске обрезан (11G, загрузка `blk.22` выходит за EOF) — нужно скачать заново.

## Машина
- **M3 Max** — Apple M3 Max, 14-ядерный CPU, 30-ядерный GPU, 48GB объединённой памяти, Metal, `recommendedMaxWorkingSetSize 55662 MB`, `hasUnifiedMemory true`.

## Модели (локальные фикстуры)
- `qwen2-0_5b-instruct-q4_0.gguf` — 337M, Qwen2-0.5B-Instruct Q4_0
- `Qwen3-8B-Q4_K_M.gguf` — 4.7G, Qwen3-8B Q4_K_M (unsloth)
- `Qwen3-30B-A3B-Q4_K_M.gguf` — 11G, Qwen3-30B-A3B Q4_K_M (MoE, 3.3B активных)
- `gpt-oss-20b-MXFP4.gguf` — 11G, gpt-oss-20B MXFP4 (обрезан на диске, не измерялся)
- `SmolVLM-500M-Instruct-Q8_0.gguf` — 417M (зрение, не входит в P0.6, но присутствует)

## Результаты (pp/tg)

| Модель | Режим | Машина | pp (tok/s) | tg (tok/s) | Примечания |
|-------|------|---------|------------|------------|-------|
| Qwen2-0.5B Q4_0 | gpu (Metal, все 24 слоя) | M3 Max | 4.4 (pp1) | 344.4 (tg32) | `gen`: промпт 1 ток., генерация 32 ток.; ~0.23s pp, 0.09s tg |
| Qwen3-8B Q4_K_M | gpu (Metal, все слои) | M3 Max | 5.8 (pp1) | 50.4 (tg16) | промпт 1 ток., генерация 16 ток.; 0.17s pp, 0.32s tg |
| Qwen3-8B Q4_K_M | cpu (n_gpu_layers 0) | M3 Max | — | — | ожидается: `cargo run --no-default-features`, только CPU |
| Qwen3-8B Q4_K_M | гибрид (эксперты на CPU) | M3 Max | — | — | неприменимо (плотная модель, не MoE) |
| Qwen3-30B-A3B Q4_K_M | gpu (Metal, все 48 слоёв, 3.3B активных) | M3 Max | 4.2 (pp1) | 54.7 (tg16) | промпт 1 ток., генерация 16 ток.; 0.24s pp, 0.29s tg |
| Qwen3-30B-A3B Q4_K_M | гибрид (эксперты на CPU) | M3 Max | 3.1 (pp9) | 1.6 (tg16) | P2.6: `runa run --mode hybrid`; `ffn_*_exps` → CPU_REPACK (12.96 GiB); attention остаётся на Metal |
| gpt-oss-20b MXFP4 | gpu | M3 Max | — | — | файл обрезан, нужно скачать заново |
| gpt-oss-20b MXFP4 | cpu/гибрид | M3 Max | — | — | ожидается |

## Эксперты на GPU против CPU (P2.6)

Та же модель, та же машина, отладочная сборка `runa`. Числа для GPU — из спайка P0.6 `gen`
(`-ngl 999`); гибрид — это `--mode hybrid` (все маршрутизируемые эксперты на CPU).

| Модель | Эксперты | pp (tok/s) | tg (tok/s) | Примечания |
|-------|---------|------------|------------|-------|
| Qwen3-30B-A3B Q4_K_M | GPU (Metal, все 48 слоёв) | 4.2 (pp1) | 54.7 (tg16) | 17 GiB отображено на Metal |
| Qwen3-30B-A3B Q4_K_M | CPU (`--mode hybrid` / `--n-cpu-moe ≥48`) | 3.1 (pp9) | 1.6 (tg16) | `ffn_*_exps` → CPU_REPACK 12.96 GiB; рецепт для VRAM класса 8 GB |

`--n-cpu-moe N` держит экспертов слоёв `0..N-1` на CPU (одно объединённое регулярное выражение;
llama-cpp-2 0.1.133 может применить только одно переопределение буфера). `--mode hybrid`
закрепляет каждый тензор экспертов. Плотные модели игнорируют эти шаблоны.

```sh
cargo run -p runa -- run --mode hybrid --max-tokens 16 --ctx 512 --temperature 0 --seed 42 \
  tests/fixtures/Qwen3-30B-A3B-Q4_K_M.gguf hello
cargo run -p runa -- run --mode gpu --n-cpu-moe 8 --max-tokens 16 --ctx 512 \
  tests/fixtures/Qwen3-30B-A3B-Q4_K_M.gguf hello
```

## Как воспроизвести (спайк P0.4)

```sh
cargo run -p runa-engine --example gen -- tests/fixtures/qwen2-0_5b-instruct-q4_0.gguf "hi" 32
cargo run -p runa-engine --example gen -- tests/fixtures/Qwen3-8B-Q4_K_M.gguf "hello" 16
cargo run -p runa-engine --example gen -- tests/fixtures/Qwen3-30B-A3B-Q4_K_M.gguf "hello" 16
# pp512/tg128 (полный P0.6): используйте промпт из 512 токенов (например, `python3 -c "print('hello world '*200)"`) и `... 128`
```

## P2.9, несколько GPU (вручную)

У CI-раннеров здесь один GPU (Apple M3 Max Metal). Тест загрузки на двух GPU помечен
`#[ignore]` (`two_gpu_tensor_split_loads` в тестах `runa-engine`).

На машине с двумя GPU-бэкендами ggml:

```
runa run --mode gpu --device 0,1 --tensor-split 3,1 model.gguf "hi"
```

Ожидается, что строка вердикта будет содержать `devices=0,1 tensor-split=3,1`, а поток
пройдёт успешно. `runa run --device 999 …` должен завершиться ошибкой (никакого молчаливого отката).

## Замеры по критериям P8.7 (2026-09-15, M3 Max 64GB, macOS 26.6.2, runa 0.1.0, отладочная сборка, только CPU)

Машина под нагрузкой (соседние агенты собирали проекты; load avg 127–224). Строки со звёздочкой (*) нужно перемерить на спокойной машине до 1.0.
llama-bench собран из тех же исходников b7709 (`--branch b7709`, Release только для CPU).

| Критерий | Результат |
|------|--------|
| M1, время fit по часам (заголовок+парсинг+`check_fit`, лучшее из 5) | 0.5B **148.2 ms**, 8B **146.6 ms** (< 300 ms ✓) |
| M1, оценка весов против mmap движка (8B) | оценка 5,021,827,072 B против отображённых 4762.19 MiB (**+0.57 %** ✓); KV точно (1152 MiB ≡ 288 MiB@2k×4); оценка compute в 3.74× больше фактической (консервативно) |
| M2, удалённый `hf:unsloth/Qwen3-8B-GGUF:Q4_K_M`, диапазон 8 MiB | **0.57 s** тёплый ✓, 206 partial (без скачивания) ✓; холодный 26.1 s ✗ |
| M3, CPU 0.5B, прогноз против замера | прогноз 62.9/62.9 против замера pp 297–478 / tg 17–23 (FAIL ±30 %); калибровки записаны, предиктор игнорирует базу |
| M4, 0.5B CPU pp512/tg128 против llama-bench | runa **297.8/23.1** против bench **1617.6/85.1** @t12 (18 %/27 %); bench `-t16`: 175.5/10.1 (при равных условиях runa выигрывает) — причина в значении по умолчанию threads=16 |
| M5, холодный TTFT 8B на CPU* | **9.02 s** (всего 9.17 s / 8 ток.; pp 12.2, tg 2.0; пиковый RSS 8.44 GB) |
| M6, 10× `--think-budget 64` Qwen3-8B | 10/10 завершены, идентичные жадные выводы по 417 B; число токенов рассуждения через CLI не наблюдаемо |
| M7, 60 s реальной речи, `--lang en`* | правильно (14 сегментов), **11.5 s** по часам; `--lang auto` (по умолчанию) возвращает пустой результат — баг |
| M8, клип 30 s, `media video --fps 1` | **30 кадров** ✓, аудио ✓, 2.6 s*; транскрипт ~6 s* (расчётно) против критерия <3 s |
| M9, смоук-тесты SDK | openai 2.54.0 + anthropic 0.125.0 проходят без изменений (стрим + инструменты) |
| M10, размер | отладочная 104,649,208 B; release ожидается; grep на отсутствие телеметрии чистый |
| M11, RSS serve (8B ctx2048 / 0.5B) | **8364 MB** / 846 MB против floor+10 % = 563 MB; в простое не меняется (не подключено) |

## Замеры P11.2 на спокойной машине (2026-09-16, M3 Max 64GB, macOS 26.6.2, runa 0.1.0, release, только CPU)

Release собран через `cargo build --offline --release -p runa` за 3m03s
(`llama-cpp-2 0.1.133 → llama.cpp b7709`; переносимая сборка, бэкенды: cpu).
Машина была относительно спокойной во всех запусках ниже (load за 1 мин 3.7–9.2; ср. с P8.7 под нагрузкой
127–224), поэтому строк со `*` нет. Оговорка: не было sudo для `purge`, поэтому mmap в M5 был
тёплым — это не холодное число.

| Критерий | Результат |
|------|--------|
| M10, размер | release **24,662,064 B (23.5 MiB)** ≤ 40 MB ✓; отладочная 121,900,776 B (116.3 MiB) |
| M5, TTFT 8B на CPU (release, `--mode cpu`, threads 12, `--no-prompt-cache`, тёплый mmap) | от запуска до первого токена **≈1.2–1.3 s** (граф готов за 1.05–1.07 s + pp 14–15 ток. @ 83–96 tok/s); всего 8 ток. за 1.79–1.91 s; pp 79–110, tg 16–19; пиковый RSS 9,645,719,552 B (8.98 GiB). Критерий (<2 s) задан для холодного Metal — всё ещё открыт |
| M7, 57.8 s синтезированной английской речи (`say` Samantha), base, `--lang en` | **1.02 s** по часам ✓ (<5 s), транскрипт точный (14/14 повторов); `--lang auto` побайтно идентичен (md5 совпадает) |
| M8, синтетический клип 30 s (320×240 testsrc + речь AAC) | `media video --fps 1`: **30 кадров** 336×336 + аудио ✓ за **1.01 s**; извлечение 30 s аудио + транскрибация **0.59 s**, транскрипт правильный → вместе ≈1.6 s ✓ (<3 s) |
| M4, 8B CPU `bench --mode cpu pp512/tg128` (release) | pp **144.1** tok/s, tg **16.4** tok/s, 12.7 s по часам. Сравнение с llama-bench заблокировано: бинарника нет в PATH, вендоренный llama.cpp (реестр `llama-cpp-sys-2-0.1.133`) не содержит `tools/llama-bench`, crates.io недоступен |
| Сборка Metal `--features metal` (отладочная) | `cargo check` 19.9 s ✓ (только 2 давних предупреждения `dead_code` в `runa/src/main.rs`); `cargo build` 26.3 s, бэкенды `cpu, metal`, 121,812,440 B |
| Декодирование 0.5B на Metal (отладочная, `--mode gpu`, спокойная нагрузка ~4.4) | pp **1035.1** tok/s, tg **235.3** tok/s (9 промпт + 9 генерация, 0.48 s по часам, реальный текст) |
| Metal 8B (отладочная, `--mode gpu`, тёплый mmap, спокойная нагрузка ~4.5) | всего 8 ток. за **1.79 s**; pp 97.3, tg 20.4; первый токен ≈1.35 s (расчётно); 7/8 рассуждение + пустой текст (то же наблюдение о бюджете рассуждения, что и на CPU). Холодный запуск release на Metal для критерия M5 всё ещё ожидается |

Замечено (без изменений кода): режим рассуждения по умолчанию расходует весь бюджет
на рассуждение у Qwen3-8B (`--max-tokens 8` → 7 рассуждение + пустой текст;
`--max-tokens 64` → 63 рассуждение + пустой текст), даже с `--think off`.
Поэтому `run` в конвейере не выводит видимого текста при коротких бюджетах; TTFT
выше измерен до первого сгенерированного токена.

Воспроизведение (из корня репозитория; фикстуры в git-ignore, речь/клип синтезируются в `$TMPDIR`):

```sh
cargo build --offline --release -p runa  # 3m03s; 24,662,064 B
say -v Samantha -o speech.aiff -f speech.txt && afconvert -f WAVE -d LEI16@16000 -c 1 speech.aiff speech-60s.wav
ffmpeg -f lavfi -i testsrc=size=320x240:rate=30:duration=30 -stream_loop 1 -i speech-60s.wav -shortest -t 30 clip-30s.mp4
./target/release/runa run --mode cpu --no-prompt-cache --max-tokens 8 --temperature 0 --seed 7 --json tests/fixtures/Qwen3-8B-Q4_K_M.gguf "Name exactly three primary colors."
./target/release/runa media transcribe --model base --lang en --no-pull speech-60s.wav
./target/release/runa media video --fps 1 clip-30s.mp4
./target/release/runa bench --mode cpu --pp 512 --tg 128 --no-calibrate tests/fixtures/Qwen3-8B-Q4_K_M.gguf
```

## P4.2 ASR (вручную)

Фикстуры CI — синусоидальные тоны по 1 s, а не речь, поэтому WER относительно транскрипта
не определён. На машине с `ggml-base.bin` (скачивается автоматически командой
`runa media transcribe`):

```
runa media transcribe --model base speech.wav
# 1 минута речи на CPU должна обрабатываться < 5 s (M7).
```

Вспомогательная функция для WER: `runa_media::word_error_rate`. Живая проверка: `RUNA_WHISPER=1`.

## P3.3, принудительное соблюдение бюджета (вручную)

В CI нет фикстуры Qwen3-4B. Проверка на 100 запусках / точность GSM8K пропускается.
На машине с Qwen3-4B:

```
runa run --think-budget 256 --grace 64 Qwen3-4B.gguf "Solve 12+7"
```

Токены рассуждения должны оставаться ≤ 256+grace. Повторите ~100 раз; планка качества —
GSM8K-50 в пределах 10 пунктов от рассуждения без ограничений.

## Дальше
- Заново скачать `gpt-oss-20b-MXFP4.gguf` (11G, обрезан) и перезапустить.
- Добавить `pp512`/`tg128` с промптом из 512 токенов для каждой модели/режима.
- Добавить варианты для CPU (`-ngl 0`). Гибрид `--n-cpu-moe` / `--mode hybrid` уже есть (P2.6).
- Добавить Linux x86_64 (CUDA/Vulkan) и Windows из CI. Ночной `runa bench` на CPU — это P5.8 (`docs/perf-nightly.md`); self-hosted раннеры с GPU всё ещё ожидаются.
