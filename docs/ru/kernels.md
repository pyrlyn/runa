---
lang: ru
---

# runa-kernels: бенчмарки и критерии внедрения

Результаты бенчмарков по каждому ядру и критерии внедрения (план D14, D23). P5.2 поставляет
каркас крейта (C). Новые ядра пишутся предпочтительно на Zig (C ABI, `@Vector`), а не на C;
C/`.S` остаются только там, где Zig хуже (FFI ggml, SME2/AMX).

## P5.2, начальная загрузка — softmax (скалярный)

| Реализация | Статус | Примечания |
|----------------|--------|-------|
| Скалярная на C (`c/scalar.c`) | поставляется | всегда линкуется |
| Эталон на Rust | поставляется | `softmax_f32_reference`, юнит-тесты |

## P5.3 — сэмплирование + SIMD softmax (Zig)

| Реализация | Статус | Примечания |
|----------------|--------|-------|
| Zig softmax (`@Vector`) | поставляется | `runa_softmax_f32_scalar` (C ABI); LLVM генерирует SIMD для хоста |
| Zig top-k / top-p / min-p | поставляется | `runa_top_k_f32` / `runa_top_p_f32` / `runa_min_p_f32`; рабочая память через `page_allocator` |
| Zig `sample_token` | поставляется | temperature + фильтры + xorshift |
| C NEON/AVX2 softmax | не линкуется | `c/` оставлен для справки; при линковке вместе с Zig символы C ABI дублируются |
| Подключение к движку | поставляется | `SamplingConfig::kernel_sampler` / `RUNA_KERNEL_SAMPLER=1` |

Измерено на этой машине (`cargo bench -p runa-kernels --bench softmax`, softmax шириной 151936, Apple Silicon aarch64):

| Операция | Время | против эталона Rust | Критерий D14 |
|----|------|-------------|----------|
| `softmax_f32` Zig (`SoftmaxImpl::ZigVector`) | 405.8 µs | **0.91×** (медленнее) | 2× изолированно → **REJECT** |
| `softmax_f32_reference` | 368.8 µs | — | — |
| `sample_token`, словарь 4k | 112.2 µs | — | ниже 5% e2e |

Генерация по умолчанию по-прежнему использует цепочку сэмплеров ggml. Собственные ядра остаются
включаемыми по желанию через `RUNA_KERNEL_SAMPLER=1`, пока не выполнен критерий D14. Сквозные 5% tg на
модели со словарём 150k не измерялись (нет фикстуры); решение принимает изолированный критерий: **REJECT**.

## P5.6 — спекулятивное декодирование на n-граммах

Черновики строятся поиском по триграммам и жадно проверяются по argmax целевой модели (`--ngram`
или неявно через `--draft`). Идентичность с выводом без ngram сохраняется при `--temperature 0`
(тест движка на qwen2-0.5B). `--draft` резервирует GGUF в планировщике fit;
вторую модель он не загружает.

Критерий ≥1.3× tg на промптах с кодом в CI пропускается (нет отдельной задачи
бенчмарка ngram). Запишите здесь отношение, когда машина будет профилирована.

## P5.4 — предобработка изображений (NEON/AVX2 против `fast_image_resize`)

Измерено на этой машине (`cargo bench -p runa-media --bench media`, 336² RGB):

| Операция | Время | против скалярной | Критерий D14 |
|----|------|-----------|----------|
| `normalize_rgb` скалярная (по умолчанию) | 40.8 µs | — | — |
| `normalize_rgb_simd` NEON | 26.8 µs | **1.52×** | 2× изолированно → **REJECT** |
| `patchify_rgb` 16×16 | 87 µs | — | ниже 5% e2e |
| nearest `resize_rgb` (P4.9) | 73 µs | — | — |
| `fast_image_resize` HQ (K3) | 1.82 ms | 0.04× против nearest | **REJECT** (оставить nearest) |

`normalize_rgb` по умолчанию остаётся скалярной. SIMD-вариант экспортируется как `normalize_rgb_simd`
для сравнения (тест эквивалентности на хвосте из 337 байт). Ресайз остаётся
методом ближайшего соседа; HQ FIR был медленнее ещё в K3.


## P5.5 — квантованное умножение матрицы на вектор на SME2 / AMX (Q4_K / Q8_0)

Исследуемый пин: `llama-cpp-2` **0.1.133** → llama.cpp **b7709** (`1051ecd`,
2026-01-12). Исходники изучены по
[`ggml/src/ggml-cpu/`](https://github.com/ggerganov/llama.cpp/tree/1051ecd28907d2ca0a15c135f190fe415d0a3d1b/ggml/src/ggml-cpu).

### Машина разработки

| Свойство | Значение |
|----------|-------|
| CPU | Apple M3 Max |
| SME2 | `hw.optional.arm.FEAT_SME2` = **0** (SME2 есть начиная с M4) |
| AMX | недоступно (x86 Sapphire Rapids+) |

На этой машине невозможно скомпилировать, запустить или проверить на эквивалентность ассемблер SME2/AMX.

### Состояние ggml на b7709

**Intel AMX (x86, `GGML_USE_AMX` / `__AMX_INT8__`)**

| Квант | Путь | Символы / файлы |
|-------|------|-----------------|
| Q4_K | AMX GEMM/GEMV только по весам | `ggml_backend_amx_mul_mat` в `amx/mmq.cpp` (ветка `GGML_TYPE_Q4_K`); буфер `ggml_backend_amx_buffer_type` в `amx/amx.cpp` |
| Q8_0 | то же | ветка `GGML_TYPE_Q8_0` в `amx/mmq.cpp` |

Быстрый путь m=1 (промпт) откатывается к AVX-512-VNNI при маленьком батче (комментарий
в `amx/mmq.cpp`).

**AArch64 KleidiAI / SME (Apple M4+ с `-DGGML_CPU_KLEIDIAI=ON`)**

| Квант | SME2 KleidiAI | Другой путь ggml для CPU |
|-------|---------------|---------------------|
| Q8_0 | да — `kai_run_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme2_mopa` (GEMM), `kai_run_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot` (GEMV) в `kleidiai/kernels.cpp`; диспетчеризация `ggml_kleidiai_select_kernels_q8_0` | таблицы i8mm/dotprod в том же файле |
| Q4_K | на этом пине записи в таблице KleidiAI **нет** | repack + gemv `ggml_gemv_q4_K_8x4_q8_K` / `gemv<block_q4_K,…>` в `repack.cpp`; `ggml_vec_dot_q4_K_q8_K` в `arch/arm/quants.c` |

KleidiAI покрывает `GGML_TYPE_Q4_0` и `GGML_TYPE_Q8_0` при `CPU_FEATURE_SME`,
но не `GGML_TYPE_Q4_K`. Умножение матрицы на вектор для промпта в Q4_K на Apple Silicon вместо этого использует пути ggml
repack / i8mm / NEON vec-dot.

### Решение: **SKIP**

- ggml уже владеет умножением матрицы на вектор для Q4_K и Q8_0 на AMX; Q8_0 (и Q4_0) на SME2 через
  KleidiAI; Q4_K на AArch64 через repack + `ggml_vec_dot_q4_K_q8_K`.
- Нет разрыва настолько большого, чтобы оправдать собственное ядро `.S`, способное обойти ggml по
  критерию D1/D14, — и эта машина всё равно не может его собрать или протестировать.
- Никаких объектов `runa-kernels`, диспетчеризации или чисел для критерия не записано.
