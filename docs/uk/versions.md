---
lang: uk
---

# Закріплені версії

План D16: тулчейн Rust, `llama-cpp-2`, `whisper-rs` і `async-openai`
закріплені. `Cargo.lock` забезпечує закріплення крейтів (щойно з'явиться робочий простір,
P0.1); цей файл фіксує, **чому відповідає кожен pin в upstream**, чому його обрано і як його
перевірено. Досліджено 2026-09-08 через API crates.io, архіви вихідного коду docs.rs
(`.cargo_vcs_info.json`), закріплення підмодулів upstream і списки тегів.

## Закріплені версії

| Компонент | Pin | Відповідник в upstream | Опубліковано | Примітки |
|-----------|-----|------------------|-----------|-------|
| Rust | `1.98` (`rust-toolchain.toml`, `mise.toml`) | rustc 1.98.1 (зібрано 2026-08-05) на момент написання | — | канал `1.98` відстежує найновіший патч 1.98.x. `mise.toml` також закріплює `components = "rustfmt,clippy"` (дзеркало rust-toolchain.toml: mise ігнорує той файл за RUSTUP_TOOLCHAIN, і без цього в CI немає cargo-fmt/clippy) |
| llama-cpp-2 | `=0.1.133` | llama-cpp-sys-2 0.1.133 → **llama.cpp `b7709`** (коміт `1051ecd`, 2026-01-12) | 2026-02-03 | точне закріплення — крейт не дотримується semver |
| whisper-rs | `=0.16.0` | whisper-rs-sys 0.15.0 → **whisper.cpp `v1.8.3`** (коміт `2eeeba5`, 2026-01-15) | 2026-03-12 | найновіший реліз whisper-rs |
| async-openai | `=0.42.0` | — | 2026-09-15 | Responses API за функцією `responses`; MSRV 1.75 |
| mistral.rs | `=0.8.1` | mistralrs 0.8.1 → mistralrs-core 0.8.1 (див. нижче) | 2026-04-02 | необов'язковий другий бекенд, `--features mistralrs` (D2, P9.2). Точне закріплення, як і для інших крейтів рушіїв; `default-features = false` залишає GPU-прискорення опційним (D13) |
| moon | `2.5.4` (`mise.toml` `aqua:moonrepo/moon`, `.moon/workspace.yml` `versionConstraint`) | moonrepo/moon `v2.5.4` (2026-09-03) | 2026-09-08 | граф задач монорепозиторію поверх робочого простору cargo (D22, K1) |
| reqwest | `=0.13.5` (default-features вимкнено; `blocking` + `rustls`) | — | 2026-09-15 | P1.2 віддалене читання заголовка; майбутній клієнт адаптера Anthropic (D9). TLS на чистому Rust, без системних бібліотек на жодній цілі CI |
| serde | `=1.0.229` (`derive`) | — | 2026-09-08 | P1.10 збереження бази калібрування |
| serde_json | `=1.0.151` (обхід `Value` без derive) | — | 2026-09-08 | P1.2 список файлів через Hub API |
| rustyline | `=18.0.1` | — | 2026-09-08 | P2.3 REPL чату (редагування рядка + історія у файлі) |
| assert_cmd | `=2.2.2` | — | 2026-09-08 | P2.3 e2e-тести CLI |
| ffmpeg | `9.0.1` (`mise.toml`) | ffmpeg `9.0.1` | 2026-09-15 | P4.5 відео через ffmpeg-sidecar (бінарник, не лінкується) |
| python | `3.13.15` (`mise.toml`) | CPython `3.13.15` (найновіший 3.13.x) | 2026-09-15 | P0.7 фікстури, P3.9/P6.2 смоук-тести SDK (лише stdlib; у 3.11.9 не було атестацій) |
| node | `24.21.0` (`mise.toml`) | Node `24.21.0` LTS | 2026-09-15 | P3.9/P6.2 смоук-тести SDK (EOL 20.x — 2026-04) |
| cargo-dist | `0.33.0` (`mise.toml` `aqua:axodotdev/cargo-dist`) | cargo-dist `0.33.0` | 2026-09-15 | P6.3 пакування. Готовий бінарник через aqua з 2026-09-14 (раніше був бекенд `cargo:`, який компілював із вихідного коду й змагався із завантаженнями rustup у CI) |
| cargo-cache | `0.8.3` (`mise.toml` `cargo:cargo-cache`) | cargo-cache `0.8.3` | 2023-09-01 | утиліта розробника: `moon run root:cache` / `cache-dry-run` / `cache-autoclean`; не є залежністю крейтів |
| zig | `0.16.0` (`mise.toml`) | zig `0.16.0` | 2026-09-15 | D23 власні ядра (`runa-kernels`) |
| keyring | `4.2.0` | — | 2026-09-15 | P3.8 сховище ключів ОС (`service = runa`) |
| symphonia | `=0.6.1` (mp3/aac/flac/ogg/pcm/wav/isomp4) | — | 2026-09-15 | P4.1 декодування аудіо |
| hound | `=3.5.1` | — | 2026-09-08 | P4.1 читання/запис WAV |
| rubato | `=5.0.0` | — | 2026-09-15 | P4.1 ресемплінг до 16 kHz |
| git-cliff | `2.13.1` (`mise.toml`) | git-cliff `2.13.1` | 2026-03-01 | P13.3 генерація `CHANGELOG.md` із заголовків комітів (`cliff.toml`); той самий pin, що й у rtok, тож нотатки релізного коміту відтворювані |

Використовуйте оператор точного закріплення `=` для трьох крейтів рушіїв/API у
`Cargo.toml`; `llama-cpp-2` явно не дотримується semver, а `whisper-rs`
залишає за собою право ламати сумісність у патч-релізах, коли йдеться про `raw-api`.

## Відповідність llama-cpp-2 → llama.cpp (перевірений ланцюжок)

1. `llama-cpp-2 0.1.133` опубліковано 2026-02-03 з
   тегу `0.1.133` репозиторію `utilityai/llama-cpp-rs` = коміт `349ae33`
   (`.cargo_vcs_info.json` крейта).
2. Дерево репозиторію на цьому коміті закріплює підмодуль
   `llama-cpp-sys-2/llama.cpp` на `1051ecd28907d2ca0a15c135f190fe415d0a3d1b`.
3. Цей коміт — це точно тег релізу llama.cpp **`b7709`** (2026-01-12,
   "vulkan: Disable large coopmat matmul configuration on proprietary AMD
   driver", #18763). Перехресна перевірка: тег `b8400` випереджає його на 691 коміт, а
   8400 − 7709 = 691 — номери тегів llama.cpp рахують коміти, тож відповідність
   точна.
4. `llama-cpp-2 0.1.133` залежить від `llama-cpp-sys-2 ^0.1.133` (функції
   передаються далі); на macOS arm64 крейт автоматично вмикає `metal` через sys-крейт.

Прапорці функцій, доступні в 0.1.133: `cuda`, `cuda-no-vmm`, `metal`,
`vulkan`, `openmp`, `mtmd`, `sampler`, `dynamic-link`, `system-ggml`,
`android-shared-stdcxx`.

> **Відхилення від плану D2:** функції `native` (наведеної в D2, використовуваної в
> P5.7) **немає в llama-cpp-2 0.1.133**. Вона з'являється в пізніших
> релізах llama-cpp-2 (є щонайменше в 0.1.156). Портабельні релізні збірки мають
> покладатися на диспетчеризацію ggml під час виконання (D13), доки ми не оновимося далі цього pin;
> диспетчеризації `runa-kernels` (P5.2) це не стосується.

### Огляд закріплень для NPU (P9.4, 2026-09-15)

Питання: чи відкриває якийсь новіший `llama-cpp-2` cargo-функції `hexagon`/`openvino`
для зрізу NPU у P9.4?

- Перевірено `llama-cpp-2` **0.1.154** (архів крейта
  `llama-cpp-2-0.1.154.crate`, git `bed81ad4`, а також сторінку функцій на docs.rs):
  повний список функцій — `android-shared-stdcxx`,
  `android-static-stdcxx`, `common`, `cuda`, `cuda-no-vmm`, `default`,
  `dynamic-backends`, `dynamic-link`, `llguidance`, `metal`, `mkl`,
  `mtmd`, `opencl`, `openmp`, `rocm`, `sampler`, `static-openmp`,
  `static-stdcxx`, `system-ggml`, `system-ggml-static`, `vulkan`.
  **Ні `hexagon`, ні `openvino` немає.** (0.1.154 додає `opencl`, `rocm`, `mkl`,
  `llguidance`, `dynamic-backends` порівняно з нашим pin 0.1.133; жодна з них не є шляхом
  до NPU.) Відповідна сторінка docs.rs для `llama-cpp-sys-2` 0.1.154 показує
  той самий набір.
- Локально перевірено, що посилання на неіснуючу функцію залежності
  (`hexagon = ["llama-cpp-2/hexagon"]`) ламає **навіть типове**
  розв'язання (`cargo build` падає на етапі розв'язання залежностей), тож функції P9.4
  не можуть передаватися до llama-cpp-2, доки upstream їх не додасть.

Рішення: **чекати** (залишаємо pin `=0.1.133`; P9.4 постачає лише перевірку +
обмеження збірки + документацію):

- `system-ggml` / `system-ggml-static` відірвали б збірку від закріпленого
  llama.cpp (b7709) і перевели б на той ggml, який надає хост, — це ламає дисципліну
  закріплень D16 і відтворюваність збірки заради шляху Tier-3.
- `dynamic-backends` лише завантажує бекенди ggml, вбудовані в той самий закріплений
  llama.cpp під час компіляції; він не може взяти нізвідки бекенди Hexagon/OpenVINO,
  без яких цей pin було зібрано.
- Запасний варіант D2 з `bindgen` (власні прив'язки поверх `llama.h`)
  непропорційний для обладнання Tier-3, що тестується вручну, без раннерів CI.

Що розблокує справжнє вивантаження на NPU, по черзі: (1) функції `hexagon`/`openvino`
в upstream `llama-cpp-2` (або реліз ggml із цими бекендами, вартий
свідомого оновлення pin через бар'єри D1/D15); (2) SDK вендорів
(`HEXAGON_SDK_ROOT`, `INTEL_OPENVINO_DIR`); (3) ручна валідація на пристрої
+ калібрування заглушок `HwSpec::{hexagon,openvino}` і
маркерів `npu_present()`. Доти cargo-функції `hexagon`/`openvino`
— це лише заглушки для перевірки: їх увімкнення не змінює розміщення (тензори
лишаються на CPU — план D12).

## Прапорці збірки (P5.7)

| Збірка | Прапорці | Примітки |
|-------|-------|-------|
| Локальна розробка (macOS arm64) | `cargo build -p runa-engine --features metal` | Metal типово ввімкнено на Apple Silicon через `llama-cpp-sys-2` |
| Локальна розробка (Linux CUDA) | `--features cuda` для `runa-engine` / робочого простору | Задача CI на ubuntu лише для CPU; CUDA деінде лише збирається |
| Портабельний реліз | без `-march=native`; диспетчеризація ggml під час виконання (D13) | `native` у `llama-cpp-2` з'являється після pin 0.1.133 — перш ніж вмикати, перейдіть на новіший pin |
| `runa-kernels` (P5.2+) | для нових ядер перевага Zig (C ABI) (D23); C/`.S` лише там, де Zig гірший | SIMD через Zig `@Vector` або диспетчеризацію C під час виконання |

Релізні бінарники мають працювати на хостах **без** AVX-512; не постачайте `-C target-cpu=native` у релізних профілях (`[profile.release]` у `Cargo.toml` не має `target-cpu`).

Основне з релізу 0.1.133, що стосується runa: прив'язки mtmd
(аудіо/зображення/відео через `LlamaContext`/контекст mtmd), виклик функцій +
допоміжні функції для перетворення у стилі OpenAI, ланцюжок семплерів. Зауважте, що згодом llama.cpp прибрав
в upstream власний OpenAI-сумісний сервер (синхронізація llama-cpp-2 0.1.147) —
нас це не стосується: `runa serve` — це наш власний сервер на axum (D11).

### Бекенд RPC: вендорений, за cargo-функцією (P9.3, 2026-09-15)

llama.cpp b7709 підтримує розподілений інференс через бекенд RPC
(`--rpc host:port`, `ggml_backend_rpc_add_server`), і жоден опублікований
`llama-cpp-2` (переглянуто 0.1.131–0.1.156 через sparse-індекс — жоден не відкриває
функцію `rpc`) не може його зібрати, але ні форк, ні підвищення pin не знадобилися.
Опублікований крейт `llama-cpp-sys-2 0.1.133` постачає всі заголовки, потрібні бекенду
(`ggml-rpc.h`, а також внутрішні `ggml-impl.h`,
`ggml-backend-impl.h`, `ggml-common.h`); вирізано лише каталог реалізації `ggml-rpc/`
на 78 KB, тож `-DGGML_RPC=ON` падає на етапі конфігурації CMake
(`add_subdirectory(ggml-rpc)` → відсутній каталог), а
`wrapper.h` ніколи не розбирає `ggml-rpc.h` (немає прив'язок `ggml_backend_rpc_*`).
`GGML_USE_RPC` у ядрі ggml лише автоматично реєструє *порожню* базову реєстрацію — справжній
потік (`ggml_backend_rpc_add_server` → `ggml_backend_register`) іде
через публічний C API, тож `runa-engine` компілює дослівний `ggml-rpc.cpp` з b7709
(вендорений у `crates/runa-engine/rpc/` з побайтово ідентичними
заголовками, див. `rpc/README.md`) у `build.rs` за новою cargo-функцією `rpc`
(C++17, `ws2_32` на Windows) і реєструє кожен ендпоінт `--rpc` у `load()`
до ініціалізації бекенду — дзеркально до upstream `common/arg.cpp:add_rpc_devices`.
Прошарок у стилі `#[link]` не знадобився: символи беруться з нашого власного
скомпільованого об'єкта, сумісного за ABI з тими самими вихідними кодами b7709, які збирає sys-крейт.
Запобіжник pin у `build.rs` зриває збірку `rpc`, якщо `llama-cpp-sys-2` відходить
від `=0.1.133` (D16; правило оновлення в `rpc/README.md`).

Використання: `runa run --rpc 192.168.1.5:50052 --device RPC0 model.gguf`
(`--rpc` лише реєструє; віддалені пристрої перелічуються як `RPC0`, `RPC1`, …
і вибираються явно через `--device` — ніколи неявно, D12).
`runa doctor` показує `rpc: true/false`; `runa fit --rpc` оцінює
локальне розміщення з явною приміткою. Протестовано через loopback із фальшивим TCP-сервером,
що відповідає на `HELLO` 3.6.0 + `DEVICE_COUNT` (`runa-engine/tests/rpc.rs`).

### Огляд оновлення рушія: `llama-cpp-2` 0.1.156 (2026-09-15, вердикт: WAIT)

Спробу підвищити `=0.1.133` → `=0.1.156` було зроблено й **відкочено**. П'ять помилок
компіляції, усі в шляху структурованого виводу / виклику інструментів (P8.1/P8.2):
upstream прибрав увесь шар oaicompat, на якому побудовано ці функції
(`apply_chat_template_oaicompat`, `OpenAIChatTemplateParams`,
`ChatTemplateResult`, `GrammarTrigger*`, `wrapper_oai.*`). Його заміна
(`common/chat.{h,cpp}` + парсери peg/auto, типово збирається через нову
cargo-функцію `common`) — це API з типами C++ (`std::string`, `std::vector`,
`nlohmann::ordered_json`) без прив'язок до Rust — відновлення P8.1/P8.2 на ньому
означає C-прошарок плюс повну повторну валідацію граматик, лінивих/енергійних тригерів
і обох циклів інструментів SDK. Згідно з планом D16 (свідомі оновлення рушія
через бар'єр бенчмарків) pin лишається на `=0.1.133`, доки це портування не
заплановано окремою задачею; вендорений RPC вище (протокол 3.6.0) лишається
узгодженим із b7709. Зауважте, що майбутнє підвищення також переведе протокол RPC на 5.0.0
(підмодуль `e79e4bf`) — оновлюйте `rpc/` разом із ним.

## Відповідність whisper-rs → whisper.cpp (перевірений ланцюжок)

1. `whisper-rs 0.16.0` (найновіший, 2026-03-12, codeberg `tazz4843/whisper-rs`,
   коміт `7558e1b7`) вимагає `whisper-rs-sys ^0.15`; єдина 0.15.x —
   0.15.0, опублікована того самого дня з того самого коміту.
2. Sys-крейт постачає whisper.cpp як підмодуль `sys/whisper.cpp`,
   закріплений на `2eeeba56e9edd762b4b38467bab96c2517163158`.
3. Цей коміт — whisper.cpp **`v1.8.3`**, буквально коміт підвищення версії
   `release : v1.8.3` (2026-01-15, CMake 1.8.2 → 1.8.3).

Прапорці функцій у 0.16.0: `cuda`, `hipblas`, `metal`, `vulkan`, `openblas`,
`raw-api`, `log_backend`, `tracing_backend`.

Upstream whisper.cpp пішов далі (v1.8.4 … v1.8.7, потім v1.9.x; найновіша
переглянута: **v1.9.3**, 2026-09-08). Отже, наш pin для ASR відстає приблизно на 2 мінорні лінії;
оновлюйтеся через бар'єр бенчмарку M7 (1 хвилина мовлення < 5 s на CPU),
коли whisper-rs перейде на новіший whisper.cpp.

## mistral.rs

0.8.1 (2026-04-02), MIT, MSRV 1.88. Фасад `mistralrs` реекспортує
білдери й типи `mistralrs-core 0.8.x`; runa використовує високорівневий
`ModelBuilder` (з автовизначенням, уміє працювати з локальним каталогом) плюс
`blocking::BlockingModel` / `BlockingStream` (власний рантайм tokio, синхронний ітератор
токенів — runa-engine лишається без рантайму). У 0.8.1 немає функції `default`,
тож `default-features = false` — це закріплення без ефекту на майбутнє;
GPU-прискорення (`metal`, `cuda`, …) лишається опційним згідно з D13 і ще не передається далі
(бекенд автоматично розподіляє пристрої). Оновлюйтеся на лінію 0.9.x, якщо/коли вона
вийде; зауважте, що текст плану/дорожньої карти, який передбачав `0.9.3`, було написано
до того, як ця версія з'явилася в upstream (найновішою на crates.io на момент
написання є 0.8.1).

## moon

2.5.4 (2026-09-03), встановлюється через mise (`aqua:moonrepo/moon` — основного
плагіна mise для moon немає, бекенд aqua віддає офіційний асет релізу
`moon_cli-<arch>.tar.xz`). `.moon/workspace.yml` вимагає
`versionConstraint >= 2.5.4`. Перевірено 2026-09-08 через `mise install` +
`moon projects` (9 проєктів: 8 крейтів + корінь). Плагін тулчейну rust
(1.0.9, вбудований WASM) **не** завантажує тулчейни Rust
(`download_prebuilt` не реалізовано) — встановлення робить mise (D21); moon
записує pin (`version: '1.98'`) для графа/гешування/кешування, але ОБИДВА прапорці
синхронізації лишаються **вимкненими**: будь-який із них переписує закріплення як вимоги semver
(`~1.98` — невалідно в `rust-version` Cargo і в `channel` rustup) і
прибирає коментарі з файлів, ламаючи кожну команду cargo (спостерігалося 2026-09-08).
Оновлюйтеся
через перевірку паритету M13 (`moon run :test` == `cargo test
--workspace`).

## async-openai

0.42.0 (2026-09-15), MIT. Увімкніть `responses` для Responses API
зі стримінгом (P3.5); `byot` (bring your own types) плюс перевизначення `base_url`
покривають OpenAI-сумісних провайдерів (OpenRouter, DeepSeek, Groq, llama-server).
Повторні спроби при обмеженні частоти запитів з експоненційною затримкою вбудовано. Адаптер
Anthropic — це наш власний клієнт на `reqwest` + SSE (D9) — жодного закріплення SDK.

## Ще не закріплено — закріпити під час першого використання (у Cargo.lock + рядок тут)

| Крейт | Задача плану |
|-------|-----------|
| `sherpa-onnx` (Parakeet; офіційна прив'язка до Rust — **не** `sherpa-rs`, архівований 2026-06). Порожню cargo-фічу `parakeet` прибрано; `transcribe_parakeet` повертає «not available yet», доки окрема задача не підключить крейт. | P4.2 |
| `ffmpeg-sidecar` | P4.5 |
| `hf-hub` | P2.4 |
| `axum` | P3.9 |
| `figment`, `clap` | P0.1/P2.3 |
| `sysinfo`, `raw-cpuid`, `objc2-metal`, `ash`, прив'язки NVML | P1.6 |
| `rusqlite` (bundled) | P1.10 |
| `criterion` | D15 |


## Нативна чи портабельна збірка (P5.7)

Типові **релізні** збірки портабельні: `llama-cpp-sys-2` задає `GGML_NATIVE=OFF`,
а ggml під час виконання вибирає AVX2 / AVX-512 / NEON / SME. Такий бінарник працює на
машині без AVX-512.

Локальні збірки, налаштовані під хост:

```sh
RUSTFLAGS='-C target-cpu=native' cargo build --release --features native
```

Сам лише `--features native` **не** передає `-march=native` у llama.cpp
(крейт не має cargo-функції `native`). `llama-cpp-sys-2` вмикає
`GGML_NATIVE=ON` лише тоді, коли бачить `-C target-cpu=native` у `RUSTFLAGS`.
`runa-kernels` дотримується того самого правила: Zig збирає з `-mcpu=baseline`, а
з `-mcpu=native` — лише за `-C target-cpu=native`. `whisper-rs-sys` збирає
власний ggml, чиє типове значення в CMake — `GGML_NATIVE=ON`; `.cargo/config.toml`
задає `GGML_NATIVE=OFF` (експортований `GGML_NATIVE=ON` однаково має перевагу). Інакше
бібліотека, зібрана на хості з AVX-512 (кешована збірка CI, релізний раннер), спричиняє
SIGILL на старішому CPU.
`runa doctor --json` показує `native_build` (true тоді й лише тоді, коли бінарник скомпільовано
з `--features native`) і `backends` (`cpu` плюс будь-які з `metal` / `cuda` /
`vulkan` / `mtmd`, вкомпільовані в нього). Тег `vX.Y.Z` запускає `.github/workflows/release.yml`
(CPU-архіви cargo-dist 0.33 + інсталятори shell/powershell/homebrew). GPU-варіанти:
`.github/workflows/release-variants.yml`. Формула Homebrew лежить у
GitHub Release; `brew install pyrlyn/runa/runa` потребує tap-репозиторію
`listepo/homebrew-runa`. Локально: `bash scripts/cargo-dist.sh generate --mode=ci --check`.

### Пакет ketch (`ketch.toml`)

Кореневий `ketch.toml` — це маніфест [ketch](https://github.com/listepo/ketch)
для цього репозиторію: файл, який `ketch registry push` пропонує до
`listepo/ketch-registry` як `runa/ketch.toml`, і те, що показує `ketch info runa`,
щойно він там з'явиться. Виведення з `github:pyrlyn/runa` працює й без
нього; маніфест якраз закріплює вибір асету.

| Поле | Значення |
|-------|-------|
| `source` | `github:pyrlyn/runa` |
| `bin` | `runa` (корінь розпакованого архіву cargo-dist) |
| `[asset] include` | `*.tar.xz`, `*windows-msvc.zip` |
| `[asset] exclude` | GPU-варіанти (`-metal`, `-vulkan`, `-cuda`), інсталятори shell / powershell / homebrew, `dist-manifest.json`, `*.sha256`, архів вихідного коду |

GPU-архіви тут важливі: `release-variants.yml` завантажує їх під тими самими
цільовими трійками, що й портабельні CPU-архіви (`runa-aarch64-apple-darwin`
проти `runa-aarch64-apple-darwin-metal`), тож без списку `exclude` оцінювання
асетів могло б дійти до варіанта, що потребує вкомпільованих GPU-функцій. Блоку
`trust` немає: реліз не публікує супровідних файлів із підписами, і політика `trust`
зривала б кожне встановлення.

Перевірте зміну перед тим, як надсилати її (дерево реєстру — це каталоги
`<package>/ketch.toml`):

```sh
TMP=$(mktemp -d); mkdir -p "$TMP/runa"; cp ketch.toml "$TMP/runa/"
ketch registry validate "$TMP"    # validated 1 package
```

### Процес релізу

Реліз — це коміт із версією плюс тег, які створює `scripts/release.sh`
(подробиці, режими та перелік того, що не під'єднано: [`release.md`](../release.md)). Саме тег
згенерований dist-ом `release.yml` перетворює на GitHub Release.

## Політика оновлень (D16)

- llama.cpp рухається швидко: ~8–12 тегів на день, і **номери тегів рахують коміти**
  (b7709 → b8400 = 691 коміт). llama-cpp-2 випускає релізи приблизно щотижня і не
  дотримується semver — завжди закріплюйте точну версію й оновлюйтеся свідомо, не частіше ніж раз на місяць.
- Кожне оновлення проходить бар'єр бенчмарків (D1: ядра мають обійти шлях ggml
  на ≥ 5 % наскрізно або ≥ 2× ізольовано; D15: CI продуктивності падає на регресії > 3 %)
  і оновлює `Cargo.lock` і цей файл у тій самій зміні.
- Найновіші версії на 2026-09-08: llama-cpp-2 **0.1.156** (llama.cpp ≈ b10405+),
  whisper-rs 0.16.0 (без змін, але whisper.cpp в upstream — на v1.9.3),
  async-openai 0.42.0.

## Перевірка

```sh
cargo --version   # 1.98.x — rust-toolchain.toml; змінна середовища RUSTUP_TOOLCHAIN
                  # локально перевизначає файл, приберіть її, щоб перевірити pin
mise ls            # rust 1.98 (mise.toml)
mise install && moon --version   # moon 2.5.4 (K1); moon projects → 9 проєктів
```

Файли, що містять закріплення: `rust-toolchain.toml`, `mise.toml`, цей файл
і `Cargo.lock`, щойно з'явиться робочий простір (P0.1).
