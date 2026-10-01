---
lang: uk
---

# docs/memory.md — публічні методи: `MemoryManager` і `TaskRegistry`

Крейт: `runa-memory` (план D17/D18, фаза P7, метрики M11/M12).
Цілі: M11 (RSS у простої ≤ нижня межа + 10 %), M12 (жодна задача не утримується двічі).
Конфігурація: `[memory] idle_timeout_s, floor_mib, max_growth_mib`;
`[agents] registry = "docs/tasks.md"`. Ескізи — у `plan.md` §5.

Домовленості: `demand_mib`/`rss_mib` — у MiB (`u64`). Часові мітки — рядки UTC
у форматі RFC 3339. Помилки задокументовано для кожного методу; усі методи
синхронні та потокобезпечні (`Send + Sync`).

---

## `MemoryManager`

Адаптивна пам'ять процесу. Простій (жодного запиту/задачі протягом `idle_timeout_s`) →
звільнити кеш промптів, буфери енкодера, чорнеткову модель, стиснути пули до
`floor_mib`. Важка задача → заздалегідь збільшити арени до вердикту fit + запас і
щонайбільше на `max_growth_mib` понад поточне використання. Ніколи не вивантажує активну
модель. Кожен перехід логує RSS до і після.

### `fn current_usage(&self) -> Usage`

- Повертає: `Usage { rss_mib, budget_mib, state }`, де `state` —
  `Idle` | `Normal` | `Heavy`, а `budget_mib` — поточна стеля
  (вердикт fit + запас + надане зростання).
- Приклад: `let u = mm.current_usage(); assert!(u.rss_mib <= u.budget_mib);`
- Примітки: чисте спостереження, без побічних ефектів; `state` виводиться з нещодавнього
  навантаження порівняно з `idle_timeout_s` і з очікуваної потреби.

### `fn touch(&self)`

- Ефект: записує, що запит/задача виконується. Скидає таймер простою.
- Коли викликати: на початку кожного generate / запиту до сервера (`grow_for` уже
  це робить).
- Примітки: не змінює `LoadState`.

### `fn maybe_idle(&self)`

- Ефект: якщо `last_activity` старший за `idle_timeout_s`, викликає
  `on_idle`. Нічого не робить, якщо нещодавно була робота (гістерезис).
- Коли викликати: опитування після завершення запиту або на такті простою сервера.
- Перевірка: `idle_timeout_s = 0` стискає одразу; тайм-аут 3600 s
  не стискає одразу після `touch`.

### `fn on_idle(&self)`

- Ефект: стискає до `floor_mib` (звільняє кеші/буфери/пули, як
  описано вище); нічого не робить, якщо вже на нижній межі чи нижче. Логує RSS до і після.
- Коли викликати: коли немає жодного запиту чи задачі протягом повного `idle_timeout_s` (або з
  `maybe_idle`).
- Перевірка: тривалий тест M11 — RSS у простої ≤ нижня межа + 10 %.
- Примітки: гістерезис — стискати лише після повної тиші, ніколи посеред сплеску
  (див. «Ризики» в `plan.md` §8).

### `fn on_heavy(&self, demand_mib: u64)`

- Параметри: `demand_mib` — очікувана додаткова пам'ять для вхідної важкої
  задачі (зростання ctx, пакет, медіа).
- Ефект: зручна обгортка — надає те, що вміщається, через `grow_for`, а
  інакше зберігає поточне розміщення (викликач переходить на менший
  ctx/квантизацію або хмару згідно з пропозицією fit). Ніколи не перевищує стелю.
- Приклад: `mm.on_heavy(2048); // make room for a 2 GiB ctx bump`

### `fn shrink_to_floor(&self)`

- Ефект: безумовне стискання до `floor_mib`, той самий набір звільнень, що й у
  `on_idle`, але негайно (використовується тестами та явними
  шляхами обслуговування на кшталт `runa doctor --shrink`).
- Примітки: однаково ніколи не вивантажує активну модель.

### `fn grow_for(&self, demand_mib: u64) -> Result<(), MemoryError>`

- Параметри: `demand_mib` — байти (MiB), потрібні задачі понад поточне використання.
- Повертає: `Ok(())` після попереднього збільшення; `Err(MemoryError::OverCeiling {
  demand_mib, ceiling_mib, suggestion })`, коли потреба перевищує
  вердикт fit + запас (`suggestion`: менший ctx, інша квантизація,
  `--kv q8_0` або хмара — той самий словник, що й у `runa fit`).
- Перевірка: тест із важким ctx проходить без OOM посеред запуску; тест на перевищення стелі
  перевіряє, що `Err` містить пропозицію.

---

## `TaskRegistry`

Кооперативні заявки над `docs/tasks.md` (зберігаються у файлі). Стани рядків:
`free` | `in progress` (+ `agent`, `started_at`). Завершення відстежується
позначкою в `plan.md`; реєстр відстежує лише живі заявки.

### `fn list_free(&self) -> Vec<String>`

- Повертає: ID усіх задач `free` (напр. `["P0.1", "P1.4", …]`), відсортовані
  за порядком фаз.
- Приклад: `for id in reg.list_free() { println!("{id}"); }`

### `fn status(&self, task_id: &str) -> Option<TaskStatus>`

- Параметри: `task_id` — напр. `"P2.6"`.
- Повертає: `None` для невідомих ID; `Some(Free)` або
  `Some(InProgress { agent, started_at })`.
- Примітки: лише читання; використовуйте, перш ніж питати про задачу `in-progress`.

### `fn claim(&self, task_id: &str, agent: &str) -> Result<TaskClaim, ClaimError>`

- Параметри: `task_id`, `agent` (ім'я агента, що подає заявку; `started_at`
  проставляє реєстр у момент заявки).
- Повертає: `Ok(TaskClaim { task_id, agent, started_at })`, і рядок
  стає `in progress`.
- Помилки: `NotFound` (невідомий ID); `AlreadyClaimed { agent, started_at }`,
  коли задача `in progress` — викликач має пройти процедуру запиту з
  `AGENTS.md` §3–4 і повторювати спробу лише після явного схвалення.
- Приклад: `reg.claim("P1.4", "fable")?; // do work …; reg.release("P1.4", "fable")?;`
- Перевірка: тест подвійної заявки — друга `claim` завершується помилкою з ім'ям і часом
  власника, перший власник не зачіпається.

### `fn release(&self, task_id: &str, agent: &str) -> Result<(), ClaimError>`

- Ефект: повертає рядок у стан `free` (agent/started очищуються). Викликайте при
  **кожній** зупинці чи завершенні, включно з невдачами та перериваннями.
- Помилки: `NotFound` (невідомий ID); `NotOwner { agent }`, коли задачу утримує інший
  агент — питайте, а не силуйте.
- Перевірка: тест звільнення — рядок повертається у `free`, і на нього знову можна подати заявку.

---

## Формат файлу реєстру (`docs/tasks.md`)

Таблиця Markdown: `| Task | Status | Agent | Started (UTC) |`.
`Status` ∈ `free` | `in progress`. Рядки `in progress` ПОВИННІ містити агента +
`started_at` у форматі RFC 3339; у рядках `free` обидва поля ПОВИННІ бути порожніми. Лінтер (P7.6)
забезпечує це, а також відсутність задач, що утримуються двічі, і позначає заявки, старші за 7 днів.

---

## P9.3 — Розміщення через RPC (розподілений інференс, заблоковано)

Крейт: `runa-engine` (`placement.rs`, `load.rs`, `prompt_cache.rs`).
Закріплений `llama-cpp-sys-2 0.1.133` вирізає бекенд RPC з ggml, тож
повноцінний розподілений інференс заблоковано (вердикт спайку + шлях розблокування в
`docs/versions.md`, «`GGML_RPC` unavailable on this pin»). Доки не з'явиться
форк sys-крейта (або підвищення pin, що поверне вихідний код), API
несе намір і гучно падає — ніколи мовчки не запускається локально, коли
було запитано розподілене виконання.

### `fn parse_rpc_list(s: &str) -> Result<Vec<String>, String>`

- Параметри: `s` — розділені комами ендпоінти `host:port` екземплярів `rpc-server`,
  напр. `"127.0.0.1:50052,10.0.0.2:50052"`.
- Повертає: обрізані непорожні записи; `Err("--rpc: empty list")`, коли
  нічого не лишається. Мінімальна перевірка форми, як у `parse_device_list`, — імена хостів,
  літерали IPv4 та IPv6 у квадратних дужках проходять без змін.
- Приклад: `parse_rpc_list("node1:50052")? // ["node1:50052"]`
- Перевірка: модульний тест `parse_rpc_list_csv` (`placement.rs`).

### `Placement::rpc_servers: Vec<String>`

- Поле в `Placement` (типово порожнє = локальний інференс у кожному
  конструкторі: `cpu`, `gpu`, `hybrid_moe`).
- Примітки: `prefix_key` гешує список, тож збірка з увімкненим RPC згодом
  ніколи не ділитиме стан кешу промптів із локальними запусками.

### `fn with_rpc_servers(self, rpc_servers: Vec<String>) -> Placement`

- Ефект: білдер, що записує ендпоінти RPC llama.cpp у розміщення.
- Приклад: `Placement::gpu().with_rpc_servers(parse_rpc_list(s)?)`
- Примітки: `load` друкує ендпоінти в рядку вердикту (суфікс `rpc=…`),
  а потім відхиляє непорожній список з
  `EngineError::Unsupported("--rpc …")` до ініціалізації бекенду — жодного
  глобального стану не зачеплено.
- Перевірка: `with_rpc_servers_preserves_mode`,
  `verdict_line_lists_rpc_servers` (`load.rs`) та інтеграційний
  тест `rpc_servers_fail_unsupported_before_backend_init`
  (`crates/runa-engine/tests/load.rs`).

## P9.4 — Перевірка NPU і заглушки швидкості (Tier 3)

Крейт: `runa-fit` (`npu`, `speed`); інтерфейс CLI в бінарнику `runa`
(`RUNA_NPU`, суфікс вердикту `runa auto`, рядки заглушок у `runa doctor`).
Лише каркас Tier-3: у `llama-cpp-2` аж до 0.1.154 немає NPU-бекенду ggml,
тож ніщо тут не переміщує тензори (огляд закріплень див. у `docs/versions.md`).

### `runa_fit::npu::NpuKind`

- Варіанти: `Hexagon` (Qualcomm Hexagon DSP/NPU), `OpenVino` (Intel NPU
  через OpenVINO).
- `fn as_str(self) -> &'static str` — канонічна назва
  (`hexagon` / `openvino`).
- `fn parse(s: &str) -> Option<NpuKind>` — назви без урахування регістру
  (`1`/`hexagon`/`qcom`/`snapdragon`, `openvino`/`ov`/`intel-npu`);
  `None` для всього іншого.
- `fn hw_spec(self) -> HwSpec` — консервативна заглушка для сімейства
  (`HwSpec::hexagon` / `HwSpec::openvino`).
- `Display` друкує `as_str`.

### `fn runa_fit::npu_present() -> Option<NpuKind>`

- Повертає: `Some(kind)`, коли NPU присутній (справжній чи імітований), інакше `None`.
  Шлях через справжнє обладнання — лише для Linux; інші ОС повідомляють про відсутність,
  доки власник пристрою не перевірить там маркер.
- Тестовий хук `RUNA_FAKE_NPU`: `1`/`hexagon` → `Hexagon`,
  `openvino` → `OpenVino`, `0`/`no`/`off`/`none` → примусова відсутність; не задано
  (або нерозпізнане значення) → апаратна евристика.
- Евристика: `/proc/device-tree/compatible` містить `qcom` → Hexagon;
  інакше існує `/dev/accel` → OpenVINO. Консервативна й не перевірена
  на пристрої (Tier 3).

### `fn runa_fit::probe_markers(device_tree_compatible: &Path, accel_dir: &Path) -> Option<NpuKind>`

- Ефект: евристика `npu_present` із підставними шляхами маркерів, щоб
  модульні тести ніколи не торкалися справжньої файлової системи чи середовища.
- Перевірка: `qcom` у файлі compat перемагає `/dev/accel`; немає маркерів →
  `None`.

### `HwSpec::hexagon() / HwSpec::openvino()`

- Повертає: консервативні некалібровані заглушки — Hexagon: 60 GB/s, 45 TOPS,
  ефективність 0.30; OpenVINO: 40 GB/s, 13 TOPS (найслабша SKU), ефективність
  0.30. Обидві за побудовою занижують прогноз порівняно з CUDA на тій самій моделі.
- Обмеження (див. `docs/fit.md`): орієнтація на Q4_0, OpenVINO лише для тексту, розбиття
  на вікна DSP Hexagon ~3.5 GiB. Перекалібруйте за `runa bench` на пристрої,
  перш ніж називати швидкості NPU.

### Інтерфейс бінарника (`runa`)

- `RUNA_NPU=hexagon|openvino` вмикає в рядку вердикту `runa auto`
  суфікс NPU: `NPU <kind> present (opt-in stub): ~X tok/s decode
  (uncalibrated …); placement stays CPU` у разі збігу або явне
  `requested but …; staying on CPU (explicit, no silent fallback)`, коли
  перевірка не підтверджує. Без `RUNA_NPU` вердикт ніколи не згадує NPU;
  нерозпізнане значення — явна помилка.
- `runa doctor` показує `hexagon-stub` / `openvino-stub` лише в бінарниках,
  зібраних із `--features hexagon` / `openvino`; типові бінарники не показують
  жодного. Скрипт збірки `runa-engine` попереджає, коли ввімкнено функцію-заглушку
  (завжди: статус заглушки; плюс у разі відсутності SDK: `HEXAGON_SDK_ROOT` /
  `INTEL_OPENVINO_DIR`).

## P9.2

Бекенд mistral.rs для моделей safetensors / omni, які ggml не може запустити.
`--backend gguf|mistral|auto` для `run` / `chat` / `serve` (типово `auto`).
Опційна cargo-функція `mistralrs` (`runa-engine`, передається далі через `runa`);
`runa doctor` показує `mistralrs`, коли її вкомпільовано.

### `runa_core::BackendKind`

- Варіанти: `Auto` (типово) | `Gguf` | `Mistral`.
- `fn parse(s: &str) -> Option<BackendKind>` — `auto` | `gguf` | `mistral`,
  без урахування регістру.
- `fn as_str(self) -> &'static str` (+ `Display`).
- `fn detect_backend(path: &Path) -> Result<BackendKind, String>` — файл `.gguf`
  → `Gguf`; каталог, що містить `config.json`, → `Mistral`; усе інше —
  явна помилка (ніколи не мовчазний запасний варіант).
- `fn resolve_backend(requested: BackendKind, path: &Path) -> Result<BackendKind, String>` —
  `Auto` визначає автоматично; явно заданий тип перевіряється щодо шляху (невідповідність
  одразу дає помилку, напр. `--backend mistral` для файлу `.gguf`).
- `fn is_gguf_file(path: &Path) -> bool`,
  `fn is_mistral_dir(path: &Path) -> bool` — два предикати, описані вище.

### `runa_engine::MistralModel` (функція `mistralrs`)

- `fn load(path: &Path) -> Result<MistralModel, EngineError>` — перевіряє
  каталог (`config.json`) до будь-якого виклику mistral.rs (швидка офлайнова
  помилка), потім завантажує через `ModelBuilder` + `blocking::BlockingModel`
  (власний рантайм tokio; не можна запускати всередині наявного рантайму — `run` /
  `chat` синхронні, `serve` використовує звичайні std-потоки рушія).
- `fn path(&self) -> &Path`.
- `fn generate(&mut self, req: GenerateRequest) -> Result<MistralGeneration, EngineError>` —
  `MistralGeneration: Iterator<Item = Result<GenEvent, EngineError>>` із
  тим самим порядком завершальних подій, що й у ggml (`Text…`, `Usage`, `Done`).
- `fn ensure_backend_available(kind: BackendKind) -> Result<(), EngineError>` —
  компілюється завжди; `Mistral` без цієї функції дає помилку з підказкою
  перезібрати (`--features mistralrs`).
- `EngineError::Mistral(String)` — збої завантаження/генерації та опції лише для GGUF,
  використані з `--backend mistral`.

### Відображення запиту (`GenerateRequest` → mistral.rs)

| Поле runa | mistral.rs |
|---|---|
| повідомлення (`system`/`user`/`assistant`) | `RequestBuilder::add_message` (ролі `tool` відхиляються) |
| `think.mode != Off` | `enable_thinking(bool)`; `think.show` керує подіями `Reasoning` |
| temperature ≤ 0 | `set_deterministic_sampler()` |
| temperature / top_k (> 0) / top_p / min_p | `set_sampler_temperature/topk/topp/minp` |
| `max_tokens` | `set_sampler_max_len` |
| `stop` | `StopTokens::Seqs` |
| завершення `length` / інше | `Done(MaxTokens)` / `Done(Eos)` |
| usage | токени промпту/відповіді + pp/tg tok/s |

Не відображаються, явно відхиляються (ніколи мовчки): `tools` (`--mcp`),
`json_schema`/`grammar`, `audio_pcm`/`images` (mtmd), `speculative`
(ngram/draft). Не відображаються, приймаються як лише для ggml (див. довідку `--backend`):
`seed` (немає відповідника в mistral.rs), `--mode`/`--ctx` (mistral автоматично розподіляє
пристрої/контекст). Запитані інструменти відхиляються, тож поява `ToolCalls`
неможлива; виклики інструментів, які модель робить із власної ініціативи (без жодного запиту), з'являються як
текст JSON — так само, як бекенд ggml показує незапитану розмітку інструментів.

### Pull / fit / serve / doctor

- `runa pull hf:<repo>:safetensors` завантажує знімок
  (`config.json`, токенізатор `*.json`, `*.index.json`, `*.safetensors`,
  лише пласка структура) у сховище моделей з перевіркою розміру + SHA-256 для кожного файлу
  та супровідними файлами `.verified`, як і для GGUF.
- `runa_fit::is_safetensors_tag(tag)` (без урахування регістру);
  `Fetcher::siblings_all(repo)` (усі rfilename; `siblings` — це
  відфільтроване за gguf подання); `RemoteError::Safetensors` відмовляє в `fetch_header`
  для посилань на safetensors, а `runa fit` відмовляє для каталогів mistral —
  обидва з явним повідомленням.
- `serve` визначає бекенд для кожної моделі (`Auto` визначає автоматично); потік пулу
  тримає `LocalEngine` (`run`/`chat` спільно використовують цей enum у `runa/src/engine.rs`);
  `/v1/embeddings` на моделі mistral явно дає помилку (лише gguf).

## P9.1 (фоновий сервіс `runa daemon`)

Демон тримає моделі «теплими» між викликами CLI, володіє одним
`MemoryManager` над справжнім бекендом RSS і обслуговує `run` / `chat` через
Unix-сокет (`~/.cache/runa/runa.sock`, перевизначається через `RUNA_DAEMON_SOCK`).
Протокол: один рядок NDJSON [`DaemonRequest`] на запит, потік рядків
[`DaemonEvent`], що завершується `done` / `error`, на відповідь
(`crates/runa/src/daemon_proto.rs`). `run` / `chat` спершу підключаються, а в разі відмови
повертаються до завантаження в процесі (`--no-daemon` пропускає підключення);
медіа, MCP, спекуляція та прапорці, що впливають на завантаження, завжди лишаються локальними.
Попередня перевірка кожного запиту — `touch()` + `on_heavy(0)`; допуск
обмежується пулом/LRU. Такт простою викликає `maybe_idle()` щонайменше кожні
`MAX_IDLE_TICK_SECS`.

### `fn SysinfoBackend::new() -> SysinfoBackend`

- Повертає: справжній бекенд RSS (P9.1). `rss_mib()` читає резидентний набір
  цього процесу через `sysinfo`; `shrink_to` / `grow` — рекомендаційні операції без ефекту,
  що повертають поточну RSS (сторінками володіє ОС — звільнення відбувається через
  `LoadedModel::on_idle` і витіснення LRU з пулу).
- Приклад: `MemoryManager::new(policy, ceiling, Box::new(SysinfoBackend::new()))`
- Примітки: замінює `FakeBackend` у місцях виклику `run` / `serve` / демона;
  модульні тести й далі використовують `FakeBackend`.

### `fn SysinfoBackend::process_rss_mib() -> u64`

- Повертає: поточну RSS процесу в MiB, `0`, коли таблицю процесів
  неможливо прочитати. Чисте спостереження, без побічних ефектів.

### `fn default_socket_path() -> PathBuf`

- Повертає: шлях до сокета демона — `RUNA_DAEMON_SOCK`, якщо задано, інакше
  `$XDG_CACHE_HOME/runa/runa.sock` або `~/.cache/runa/runa.sock`.

### `fn ModelPool::insert_spec(&mut self, path: &Path) -> Result<String, String>`

- Параметри: `path` — файл моделі, яку слід обслуговувати на вимогу.
- Повертає: id у пулі (`Ok`): наявний id, якщо шлях відомий,
  інакше основу назви файлу (`stem-2`, … у разі колізії). `Err`, коли файл
  відсутній. Ніколи не вивантажує моделі.

### `fn resolve_or_insert(pool: &Mutex<ModelPool>, model: Option<&str>) -> Result<String, String>`

- Ефект: спершу `resolve_id`; коли модель — це шлях на диску, якого пул
  ще не знає, викликає для нього `insert_spec` і повертає новий id.
- Помилки: `model <id> not found`, коли модель не є ні відомим id,
  ні наявним файлом.

### `fn generate(pool: &Arc<Mutex<ModelPool>>, model_id: &str, req: GenerateRequest) -> Result<Vec<GenEvent>, String>`

- Ефект: розв'язує + (блокувально) завантажує рушій, запускає одну генерацію в
  його потоці, збирає події. Асинхронна обгортка — ніколи не блокує
  виконавця. `serve` має власний варіант із відображенням на статуси.

### `fn request_sync(socket: &Path, req: &DaemonRequest, timeout: Duration) -> Result<Vec<DaemonEvent>, String>`

- Повертає: події демона аж до `done` / `error` включно.
- Помилки: збої транспорту означають «демона немає» (викликач повертається до запасного шляху);
  збій на боці демона надходить як `DaemonEvent::Error` всередині `Ok`.
  Заглушка для не-Unix систем завжди повертає помилку (лише Unix-сокети).

### `fn install_daemon(home: &Path, exe: &Path, argv: &[String]) -> Result<Vec<PathBuf>, String>`

- Ефект: записує plist для launchd
  (`~/Library/LaunchAgents/ai.runa.daemon.plist`) і користувацький юніт systemd
  (`~/.config/systemd/user/runa-daemon.service`) для `exe argv…`;
  повертає обидва шляхи. Перезапис ідемпотентний.

### `fn uninstall_daemon(home: &Path) -> Result<Vec<PathBuf>, String>`

- Ефект: видаляє обидва юніти; повертає видалені шляхи (порожньо, коли
  нічого не було встановлено). Відсутні файли не є помилкою.

### `fn launchd_plist(exe: &Path, argv: &[String]) -> String`

- Повертає: текст plist для launchd (`ai.runa.daemon`, `RunAtLoad` +
  `KeepAlive`, логи в `~/.cache/runa/daemon.{out,err}.log`).

### `fn systemd_unit(exe: &Path, argv: &[String]) -> String`

- Повертає: текст користувацького юніта systemd (`Restart=on-failure`,
  `WantedBy=default.target`).

## P10.1 — Прогнози швидкості з урахуванням калібрування (продовження M3)

Крейт: `runa-fit` (`speed::apply_efficiency`); під'єднано в бінарнику `runa`
(`bench::predicted_speeds`, `fit::calibrate_report`,
`fit::calibrate_pick`). `runa bench` записує пари виміряне/прогнозоване;
тепер прогнози масштабуються за медіаною відношення виміряне/прогнозоване для
точної трійки `(device, backend, quant)`. Порожня або відсутня база нічого
не змінює (сирий вивід моделі, поведінка M3 без калібрування).

### `fn runa_fit::apply_efficiency(pp: f64, tg: f64, eff: Option<&Efficiency>) -> (f64, f64)`

- Параметри: сирі прогнози префілу/декодування + результат `CalibrationDb::get_efficiency`
  для `(device, backend, quant)` запуску.
- Повертає: `(pp × pp_efficiency, tg × tg_efficiency)`. `None` повертає
  вхідні значення без змін; недодатний або нескінченний коефіцієнт вважається
  відсутнім лише для цієї осі (один поганий запуск бенчмарку ніколи не обнуляє рядок).
- Перевірка: `apply_efficiency_scales_both_axes`,
  `apply_efficiency_missing_db_keeps_raw`,
  `apply_efficiency_ignores_bad_factors_per_axis` (`speed.rs`).

### Інтерфейс бінарника (`runa`)

- `bench::predicted_speeds` завантажує `default_calibration_path()`
  (з урахуванням `RUNA_CALIBRATION`) і бере ключ `(device_backend(placement),
  quant_from_name(path))`, перш ніж друкувати чи записувати прогнози.
  Примітка: записані значення `predicted_*` — уже після калібрування, тож
  відношення наступного зразка вимірює залишкову похибку, а не сиру модель.
- `fit::calibrate_report` масштабує рядок `speed_gpu` коефіцієнтом ефективності для GPU
  (`metal:0`/`metal` на macOS, інакше `cuda:0`/`cuda`), а рядок
  `speed_cpu` — за `("cpu", "cpu", quant)`; `quant` береться з
  `calibration_quant` (локальна назва файлу, файл/квантизація HF або хвіст URL).
  Попередження `DecodeSlow` і далі використовують некалібровану перевірку всередині
  `check_fit` (бар'єр на рівні порядку величини).
- `fit::calibrate_pick` масштабує кожен вибір `--recommend` для того боку, який
  обрав його вердикт (гібридний використовує коефіцієнт GPU — те саме
  наближення, що задокументоване в `decode_for`).
- Перевірка: `fit::tests::{entry_quant_…, calibration_quant_…,
  calibrate_pick_…}`; e2e `fit_calibration_db_scales_predictions`
  (зразок 2.0× подвоює декодування в `runa fit --json` на фікстурі qwen2).

## P10.2 — Параметр `--threads` + типове значення з урахуванням P-ядер (продовження M4)

Крейт: `runa-engine` (`load::default_threads`); інтерфейс CLI в
бінарнику `runa` (`--threads` для `run` / `chat` / `bench` / `serve`,
`RUNA_THREADS`, `[defaults] threads`). Раніше кожне завантаження використовувало всі
логічні CPU разом з E-ядрами і програвало автоматичному вибору потоків у `llama-bench`
(M4); тепер типове значення збігається з власним значенням llama.cpp (`cpu_get_num_math`):
P-ядра на Apple Silicon, логічні CPU деінде.

### `fn default_threads() -> i32` (`runa-engine/src/load.rs`)

- Повертає: `hw.perflevel0.logicalcpu` на macOS (P-ядра через
  `sysctlbyname`, залежність `libc` лише для macOS), інакше
  `available_parallelism`; завжди `>= 1` (запасне значення 4, коли визначити неможливо).
- `fn apple_pcore_threads() -> Option<i32>` — сире читання sysctl;
  `None` у разі будь-якого збою (Intel Mac без рівнів продуктивності, неповне читання,
  абсурдне значення), тож викликач мовчки повертається до логічних CPU.
- Перевірка: `threads_tests::{default_threads_is_sane,
  apple_pcore_reading_is_plausible}`.

### `fn config::resolve_threads(cli: Option<i32>) -> Result<Option<i32>, String>`

- Пріоритет: CLI `--threads` > `RUNA_THREADS` > `[defaults] threads`
  у файлах конфігурації (пізніші файли перемагають) > `None` (типове значення рушія).
- Помилки: будь-яке значення `< 1` (`--threads 0`, `RUNA_THREADS=lots` або
  некоректне `[defaults] threads`) дає помилку ще до будь-якого завантаження.
- Примітки: лише для завантажень gguf — `run`/`chat` з `--backend mistral` явно відхиляють
  `--threads` (mistral.rs сам керує своїми потоками); `--threads`
  також вимикає для `run` використання демона (прапорець, що впливає на завантаження, P9.1), як
  `--device`/`--kv*`.
- Перевірка: `config::tests::{defaults_threads_toml,
  resolve_threads_precedence}` (збереження/відновлення `RUNA_THREADS`),
  `daemon_gate_tests` (`threads` лишається локальним), e2e
  `threads_zero_fails_with_usage_error`; фікстури довідки trycmd для `run`/`chat`/`serve`
  + перегенерований `docs/runa-run.1`.

## P10.3 — Виправлення ASR для `--lang auto` (блокер M7)

Крейт: `runa-media` (`asr::AsrEngine::transcribe`). Типовий
`--lang auto` показував правильно визначену мову з порожнім
транскриптом; `--lang en` на тому самому аудіо працював правильно.

- Першопричина (перевірено у вендореному `whisper.cpp`,
  `whisper_full`): прапорець `detect_language` означає *лише визначити* —
  функція повертає 0 одразу після визначення, не декодуючи.
  Встановлення і прапорця, і `language = "auto"` (старий код) або
  лише прапорця (перша спроба виправлення) завжди дає нуль сегментів.
  Правильний виклик — `language = "auto"` з незаданим прапорцем:
  тоді whisper.cpp визначає мову автоматично *і* декодує.
- Виправлення: для `None | Some("auto")` лише `set_language(Some("auto"))`,
  ніколи не `set_detect_language(true)`.
- Перевірено наживо (кешований `ggml-base.bin`, CPU Apple Silicon):
  англійське мовлення, синтезоване через `say`, транскрибується побайтово однаково з
  `--lang auto` і `--lang en`; синусоїдна фікстура теж декодується однаково.
- Перевірка: `asr::tests::auto_detect_decodes_like_explicit` —
  запускається наживо лише за наявності кешованої моделі (у CI пропускається, як і інші
  тести, що залежать від моделі); доведено, що він ПАДАЄ на коді до виправлення і
  проходить після. Рядок M7 у `docs/release-1.0.md` оновлено (повторне вимірювання
  часу ще очікується).

## P10.5 — Такт простою serve/демона викликає `on_idle` (продовження M11)

`LoadedModel::on_idle` існував, але поза модульними тестами його ніщо не
викликало: `maybe_idle` демона керував лише (рекомендаційним, без ефекту)
`SysinfoBackend`, а serve узагалі не мав менеджера. Тепер пул володіє
активністю кожного рушія, і обидва довгоживучі процеси її обходять.

### `EngineJob::Idle` (`pool.rs`)

- Задача за принципом «запустив і забув»: потік рушія виконує `LocalEngine::on_idle`
  (ggml відмаплює кеш промптів LMDB, модель лишається; для mistral — нічого) і
  ніколи не зриває такт. Стає в чергу за будь-яким запитом, що виконується.
- `fn LocalEngine::on_idle(&mut self)` (`engine.rs`) — прямий виклик
  бекенду.

### Відстеження простою в `ModelPool` (`pool.rs`)

- Поля `last_used: HashMap<String, Instant>` (оновлюється
  `touch_lru` під час кожного звернення та нового завантаження, очищується під час витіснення LRU) і
  `idle_timeout: Duration` (`Duration::MAX` = вимкнено; serve/демон
  задають його з політики пам'яті через `with_idle_timeout`).
- `fn due_for_idle(&self, now: Instant) -> Vec<String>` — чисте
  рішення (рушії, не використовувані ≥ тайм-аут), тестується модульно без моделей.
- `fn idle_sweep(&mut self) -> Vec<String>` — надсилає `Idle` кожному
  рушію, якому вже час, заново проставляє йому мітку (один обхід на тайм-аут, без спаму на кожному такті),
  забуває мертві потоки рушіїв. Може ненадовго заблокуватися за зайнятим рушієм
  — викликайте лише з блокувального потоку.
- Перевірка: `idle_sweep_fires_once_per_timeout`,
  `idle_sweep_disabled_by_default`, `idle_sweep_drops_dead_engines`
  (кінці каналів підміняють потоки рушіїв).

### `async fn idle_tick(pool, mm, tag)` (`pool.rs`)

- Один такт для обох циклів: `mm.maybe_idle()` (стан/логи менеджера)
  плюс обхід пулу в іншому потоці (`spawn_blocking`), по одному рядку
  `<tag>: idle <id>: prompt cache released (model kept)` на кожен
  оброблений рушій.
- Serve створює власний `MemoryManager` у `listen` (політика з
  конфігурації, такт обмежено `MAX_IDLE_TICK_SECS`); демон повторно використовує
  наявний менеджер, і його такт тепер викликає `idle_tick` замість
  голого `maybe_idle`. Ендпоінти генерації (`chat/completions`,
  `messages`, `embeddings`, `transcriptions`, `serve_request` демона)
  викликають `touch` менеджера; `/health` і `/v1/models` навмисно цього
  не роблять, тож опитування моніторингу не можуть тримати рушії активними.
- Перевірка: e2e `serve_idle_tick_releases_prompt_cache`
  (`RUNA_MEMORY_IDLE_TIMEOUT_S=1`, перевіряє рядок логу обходу).
- Вердикт M11 (виміряно, `docs/release-1.0.md`): звільнення справжнє,
  але мізерне за RSS порівняно з резидентною моделлю (772.9 MiB до
  і після на qwen2-0.5B) — бар'єр лишається недосяжним без вивантаження
  моделі, яке D17 забороняє.

## P10.6 — Підтримка інструментів у потоковому SSE Anthropic

Крейт: `runa-cloud` (`anthropic::parse_sse`); цикл інструментів CLI
(`drain_anthropic`) уже обробляє `ToolUse` — лише потоковий
парсер губив блоки, тож виклик інструмента зі `stream: true` зникав.

### `struct SseTools` + `SseBlock` (`anthropic.rs`)

- Накопичує `content_block_start` (id/name для tool_use, типи блоків text/thinking)
  плюс фрагменти `input_json_delta` / `text_delta` /
  `thinking_delta` / `signature_delta` за індексом блоку
  (`BTreeMap` зберігає порядок у потоці).
- `fn flush(&mut self) -> Option<AnthropicEvent>` відновлює
  масив блоків вмісту та `ToolCall` (JSON із поганих фрагментів стає
  `{}`); блоки міркування зберігають свій потоковий підпис, якщо він є.
  Викидається один раз: на `message_delta` зі `stop_reason` (перед
  `Done`, на тій самій позиції, що й у `parse_message`) або в кінці транскрипту для
  обірваного потоку. Потоки звичайного тексту лишаються тихими.
- `push_sse_event` додатково поглинає фрагменти інструментів/підписів, щоб
  частковий JSON ніколи не просочувався в події `Text`/`Reasoning`.
- CLI навмисно лишає `stream: false`: лише відповідь цілим повідомленням
  зберігає підписи міркування для наступного раунду інструментів
  (коментар у місці запиту в `cloud.rs`).
- Перевірка: `sse_tool_use_reassembled_across_fragments` (два виклики,
  фрагментовані аргументи, порядок блоків, ToolUse перед Done, жодного просочення тексту),
  `sse_without_tools_emits_no_tool_use`; наявні
  `thinking_delta_and_refusal` + тести wiremock без змін.

## P10.7 — Офлайновий `--recommend` враховує KV + обчислення

Раніше `probe_offline` перевіряв вміщення лише за розмірами файлів, тож довгий контекст на
великій моделі вміщався офлайн, але не онлайн. Тепер каталог містить
два виміряні числа, єдиним джерелом яких раніше був заголовок.

### `CatalogEntry::{kv_mib_per_1k, compute_mib}` (`recommend.rs`)

- `kv_mib_per_1k`: MiB KV на 1024 токени ctx у f16, вимірюється один раз
  для кожного запису (`runa fit --json --ctx 8192` за заголовком; KV
  строго лінійний за ctx). `q8_0` зменшує його вдвічі, `q4_0` — вчетверо;
  невідомі рядки `kv_type` зберігають значення для f16 (ніколи мовчки не
  зменшувати потребу).
- `compute_mib`: MiB обчислювального буфера за ubatch 512, лінійно за
  `n_ubatch` (дзеркально до масштабування `estimate_compute`).
- `fn kv_bytes_for(ctx_len, kv_type) -> u64` і
  `fn compute_bytes_for(n_ubatch) -> u64` виконують масштабування (з округленням угору,
  не менше нуля).
- Потреба в `probe_offline` = ваги + проєктор + KV(ctx) +
  compute(ubatch); швидкість використовує ту саму форму байтів на токен, що й онлайн
  (active + KV/2). `catalog_is_sane` вимагає, щоб обидва числа були > 0 у
  кожному записі.
- Перевірка: `offline_counts_kv_and_compute` (KV перемикає Gpu→Cpu→NoFit,
  `q8_0` повертає Gpu, допоміжні функції масштабування); e2e
  `fit_recommend_offline_ranks_the_catalog` лишається зеленим без змін.

## P10.9 — Чат зберігає історію між ходами

`Session.history: Vec<ChatMessage>` передається з кожним ходом REPL/TUI
(P10.9). Раніше кожен хід надсилав одне повідомлення користувача; раунди інструментів
уже лишалися в межах ходу, але відповіді ніщо не переносило далі.

- `fn turn_messages(session, input) -> Vec<ChatMessage>` — додає
  нове повідомлення користувача, обрізає до ~75 % `session.ctx`, повертає повний
  транскрипт для запиту. Обрізання друкує `(history trimmed to
  fit ctx)`.
- `fn commit_history(session, sent, answer)` — після ходу з відповіддю
  історією стають надіслані повідомлення (вони пройшли весь цикл
  інструментів, тож кожен раунд там є) плюс фінальна відповідь асистента.
  Невдалі ходи нічого не записують (повторна спроба надсилає ту саму історію).
- `fn trim_history(history, budget) -> usize` — відкидає найстаріші *ходи*
  (до наступного повідомлення `user`, тож повідомлення `tool` ніколи не відриваються
  від свого виклику `assistant`); найновіше повідомлення завжди зберігається.
  `fn estimate_history_tokens` — евристичний запобіжник «символи/4» —
  рушій однаково гучно падає за межами справжнього ctx.
- `/reset` (REPL + TUI) і `/model` (нові ваги, застарілий транскрипт)
  очищають історію; `/mode` її зберігає. Протокол демона вже
  передає повні `messages` (`ProtoMessage` разом із викликами інструментів), тож
  змінювати протокол не довелося.
- Перевірка: `history_tests` (порядок обрізання/найновіше зберігається/жодних відірваних
  повідомлень tool/форма коміту); e2e `chat_second_turn_remembers_history`
  (фікстура qwen2 згадує «Ada», 3/3 локально) поруч із наявним
  тестом повторного використання контексту.

### P11.4 — Точне за токенізатором обрізання історії

Локальні ходи gguf рахують справжні токени: `LoadedModel::
count_history_tokens(messages)` токенізує вміст кожного повідомлення
власним токенізатором моделі (`AddBos::Never`) плюс ті самі +4 токени
накладних витрат шаблону на повідомлення, що й в оцінці; запобіжник ~75 % ctx
у `turn_messages` повторно перевіряється за цим точним підрахунком.
Шлях через сокет демона не має токенізатора (і бекенд mistral
його не відкриває), тож там лишається `estimate_history_tokens` (символи/4).
`trim_history` лишається обгорткою на основі оцінки над узагальненою
`trim_history_with`; `count_history_tokens_with` фіксує точний шлях
проти заглушки-лічильника в модульних тестах.
- Перевірка: `history_tests::{exact_counter_counts_known_string,
  trim_with_exact_counter_drops_oldest_turn_first,
  fallback_estimate_locked_on_known_string}`.

## P10.10 — Моделі split-GGUF у каталозі

`CatalogEntry.parts: Vec<String>` містить посилання на шарди 2..N (`ref` лишається
частиною 1); `size` для таких записів — сума байтів усіх частин.

### `fn CatalogEntry::all_refs(&self) -> Vec<&str>` + `fn combine_shard_weights(first, rest)` (`recommend.rs`)

- `probe_remote` завантажує заголовок кожної частини, будує дескриптор
  із частини 1 (шарди повторюють повні метадані) і підсумовує чотири
  групи ваг за всіма частинами перед `check_fit`. Однофайлові записи
  йдуть тим самим шляхом з одним посиланням — там поведінка не змінюється.
- `catalog_is_sane` тепер також вимагає, щоб кожне посилання (частина 1 + parts)
  було посиланням `hf:….gguf` без дублікатів. Жодного розділеного запису поки не постачається:
  кожна поточна модель уміщається в один файл, тож `parts` — це схема +
  підтримка перевірки з покриттям тестами, що чекає на першу шардовану модель,
  яка вміститься на обладнання Tier-1/2.
- Перевірка: `split_tests::{all_refs_orders_part_one_first,
  split_parts_combine_weights}` (справжні метадані заголовка qwen2 з
  перевизначеними групами — без третьої копії білдера синтетичного GGUF).

## P10.11 — Покриття формату інструментів Harmony + `thinking_forced_open`

Обидва залишки P8.2 закрито перевіркою, а не новими гілками коду:
власний обробник llama.cpp уже підтримує обидва формати.

- Harmony: закріплений llama.cpp постачає `common_chat_parse_gpt_oss`
  (канали commentary/analysis, отримувач `to=functions.<name>` у
  будь-якому порядку заголовків). `harmony_tool_reply_parses`
  (`structured.rs`) рендерить обов'язковий виклик інструмента через СПРАВЖНІЙ
  шаблон gpt-oss-20B і розбирає заготовлені відповіді в обох порядках
  заголовків — 1 виклик, `get_weather`, жодного просочення розмітки. Залежить від фікстури
  (модель на 11 GiB, лише для розробників, у CI пропускається).
- `thinking_forced_open`: внутрішній прапорець llama.cpp (задається з
  шаблону + нашого `enable_thinking`, споживається його парсером/граматикою);
  наш шлях рендерингу вже передає йому все потрібне. Доведено
  розширеним `think_budget_reports_reasoning_tokens` (інтеграційний тест `generate.rs`):
  Qwen3-8B завершує *обов'язковий* виклик інструмента в межах
  бюджету 64+8, а показана кількість токенів міркування лишається в межах budget+grace.
- Попутно відремонтовано фікстуру: `tests/fixtures/gpt-oss-20b-MXFP4.gguf`
  був обрізаний (11.4 з 12.1 GiB — тензори за межами файлу);
  завантаження відновлено з Hub до точного розміру в байтах із каталогу. Файли фікстур —
  це ваги в git-ignore, тож відстежувані файли не зачеплено.
- Перевірка: `cargo test -p runa-engine --lib harmony` +
  `--test generate think_budget`; рядок формату в `docs/structured.md`.

## P10.13 — Обмеження навантаження на систему `--max-load-percent` (чужа незавершена робота, доведена до кінця)

Знайдено в дереві без заявки й у стані, що не компілювався (поля CLI без
відповідних сигнатур); доведено до кінця тут: відсутнє прокидання сигнатур/полів,
відмова від демона, відхилення в бекенді mistral, документація.
Обмежує частку загальних системних ресурсів, яку може використати запуск, у відсотках
1..=100. За задумом лише з попередженнями: запуск завжди продовжується, рішення
ухвалює користувач (план D12 — гучно, але ніколи не блокуючи).

### `const DEFAULT_MAX_LOAD_PERCENT: u8` (= 80)

### `fn config::resolve_max_load_percent(cli: Option<u8>) -> Result<u8, String>`

- Пріоритет: CLI `--max-load-percent` > `RUNA_MAX_LOAD_PERCENT` >
  `[system] max_load_percent` (пізніші файли перемагають) > 80.
- Помилки: будь-яке значення поза 1..=100 на будь-якому рівні.

### `struct config::SystemSnapshot` + `fn config::read_system_snapshot() -> SystemSnapshot`

- Знімок на момент часу `{total_ram_bytes, avail_ram_bytes, cpu_count}` через
  `sysinfo`. `RUNA_FAKE_TOTAL_RAM_MIB` / `RUNA_FAKE_AVAIL_RAM_MIB` /
  `RUNA_FAKE_CPU_COUNT` перевизначають усі три значення для тестів.

### `fn config::system_load_warnings(snap, demand_bytes, threads, limit) -> Vec<String>`

- Чиста математика, по одному рядку `warning:` на кожен перевищений ресурс: потреба моделі в RAM
  понад обмеження, фоновий тиск на системну RAM понад обмеження,
  частка CPU для `--threads` понад обмеження. Кожен рядок називає значення, яке треба
  встановити, щоб запуск умістився.
- `fn config::warn_if_over_system_limit(demand, threads, cli)` —
  визначає обмеження й друкує рядки (шлях запуску: `run`,
  `chat`, `bench`, `serve`, `daemon`); повертає обмеження.
- `fn config::warn_if_demand_over_limit(demand, cli)` — лише рядок
  про потребу моделі (`preflight_grow`, після читання заголовка).
- Перевірка: `config::tests::max_load_*` (пріоритет TOML/середовища/CLI,
  попередження на підставному знімку); шляхи рівня e2e перевірено смоук-тестами через CLI
  (`--max-load-percent 0` і `RUNA_MAX_LOAD_PERCENT=lots` дають помилку
  до будь-якого завантаження).
- Примітки: `--max-load-percent` вимикає для `run` використання демона (демон
  керує розміщенням і не знає обмежень на окремий запит) і
  відхиляється з `--backend mistral` (там ніколи не попереджає, тож ніколи
  не приймається мовчки).
