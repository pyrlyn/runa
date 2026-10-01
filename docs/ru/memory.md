---
lang: ru
---

# docs/memory.md — публичные методы: `MemoryManager` и `TaskRegistry`

Крейт: `runa-memory` (план D17/D18, фаза P7, метрики M11/M12).
Цели: M11 (RSS в простое ≤ floor + 10 %), M12 (ни одна задача не захвачена дважды).
Конфигурация: `[memory] idle_timeout_s, floor_mib, max_growth_mib`;
`[agents] registry = "docs/tasks.md"`. Наброски — в `plan.md` §5.

Соглашения: `demand_mib`/`rss_mib` указываются в MiB (`u64`). Временные метки — строки UTC
в формате RFC 3339. Ошибки описаны для каждого метода; все методы
синхронные и потокобезопасные (`Send + Sync`).

---

## `MemoryManager`

Адаптивная память процесса. Простой (нет запросов/заданий в течение `idle_timeout_s`) →
освободить промпт-кеш, буферы энкодера, draft-модель, сжать пулы до
`floor_mib`. Тяжёлая задача → заранее нарастить арены до вердикта fit + запас, но
не более чем на `max_growth_mib` сверх текущего использования. Никогда не выгружает активную
модель. Каждый переход логирует RSS до и после.

### `fn current_usage(&self) -> Usage`

- Возвращает: `Usage { rss_mib, budget_mib, state }`, где `state` —
  `Idle` | `Normal` | `Heavy`, а `budget_mib` — текущий потолок
  (вердикт fit + запас + выделенный рост).
- Пример: `let u = mm.current_usage(); assert!(u.rss_mib <= u.budget_mib);`
- Примечания: чистое наблюдение, без побочных эффектов; `state` выводится из недавней
  нагрузки относительно `idle_timeout_s` и ожидающей потребности.

### `fn touch(&self)`

- Эффект: фиксирует, что запрос/задание выполняется. Сбрасывает таймер простоя.
- Когда вызывать: при старте каждой генерации / запроса к серверу (`grow_for` уже
  делает это).
- Примечания: не меняет `LoadState`.

### `fn maybe_idle(&self)`

- Эффект: если `last_activity` старше `idle_timeout_s`, вызывает
  `on_idle`. Ничего не делает, если работа была недавно (гистерезис).
- Когда вызывать: опрос после завершения запроса или на тике простоя сервера.
- Проверка: `idle_timeout_s = 0` сжимает немедленно; таймаут 3600 s
  не сжимает сразу после `touch`.

### `fn on_idle(&self)`

- Эффект: сжатие до `floor_mib` (освобождение кешей/буферов/пулов, как
  выше); ничего не делает, если уже на нижней границе или ниже. Логирует RSS до и после.
- Когда вызывать: нет ни запросов, ни заданий в течение полного `idle_timeout_s` (или из
  `maybe_idle`).
- Проверка: длительный тест M11 — RSS в простое ≤ floor + 10 %.
- Примечания: гистерезис — сжимать только после полной тишины, никогда посреди всплеска
  (см. Risks в `plan.md` §8).

### `fn on_heavy(&self, demand_mib: u64)`

- Параметры: `demand_mib` — ожидаемая дополнительная память для поступающей тяжёлой
  задачи (рост ctx, батч, медиа).
- Эффект: удобная обёртка — выделяет то, что помещается, через `grow_for`, а
  иначе сохраняет текущее размещение (вызывающий код откатывается к меньшему
  ctx/кванту или к облаку согласно совету fit). Никогда не превышает потолок.
- Пример: `mm.on_heavy(2048); // make room for a 2 GiB ctx bump`

### `fn shrink_to_floor(&self)`

- Эффект: безусловное сжатие до `floor_mib`, тот же набор освобождаемого, что и в
  `on_idle`, но немедленно (используется тестами и явными
  путями обслуживания в стиле `runa doctor --shrink`).
- Примечания: активная модель всё равно никогда не выгружается.

### `fn grow_for(&self, demand_mib: u64) -> Result<(), MemoryError>`

- Параметры: `demand_mib` — байты (MiB), которые нужны задаче сверх текущего использования.
- Возвращает: `Ok(())` после предварительного роста; `Err(MemoryError::OverCeiling {
  demand_mib, ceiling_mib, suggestion })`, когда потребность превышает
  вердикт fit + запас (`suggestion`: меньший ctx, другой квант,
  `--kv q8_0` или облако — тот же словарь, что и у `runa fit`).
- Проверка: тест с тяжёлым ctx проходит без OOM посреди запуска; тест превышения потолка
  проверяет, что `Err` содержит совет.

---

## `TaskRegistry`

Кооперативный захват задач в `docs/tasks.md` (хранится в файле). Состояния строк:
`free` | `in progress` (+ `agent`, `started_at`). Завершение отслеживается
отметкой в `plan.md`; реестр отслеживает только активные захваты.

### `fn list_free(&self) -> Vec<String>`

- Возвращает: ID всех задач `free` (например, `["P0.1", "P1.4", …]`), отсортированные
  в порядке фаз.
- Пример: `for id in reg.list_free() { println!("{id}"); }`

### `fn status(&self, task_id: &str) -> Option<TaskStatus>`

- Параметры: `task_id` — например, `"P2.6"`.
- Возвращает: `None` для неизвестных ID; `Some(Free)` или
  `Some(InProgress { agent, started_at })`.
- Примечания: только чтение; используйте, прежде чем спрашивать о задаче `in-progress`.

### `fn claim(&self, task_id: &str, agent: &str) -> Result<TaskClaim, ClaimError>`

- Параметры: `task_id`, `agent` (имя захватывающего агента; `started_at`
  проставляет реестр в момент захвата).
- Возвращает: `Ok(TaskClaim { task_id, agent, started_at })`, и строка
  становится `in progress`.
- Ошибки: `NotFound` (неизвестный ID); `AlreadyClaimed { agent, started_at }`,
  когда задача `in progress`, — вызывающий должен следовать процедуре запроса из
  `AGENTS.md` §3–4 и повторять попытку только при явном одобрении.
- Пример: `reg.claim("P1.4", "fable")?; // do work …; reg.release("P1.4", "fable")?;`
- Проверка: тест двойного захвата — второй `claim` завершается ошибкой с именем владельца
  и временем, первый владелец не затронут.

### `fn release(&self, task_id: &str, agent: &str) -> Result<(), ClaimError>`

- Эффект: сбрасывает строку в `free` (agent/started очищаются). Вызывайте при
  **каждой** остановке или завершении, включая сбои и прерывания.
- Ошибки: `NotFound` (неизвестный ID); `NotOwner { agent }`, когда задачу держит другой
  агент, — спросите, не применяйте силу.
- Проверка: тест освобождения — строка возвращается в `free` и снова доступна для захвата.

---

## Формат файла реестра (`docs/tasks.md`)

Таблица Markdown: `| Task | Status | Agent | Started (UTC) |`.
`Status` ∈ `free` | `in progress`. Строки `in progress` ОБЯЗАНЫ содержать агента +
`started_at` в RFC 3339; у строк `free` оба поля ОБЯЗАНЫ быть пустыми. Линтер (P7.6)
проверяет это, а также отсутствие двойного захвата и помечает захваты старше 7 дней.

---

## P9.3 — размещение через RPC (распределённый инференс, заблокировано)

Крейт: `runa-engine` (`placement.rs`, `load.rs`, `prompt_cache.rs`).
Закреплённый `llama-cpp-sys-2 0.1.133` вырезает бэкенд RPC из ggml, поэтому
полноценный распределённый инференс заблокирован (вердикт спайка + путь разблокировки в
`docs/versions.md`, "`GGML_RPC` unavailable on this pin"). Пока не появится
форк sys-крейта (или повышение пина, возвращающее исходники), API
выражает намерение и громко падает — никогда не запускается молча локально, если
было запрошено распределение.

### `fn parse_rpc_list(s: &str) -> Result<Vec<String>, String>`

- Параметры: `s` — эндпоинты `host:port` экземпляров `rpc-server` через запятую,
  например `"127.0.0.1:50052,10.0.0.2:50052"`.
- Возвращает: обрезанные непустые записи; `Err("--rpc: empty list")`, когда
  ничего не остаётся. Проверка формы мягкая, как у `parse_device_list`: имена хостов,
  IPv4 и литералы IPv6 в квадратных скобках проходят.
- Пример: `parse_rpc_list("node1:50052")? // ["node1:50052"]`
- Проверка: юнит-тест `parse_rpc_list_csv` (`placement.rs`).

### `Placement::rpc_servers: Vec<String>`

- Поле `Placement` (по умолчанию пустое = локальный инференс во всех
  конструкторах: `cpu`, `gpu`, `hybrid_moe`).
- Примечания: `prefix_key` хеширует список, поэтому сборка с поддержкой RPC в будущем
  никогда не разделит состояние промпт-кеша с локальными запусками.

### `fn with_rpc_servers(self, rpc_servers: Vec<String>) -> Placement`

- Эффект: билдер, записывающий эндпоинты RPC llama.cpp в размещение.
- Пример: `Placement::gpu().with_rpc_servers(parse_rpc_list(s)?)`
- Примечания: `load` выводит эндпоинты в строке вердикта (суффикс `rpc=…`),
  а затем отклоняет непустой список с
  `EngineError::Unsupported("--rpc …")` до инициализации бэкенда — никакое
  глобальное состояние не затрагивается.
- Проверка: `with_rpc_servers_preserves_mode`,
  `verdict_line_lists_rpc_servers` (`load.rs`) и интеграционный
  тест `rpc_servers_fail_unsupported_before_backend_init`
  (`crates/runa-engine/tests/load.rs`).

## P9.4 — проба NPU и заглушки скорости (Tier 3)

Крейт: `runa-fit` (`npu`, `speed`); поверхность CLI в бинарнике `runa`
(`RUNA_NPU`, суффикс вердикта `runa auto`, строки заглушек в `runa doctor`).
Только заготовка Tier-3: в `llama-cpp-2` вплоть до 0.1.154 нет бэкенда NPU для ggml,
поэтому ничто здесь не перемещает тензоры (см. обзор пинов в `docs/versions.md`).

### `runa_fit::npu::NpuKind`

- Варианты: `Hexagon` (Qualcomm Hexagon DSP/NPU), `OpenVino` (Intel NPU
  через OpenVINO).
- `fn as_str(self) -> &'static str` — каноническое имя
  (`hexagon` / `openvino`).
- `fn parse(s: &str) -> Option<NpuKind>` — имена без учёта регистра
  (`1`/`hexagon`/`qcom`/`snapdragon`, `openvino`/`ov`/`intel-npu`);
  `None` для всего остального.
- `fn hw_spec(self) -> HwSpec` — консервативная заглушка для семейства
  (`HwSpec::hexagon` / `HwSpec::openvino`).
- `Display` выводит `as_str`.

### `fn runa_fit::npu_present() -> Option<NpuKind>`

- Возвращает: `Some(kind)`, когда NPU присутствует (реальный или имитированный), иначе `None`.
  Чисто аппаратный путь есть только в Linux; другие ОС сообщают об отсутствии,
  пока владелец устройства не проверит там маркер.
- Тестовый хук `RUNA_FAKE_NPU`: `1`/`hexagon` → `Hexagon`,
  `openvino` → `OpenVino`, `0`/`no`/`off`/`none` → принудительное отсутствие; не задан
  (или не распознан) → аппаратная эвристика.
- Эвристика: `/proc/device-tree/compatible` содержит `qcom` → Hexagon;
  иначе, если существует `/dev/accel` → OpenVINO. Консервативна и не проверена
  на устройстве (Tier 3).

### `fn runa_fit::probe_markers(device_tree_compatible: &Path, accel_dir: &Path) -> Option<NpuKind>`

- Эффект: эвристика `npu_present` с внедряемыми путями маркеров, чтобы
  юнит-тесты никогда не трогали реальную файловую систему или окружение.
- Проверка: `qcom` в файле compat побеждает `/dev/accel`; нет маркеров →
  `None`.

### `HwSpec::hexagon() / HwSpec::openvino()`

- Возвращает: консервативные некалиброванные заглушки — Hexagon: 60 GB/s, 45 TOPS,
  эффективность 0.30; OpenVINO: 40 GB/s, 13 TOPS (младшая модель), эффективность
  0.30. Обе по построению дают заниженный прогноз по сравнению с CUDA на той же модели.
- Ограничения (см. `docs/fit.md`): ориентация на Q4_0, OpenVINO только для текста, у Hexagon
  разбиение по окну DSP ~3.5 GiB. Перекалибруйте по `runa bench` на устройстве,
  прежде чем называть скорости NPU.

### Поверхность бинарника (`runa`)

- `RUNA_NPU=hexagon|openvino` включает в строке вердикта `runa auto`
  суффикс NPU: `NPU <kind> present (opt-in stub): ~X tok/s decode
  (uncalibrated …); placement stays CPU` при совпадении или явное
  `requested but …; staying on CPU (explicit, no silent fallback)`, когда
  проба с этим не согласна. Без `RUNA_NPU` вердикт никогда не упоминает NPU;
  нераспознанное значение — явная ошибка.
- `runa doctor` перечисляет `hexagon-stub` / `openvino-stub` только в бинарниках,
  собранных с `--features hexagon` / `openvino`; бинарники по умолчанию не перечисляют
  ни то, ни другое. Скрипт сборки `runa-engine` выдаёт предупреждение, когда включена фича-заглушка
  (всегда: статус заглушки; плюс при отсутствии SDK: `HEXAGON_SDK_ROOT` /
  `INTEL_OPENVINO_DIR`).

## P9.2

Бэкенд mistral.rs для моделей safetensors / omni, которые ggml не может запустить.
`--backend gguf|mistral|auto` для `run` / `chat` / `serve` (по умолчанию `auto`).
Включаемая по желанию фича cargo `mistralrs` (`runa-engine`, пробрасывается из `runa`);
`runa doctor` сообщает `mistralrs`, когда она вкомпилирована.

### `runa_core::BackendKind`

- Варианты: `Auto` (по умолчанию) | `Gguf` | `Mistral`.
- `fn parse(s: &str) -> Option<BackendKind>` — `auto` | `gguf` | `mistral`,
  без учёта регистра.
- `fn as_str(self) -> &'static str` (+ `Display`).
- `fn detect_backend(path: &Path) -> Result<BackendKind, String>` — файл `.gguf`
  → `Gguf`; каталог с `config.json` → `Mistral`; всё остальное
  — явная ошибка (никогда не молчаливый откат).
- `fn resolve_backend(requested: BackendKind, path: &Path) -> Result<BackendKind, String>` —
  `Auto` определяет автоматически; явно указанный вид проверяется на соответствие пути (несоответствие
  сразу приводит к ошибке, например `--backend mistral` для файла `.gguf`).
- `fn is_gguf_file(path: &Path) -> bool`,
  `fn is_mistral_dir(path: &Path) -> bool` — два предиката выше.

### `runa_engine::MistralModel` (фича `mistralrs`)

- `fn load(path: &Path) -> Result<MistralModel, EngineError>` — проверяет
  каталог (`config.json`) до любого вызова mistral.rs (быстрая ошибка
  офлайн), затем загружает через `ModelBuilder` + `blocking::BlockingModel`
  (собственный рантайм tokio; нельзя запускать внутри существующего рантайма — `run` /
  `chat` синхронные, `serve` использует обычные std-потоки движка).
- `fn path(&self) -> &Path`.
- `fn generate(&mut self, req: GenerateRequest) -> Result<MistralGeneration, EngineError>` —
  `MistralGeneration: Iterator<Item = Result<GenEvent, EngineError>>` с
  тем же завершающим порядком, что и в ggml (`Text…`, `Usage`, `Done`).
- `fn ensure_backend_available(kind: BackendKind) -> Result<(), EngineError>` —
  компилируется всегда; `Mistral` без фичи завершается ошибкой с подсказкой
  пересобрать (`--features mistralrs`).
- `EngineError::Mistral(String)` — сбои загрузки/генерации и опции только для GGUF,
  использованные с `--backend mistral`.

### Сопоставление запроса (`GenerateRequest` → mistral.rs)

| Поле runa | mistral.rs |
|---|---|
| messages (`system`/`user`/`assistant`) | `RequestBuilder::add_message` (роли `tool` отклоняются) |
| `think.mode != Off` | `enable_thinking(bool)`; `think.show` управляет событиями `Reasoning` |
| temperature ≤ 0 | `set_deterministic_sampler()` |
| temperature / top_k (> 0) / top_p / min_p | `set_sampler_temperature/topk/topp/minp` |
| `max_tokens` | `set_sampler_max_len` |
| `stop` | `StopTokens::Seqs` |
| завершение `length` / другое | `Done(MaxTokens)` / `Done(Eos)` |
| usage | токены промпта/ответа + pp/tg tok/s |

Не сопоставляются и явно отклоняются (никогда молча): `tools` (`--mcp`),
`json_schema`/`grammar`, `audio_pcm`/`images` (mtmd), `speculative`
(ngram/draft). Не сопоставляются, но принимаются как опции только для ggml (см. справку `--backend`):
`seed` (в mistral.rs нет аналога), `--mode`/`--ctx` (mistral сам сопоставляет
устройства/контекст). Запрошенные инструменты отклоняются, поэтому выдача `ToolCalls`
невозможна; вызовы инструментов, сделанные по собственной инициативе (без какого-либо запроса), выдаются как
текст JSON — так же, как бэкенд ggml выдаёт незапрошенную разметку инструментов.

### Pull / fit / serve / doctor

- `runa pull hf:<repo>:safetensors` скачивает снапшот
  (`config.json`, `*.json` токенизатора, `*.index.json`, `*.safetensors`,
  только плоская структура) в хранилище моделей с проверкой размера + SHA-256 для каждого файла
  и файлами `.verified` рядом, как при скачивании GGUF.
- `runa_fit::is_safetensors_tag(tag)` (без учёта регистра);
  `Fetcher::siblings_all(repo)` (все rfilenames; `siblings` — представление,
  отфильтрованное по gguf); `RemoteError::Safetensors` отказывает в `fetch_header`
  для ссылок на safetensors, а `runa fit` отказывается работать с каталогами mistral —
  в обоих случаях с явным сообщением.
- `serve` разрешает бэкенд для каждой модели (`Auto` определяет автоматически); поток пула
  держит `LocalEngine` (`run`/`chat` используют это же перечисление в `runa/src/engine.rs`);
  `/v1/embeddings` на модели mistral явно завершается ошибкой (только gguf).

## P9.1 (фоновый сервис `runa daemon`)

Демон держит модели прогретыми между вызовами CLI, владеет одним
`MemoryManager` поверх реального бэкенда RSS и обслуживает `run` / `chat` через
Unix-сокет (`~/.cache/runa/runa.sock`, переопределяется через `RUNA_DAEMON_SOCK`).
Протокол: одна строка NDJSON [`DaemonRequest`] на запрос, поток строк
[`DaemonEvent`], заканчивающийся `done` / `error`, на каждый ответ
(`crates/runa/src/daemon_proto.rs`). `run` / `chat` сначала подключаются и при отказе
откатываются к загрузке внутри процесса (`--no-daemon` пропускает подключение);
медиа, MCP, спекуляция и флаги, влияющие на загрузку, всегда остаются локальными.
Предварительная проверка каждого запроса — `touch()` + `on_heavy(0)`; допуск
ограничен пулом/LRU. Тик простоя вызывает `maybe_idle()` не реже чем раз в
`MAX_IDLE_TICK_SECS`.

### `fn SysinfoBackend::new() -> SysinfoBackend`

- Возвращает: реальный бэкенд RSS (P9.1). `rss_mib()` читает резидентный набор
  этого процесса через `sysinfo`; `shrink_to` / `grow` — рекомендательные пустые операции,
  возвращающие текущий RSS (страницами владеет ОС — освобождение происходит через
  `LoadedModel::on_idle` и вытеснение LRU из пула).
- Пример: `MemoryManager::new(policy, ceiling, Box::new(SysinfoBackend::new()))`
- Примечания: заменяет `FakeBackend` в местах вызова `run` / `serve` / демона;
  юнит-тесты продолжают использовать `FakeBackend`.

### `fn SysinfoBackend::process_rss_mib() -> u64`

- Возвращает: текущий RSS процесса в MiB, `0`, если таблицу процессов
  невозможно прочитать. Чистое наблюдение, без побочных эффектов.

### `fn default_socket_path() -> PathBuf`

- Возвращает: путь к сокету демона — `RUNA_DAEMON_SOCK`, если задан, иначе
  `$XDG_CACHE_HOME/runa/runa.sock` или `~/.cache/runa/runa.sock`.

### `fn ModelPool::insert_spec(&mut self, path: &Path) -> Result<String, String>`

- Параметры: `path` — файл модели, который нужно обслуживать по запросу.
- Возвращает: id в пуле (`Ok`): существующий id, если путь уже известен,
  иначе основу имени файла (`stem-2`, … при коллизии). `Err`, если файл
  отсутствует. Никогда не выгружает модели.

### `fn resolve_or_insert(pool: &Mutex<ModelPool>, model: Option<&str>) -> Result<String, String>`

- Эффект: сначала `resolve_id`; если модель — путь на диске, который пул
  ещё не знает, выполняет для неё `insert_spec` и возвращает новый id.
- Ошибки: `model <id> not found`, если модель не является ни известным id,
  ни существующим файлом.

### `fn generate(pool: &Arc<Mutex<ModelPool>>, model_id: &str, req: GenerateRequest) -> Result<Vec<GenEvent>, String>`

- Эффект: разрешает и (блокирующе) загружает движок, выполняет одну генерацию в
  его потоке, собирает события. Асинхронная обёртка — никогда не блокирует
  исполнитель. У `serve` есть собственный вариант с сопоставлением статусов.

### `fn request_sync(socket: &Path, req: &DaemonRequest, timeout: Duration) -> Result<Vec<DaemonEvent>, String>`

- Возвращает: события демона вплоть до `done` / `error` включительно.
- Ошибки: сбои транспорта означают «демона нет» (вызывающий откатывается);
  сбой на стороне демона приходит как `DaemonEvent::Error` внутри `Ok`.
  Заглушка для не-unix систем всегда возвращает ошибку (только unix-сокеты).

### `fn install_daemon(home: &Path, exe: &Path, argv: &[String]) -> Result<Vec<PathBuf>, String>`

- Эффект: записывает plist для launchd
  (`~/Library/LaunchAgents/ai.runa.daemon.plist`) и пользовательский юнит systemd
  (`~/.config/systemd/user/runa-daemon.service`) для `exe argv…`;
  возвращает оба пути. Перезапись идемпотентна.

### `fn uninstall_daemon(home: &Path) -> Result<Vec<PathBuf>, String>`

- Эффект: удаляет оба юнита; возвращает удалённые пути (пусто, если
  ничего не было установлено). Отсутствующие файлы не считаются ошибкой.

### `fn launchd_plist(exe: &Path, argv: &[String]) -> String`

- Возвращает: текст plist для launchd (`ai.runa.daemon`, `RunAtLoad` +
  `KeepAlive`, логи в `~/.cache/runa/daemon.{out,err}.log`).

### `fn systemd_unit(exe: &Path, argv: &[String]) -> String`

- Возвращает: текст пользовательского юнита systemd (`Restart=on-failure`,
  `WantedBy=default.target`).

## P10.1 — прогнозы скорости с учётом калибровки (продолжение M3)

Крейт: `runa-fit` (`speed::apply_efficiency`); подключено в бинарнике `runa`
(`bench::predicted_speeds`, `fit::calibrate_report`,
`fit::calibrate_pick`). `runa bench` записывает пары измеренное/прогнозное;
теперь прогнозы масштабируются по медиане отношения измеренное/прогнозное для
точной тройки `(device, backend, quant)`. Пустая или отсутствующая база ничего
не меняет (сырой вывод модели, поведение M3 без калибровки).

### `fn runa_fit::apply_efficiency(pp: f64, tg: f64, eff: Option<&Efficiency>) -> (f64, f64)`

- Параметры: сырые прогнозы префилла/декодирования + результат `CalibrationDb::get_efficiency`
  для `(device, backend, quant)` запуска.
- Возвращает: `(pp × pp_efficiency, tg × tg_efficiency)`. `None` возвращает
  входные значения без изменений; неположительный или нечисловой коэффициент считается
  отсутствующим только для этой оси (один плохой запуск бенчмарка никогда не обнуляет строку).
- Проверка: `apply_efficiency_scales_both_axes`,
  `apply_efficiency_missing_db_keeps_raw`,
  `apply_efficiency_ignores_bad_factors_per_axis` (`speed.rs`).

### Поверхность бинарника (`runa`)

- `bench::predicted_speeds` загружает `default_calibration_path()`
  (учитывает `RUNA_CALIBRATION`) и строит ключ `(device_backend(placement),
  quant_from_name(path))`, прежде чем выводить или записывать прогнозы.
  Примечание: записанные значения `predicted_*` уже откалиброваны, поэтому
  отношение в следующем замере измеряет остаточную ошибку, а не сырую модель.
- `fit::calibrate_report` масштабирует строку `speed_gpu` по эффективности
  на стороне GPU (`metal:0`/`metal` на macOS, иначе `cuda:0`/`cuda`), а
  строку `speed_cpu` — по `("cpu", "cpu", quant)`; `quant` берётся из
  `calibration_quant` (локальное имя файла, файл/квант HF или хвост URL).
  Предупреждения `DecodeSlow` по-прежнему используют некалиброванную проверку внутри
  `check_fit` (критерий порядка величины).
- `fit::calibrate_pick` масштабирует каждый вариант `--recommend` для той стороны, которую
  выбрал его вердикт (гибрид использует коэффициент стороны GPU — то же
  приближение, которое описывает `decode_for`).
- Проверка: `fit::tests::{entry_quant_…, calibration_quant_…,
  calibrate_pick_…}`; e2e `fit_calibration_db_scales_predictions`
  (замер 2.0× удваивает декодирование в `runa fit --json` на фикстуре qwen2).

## P10.2 — настройка `--threads` + значение по умолчанию с учётом P-ядер (продолжение M4)

Крейт: `runa-engine` (`load::default_threads`); поверхность CLI в
бинарнике `runa` (`--threads` для `run` / `chat` / `bench` / `serve`,
`RUNA_THREADS`, `[defaults] threads`). Раньше каждая загрузка использовала все
логические CPU, включая E-ядра, и проигрывала автоматическому выбору потоков в `llama-bench`
(M4); теперь значение по умолчанию совпадает с собственным значением llama.cpp (`cpu_get_num_math`):
P-ядра на Apple Silicon, логические CPU на остальных платформах.

### `fn default_threads() -> i32` (`runa-engine/src/load.rs`)

- Возвращает: `hw.perflevel0.logicalcpu` на macOS (P-ядра через
  `sysctlbyname`, зависимость `libc` только для macOS), иначе
  `available_parallelism`; всегда `>= 1` (запасное значение 4, если определить невозможно).
- `fn apple_pcore_threads() -> Option<i32>` — сырое чтение sysctl;
  `None` при любой неудаче (Intel Mac без уровней производительности, неполное чтение,
  абсурдное значение), чтобы вызывающий молча откатился к логическим CPU.
- Проверка: `threads_tests::{default_threads_is_sane,
  apple_pcore_reading_is_plausible}`.

### `fn config::resolve_threads(cli: Option<i32>) -> Result<Option<i32>, String>`

- Приоритет: CLI `--threads` > `RUNA_THREADS` > `[defaults] threads`
  в файлах конфигурации (более поздние файлы побеждают) > `None` (значение движка по умолчанию).
- Ошибки: любое значение `< 1` (`--threads 0`, `RUNA_THREADS=lots` или
  некорректный `[defaults] threads`) приводит к ошибке до загрузки.
- Примечания: только для загрузок gguf — `run`/`chat` с `--backend mistral` явно отклоняют
  `--threads` (mistral.rs сам управляет своими потоками); `--threads`
  также заставляет `run` обойти демон (флаг, влияющий на загрузку, P9.1), как
  `--device`/`--kv*`.
- Проверка: `config::tests::{defaults_threads_toml,
  resolve_threads_precedence}` (сохранение/восстановление `RUNA_THREADS`),
  `daemon_gate_tests` (`threads` остаётся локальным), e2e
  `threads_zero_fails_with_usage_error`; фикстуры справки trycmd для `run`/`chat`/`serve`
  + перегенерированный `docs/runa-run.1`.

## P10.3 — исправление ASR для `--lang auto` (блокер M7)

Крейт: `runa-media` (`asr::AsrEngine::transcribe`). `--lang auto` по умолчанию
сообщал правильно определённый язык и пустой
транскрипт; `--lang en` на том же аудио работал правильно.

- Первопричина (проверено в вендоренном `whisper.cpp`,
  `whisper_full`): флаг `detect_language` означает *только определить* —
  функция возвращает 0 сразу после определения, без декодирования.
  Установка и флага, и `language = "auto"` (старый код) или
  только флага (первая попытка исправления) всегда даёт ноль сегментов.
  Правильный вызов — `language = "auto"` без установленного флага:
  тогда whisper.cpp и определяет язык автоматически, *и* декодирует.
- Исправление: при `None | Some("auto")` только `set_language(Some("auto"))`,
  никогда `set_detect_language(true)`.
- Проверено вживую (закешированный `ggml-base.bin`, CPU Apple Silicon):
  английская речь, синтезированная через `say`, транскрибируется побайтно одинаково при
  `--lang auto` и `--lang en`; синусоидальная фикстура тоже декодируется одинаково.
- Проверка: `asr::tests::auto_detect_decodes_like_explicit` —
  запускается вживую только при наличии закешированной модели (в CI пропускается, как и другие
  тесты, зависящие от модели); доказано, что он ПАДАЕТ на коде до исправления и
  проходит после. Строка M7 в `docs/release-1.0.md` обновлена (повторный замер
  времени всё ещё ожидается).

## P10.5 — тик простоя serve/daemon вызывает `on_idle` (продолжение M11)

`LoadedModel::on_idle` существовал, но ничто не вызывало его, кроме юнит-
тестов: `maybe_idle` демона управлял только (рекомендательным, пустым)
`SysinfoBackend`, а у serve менеджера не было вовсе. Теперь пул владеет
активностью каждого движка, и оба долгоживущих процесса её обходят.

### `EngineJob::Idle` (`pool.rs`)

- Задание «запустил и забыл»: поток движка выполняет `LocalEngine::on_idle`
  (ggml снимает отображение промпт-кеша LMDB, модель остаётся; для mistral — пустая операция) и
  никогда не роняет тик. Ставится в очередь за любым выполняющимся запросом.
- `fn LocalEngine::on_idle(&mut self)` (`engine.rs`) — сквозная передача
  в бэкенд.

### Отслеживание простоя в `ModelPool` (`pool.rs`)

- Поля `last_used: HashMap<String, Instant>` (проставляется
  `touch_lru` при каждом попадании и новой загрузке, очищается при вытеснении LRU) и
  `idle_timeout: Duration` (`Duration::MAX` = отключено; serve/daemon
  задают его из политики памяти через `with_idle_timeout`).
- `fn due_for_idle(&self, now: Instant) -> Vec<String>` — чистое
  решение (движки, не используемые ≥ таймаута), тестируется без моделей.
- `fn idle_sweep(&mut self) -> Vec<String>` — отправляет `Idle` каждому
  движку, которому пора, заново проставляет ему время (один обход на таймаут, без спама на каждом тике),
  забывает мёртвые потоки движков. Может ненадолго заблокироваться за занятым движком
  — вызывать только из блокирующего потока.
- Проверка: `idle_sweep_fires_once_per_timeout`,
  `idle_sweep_disabled_by_default`, `idle_sweep_drops_dead_engines`
  (концы каналов заменяют потоки движков).

### `async fn idle_tick(pool, mm, tag)` (`pool.rs`)

- Один тик для обоих циклов: `mm.maybe_idle()` (состояние/логи менеджера)
  плюс обход пула в отдельном потоке (`spawn_blocking`), по одной
  строке `<tag>: idle <id>: prompt cache released (model kept)` на каждый
  обойдённый движок.
- Serve создаёт собственный `MemoryManager` в `listen` (политика из
  конфигурации, тик ограничен `MAX_IDLE_TICK_SECS`); демон повторно использует
  свой существующий менеджер, и его тик теперь вызывает `idle_tick` вместо
  голого `maybe_idle`. Эндпоинты генерации (`chat/completions`,
  `messages`, `embeddings`, `transcriptions`, `serve_request` демона)
  вызывают `touch` у менеджера; `/health` и `/v1/models` намеренно
  этого не делают, чтобы опросы мониторинга не могли удерживать движки активными.
- Проверка: e2e `serve_idle_tick_releases_prompt_cache`
  (`RUNA_MEMORY_IDLE_TIMEOUT_S=1`, проверяет строку лога обхода).
- Вердикт M11 (измерено, `docs/release-1.0.md`): освобождение реальное,
  но по RSS ничтожное по сравнению с резидентной моделью (772.9 MiB до
  и после на qwen2-0.5B) — критерий остаётся недостижимым без выгрузки
  модели, которую D17 запрещает.

## P10.6 — поддержка инструментов в SSE-стриминге Anthropic

Крейт: `runa-cloud` (`anthropic::parse_sse`); цикл инструментов CLI
(`drain_anthropic`) уже обрабатывает `ToolUse` — блоки терял только потоковый
парсер, поэтому вызов инструмента при `stream: true` пропадал.

### `struct SseTools` + `SseBlock` (`anthropic.rs`)

- Накапливает `content_block_start` (id/имя tool_use, виды блоков text/thinking)
  плюс фрагменты `input_json_delta` / `text_delta` /
  `thinking_delta` / `signature_delta` с ключом по индексу
  блока (`BTreeMap` сохраняет порядок из протокола).
- `fn flush(&mut self) -> Option<AnthropicEvent>` восстанавливает
  массив блоков содержимого и `ToolCall` (JSON из плохих фрагментов становится
  `{}`); блоки рассуждения сохраняют потоковую подпись, если она есть.
  Выдаётся один раз: на `message_delta` с `stop_reason` (до
  `Done`, на той же позиции, что в `parse_message`) или в конце транскрипта для
  оборванного потока. Потоки с обычным текстом ничего не выдают.
- `push_sse_event` дополнительно поглощает фрагменты инструментов/подписей, чтобы
  частичный JSON никогда не просачивался в события `Text`/`Reasoning`.
- CLI намеренно сохраняет `stream: false`: только ответ целым сообщением
  сохраняет подписи рассуждения для следующего раунда инструментов
  (комментарий в месте запроса в `cloud.rs`).
- Проверка: `sse_tool_use_reassembled_across_fragments` (два вызова,
  фрагментированные аргументы, порядок блоков, ToolUse до Done, без утечки текста),
  `sse_without_tools_emits_no_tool_use`; существующие
  `thinking_delta_and_refusal` + тесты на wiremock не изменились.

## P10.7 — офлайн `--recommend` учитывает KV + вычисления

`probe_offline` раньше проверял только размеры файлов, поэтому длинный контекст на
большой модели помещался офлайн, но не онлайн. Теперь каталог содержит
два измеренных числа, единственным источником которых раньше был заголовок.

### `CatalogEntry::{kv_mib_per_1k, compute_mib}` (`recommend.rs`)

- `kv_mib_per_1k`: MiB KV на 1024 токена ctx в f16, измеряется один раз
  для каждой записи (`runa fit --json --ctx 8192` по заголовку; KV
  строго линеен по ctx). `q8_0` уменьшает его вдвое, `q4_0` — вчетверо;
  неизвестные строки `kv_type` сохраняют значение для f16 (никогда молча
  не занижают потребность).
- `compute_mib`: MiB вычислительного буфера при ubatch 512, линейно по
  `n_ubatch` (повторяет масштабирование `estimate_compute`).
- `fn kv_bytes_for(ctx_len, kv_type) -> u64` и
  `fn compute_bytes_for(n_ubatch) -> u64` выполняют масштабирование (с округлением вверх,
  с нижней границей ноль).
- Потребность в `probe_offline` = веса + проектор + KV(ctx) +
  compute(ubatch); скорость использует ту же форму байтов на токен, что и онлайн
  (активные + KV/2). `catalog_is_sane` требует, чтобы оба числа были > 0 в
  каждой записи.
- Проверка: `offline_counts_kv_and_compute` (KV переводит Gpu→Cpu→NoFit,
  `q8_0` возвращает Gpu, вспомогательные функции масштабирования); e2e
  `fit_recommend_offline_ranks_the_catalog` по-прежнему зелёный.

## P10.9 — чат хранит историю между ходами

`Session.history: Vec<ChatMessage>` передаётся с каждым ходом REPL/TUI
(P10.9). Раньше каждый ход отправлял одно сообщение пользователя; раунды инструментов
и так оставались внутри хода, но ничто не переносило ответы дальше.

- `fn turn_messages(session, input) -> Vec<ChatMessage>` — добавляет
  новое сообщение пользователя, обрезает до ~75% от `session.ctx`, возвращает полный
  транскрипт для запроса. При обрезке выводится `(history trimmed to
  fit ctx)`.
- `fn commit_history(session, sent, answer)` — после хода с ответом
  история становится отправленными сообщениями (они прошли весь цикл
  инструментов, так что там есть каждый раунд) плюс итоговый ответ ассистента.
  Неудачные ходы ничего не записывают (повтор отправляет ту же историю).
- `fn trim_history(history, budget) -> usize` — отбрасывает самые старые *ходы*
  (до следующего сообщения `user`, чтобы сообщения `tool` никогда не отрывались
  от своего вызова `assistant`); самое новое сообщение всегда сохраняется.
  `fn estimate_history_tokens` — эвристическое ограждение «символы/4»;
  движок всё равно громко падает при превышении реального ctx.
- `/reset` (REPL + TUI) и `/model` (новые веса, устаревший транскрипт)
  очищают историю; `/mode` сохраняет её. Протокол демона уже
  передаёт полные `messages` (`ProtoMessage`, включая вызовы инструментов), поэтому
  менять протокол не понадобилось.
- Проверка: `history_tests` (порядок обрезки / сохранение самого нового / нет оторванных
  сообщений инструментов / форма коммита); e2e `chat_second_turn_remembers_history`
  (фикстура qwen2 вспоминает "Ada", 3/3 локально) рядом с существующим
  тестом повторного использования контекста.

### P11.4 — обрезка истории с точным подсчётом токенов

Локальные ходы gguf считают реальные токены: `LoadedModel::
count_history_tokens(messages)` токенизирует содержимое каждого сообщения
собственным токенизатором модели (`AddBos::Never`) плюс те же +4
токена накладных расходов шаблона на сообщение, что и в оценке; ограждение ~75% от ctx
в `turn_messages` перепроверяется по этому точному числу.
У пути через сокет демона нет токенизатора (и бэкенд mistral
его не предоставляет), поэтому там остаётся `estimate_history_tokens` (символы/4).
`trim_history` остаётся обёрткой на основе оценки над обобщённым
`trim_history_with`; `count_history_tokens_with` фиксирует точный путь
относительно счётчика-заглушки в юнит-тестах.
- Проверка: `history_tests::{exact_counter_counts_known_string,
  trim_with_exact_counter_drops_oldest_turn_first,
  fallback_estimate_locked_on_known_string}`.

## P10.10 — модели split-GGUF в каталоге

`CatalogEntry.parts: Vec<String>` содержит ссылки на шарды 2..N (`ref` остаётся
частью 1); для таких записей `size` — сумма байтов частей.

### `fn CatalogEntry::all_refs(&self) -> Vec<&str>` + `fn combine_shard_weights(first, rest)` (`recommend.rs`)

- `probe_remote` загружает заголовок каждой части, строит дескриптор
  по части 1 (шарды повторяют полные метаданные) и суммирует четыре
  группы весов по всем частям перед `check_fit`. Однофайловые записи
  проходят тот же путь с одной ссылкой — для них поведение не меняется.
- `catalog_is_sane` теперь также требует, чтобы каждая ссылка (часть 1 + parts)
  была ссылкой `hf:….gguf` без дубликатов. Ни одной разделённой записи пока не поставляется:
  каждая текущая модель помещается в один файл, поэтому `parts` — это схема +
  поддержка в пробе с покрытием тестами, ожидающая первой шардированной модели,
  которая поместится на оборудование Tier-1/2.
- Проверка: `split_tests::{all_refs_orders_part_one_first,
  split_parts_combine_weights}` (реальные метаданные заголовка qwen2 с
  переопределёнными группами — без третьей копии билдера синтетического GGUF).

## P10.11 — покрытие формата инструментов Harmony + `thinking_forced_open`

Оба хвоста P8.2 закрыты проверкой, а не новыми ветками кода:
собственный обработчик llama.cpp уже владеет обоими форматами.

- Harmony: закреплённый llama.cpp поставляет `common_chat_parse_gpt_oss`
  (каналы commentary/analysis, получатель `to=functions.<name>` при
  любом порядке заголовков). `harmony_tool_reply_parses`
  (`structured.rs`) формирует обязательный вызов инструмента через НАСТОЯЩИЙ
  шаблон gpt-oss-20B и разбирает заготовленные ответы при обоих порядках
  заголовков — 1 вызов, `get_weather`, без утечки разметки. Запускается только при наличии фикстуры
  (модель 11 GiB, только для разработки, в CI пропускается).
- `thinking_forced_open`: внутренний флаг llama.cpp (выставляется из
  шаблона + нашего `enable_thinking`, потребляется её парсером/грамматикой);
  наш путь формирования промпта уже передаёт ему всё необходимое. Подтверждено
  расширенным `think_budget_reports_reasoning_tokens` (интеграционный тест `generate.rs`):
  Qwen3-8B завершает *обязательный* вызов инструмента в рамках бюджета
  64+8, а выводимое число токенов рассуждения укладывается в budget+grace.
- Попутный ремонт фикстуры: `tests/fixtures/gpt-oss-20b-MXFP4.gguf`
  был обрезан (11.4 из 12.1 GiB — тензоры выходили за границы файла);
  докачан с Hub до точного размера в байтах из каталога. Файлы фикстур —
  веса в git-ignore, поэтому отслеживаемые файлы это не затрагивает.
- Проверка: `cargo test -p runa-engine --lib harmony` +
  `--test generate think_budget`; строка о формате в `docs/structured.md`.

## P10.13 — ограничение нагрузки на систему `--max-load-percent` (чужая незавершённая работа, завершена)

Найдено в дереве незахваченным и некомпилирующимся (поля CLI без
соответствующих сигнатур); доделано здесь: недостающая прокладка сигнатур/полей,
отказ от демона, отклонение на бэкенде mistral, документация.
Ограничивает долю общих ресурсов системы, которую может использовать запуск, в процентах
1..=100. По замыслу только предупреждает: запуск всегда продолжается, решает
пользователь (план D12 — громко, но никогда не блокируя).

### `const DEFAULT_MAX_LOAD_PERCENT: u8` (= 80)

### `fn config::resolve_max_load_percent(cli: Option<u8>) -> Result<u8, String>`

- Приоритет: CLI `--max-load-percent` > `RUNA_MAX_LOAD_PERCENT` >
  `[system] max_load_percent` (более поздние файлы побеждают) > 80.
- Ошибки: любое значение вне 1..=100 на любом уровне.

### `struct config::SystemSnapshot` + `fn config::read_system_snapshot() -> SystemSnapshot`

- Снимок на момент времени `{total_ram_bytes, avail_ram_bytes, cpu_count}` через
  `sysinfo`. `RUNA_FAKE_TOTAL_RAM_MIB` / `RUNA_FAKE_AVAIL_RAM_MIB` /
  `RUNA_FAKE_CPU_COUNT` переопределяют все три значения для тестов.

### `fn config::system_load_warnings(snap, demand_bytes, threads, limit) -> Vec<String>`

- Чистая математика, по одной строке `warning:` на каждый превышенный ресурс: потребность модели в RAM
  выше ограничения, фоновое давление на системную RAM выше ограничения,
  доля CPU для `--threads` выше ограничения. В каждой строке указано значение, которое нужно
  задать, чтобы запуск уложился.
- `fn config::warn_if_over_system_limit(demand, threads, cli)` —
  определяет ограничение и выводит строки (путь запуска: `run`,
  `chat`, `bench`, `serve`, `daemon`); возвращает лимит.
- `fn config::warn_if_demand_over_limit(demand, cli)` — только строка
  о потребности модели (`preflight_grow`, после чтения заголовка).
- Проверка: `config::tests::max_load_*` (приоритет TOML/env/CLI,
  предупреждения на фейковом снимке); пути уровня e2e проверены смоук-тестами через CLI
  (`--max-load-percent 0` и `RUNA_MAX_LOAD_PERCENT=lots` завершаются ошибкой
  до загрузки).
- Примечания: `--max-load-percent` заставляет `run` обойти демон (демон
  владеет размещением и не знает ограничений для отдельного запроса), а с
  `--backend mistral` флаг отклоняется (там предупреждений не бывает, поэтому он никогда
  не принимается молча).
