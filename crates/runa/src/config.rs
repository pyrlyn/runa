//! Minimal `runa.toml` reader (plan D10, task P2.4 slice).
//!
//! Only the `[models.<name>]` alias table is read here:
//! full figment layering (defaults < user < local < env < flags, profiles)
//! lands with the config task (P6.4 documents every key). Files merge with
//! `./runa.toml` winning over `~/.config/runa/config.toml`.

use std::collections::HashMap;
use std::path::PathBuf;

use runa_cloud::reject_inline_secrets;
use runa_core::{Effort, ThinkConfig, ThinkOverrides, parse_budget};
use runa_engine::{LoraSpec, parse_lora_spec};
use runa_media::AudioRoutePref;

/// A named model alias: `[models.qwen] source = "hf:org/model:Q4_K_M"`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelAlias {
    /// Model reference (anything [`parse_model_ref`] accepts).
    pub source: String,
}

/// Alias table: name → alias.
#[derive(Debug, Clone, Default)]
pub struct AliasTable {
    /// In merge order (later files win).
    pub models: HashMap<String, ModelAlias>,
}

impl AliasTable {
    /// Look up an alias by name.
    pub fn get(&self, name: &str) -> Option<&ModelAlias> {
        self.models.get(name)
    }
}

/// What to do when the planner returns [`runa_fit::Verdict::NoFit`] (D12).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum OnUnfit {
    /// Exit 2. Default in `runa.toml`.
    Error,
    /// Load on CPU and warn. Never silent.
    Cpu,
    /// Cloud route (`cloud:<backend>:<model>`). Applied in P3.7.
    Cloud(String),
}

impl OnUnfit {
    pub(crate) fn parse(s: &str) -> Result<Self, String> {
        let s = s.trim();
        if s.eq_ignore_ascii_case("error") {
            return Ok(Self::Error);
        }
        if s.eq_ignore_ascii_case("cpu") {
            return Ok(Self::Cpu);
        }
        if let Some(rest) = s.strip_prefix("cloud:") {
            if rest.is_empty() {
                return Err("on_unfit cloud: needs backend:model".into());
            }
            return Ok(Self::Cloud(rest.to_owned()));
        }
        Err(format!(
            "{s}: on_unfit must be error | cpu | cloud:<backend>:<model>"
        ))
    }
}

impl std::fmt::Display for OnUnfit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Error => write!(f, "error"),
            Self::Cpu => write!(f, "cpu"),
            Self::Cloud(spec) => write!(f, "cloud:{spec}"),
        }
    }
}

/// Candidate config files in increasing precedence.
pub fn config_paths() -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Ok(home) = std::env::var("HOME")
        && !home.is_empty()
    {
        out.push(
            PathBuf::from(home)
                .join(".config")
                .join("runa")
                .join("config.toml"),
        );
    }
    out.push(PathBuf::from("runa.toml"));
    out
}

/// Read a config file and reject inline API keys (P3.8). Missing files are skipped.
fn read_config_text(path: &std::path::Path) -> Result<Option<String>, String> {
    let Ok(text) = std::fs::read_to_string(path) else {
        return Ok(None);
    };
    runa_cloud::reject_inline_secrets(&text, &path.display().to_string())
        .map_err(|e| e.to_string())?;
    Ok(Some(text))
}

/// Scan every candidate config file for inline API keys (CLI startup).
pub(crate) fn reject_inline_in_config_files() -> Result<(), String> {
    for path in config_paths() {
        let _ = read_config_text(&path)?;
    }
    Ok(())
}

/// Load and merge `[models]` tables. Missing/unreadable files are skipped;
/// a present-but-invalid file is an error (fail loud, not silent).
pub fn load_aliases() -> Result<AliasTable, String> {
    let mut table = AliasTable::default();
    for path in config_paths() {
        let Some(text) = read_config_text(&path)? else {
            continue;
        };
        merge_aliases_toml(&mut table, &text, &path.display().to_string())?;
    }
    Ok(table)
}

/// CLI `--threads` > `RUNA_THREADS` > `[defaults] threads` > engine
/// default (`None` = P-cores on macOS, else logical CPUs, P10.2).
pub(crate) fn resolve_threads(cli: Option<i32>) -> Result<Option<i32>, String> {
    if let Some(n) = cli {
        return check_threads(n, "--threads");
    }
    if let Ok(s) = std::env::var("RUNA_THREADS")
        && !s.trim().is_empty()
    {
        let n: i32 = s
            .trim()
            .parse()
            .map_err(|_| format!("RUNA_THREADS={s}: expected an integer >= 1"))?;
        return check_threads(n, "RUNA_THREADS");
    }
    let mut found = None;
    for path in config_paths() {
        let Some(text) = read_config_text(&path)? else {
            continue;
        };
        if let Some(n) = defaults_threads_from_toml(&text, &path.display().to_string())? {
            found = Some(n);
        }
    }
    found.map_or(Ok(None), |n| check_threads(n, "[defaults] threads"))
}

fn check_threads(n: i32, origin: &str) -> Result<Option<i32>, String> {
    if n < 1 {
        return Err(format!("{origin}: threads must be >= 1 (got {n})"));
    }
    Ok(Some(n))
}

fn defaults_threads_from_toml(text: &str, origin: &str) -> Result<Option<i32>, String> {
    let value: toml::Value =
        toml::from_str(text).map_err(|e| format!("{origin}: invalid TOML: {e}"))?;
    let Some(table) = value.get("defaults").and_then(|v| v.as_table()) else {
        return Ok(None);
    };
    match table.get("threads") {
        None => Ok(None),
        Some(toml::Value::Integer(n)) if *n >= 1 && *n <= i64::from(i32::MAX) => {
            Ok(Some(*n as i32))
        }
        _ => Err(format!(
            "{origin}: defaults.threads must be an integer >= 1"
        )),
    }
}

/// Default cap on the share of total system resources (RAM budget and
/// CPU thread share) this app may use, in percent of the total.
pub const DEFAULT_MAX_LOAD_PERCENT: u8 = 80;

/// CLI `--max-load-percent` > `RUNA_MAX_LOAD_PERCENT` >
/// `[system] max_load_percent` > 80.
pub(crate) fn resolve_max_load_percent(cli: Option<u8>) -> Result<u8, String> {
    if let Some(n) = cli {
        return check_max_load(i32::from(n), "--max-load-percent");
    }
    if let Ok(s) = std::env::var("RUNA_MAX_LOAD_PERCENT")
        && !s.trim().is_empty()
    {
        let n: i32 = s
            .trim()
            .parse()
            .map_err(|_| format!("RUNA_MAX_LOAD_PERCENT={s}: expected an integer 1..=100"))?;
        return check_max_load(n, "RUNA_MAX_LOAD_PERCENT");
    }
    let mut found = None;
    for path in config_paths() {
        let Some(text) = read_config_text(&path)? else {
            continue;
        };
        if let Some(n) = max_load_from_toml(&text, &path.display().to_string())? {
            found = Some(n);
        }
    }
    found.map_or(Ok(DEFAULT_MAX_LOAD_PERCENT), |n| {
        check_max_load(i32::from(n), "[system] max_load_percent")
    })
}

fn check_max_load(n: i32, origin: &str) -> Result<u8, String> {
    if (1..=100).contains(&n) {
        Ok(n as u8)
    } else {
        Err(format!(
            "{origin}: max_load_percent must be 1..=100 (got {n})"
        ))
    }
}

fn max_load_from_toml(text: &str, origin: &str) -> Result<Option<u8>, String> {
    let value: toml::Value =
        toml::from_str(text).map_err(|e| format!("{origin}: invalid TOML: {e}"))?;
    let Some(table) = value.get("system").and_then(|v| v.as_table()) else {
        return Ok(None);
    };
    match table.get("max_load_percent") {
        None => Ok(None),
        Some(toml::Value::Integer(n)) if (1..=100).contains(n) => Ok(Some(*n as u8)),
        _ => Err(format!(
            "{origin}: system.max_load_percent must be an integer 1..=100"
        )),
    }
}

/// Point-in-time system totals for the startup load-cap check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SystemSnapshot {
    pub total_ram_bytes: u64,
    pub avail_ram_bytes: u64,
    pub cpu_count: u32,
}

fn fake_mib(name: &str) -> Option<u64> {
    std::env::var(name).ok()?.trim().parse::<u64>().ok()
}

/// Live totals via `sysinfo` (`RUNA_FAKE_TOTAL_RAM_MIB`,
/// `RUNA_FAKE_AVAIL_RAM_MIB`, `RUNA_FAKE_CPU_COUNT` override for tests).
pub(crate) fn read_system_snapshot() -> SystemSnapshot {
    let cpu_count = fake_mib("RUNA_FAKE_CPU_COUNT")
        .and_then(|n| u32::try_from(n).ok())
        .filter(|n| *n > 0)
        .unwrap_or_else(real_cpu_count);
    if let (Some(total_mib), Some(avail_mib)) = (
        fake_mib("RUNA_FAKE_TOTAL_RAM_MIB"),
        fake_mib("RUNA_FAKE_AVAIL_RAM_MIB"),
    ) {
        let total = total_mib.saturating_mul(1024 * 1024);
        let avail = avail_mib.saturating_mul(1024 * 1024).min(total);
        return SystemSnapshot {
            total_ram_bytes: total,
            avail_ram_bytes: avail,
            cpu_count,
        };
    }
    let mut sys = sysinfo::System::new();
    sys.refresh_memory();
    let total = sys.total_memory();
    let avail = sys.available_memory().min(total);
    SystemSnapshot {
        total_ram_bytes: total,
        avail_ram_bytes: avail,
        cpu_count,
    }
}

fn real_cpu_count() -> u32 {
    let mut sys = sysinfo::System::new();
    sys.refresh_cpu_list(sysinfo::CpuRefreshKind::nothing());
    let n = sys.cpus().len();
    if n > 0 {
        return u32::try_from(n).unwrap_or(u32::MAX);
    }
    std::thread::available_parallelism()
        .map(|n| u32::try_from(n.get()).unwrap_or(u32::MAX))
        .unwrap_or(1)
}

/// Smallest integer percent `p` with `total * p / 100 >= part` (0 when
/// `part` is 0 or `total` is 0). Pure so tests avoid touching hardware.
fn need_percent(part: u64, total: u64) -> u8 {
    if part == 0 || total == 0 {
        return 0;
    }
    ((u128::from(part) * 100).div_ceil(u128::from(total)) as u64).min(100) as u8
}

/// Pure startup-cap math: `demand_bytes` is the model's estimated RAM need
/// (when known, e.g. fit/preflight), `threads` the requested worker threads
/// (when known). Returns one `warning:` line per breached resource, each
/// naming the value to set so the run fits the cap.
pub(crate) fn system_load_warnings(
    snap: &SystemSnapshot,
    demand_bytes: Option<u64>,
    threads: Option<i32>,
    limit: u8,
) -> Vec<String> {
    let mut out = Vec::new();
    let total = snap.total_ram_bytes;
    if total == 0 {
        return out;
    }
    let mib = |b: u64| b.div_ceil(1024 * 1024);
    let budget = total * u64::from(limit) / 100;
    let origin = "[system] max_load_percent, RUNA_MAX_LOAD_PERCENT, or --max-load-percent";
    if let Some(line) = demand_warning(snap, demand_bytes, limit) {
        out.push(line);
    }
    let used = total.saturating_sub(snap.avail_ram_bytes.min(total));
    if used > budget {
        let pct = need_percent(used, total);
        out.push(format!(
            "warning: system already uses ~{} MiB RAM ({pct}% of {} MiB), above max_load_percent={limit} — set max_load_percent to at least {pct} ({origin})",
            mib(used),
            mib(total),
        ));
    }
    if let Some(t) = threads.filter(|t| *t > 0)
        && snap.cpu_count > 0
        && u64::from(t as u32) * 100 > u64::from(snap.cpu_count) * u64::from(limit)
    {
        let pct = need_percent(u64::from(t as u32), u64::from(snap.cpu_count));
        let max_t = (u64::from(snap.cpu_count) * u64::from(limit) / 100).max(1);
        out.push(format!(
            "warning: --threads {t} wants ~{pct}% of {} CPUs, above max_load_percent={limit} — set max_load_percent to at least {pct} ({origin}) or lower --threads to {max_t}",
            snap.cpu_count,
        ));
    }
    out
}

/// The model-need line of [`system_load_warnings`] on its own (`None`
/// when the demand fits the cap). Shared by the full check and the
/// demand-only [`warn_if_demand_over_limit`].
fn demand_warning(snap: &SystemSnapshot, demand_bytes: Option<u64>, limit: u8) -> Option<String> {
    let total = snap.total_ram_bytes;
    let demand = demand_bytes.filter(|d| *d > 0)?;
    if total == 0 {
        return None;
    }
    let budget = total * u64::from(limit) / 100;
    if demand <= budget {
        return None;
    }
    let need = need_percent(demand, total);
    let mib = |b: u64| b.div_ceil(1024 * 1024);
    Some(format!(
        "warning: model needs ~{} MiB RAM but max_load_percent={} allows only ~{} MiB of {} MiB total — set max_load_percent to at least {need} ([system] max_load_percent, RUNA_MAX_LOAD_PERCENT, or --max-load-percent)",
        mib(demand),
        limit,
        mib(budget),
        mib(total),
    ))
}

/// Demand-only half of [`system_load_warnings`]: just the model-need
/// line (no ambient RAM-pressure or thread-share lines). Used by
/// `preflight_grow`, which runs after the command already printed the
/// ambient check at startup.
pub(crate) fn warn_if_demand_over_limit(demand_bytes: u64, cli: Option<u8>) -> Result<u8, String> {
    let limit = resolve_max_load_percent(cli)?;
    if let Some(line) = demand_warning(&read_system_snapshot(), Some(demand_bytes), limit) {
        eprintln!("{line}");
    }
    Ok(limit)
}
/// Resolve the cap and print one `warning:` per breached resource to
/// stderr (never fatal: the run proceeds, the user decides). Returns the
/// resolved limit so callers can reuse it without re-reading config.
pub(crate) fn warn_if_over_system_limit(
    demand_bytes: Option<u64>,
    threads: Option<i32>,
    cli: Option<u8>,
) -> Result<u8, String> {
    let limit = resolve_max_load_percent(cli)?;
    for w in system_load_warnings(&read_system_snapshot(), demand_bytes, threads, limit) {
        eprintln!("{w}");
    }
    Ok(limit)
}

/// CLI `--on-unfit` > `RUNA_ON_UNFIT` > config files > `error`.
pub(crate) fn resolve_on_unfit(cli: Option<&str>) -> Result<OnUnfit, String> {
    if let Some(s) = cli {
        return OnUnfit::parse(s);
    }
    load_on_unfit()
}

fn load_on_unfit() -> Result<OnUnfit, String> {
    if let Ok(s) = std::env::var("RUNA_ON_UNFIT")
        && !s.is_empty()
    {
        return OnUnfit::parse(&s);
    }
    let mut found = OnUnfit::Error;
    for path in config_paths() {
        let Some(text) = read_config_text(&path)? else {
            continue;
        };
        if let Some(v) = on_unfit_from_toml(&text, &path.display().to_string())? {
            found = v;
        }
    }
    Ok(found)
}

fn on_unfit_from_toml(text: &str, origin: &str) -> Result<Option<OnUnfit>, String> {
    let value: toml::Value =
        toml::from_str(text).map_err(|e| format!("{origin}: invalid TOML: {e}"))?;
    match value.get("on_unfit") {
        None => Ok(None),
        Some(v) => {
            let s = v
                .as_str()
                .ok_or_else(|| format!("{origin}: on_unfit must be a string"))?;
            OnUnfit::parse(s).map(Some)
        }
    }
}

pub(crate) fn resolve_think(cli: ThinkOverrides) -> Result<ThinkConfig, String> {
    let mut cfg = ThinkConfig::default();
    for path in config_paths() {
        let Some(text) = read_config_text(&path)? else {
            continue;
        };
        if let Some(o) = think_from_toml(&text, &path.display().to_string())? {
            cfg = cfg.apply(&o)?;
        }
    }
    cfg = cfg.apply(&think_from_env()?)?;
    cfg.apply(&cli)
}

fn think_from_env() -> Result<ThinkOverrides, String> {
    let mut o = ThinkOverrides::default();
    if let Ok(s) = std::env::var("RUNA_THINK")
        && !s.is_empty()
    {
        o.think = Some(ThinkOverrides::parse_think(&s)?);
    }
    if let Ok(s) = std::env::var("RUNA_THINK_BUDGET")
        && !s.is_empty()
    {
        o.budget = Some(parse_budget(&s)?);
    }
    if let Ok(s) = std::env::var("RUNA_THINK_GRACE")
        && !s.is_empty()
    {
        o.grace = Some(
            s.parse::<u32>()
                .map_err(|_| format!("{s}: RUNA_THINK_GRACE must be an integer"))?,
        );
    }
    if let Ok(s) = std::env::var("RUNA_EFFORT")
        && !s.is_empty()
    {
        o.effort = Some(Effort::parse(&s)?);
    }
    if let Ok(s) = std::env::var("RUNA_SHOW_REASONING")
        && !s.is_empty()
    {
        o.show = Some(ThinkOverrides::parse_show(&s)?);
    }
    Ok(o)
}

fn think_from_toml(text: &str, origin: &str) -> Result<Option<ThinkOverrides>, String> {
    let value: toml::Value =
        toml::from_str(text).map_err(|e| format!("{origin}: invalid TOML: {e}"))?;
    let Some(table) = value.get("think").and_then(|v| v.as_table()) else {
        return Ok(None);
    };
    overrides_from_think_table(table, origin).map(Some)
}

/// CLI `--audio-route` > `RUNA_AUDIO_ROUTE` > `[audio] route` > `auto`.
pub(crate) fn resolve_audio_route(cli: Option<&str>) -> Result<AudioRoutePref, String> {
    if let Some(s) = cli {
        return AudioRoutePref::parse(s);
    }
    if let Ok(s) = std::env::var("RUNA_AUDIO_ROUTE") {
        let t = s.trim();
        if !t.is_empty() {
            return AudioRoutePref::parse(t);
        }
    }
    for path in config_paths() {
        let Some(text) = read_config_text(&path)? else {
            continue;
        };
        if let Some(r) = audio_route_from_toml(&text, &path.display().to_string())? {
            return Ok(r);
        }
    }
    Ok(AudioRoutePref::Auto)
}

pub(crate) fn resolve_memory_policy() -> Result<runa_memory::MemoryPolicy, String> {
    let mut p = runa_memory::MemoryPolicy::default();
    for path in config_paths() {
        let Some(text) = read_config_text(&path)? else {
            continue;
        };
        if let Some(parsed) = memory_from_toml(&text, &path.display().to_string())? {
            p = parsed;
            break;
        }
    }
    if let Ok(s) = std::env::var("RUNA_MEMORY_IDLE_TIMEOUT_S")
        && let Ok(n) = s.parse()
    {
        p.idle_timeout_s = n;
    }
    if let Ok(s) = std::env::var("RUNA_MEMORY_FLOOR_MIB")
        && let Ok(n) = s.parse()
    {
        p.floor_mib = n;
    }
    if let Ok(s) = std::env::var("RUNA_MEMORY_MAX_GROWTH_MIB")
        && let Ok(n) = s.parse()
    {
        p.max_growth_mib = n;
    }
    Ok(p)
}

fn memory_from_toml(text: &str, origin: &str) -> Result<Option<runa_memory::MemoryPolicy>, String> {
    let value: toml::Value =
        toml::from_str(text).map_err(|e| format!("{origin}: invalid TOML: {e}"))?;
    let Some(table) = value.get("memory").and_then(|v| v.as_table()) else {
        return Ok(None);
    };
    let mut p = runa_memory::MemoryPolicy::default();
    if let Some(v) = table.get("idle_timeout_s") {
        p.idle_timeout_s = u64::from(toml_u32(v, origin, "idle_timeout_s")?);
    }
    if let Some(v) = table.get("floor_mib") {
        p.floor_mib = u64::from(toml_u32(v, origin, "floor_mib")?);
    }
    if let Some(v) = table.get("max_growth_mib") {
        p.max_growth_mib = u64::from(toml_u32(v, origin, "max_growth_mib")?);
    }
    Ok(Some(p))
}

fn audio_route_from_toml(text: &str, origin: &str) -> Result<Option<AudioRoutePref>, String> {
    let value: toml::Value =
        toml::from_str(text).map_err(|e| format!("{origin}: invalid TOML: {e}"))?;
    let Some(table) = value.get("audio").and_then(|v| v.as_table()) else {
        return Ok(None);
    };
    match table.get("route") {
        None => Ok(None),
        Some(v) => {
            let s = v
                .as_str()
                .ok_or_else(|| format!("{origin}: audio.route must be a string"))?;
            AudioRoutePref::parse(s)
                .map(Some)
                .map_err(|e| format!("{origin}: {e}"))
        }
    }
}

fn toml_u32(v: &toml::Value, origin: &str, key: &str) -> Result<u32, String> {
    match v {
        toml::Value::Integer(n) if *n >= 0 && *n <= i64::from(u32::MAX) => Ok(*n as u32),
        toml::Value::String(s) => s
            .parse::<u32>()
            .map_err(|_| format!("{origin}: think.{key} must be an integer")),
        _ => Err(format!("{origin}: think.{key} must be an integer")),
    }
}

fn overrides_from_think_table(
    table: &toml::map::Map<String, toml::Value>,
    origin: &str,
) -> Result<ThinkOverrides, String> {
    let mut o = ThinkOverrides::default();
    if let Some(v) = table.get("grace") {
        o.grace = Some(toml_u32(v, origin, "grace")?);
    }
    if let Some(v) = table.get("show") {
        o.show = Some(
            v.as_bool()
                .ok_or_else(|| format!("{origin}: think.show must be a boolean"))?,
        );
    }
    let budget = match table.get("budget") {
        Some(v) => Some(parse_budget(&toml_u32(v, origin, "budget")?.to_string())?),
        None => None,
    };
    let effort = match table.get("effort") {
        Some(v) => {
            let s = v
                .as_str()
                .ok_or_else(|| format!("{origin}: think.effort must be a string"))?;
            Some(Effort::parse(s).map_err(|e| format!("{origin}: {e}"))?)
        }
        None => None,
    };
    match table.get("mode").and_then(|v| v.as_str()) {
        None => {
            o.budget = budget;
            o.effort = effort;
        }
        Some("off") => o.think = Some(false),
        Some("on") => o.think = Some(true),
        Some("budget") => {
            o.budget =
                Some(budget.ok_or_else(|| {
                    format!("{origin}: think.mode = \"budget\" needs think.budget")
                })?);
        }
        Some("effort") => {
            o.effort =
                Some(effort.ok_or_else(|| {
                    format!("{origin}: think.mode = \"effort\" needs think.effort")
                })?);
        }
        Some(other) => {
            return Err(format!(
                "{origin}: think.mode must be off | on | budget | effort, not {other}"
            ));
        }
    }
    Ok(o)
}

/// `[mcp.servers.<name>]` from every config file; a later file replaces a
/// server with the same name (P8.3).
pub(crate) fn load_mcp_servers() -> Result<Vec<crate::mcp::McpServer>, String> {
    let mut servers = std::collections::BTreeMap::new();
    for path in config_paths() {
        let Some(text) = read_config_text(&path)? else {
            continue;
        };
        merge_mcp_toml(&mut servers, &text, &path.display().to_string())?;
    }
    Ok(servers.into_values().collect())
}

fn merge_mcp_toml(
    servers: &mut std::collections::BTreeMap<String, crate::mcp::McpServer>,
    text: &str,
    origin: &str,
) -> Result<(), String> {
    let value: toml::Value =
        toml::from_str(text).map_err(|e| format!("{origin}: invalid TOML: {e}"))?;
    let Some(table) = value
        .get("mcp")
        .and_then(|m| m.get("servers"))
        .and_then(|s| s.as_table())
    else {
        return Ok(());
    };
    for (name, entry) in table {
        let bad = |what: &str| format!("{origin}: [mcp.servers.{name}] {what}");
        let command = entry
            .get("command")
            .and_then(|c| c.as_str())
            .ok_or_else(|| bad("needs a `command` string"))?;
        let strings = |v: &toml::Value| v.as_str().map(str::to_owned);
        let args = match entry.get("args") {
            None => Vec::new(),
            Some(a) => a
                .as_array()
                .and_then(|a| a.iter().map(strings).collect::<Option<Vec<_>>>())
                .ok_or_else(|| bad("`args` must be an array of strings"))?,
        };
        let env = match entry.get("env") {
            None => Default::default(),
            Some(e) => e
                .as_table()
                .and_then(|t| {
                    t.iter()
                        .map(|(k, v)| strings(v).map(|v| (k.clone(), v)))
                        .collect::<Option<_>>()
                })
                .ok_or_else(|| bad("`env` must be a table of strings"))?,
        };
        servers.insert(
            name.clone(),
            crate::mcp::McpServer {
                name: name.clone(),
                command: command.to_owned(),
                args,
                env,
            },
        );
    }
    Ok(())
}

fn merge_aliases_toml(table: &mut AliasTable, text: &str, origin: &str) -> Result<(), String> {
    reject_inline_secrets(text, origin).map_err(|e| e.to_string())?;
    let value: toml::Value =
        toml::from_str(text).map_err(|e| format!("{origin}: invalid TOML: {e}"))?;
    if let Some(models) = value.get("models").and_then(|m| m.as_table()) {
        for (name, entry) in models {
            let Some(source) = entry.get("source").and_then(|s| s.as_str()) else {
                return Err(format!("{origin}: [models.{name}] needs a `source` string"));
            };
            table.models.insert(
                name.clone(),
                ModelAlias {
                    source: source.to_owned(),
                },
            );
        }
    }
    Ok(())
}

/// LoRA adapters for one run (P8.5): global `[model] lora`, then the
/// `[models.<alias>] lora` of the alias being run, then `--lora` flags.
///
/// Later entries win on overlap (llama.cpp applies adapters in order), so
/// CLI flags come last. Every entry is `path[:scale]`.
pub(crate) fn resolve_loras(cli: &[String], model_ref: &str) -> Result<Vec<LoraSpec>, String> {
    let mut files = Vec::new();
    for path in config_paths() {
        let Some(text) = read_config_text(&path)? else {
            continue;
        };
        let origin = path.display().to_string();
        let (global, per_alias) = lora_lists_from_toml(&text, &origin)?;
        files.push((origin, global, per_alias));
    }
    let raws = select_lora_raws(&files, model_ref, cli);
    raws.into_iter()
        .map(|(origin, s)| parse_lora_spec(&s).map_err(|e| format!("{origin}: lora {s:?}: {e}")))
        .collect()
}

/// `(alias, source, specs)` from one `[models.<alias>]` table (P8.5).
type AliasLoras = (String, String, Vec<String>);
/// One file's LoRA lists: origin path, global specs, per-alias specs.
type FileLoras = (String, Vec<String>, Vec<AliasLoras>);

/// Order raw `path[:scale]` strings: global, then the matching alias,
/// then CLI flags. Pure so tests avoid touching real config files.
fn select_lora_raws(files: &[FileLoras], model_ref: &str, cli: &[String]) -> Vec<(String, String)> {
    let mut raws = Vec::new();
    for (origin, global, per_alias) in files {
        for s in global {
            raws.push((format!("{origin} [model]"), s.clone()));
        }
        for (name, source, specs) in per_alias {
            if model_ref == name || (!source.is_empty() && model_ref == source) {
                for s in specs {
                    raws.push((format!("{origin} [models.{name}]"), s.clone()));
                }
            }
        }
    }
    for s in cli {
        raws.push(("--lora".to_owned(), s.clone()));
    }
    raws
}

/// `lora` entries from one config file: global `[model] lora` plus
/// `(name, source, entries)` for every `[models.<name>]` table.
fn lora_lists_from_toml(
    text: &str,
    origin: &str,
) -> Result<(Vec<String>, Vec<AliasLoras>), String> {
    let value: toml::Value =
        toml::from_str(text).map_err(|e| format!("{origin}: invalid TOML: {e}"))?;
    let global = match value.get("model") {
        None => Vec::new(),
        Some(m) => lora_strings(m.get("lora"), origin, "[model]")?,
    };
    let mut per_alias = Vec::new();
    if let Some(models) = value.get("models").and_then(|m| m.as_table()) {
        for (name, entry) in models {
            let Some(lora) = entry.get("lora") else {
                continue;
            };
            let specs = lora_strings(Some(lora), origin, &format!("[models.{name}]"))?;
            let source = entry
                .get("source")
                .and_then(|s| s.as_str())
                .unwrap_or("")
                .to_owned();
            per_alias.push((name.clone(), source, specs));
        }
    }
    Ok((global, per_alias))
}

/// A `lora` value is one string or an array of strings.
fn lora_strings(
    value: Option<&toml::Value>,
    origin: &str,
    section: &str,
) -> Result<Vec<String>, String> {
    match value {
        None => Ok(Vec::new()),
        Some(toml::Value::String(s)) => Ok(vec![s.clone()]),
        Some(toml::Value::Array(items)) => items
            .iter()
            .map(|item| {
                item.as_str().map(str::to_owned).ok_or_else(|| {
                    format!("{origin}: {section} lora must be a string or array of strings")
                })
            })
            .collect(),
        Some(_) => Err(format!(
            "{origin}: {section} lora must be a string or array of strings"
        )),
    }
}

/// Fuzzing entry (`cargo fuzz`, fuzz/README.md): every TOML reader above
/// on one config file's text, as `runa` does at startup. No env, no disk.
#[cfg(fuzzing)]
pub(crate) fn fuzz_config_text(text: &str) {
    let origin = "fuzz.toml";
    let _ = runa_cloud::reject_inline_secrets(text, origin);
    let _ = defaults_threads_from_toml(text, origin);
    let _ = max_load_from_toml(text, origin);
    let _ = on_unfit_from_toml(text, origin);
    let _ = think_from_toml(text, origin);
    let _ = memory_from_toml(text, origin);
    let _ = audio_route_from_toml(text, origin);
    let _ = merge_mcp_toml(&mut std::collections::BTreeMap::new(), text, origin);
    let _ = merge_aliases_toml(&mut AliasTable::default(), text, origin);
    if let Ok((global, per_alias)) = lora_lists_from_toml(text, origin) {
        let model_ref = per_alias.first().map(|a| a.0.clone()).unwrap_or_default();
        let files = [(origin.to_owned(), global, per_alias)];
        for (_, raw) in select_lora_raws(&files, &model_ref, &[]) {
            let _ = parse_lora_spec(&raw);
        }
    }
    let _ = OnUnfit::parse(text);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_and_merges_models_tables() {
        let mut table = AliasTable::default();
        merge_aliases_toml(
            &mut table,
            "[models.qwen]\nsource = \"hf:unsloth/Qwen3-8B-GGUF:Q4_K_M\"\n",
            "user",
        )
        .unwrap();
        merge_aliases_toml(
            &mut table,
            "[models.qwen]\nsource = \"hf:other/model:Q8_0\"\n[models.fast]\nsource = \"./tiny.gguf\"\n",
            "local",
        )
        .unwrap();
        // Local file wins.
        assert_eq!(table.get("qwen").unwrap().source, "hf:other/model:Q8_0");
        assert_eq!(table.get("fast").unwrap().source, "./tiny.gguf");
        assert!(table.get("nope").is_none());
    }

    #[test]
    fn rejects_alias_without_source() {
        let mut table = AliasTable::default();
        assert!(merge_aliases_toml(&mut table, "[models.qwen]\nfoo = 1\n", "x").is_err());
        assert!(merge_aliases_toml(&mut table, "models = 42\n", "x").is_ok());
    }

    #[test]
    fn parses_on_unfit() {
        assert_eq!(OnUnfit::parse("error").unwrap(), OnUnfit::Error);
        assert_eq!(OnUnfit::parse("CPU").unwrap(), OnUnfit::Cpu);
        assert_eq!(
            OnUnfit::parse("cloud:anthropic:claude-sonnet-5").unwrap(),
            OnUnfit::Cloud("anthropic:claude-sonnet-5".into())
        );
        assert!(OnUnfit::parse("gpu").is_err());
        assert!(OnUnfit::parse("cloud:").is_err());
    }

    #[test]
    fn rejects_inline_api_key_in_config() {
        let mut table = AliasTable::default();
        let err = merge_aliases_toml(&mut table, "openai_api_key = \"sk-live\"\n", "runa.toml")
            .unwrap_err();
        assert!(err.contains("rejected"));
        assert!(err.contains("openai_api_key"));
    }

    #[test]
    fn on_unfit_toml_top_level() {
        assert_eq!(
            on_unfit_from_toml("on_unfit = \"cpu\"\n", "x").unwrap(),
            Some(OnUnfit::Cpu)
        );
        assert_eq!(on_unfit_from_toml("[models]\n", "x").unwrap(), None);
        assert!(on_unfit_from_toml("on_unfit = 1\n", "x").is_err());
    }

    #[test]
    fn think_toml_table() {
        let o = think_from_toml(
            "[think]\nmode = \"budget\"\nbudget = 2048\ngrace = 32\nshow = true\n",
            "x",
        )
        .unwrap()
        .unwrap();
        let cfg = ThinkConfig::default().apply(&o).unwrap();
        assert_eq!(
            cfg.mode,
            runa_core::ThinkMode::Budget {
                tokens: 2048,
                grace: 32
            }
        );
        assert!(cfg.show);
    }

    #[test]
    fn think_inline_table() {
        let o = think_from_toml("think = { mode = \"off\" }\n", "x")
            .unwrap()
            .unwrap();
        let cfg = ThinkConfig::default().apply(&o).unwrap();
        assert_eq!(cfg.mode, runa_core::ThinkMode::Off);
    }

    #[test]
    fn think_toml_effort_mode() {
        let o = think_from_toml("[think]\nmode = \"effort\"\neffort = \"high\"\n", "x")
            .unwrap()
            .unwrap();
        let cfg = ThinkConfig::default().apply(&o).unwrap();
        assert_eq!(cfg.mode, runa_core::ThinkMode::Effort(Effort::High));
    }

    #[test]
    fn think_toml_budget_mode_requires_budget() {
        assert!(think_from_toml("[think]\nmode = \"budget\"\n", "x").is_err());
    }

    #[test]
    fn think_cli_overrides_file() {
        let file = think_from_toml("[think]\nmode = \"on\"\nshow = true\n", "x")
            .unwrap()
            .unwrap();
        let cfg = ThinkConfig::default()
            .apply(&file)
            .unwrap()
            .apply(&ThinkOverrides {
                think: Some(false),
                show: Some(false),
                ..ThinkOverrides::default()
            })
            .unwrap();
        assert_eq!(cfg.mode, runa_core::ThinkMode::Off);
        assert!(!cfg.show);
    }

    #[test]
    fn memory_toml_table() {
        let t = r#"
[memory]
idle_timeout_s = 12
floor_mib = 256
max_growth_mib = 64
"#;
        let p = super::memory_from_toml(t, "t").unwrap().unwrap();
        assert_eq!(p.idle_timeout_s, 12);
        assert_eq!(p.floor_mib, 256);
        assert_eq!(p.max_growth_mib, 64);
    }

    #[test]
    fn defaults_threads_toml() {
        assert_eq!(
            super::defaults_threads_from_toml("[defaults]\nthreads = 6\n", "t").unwrap(),
            Some(6)
        );
        assert_eq!(
            super::defaults_threads_from_toml("[think]\nmode = \"on\"\n", "t").unwrap(),
            None
        );
        assert!(super::defaults_threads_from_toml("[defaults]\nthreads = 0\n", "t").is_err());
        assert!(
            super::defaults_threads_from_toml("[defaults]\nthreads = \"many\"\n", "t").is_err()
        );
    }

    #[test]
    fn max_load_toml() {
        assert_eq!(
            super::max_load_from_toml("[system]\nmax_load_percent = 60\n", "t").unwrap(),
            Some(60)
        );
        assert_eq!(
            super::max_load_from_toml("[think]\nmode = \"on\"\n", "t").unwrap(),
            None
        );
        assert!(super::max_load_from_toml("[system]\nmax_load_percent = 0\n", "t").is_err());
        assert!(super::max_load_from_toml("[system]\nmax_load_percent = 101\n", "t").is_err());
        assert!(super::max_load_from_toml("[system]\nmax_load_percent = \"lots\"\n", "t").is_err());
    }

    #[test]
    fn resolve_max_load_precedence_and_default() {
        let saved_env = std::env::var("RUNA_MAX_LOAD_PERCENT").ok();
        let saved_home = std::env::var("HOME").ok();
        // Hermetic HOME so a real user config cannot leak into the test.
        let fake_home = std::env::temp_dir().join(format!(
            "runa-max-load-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&fake_home).unwrap();
        unsafe {
            std::env::set_var("HOME", &fake_home);
            std::env::remove_var("RUNA_MAX_LOAD_PERCENT");
        }
        // No env and no files: the default is 80.
        assert_eq!(super::resolve_max_load_percent(None).unwrap(), 80);
        // CLI wins and validates.
        assert_eq!(super::resolve_max_load_percent(Some(50)).unwrap(), 50);
        assert!(super::resolve_max_load_percent(Some(0)).is_err());
        assert!(super::resolve_max_load_percent(Some(101)).is_err());
        // Env beats the file layer.
        unsafe { std::env::set_var("RUNA_MAX_LOAD_PERCENT", "70") };
        assert_eq!(super::resolve_max_load_percent(None).unwrap(), 70);
        unsafe { std::env::set_var("RUNA_MAX_LOAD_PERCENT", "0") };
        assert!(super::resolve_max_load_percent(None).is_err());
        unsafe { std::env::set_var("RUNA_MAX_LOAD_PERCENT", "lots") };
        assert!(super::resolve_max_load_percent(None).is_err());
        match saved_env {
            Some(v) => unsafe { std::env::set_var("RUNA_MAX_LOAD_PERCENT", v) },
            None => unsafe { std::env::remove_var("RUNA_MAX_LOAD_PERCENT") },
        }
        match saved_home {
            Some(v) => unsafe { std::env::set_var("HOME", v) },
            None => unsafe { std::env::remove_var("HOME") },
        }
    }

    fn load_snap(total_mib: u64, avail_mib: u64, cpus: u32) -> super::SystemSnapshot {
        super::SystemSnapshot {
            total_ram_bytes: total_mib * 1024 * 1024,
            avail_ram_bytes: avail_mib * 1024 * 1024,
            cpu_count: cpus,
        }
    }

    #[test]
    fn load_warnings_quiet_inside_cap() {
        // 16 GiB total, 4 used, 8 CPUs, 4 threads, no model demand.
        let w = super::system_load_warnings(&load_snap(16384, 12288, 8), None, Some(4), 80);
        assert!(w.is_empty(), "{w:?}");
    }

    #[test]
    fn load_warnings_demand_over_budget_names_needed_percent() {
        // Demand 14 GiB of 16 total with limit 80 (budget 12.8): need 88%.
        let w = super::system_load_warnings(
            &load_snap(16384, 16384, 8),
            Some(14 * 1024 * 1024 * 1024),
            None,
            80,
        );
        assert_eq!(w.len(), 1, "{w:?}");
        assert!(w[0].starts_with("warning:"), "{w:?}");
        assert!(w[0].contains("at least 88"), "{w:?}");
        assert!(w[0].contains("max_load_percent=80"), "{w:?}");
    }

    #[test]
    fn load_warnings_used_ram_over_cap() {
        // 15 of 16 GiB already used (94%) with limit 80.
        let w = super::system_load_warnings(&load_snap(16384, 1024, 8), None, None, 80);
        assert_eq!(w.len(), 1, "{w:?}");
        assert!(w[0].contains("already uses"), "{w:?}");
        assert!(w[0].contains("at least 94"), "{w:?}");
    }

    #[test]
    fn load_warnings_threads_over_cap() {
        // 8 threads on 8 CPUs is 100%: over a 50% cap, needs 100 or fewer threads.
        let w = super::system_load_warnings(&load_snap(16384, 16384, 8), None, Some(8), 50);
        assert_eq!(w.len(), 1, "{w:?}");
        assert!(w[0].contains("--threads 8"), "{w:?}");
        assert!(w[0].contains("at least 100"), "{w:?}");
        assert!(w[0].contains("--threads to 4"), "{w:?}");
    }

    #[test]
    fn load_warnings_zero_total_never_warns() {
        let snap = super::SystemSnapshot {
            total_ram_bytes: 0,
            avail_ram_bytes: 0,
            cpu_count: 0,
        };
        assert!(super::system_load_warnings(&snap, Some(u64::MAX), Some(999), 80).is_empty());
    }

    #[test]
    fn resolve_threads_precedence() {
        let saved = std::env::var("RUNA_THREADS").ok();
        unsafe { std::env::remove_var("RUNA_THREADS") };
        // CLI wins and validates.
        assert_eq!(super::resolve_threads(Some(4)).unwrap(), Some(4));
        assert!(super::resolve_threads(Some(0)).is_err());
        // Env beats the (absent here) file layer.
        unsafe { std::env::set_var("RUNA_THREADS", "3") };
        assert_eq!(super::resolve_threads(None).unwrap(), Some(3));
        unsafe { std::env::set_var("RUNA_THREADS", "0") };
        assert!(super::resolve_threads(None).is_err());
        unsafe { std::env::set_var("RUNA_THREADS", "lots") };
        assert!(super::resolve_threads(None).is_err());
        match saved {
            Some(v) => unsafe { std::env::set_var("RUNA_THREADS", v) },
            None => unsafe { std::env::remove_var("RUNA_THREADS") },
        }
    }

    #[test]
    fn reject_inline_api_key_in_toml() {
        let err = runa_cloud::reject_inline_secrets("openai_api_key = \"sk-live\"\n", "runa.toml")
            .unwrap_err()
            .to_string();
        assert!(err.contains("rejected"), "{err}");
        assert!(err.contains("OPENAI_API_KEY"), "{err}");
    }

    #[test]
    fn audio_route_toml() {
        assert_eq!(
            audio_route_from_toml("[audio]\nroute = \"asr\"\n", "x")
                .unwrap()
                .unwrap(),
            AudioRoutePref::Asr
        );
        assert_eq!(
            audio_route_from_toml("[think]\nmode = \"on\"\n", "x").unwrap(),
            None
        );
        assert!(audio_route_from_toml("[audio]\nroute = \"cloud\"\n", "x").is_err());
        assert_eq!(
            resolve_audio_route(Some("native")).unwrap(),
            AudioRoutePref::Native
        );
    }

    #[test]
    fn mcp_servers_toml() {
        let mut servers = std::collections::BTreeMap::new();
        let text = r#"
[mcp.servers.fs]
command = "npx"
args = ["-y", "@modelcontextprotocol/server-filesystem", "."]
env = { DEBUG = "1" }
"#;
        merge_mcp_toml(&mut servers, text, "a.toml").unwrap();
        merge_mcp_toml(
            &mut servers,
            "[mcp.servers.fs]\ncommand = \"fs-mcp\"",
            "b.toml",
        )
        .unwrap();
        let fs = &servers["fs"];
        assert_eq!(fs.command, "fs-mcp");
        assert!(fs.args.is_empty());
        merge_mcp_toml(&mut servers, text, "a.toml").unwrap();
        assert_eq!(servers["fs"].args.len(), 3);
        assert_eq!(servers["fs"].env["DEBUG"], "1");
        let err = merge_mcp_toml(&mut servers, "[mcp.servers.x]\nargs = []", "c.toml").unwrap_err();
        assert!(err.contains("command"), "{err}");
        let err = merge_mcp_toml(
            &mut servers,
            "[mcp.servers.x]\ncommand = \"x\"\nargs = [1]",
            "c.toml",
        )
        .unwrap_err();
        assert!(err.contains("args"), "{err}");
    }

    #[test]
    fn lora_toml_global_and_per_alias() {
        let text = r#"
[model]
lora = ["base-lora.gguf:0.5"]

[models.qwen]
source = "hf:org/model:Q4_K_M"
lora = "qwen-lora.gguf"

[models.plain]
source = "./tiny.gguf"
"#;
        let (global, per_alias) = lora_lists_from_toml(text, "x").unwrap();
        assert_eq!(global, ["base-lora.gguf:0.5"]);
        assert_eq!(per_alias.len(), 1);
        assert_eq!(per_alias[0].0, "qwen");
        assert_eq!(per_alias[0].1, "hf:org/model:Q4_K_M");
        assert_eq!(per_alias[0].2, ["qwen-lora.gguf"]);
    }

    #[test]
    fn lora_toml_rejects_non_strings() {
        assert!(lora_lists_from_toml("[model]\nlora = 42\n", "x").is_err());
        assert!(lora_lists_from_toml("[model]\nlora = [1]\n", "x").is_err());
        assert!(lora_lists_from_toml("[model]\n", "x").unwrap().0.is_empty());
    }

    #[test]
    fn lora_raws_order_global_alias_cli() {
        let files = vec![(
            "a.toml".to_owned(),
            vec!["g.gguf".to_owned()],
            vec![(
                "qwen".to_owned(),
                "hf:org/model:Q4_K_M".to_owned(),
                vec!["q.gguf:0.5".to_owned()],
            )],
        )];
        let cli = vec!["c.gguf".to_owned()];
        // Alias name matches.
        let raws = select_lora_raws(&files, "qwen", &cli);
        let specs: Vec<&str> = raws.iter().map(|(_, s)| s.as_str()).collect();
        assert_eq!(specs, ["g.gguf", "q.gguf:0.5", "c.gguf"]);
        // Alias source matches too.
        let raws = select_lora_raws(&files, "hf:org/model:Q4_K_M", &cli);
        let specs: Vec<&str> = raws.iter().map(|(_, s)| s.as_str()).collect();
        assert_eq!(specs, ["g.gguf", "q.gguf:0.5", "c.gguf"]);
        // Unrelated refs only get the global entry plus CLI.
        let raws = select_lora_raws(&files, "other", &cli);
        let specs: Vec<&str> = raws.iter().map(|(_, s)| s.as_str()).collect();
        assert_eq!(specs, ["g.gguf", "c.gguf"]);
    }

    /// P6.4: every parsed config key must appear in docs/config.md.
    const CONFIG_KEYS: &[&str] = &[
        "[mcp.servers",
        "command",
        "args",
        "env",
        "on_unfit",
        "source",
        "[model]",
        "lora",
        "[defaults]",
        "threads",
        "[system]",
        "max_load_percent",
        "[think]",
        "mode",
        "budget",
        "grace",
        "effort",
        "show",
        "[audio]",
        "route",
        "[memory]",
        "idle_timeout_s",
        "floor_mib",
        "max_growth_mib",
        "[models",
        "RUNA_ON_UNFIT",
        "RUNA_THREADS",
        "RUNA_MAX_LOAD_PERCENT",
        "RUNA_THINK",
        "RUNA_THINK_BUDGET",
        "RUNA_THINK_GRACE",
        "RUNA_EFFORT",
        "RUNA_SHOW_REASONING",
        "RUNA_AUDIO_ROUTE",
        "RUNA_MEMORY_IDLE_TIMEOUT_S",
        "RUNA_MEMORY_FLOOR_MIB",
        "RUNA_MEMORY_MAX_GROWTH_MIB",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "OPENAI_BASE_URL",
    ];

    #[test]
    fn config_keys_in_config_md() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/config.md");
        let doc = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        for key in CONFIG_KEYS {
            assert!(doc.contains(key), "docs/config.md missing `{key}`");
        }
    }
}
