// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! Append-only usage journal for `runa stats` and `runa web`.
//!
//! One JSON object per line. `run`, `chat`, and `serve` append a line when a
//! generation finishes. Nothing here is sent off the machine.

use std::fs::{self, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::Once;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use chrono::{DateTime, TimeZone, Utc};
use runa_engine::Usage;
use serde::{Deserialize, Serialize};

const RECENT_CAP: usize = 20;
const MODEL_CAP: usize = 120;

/// Where finished generations are appended.
/// `RUNA_USAGE_LOG` wins; otherwise `$XDG_CACHE_HOME/runa/usage.jsonl`
/// (or `~/.cache/runa/usage.jsonl`).
pub(crate) fn journal_path() -> PathBuf {
    if let Ok(p) = std::env::var("RUNA_USAGE_LOG") {
        let p = p.trim();
        if !p.is_empty() {
            return PathBuf::from(p);
        }
    }
    crate::daemon_proto::cache_dir().join("usage.jsonl")
}

/// One finished generation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct UsageRecord {
    pub ts: u64,
    pub source: String,
    pub model: String,
    pub prompt_tokens: u64,
    pub generated_tokens: u64,
    pub reasoning_tokens: u64,
    pub pp_toks_per_s: f64,
    pub tg_toks_per_s: f64,
}

impl UsageRecord {
    pub(crate) fn from_usage(source: &str, model: &str, usage: &Usage) -> Self {
        Self::from_counts(
            source,
            model,
            u64::from(usage.prompt_tokens),
            u64::from(usage.generated_tokens),
            u64::from(usage.reasoning_tokens),
            usage.pp_toks_per_s,
            usage.tg_toks_per_s,
        )
    }

    pub(crate) fn from_counts(
        source: &str,
        model: &str,
        prompt: u64,
        generated: u64,
        reasoning: u64,
        pp: f64,
        tg: f64,
    ) -> Self {
        Self {
            ts: unix_now(),
            source: source.to_owned(),
            model: clip_model(model),
            prompt_tokens: prompt,
            generated_tokens: generated,
            reasoning_tokens: reasoning,
            pp_toks_per_s: finite(pp),
            tg_toks_per_s: finite(tg),
        }
    }
}

/// Best-effort append. A full disk must not fail the generation.
/// Unit tests skip the write unless `RUNA_USAGE_LOG` is set, so dashboard
/// tests do not touch the real cache.
pub(crate) fn record(rec: &UsageRecord) {
    if cfg!(test) && std::env::var_os("RUNA_USAGE_LOG").is_none() {
        return;
    }
    if let Err(e) = append(&journal_path(), rec) {
        static ONCE: Once = Once::new();
        ONCE.call_once(|| eprintln!("runa: usage journal: {e}"));
    }
}

pub(crate) fn record_usage(source: &str, model: &str, usage: &Usage) {
    record(&UsageRecord::from_usage(source, model, usage));
}

pub(crate) fn append(path: &Path, rec: &UsageRecord) -> Result<(), String> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
    }
    let mut line = serde_json::to_string(rec).map_err(|e| e.to_string())?;
    line.push('\n');
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    file.write_all(line.as_bytes())
        .map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(())
}

/// `24h` or `7d` (also `60d`). Zero and other suffixes are an error.
pub(crate) fn parse_since(raw: &str) -> Result<Duration, String> {
    let s = raw.trim();
    let (num, mult) = if let Some(n) = s.strip_suffix(['d', 'D']) {
        (n, 86_400u64)
    } else if let Some(n) = s.strip_suffix(['h', 'H']) {
        (n, 3_600u64)
    } else {
        return Err(format!("since: {raw} (use 24h or 7d)"));
    };
    if num.is_empty() || !num.bytes().all(|b| b.is_ascii_digit()) {
        return Err(format!("since: {raw} (use 24h or 7d)"));
    }
    let n: u64 = num
        .parse()
        .map_err(|_| format!("since: {raw} (use 24h or 7d)"))?;
    if n == 0 {
        return Err(format!("since: {raw} (use 24h or 7d)"));
    }
    Ok(Duration::from_secs(n.saturating_mul(mult)))
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) struct Summary {
    pub requests: u64,
    pub prompt_tokens: u64,
    pub generated_tokens: u64,
    pub reasoning_tokens: u64,
    pub avg_tg_toks_per_s: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) struct ModelRow {
    pub model: String,
    pub requests: u64,
    pub prompt_tokens: u64,
    pub generated_tokens: u64,
    pub reasoning_tokens: u64,
    pub avg_tg_toks_per_s: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) struct SourceRow {
    pub source: String,
    pub requests: u64,
    pub prompt_tokens: u64,
    pub generated_tokens: u64,
    pub reasoning_tokens: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) struct DayRow {
    pub day: String,
    pub requests: u64,
    pub prompt_tokens: u64,
    pub generated_tokens: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) struct HourRow {
    /// Unix seconds of the hour, UTC.
    pub hour: u64,
    pub requests: u64,
    pub generated_tokens: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) struct Report {
    pub journal: String,
    pub now: u64,
    pub summary: Summary,
    pub by_model: Vec<ModelRow>,
    pub by_source: Vec<SourceRow>,
    pub by_day: Vec<DayRow>,
    pub by_hour: Vec<HourRow>,
    pub recent: Vec<UsageRecord>,
}

impl Report {
    pub(crate) fn to_json(&self) -> Result<String, String> {
        serde_json::to_string_pretty(self).map_err(|e| e.to_string())
    }

    pub(crate) fn to_table(&self) -> String {
        let mut out = String::new();
        let s = &self.summary;
        push_kv(&mut out, "requests", &grouped(s.requests));
        push_kv(&mut out, "prompt tokens", &grouped(s.prompt_tokens));
        push_kv(&mut out, "generated tokens", &grouped(s.generated_tokens));
        push_kv(&mut out, "reasoning tokens", &grouped(s.reasoning_tokens));
        push_kv(
            &mut out,
            "avg decode",
            &format!("{:.1} tok/s", s.avg_tg_toks_per_s),
        );
        if !self.by_source.is_empty() {
            let parts: Vec<String> = self
                .by_source
                .iter()
                .map(|row| format!("{} {}", row.source, grouped(row.requests)))
                .collect();
            push_kv(&mut out, "sources", &parts.join(" · "));
        }
        out.push('\n');
        if s.requests == 0 {
            out.push_str("journal empty\n");
        } else {
            model_table(&mut out, &self.by_model);
            day_table(&mut out, &self.by_day);
            recent_table(&mut out, &self.recent);
        }
        out.push_str(&format!("journal: {}\n", self.journal));
        out
    }
}

pub(crate) fn load_default(since: Option<Duration>) -> Result<Report, String> {
    let path = journal_path();
    load(&path, since, unix_now())
}

pub(crate) fn load(path: &Path, since: Option<Duration>, now: u64) -> Result<Report, String> {
    let cutoff = since.map(|d| now.saturating_sub(d.as_secs()));
    let hour_floor = now.saturating_sub(24 * 3_600);
    let mut rows = Vec::new();
    if path.is_file() {
        let file = fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        for line in BufReader::new(file).lines() {
            let line = line.map_err(|e| format!("{}: {e}", path.display()))?;
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let Ok(rec) = serde_json::from_str::<UsageRecord>(line) else {
                continue;
            };
            if cutoff.is_some_and(|cut| rec.ts < cut) {
                continue;
            }
            rows.push(rec);
        }
    }
    Ok(fold(path, now, hour_floor, rows))
}

struct Acc {
    requests: u64,
    prompt: u64,
    generated: u64,
    reasoning: u64,
    tps_weight: f64,
    tps_tokens: u64,
}

impl Acc {
    fn new() -> Self {
        Self {
            requests: 0,
            prompt: 0,
            generated: 0,
            reasoning: 0,
            tps_weight: 0.0,
            tps_tokens: 0,
        }
    }

    fn add(&mut self, rec: &UsageRecord) {
        self.requests = self.requests.saturating_add(1);
        self.prompt = self.prompt.saturating_add(rec.prompt_tokens);
        self.generated = self.generated.saturating_add(rec.generated_tokens);
        self.reasoning = self.reasoning.saturating_add(rec.reasoning_tokens);
        if rec.tg_toks_per_s > 0.0 && rec.generated_tokens > 0 {
            self.tps_weight += rec.tg_toks_per_s * rec.generated_tokens as f64;
            self.tps_tokens = self.tps_tokens.saturating_add(rec.generated_tokens);
        }
    }

    fn avg(&self) -> f64 {
        if self.tps_tokens == 0 {
            0.0
        } else {
            self.tps_weight / self.tps_tokens as f64
        }
    }
}

fn fold(path: &Path, now: u64, hour_floor: u64, rows: Vec<UsageRecord>) -> Report {
    let mut summary = Acc::new();
    let mut models: std::collections::BTreeMap<String, Acc> = std::collections::BTreeMap::new();
    let mut sources: std::collections::BTreeMap<String, Acc> = std::collections::BTreeMap::new();
    let mut days: std::collections::BTreeMap<String, (u64, u64, u64)> =
        std::collections::BTreeMap::new();
    let mut hours: std::collections::BTreeMap<u64, (u64, u64)> = std::collections::BTreeMap::new();
    let mut recent = std::collections::VecDeque::new();

    for rec in rows {
        summary.add(&rec);
        models
            .entry(rec.model.clone())
            .or_insert_with(Acc::new)
            .add(&rec);
        sources
            .entry(rec.source.clone())
            .or_insert_with(Acc::new)
            .add(&rec);
        if let Some(dt) = stamp(rec.ts) {
            let day = dt.format("%Y-%m-%d").to_string();
            let slot = days.entry(day).or_insert((0, 0, 0));
            slot.0 = slot.0.saturating_add(1);
            slot.1 = slot.1.saturating_add(rec.prompt_tokens);
            slot.2 = slot.2.saturating_add(rec.generated_tokens);
        }
        if rec.ts >= hour_floor {
            let hour = rec.ts - (rec.ts % 3_600);
            let slot = hours.entry(hour).or_insert((0, 0));
            slot.0 = slot.0.saturating_add(1);
            slot.1 = slot.1.saturating_add(rec.generated_tokens);
        }
        recent.push_back(rec);
        if recent.len() > RECENT_CAP {
            recent.pop_front();
        }
    }

    let mut by_model: Vec<ModelRow> = models
        .into_iter()
        .map(|(model, acc)| ModelRow {
            model,
            requests: acc.requests,
            prompt_tokens: acc.prompt,
            generated_tokens: acc.generated,
            reasoning_tokens: acc.reasoning,
            avg_tg_toks_per_s: acc.avg(),
        })
        .collect();
    by_model.sort_by(|a, b| {
        b.generated_tokens
            .cmp(&a.generated_tokens)
            .then_with(|| a.model.cmp(&b.model))
    });

    let mut by_source: Vec<SourceRow> = sources
        .into_iter()
        .map(|(source, acc)| SourceRow {
            source,
            requests: acc.requests,
            prompt_tokens: acc.prompt,
            generated_tokens: acc.generated,
            reasoning_tokens: acc.reasoning,
        })
        .collect();
    by_source.sort_by(|a, b| {
        b.generated_tokens
            .cmp(&a.generated_tokens)
            .then_with(|| a.source.cmp(&b.source))
    });

    let by_day = days
        .into_iter()
        .map(|(day, (requests, prompt, generated))| DayRow {
            day,
            requests,
            prompt_tokens: prompt,
            generated_tokens: generated,
        })
        .collect();
    let by_hour = hours
        .into_iter()
        .map(|(hour, (requests, generated))| HourRow {
            hour,
            requests,
            generated_tokens: generated,
        })
        .collect();
    let recent: Vec<UsageRecord> = recent.into_iter().rev().collect();

    Report {
        journal: path.display().to_string(),
        now,
        summary: Summary {
            requests: summary.requests,
            prompt_tokens: summary.prompt,
            generated_tokens: summary.generated,
            reasoning_tokens: summary.reasoning,
            avg_tg_toks_per_s: summary.avg(),
        },
        by_model,
        by_source,
        by_day,
        by_hour,
        recent,
    }
}

fn stamp(ts: u64) -> Option<DateTime<Utc>> {
    Utc.timestamp_opt(ts as i64, 0).single()
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn finite(n: f64) -> f64 {
    if n.is_finite() && n > 0.0 { n } else { 0.0 }
}

fn clip_model(model: &str) -> String {
    let flat: String = model
        .chars()
        .map(|c| if c.is_control() { ' ' } else { c })
        .take(MODEL_CAP)
        .collect();
    flat.trim().to_owned()
}

fn grouped(n: u64) -> String {
    let s = n.to_string();
    let bytes = s.as_bytes();
    let mut out = String::with_capacity(s.len() + s.len() / 3);
    for (i, b) in bytes.iter().enumerate() {
        if i > 0 && (bytes.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(*b as char);
    }
    out
}

fn push_kv(out: &mut String, key: &str, value: &str) {
    out.push_str(&format!("{key:<20}{value}\n"));
}

fn clip_cell(s: &str, n: usize) -> String {
    let count = s.chars().count();
    if count <= n {
        return s.to_owned();
    }
    let mut out: String = s.chars().take(n.saturating_sub(1)).collect();
    out.push('…');
    out
}

fn model_table(out: &mut String, rows: &[ModelRow]) {
    out.push_str(&format!(
        "{:<28} {:>8} {:>10} {:>10} {:>10} {:>8}\n",
        "model", "requests", "prompt", "generated", "reasoning", "tok/s"
    ));
    for row in rows {
        out.push_str(&format!(
            "{:<28} {:>8} {:>10} {:>10} {:>10} {:>8.1}\n",
            clip_cell(&row.model, 28),
            grouped(row.requests),
            grouped(row.prompt_tokens),
            grouped(row.generated_tokens),
            grouped(row.reasoning_tokens),
            row.avg_tg_toks_per_s
        ));
    }
    out.push('\n');
}

fn day_table(out: &mut String, rows: &[DayRow]) {
    if rows.is_empty() {
        return;
    }
    out.push_str(&format!(
        "{:<12} {:>8} {:>10}\n",
        "day", "requests", "generated"
    ));
    for row in rows {
        out.push_str(&format!(
            "{:<12} {:>8} {:>10}\n",
            row.day,
            grouped(row.requests),
            grouped(row.generated_tokens)
        ));
    }
    out.push('\n');
}

fn recent_table(out: &mut String, rows: &[UsageRecord]) {
    out.push_str("recent\n");
    for row in rows {
        let when = stamp(row.ts)
            .map(|dt| dt.format("%Y-%m-%d %H:%M:%S").to_string())
            .unwrap_or_else(|| row.ts.to_string());
        out.push_str(&format!(
            "{when}  {:<5}  {:<28}  prompt {}  generated {}  {:.1} tok/s\n",
            row.source,
            clip_cell(&row.model, 28),
            grouped(row.prompt_tokens),
            grouped(row.generated_tokens),
            row.tg_toks_per_s
        ));
    }
    out.push('\n');
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn rec(
        ts: u64,
        source: &str,
        model: &str,
        prompt: u64,
        generated: u64,
        tg: f64,
    ) -> UsageRecord {
        UsageRecord {
            ts,
            source: source.into(),
            model: model.into(),
            prompt_tokens: prompt,
            generated_tokens: generated,
            reasoning_tokens: 0,
            pp_toks_per_s: 10.0,
            tg_toks_per_s: tg,
        }
    }

    #[test]
    fn since_parses_hours_and_days() {
        assert_eq!(parse_since("24h").unwrap(), Duration::from_secs(24 * 3600));
        assert_eq!(parse_since("7d").unwrap(), Duration::from_secs(7 * 86_400));
        assert_eq!(
            parse_since(" 60D ").unwrap(),
            Duration::from_secs(60 * 86_400)
        );
        assert!(parse_since("0d").is_err());
        assert!(parse_since("week").is_err());
        assert!(parse_since("").is_err());
    }

    #[test]
    fn fold_skips_junk_and_weights_speed_by_generated_tokens() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("usage.jsonl");
        append(&path, &rec(1_700_000_000, "run", "qwen", 10, 4, 10.0)).unwrap();
        append(&path, &rec(1_700_000_100, "serve", "qwen", 5, 6, 20.0)).unwrap();
        append(&path, &rec(1_700_000_200, "chat", "cloud", 1, 0, 0.0)).unwrap();
        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap();
        writeln!(file, "not json").unwrap();
        writeln!(file).unwrap();

        let report = load(&path, None, 1_700_000_500).unwrap();
        assert_eq!(report.summary.requests, 3);
        assert_eq!(report.summary.prompt_tokens, 16);
        assert_eq!(report.summary.generated_tokens, 10);
        // (10*4 + 20*6) / 10 = 16
        assert!((report.summary.avg_tg_toks_per_s - 16.0).abs() < 1e-9);
        assert_eq!(report.by_model[0].model, "qwen");
        assert_eq!(report.by_model[0].generated_tokens, 10);
        assert_eq!(report.by_day.len(), 1);
        assert_eq!(report.by_day[0].day, "2023-11-14");
        assert_eq!(report.recent[0].source, "chat");
        assert_eq!(report.recent.len(), 3);
        let table = report.to_table();
        assert!(table.contains("requests"), "{table}");
        assert!(table.contains("qwen"), "{table}");
        assert!(table.contains("journal:"), "{table}");
        let json = report.to_json().unwrap();
        assert!(json.contains("\"requests\": 3"));
    }

    #[test]
    fn since_drops_older_rows() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("usage.jsonl");
        append(&path, &rec(1_699_990_000, "run", "old", 1, 1, 1.0)).unwrap();
        append(&path, &rec(1_700_003_000, "run", "new", 2, 2, 2.0)).unwrap();
        let now = 1_700_003_100;
        let report = load(&path, Some(Duration::from_secs(3600)), now).unwrap();
        assert_eq!(report.summary.requests, 1);
        assert_eq!(report.by_model[0].model, "new");
    }

    #[test]
    fn missing_file_is_an_empty_journal() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("missing.jsonl");
        let report = load(&path, None, 1_700_000_000).unwrap();
        assert_eq!(report.summary.requests, 0);
        assert!(report.to_table().contains("journal empty"));
    }

    #[test]
    fn model_names_lose_control_characters() {
        let rec = UsageRecord::from_counts("run", "a\nb\t", 1, 1, 0, f64::NAN, -1.0);
        assert_eq!(rec.model, "a b");
        assert_eq!(rec.pp_toks_per_s, 0.0);
        assert_eq!(rec.tg_toks_per_s, 0.0);
    }
}
