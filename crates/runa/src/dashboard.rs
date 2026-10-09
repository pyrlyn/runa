// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! Live snapshot for `GET /dashboard` (P14.4).
//!
//! Rings are bounded (60s window, [`LATENCY_RING`] latencies, [`ERROR_RING`]
//! errors). Nothing here is sent off-box. Quotas are not config keys: the
//! parallel semaphore plus the defaults below decide `hitting_ceiling`.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use axum::extract::ws::{Message, WebSocket};
use serde_json::{Value, json};
use tokio::sync::{OwnedSemaphorePermit, Semaphore, watch};

/// Token and request counters use this window.
pub(crate) const WINDOW_SECS: u64 = 60;
/// Documented default. The 60s window hits the ceiling at this many tokens.
pub(crate) const DEFAULT_TOKENS_PER_MINUTE: u64 = 100_000;
/// Documented default. The 60s window hits the ceiling when
/// `requests / WINDOW_SECS` reaches this rate.
pub(crate) const DEFAULT_REQUESTS_PER_SECOND: f64 = 16.0;
const ERROR_RING: usize = 16;
const LATENCY_RING: usize = 512;
const CLIP: usize = 2_000;

const INDEX_HTML: &str = include_str!("../../../dashboard/dist/index.html");
const DASHBOARD_JS: &str = include_str!("../../../dashboard/dist/dashboard.js");
const DASHBOARD_CSS: &str = include_str!("../../../dashboard/dist/dashboard.css");

pub(crate) fn index_html() -> &'static str {
    INDEX_HTML
}

pub(crate) fn dashboard_js() -> &'static str {
    DASHBOARD_JS
}

pub(crate) fn dashboard_css() -> &'static str {
    DASHBOARD_CSS
}

/// Nearest-rank percentile. `sorted` must be ascending.
/// Index is `ceil(p * n) - 1`, clamped to the last sample. Empty input is 0.
pub(crate) fn percentile(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    let rank = ((p * n as f64).ceil() as usize).clamp(1, n);
    sorted[rank - 1]
}

/// True when every parallel slot is taken, or the 60s window is at or over
/// [`DEFAULT_TOKENS_PER_MINUTE`] or [`DEFAULT_REQUESTS_PER_SECOND`].
pub(crate) fn hitting_ceiling(
    permits_available: usize,
    tokens_in_window: u64,
    requests_in_window: u64,
) -> bool {
    permits_available == 0
        || tokens_in_window >= DEFAULT_TOKENS_PER_MINUTE
        || (requests_in_window as f64) / (WINDOW_SECS as f64) >= DEFAULT_REQUESTS_PER_SECOND
}

/// First lines of a forced backtrace, for the error ring.
pub(crate) fn short_trace() -> String {
    let text = std::backtrace::Backtrace::force_capture().to_string();
    let mut out = String::new();
    for (i, line) in text.lines().take(8).enumerate() {
        if i > 0 {
            out.push('\n');
        }
        out.push_str(line);
    }
    if out.is_empty() {
        "no traceback".to_owned()
    } else {
        out
    }
}

struct LiveReq {
    id: u64,
    route: &'static str,
    model: String,
    started: Instant,
    budget: Option<u32>,
}

struct Waiter {
    id: u64,
    route: &'static str,
    started: Instant,
}

struct TokenMark {
    at: Instant,
    prompt: u64,
    generated: u64,
}

struct StoredError {
    message: String,
    traceback: String,
    at_unix_ms: u64,
}

struct Inner {
    in_flight: Vec<LiveReq>,
    queue: Vec<Waiter>,
    latencies_ms: VecDeque<f64>,
    errors: VecDeque<StoredError>,
    error_count: u64,
    token_marks: VecDeque<TokenMark>,
    request_marks: VecDeque<Instant>,
    reasoning_spent: u64,
    decode_tps: f64,
    next_id: u64,
}

impl Default for Inner {
    fn default() -> Self {
        Self {
            in_flight: Vec::new(),
            queue: Vec::new(),
            latencies_ms: VecDeque::new(),
            errors: VecDeque::new(),
            error_count: 0,
            token_marks: VecDeque::new(),
            request_marks: VecDeque::new(),
            reasoning_spent: 0,
            decode_tps: 0.0,
            next_id: 1,
        }
    }
}

pub(crate) struct DashboardMetrics {
    parallel: Arc<Semaphore>,
    limit: usize,
    inner: Mutex<Inner>,
    load: Mutex<sysinfo::System>,
    notify: tokio::sync::Notify,
}

/// Permit plus the flight record dropped when the response finishes.
pub(crate) struct Admission {
    pub permit: OwnedSemaphorePermit,
    pub flight: Flight,
}

pub(crate) struct Flight {
    metrics: Arc<DashboardMetrics>,
    id: u64,
    started: Instant,
    prompt: u64,
    generated: u64,
    reasoning: u64,
    decode_tps: f64,
    error: Option<(String, String)>,
}

struct QueueTicket {
    metrics: Arc<DashboardMetrics>,
    id: u64,
}

impl Drop for QueueTicket {
    fn drop(&mut self) {
        self.metrics.dequeue(self.id);
    }
}

struct Finished {
    id: u64,
    started: Instant,
    prompt: u64,
    generated: u64,
    reasoning: u64,
    decode_tps: f64,
    error: Option<(String, String)>,
}

impl Drop for Flight {
    fn drop(&mut self) {
        let err = if std::thread::panicking() {
            None
        } else {
            self.error.take()
        };
        self.metrics.complete(Finished {
            id: self.id,
            started: self.started,
            prompt: self.prompt,
            generated: self.generated,
            reasoning: self.reasoning,
            decode_tps: self.decode_tps,
            error: err,
        });
    }
}

impl Flight {
    pub(crate) fn set_model(&mut self, model: &str) {
        self.metrics.set_model(self.id, model);
    }

    pub(crate) fn note_budget(&mut self, tokens: u32) {
        self.metrics.set_budget(self.id, tokens);
    }

    pub(crate) fn add_tokens(
        &mut self,
        prompt: u32,
        generated: u32,
        reasoning: u32,
        decode_tps: f64,
    ) {
        self.prompt = self.prompt.saturating_add(u64::from(prompt));
        self.generated = self.generated.saturating_add(u64::from(generated));
        self.reasoning = self.reasoning.saturating_add(u64::from(reasoning));
        if decode_tps.is_finite() && decode_tps > 0.0 {
            self.decode_tps = decode_tps;
        }
    }

    pub(crate) fn fail(&mut self, message: &str, traceback: &str) {
        if self.error.is_none() {
            self.error = Some((clip(message), clip(traceback)));
        }
    }
}

impl DashboardMetrics {
    pub(crate) fn new(parallel: Arc<Semaphore>, limit: usize) -> Self {
        Self {
            parallel,
            limit,
            inner: Mutex::new(Inner::default()),
            load: Mutex::new(sysinfo::System::new()),
            notify: tokio::sync::Notify::new(),
        }
    }

    /// Take a parallel slot. Waiters show up in `queue` until a permit is free.
    pub(crate) async fn admit(self: &Arc<Self>, route: &'static str) -> Result<Admission, ()> {
        let sem = Arc::clone(&self.parallel);
        if let Ok(permit) = sem.clone().try_acquire_owned() {
            return Ok(Admission {
                permit,
                flight: self.start(route),
            });
        }
        let ticket = self.enqueue(route);
        let permit = sem.acquire_owned().await.map_err(|_| ())?;
        drop(ticket);
        Ok(Admission {
            permit,
            flight: self.start(route),
        })
    }

    pub(crate) fn record_error(&self, message: &str, traceback: &str) {
        {
            let mut g = self.lock();
            push_error(&mut g, clip(message), clip(traceback));
        }
        self.notify.notify_one();
    }

    pub(crate) async fn changed(&self) {
        self.notify.notified().await;
    }

    /// JSON body for `GET /dashboard/snapshot` and the websocket.
    /// `loaded` is `(id, approximate weight bytes)`.
    pub(crate) fn to_json(&self, loaded: &[(String, u64)], configured: usize) -> Value {
        let (cpu, rss_mib) = self.sample_load();
        let now = Instant::now();
        let mut g = self.lock();
        prune_tokens(&mut g.token_marks, now);
        prune_instants(&mut g.request_marks, now);
        let (prompt, generated) = token_sums(&g.token_marks);
        let tokens = prompt.saturating_add(generated);
        let requests = g.request_marks.len() as u64;
        let available = self.parallel.available_permits();
        let (p50, p95, p99, nlat) = percentiles(&g.latencies_ms);
        let history = history_buckets(&g.token_marks, now);
        let in_flight: Vec<Value> = g
            .in_flight
            .iter()
            .map(|r| {
                json!({
                    "id": r.id,
                    "route": r.route,
                    "model": r.model,
                    "elapsed_ms": elapsed_ms(r.started),
                })
            })
            .collect();
        let waiting: Vec<Value> = g
            .queue
            .iter()
            .map(|w| {
                json!({
                    "id": w.id,
                    "route": w.route,
                    "wait_ms": elapsed_ms(w.started),
                })
            })
            .collect();
        let recent: Vec<Value> = g
            .errors
            .iter()
            .rev()
            .map(|e| {
                json!({
                    "message": e.message,
                    "traceback": e.traceback,
                    "at_unix_ms": e.at_unix_ms,
                })
            })
            .collect();
        let active_budget = active_budget(&g.in_flight);
        let models: Vec<Value> = loaded
            .iter()
            .map(|(id, bytes)| json!({"id": id, "memory_bytes": bytes}))
            .collect();
        json!({
            "token_speed": {
                "toks_per_s": finite(generated as f64 / WINDOW_SECS as f64),
                "decode_toks_per_s": finite(g.decode_tps),
                "history": history,
            },
            "in_flight": in_flight,
            "system_load": {
                "process_cpu_percent": finite(f64::from(cpu)),
                "process_rss_mib": rss_mib,
            },
            "latency": {
                "p50_ms": p50,
                "p95_ms": p95,
                "p99_ms": p99,
                "samples": nlat,
            },
            "errors": {
                "count": g.error_count,
                "recent": recent,
            },
            "quotas": {
                "parallel_limit": self.limit,
                "parallel_in_use": self.limit.saturating_sub(available),
                "tokens_per_minute": tokens,
                "requests_per_second": finite(requests as f64 / WINDOW_SECS as f64),
                "requests_in_window": requests,
                "window_secs": WINDOW_SECS,
                "default_tokens_per_minute": DEFAULT_TOKENS_PER_MINUTE,
                "default_requests_per_second": DEFAULT_REQUESTS_PER_SECOND,
                "hitting_ceiling": hitting_ceiling(available, tokens, requests),
            },
            "daemon": {
                "status": "not_this_process",
                "alive": Value::Null,
                "detail": "runa serve does not host the warm daemon (that is `runa daemon`).",
            },
            "models": {
                "configured": configured,
                "loaded_count": loaded.len(),
                "loaded": models,
            },
            "mcp_tools": [],
            "queue": {
                "depth": g.queue.len(),
                "waiting": waiting,
            },
            "reasoning": {
                "tokens_spent": g.reasoning_spent,
                "active_budget_tokens": active_budget,
            },
        })
    }

    pub(crate) fn start(self: &Arc<Self>, route: &'static str) -> Flight {
        let started = Instant::now();
        let id = {
            let mut g = self.lock();
            let id = g.next_id;
            g.next_id = g.next_id.saturating_add(1);
            g.request_marks.push_back(started);
            prune_instants(&mut g.request_marks, started);
            g.in_flight.push(LiveReq {
                id,
                route,
                model: String::new(),
                started,
                budget: None,
            });
            id
        };
        self.notify.notify_one();
        Flight {
            metrics: Arc::clone(self),
            id,
            started,
            prompt: 0,
            generated: 0,
            reasoning: 0,
            decode_tps: 0.0,
            error: None,
        }
    }

    fn enqueue(self: &Arc<Self>, route: &'static str) -> QueueTicket {
        let id = {
            let mut g = self.lock();
            let id = g.next_id;
            g.next_id = g.next_id.saturating_add(1);
            g.queue.push(Waiter {
                id,
                route,
                started: Instant::now(),
            });
            id
        };
        self.notify.notify_one();
        QueueTicket {
            metrics: Arc::clone(self),
            id,
        }
    }

    fn dequeue(&self, id: u64) {
        {
            let mut g = self.lock();
            g.queue.retain(|w| w.id != id);
        }
        self.notify.notify_one();
    }

    fn set_model(&self, id: u64, model: &str) {
        let mut g = self.lock();
        if let Some(req) = g.in_flight.iter_mut().find(|r| r.id == id) {
            req.model = model.to_owned();
        }
        drop(g);
        self.notify.notify_one();
    }

    fn set_budget(&self, id: u64, tokens: u32) {
        let mut g = self.lock();
        if let Some(req) = g.in_flight.iter_mut().find(|r| r.id == id) {
            req.budget = Some(tokens);
        }
        drop(g);
        self.notify.notify_one();
    }

    fn complete(&self, done: Finished) {
        let now = Instant::now();
        {
            let mut g = self.lock();
            g.in_flight.retain(|r| r.id != done.id);
            push_latency(
                &mut g.latencies_ms,
                done.started.elapsed().as_secs_f64() * 1000.0,
            );
            if done.prompt > 0 || done.generated > 0 {
                g.token_marks.push_back(TokenMark {
                    at: now,
                    prompt: done.prompt,
                    generated: done.generated,
                });
            }
            prune_tokens(&mut g.token_marks, now);
            g.reasoning_spent = g.reasoning_spent.saturating_add(done.reasoning);
            if done.decode_tps.is_finite() && done.decode_tps > 0.0 {
                g.decode_tps = done.decode_tps;
            }
            if let Some((message, traceback)) = done.error {
                push_error(&mut g, message, traceback);
            }
        }
        self.notify.notify_one();
    }

    #[cfg(test)]
    fn push_token_mark(&self, at: Instant, prompt: u64, generated: u64) {
        let mut g = self.lock();
        g.token_marks.push_back(TokenMark {
            at,
            prompt,
            generated,
        });
    }

    #[cfg(test)]
    fn push_latency_ms(&self, ms: f64) {
        let mut g = self.lock();
        push_latency(&mut g.latencies_ms, ms);
    }

    fn lock(&self) -> MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn sample_load(&self) -> (f32, u64) {
        let mut sys = self.load.lock().unwrap_or_else(|e| e.into_inner());
        let pid = sysinfo::Pid::from(std::process::id() as usize);
        sys.refresh_processes_specifics(
            sysinfo::ProcessesToUpdate::Some(std::slice::from_ref(&pid)),
            false,
            sysinfo::ProcessRefreshKind::nothing()
                .with_cpu()
                .with_memory(),
        );
        let Some(proc) = sys.process(pid) else {
            return (0.0, 0);
        };
        let rss_mib = proc.memory().div_ceil(1024 * 1024);
        (proc.cpu_usage(), rss_mib)
    }
}

fn push_error(g: &mut Inner, message: String, traceback: String) {
    g.error_count = g.error_count.saturating_add(1);
    if g.errors.len() == ERROR_RING {
        g.errors.pop_front();
    }
    g.errors.push_back(StoredError {
        message,
        traceback,
        at_unix_ms: unix_ms(),
    });
}

fn push_latency(buf: &mut VecDeque<f64>, ms: f64) {
    if !ms.is_finite() {
        return;
    }
    if buf.len() == LATENCY_RING {
        buf.pop_front();
    }
    buf.push_back(ms);
}

fn prune_tokens(marks: &mut VecDeque<TokenMark>, now: Instant) {
    let window = Duration::from_secs(WINDOW_SECS);
    while marks
        .front()
        .is_some_and(|m| now.saturating_duration_since(m.at) >= window)
    {
        marks.pop_front();
    }
}

fn prune_instants(marks: &mut VecDeque<Instant>, now: Instant) {
    let window = Duration::from_secs(WINDOW_SECS);
    while marks
        .front()
        .is_some_and(|t| now.saturating_duration_since(*t) >= window)
    {
        marks.pop_front();
    }
}

fn token_sums(marks: &VecDeque<TokenMark>) -> (u64, u64) {
    let mut prompt = 0u64;
    let mut generated = 0u64;
    for m in marks {
        prompt = prompt.saturating_add(m.prompt);
        generated = generated.saturating_add(m.generated);
    }
    (prompt, generated)
}

fn history_buckets(marks: &VecDeque<TokenMark>, now: Instant) -> Vec<u64> {
    let mut buckets = vec![0u64; WINDOW_SECS as usize];
    let window = Duration::from_secs(WINDOW_SECS);
    for mark in marks {
        let age = now.saturating_duration_since(mark.at);
        if age >= window {
            continue;
        }
        let secs_ago = age.as_secs().min(WINDOW_SECS - 1);
        let idx = (WINDOW_SECS - 1 - secs_ago) as usize;
        buckets[idx] = buckets[idx].saturating_add(mark.generated);
    }
    buckets
}

fn percentiles(samples: &VecDeque<f64>) -> (f64, f64, f64, usize) {
    if samples.is_empty() {
        return (0.0, 0.0, 0.0, 0);
    }
    let mut sorted: Vec<f64> = samples.iter().copied().collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    (
        percentile(&sorted, 0.50),
        percentile(&sorted, 0.95),
        percentile(&sorted, 0.99),
        sorted.len(),
    )
}

fn active_budget(live: &[LiveReq]) -> Value {
    if !live.iter().any(|r| r.budget.is_some()) {
        return Value::Null;
    }
    let sum = live
        .iter()
        .filter_map(|r| r.budget.map(u64::from))
        .fold(0u64, u64::saturating_add);
    json!(sum)
}

fn elapsed_ms(since: Instant) -> u64 {
    u64::try_from(since.elapsed().as_millis()).unwrap_or(u64::MAX)
}

fn unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
        .unwrap_or(0)
}

fn finite(n: f64) -> f64 {
    if n.is_finite() { n } else { 0.0 }
}

fn clip(s: &str) -> String {
    if s.chars().count() <= CLIP {
        s.to_owned()
    } else {
        s.chars().take(CLIP).collect()
    }
}

/// Push the latest snapshot about once a second and whenever it changes.
pub(crate) async fn push_snapshots(mut socket: WebSocket, mut rx: watch::Receiver<String>) {
    let initial = rx.borrow().clone();
    let _ = send_snapshot(&mut socket, initial).await;
    loop {
        tokio::select! {
            _ = tokio::time::sleep(Duration::from_secs(1)) => {
                let body = rx.borrow().clone();
                if send_snapshot(&mut socket, body).await.is_err() {
                    break;
                }
            }
            changed = rx.changed() => {
                if changed.is_err() {
                    break;
                }
                let body = rx.borrow_and_update().clone();
                if send_snapshot(&mut socket, body).await.is_err() {
                    break;
                }
            }
            incoming = socket.recv() => {
                match incoming {
                    Some(Ok(Message::Ping(payload))) => {
                        if socket.send(Message::Pong(payload)).await.is_err() {
                            break;
                        }
                    }
                    Some(Ok(Message::Close(_))) | None | Some(Err(_)) => break,
                    _ => {}
                }
            }
        }
    }
}

async fn send_snapshot(socket: &mut WebSocket, body: String) -> Result<(), ()> {
    if body.is_empty() {
        return Ok(());
    }
    socket
        .send(Message::Text(body.into()))
        .await
        .map_err(|_| ())
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::Router;
    use axum::extract::State;
    use axum::routing::get;
    use serde_json::Value;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    fn metrics(limit: usize) -> Arc<DashboardMetrics> {
        let sem = Arc::new(Semaphore::new(limit));
        Arc::new(DashboardMetrics::new(sem, limit))
    }

    #[test]
    fn percentile_empty_is_zero() {
        assert_eq!(percentile(&[], 0.50), 0.0);
        assert_eq!(percentile(&[], 0.99), 0.0);
    }

    #[test]
    fn percentile_nearest_rank() {
        let s = [10.0, 20.0, 30.0, 40.0, 100.0];
        assert_eq!(percentile(&s, 0.50), 30.0);
        assert_eq!(percentile(&s, 0.95), 100.0);
        assert_eq!(percentile(&s, 0.99), 100.0);
        assert_eq!(percentile(&[7.0], 0.99), 7.0);
    }

    #[test]
    fn ceiling_flag_matches_the_documented_defaults() {
        assert!(!hitting_ceiling(2, 0, 0));
        assert!(hitting_ceiling(0, 0, 0));
        assert!(hitting_ceiling(4, DEFAULT_TOKENS_PER_MINUTE, 0));
        assert!(!hitting_ceiling(4, DEFAULT_TOKENS_PER_MINUTE - 1, 0));
        let under = 16 * WINDOW_SECS - 1;
        let at = 16 * WINDOW_SECS;
        assert!(!hitting_ceiling(4, 0, under));
        assert!(hitting_ceiling(4, 0, at));
    }

    #[test]
    fn ceiling_when_semaphore_is_saturated() {
        let sem = Arc::new(Semaphore::new(1));
        let m = DashboardMetrics::new(Arc::clone(&sem), 1);
        let _hold = sem.try_acquire().unwrap();
        let v = m.to_json(&[], 0);
        assert_eq!(v["quotas"]["hitting_ceiling"], true);
        assert_eq!(v["quotas"]["parallel_in_use"], 1);
        assert_eq!(v["quotas"]["parallel_limit"], 1);
    }

    #[test]
    fn ceiling_when_token_window_exceeds_default() {
        let m = metrics(4);
        let mut flight = m.start("/v1/chat/completions");
        flight.add_tokens(DEFAULT_TOKENS_PER_MINUTE as u32, 0, 0, 0.0);
        drop(flight);
        let v = m.to_json(&[], 0);
        assert_eq!(v["quotas"]["tokens_per_minute"], DEFAULT_TOKENS_PER_MINUTE);
        assert_eq!(v["quotas"]["hitting_ceiling"], true);
    }

    #[test]
    fn stale_tokens_leave_the_window() {
        let m = metrics(4);
        m.push_token_mark(
            Instant::now() - Duration::from_secs(WINDOW_SECS + 1),
            DEFAULT_TOKENS_PER_MINUTE,
            0,
        );
        let v = m.to_json(&[], 0);
        assert_eq!(v["quotas"]["tokens_per_minute"], 0);
        assert_eq!(v["quotas"]["hitting_ceiling"], false);
    }

    #[test]
    fn error_ring_drops_oldest_and_counts_all() {
        let m = metrics(2);
        for i in 0..20 {
            m.record_error(&format!("e{i}"), "trace-line");
        }
        let v = m.to_json(&[], 0);
        assert_eq!(v["errors"]["count"], 20);
        let recent = v["errors"]["recent"].as_array().unwrap();
        assert_eq!(recent.len(), ERROR_RING);
        // Newest first. 20 records, ring keeps the last 16: e4 .. e19.
        assert_eq!(recent[0]["message"], "e19");
        assert_eq!(recent[ERROR_RING - 1]["message"], "e4");
        assert_eq!(recent[0]["traceback"], "trace-line");
    }

    #[test]
    fn latency_snapshot_uses_percentiles() {
        let m = metrics(2);
        for ms in [10.0, 20.0, 30.0, 40.0, 100.0] {
            m.push_latency_ms(ms);
        }
        let v = m.to_json(&[], 0);
        assert_eq!(v["latency"]["samples"], 5);
        assert_eq!(v["latency"]["p50_ms"], 30.0);
        assert_eq!(v["latency"]["p95_ms"], 100.0);
        assert_eq!(v["latency"]["p99_ms"], 100.0);
    }

    #[test]
    fn queue_and_inflight_show_and_clear() {
        let sem = Arc::new(Semaphore::new(1));
        let m = Arc::new(DashboardMetrics::new(Arc::clone(&sem), 1));
        let _hold = sem.try_acquire_owned().unwrap();
        let ticket = m.enqueue("/v1/messages");
        let queued = m.to_json(&[], 2);
        assert_eq!(queued["queue"]["depth"], 1);
        assert_eq!(queued["queue"]["waiting"][0]["route"], "/v1/messages");
        drop(ticket);
        assert_eq!(m.to_json(&[], 2)["queue"]["depth"], 0);

        let mut flight = m.start("/v1/chat/completions");
        flight.set_model("qwen");
        flight.note_budget(256);
        flight.add_tokens(3, 5, 2, 12.5);
        let live = m.to_json(&[("qwen".into(), 4096)], 2);
        assert_eq!(live["in_flight"][0]["model"], "qwen");
        assert_eq!(live["in_flight"][0]["route"], "/v1/chat/completions");
        assert_eq!(live["models"]["loaded_count"], 1);
        assert_eq!(live["models"]["loaded"][0]["memory_bytes"], 4096);
        assert_eq!(live["reasoning"]["active_budget_tokens"], 256);
        drop(flight);
        let done = m.to_json(&[], 2);
        assert!(done["in_flight"].as_array().unwrap().is_empty());
        assert_eq!(done["reasoning"]["tokens_spent"], 2);
        assert_eq!(done["reasoning"]["active_budget_tokens"], Value::Null);
        assert_eq!(done["token_speed"]["decode_toks_per_s"], 12.5);
        assert_eq!(done["daemon"]["status"], "not_this_process");
        assert_eq!(done["daemon"]["alive"], Value::Null);
        assert!(done["mcp_tools"].as_array().unwrap().is_empty());
    }

    async fn serve_snapshot(State(metrics): State<Arc<DashboardMetrics>>) -> axum::Json<Value> {
        axum::Json(metrics.to_json(&[], 0))
    }

    #[tokio::test]
    async fn snapshot_route_has_required_keys() {
        let app = Router::new()
            .route("/dashboard/snapshot", get(serve_snapshot))
            .with_state(metrics(2));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        let body = http_get(addr, "/dashboard/snapshot").await;
        server.abort();
        let json_text = body.split_once("\r\n\r\n").map(|(_, b)| b).unwrap_or("");
        let v: Value = serde_json::from_str(json_text.trim()).expect(json_text);
        for key in [
            "token_speed",
            "in_flight",
            "system_load",
            "latency",
            "errors",
            "quotas",
            "daemon",
            "models",
            "mcp_tools",
            "queue",
            "reasoning",
        ] {
            assert!(v.get(key).is_some(), "missing {key} in {v}");
        }
        assert!(v["latency"].get("p50_ms").is_some());
        assert!(v["latency"].get("p95_ms").is_some());
        assert!(v["latency"].get("p99_ms").is_some());
        assert!(v["errors"].get("count").is_some());
        assert!(v["errors"].get("recent").is_some());
        assert!(v["quotas"].get("hitting_ceiling").is_some());
        assert_eq!(v["quotas"]["window_secs"], WINDOW_SECS);
        assert_eq!(
            v["quotas"]["default_tokens_per_minute"],
            DEFAULT_TOKENS_PER_MINUTE
        );
        assert!(v["system_load"].get("process_cpu_percent").is_some());
        assert!(v["system_load"].get("process_rss_mib").is_some());
    }

    async fn http_get(addr: std::net::SocketAddr, path: &str) -> String {
        let mut sock = None;
        let mut last = String::new();
        for _ in 0..50 {
            match tokio::net::TcpStream::connect(addr).await {
                Ok(s) => {
                    sock = Some(s);
                    break;
                }
                Err(e) => {
                    last = e.to_string();
                    tokio::time::sleep(Duration::from_millis(20)).await;
                }
            }
        }
        let mut sock = sock.unwrap_or_else(|| panic!("connect {addr}: {last}"));
        let req = format!("GET {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n");
        sock.write_all(req.as_bytes()).await.unwrap();
        let mut buf = Vec::new();
        sock.read_to_end(&mut buf).await.unwrap();
        String::from_utf8(buf).unwrap()
    }
}
