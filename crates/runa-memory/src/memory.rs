// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! Adaptive memory manager (plan D17, P7.1).

use std::sync::{Mutex, MutexGuard, PoisonError};
use std::time::Instant;

/// Idle / normal / heavy load state (plan §5).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadState {
    Idle,
    Normal,
    Heavy,
}

/// Policy knobs from `[memory]` in config (plan §6).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryPolicy {
    pub idle_timeout_s: u64,
    pub floor_mib: u64,
    pub max_growth_mib: u64,
}

impl Default for MemoryPolicy {
    fn default() -> Self {
        Self {
            idle_timeout_s: 300,
            floor_mib: 512,
            max_growth_mib: 4096,
        }
    }
}

/// Snapshot returned by [`MemoryManager::current_usage`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Usage {
    pub rss_mib: u64,
    pub budget_mib: u64,
    pub state: LoadState,
}

/// Grow failures (plan P7.3 vocabulary for suggestions).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemoryError {
    OverCeiling {
        demand_mib: u64,
        ceiling_mib: u64,
        suggestion: String,
    },
}

impl std::fmt::Display for MemoryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OverCeiling {
                demand_mib,
                ceiling_mib,
                suggestion,
            } => write!(
                f,
                "need {demand_mib} MiB but ceiling is {ceiling_mib} MiB — try {suggestion}"
            ),
        }
    }
}

impl std::error::Error for MemoryError {}

/// Backend hook for RSS changes (fake in unit tests, real RSS in P7.2).
pub trait MemoryBackend: Send {
    fn rss_mib(&self) -> u64;
    /// Release caches/buffers down to at most `target_mib` (never below model floor).
    fn shrink_to(&mut self, target_mib: u64);
    /// Pre-grow by `additional_mib`; returns new RSS.
    fn grow(&mut self, additional_mib: u64) -> u64;
}

/// In-memory backend for unit tests.
#[derive(Debug)]
pub struct FakeBackend {
    rss_mib: u64,
    /// Model weights that must stay resident (never unloaded).
    model_floor_mib: u64,
}

impl FakeBackend {
    pub fn new(rss_mib: u64, model_floor_mib: u64) -> Self {
        Self {
            rss_mib,
            model_floor_mib,
        }
    }
}

impl MemoryBackend for FakeBackend {
    fn rss_mib(&self) -> u64 {
        self.rss_mib
    }

    fn shrink_to(&mut self, target_mib: u64) {
        let floor = self.model_floor_mib;
        self.rss_mib = target_mib.max(floor).min(self.rss_mib);
    }

    fn grow(&mut self, additional_mib: u64) -> u64 {
        self.rss_mib += additional_mib;
        self.rss_mib
    }
}

/// Real RSS backend over `sysinfo` (P9.1).
///
/// `rss_mib` reads this process's resident set from the OS. `shrink_to`
/// and `grow` are advisory no-ops returning current RSS: the OS owns the
/// pages, so actual release happens through the owner's release paths
/// (`LoadedModel::on_idle`, pool LRU eviction) while the manager's state
/// machine and logs still apply. Unit tests keep using [`FakeBackend`].
#[derive(Debug, Default)]
pub struct SysinfoBackend;

impl SysinfoBackend {
    pub fn new() -> Self {
        Self
    }

    /// Current process RSS in MiB (0 when the process table is unreadable).
    pub fn process_rss_mib() -> u64 {
        let pid = sysinfo::Pid::from(std::process::id() as usize);
        let mut sys = sysinfo::System::new();
        sys.refresh_processes(
            sysinfo::ProcessesToUpdate::Some(std::slice::from_ref(&pid)),
            false,
        );
        sys.process(pid)
            .map(|p| p.memory().div_ceil(1024 * 1024))
            .unwrap_or(0)
    }
}

impl MemoryBackend for SysinfoBackend {
    fn rss_mib(&self) -> u64 {
        Self::process_rss_mib()
    }

    fn shrink_to(&mut self, _target_mib: u64) {}

    fn grow(&mut self, _additional_mib: u64) -> u64 {
        self.rss_mib()
    }
}

/// Adaptive process memory (plan D17, docs/memory.md).
pub struct MemoryManager {
    policy: MemoryPolicy,
    fit_ceiling_mib: u64,
    backend: Mutex<Box<dyn MemoryBackend>>,
    state: Mutex<LoadState>,
    granted_growth_mib: Mutex<u64>,
    last_activity: Mutex<Instant>,
}

impl MemoryManager {
    /// `fit_ceiling_mib` is the fit verdict + margin (MiB).
    pub fn new(
        policy: MemoryPolicy,
        fit_ceiling_mib: u64,
        backend: Box<dyn MemoryBackend>,
    ) -> Self {
        Self {
            policy,
            fit_ceiling_mib,
            backend: Mutex::new(backend),
            state: Mutex::new(LoadState::Normal),
            granted_growth_mib: Mutex::new(0),
            last_activity: Mutex::new(Instant::now()),
        }
    }

    fn ceiling_mib(&self, current_rss: u64) -> u64 {
        let growth_cap = current_rss.saturating_add(self.policy.max_growth_mib);
        self.fit_ceiling_mib.min(growth_cap)
    }

    fn log_transition(&self, action: &str, before: u64, after: u64) {
        eprintln!("memory: {action}: rss {before} -> {after} MiB");
    }

    pub fn current_usage(&self) -> Usage {
        let rss = lock(&self.backend).rss_mib();
        let state = *lock(&self.state);
        Usage {
            rss_mib: rss,
            budget_mib: self.ceiling_mib(rss),
            state,
        }
    }

    /// Record a request/job so idle shrink waits a full `idle_timeout_s`.
    pub fn touch(&self) {
        *lock(&self.last_activity) = Instant::now();
    }

    /// Shrink if nothing has `touch`ed for `idle_timeout_s` (P7.2).
    pub fn maybe_idle(&self) {
        let elapsed = lock(&self.last_activity).elapsed();
        if elapsed.as_secs() >= self.policy.idle_timeout_s {
            self.on_idle();
        }
    }

    pub fn on_idle(&self) {
        let before = lock(&self.backend).rss_mib();
        lock(&self.backend).shrink_to(self.policy.floor_mib);
        let after = lock(&self.backend).rss_mib();
        if before != after {
            self.log_transition("on_idle", before, after);
        }
        *lock(&self.state) = LoadState::Idle;
        *lock(&self.granted_growth_mib) = 0;
        *lock(&self.last_activity) = Instant::now();
    }

    pub fn on_heavy(&self, demand_mib: u64) {
        let _ = self.grow_for(demand_mib);
    }

    pub fn shrink_to_floor(&self) {
        let before = lock(&self.backend).rss_mib();
        lock(&self.backend).shrink_to(self.policy.floor_mib);
        let after = lock(&self.backend).rss_mib();
        if before != after {
            self.log_transition("shrink_to_floor", before, after);
        }
        *lock(&self.state) = LoadState::Idle;
        *lock(&self.granted_growth_mib) = 0;
    }

    pub fn grow_for(&self, demand_mib: u64) -> Result<(), MemoryError> {
        self.touch();
        let current = lock(&self.backend).rss_mib();
        let ceiling = self.ceiling_mib(current);
        if current.saturating_add(demand_mib) > ceiling {
            return Err(MemoryError::OverCeiling {
                demand_mib,
                ceiling_mib: ceiling,
                suggestion: over_ceiling_suggestion(),
            });
        }
        let before = current;
        let after = lock(&self.backend).grow(demand_mib);
        self.log_transition("grow_for", before, after);
        *lock(&self.granted_growth_mib) += demand_mib;
        *lock(&self.state) = LoadState::Heavy;
        Ok(())
    }
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

fn over_ceiling_suggestion() -> String {
    "smaller --ctx, another quant, --kv q8_0, or cloud".to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mgr(rss: u64, ceiling: u64) -> MemoryManager {
        let policy = MemoryPolicy {
            idle_timeout_s: 300,
            floor_mib: 512,
            max_growth_mib: 1024,
        };
        MemoryManager::new(policy, ceiling, Box::new(FakeBackend::new(rss, 256)))
    }

    #[test]
    fn current_usage_within_ceiling() {
        let mm = mgr(800, 2048);
        let u = mm.current_usage();
        assert_eq!(u.rss_mib, 800);
        assert!(u.rss_mib <= u.budget_mib);
        assert_eq!(u.state, LoadState::Normal);
    }

    #[test]
    fn shrink_to_floor_never_below_model() {
        let mm = mgr(900, 2048);
        mm.shrink_to_floor();
        assert_eq!(mm.current_usage().rss_mib, 512);
        assert_eq!(mm.current_usage().state, LoadState::Idle);
    }

    #[test]
    fn grow_for_heavy_ctx() {
        let mm = mgr(600, 4096);
        mm.grow_for(512).unwrap();
        assert_eq!(mm.current_usage().rss_mib, 1112);
        assert_eq!(mm.current_usage().state, LoadState::Heavy);
    }

    #[test]
    fn grow_for_over_ceiling_errors_with_suggestion() {
        let mm = mgr(900, 1000);
        let err = mm.grow_for(500).unwrap_err();
        assert_eq!(
            err,
            MemoryError::OverCeiling {
                demand_mib: 500,
                ceiling_mib: 1000,
                suggestion: over_ceiling_suggestion(),
            }
        );
    }

    #[test]
    fn grow_for_respects_max_growth_cap() {
        let mm = mgr(600, 4096);
        let err = mm.grow_for(1025).unwrap_err();
        match err {
            MemoryError::OverCeiling {
                demand_mib,
                ceiling_mib,
                suggestion,
            } => {
                assert_eq!(demand_mib, 1025);
                assert_eq!(ceiling_mib, 1624);
                assert!(suggestion.contains("smaller --ctx"));
            }
        }
    }

    #[test]
    fn on_idle_shrinks_toward_floor() {
        let mm = mgr(1200, 4096);
        mm.grow_for(200).unwrap();
        mm.on_idle();
        assert_eq!(mm.current_usage().rss_mib, 512);
        assert_eq!(mm.current_usage().state, LoadState::Idle);
    }

    #[test]
    fn maybe_idle_skips_when_recently_touched() {
        let policy = MemoryPolicy {
            idle_timeout_s: 3600,
            floor_mib: 512,
            max_growth_mib: 1024,
        };
        let mm = MemoryManager::new(policy, 4096, Box::new(FakeBackend::new(1200, 256)));
        mm.grow_for(200).unwrap();
        mm.maybe_idle();
        assert_eq!(mm.current_usage().state, LoadState::Heavy);
    }

    #[test]
    fn maybe_idle_shrinks_after_timeout() {
        let policy = MemoryPolicy {
            idle_timeout_s: 0,
            floor_mib: 512,
            max_growth_mib: 1024,
        };
        let mm = MemoryManager::new(policy, 4096, Box::new(FakeBackend::new(1200, 256)));
        mm.grow_for(200).unwrap();
        mm.maybe_idle();
        assert_eq!(mm.current_usage().state, LoadState::Idle);
        assert_eq!(mm.current_usage().rss_mib, 512);
        assert!(mm.current_usage().rss_mib <= 512 + 512 / 10);
    }

    #[test]
    fn sysinfo_backend_reports_live_rss() {
        let mut backend = SysinfoBackend::new();
        // The test process is alive, so RSS is non-zero; advisory
        // shrink/grow never fail and keep reporting live RSS (which may
        // move between reads as the allocator works).
        assert!(backend.rss_mib() > 0, "live RSS must be non-zero");
        backend.shrink_to(1);
        assert!(backend.rss_mib() > 0);
        assert!(backend.grow(512) > 0);
    }
}
