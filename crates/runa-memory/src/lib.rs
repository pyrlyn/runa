// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! runa-memory — adaptive memory management (plan D17): shrink toward a
//! floor after `idle_timeout_s` with no request/job, bounded pre-grow for
//! heavy tasks; every transition logged with before/after RSS. Also hosts
//! the task-claim registry over `docs/tasks.md` (plan D18, P7.4).
//!
//! Public-method docs: `docs/memory.md`.

mod memory;
mod registry;

pub use memory::{
    FakeBackend, LoadState, MemoryBackend, MemoryError, MemoryManager, MemoryPolicy,
    SysinfoBackend, Usage,
};
pub use registry::{ClaimError, TaskClaim, TaskRegistry, TaskStatus};

/// Every public method documented in `docs/memory.md` (P7.5 / P6.4 lint).
#[cfg(test)]
mod docs_lint {
    const METHODS: &[&str] = &[
        "current_usage",
        "touch",
        "maybe_idle",
        "on_idle",
        "on_heavy",
        "shrink_to_floor",
        "grow_for",
        "list_free",
        "status",
        "claim",
        "release",
    ];

    #[test]
    fn public_methods_in_memory_md() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/memory.md");
        let doc = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        for method in METHODS {
            let needle = format!("fn {method}(");
            assert!(doc.contains(&needle), "docs/memory.md missing `{needle}`");
        }
    }
}
