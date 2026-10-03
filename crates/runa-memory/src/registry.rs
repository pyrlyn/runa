// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! Task-claim registry over a markdown table (plan D18, P7.4).

use std::fs::{self, File, OpenOptions};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use chrono::Utc;

/// Live claim returned on successful [`TaskRegistry::claim`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TaskClaim {
    pub task_id: String,
    pub agent: String,
    pub started_at: String,
}

/// Row status (read-only view).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TaskStatus {
    Free,
    InProgress { agent: String, started_at: String },
}

/// Claim/release failures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClaimError {
    NotFound,
    AlreadyClaimed {
        agent: String,
        started_at: String,
    },
    NotOwner {
        agent: String,
    },
    /// The registry file could not be read or written. Distinct from
    /// [`ClaimError::NotFound`] so a bad path is not reported as a bad id.
    Io(String),
    /// Empty, `|`, or a control character. Those bytes break the markdown row.
    InvalidAgent,
}

impl std::fmt::Display for ClaimError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound => write!(f, "task not found"),
            Self::AlreadyClaimed { agent, started_at } => {
                write!(f, "already claimed by {agent} since {started_at}")
            }
            Self::NotOwner { agent } => write!(f, "held by another agent (not {agent})"),
            Self::Io(msg) => write!(f, "{msg}"),
            Self::InvalidAgent => write!(
                f,
                "agent name must be non-empty and must not contain '|' or control characters"
            ),
        }
    }
}

impl std::error::Error for ClaimError {}

/// Cooperative claims backed by `docs/tasks.md` (file-backed, P7.4).
pub struct TaskRegistry {
    path: PathBuf,
    lock: Mutex<()>,
}

impl TaskRegistry {
    pub fn open(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            lock: Mutex::new(()),
        }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn list_free(&self) -> Result<Vec<String>, ClaimError> {
        self.locked(|| {
            let rows = self.read_rows()?;
            let mut free: Vec<String> = rows
                .into_iter()
                .filter(|r| r.status == "free")
                .map(|r| r.task_id)
                .collect();
            free.sort_by_key(|a| task_order_key(a));
            Ok(free)
        })
    }

    pub fn status(&self, task_id: &str) -> Result<Option<TaskStatus>, ClaimError> {
        self.locked(|| {
            let rows = self.read_rows()?;
            Ok(rows
                .into_iter()
                .find(|r| r.task_id == task_id)
                .map(|r| r.into_status()))
        })
    }

    pub fn claim(&self, task_id: &str, agent: &str) -> Result<TaskClaim, ClaimError> {
        let agent = validate_agent(agent)?;
        self.locked(|| {
            let mut rows = self.read_rows()?;
            let idx = rows
                .iter()
                .position(|r| r.task_id == task_id)
                .ok_or(ClaimError::NotFound)?;
            let row = &rows[idx];
            if row.status == "in progress" {
                return Err(ClaimError::AlreadyClaimed {
                    agent: row.agent.clone(),
                    started_at: row.started.clone(),
                });
            }
            let started_at = Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string();
            rows[idx].status = "in progress".to_owned();
            rows[idx].agent = agent.to_owned();
            rows[idx].started = started_at.clone();
            self.write_rows(&rows)?;
            Ok(TaskClaim {
                task_id: task_id.to_owned(),
                agent: agent.to_owned(),
                started_at,
            })
        })
    }

    pub fn release(&self, task_id: &str, agent: &str) -> Result<(), ClaimError> {
        let agent = validate_agent(agent)?;
        self.locked(|| {
            let mut rows = self.read_rows()?;
            let idx = rows
                .iter()
                .position(|r| r.task_id == task_id)
                .ok_or(ClaimError::NotFound)?;
            let row = &rows[idx];
            if row.status == "in progress" && row.agent != agent {
                return Err(ClaimError::NotOwner {
                    agent: row.agent.clone(),
                });
            }
            rows[idx].status = "free".to_owned();
            rows[idx].agent.clear();
            rows[idx].started.clear();
            self.write_rows(&rows)?;
            Ok(())
        })
    }

    /// In-process mutex, then an OS lock on a sidecar, so two CLI processes
    /// cannot both win the same claim. The guard is dropped by the caller
    /// after the write is renamed into place.
    fn locked<T>(&self, f: impl FnOnce() -> Result<T, ClaimError>) -> Result<T, ClaimError> {
        let _guard = self.lock.lock().unwrap_or_else(|e| e.into_inner());
        let file = self.file_lock()?;
        let out = f();
        let _ = file.unlock();
        out
    }

    fn file_lock(&self) -> Result<File, ClaimError> {
        let path = lock_path(&self.path);
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)
            .map_err(|e| io_err("lock", &path, &e))?;
        file.lock().map_err(|e| io_err("lock", &path, &e))?;
        Ok(file)
    }

    fn read_rows(&self) -> Result<Vec<Row>, ClaimError> {
        let text = fs::read_to_string(&self.path).map_err(|e| io_err("read", &self.path, &e))?;
        Ok(parse_rows(&text))
    }

    fn write_rows(&self, rows: &[Row]) -> Result<(), ClaimError> {
        let text = fs::read_to_string(&self.path).map_err(|e| io_err("read", &self.path, &e))?;
        let updated = rewrite_rows(&text, rows);
        let tmp = temp_path(&self.path);
        fs::write(&tmp, updated).map_err(|e| io_err("write", &self.path, &e))?;
        fs::rename(&tmp, &self.path).map_err(|e| {
            let _ = fs::remove_file(&tmp);
            io_err("write", &self.path, &e)
        })
    }
}

fn lock_path(path: &Path) -> PathBuf {
    let mut name = path
        .file_name()
        .map(|s| s.to_os_string())
        .unwrap_or_else(|| "tasks.md".into());
    name.push(".lock");
    path.with_file_name(name)
}

fn temp_path(path: &Path) -> PathBuf {
    let name = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("tasks.md");
    path.with_file_name(format!("{name}.tmp"))
}

fn io_err(op: &str, path: &Path, err: &std::io::Error) -> ClaimError {
    ClaimError::Io(format!("cannot {op} {}: {err}", path.display()))
}

fn validate_agent(agent: &str) -> Result<&str, ClaimError> {
    let agent = agent.trim();
    if agent.is_empty() || agent.contains('|') || agent.chars().any(|c| c.is_control()) {
        return Err(ClaimError::InvalidAgent);
    }
    Ok(agent)
}

#[derive(Debug, Clone)]
struct Row {
    task_id: String,
    status: String,
    agent: String,
    started: String,
}

impl Row {
    fn into_status(self) -> TaskStatus {
        if self.status == "free" {
            TaskStatus::Free
        } else {
            TaskStatus::InProgress {
                agent: self.agent,
                started_at: self.started,
            }
        }
    }
}

fn parse_rows(text: &str) -> Vec<Row> {
    let mut out = Vec::new();
    for line in text.lines() {
        if line.starts_with('#') {
            continue;
        }
        let Some(row) = parse_row_line(line) else {
            continue;
        };
        out.push(row);
    }
    out
}

fn parse_row_line(line: &str) -> Option<Row> {
    let line = line.trim();
    if !line.starts_with('|') {
        return None;
    }
    let parts: Vec<&str> = line.split('|').map(str::trim).collect();
    // | Task | Status | Agent | Started |
    if parts.len() < 6 {
        return None;
    }
    let task = parts[1];
    if task == "Task" || task == "------" {
        return None;
    }
    if !is_task_id(task) {
        return None;
    }
    Some(Row {
        task_id: task.to_owned(),
        status: parts[2].to_owned(),
        agent: parts[3].to_owned(),
        started: parts[4].to_owned(),
    })
}

fn is_task_id(s: &str) -> bool {
    if let Some(rest) = s.strip_prefix('P')
        && let Some((phase, sub)) = rest.split_once('.')
    {
        return !phase.is_empty()
            && phase.chars().all(|c| c.is_ascii_digit())
            && !sub.is_empty()
            && sub.chars().all(|c| c.is_ascii_digit());
    }
    if let Some(rest) = s.strip_prefix('K') {
        return !rest.is_empty() && rest.chars().all(|c| c.is_ascii_digit());
    }
    false
}

fn rewrite_rows(text: &str, rows: &[Row]) -> String {
    let map: std::collections::HashMap<&str, &Row> =
        rows.iter().map(|r| (r.task_id.as_str(), r)).collect();
    let mut out = String::new();
    for line in text.lines() {
        if let Some(row) = parse_row_line(line)
            && let Some(updated) = map.get(row.task_id.as_str())
        {
            out.push_str(&format!(
                "| {} | {} | {} | {} |\n",
                updated.task_id, updated.status, updated.agent, updated.started
            ));
            continue;
        }
        out.push_str(line);
        out.push('\n');
    }
    if !text.ends_with('\n') && out.ends_with('\n') {
        out.pop();
    }
    out
}

fn task_order_key(id: &str) -> (u32, u32, u32) {
    if let Some(rest) = id.strip_prefix('P')
        && let Some((phase, sub)) = rest.split_once('.')
    {
        return (0, phase.parse().unwrap_or(999), sub.parse().unwrap_or(999));
    }
    if let Some(rest) = id.strip_prefix('K') {
        return (1, rest.parse().unwrap_or(999), 0);
    }
    (2, 999, 999)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn sample_registry() -> String {
        "| Task | Status | Agent | Started (UTC) |\n\
         |------|--------|-------|---------------|\n\
         | P9.1 | free | | |\n\
         | P9.2 | in progress | alice | 2026-09-01T10:00:00Z |\n\
         | K1 | free | | |\n"
            .to_owned()
    }

    #[test]
    fn list_free_sorted() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tasks.md");
        fs::write(&path, sample_registry()).unwrap();
        let reg = TaskRegistry::open(path);
        assert_eq!(
            reg.list_free().unwrap(),
            vec!["P9.1".to_owned(), "K1".to_owned()]
        );
    }

    #[test]
    fn claim_and_release_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tasks.md");
        fs::write(&path, sample_registry()).unwrap();
        let reg = TaskRegistry::open(path.clone());

        let claim = reg.claim("P9.1", "bob").unwrap();
        assert_eq!(claim.task_id, "P9.1");
        assert_eq!(claim.agent, "bob");
        assert!(claim.started_at.ends_with('Z'));

        assert!(matches!(
            reg.status("P9.1").unwrap(),
            Some(TaskStatus::InProgress { .. })
        ));

        reg.release("P9.1", "bob").unwrap();
        assert_eq!(reg.status("P9.1").unwrap(), Some(TaskStatus::Free));
        assert!(reg.list_free().unwrap().contains(&"P9.1".to_owned()));
    }

    #[test]
    fn double_claim_fails_with_owner() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tasks.md");
        fs::write(&path, sample_registry()).unwrap();
        let reg = TaskRegistry::open(path);

        let err = reg.claim("P9.2", "bob").unwrap_err();
        assert_eq!(
            err,
            ClaimError::AlreadyClaimed {
                agent: "alice".to_owned(),
                started_at: "2026-09-01T10:00:00Z".to_owned(),
            }
        );
        // first holder unaffected
        assert_eq!(
            reg.status("P9.2").unwrap(),
            Some(TaskStatus::InProgress {
                agent: "alice".to_owned(),
                started_at: "2026-09-01T10:00:00Z".to_owned(),
            })
        );
    }

    #[test]
    fn release_wrong_agent_fails() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tasks.md");
        fs::write(&path, sample_registry()).unwrap();
        let reg = TaskRegistry::open(path);

        let err = reg.release("P9.2", "bob").unwrap_err();
        assert_eq!(
            err,
            ClaimError::NotOwner {
                agent: "alice".to_owned(),
            }
        );
    }

    #[test]
    fn claim_persists_to_disk() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tasks.md");
        let mut file = fs::File::create(&path).unwrap();
        write!(file, "{}", sample_registry()).unwrap();
        let reg = TaskRegistry::open(path.clone());
        reg.claim("K1", "carol").unwrap();
        let text = fs::read_to_string(path).unwrap();
        assert!(text.contains("| K1 | in progress | carol |"));
    }

    #[test]
    fn agent_names_with_pipes_or_empty_are_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tasks.md");
        fs::write(&path, sample_registry()).unwrap();
        let reg = TaskRegistry::open(&path);
        assert_eq!(
            reg.claim("P9.1", "a|b").unwrap_err(),
            ClaimError::InvalidAgent
        );
        assert_eq!(reg.claim("P9.1", "").unwrap_err(), ClaimError::InvalidAgent);
        assert_eq!(
            reg.claim("P9.1", " \n").unwrap_err(),
            ClaimError::InvalidAgent
        );
        let text = fs::read_to_string(&path).unwrap();
        assert!(text.contains("| P9.1 | free |"));
    }

    #[test]
    fn missing_registry_is_an_io_error() {
        let path = std::env::temp_dir().join(format!(
            "runa-missing-registry-{}-{}.md",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let reg = TaskRegistry::open(&path);
        let err = reg.claim("P9.1", "qa").unwrap_err();
        match err {
            ClaimError::Io(msg) => {
                assert!(msg.contains("cannot read"), "{msg}");
                // Windows and Unix phrase a missing path differently; both are NotFound.
                assert!(msg.contains("os error 2"), "{msg}");
            }
            other => panic!("expected io, got {other}"),
        }
    }

    #[test]
    fn two_registries_cannot_both_claim() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tasks.md");
        fs::write(&path, sample_registry()).unwrap();
        let a = TaskRegistry::open(&path);
        let b = TaskRegistry::open(path);
        for _ in 0..8 {
            if a.status("P9.1").unwrap() != Some(TaskStatus::Free) {
                a.release("P9.1", "alice").ok();
                a.release("P9.1", "bob").ok();
            }
            let (left, right) = std::thread::scope(|s| {
                let h1 = s.spawn(|| a.claim("P9.1", "alice"));
                let h2 = s.spawn(|| b.claim("P9.1", "bob"));
                (h1.join().unwrap(), h2.join().unwrap())
            });
            let wins = [&left, &right].iter().filter(|r| r.is_ok()).count();
            assert_eq!(wins, 1, "left={left:?} right={right:?}");
        }
    }
}
