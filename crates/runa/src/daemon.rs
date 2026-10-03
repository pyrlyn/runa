// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! Background daemon (P9.1): keeps models warm between CLI calls, owns
//! the adaptive [`MemoryManager`](runa_memory::MemoryManager), and serves
//! `run` / `chat` over a Unix socket ([`daemon_proto::default_socket_path`]).
//!
//! Foreground only: process supervision (restart, logging) belongs to
//! launchd/systemd — see `--install` / `--uninstall`.

use std::path::{Path, PathBuf};
#[cfg(unix)]
use std::sync::atomic::AtomicU32;
use std::sync::{Arc, Mutex};

#[cfg(any(unix, test))]
use runa_core::BackendKind;
use runa_engine::{LoadConfig, Mode, Placement};
use runa_memory::MemoryManager;
#[cfg(unix)]
use runa_memory::SysinfoBackend;
#[cfg(unix)]
use tokio::net::{UnixListener, UnixStream};

#[cfg(any(unix, test))]
use crate::daemon_proto::encode_line;
use crate::daemon_proto::{
    DaemonEvent, DaemonRequest, PROTOCOL_VERSION, decode_line, default_socket_path,
};
#[cfg(unix)]
use crate::daemon_proto::{MAX_IDLE_TICK_SECS, read_line, write_line};
use crate::pool::ModelPool;
#[cfg(unix)]
use crate::pool::Warmup;

/// Admission preflight per request (MiB). Model residency is pool/LRU
/// bound, so the preflight only records activity and lets the manager
/// clamp growth; see `serve_request`. Unix-only: the serving path.
#[cfg_attr(not(unix), allow(dead_code))]
const REQUEST_DEMAND_MIB: u64 = 0;

/// How long a daemon client waits for one request (model load happens
/// inside the daemon, so this must cover cold starts).
pub(crate) const REQUEST_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(1800);

/// CLI bundle for `runa daemon`.
pub(crate) struct DaemonOpts {
    pub models: Vec<(String, PathBuf)>,
    pub mode: String,
    pub ctx: u32,
    pub max_loaded: Option<usize>,
    /// CLI `--max-load-percent` for the startup cap check (warn-only).
    pub max_load_percent: Option<u8>,
    /// LoRA adapters applied to every daemon model.
    pub loras: Vec<runa_engine::LoraSpec>,
    /// Socket path override (`None` = [`default_socket_path`]).
    pub socket: Option<PathBuf>,
    pub install: bool,
    pub uninstall: bool,
    /// Raw argv fragment baked into installed units (see [`daemon_argv`]).
    pub argv: Vec<String>,
}

pub(crate) fn cmd_daemon(opts: DaemonOpts) -> Result<(), String> {
    let home = home_dir()?;
    if opts.install {
        if opts.models.is_empty() {
            return Err("daemon --install needs at least one model".into());
        }
        let exe = std::env::current_exe().map_err(|e| e.to_string())?;
        let paths = install_daemon(&home, &exe, &opts.argv)?;
        for p in &paths {
            println!("installed {}", p.display());
        }
        return Ok(());
    }
    if opts.uninstall {
        let removed = uninstall_daemon(&home)?;
        if removed.is_empty() {
            println!("no daemon units installed");
        }
        for p in &removed {
            println!("removed {}", p.display());
        }
        return Ok(());
    }
    if opts.models.is_empty() {
        return Err("daemon: need at least one model (positional or --models)".into());
    }
    crate::config::warn_if_over_system_limit(None, None, opts.max_load_percent)?;
    let placement_base = match crate::parse_mode_choice(&opts.mode)? {
        crate::ModeChoice::Fixed(m) => Placement::from_mode(m),
        crate::ModeChoice::Auto => Placement::from_mode(Mode::Cpu),
    };
    let config = LoadConfig {
        n_ctx: opts.ctx,
        loras: opts.loras,
        ..LoadConfig::default()
    };
    let max_loaded = opts.max_loaded.unwrap_or_else(|| opts.models.len().max(1));
    let default_id = opts.models[0].0.clone();
    let socket = opts.socket.clone().unwrap_or_else(default_socket_path);
    let rt = tokio::runtime::Runtime::new().map_err(|e| e.to_string())?;
    rt.block_on(serve(
        opts.models,
        placement_base,
        opts.mode,
        config,
        default_id,
        max_loaded,
        socket,
        opts.max_load_percent,
    ))
}

#[allow(clippy::too_many_arguments)]
#[cfg(unix)]
async fn serve(
    models: Vec<(String, PathBuf)>,
    placement_base: Placement,
    mode: String,
    config: LoadConfig,
    default_id: String,
    max_loaded: usize,
    socket: PathBuf,
    max_load_percent: Option<u8>,
) -> Result<(), String> {
    let progress = Arc::new(AtomicU32::new(0));
    // Single slot by design: only the startup warm-up reports it, so later LRU reload overwrites go unread.
    let config = LoadConfig {
        progress: Some(Arc::clone(&progress)),
        ..config
    };
    let policy = crate::config::resolve_memory_policy()?;
    let tick_secs = policy.idle_timeout_s.clamp(1, MAX_IDLE_TICK_SECS);
    let idle_timeout = std::time::Duration::from_secs(policy.idle_timeout_s.max(1));
    let pool = ModelPool::new(
        models,
        BackendKind::Auto,
        placement_base,
        mode,
        config,
        max_loaded,
    )?
    .with_idle_timeout(idle_timeout)
    .with_max_load_percent(max_load_percent);
    let pool = Arc::new(Mutex::new(pool));
    let warm = Arc::new(Warmup::new(default_id, Arc::clone(&progress)));
    let mm = Arc::new(MemoryManager::new(
        policy,
        crate::memory_ceiling_mib(),
        Box::new(SysinfoBackend::new()),
    ));
    if let Some(parent) = socket.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    // A live socket means another daemon is serving it. Only a refused
    // connect (dead process, leftover path) is replaced.
    let (listener, bound_id) = bind_daemon_socket(&socket)?;
    eprintln!("listening on {}", socket.display());
    // Requests that arrive meanwhile queue on the pool lock the warm-up holds.
    let (warm_pool, warm_state) = (Arc::clone(&pool), Arc::clone(&warm));
    tokio::task::spawn_blocking(move || crate::pool::warm_up(&warm_pool, &warm_state, "daemon"));
    tokio::spawn(crate::pool::report_progress(Arc::clone(&warm), "daemon"));
    let idle_mm = Arc::clone(&mm);
    let idle_pool = Arc::clone(&pool);
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(std::time::Duration::from_secs(tick_secs));
        loop {
            tick.tick().await;
            // P10.5: manager shrink decision plus the pool sweep
            // (prompt caches released, models kept).
            crate::pool::idle_tick(&idle_pool, &idle_mm, "daemon").await;
        }
    });
    loop {
        let (stream, _) = tokio::select! {
            accepted = listener.accept() => accepted.map_err(|e| e.to_string())?,
            _ = tokio::signal::ctrl_c() => break,
        };
        let (pool, mm) = (Arc::clone(&pool), Arc::clone(&mm));
        tokio::spawn(async move {
            if let Err(e) = serve_conn(stream, &pool, &mm).await {
                eprintln!("daemon: connection: {e}");
            }
        });
    }
    remove_owned_socket(&socket, bound_id);
    Ok(())
}

#[cfg(unix)]
fn socket_id(path: &Path) -> Option<(u64, u64)> {
    use std::os::unix::fs::MetadataExt;
    let meta = std::fs::metadata(path).ok()?;
    Some((meta.dev(), meta.ino()))
}

/// Bind `socket`, replacing it only when nothing is accepting connections.
#[cfg(unix)]
fn bind_daemon_socket(socket: &Path) -> Result<(UnixListener, (u64, u64)), String> {
    if socket.exists() {
        match std::os::unix::net::UnixStream::connect(socket) {
            Ok(_) => {
                return Err(format!("daemon already running on {}", socket.display()));
            }
            Err(e)
                if e.kind() == std::io::ErrorKind::ConnectionRefused
                    || e.kind() == std::io::ErrorKind::NotFound =>
            {
                let _ = std::fs::remove_file(socket);
            }
            Err(e) => return Err(format!("socket {}: {e}", socket.display())),
        }
    }
    let listener =
        UnixListener::bind(socket).map_err(|e| format!("bind {}: {e}", socket.display()))?;
    let id = socket_id(socket)
        .ok_or_else(|| format!("socket {}: bound but cannot stat", socket.display()))?;
    Ok((listener, id))
}

/// Unlink the socket only when it is still the one this daemon bound.
#[cfg(unix)]
fn remove_owned_socket(socket: &Path, owned: (u64, u64)) {
    if socket_id(socket) == Some(owned) {
        let _ = std::fs::remove_file(socket);
    }
}

/// Non-unix stub: the daemon speaks over a Unix socket.
#[cfg(not(unix))]
#[allow(clippy::too_many_arguments)]
async fn serve(
    _models: Vec<(String, PathBuf)>,
    _placement_base: Placement,
    _mode: String,
    _config: LoadConfig,
    _default_id: String,
    _max_loaded: usize,
    _socket: PathBuf,
    _max_load_percent: Option<u8>,
) -> Result<(), String> {
    Err("runa daemon needs a Unix socket (not supported on Windows)".into())
}

#[cfg(unix)]
async fn serve_conn(
    stream: UnixStream,
    pool: &Arc<Mutex<ModelPool>>,
    mm: &Arc<MemoryManager>,
) -> Result<(), String> {
    let (reader, mut writer) = stream.into_split();
    let mut reader = tokio::io::BufReader::new(reader);
    while let Some(line) = read_line(&mut reader).await.map_err(|e| e.to_string())? {
        if line.trim().is_empty() {
            continue;
        }
        for ev in serve_request(pool, mm, &line).await {
            let encoded = encode_line(&ev)?;
            write_line(&mut writer, &encoded)
                .await
                .map_err(|e| e.to_string())?;
            if ev.is_terminal() {
                break;
            }
        }
    }
    Ok(())
}

/// One NDJSON request line → the full event stream (always terminal).
/// Unix-only: the serving path (kept compiling everywhere so unit tests
/// stay portable).
#[cfg_attr(not(unix), allow(dead_code))]
async fn serve_request(
    pool: &Arc<Mutex<ModelPool>>,
    mm: &Arc<MemoryManager>,
    line: &str,
) -> Vec<DaemonEvent> {
    let started = std::time::Instant::now();
    let req: DaemonRequest = match decode_line(line) {
        Ok(r) => r,
        Err(e) => {
            return vec![DaemonEvent::Error {
                message: format!("bad request: {e}"),
            }];
        }
    };
    if req.v != PROTOCOL_VERSION {
        return vec![DaemonEvent::Error {
            message: format!(
                "protocol v{} != daemon v{PROTOCOL_VERSION}; upgrade runa",
                req.v
            ),
        }];
    }
    let request = match req.request.into_generate() {
        Ok(g) => g,
        Err(e) => {
            return vec![DaemonEvent::Error {
                message: format!("bad request: {e}"),
            }];
        }
    };
    // Per-request preflight (P9.1): record activity, then bounded pre-grow.
    // `on_heavy` keeps current placement when over ceiling instead of
    // refusing the request — admission is pool/LRU bound.
    mm.touch();
    mm.on_heavy(REQUEST_DEMAND_MIB);
    let id = match crate::pool::resolve_or_insert(pool, Some(&req.model)) {
        Ok(id) => id,
        Err(e) => {
            return vec![DaemonEvent::Error { message: e }];
        }
    };
    match crate::pool::generate(pool, &id, request).await {
        Ok(events) => {
            let mut out: Vec<DaemonEvent> = events
                .iter()
                .filter_map(DaemonEvent::from_gen_event)
                .collect();
            if !out.iter().any(|e| e.is_terminal()) {
                out.push(DaemonEvent::Done {
                    stop: "max_tokens".into(),
                });
            }
            eprintln!(
                "daemon: served {id} ({} events, {:.1}s)",
                out.len(),
                started.elapsed().as_secs_f64()
            );
            out
        }
        Err(e) => {
            eprintln!("daemon: error for {id}: {e}");
            let message = if e.contains("context_length_exceeded") {
                format!("bad request: {e}")
            } else {
                e
            };
            vec![DaemonEvent::Error { message }]
        }
    }
}

// --- supervision units (std only) ------------------------------------------

fn home_dir() -> Result<PathBuf, String> {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .filter(|p| !p.as_os_str().is_empty())
        .ok_or_else(|| "HOME is not set".to_string())
}

/// `~/Library/LaunchAgents/ai.runa.daemon.plist`.
pub fn launchd_path(home: &Path) -> PathBuf {
    home.join("Library")
        .join("LaunchAgents")
        .join("ai.runa.daemon.plist")
}

/// `~/.config/systemd/user/runa-daemon.service`.
pub fn systemd_path(home: &Path) -> PathBuf {
    home.join(".config")
        .join("systemd")
        .join("user")
        .join("runa-daemon.service")
}

/// launchd plist: run `exe argv…` at load, keep alive, log to
/// `~/.cache/runa/daemon.{out,err}.log`.
pub fn launchd_plist(exe: &Path, argv: &[String]) -> String {
    let mut args = format!(
        "    <string>{}</string>",
        xml_escape(&exe.display().to_string())
    );
    for a in argv {
        args.push_str(&format!("\n    <string>{}</string>", xml_escape(a)));
    }
    format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>ai.runa.daemon</string>
  <key>ProgramArguments</key>
  <array>
{args}
  </array>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>StandardOutPath</key>
  <string>~/.cache/runa/daemon.out.log</string>
  <key>StandardErrorPath</key>
  <string>~/.cache/runa/daemon.err.log</string>
</dict>
</plist>
"#
    )
}

/// systemd user unit: `exe argv…`, restart on failure, WantedBy default.
pub fn systemd_unit(exe: &Path, argv: &[String]) -> String {
    let mut cmd = shell_escape(&exe.display().to_string());
    for a in argv {
        cmd.push(' ');
        cmd.push_str(&shell_escape(a));
    }
    format!(
        r#"[Unit]
Description=runa daemon (warm models for run/chat)
After=network-online.target

[Service]
ExecStart={cmd}
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
"#
    )
}

fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

fn shell_escape(s: &str) -> String {
    if s.chars()
        .all(|c| c.is_alphanumeric() || "/._-:@".contains(c))
    {
        return s.to_owned();
    }
    format!("'{}'", s.replace('\'', "'\\''"))
}

/// Write both supervision units under `home`; returns the paths written.
pub fn install_daemon(home: &Path, exe: &Path, argv: &[String]) -> Result<Vec<PathBuf>, String> {
    let plist = launchd_path(home);
    let unit = systemd_path(home);
    if let Some(parent) = plist.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    if let Some(parent) = unit.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    std::fs::write(&plist, launchd_plist(exe, argv)).map_err(|e| e.to_string())?;
    std::fs::write(&unit, systemd_unit(exe, argv)).map_err(|e| e.to_string())?;
    Ok(vec![plist, unit])
}

/// Remove both supervision units under `home`; returns the paths removed.
pub fn uninstall_daemon(home: &Path) -> Result<Vec<PathBuf>, String> {
    let mut removed = Vec::new();
    for path in [launchd_path(home), systemd_path(home)] {
        match std::fs::remove_file(&path) {
            Ok(()) => removed.push(path),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(e.to_string()),
        }
    }
    Ok(removed)
}

/// Argv baked into installed units: `daemon <model> [--models …]
/// [--mode …] [--ctx …] …`, mirroring the flags the daemon accepts.
#[allow(clippy::too_many_arguments)]
pub(crate) fn daemon_argv(
    first: Option<&str>,
    rest: &[String],
    mode: &str,
    ctx: u32,
    max_loaded: Option<usize>,
    loras: &[String],
    socket: Option<&Path>,
    max_load_percent: Option<u8>,
) -> Vec<String> {
    let mut argv = vec!["daemon".to_owned()];
    if let Some(m) = first {
        argv.push(m.to_owned());
    }
    if !rest.is_empty() {
        argv.push("--models".to_owned());
        argv.push(rest.join(","));
    }
    argv.push("--mode".to_owned());
    argv.push(mode.to_owned());
    argv.push("--ctx".to_owned());
    argv.push(ctx.to_string());
    if let Some(n) = max_loaded {
        argv.push("--max-loaded".to_owned());
        argv.push(n.to_string());
    }
    for l in loras {
        argv.push("--lora".to_owned());
        argv.push(l.clone());
    }
    if let Some(s) = socket {
        argv.push("--socket".to_owned());
        argv.push(s.display().to_string());
    }
    if let Some(n) = max_load_percent {
        argv.push("--max-load-percent".to_owned());
        argv.push(n.to_string());
    }
    argv
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::daemon_proto::ProtoGenerateRequest;

    #[test]
    fn units_render_exe_and_args() {
        let exe = Path::new("/opt/runa/runa");
        let argv = daemon_argv(
            Some("qwen.gguf"),
            &["extra.gguf".to_owned()],
            "cpu",
            4096,
            Some(2),
            &["a.gguf:0.5".to_owned()],
            None,
            None,
        );
        assert_eq!(
            argv,
            [
                "daemon",
                "qwen.gguf",
                "--models",
                "extra.gguf",
                "--mode",
                "cpu",
                "--ctx",
                "4096",
                "--max-loaded",
                "2",
                "--lora",
                "a.gguf:0.5"
            ]
        );
        let plist = launchd_plist(exe, &argv);
        assert!(plist.contains("ai.runa.daemon"), "{plist}");
        assert!(plist.contains("<string>/opt/runa/runa</string>"), "{plist}");
        assert!(plist.contains("<string>qwen.gguf</string>"), "{plist}");
        assert!(plist.contains("KeepAlive"), "{plist}");
        let unit = systemd_unit(exe, &argv);
        assert!(
            unit.contains("ExecStart=/opt/runa/runa daemon qwen.gguf"),
            "{unit}"
        );
        assert!(unit.contains("Restart=on-failure"), "{unit}");
        assert!(unit.contains("WantedBy=default.target"), "{unit}");
    }

    #[test]
    fn shell_escape_quotes_spaces() {
        assert_eq!(shell_escape("/a/b"), "/a/b");
        assert_eq!(shell_escape("a b"), "'a b'");
        let unit = systemd_unit(Path::new("/a b/runa"), &["daemon".to_owned()]);
        assert!(unit.contains("ExecStart='/a b/runa' daemon"), "{unit}");
        assert_eq!(xml_escape("a&<b>"), "a&amp;&lt;b&gt;");
    }

    #[test]
    fn install_uninstall_roundtrip() {
        let home = std::env::temp_dir().join(format!(
            "runa-daemon-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let exe = Path::new("/bin/runa");
        let argv = vec!["daemon".to_owned(), "m.gguf".to_owned()];
        let paths = install_daemon(&home, exe, &argv).unwrap();
        assert_eq!(paths.len(), 2);
        for p in &paths {
            let text = std::fs::read_to_string(p).unwrap();
            assert!(text.contains("m.gguf"), "{p:?}");
        }
        // Install twice: idempotent overwrite.
        install_daemon(&home, exe, &argv).unwrap();
        let removed = uninstall_daemon(&home).unwrap();
        assert_eq!(removed.len(), 2);
        assert!(uninstall_daemon(&home).unwrap().is_empty());
        let _ = std::fs::remove_dir_all(&home);
    }

    #[test]
    fn bad_ndjson_line_becomes_error_event() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let pool = Arc::new(Mutex::new(
                ModelPool::new(
                    vec![],
                    BackendKind::Auto,
                    Placement::cpu(),
                    "cpu".into(),
                    LoadConfig::default(),
                    1,
                )
                .unwrap(),
            ));
            let policy = runa_memory::MemoryPolicy::default();
            let mm = Arc::new(MemoryManager::new(
                policy,
                4096,
                Box::new(runa_memory::FakeBackend::new(0, 0)),
            ));
            let events = serve_request(&pool, &mm, "not json").await;
            assert_eq!(events.len(), 1);
            assert!(matches!(events[0], DaemonEvent::Error { .. }));
            let stale = encode_line(&DaemonRequest {
                v: PROTOCOL_VERSION + 99,
                model: "m".into(),
                request: ProtoGenerateRequest::from_generate(
                    &runa_engine::GenerateRequest::default(),
                ),
            })
            .unwrap();
            let events = serve_request(&pool, &mm, &stale).await;
            assert!(matches!(events[0], DaemonEvent::Error { .. }));
        });
    }

    #[cfg(unix)]
    #[test]
    fn second_daemon_does_not_unlink_a_live_socket() {
        let dir = std::env::temp_dir().join(format!(
            "runa-sock-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("d.sock");
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(async {
            let (listener, id) = bind_daemon_socket(&path).unwrap();
            let err = bind_daemon_socket(&path).unwrap_err();
            assert!(err.contains("already running"), "{err}");
            assert!(path.exists());
            drop(listener);
            let (listener, id2) = bind_daemon_socket(&path).unwrap();
            assert_ne!(id, id2);
            remove_owned_socket(&path, id);
            assert!(
                path.exists(),
                "a stale inode must not remove the new socket"
            );
            remove_owned_socket(&path, id2);
            assert!(!path.exists());
            drop(listener);
        });
        let _ = std::fs::remove_dir_all(&dir);
    }
}
