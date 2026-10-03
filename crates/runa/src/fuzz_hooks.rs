//! libFuzzer entry points, compiled only by `cargo fuzz` (`--cfg fuzzing`).
//!
//! `fuzz/Cargo.toml` builds this binary crate once per target (`cli-argv`,
//! `cli-config`, `serve-body`, `daemon-proto`); the bin name picks the
//! surface. Everything here is in-process and side-effect free: no model
//! load, network, process spawn, or file writes. See fuzz/README.md.

use std::ffi::OsString;

use clap::Parser;

use crate::{Cli, Commands, ToolArgs};

pub(crate) fn run(data: &[u8]) {
    match env!("CARGO_BIN_NAME") {
        "cli-argv" => cli_argv(data),
        "cli-config" => cli_config(data),
        "serve-body" => crate::serve::fuzz_request_body(data),
        "daemon-proto" => daemon_proto(data),
        other => panic!("fuzz_hooks: unknown fuzz target {other}"),
    }
}

/// NUL-separated argv (raw bytes, so non-UTF-8 arguments too) through the
/// real clap parser, then the pure flag validators each command runs
/// before loading anything.
fn cli_argv(data: &[u8]) {
    let mut argv = vec![OsString::from("runa")];
    argv.extend(data.split(|&b| b == 0).map(os_string));
    let cli = match Cli::try_parse_from(argv) {
        Ok(cli) => cli,
        Err(e) => {
            // Help / version / usage errors render the whole command tree.
            let _ = e.render().to_string();
            return;
        }
    };
    match cli.command {
        Commands::Run(a) => {
            flags(
                &a.backend,
                &a.mode,
                a.device.as_deref(),
                a.rpc.as_deref(),
                &a.lora,
            );
            let _ = crate::resolve_kv(a.kv.as_deref(), a.kv_k.as_deref(), a.kv_v.as_deref());
            if let Some(s) = a.tensor_split.as_deref() {
                let _ = runa_engine::parse_tensor_split(s);
            }
            if let Some(s) = a.on_unfit.as_deref() {
                let _ = crate::config::OnUnfit::parse(s);
            }
            think(a.think.as_deref(), a.effort.as_deref());
            if let Some(s) = a.audio_route.as_deref() {
                let _ = runa_media::AudioRoutePref::parse(s);
            }
            model_ref(&a.model);
            tools(&a.tools);
        }
        Commands::Chat {
            model,
            backend,
            mode,
            lora,
            think: t,
            effort,
            rpc,
            tools: tool_args,
            ..
        } => {
            flags(&backend, &mode, None, rpc.as_deref(), &lora);
            think(t.as_deref(), effort.as_deref());
            if let Some(m) = model {
                model_ref(&m);
            }
            tools(&tool_args);
        }
        Commands::Bench {
            model,
            mode,
            kv,
            kv_k,
            kv_v,
            device,
            tensor_split,
            ..
        } => {
            flags("auto", &mode, device.as_deref(), None, &[]);
            let _ = crate::resolve_kv(kv.as_deref(), kv_k.as_deref(), kv_v.as_deref());
            if let Some(s) = tensor_split.as_deref() {
                let _ = runa_engine::parse_tensor_split(s);
            }
            model_ref(&model);
        }
        Commands::Serve {
            model,
            models,
            backend,
            mode,
            lora,
            device,
            tensor_split,
            rpc,
            ..
        } => {
            flags(&backend, &mode, device.as_deref(), rpc.as_deref(), &lora);
            if let Some(s) = tensor_split.as_deref() {
                let _ = runa_engine::parse_tensor_split(s);
            }
            for m in model.iter().chain(&models) {
                model_ref(m);
            }
        }
        Commands::Daemon {
            model,
            models,
            mode,
            lora,
            ..
        } => {
            flags("auto", &mode, None, None, &lora);
            for m in model.iter().chain(&models) {
                model_ref(m);
            }
        }
        Commands::Pull { model } => model_ref(&model),
        _ => {}
    }
}

fn flags(backend: &str, mode: &str, device: Option<&str>, rpc: Option<&str>, lora: &[String]) {
    let _ = runa_core::BackendKind::parse(backend);
    let _ = crate::parse_mode_choice(mode);
    if let Some(s) = device {
        let _ = runa_engine::parse_device_list(s);
    }
    if let Some(s) = rpc {
        let _ = runa_engine::parse_rpc_list(s);
    }
    for s in lora {
        let _ = runa_engine::parse_lora_spec(s);
    }
}

fn think(think: Option<&str>, effort: Option<&str>) {
    if let Some(s) = think {
        let _ = runa_core::ThinkOverrides::parse_think(s);
    }
    if let Some(s) = effort {
        let _ = runa_core::Effort::parse(s);
    }
}

fn model_ref(s: &str) {
    let _ = runa_cloud::parse_cloud_ref(s);
    let _ = runa_fit::parse_model_ref(s);
}

fn tools(t: &ToolArgs) {
    for spec in &t.mcp {
        let _ = crate::mcp::McpServer::from_flag(spec);
    }
}

/// A config file's text (`runa.toml`), plus the REPL / TUI slash-command
/// and `--mcp` quoting parsers on the same string.
fn cli_config(data: &[u8]) {
    let Ok(text) = std::str::from_utf8(data) else {
        return;
    };
    crate::config::fuzz_config_text(text);
    let _ = crate::tui::parse_slash(text);
    let _ = crate::mcp::McpServer::from_flag(text);
}

/// Client ↔ daemon NDJSON lines over the local socket.
fn daemon_proto(data: &[u8]) {
    use crate::daemon_proto::{DaemonEvent, DaemonRequest, decode_line, encode_line, parse_stop};
    let Ok(line) = std::str::from_utf8(data) else {
        return;
    };
    if let Ok(req) = decode_line::<DaemonRequest>(line) {
        let _ = encode_line(&req);
        let _ = req.request.into_generate();
    }
    if let Ok(ev) = decode_line::<DaemonEvent>(line) {
        let _ = ev.is_terminal();
        let _ = encode_line(&ev);
    }
    let _ = parse_stop(line);
}

#[cfg(unix)]
fn os_string(bytes: &[u8]) -> OsString {
    use std::os::unix::ffi::OsStrExt;
    std::ffi::OsStr::from_bytes(bytes).to_owned()
}

#[cfg(not(unix))]
fn os_string(bytes: &[u8]) -> OsString {
    OsString::from(String::from_utf8_lossy(bytes).into_owned())
}
