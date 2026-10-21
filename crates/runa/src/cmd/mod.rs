// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! Command dispatch: config bootstrap, one call per subcommand, exit code.

use clap::Parser;
use runa_engine::{parse_device_list, parse_rpc_list, parse_tensor_split};

use crate::cli::input::resolve_kv;
use crate::cli::{Cli, Commands, MediaAction, config};
use crate::{bench, daemon, fit, placer, pull, serve, stats_web};

mod doctor;
mod media;
mod models;
mod run;
mod stats;
mod tasks;

#[cfg(not(fuzzing))]
pub fn run() {
    let cli = Cli::parse();
    if let Err(e) = config::reject_inline_in_config_files() {
        eprintln!("runa: error: {e}");
        std::process::exit(2);
    }
    match config::load_store() {
        Ok(store) => pull::init_store(store),
        Err(e) => {
            eprintln!("runa: error: {e}");
            std::process::exit(2);
        }
    }
    let rc = match cli.command {
        Commands::Run(args) => run::cmd_run(&args),
        Commands::Fit(args) => fit::cmd_fit(&args),
        Commands::Chat {
            model,
            backend,
            mode,
            ctx,
            threads,
            max_load_percent,
            lora,
            think,
            think_budget,
            effort,
            show_reasoning,
            no_show_reasoning,
            tui,
            no_daemon,
            rpc,
            tools,
        } => crate::session::cmd_chat(
            model.as_deref(),
            &backend,
            &mode,
            ctx,
            threads,
            max_load_percent,
            lora,
            think,
            think_budget,
            effort,
            show_reasoning,
            no_show_reasoning,
            tui,
            no_daemon,
            rpc,
            &tools,
        ),
        Commands::Pull { model } => models::cmd_pull(&model),
        Commands::Models {} => models::cmd_models(),
        Commands::Bench {
            model,
            mode,
            ctx,
            pp,
            tg,
            json,
            no_calibrate,
            kv,
            kv_k,
            kv_v,
            device,
            tensor_split,
            threads,
            max_load_percent,
        } => match resolve_kv(kv.as_deref(), kv_k.as_deref(), kv_v.as_deref()) {
            Ok((kv_k, kv_v)) => bench::cmd_bench(
                &model,
                &mode,
                ctx,
                pp,
                tg,
                json,
                no_calibrate,
                kv_k,
                kv_v,
                device.as_deref(),
                tensor_split.as_deref(),
                threads,
                max_load_percent,
            ),
            Err(e) => Err(e),
        },
        Commands::Media { action } => match action {
            MediaAction::Probe { path, json } => media::cmd_media_probe(&path, json),
            MediaAction::Video {
                path,
                fps,
                max_frames,
                json,
            } => media::cmd_media_video(&path, fps, max_frames, json),
            MediaAction::Transcribe {
                path,
                model,
                lang,
                json,
                no_pull,
            } => media::cmd_media_transcribe(&path, &model, &lang, json, no_pull),
        },
        Commands::Serve {
            model,
            models,
            backend,
            parallel,
            max_loaded,
            host,
            port,
            api_key,
            mode,
            ctx,
            lora,
            device,
            tensor_split,
            main_gpu,
            rpc,
            threads,
            max_load_percent,
        } => {
            let mut paths = Vec::new();
            if let Some(m) = model {
                paths.push(std::path::PathBuf::from(m));
            }
            for entry in models {
                for part in entry.split(',') {
                    let p = part.trim();
                    if !p.is_empty() {
                        paths.push(std::path::PathBuf::from(p));
                    }
                }
            }
            // Serve hosts several models: only the global `[model] lora`
            // plus `--lora` apply, to every model (P8.5).
            config::resolve_loras(&lora, "").and_then(|loras| {
                serve::model_specs_from_paths(paths).and_then(|models| {
                    let requested = runa_core::BackendKind::parse(&backend).ok_or_else(|| {
                        format!("{backend}: --backend must be gguf | mistral | auto")
                    })?;
                    runa_engine::ensure_backend_available(requested).map_err(|e| e.to_string())?;
                    // P2.9/P9.3: placement flags apply to gguf models only —
                    // the pool loads mistral models with `Placement::cpu()`.
                    // Parsed here so a typo fails fast at startup, before
                    // binding the port; applied in `placement_for` on top of
                    // both fixed and `auto` placements (never dropped).
                    let overrides = placer::PlacementOverrides {
                        devices: device
                            .as_deref()
                            .map(parse_device_list)
                            .transpose()?
                            .unwrap_or_default(),
                        tensor_split: tensor_split
                            .as_deref()
                            .map(parse_tensor_split)
                            .transpose()?
                            .unwrap_or_default(),
                        main_gpu,
                        rpc_servers: rpc
                            .as_deref()
                            .map(parse_rpc_list)
                            .transpose()?
                            .unwrap_or_default(),
                    };
                    serve::cmd_serve(serve::ServeOpts {
                        models,
                        backend: requested,
                        host,
                        port,
                        api_key,
                        mode,
                        ctx,
                        parallel,
                        max_loaded,
                        loras,
                        overrides,
                        threads: config::resolve_threads(threads)?,
                        max_load_percent,
                    })
                })
            })
        }
        Commands::Daemon {
            model,
            models,
            mode,
            ctx,
            max_loaded,
            max_load_percent,
            lora,
            socket,
            install,
            uninstall,
        } => {
            let mut paths = Vec::new();
            if let Some(m) = model {
                paths.push(std::path::PathBuf::from(m));
            }
            for entry in &models {
                for part in entry.split(',') {
                    let p = part.trim();
                    if !p.is_empty() {
                        paths.push(std::path::PathBuf::from(p));
                    }
                }
            }
            // Like serve: only the global `[model] lora` plus `--lora`
            // apply, to every model (P8.5).
            let first = paths.first().map(|p| p.display().to_string());
            let rest: Vec<String> = paths
                .iter()
                .skip(1)
                .map(|p| p.display().to_string())
                .collect();
            let argv = daemon::daemon_argv(
                first.as_deref(),
                &rest,
                &mode,
                ctx,
                max_loaded,
                &lora,
                socket.as_deref(),
                max_load_percent,
            );
            config::resolve_loras(&lora, "").and_then(|loras| {
                serve::model_specs_from_paths(paths).and_then(|models| {
                    daemon::cmd_daemon(daemon::DaemonOpts {
                        models,
                        mode,
                        ctx,
                        max_loaded,
                        max_load_percent,
                        loras,
                        socket,
                        install,
                        uninstall,
                        argv,
                    })
                })
            })
        }
        Commands::Doctor { json } => {
            doctor::doctor(json);
            Ok(())
        }
        Commands::Tasks { action } => tasks::cmd_tasks(action),
        Commands::Stats { since, json } => stats::cmd_stats(since.as_deref(), json),
        Commands::Web { host, port } => stats_web::cmd_web(&host, port),
    };
    if let Err(e) = rc {
        let (code, msg) = if let Some(msg) = e.strip_prefix("unfit: ") {
            (2, msg)
        } else {
            (1, e.as_str())
        };
        eprintln!("runa: error: {msg}");
        std::process::exit(code);
    }
}
