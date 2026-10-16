// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! Command parsing and the subcommands other than the chat session.

use std::fs;
use std::io::{self, IsTerminal, Read, Write};
use std::path::{Path, PathBuf};

use clap::{Parser, Subcommand};
use runa_core::{BackendKind, ThinkConfig, ThinkOverrides, parse_budget};
use runa_engine::{
    ChatMessage, GenerateRequest, KvKind, LoadConfig, LoraSpec, Mode, Placement, PromptCache,
    SamplingConfig, StopReason, ToolCall, Usage, VisionFrame, VisionSource, parse_device_list,
    parse_rpc_list, parse_tensor_split, planner_kv_type,
};
use runa_fit::{
    Descriptor, NpuKind, Reader, check_fit, estimate_compute, estimate_kv, estimate_speed_single,
    npu_present, read_local_prefix,
};
use runa_memory::{ClaimError, MemoryManager, SysinfoBackend, TaskRegistry};

pub(crate) mod config;

use crate::bench;
use crate::cloud;
use crate::daemon;
use crate::daemon_proto;
use crate::fit;
use crate::mcp;
use crate::placer;
use crate::pull;
use crate::serve;
use crate::stats_web;
use crate::usage_log;
use runa_pool::engine;

/// Run AI models locally or through the OpenAI and Anthropic APIs.
#[derive(Debug, Parser)]
#[command(name = "runa", version, about, long_about = None, arg_required_else_help = true)]
pub(crate) struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
// CLI subcommands are inherently size-diverse and parsed once per process;
// boxing the large variants would ripple through every clap match site.
#[allow(clippy::large_enum_variant)]
pub(crate) enum Commands {
    /// One-shot generation: `runa run <model> [prompt]`.
    Run(RunArgs),
    /// Will a model run here, and how fast? No download (`--recommend` ranks a catalog).
    Fit(fit::FitArgs),
    /// Interactive chat (history, `/think`, `/mode`, `/model`, `\` continuation).
    Chat {
        /// Local model path: `.gguf` file or mistral safetensors directory.
        model: Option<String>,
        /// Local backend: gguf | mistral | auto (default: auto-detect from the path).
        #[arg(long, default_value = "auto")]
        backend: String,
        /// Compute mode (default: cpu).
        #[arg(long, default_value = "cpu")]
        mode: String,
        /// Context length.
        #[arg(long, default_value_t = 8192)]
        ctx: u32,
        /// Worker threads (default: P-cores on macOS, else all logical
        /// CPUs). GGUF backend only (P10.2).
        #[arg(long, value_name = "N")]
        threads: Option<i32>,
        /// Max share of total system resources (RAM budget + CPU thread
        /// share) this chat may use, percent 1..=100 (default: 80).
        #[arg(long, value_name = "N")]
        max_load_percent: Option<u8>,
        /// LoRA adapter GGUF (`path[:scale]`). Repeatable (P8.5).
        #[arg(long, value_name = "PATH[:SCALE]")]
        lora: Vec<String>,
        /// Thinking: on | off (P3.1).
        #[arg(long, value_name = "on|off")]
        think: Option<String>,
        /// Reasoning token budget.
        #[arg(long, value_name = "N")]
        think_budget: Option<u32>,
        /// Effort level: low | medium | high | max.
        #[arg(long, value_name = "LEVEL")]
        effort: Option<String>,
        /// Print reasoning.
        #[arg(long, default_value_t = false)]
        show_reasoning: bool,
        /// Hide reasoning even if config enables it.
        #[arg(long, default_value_t = false)]
        no_show_reasoning: bool,
        /// Full-screen TUI (transcript, reasoning toggle, status bar).
        #[arg(long, default_value_t = false)]
        tui: bool,
        /// Skip the daemon (`~/.cache/runa/runa.sock`) and load the model
        /// in-process, even when `runa daemon` is running (P9.1).
        #[arg(long, default_value_t = false)]
        no_daemon: bool,
        /// llama.cpp RPC endpoints (`host:port,…`); registers only — select
        /// with `--device RPC0` (needs the `rpc` cargo feature, else an
        /// explicit error, P9.3).
        #[arg(long, value_name = "LIST")]
        rpc: Option<String>,
        #[command(flatten)]
        tools: ToolArgs,
    },
    /// Download a model: `runa pull hf:<repo>:<file-or-quant>` (P2.4).
    Pull {
        /// Model reference (`hf:<repo>:<file.gguf>` or `hf:<repo>:<quant>`).
        model: String,
    },
    /// List downloaded models + configured aliases (P2.4).
    Models {},
    /// Prefill/decode throughput like llama-bench (pp512/tg128).
    Bench {
        /// Local GGUF file.
        model: String,
        /// Compute mode: cpu | gpu | hybrid | auto (default: gpu).
        #[arg(long, default_value = "gpu")]
        mode: String,
        /// Context length (must fit pp + tg + 1).
        #[arg(long, default_value_t = 8192)]
        ctx: u32,
        /// Prompt-processing tokens (llama-bench `-p`).
        #[arg(long, default_value_t = 512)]
        pp: u32,
        /// Generated tokens (llama-bench `-n`).
        #[arg(long, default_value_t = 128)]
        tg: u32,
        /// Emit one JSON object instead of the human summary.
        #[arg(long, default_value_t = false)]
        json: bool,
        /// Do not append this run to the calibration DB.
        #[arg(long, default_value_t = false)]
        no_calibrate: bool,
        /// KV cache type for both K and V (`f16`, `q8_0`, `q4_0`).
        #[arg(long, value_name = "TYPE")]
        kv: Option<String>,
        /// KV type for K only (overrides `--kv`).
        #[arg(long, value_name = "TYPE")]
        kv_k: Option<String>,
        /// KV type for V only (overrides `--kv`).
        #[arg(long, value_name = "TYPE")]
        kv_v: Option<String>,
        /// ggml backends to use (`0,1` or `CUDA0,CUDA1`).
        #[arg(long, value_name = "LIST")]
        device: Option<String>,
        /// Per-GPU proportions (`3,1`). Requires multiple GPUs.
        #[arg(long, value_name = "LIST")]
        tensor_split: Option<String>,
        /// Worker threads (default: P-cores on macOS, else all logical CPUs).
        /// GGUF backend only (P10.2).
        #[arg(long, value_name = "N")]
        threads: Option<i32>,
        /// Max share of total system resources (RAM budget + CPU thread
        /// share) this bench may use, percent 1..=100 (default: 80).
        #[arg(long, value_name = "N")]
        max_load_percent: Option<u8>,
    },
    /// OpenAI-compatible HTTP server (P3.9 / P6.1).
    Serve {
        /// Local model to load: `.gguf` file or mistral safetensors directory.
        model: Option<String>,
        /// Additional model paths (comma-separated or repeatable).
        #[arg(long, value_delimiter = ',')]
        models: Vec<String>,
        /// Local backend: gguf | mistral | auto (default: auto-detect per model).
        #[arg(long, default_value = "auto")]
        backend: String,
        /// Max in-flight HTTP generations (queued beyond this).
        #[arg(long, default_value_t = 1)]
        parallel: usize,
        /// Max models kept loaded (LRU unloads the rest).
        #[arg(long)]
        max_loaded: Option<usize>,
        /// Bind address (D11: 127.0.0.1 by default).
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        /// Bind port. `0` picks an ephemeral port.
        #[arg(long, default_value_t = 8080)]
        port: u16,
        /// Bearer token required on `/v1`. Required when `--host` is not
        /// a loopback address (`127.0.0.0/8` or `::1`).
        #[arg(long, value_name = "KEY")]
        api_key: Option<String>,
        /// Compute mode: cpu | gpu | hybrid | auto.
        #[arg(long, default_value = "cpu")]
        mode: String,
        /// Context length.
        #[arg(long, default_value_t = 4096)]
        ctx: u32,
        /// LoRA adapter GGUF (`path[:scale]`). Repeatable; applies to every
        /// served model (P8.5).
        #[arg(long, value_name = "PATH[:SCALE]")]
        lora: Vec<String>,
        /// ggml backends to use (`0,1` or `CUDA0,CUDA1`, gguf models only).
        #[arg(long, value_name = "LIST")]
        device: Option<String>,
        /// Per-GPU proportions (`3,1`, gguf models only).
        #[arg(long, value_name = "LIST")]
        tensor_split: Option<String>,
        /// Main GPU for scratch/small tensors (gguf models only).
        #[arg(long, value_name = "N")]
        main_gpu: Option<i32>,
        /// llama.cpp RPC endpoints (`host:port,…`); registers only — select
        /// with `--device RPC0` (needs the `rpc` cargo feature, else an
        /// explicit error, P9.3).
        #[arg(long, value_name = "LIST")]
        rpc: Option<String>,
        /// Worker threads (default: P-cores on macOS, else all logical CPUs).
        /// GGUF backend only (P10.2).
        #[arg(long, value_name = "N")]
        threads: Option<i32>,
        /// Max share of total system resources (RAM budget + CPU thread
        /// share) the server may use, percent 1..=100 (default: 80).
        #[arg(long, value_name = "N")]
        max_load_percent: Option<u8>,
    },
    /// Background service: warm models + Unix-socket server for `run`/`chat` (P9.1).
    Daemon {
        /// Local GGUF file to keep warm (single-model shorthand).
        model: Option<String>,
        /// Additional GGUF paths (comma-separated or repeatable).
        #[arg(long, value_delimiter = ',')]
        models: Vec<String>,
        /// Compute mode: cpu | gpu | hybrid | auto.
        #[arg(long, default_value = "cpu")]
        mode: String,
        /// Context length.
        #[arg(long, default_value_t = 4096)]
        ctx: u32,
        /// Max models kept loaded (LRU unloads the rest).
        #[arg(long)]
        max_loaded: Option<usize>,
        /// Max share of total system resources (RAM budget + CPU thread
        /// share) the daemon may use, percent 1..=100 (default: 80).
        #[arg(long, value_name = "N")]
        max_load_percent: Option<u8>,
        /// LoRA adapter GGUF (`path[:scale]`). Repeatable; applies to every
        /// daemon model.
        #[arg(long, value_name = "PATH[:SCALE]")]
        lora: Vec<String>,
        /// Socket path (default: `~/.cache/runa/runa.sock`).
        #[arg(long)]
        socket: Option<PathBuf>,
        /// Write launchd/systemd units (needs a model) and exit.
        #[arg(long, default_value_t = false, conflicts_with = "uninstall")]
        install: bool,
        /// Remove launchd/systemd units and exit.
        #[arg(long, default_value_t = false)]
        uninstall: bool,
    },
    /// Report compiled-in backends and native-build flags (P6.3).
    Doctor {
        /// Emit machine-readable JSON (`backends`, `native_build`).
        #[arg(long)]
        json: bool,
    },
    /// Decode / inspect media files (P4.1).
    Media {
        #[command(subcommand)]
        action: MediaAction,
    },
    /// Plan task claims (`docs/tasks.md`, P7.4).
    Tasks {
        #[command(subcommand)]
        action: TaskAction,
    },
    /// Usage totals from the local journal (`run`, `chat`, `serve`).
    Stats {
        /// How far back to read (`24h`, `7d`, `60d`).
        #[arg(long, value_name = "SPAN")]
        since: Option<String>,
        /// JSON instead of the table.
        #[arg(long, default_value_t = false)]
        json: bool,
    },
    /// Live usage page on this machine (websocket, loopback only).
    Web {
        /// Bind address. Must be a loopback address.
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        /// Bind port. `0` picks an ephemeral port.
        #[arg(long, default_value_t = 3939)]
        port: u16,
    },
}

#[derive(Debug, Subcommand)]
pub(crate) enum MediaAction {
    /// Decode audio to f32 mono 16 kHz and print codec / PCM hash.
    Probe {
        /// Audio file (wav, flac, ogg, mp3, aac).
        path: PathBuf,
        /// Machine-readable JSON (`AudioProbe`).
        #[arg(long, default_value_t = false)]
        json: bool,
    },
    /// Sample video frames (P4.5): fps, max-frames, scene-change.
    Video {
        path: PathBuf,
        #[arg(long, default_value_t = 1.0)]
        fps: f32,
        #[arg(long, default_value_t = 32)]
        max_frames: usize,
        #[arg(long, default_value_t = false)]
        json: bool,
    },
    /// Transcribe audio via whisper.cpp (P4.2). Auto-pulls ggml-base / turbo.
    Transcribe {
        /// Audio file (wav, flac, ogg, mp3, aac).
        path: PathBuf,
        /// `base` or `large-v3-turbo`.
        #[arg(long, default_value = "base")]
        model: String,
        /// ISO-639-1 or `auto`.
        #[arg(long, default_value = "auto")]
        lang: String,
        #[arg(long, default_value_t = false)]
        json: bool,
        /// Do not download missing ggml files.
        #[arg(long, default_value_t = false)]
        no_pull: bool,
    },
}

#[derive(Debug, Subcommand)]
pub(crate) enum TaskAction {
    /// List `free` task IDs.
    List {},
    /// Claim a task: `runa tasks claim P7.1 --agent <name>`.
    Claim {
        task_id: String,
        #[arg(long)]
        agent: String,
    },
    /// Release a claim.
    Release {
        task_id: String,
        #[arg(long)]
        agent: String,
    },
}

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
        Commands::Run(args) => cmd_run(&args),
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
        Commands::Pull { model } => cmd_pull(&model),
        Commands::Models {} => cmd_models(),
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
            MediaAction::Probe { path, json } => cmd_media_probe(&path, json),
            MediaAction::Video {
                path,
                fps,
                max_frames,
                json,
            } => cmd_media_video(&path, fps, max_frames, json),
            MediaAction::Transcribe {
                path,
                model,
                lang,
                json,
                no_pull,
            } => cmd_media_transcribe(&path, &model, &lang, json, no_pull),
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
            doctor(json);
            Ok(())
        }
        Commands::Tasks { action } => cmd_tasks(action),
        Commands::Stats { since, json } => cmd_stats(since.as_deref(), json),
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

#[derive(Debug, clap::Args)]
pub(crate) struct RunArgs {
    /// Local model path: `.gguf` file or mistral safetensors directory.
    model: String,
    /// Local backend: gguf | mistral | auto (default: auto-detect from the path).
    /// The mistral backend maps sampling / think / stop / usage; `--mode`,
    /// `--ctx` and `--seed` are gguf-only and ignored there (see docs/memory.md P9.2).
    #[arg(long, default_value = "auto")]
    backend: String,
    /// Prompt text (else read from stdin when piped).
    prompt: Option<String>,
    /// Compute mode: cpu | gpu | hybrid | auto (default: auto).
    #[arg(long, default_value = "auto")]
    mode: String,
    /// Context length.
    #[arg(long, default_value_t = 8192)]
    ctx: u32,
    /// Max new tokens.
    #[arg(long, default_value_t = 512)]
    max_tokens: u32,
    /// Sampling temperature (<= 0 = greedy).
    #[arg(long, default_value_t = 0.8)]
    temperature: f32,
    /// Sampler seed.
    #[arg(long, default_value_t = 42)]
    seed: u32,
    /// Worker threads (default: P-cores on macOS, else all logical CPUs).
    /// GGUF backend only (P10.2).
    #[arg(long, value_name = "N")]
    threads: Option<i32>,
    /// Max share of total system resources (RAM budget + CPU thread share)
    /// this run may use, percent 1..=100 (default: 80, `[system]`).
    #[arg(long, value_name = "N")]
    max_load_percent: Option<u8>,
    /// Emit one JSON object instead of streaming text.
    #[arg(long, default_value_t = false)]
    json: bool,
    /// When auto cannot fit: error | cpu | cloud:<backend>:<model> (D12).
    #[arg(long)]
    on_unfit: Option<String>,
    /// Keep MoE expert tensors of the first N layers on CPU (`--n-cpu-moe`).
    #[arg(long, value_name = "N")]
    n_cpu_moe: Option<u32>,
    /// LMDB dir for prompt KV cache (default: ~/.cache/runa/kv).
    #[arg(long, value_name = "DIR")]
    prompt_cache: Option<PathBuf>,
    /// Disable the LMDB prompt cache (P2.8).
    #[arg(long, default_value_t = false)]
    no_prompt_cache: bool,
    /// KV cache type for both K and V (`f16`, `q8_0`, `q4_0`). Requires flash-attn.
    #[arg(long, value_name = "TYPE")]
    kv: Option<String>,
    /// KV type for K only (overrides `--kv`).
    #[arg(long, value_name = "TYPE")]
    kv_k: Option<String>,
    /// KV type for V only (overrides `--kv`).
    #[arg(long, value_name = "TYPE")]
    kv_v: Option<String>,
    /// Thinking: on | off (P3.1). Budget/effort override `on`.
    #[arg(long, value_name = "on|off")]
    think: Option<String>,
    /// Reasoning token budget (`ThinkMode::Budget`).
    #[arg(long, value_name = "N")]
    think_budget: Option<u32>,
    /// Effort level: low | medium | high | max.
    #[arg(long, value_name = "LEVEL")]
    effort: Option<String>,
    /// Print reasoning (`Event::Reasoning` lands in P3.2).
    #[arg(long, default_value_t = false)]
    show_reasoning: bool,
    /// Hide reasoning even if config `[think] show = true`.
    #[arg(long, default_value_t = false)]
    no_show_reasoning: bool,
    /// ggml backends to use (`0,1` or `CUDA0,CUDA1`).
    #[arg(long, value_name = "LIST")]
    device: Option<String>,
    /// Per-GPU proportions (`3,1`). Requires multiple GPUs.
    #[arg(long, value_name = "LIST")]
    tensor_split: Option<String>,
    /// Main GPU for scratch/small tensors (`--main-gpu N`, P2.9).
    #[arg(long, value_name = "N")]
    main_gpu: Option<i32>,
    /// llama.cpp RPC endpoints (`host:port,…`); registers only — select
    /// with `--device RPC0` (needs the `rpc` cargo feature, else an
    /// explicit error, P9.3).
    #[arg(long, value_name = "LIST")]
    rpc: Option<String>,
    /// Audio file (PCM 16 kHz). Routed by `--audio-route`.
    #[arg(long, value_name = "PATH")]
    audio: Option<PathBuf>,
    /// Audio/vision mmproj GGUF. Sibling `*mmproj*.gguf` if `--audio` and omitted.
    #[arg(long, value_name = "PATH")]
    mmproj: Option<PathBuf>,
    /// Audio route: auto | native | asr (P4.4).
    #[arg(long, value_name = "auto|native|asr")]
    audio_route: Option<String>,
    /// Image file(s) for native mtmd vision (P4.6).
    #[arg(long, value_name = "PATH")]
    image: Vec<PathBuf>,
    /// Video file: sampled frames with `[t=12.0s]` markers (P4.6).
    #[arg(long, value_name = "PATH")]
    video: Option<PathBuf>,
    /// Trigram speculative decoding; greedy-verified (use --temperature 0).
    #[arg(long, default_value_t = false)]
    ngram: bool,
    /// Draft-model GGUF: counted in fit; speculation still uses n-gram.
    #[arg(long, value_name = "PATH")]
    draft: Option<PathBuf>,
    /// LoRA adapter GGUF (`path[:scale]`). Repeatable; adds to `[model]` /
    /// `[models]` `lora` in config, counted in fit (P8.5).
    #[arg(long, value_name = "PATH[:SCALE]")]
    lora: Vec<String>,
    /// Constrain the answer to a JSON Schema (file path or inline JSON).
    #[arg(long, value_name = "FILE|JSON", conflicts_with = "grammar")]
    json_schema: Option<String>,
    /// Constrain the answer to a GBNF grammar file.
    #[arg(long, value_name = "FILE")]
    grammar: Option<PathBuf>,
    /// Skip the daemon (`~/.cache/runa/runa.sock`) and load the model
    /// in-process, even when `runa daemon` is running (P9.1).
    #[arg(long, default_value_t = false)]
    no_daemon: bool,
    #[command(flatten)]
    tools: ToolArgs,
}

/// MCP tool loop flags shared by `run` and `chat` (P8.3).
#[derive(Debug, clap::Args)]
pub(crate) struct ToolArgs {
    /// Stdio MCP server whose tools the model may call: '<command args>'
    /// (shell quoting groups spaces, e.g. `--mcp "python3 'my dir/s.py'"`;
    /// repeatable; adds to `[mcp.servers]` in config).
    #[arg(long, value_name = "COMMAND")]
    pub(crate) mcp: Vec<String>,
    /// Stop after this many tool-call rounds without an answer.
    #[arg(long, value_name = "N", default_value_t = 8)]
    pub(crate) max_tool_rounds: u32,
}

/// Resolve a model reference: local path → alias → pull store (P2.4).
/// Never downloads (that is `pull`'s job — plan D12, no silent fetches).
pub(crate) fn resolve_model(model_ref: &str) -> Result<PathBuf, String> {
    let aliases = config::load_aliases()?;
    pull::find_local(model_ref, &aliases)
}

pub(crate) fn parse_mode(mode: &str) -> Result<Mode, String> {
    Mode::parse(mode).ok_or_else(|| format!("{mode}: --mode must be cpu | gpu | hybrid"))
}

pub(crate) enum ModeChoice {
    Auto,
    Fixed(Mode),
}

pub(crate) fn parse_mode_choice(mode: &str) -> Result<ModeChoice, String> {
    if mode.eq_ignore_ascii_case("auto") {
        return Ok(ModeChoice::Auto);
    }
    Mode::parse(mode)
        .map(ModeChoice::Fixed)
        .ok_or_else(|| format!("{mode}: --mode must be cpu | gpu | hybrid | auto"))
}

pub(crate) fn vram_bytes() -> Result<(u64, &'static str), String> {
    if let Ok(s) = std::env::var("RUNA_FAKE_VRAM") {
        let mib: u64 = s
            .parse()
            .map_err(|_| format!("RUNA_FAKE_VRAM={s}: expected integer MiB"))?;
        return Ok((mib.saturating_mul(1024 * 1024), "RUNA_FAKE_VRAM"));
    }
    #[cfg(target_os = "macos")]
    {
        let mut sys = sysinfo::System::new();
        sys.refresh_memory();
        // Unified memory stand-in until the P1.6 wired_limit probe is wired here.
        let assumed = sys.total_memory().saturating_mul(3) / 4;
        Ok((assumed, "macos-unified-75pct"))
    }
    #[cfg(not(target_os = "macos"))]
    {
        Ok((0, "unknown"))
    }
}

pub(crate) fn ram_bytes() -> u64 {
    // Test hook like RUNA_FAKE_VRAM: forces a CPU NO FIT on big hosts.
    if let Some(mib) = std::env::var("RUNA_FAKE_RAM")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
    {
        return mib.saturating_mul(1024 * 1024);
    }
    let mut sys = sysinfo::System::new();
    sys.refresh_memory();
    sys.total_memory()
}

fn bytes_to_mib(bytes: u64) -> u64 {
    bytes.div_ceil(1024 * 1024)
}

pub(crate) fn memory_ceiling_mib() -> u64 {
    if let Ok(s) = std::env::var("RUNA_MEMORY_CEILING_MIB")
        && let Ok(n) = s.parse::<u64>()
    {
        return n;
    }
    ram_bytes() / (1024 * 1024)
}

/// P7.3: refuse a run whose KV+compute demand exceeds the fit ceiling
/// and `max_growth_mib` before llama allocates arenas. Warns first when
/// the demand alone breaches the `[system] max_load_percent` cap.
pub(crate) fn preflight_grow(
    path: &Path,
    ctx: u32,
    kv_type: &str,
    extra_bytes: u64,
    max_load_percent: Option<u8>,
) -> Result<(), String> {
    let policy = config::resolve_memory_policy()?;
    let header = read_local_prefix(path).map_err(|e| e.to_string())?;
    let reader = Reader::parse(&header.bytes).map_err(|e| e.to_string())?;
    let desc = Descriptor::from_reader(&reader).map_err(|e| e.to_string())?;
    let kv = estimate_kv(&desc, u64::from(ctx), kv_type);
    let compute = estimate_compute(&desc, 512);
    let demand_bytes = kv
        .kv_bytes
        .saturating_add(compute.compute_bytes)
        .saturating_add(extra_bytes);
    config::warn_if_demand_over_limit(demand_bytes, max_load_percent)?;
    let demand = bytes_to_mib(demand_bytes);
    if demand == 0 {
        return Ok(());
    }
    let mm = MemoryManager::new(
        policy,
        memory_ceiling_mib(),
        Box::new(SysinfoBackend::new()),
    );
    mm.grow_for(demand).map_err(|e| format!("unfit: {e}"))
}

fn cmd_media_probe(path: &Path, json: bool) -> Result<(), String> {
    let decoded = runa_media::decode_audio(path).map_err(|e| e.to_string())?;
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&decoded.probe).map_err(|e| e.to_string())?
        );
    } else {
        println!("{}", decoded.probe);
    }
    Ok(())
}

fn cmd_media_video(path: &Path, fps: f32, max_frames: usize, json: bool) -> Result<(), String> {
    let opts = runa_media::VideoOpts {
        fps,
        max_frames,
        ..runa_media::VideoOpts::default()
    };
    let sampled = runa_media::sample_video(path, &opts).map_err(|e| e.to_string())?;
    if json {
        let times: Vec<f32> = sampled.frames.iter().map(|f| f.t_sec).collect();
        println!(
            "{}",
            serde_json::json!({
                "n_frames": sampled.frames.len(),
                "t_sec": times,
                "audio": sampled.audio.is_some(),
            })
        );
    } else {
        println!("frames: {}", sampled.frames.len());
        for f in &sampled.frames {
            println!("  [t={:.1}s] {}x{}", f.t_sec, f.width, f.height);
        }
        if sampled.audio.is_some() {
            println!("audio: yes");
        }
    }
    Ok(())
}

fn cmd_media_transcribe(
    path: &Path,
    model: &str,
    lang: &str,
    json: bool,
    no_pull: bool,
) -> Result<(), String> {
    let kind = runa_media::WhisperKind::parse(model).map_err(|e| e.to_string())?;
    let opts = runa_media::AsrOptions {
        kind,
        pull: !no_pull,
        language: match lang {
            "auto" | "" => None,
            other => Some(other.to_string()),
        },
        ..runa_media::AsrOptions::default()
    };
    let t = runa_media::transcribe_file(path, &opts).map_err(|e| e.to_string())?;
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&t).map_err(|e| e.to_string())?
        );
    } else {
        if let Some(lang) = &t.language {
            eprintln!("language: {lang}");
        }
        println!("{}", t.text);
    }
    Ok(())
}

fn auto_verdict_line(report: &runa_fit::FitReport, vram: u64, src: &str) -> String {
    let gpu_mib = report.plan.gpu_total_bytes as f64 / 1_048_576.0;
    let vram_mib = vram / 1_048_576;
    let speed = report
        .speed_gpu
        .as_ref()
        .or(report.speed_cpu.as_ref())
        .map(|s| format!(" · {:.1} tok/s decode", s.decode_toks_per_sec))
        .unwrap_or_default();
    format!(
        "runa auto {} · GPU {gpu_mib:.1} MiB · VRAM {vram_mib} MiB ({src}){speed}",
        report.verdict
    )
}

fn placement_from_report(report: &runa_fit::FitReport) -> Placement {
    match &report.verdict {
        runa_fit::Verdict::Gpu => Placement::gpu(),
        runa_fit::Verdict::Cpu => Placement::cpu(),
        runa_fit::Verdict::Hybrid {
            gpu_layers,
            total_layers,
        } => {
            if *gpu_layers >= *total_layers && *total_layers > 0 {
                Placement::hybrid_moe()
            } else {
                Placement {
                    n_gpu_layers: *gpu_layers as u32,
                    cpu_patterns: Vec::new(),
                    main_gpu: 0,
                    devices: Vec::new(),
                    tensor_split: Vec::new(),
                    rpc_servers: Vec::new(),
                }
            }
        }
        runa_fit::Verdict::NoFit => Placement::cpu(),
    }
}

pub(crate) enum AutoPlacement {
    Local(Placement),
    Cloud(runa_cloud::CloudRef),
}

/// P9.4: explicit NPU opt-in (`RUNA_NPU=hexagon|openvino`). `None` = not
/// requested (the default; placement never touches NPU — plan D12). An
/// unrecognized value is an explicit error, never silently ignored.
fn npu_request() -> Result<Option<NpuKind>, String> {
    match std::env::var("RUNA_NPU") {
        Err(_) => Ok(None),
        Ok(raw) if raw.trim().is_empty() => Ok(None),
        Ok(raw) => match NpuKind::parse(&raw) {
            Some(kind) => Ok(Some(kind)),
            None => Err(format!("RUNA_NPU={raw}: expected hexagon|openvino")),
        },
    }
}

/// P9.4: NPU status suffix for the `runa auto` verdict line. Pure (request
/// and probe result passed explicitly) so unit tests need no env. Empty
/// unless the user opted in — the verdict never mentions NPU by default.
fn npu_verdict_note(
    request: Option<NpuKind>,
    present: Option<NpuKind>,
    stub_decode_tps: f64,
) -> String {
    let Some(want) = request else {
        return String::new();
    };
    match present {
        Some(have) if have == want => format!(
            " · NPU {want} present (opt-in stub): ~{stub_decode_tps:.1} tok/s decode \
             (uncalibrated Tier-3 estimate); placement stays CPU — no ggml backend in llama-cpp-2"
        ),
        Some(have) => format!(
            " · NPU {want} requested but only {have} present; \
             staying on CPU (explicit, no silent fallback)"
        ),
        None => format!(
            " · NPU {want} requested but not present; \
             staying on CPU (explicit, no silent fallback)"
        ),
    }
}

pub(crate) fn auto_placement(
    path: &Path,
    ctx: u32,
    on_unfit: &config::OnUnfit,
    kv_type: &str,
    mmproj: Option<&Path>,
    draft: Option<&Path>,
    loras: &[LoraSpec],
) -> Result<AutoPlacement, String> {
    let header = read_local_prefix(path).map_err(|e| e.to_string())?;
    let reader = Reader::parse(&header.bytes).map_err(|e| e.to_string())?;
    let desc = Descriptor::from_reader(&reader).map_err(|e| e.to_string())?;
    let mmproj_path = mmproj
        .map(Path::to_path_buf)
        .or_else(|| runa_fit::sibling_mmproj(path));
    let mmproj_bytes = mmproj_path
        .as_ref()
        .map(|p| runa_fit::mmproj_file_bytes(p))
        .unwrap_or(0);
    let draft_bytes = draft.map(runa_fit::mmproj_file_bytes).unwrap_or(0);
    // LoRA adapters (P8.5): file sizes join the weight budget like the
    // draft model does.
    let lora_bytes: u64 = loras
        .iter()
        .map(|s| runa_fit::mmproj_file_bytes(&s.path))
        .sum();
    let (mut config, vram_src) =
        fit::machine_config(ctx, kv_type, mmproj_bytes.saturating_add(draft_bytes))?;
    config.planner.lora_bytes = lora_bytes;
    let report = check_fit(&desc, &config);
    // P9.4: NPU is opt-in only (`RUNA_NPU`) and stub-only (no ggml backend):
    // the note names the request/probe outcome explicitly; without a request
    // the verdict line is byte-identical to the pre-NPU format.
    let npu_note = match npu_request()? {
        None => String::new(),
        Some(want) => {
            let kv = estimate_kv(&desc, u64::from(ctx), kv_type);
            let tps = estimate_speed_single(&desc, &kv, u64::from(ctx), 1024, &want.hw_spec())
                .decode_toks_per_sec;
            npu_verdict_note(Some(want), npu_present(), tps)
        }
    };
    let line = format!(
        "{}{}",
        auto_verdict_line(&report, config.planner.vram_bytes, vram_src),
        npu_note
    );
    match &report.verdict {
        runa_fit::Verdict::NoFit => match on_unfit {
            config::OnUnfit::Cpu => {
                eprintln!("{line} · on_unfit=cpu → CPU");
                Ok(AutoPlacement::Local(Placement::cpu()))
            }
            config::OnUnfit::Error => Err(format!("unfit: {line} · on_unfit=error")),
            config::OnUnfit::Cloud(spec) => {
                let cloud = cloud::cloud_from_on_unfit(spec)?;
                eprintln!("{line} · on_unfit=cloud:{spec}");
                Ok(AutoPlacement::Cloud(cloud))
            }
        },
        _ => {
            eprintln!("{line}");
            Ok(AutoPlacement::Local(placement_from_report(&report)))
        }
    }
}

/// Prompt from argv, else from piped stdin.
fn read_prompt(arg: Option<String>) -> Result<String, String> {
    if let Some(p) = arg {
        if p == "-" {
            return read_stdin();
        }
        return Ok(p);
    }
    if io::stdin().is_terminal() {
        return Err("no prompt: pass one as an argument or pipe it on stdin".into());
    }
    read_stdin()
}

fn read_stdin() -> Result<String, String> {
    let mut s = String::new();
    io::stdin()
        .read_to_string(&mut s)
        .map_err(|e| format!("stdin: {e}"))?;
    if s.trim().is_empty() {
        return Err("empty prompt on stdin".into());
    }
    Ok(s)
}

/// `--json-schema`: inline JSON when it starts with `{`, else a file path.
/// Returned verbatim after a parse check.
fn read_json_schema(arg: &str) -> Result<String, String> {
    let text = if arg.trim_start().starts_with('{') {
        arg.to_owned()
    } else {
        fs::read_to_string(arg).map_err(|e| format!("--json-schema {arg}: {e}"))?
    };
    serde_json::from_str::<serde_json::Value>(&text)
        .map_err(|e| format!("--json-schema is not valid JSON: {e}"))?;
    Ok(text)
}

fn default_prompt_cache_dir() -> PathBuf {
    if let Ok(p) = std::env::var("RUNA_PROMPT_CACHE")
        && !p.is_empty()
    {
        return PathBuf::from(p);
    }
    if let Ok(p) = std::env::var("XDG_CACHE_HOME")
        && !p.is_empty()
    {
        return PathBuf::from(p).join("runa").join("kv");
    }
    match config::home_dir() {
        Some(home) => home.join(".cache").join("runa").join("kv"),
        None => std::env::temp_dir().join("runa-kv"),
    }
}

pub(crate) fn attach_prompt_cache(
    engine: &mut engine::LocalEngine,
    disabled: bool,
    explicit: Option<&Path>,
) -> Result<(), String> {
    if !engine.is_gguf() {
        // The mistral backend manages its own KV; there is nothing to attach.
        return Ok(());
    }
    if disabled || std::env::var_os("RUNA_NO_PROMPT_CACHE").is_some() {
        return Ok(());
    }
    let (dir, required) = match explicit {
        Some(p) => (p.to_path_buf(), true),
        None => (default_prompt_cache_dir(), false),
    };
    match PromptCache::open(&dir) {
        Ok(cache) => {
            engine.attach_prompt_cache(cache);
            Ok(())
        }
        Err(e) if required => Err(format!("prompt-cache: {e}")),
        Err(e) => {
            eprintln!("prompt-cache: disabled ({e})");
            Ok(())
        }
    }
}

/// `attach_prompt_cache` for a bare ggml load (P9.1 daemon path): the
/// caller holds a `LoadedModel`, not a `LocalEngine`.
pub(crate) fn attach_prompt_cache_local(
    loaded: &mut runa_engine::LoadedModel,
    disabled: bool,
    explicit: Option<&Path>,
) -> Result<(), String> {
    if disabled || std::env::var_os("RUNA_NO_PROMPT_CACHE").is_some() {
        return Ok(());
    }
    let (dir, required) = match explicit {
        Some(p) => (p.to_path_buf(), true),
        None => (default_prompt_cache_dir(), false),
    };
    match PromptCache::open(&dir) {
        Ok(cache) => {
            loaded.attach_prompt_cache(cache);
            Ok(())
        }
        Err(e) if required => Err(format!("prompt-cache: {e}")),
        Err(e) => {
            eprintln!("prompt-cache: disabled ({e})");
            Ok(())
        }
    }
}

fn parse_kv_kind(raw: &str) -> Result<KvKind, String> {
    raw.parse::<KvKind>().map_err(|e| format!("--kv: {e}"))
}

pub(crate) fn resolve_kv(
    kv: Option<&str>,
    kv_k: Option<&str>,
    kv_v: Option<&str>,
) -> Result<(Option<KvKind>, Option<KvKind>), String> {
    let both = kv.map(parse_kv_kind).transpose()?;
    let k = match kv_k {
        Some(s) => Some(parse_kv_kind(s)?),
        None => both,
    };
    let v = match kv_v {
        Some(s) => Some(parse_kv_kind(s)?),
        None => both,
    };
    Ok((k, v))
}

pub(crate) fn cli_think(
    think: Option<&str>,
    think_budget: Option<u32>,
    effort: Option<&str>,
    show_reasoning: bool,
    no_show_reasoning: bool,
) -> Result<ThinkConfig, String> {
    let mut o = ThinkOverrides::default();
    if let Some(s) = think {
        o.think = Some(ThinkOverrides::parse_think(s)?);
    }
    if let Some(n) = think_budget {
        o.budget = Some(parse_budget(&n.to_string())?);
    }
    if let Some(s) = effort {
        o.effort = Some(runa_core::Effort::parse(s)?);
    }
    if no_show_reasoning {
        o.show = Some(false);
    } else if show_reasoning {
        o.show = Some(true);
    }
    config::resolve_think(o)
}

fn collect_vision_frames(
    images: &[PathBuf],
    video: Option<&Path>,
) -> Result<Vec<VisionFrame>, String> {
    let mut out = Vec::new();
    for p in images {
        out.push(VisionFrame {
            t_sec: None,
            source: VisionSource::Path(p.clone()),
        });
    }
    if let Some(v) = video {
        let sampled = runa_media::sample_video(v, &runa_media::VideoOpts::default())
            .map_err(|e| e.to_string())?;
        for f in sampled.frames {
            out.push(VisionFrame {
                t_sec: Some(f.t_sec),
                source: VisionSource::Rgb {
                    width: f.width,
                    height: f.height,
                    rgb: f.rgb,
                },
            });
        }
    }
    Ok(out)
}

fn cmd_run(args: &RunArgs) -> Result<(), String> {
    let requested = BackendKind::parse(&args.backend)
        .ok_or_else(|| format!("{}: --backend must be gguf | mistral | auto", args.backend))?;
    // Startup cap check (warn-only): RAM pressure + thread share against
    // `[system] max_load_percent` (default 80). Model demand is checked
    // again in `preflight_grow` once the header is read.
    let threads = config::resolve_threads(args.threads)?;
    config::warn_if_over_system_limit(None, threads, args.max_load_percent)?;
    let think = cli_think(
        args.think.as_deref(),
        args.think_budget,
        args.effort.as_deref(),
        args.show_reasoning,
        args.no_show_reasoning,
    )?;
    let mut prompt = read_prompt(args.prompt.clone())?;
    let audio_pref = config::resolve_audio_route(args.audio_route.as_deref())?;
    let json_schema = args
        .json_schema
        .as_deref()
        .map(read_json_schema)
        .transpose()?;
    let grammar = args
        .grammar
        .as_ref()
        .map(|p| fs::read_to_string(p).map_err(|e| format!("--grammar {}: {e}", p.display())))
        .transpose()?;
    // Daemon first (P9.1): a warm daemon answers text-only requests over
    // the socket; anything it cannot serve (media, MCP, speculation,
    // load-shaping flags) or any dial failure falls back to the local
    // path below. Cloud refs never touch the daemon.
    if runa_cloud::parse_cloud_ref(&args.model).is_none()
        && !args.no_daemon
        && daemon_compatible_run(args)
        && let Some(outcome) = try_daemon_run(
            args,
            &think,
            &prompt,
            json_schema.as_deref(),
            grammar.as_deref(),
        )
    {
        return outcome;
    }
    let hub = mcp::start(&args.tools.mcp)?;
    let cloud_run = |prompt: &str, cloud: &runa_cloud::CloudRef| {
        if grammar.is_some() {
            return Err("--grammar is local-only; use --json-schema with cloud models".into());
        }
        cloud::run_cloud(
            cloud,
            &cloud::CloudRun {
                prompt,
                think,
                max_tokens: args.max_tokens,
                json: args.json,
                audio: args.audio.as_deref(),
                audio_pref,
                json_schema: json_schema.as_deref(),
                mcp: hub.as_ref(),
                max_tool_rounds: args.tools.max_tool_rounds,
            },
        )
    };
    if let Some(cloud) = runa_cloud::parse_cloud_ref(&args.model) {
        if requested != BackendKind::Auto {
            return Err("--backend is local-only; it does not apply to cloud models".into());
        }
        if !args.image.is_empty() || args.video.is_some() {
            return Err(
                "cloud --image/--video: native mtmd is local-only; send images through the API adapters (P4.7) or run a local VL model".into(),
            );
        }
        return cloud_run(&prompt, &cloud);
    }
    let path = resolve_model(&args.model)?;
    let kind = engine::resolve_requested(requested, &path)?;
    if kind == BackendKind::Mistral {
        return cmd_run_mistral(args, &path, think, prompt);
    }
    if let Some(p) = args.draft.as_ref()
        && !p.is_file()
    {
        return Err(format!("draft model not found: {}", p.display()));
    }
    let loras = config::resolve_loras(&args.lora, &args.model)?;
    let on_unfit = config::resolve_on_unfit(args.on_unfit.as_deref())?;
    let (kv_k, kv_v) = resolve_kv(
        args.kv.as_deref(),
        args.kv_k.as_deref(),
        args.kv_v.as_deref(),
    )?;
    let mut placement = match parse_mode_choice(&args.mode)? {
        ModeChoice::Fixed(mode) => Placement::from_mode(mode),
        ModeChoice::Auto => {
            match auto_placement(
                &path,
                args.ctx,
                &on_unfit,
                planner_kv_type(kv_k, kv_v),
                args.mmproj.as_deref(),
                args.draft.as_deref(),
                &loras,
            )? {
                AutoPlacement::Local(p) => p,
                AutoPlacement::Cloud(cloud) => return cloud_run(&prompt, &cloud),
            }
        }
    };
    if let Some(n) = args.n_cpu_moe {
        placement = placement.with_n_cpu_moe(n);
    }
    if let Some(s) = args.device.as_deref() {
        placement = placement.with_devices(parse_device_list(s)?);
    }
    if let Some(s) = args.tensor_split.as_deref() {
        placement = placement.with_tensor_split(parse_tensor_split(s)?);
    }
    if let Some(n) = args.main_gpu {
        placement = placement.with_main_gpu(n);
    }
    if let Some(s) = args.rpc.as_deref() {
        placement = placement.with_rpc_servers(parse_rpc_list(s)?);
    }
    let mmproj_for_audio = args.mmproj.clone().or_else(|| {
        args.audio
            .as_ref()
            .and_then(|_| runa_fit::sibling_mmproj(&path))
    });
    let audio_plan = if args.audio.is_some() {
        Some(runa_media::select_audio_route(
            audio_pref,
            runa_media::AudioBackend::Local {
                has_audio_mmproj: mmproj_for_audio.is_some(),
            },
        )?)
    } else {
        None
    };
    let mmproj = args.mmproj.clone().or_else(|| {
        if matches!(audio_plan, Some(runa_media::AudioPlan::Native))
            || !args.image.is_empty()
            || args.video.is_some()
        {
            runa_fit::sibling_mmproj(&path)
        } else {
            None
        }
    });
    if let (Some(p), Some(runa_media::AudioPlan::Transcribe)) = (&args.audio, audio_plan) {
        prompt = cloud::fold_audio_transcript(&prompt, p)?;
    }
    let config = LoadConfig {
        n_ctx: args.ctx,
        kv_k,
        kv_v,
        mmproj,
        loras,
        threads,
        ..LoadConfig::default()
    };
    let extra_bytes = args
        .draft
        .as_ref()
        .map(|p| runa_fit::mmproj_file_bytes(p))
        .unwrap_or(0)
        .saturating_add(
            config
                .loras
                .iter()
                .map(|s| runa_fit::mmproj_file_bytes(&s.path))
                .sum::<u64>(),
        );
    preflight_grow(
        &path,
        args.ctx,
        planner_kv_type(kv_k, kv_v),
        extra_bytes,
        args.max_load_percent,
    )?;
    if (args.ngram || args.draft.is_some()) && args.temperature > 0.0 {
        eprintln!("ngram: skipped (needs --temperature 0)");
    }
    let sampling = SamplingConfig {
        temperature: args.temperature,
        seed: args.seed,
        ..SamplingConfig::default()
    };
    let mut engine = engine::LocalEngine::load(BackendKind::Gguf, &path, &placement, &config)?;
    attach_prompt_cache(
        &mut engine,
        args.no_prompt_cache,
        args.prompt_cache.as_deref(),
    )?;
    let audio_pcm = match (&args.audio, audio_plan) {
        (None, _) | (Some(_), None) => None,
        (Some(_), Some(runa_media::AudioPlan::Transcribe)) => None,
        (Some(p), Some(runa_media::AudioPlan::Native)) => {
            if engine.supports_native_audio() {
                Some(
                    runa_media::decode_audio(p)
                        .map_err(|e| e.to_string())?
                        .samples,
                )
            } else if audio_pref == runa_media::AudioRoutePref::Auto {
                prompt = cloud::fold_audio_transcript(&prompt, p)?;
                None
            } else {
                return Err(
                    "loaded mmproj has no native audio; use --audio-route asr or auto".into(),
                );
            }
        }
        (Some(_), Some(runa_media::AudioPlan::OpenAiInputAudio)) => {
            return Err("openai input_audio is cloud-only".into());
        }
    };
    let mut req = GenerateRequest {
        messages: vec![ChatMessage::user(&prompt)],
        sampling,
        max_tokens: args.max_tokens,
        stop: Vec::new(),
        add_generation_prompt: true,
        think,
        audio_pcm,
        images: collect_vision_frames(&args.image, args.video.as_deref())?,
        speculative: runa_engine::Speculative {
            ngram: args.ngram || args.draft.is_some(),
            draft_n: 4,
            draft: args.draft.clone(),
        },
        json_schema,
        grammar,
        tools: hub.as_ref().map(|h| h.tools_json().to_string()),
        ..GenerateRequest::default()
    };
    let mut last = None;
    mcp::tool_loop(
        args.tools.max_tool_rounds,
        mcp::call_with(hub.as_ref()),
        |results| {
            crate::session::push_tool_results(&mut req.messages, results);
            let stream = engine.generate(req.clone()).map_err(|e| e.to_string())?;
            let turn = crate::session::drain_local(stream, !args.json)?;
            let calls = crate::session::push_tool_calls(&mut req.messages, &turn, !args.json);
            last = Some(turn);
            Ok(calls)
        },
    )?;
    let turn = last.expect("tool_loop runs at least one round");
    finish_run(&turn, args.json, &args.model);
    Ok(())
}

/// GGUF-only flags fail explicitly on the mistral backend instead of being
/// silently ignored (`--mode`/`--ctx`/`--seed` are documented in `--backend`
/// help as ggml-only and simply have no mistral equivalent).
fn reject_mistral_flags(args: &RunArgs) -> Result<(), String> {
    if args.on_unfit.is_some() {
        return Err("--on-unfit needs the gguf backend (fit is GGUF-only)".into());
    }
    if args.n_cpu_moe.is_some() {
        return Err("--n-cpu-moe needs the gguf backend".into());
    }
    if args.prompt_cache.is_some() {
        return Err("--prompt-cache needs the gguf backend".into());
    }
    if args.kv.is_some() || args.kv_k.is_some() || args.kv_v.is_some() {
        return Err("--kv/--kv-k/--kv-v need the gguf backend".into());
    }
    if args.threads.is_some() {
        return Err("--threads needs the gguf backend".into());
    }
    if args.max_load_percent.is_some() {
        return Err("--max-load-percent needs the gguf backend".into());
    }
    if args.device.is_some() || args.tensor_split.is_some() || args.main_gpu.is_some() {
        return Err("--device/--tensor-split/--main-gpu need the gguf backend".into());
    }
    if args.rpc.is_some() {
        return Err("--rpc needs the gguf backend (and no RPC backend in this build, P9.3)".into());
    }
    if args.audio.is_some()
        || args.mmproj.is_some()
        || !args.image.is_empty()
        || args.video.is_some()
    {
        return Err("--audio/--mmproj/--image/--video need the gguf backend (native mtmd)".into());
    }
    if args.ngram || args.draft.is_some() {
        return Err("--ngram/--draft need the gguf backend".into());
    }
    if !args.lora.is_empty() {
        return Err("--lora needs the gguf backend".into());
    }
    if args.json_schema.is_some() || args.grammar.is_some() {
        return Err("--json-schema/--grammar need the gguf backend".into());
    }
    if !args.tools.mcp.is_empty() {
        return Err("--mcp tools need the gguf backend".into());
    }
    Ok(())
}

/// One-shot generation through the mistral backend: sampling / think /
/// `max_tokens` map over; everything GGUF-only was rejected above.
fn cmd_run_mistral(
    args: &RunArgs,
    path: &Path,
    think: ThinkConfig,
    prompt: String,
) -> Result<(), String> {
    reject_mistral_flags(args)?;
    if !config::load_mcp_servers()?.is_empty() {
        return Err("--mcp tools need the gguf backend (config enables servers)".into());
    }
    let sampling = SamplingConfig {
        temperature: args.temperature,
        seed: args.seed,
        ..SamplingConfig::default()
    };
    let mut engine = engine::LocalEngine::load(
        BackendKind::Mistral,
        path,
        &Placement::cpu(),
        &LoadConfig {
            n_ctx: args.ctx,
            ..LoadConfig::default()
        },
    )?;
    let req = GenerateRequest {
        messages: vec![ChatMessage::user(&prompt)],
        sampling,
        max_tokens: args.max_tokens,
        think,
        ..GenerateRequest::default()
    };
    let stream = engine.generate(req)?;
    let turn = crate::session::drain_local(stream, !args.json)?;
    finish_run(&turn, args.json, &args.model);
    Ok(())
}

/// `run` requests the daemon cannot serve (P9.1): media, MCP servers,
/// speculation, an explicit prompt-cache dir, and load-shaping flags
/// (`--device`, `--tensor-split`, `--main-gpu`, `--rpc`, `--n-cpu-moe`,
/// `--threads`, `--max-load-percent`, `--kv*`). The daemon owns
/// ctx/mode/placement — use `--no-daemon` for exact local control.
fn daemon_compatible_run(args: &RunArgs) -> bool {
    args.audio.is_none()
        && args.image.is_empty()
        && args.video.is_none()
        && args.tools.mcp.is_empty()
        && !args.ngram
        && args.draft.is_none()
        && args.prompt_cache.is_none()
        && args.device.is_none()
        && args.tensor_split.is_none()
        && args.main_gpu.is_none()
        && args.rpc.is_none()
        && args.n_cpu_moe.is_none()
        && args.threads.is_none()
        && args.max_load_percent.is_none()
        && args.kv.is_none()
        && args.kv_k.is_none()
        && args.kv_v.is_none()
}

/// Try one daemon generation. `None` = no daemon listening (the caller
/// falls back to in-process load); `Some` is the served outcome, `Err`
/// included (a daemon-side failure is real, not a refusal).
fn try_daemon_run(
    args: &RunArgs,
    think: &ThinkConfig,
    prompt: &str,
    json_schema: Option<&str>,
    grammar: Option<&str>,
) -> Option<Result<(), String>> {
    let path = resolve_model(&args.model).ok()?;
    let req = GenerateRequest {
        messages: vec![ChatMessage::user(prompt)],
        sampling: SamplingConfig {
            temperature: args.temperature,
            seed: args.seed,
            ..SamplingConfig::default()
        },
        max_tokens: args.max_tokens,
        think: *think,
        json_schema: json_schema.map(str::to_owned),
        grammar: grammar.map(str::to_owned),
        ..GenerateRequest::default()
    };
    let events = match daemon_generate(&path.display().to_string(), &req) {
        Ok(events) => events,
        Err(_) => return None,
    };
    Some(drain_daemon(&events, !args.json).map(|turn| finish_run(&turn, args.json, &args.model)))
}

/// Send one request to the daemon; transport errors mean "no daemon".
pub(crate) fn daemon_generate(
    model: &str,
    req: &GenerateRequest,
) -> Result<Vec<daemon_proto::DaemonEvent>, String> {
    let dreq = daemon_proto::DaemonRequest::new(model.to_owned(), req);
    daemon_proto::request_sync(
        &daemon_proto::default_socket_path(),
        &dreq,
        daemon::REQUEST_TIMEOUT,
    )
}

/// Drain daemon events like [`drain_local`]: answer text to stdout,
/// reasoning to stderr. A daemon-side `error` fails the request.
pub(crate) fn drain_daemon(
    events: &[daemon_proto::DaemonEvent],
    print: bool,
) -> Result<crate::session::LocalTurn, String> {
    use daemon_proto::DaemonEvent;
    let mut turn = crate::session::LocalTurn {
        text: String::new(),
        calls: Vec::new(),
        usage: None,
        reason: StopReason::MaxTokens,
    };
    for ev in events {
        match ev {
            DaemonEvent::Text { text } => {
                if print {
                    print!("{text}");
                    io::stdout().flush().map_err(|e| format!("stdout: {e}"))?;
                }
                turn.text.push_str(text);
            }
            DaemonEvent::Reasoning { text } => {
                if print {
                    eprint!("{text}");
                    io::stderr().flush().map_err(|e| format!("stderr: {e}"))?;
                }
            }
            DaemonEvent::ToolCalls { calls } => {
                turn.calls = calls.iter().cloned().map(ToolCall::from).collect();
            }
            DaemonEvent::Usage {
                prompt_tokens,
                generated_tokens,
                reasoning_tokens,
            } => {
                turn.usage = Some(Usage {
                    prompt_tokens: *prompt_tokens,
                    generated_tokens: *generated_tokens,
                    reasoning_tokens: *reasoning_tokens,
                    pp_toks_per_s: 0.0,
                    tg_toks_per_s: 0.0,
                });
            }
            DaemonEvent::Done { stop } => turn.reason = daemon_proto::parse_stop(stop),
            DaemonEvent::Error { message } => return Err(message.clone()),
        }
    }
    Ok(turn)
}

/// Shared `run` epilogue (local and daemon paths): `--json` object or a
/// trailing newline plus the usage line on stderr.
fn finish_run(turn: &crate::session::LocalTurn, json: bool, model: &str) {
    let usage = turn.usage.clone().unwrap_or(Usage {
        prompt_tokens: 0,
        generated_tokens: 0,
        reasoning_tokens: 0,
        pp_toks_per_s: 0.0,
        tg_toks_per_s: 0.0,
    });
    if let Some(u) = &turn.usage {
        usage_log::record_usage("run", model, u);
    }
    if json {
        println!(
            "{}",
            crate::session::run_json(&turn.text, &usage, &turn.reason)
        );
    } else {
        println!();
        if turn.usage.is_some() {
            eprintln!(
                "tokens: prompt {} / generated {} · {:.1} pp tok/s · {:.1} tg tok/s",
                usage.prompt_tokens,
                usage.generated_tokens,
                usage.pp_toks_per_s,
                usage.tg_toks_per_s
            );
        }
    }
}

/// One local generation, drained.
/// Minimal JSON string escaping (no serde in the binary crate).
pub(crate) fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

fn cmd_pull(model_ref: &str) -> Result<(), String> {
    let pulled = pull::pull(model_ref)?;
    if pulled.fresh {
        println!(
            "{} ({}/{}; {} bytes)",
            pulled.path.display(),
            pulled.repo,
            pulled.file,
            pulled.size
        );
    } else {
        println!(
            "{} ({}/{}; already present, verified)",
            pulled.path.display(),
            pulled.repo,
            pulled.file
        );
    }
    Ok(())
}

fn task_registry_path() -> PathBuf {
    std::env::var("RUNA_TASK_REGISTRY")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("docs/tasks.md"))
}

fn cmd_stats(since: Option<&str>, json: bool) -> Result<(), String> {
    let window = match since {
        Some(s) => Some(usage_log::parse_since(s)?),
        None => None,
    };
    let report = usage_log::load_default(window)?;
    if json {
        println!("{}", report.to_json()?);
    } else {
        print!("{}", report.to_table());
    }
    Ok(())
}

fn cmd_tasks(action: TaskAction) -> Result<(), String> {
    let reg = TaskRegistry::open(task_registry_path());
    match action {
        TaskAction::List {} => {
            for id in reg.list_free().map_err(|e| e.to_string())? {
                println!("{id}");
            }
            Ok(())
        }
        TaskAction::Claim { task_id, agent } => match reg.claim(&task_id, &agent) {
            Ok(claim) => {
                println!(
                    "claimed {task_id} as {agent} at {started}",
                    started = claim.started_at
                );
                Ok(())
            }
            Err(ClaimError::AlreadyClaimed {
                agent: owner,
                started_at,
            }) => Err(format!(
                "{task_id}: in progress (owner: {owner}, since {started_at})"
            )),
            Err(e) => Err(e.to_string()),
        },
        TaskAction::Release { task_id, agent } => {
            reg.release(&task_id, &agent).map_err(|e| e.to_string())?;
            println!("released {task_id}");
            Ok(())
        }
    }
}

fn cmd_models() -> Result<(), String> {
    let aliases = config::load_aliases()?;
    println!("models in {}:", pull::models_dir().display());
    for dir in pull::extra_dirs() {
        println!("  and {}", dir.display());
    }
    for p in pull::list_models() {
        let size = std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0);
        println!("  {} ({size} bytes)", p.display());
    }
    if !aliases.models.is_empty() {
        println!("aliases:");
        let mut names: Vec<&String> = aliases.models.keys().collect();
        names.sort();
        for n in names {
            println!("  {n} -> {}", aliases.models[n].source);
        }
    }
    Ok(())
}

/// Cargo features compiled into this binary (plan D13 / P6.3).
fn compiled_backends() -> Vec<&'static str> {
    let mut out = vec!["cpu"];
    if cfg!(feature = "metal") {
        out.push("metal");
    }
    if cfg!(feature = "cuda") {
        out.push("cuda");
    }
    if cfg!(feature = "vulkan") {
        out.push("vulkan");
    }
    // P9.4: probe-only stubs (Tier 3) — reported only when built with the
    // feature; default binaries never list NPU capabilities (plan D12).
    if cfg!(feature = "hexagon") {
        out.push("hexagon-stub");
    }
    if cfg!(feature = "openvino") {
        out.push("openvino-stub");
    }
    if cfg!(feature = "mtmd") {
        out.push("mtmd");
    }
    // P9.2 safetensors backend (`--backend mistral`).
    if cfg!(feature = "mistralrs") {
        out.push("mistralrs");
    }
    out
}

fn doctor(json: bool) {
    let native_build = cfg!(feature = "native");
    let backends = compiled_backends();
    // P9.3: the vendored ggml RPC backend compiles only with `--features
    // rpc` (see `docs/versions.md`); reported explicitly either way.
    let rpc = cfg!(feature = "rpc");
    if json {
        let payload = serde_json::json!({
            "backends": backends,
            "native_build": native_build,
            "rpc": rpc,
        });
        println!("{payload}");
    } else {
        println!("runa doctor");
        println!(
            "native build (-C target-cpu=native): {}",
            if native_build {
                "yes"
            } else {
                "no (portable; ggml runtime dispatch)"
            }
        );
        println!("backends compiled in: {}", backends.join(", "));
        if rpc {
            println!("rpc: ggml RPC backend compiled in (`--rpc host:port` + `--device RPC0`)");
        } else {
            println!(
                "rpc: no ggml RPC backend in this build (rebuild with --features rpc; `--rpc` errors explicitly)"
            );
        }
        if backends.iter().any(|b| b.ends_with("-stub")) {
            println!(
                "NPU entries are probe-only stubs (Tier 3): no ggml backend in llama-cpp-2; \
                 placement stays CPU"
            );
        }
    }
}

#[cfg(test)]
mod npu_tests {
    use super::{NpuKind, npu_verdict_note};

    #[test]
    fn no_request_means_no_note() {
        assert_eq!(npu_verdict_note(None, Some(NpuKind::Hexagon), 12.0), "");
        assert_eq!(npu_verdict_note(None, None, 12.0), "");
    }

    #[test]
    fn present_match_names_npu_and_stays_cpu() {
        let note = npu_verdict_note(Some(NpuKind::Hexagon), Some(NpuKind::Hexagon), 12.34);
        assert!(note.contains("NPU hexagon present"), "{note}");
        assert!(note.contains("12.3"), "{note}");
        assert!(note.contains("placement stays CPU"), "{note}");
    }

    #[test]
    fn mismatch_and_absent_are_explicit() {
        let note = npu_verdict_note(Some(NpuKind::Hexagon), Some(NpuKind::OpenVino), 1.0);
        assert!(note.contains("only openvino present"), "{note}");
        assert!(note.contains("no silent fallback"), "{note}");

        let note = npu_verdict_note(Some(NpuKind::OpenVino), None, 1.0);
        assert!(
            note.contains("NPU openvino requested but not present"),
            "{note}"
        );
        assert!(note.contains("no silent fallback"), "{note}");
    }

    #[test]
    fn request_parsing_accepts_known_names() {
        assert_eq!(NpuKind::parse("hexagon"), Some(NpuKind::Hexagon));
        assert_eq!(NpuKind::parse("openvino"), Some(NpuKind::OpenVino));
        assert_eq!(NpuKind::parse("tpu"), None);
    }
}

#[cfg(test)]
mod daemon_gate_tests {
    use super::*;

    fn test_args() -> RunArgs {
        RunArgs {
            model: "m.gguf".into(),
            backend: "auto".into(),
            prompt: Some("hi".into()),
            mode: "auto".into(),
            ctx: 8192,
            max_tokens: 512,
            temperature: 0.8,
            seed: 42,
            threads: None,
            max_load_percent: None,
            json: false,
            on_unfit: None,
            n_cpu_moe: None,
            prompt_cache: None,
            no_prompt_cache: false,
            kv: None,
            kv_k: None,
            kv_v: None,
            think: None,
            think_budget: None,
            effort: None,
            show_reasoning: false,
            no_show_reasoning: false,
            device: None,
            tensor_split: None,
            main_gpu: None,
            rpc: None,
            audio: None,
            mmproj: None,
            audio_route: None,
            image: Vec::new(),
            video: None,
            ngram: false,
            draft: None,
            lora: Vec::new(),
            json_schema: None,
            grammar: None,
            no_daemon: false,
            tools: ToolArgs {
                mcp: Vec::new(),
                max_tool_rounds: 8,
            },
        }
    }

    #[test]
    fn plain_text_run_is_daemon_compatible() {
        assert!(daemon_compatible_run(&test_args()));
    }

    #[test]
    fn media_mcp_speculative_and_load_flags_stay_local() {
        let mut a = test_args();
        a.audio = Some(PathBuf::from("a.wav"));
        assert!(!daemon_compatible_run(&a));
        let mut a = test_args();
        a.image.push(PathBuf::from("i.png"));
        assert!(!daemon_compatible_run(&a));
        let mut a = test_args();
        a.video = Some(PathBuf::from("v.mp4"));
        assert!(!daemon_compatible_run(&a));
        let mut a = test_args();
        a.tools.mcp.push("server --flag".into());
        assert!(!daemon_compatible_run(&a));
        let mut a = test_args();
        a.ngram = true;
        assert!(!daemon_compatible_run(&a));
        let mut a = test_args();
        a.draft = Some(PathBuf::from("d.gguf"));
        assert!(!daemon_compatible_run(&a));
        let mut a = test_args();
        a.prompt_cache = Some(PathBuf::from("/tmp/kv"));
        assert!(!daemon_compatible_run(&a));
        let mut a = test_args();
        a.device = Some("0".into());
        assert!(!daemon_compatible_run(&a));
        let mut a = test_args();
        a.tensor_split = Some("3,1".into());
        assert!(!daemon_compatible_run(&a));
        let mut a = test_args();
        a.main_gpu = Some(1);
        assert!(!daemon_compatible_run(&a));
        let mut a = test_args();
        a.rpc = Some("127.0.0.1:50052".into());
        assert!(!daemon_compatible_run(&a));
        let mut a = test_args();
        a.n_cpu_moe = Some(2);
        assert!(!daemon_compatible_run(&a));
        let mut a = test_args();
        a.kv = Some("q8_0".into());
        assert!(!daemon_compatible_run(&a));
        let mut a = test_args();
        a.threads = Some(4);
        assert!(!daemon_compatible_run(&a));
        let mut a = test_args();
        a.max_load_percent = Some(50);
        assert!(!daemon_compatible_run(&a));
        // Per-request knobs stay daemon-compatible.
        let mut a = test_args();
        a.think_budget = Some(256);
        a.json_schema = Some(r#"{"type":"object"}"#.into());
        a.max_tokens = 64;
        assert!(daemon_compatible_run(&a));
    }
}

#[cfg(test)]
mod docs_lint {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn man_page_via_clap_mangen() {
        let cmd = Cli::command();
        let mut buf = Vec::new();
        clap_mangen::Man::new(cmd.clone())
            .render(&mut buf)
            .expect("render man");
        let s = String::from_utf8(buf).expect("utf8 man");
        assert!(s.contains("runa"), "{s}");
        assert!(s.contains("SUBCOMMANDS"), "{s}");
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("runa.1"), &s).expect("write runa.1");
        let run = cmd.find_subcommand("run").expect("run subcommand").clone();
        let mut run_buf = Vec::new();
        clap_mangen::Man::new(run)
            .render(&mut run_buf)
            .expect("render run man");
        let run_s = String::from_utf8(run_buf).expect("utf8 run man");
        assert!(
            run_s.contains("audio") && run_s.contains("route"),
            "{run_s}"
        );
        std::fs::write(dir.path().join("runa-run.1"), &run_s).expect("write runa-run.1");
    }
}
