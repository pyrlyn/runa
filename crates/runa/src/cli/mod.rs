// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! The clap surface: `Cli`, its subcommands and their shared flag groups.

use std::path::PathBuf;

use clap::{Parser, Subcommand};

pub(crate) mod config;
pub(crate) mod input;

use crate::fit;

/// Run AI models locally or through the OpenAI and Anthropic APIs.
#[derive(Debug, Parser)]
#[command(name = "runa", version, about, long_about = None, arg_required_else_help = true)]
pub(crate) struct Cli {
    #[command(subcommand)]
    pub(crate) command: Commands,
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

#[derive(Debug, clap::Args)]
pub(crate) struct RunArgs {
    /// Local model path: `.gguf` file or mistral safetensors directory.
    pub(crate) model: String,
    /// Local backend: gguf | mistral | auto (default: auto-detect from the path).
    /// The mistral backend maps sampling / think / stop / usage; `--mode`,
    /// `--ctx` and `--seed` are gguf-only and ignored there (see docs/memory.md P9.2).
    #[arg(long, default_value = "auto")]
    pub(crate) backend: String,
    /// Prompt text (else read from stdin when piped).
    pub(crate) prompt: Option<String>,
    /// Compute mode: cpu | gpu | hybrid | auto (default: auto).
    #[arg(long, default_value = "auto")]
    pub(crate) mode: String,
    /// Context length.
    #[arg(long, default_value_t = 8192)]
    pub(crate) ctx: u32,
    /// Max new tokens.
    #[arg(long, default_value_t = 512)]
    pub(crate) max_tokens: u32,
    /// Sampling temperature (<= 0 = greedy).
    #[arg(long, default_value_t = 0.8)]
    pub(crate) temperature: f32,
    /// Sampler seed.
    #[arg(long, default_value_t = 42)]
    pub(crate) seed: u32,
    /// Worker threads (default: P-cores on macOS, else all logical CPUs).
    /// GGUF backend only (P10.2).
    #[arg(long, value_name = "N")]
    pub(crate) threads: Option<i32>,
    /// Max share of total system resources (RAM budget + CPU thread share)
    /// this run may use, percent 1..=100 (default: 80, `[system]`).
    #[arg(long, value_name = "N")]
    pub(crate) max_load_percent: Option<u8>,
    /// Emit one JSON object instead of streaming text.
    #[arg(long, default_value_t = false)]
    pub(crate) json: bool,
    /// When auto cannot fit: error | cpu | cloud:<backend>:<model> (D12).
    #[arg(long)]
    pub(crate) on_unfit: Option<String>,
    /// Keep MoE expert tensors of the first N layers on CPU (`--n-cpu-moe`).
    #[arg(long, value_name = "N")]
    pub(crate) n_cpu_moe: Option<u32>,
    /// LMDB dir for prompt KV cache (default: ~/.cache/runa/kv).
    #[arg(long, value_name = "DIR")]
    pub(crate) prompt_cache: Option<PathBuf>,
    /// Disable the LMDB prompt cache (P2.8).
    #[arg(long, default_value_t = false)]
    pub(crate) no_prompt_cache: bool,
    /// KV cache type for both K and V (`f16`, `q8_0`, `q4_0`). Requires flash-attn.
    #[arg(long, value_name = "TYPE")]
    pub(crate) kv: Option<String>,
    /// KV type for K only (overrides `--kv`).
    #[arg(long, value_name = "TYPE")]
    pub(crate) kv_k: Option<String>,
    /// KV type for V only (overrides `--kv`).
    #[arg(long, value_name = "TYPE")]
    pub(crate) kv_v: Option<String>,
    /// Thinking: on | off (P3.1). Budget/effort override `on`.
    #[arg(long, value_name = "on|off")]
    pub(crate) think: Option<String>,
    /// Reasoning token budget (`ThinkMode::Budget`).
    #[arg(long, value_name = "N")]
    pub(crate) think_budget: Option<u32>,
    /// Effort level: low | medium | high | max.
    #[arg(long, value_name = "LEVEL")]
    pub(crate) effort: Option<String>,
    /// Print reasoning (`Event::Reasoning` lands in P3.2).
    #[arg(long, default_value_t = false)]
    pub(crate) show_reasoning: bool,
    /// Hide reasoning even if config `[think] show = true`.
    #[arg(long, default_value_t = false)]
    pub(crate) no_show_reasoning: bool,
    /// ggml backends to use (`0,1` or `CUDA0,CUDA1`).
    #[arg(long, value_name = "LIST")]
    pub(crate) device: Option<String>,
    /// Per-GPU proportions (`3,1`). Requires multiple GPUs.
    #[arg(long, value_name = "LIST")]
    pub(crate) tensor_split: Option<String>,
    /// Main GPU for scratch/small tensors (`--main-gpu N`, P2.9).
    #[arg(long, value_name = "N")]
    pub(crate) main_gpu: Option<i32>,
    /// llama.cpp RPC endpoints (`host:port,…`); registers only — select
    /// with `--device RPC0` (needs the `rpc` cargo feature, else an
    /// explicit error, P9.3).
    #[arg(long, value_name = "LIST")]
    pub(crate) rpc: Option<String>,
    /// Audio file (PCM 16 kHz). Routed by `--audio-route`.
    #[arg(long, value_name = "PATH")]
    pub(crate) audio: Option<PathBuf>,
    /// Audio/vision mmproj GGUF. Sibling `*mmproj*.gguf` if `--audio` and omitted.
    #[arg(long, value_name = "PATH")]
    pub(crate) mmproj: Option<PathBuf>,
    /// Audio route: auto | native | asr (P4.4).
    #[arg(long, value_name = "auto|native|asr")]
    pub(crate) audio_route: Option<String>,
    /// Image file(s) for native mtmd vision (P4.6).
    #[arg(long, value_name = "PATH")]
    pub(crate) image: Vec<PathBuf>,
    /// Video file: sampled frames with `[t=12.0s]` markers (P4.6).
    #[arg(long, value_name = "PATH")]
    pub(crate) video: Option<PathBuf>,
    /// Trigram speculative decoding; greedy-verified (use --temperature 0).
    #[arg(long, default_value_t = false)]
    pub(crate) ngram: bool,
    /// Draft-model GGUF: counted in fit; speculation still uses n-gram.
    #[arg(long, value_name = "PATH")]
    pub(crate) draft: Option<PathBuf>,
    /// LoRA adapter GGUF (`path[:scale]`). Repeatable; adds to `[model]` /
    /// `[models]` `lora` in config, counted in fit (P8.5).
    #[arg(long, value_name = "PATH[:SCALE]")]
    pub(crate) lora: Vec<String>,
    /// Constrain the answer to a JSON Schema (file path or inline JSON).
    #[arg(long, value_name = "FILE|JSON", conflicts_with = "grammar")]
    pub(crate) json_schema: Option<String>,
    /// Constrain the answer to a GBNF grammar file.
    #[arg(long, value_name = "FILE")]
    pub(crate) grammar: Option<PathBuf>,
    /// Skip the daemon (`~/.cache/runa/runa.sock`) and load the model
    /// in-process, even when `runa daemon` is running (P9.1).
    #[arg(long, default_value_t = false)]
    pub(crate) no_daemon: bool,
    #[command(flatten)]
    pub(crate) tools: ToolArgs,
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
