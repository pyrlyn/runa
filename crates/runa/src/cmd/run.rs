// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! `runa run`: one-shot generation (daemon first, else in-process).

use std::fs;
use std::path::Path;

use runa_core::{BackendKind, ThinkConfig};
use runa_engine::{
    ChatMessage, GenerateRequest, LoadConfig, Placement, SamplingConfig, Usage, parse_device_list,
    parse_rpc_list, parse_tensor_split, planner_kv_type,
};
use runa_pool::engine;

use crate::cli::RunArgs;
use crate::cli::input::{
    attach_prompt_cache, cli_think, collect_vision_frames, read_json_schema, read_prompt,
    resolve_kv,
};
use crate::daemon::daemon_generate;
use crate::placement::{
    AutoPlacement, ModeChoice, auto_placement, parse_mode_choice, preflight_grow, resolve_model,
};
use crate::session::drain_daemon;
use crate::{cloud, config, mcp, usage_log};

pub(super) fn cmd_run(args: &RunArgs) -> Result<(), String> {
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

#[cfg(test)]
mod daemon_gate_tests {
    use std::path::PathBuf;

    use super::*;
    use crate::cli::ToolArgs;

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
