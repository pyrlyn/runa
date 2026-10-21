// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! Flag and stdin input shared by `run`, `chat` and `bench`: prompt,
//! JSON schema, KV kinds, think flags, vision frames, prompt-cache attach.

use std::fs;
use std::io::{self, IsTerminal, Read};
use std::path::{Path, PathBuf};

use runa_core::{ThinkConfig, ThinkOverrides, parse_budget};
use runa_engine::{KvKind, PromptCache, VisionFrame, VisionSource};
use runa_pool::engine;

use crate::config;

/// Prompt from argv, else from piped stdin.
pub(crate) fn read_prompt(arg: Option<String>) -> Result<String, String> {
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
pub(crate) fn read_json_schema(arg: &str) -> Result<String, String> {
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

pub(crate) fn collect_vision_frames(
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
