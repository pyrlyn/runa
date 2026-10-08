// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! `runa bench` — pp/tg like llama-bench, JSON, calibration DB (P2.10).

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use runa_engine::{
    KvKind, LoadConfig, Placement, load, parse_device_list, parse_tensor_split, planner_kv_type,
};
use runa_fit::calibration::{CalibrationDb, CalibrationSample};
use runa_fit::{
    Descriptor, HwSpec, Reader, apply_efficiency, estimate_kv, estimate_speed_single,
    read_local_prefix,
};
use sha2::{Digest, Sha256};

use crate::{AutoPlacement, ModeChoice, auto_placement, parse_mode_choice, resolve_model};

/// SHA-256 of the first 1 MiB concatenated with the last 1 MiB.
pub(crate) fn model_hash(path: &Path) -> Result<String, String> {
    const CHUNK: u64 = 1024 * 1024;
    let mut f = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let len = f
        .metadata()
        .map_err(|e| format!("stat {}: {e}", path.display()))?
        .len();
    let first_n = CHUNK.min(len);
    let mut first = vec![0u8; first_n as usize];
    f.read_exact(&mut first)
        .map_err(|e| format!("read {}: {e}", path.display()))?;
    let mut last = Vec::new();
    if len > 0 {
        let last_n = CHUNK.min(len);
        f.seek(SeekFrom::End(-(last_n as i64)))
            .map_err(|e| format!("seek {}: {e}", path.display()))?;
        last.resize(last_n as usize, 0);
        f.read_exact(&mut last)
            .map_err(|e| format!("read {}: {e}", path.display()))?;
    }
    let mut h = Sha256::new();
    h.update(&first);
    h.update(&last);
    Ok(format!(
        "sha256:{}",
        h.finalize()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>()
    ))
}

pub(crate) fn quant_from_name(path: &Path) -> String {
    let name = path
        .file_stem()
        .unwrap_or_default()
        .to_string_lossy()
        .to_ascii_uppercase();
    const TAGS: &[&str] = &[
        "Q4_K_M", "Q5_K_M", "Q4_K_S", "Q5_K_S", "Q3_K_M", "Q6_K", "Q2_K", "IQ4_XS", "MXFP4",
        "Q8_0", "Q5_0", "Q4_0", "Q6_0",
    ];
    for tag in TAGS {
        if name.contains(tag) {
            return (*tag).to_string();
        }
    }
    "unknown".into()
}

fn placement_name(p: &Placement) -> &'static str {
    if p.n_gpu_layers == 0 {
        "cpu"
    } else if !p.cpu_patterns.is_empty() {
        "hybrid"
    } else {
        "gpu"
    }
}

pub(crate) fn device_backend(placement: &str) -> (&'static str, &'static str) {
    match placement {
        "cpu" => ("cpu", "cpu"),
        _ if cfg!(target_os = "macos") => ("metal:0", "metal"),
        _ => ("cuda:0", "cuda"),
    }
}

pub(crate) fn hw_for(placement: &str) -> HwSpec {
    match placement {
        "cpu" => HwSpec::cpu(),
        _ if cfg!(target_os = "macos") => HwSpec::metal(),
        _ => HwSpec::cuda(),
    }
}

pub(crate) fn default_calibration_path() -> PathBuf {
    if let Ok(p) = std::env::var("RUNA_CALIBRATION")
        && !p.is_empty()
    {
        return PathBuf::from(p);
    }
    match std::env::var_os("HOME") {
        Some(home) => PathBuf::from(home).join(".runa").join("calibration.json"),
        None => std::env::temp_dir().join("runa-calibration.json"),
    }
}

#[derive(Debug, serde::Serialize)]
struct BenchResult {
    model: String,
    model_hash: String,
    quant: String,
    placement: String,
    device: String,
    backend: String,
    ctx: u32,
    n_prompt: u32,
    n_gen: u32,
    prompt_tokens: u32,
    generated_tokens: u32,
    pp_tok_s: f64,
    tg_tok_s: f64,
    predicted_pp: f64,
    predicted_tg: f64,
}

fn bench_json(result: &BenchResult) -> String {
    serde_json::to_string(result).expect("BenchResult is always serializable")
}

fn predicted_speeds(
    path: &Path,
    ctx: u32,
    n_prompt: u32,
    placement: &str,
    kv_type: &str,
) -> (f64, f64) {
    let Ok(header) = read_local_prefix(path) else {
        return (0.0, 0.0);
    };
    let Ok(reader) = Reader::parse(&header.bytes) else {
        return (0.0, 0.0);
    };
    let Ok(desc) = Descriptor::from_reader(&reader) else {
        return (0.0, 0.0);
    };
    let kv = estimate_kv(&desc, u64::from(ctx), kv_type);
    let est = estimate_speed_single(
        &desc,
        &kv,
        u64::from(ctx),
        u64::from(n_prompt),
        &hw_for(placement),
    );
    // P10.1 (M3): scale by the calibration DB when this exact
    // (device, backend, quant) was measured before; an empty or
    // missing DB leaves the raw model untouched.
    let (device, backend) = device_backend(placement);
    let db = CalibrationDb::load(&default_calibration_path());
    let eff = db.get_efficiency(device, backend, &quant_from_name(path));
    apply_efficiency(
        est.prefill_toks_per_sec,
        est.decode_toks_per_sec,
        eff.as_ref(),
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn cmd_bench(
    model: &str,
    mode: &str,
    ctx: u32,
    n_prompt: u32,
    n_gen: u32,
    json: bool,
    no_calibrate: bool,
    kv_k: Option<KvKind>,
    kv_v: Option<KvKind>,
    device: Option<&str>,
    tensor_split: Option<&str>,
    threads: Option<i32>,
    max_load_percent: Option<u8>,
) -> Result<(), String> {
    if n_prompt == 0 {
        return Err("--pp must be > 0".into());
    }
    let need = n_prompt.saturating_add(n_gen).saturating_add(1);
    if ctx < need {
        return Err(format!(
            "--ctx {ctx} is too small for pp{n_prompt}+tg{n_gen} (need >= {need})"
        ));
    }
    let path = resolve_model(model)?;
    let on_unfit = crate::config::resolve_on_unfit(None)?;
    let threads = crate::config::resolve_threads(threads)?;
    // Startup cap check (warn-only): RAM pressure + thread share against
    // `[system] max_load_percent` (default 80).
    crate::config::warn_if_over_system_limit(None, threads, max_load_percent)?;
    let kv_type = planner_kv_type(kv_k, kv_v);
    let mut placement = match parse_mode_choice(mode)? {
        ModeChoice::Fixed(m) => Placement::from_mode(m),
        ModeChoice::Auto => {
            match auto_placement(&path, ctx, &on_unfit, kv_type, None, None, &[])? {
                AutoPlacement::Local(p) => p,
                AutoPlacement::Cloud(_) => {
                    return Err(
                        "bench requires a local model (cloud on_unfit fallback is not supported)"
                            .into(),
                    );
                }
            }
        }
    };
    if let Some(s) = device {
        placement = placement.with_devices(parse_device_list(s)?);
    }
    if let Some(s) = tensor_split {
        placement = placement.with_tensor_split(parse_tensor_split(s)?);
    }
    let config = LoadConfig {
        n_ctx: ctx,
        n_batch: n_prompt.max(1),
        n_ubatch: n_prompt.max(1),
        kv_k,
        kv_v,
        threads,
        ..LoadConfig::default()
    };
    let mut loaded = load(&path, &placement, &config).map_err(|e| e.to_string())?;
    let usage = loaded.bench(n_prompt, n_gen).map_err(|e| e.to_string())?;
    let place = placement_name(&placement);
    let (device, backend) = device_backend(place);
    let (predicted_pp, predicted_tg) = predicted_speeds(&path, ctx, n_prompt, place, kv_type);
    let result = BenchResult {
        model: path.display().to_string(),
        model_hash: model_hash(&path)?,
        quant: quant_from_name(&path),
        placement: place.to_string(),
        device: device.to_string(),
        backend: backend.to_string(),
        ctx,
        n_prompt,
        n_gen,
        prompt_tokens: usage.prompt_tokens,
        generated_tokens: usage.generated_tokens,
        pp_tok_s: usage.pp_toks_per_s,
        tg_tok_s: usage.tg_toks_per_s,
        predicted_pp,
        predicted_tg,
    };
    if !no_calibrate {
        let db_path = default_calibration_path();
        let mut db = CalibrationDb::load(&db_path);
        db.insert(CalibrationSample {
            model_hash: result.model_hash.clone(),
            quant: result.quant.clone(),
            placement: result.placement.clone(),
            ctx: u64::from(ctx),
            measured_pp: result.pp_tok_s,
            measured_tg: result.tg_tok_s,
            predicted_pp,
            predicted_tg,
            device: result.device.clone(),
            backend: result.backend.clone(),
            timestamp: chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
        });
        db.save(&db_path)?;
    }
    if json {
        println!("{}", bench_json(&result));
    } else {
        println!(
            "pp{n_prompt}: {pp:.1} tok/s (pred {ppp:.1})\ntg{n_gen}: {tg:.1} tok/s (pred {ptg:.1})",
            pp = result.pp_tok_s,
            ppp = result.predicted_pp,
            tg = result.tg_tok_s,
            ptg = result.predicted_tg,
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn json_schema_has_required_keys() {
        let result = BenchResult {
            model: "m.gguf".into(),
            model_hash: "sha256:abc".into(),
            quant: "Q4_0".into(),
            placement: "cpu".into(),
            device: "cpu".into(),
            backend: "cpu".into(),
            ctx: 2048,
            n_prompt: 512,
            n_gen: 128,
            prompt_tokens: 512,
            generated_tokens: 128,
            pp_tok_s: 100.0,
            tg_tok_s: 20.0,
            predicted_pp: 90.0,
            predicted_tg: 18.0,
        };
        let v: serde_json::Value = serde_json::from_str(&bench_json(&result)).unwrap();
        for key in [
            "model",
            "model_hash",
            "quant",
            "placement",
            "device",
            "backend",
            "ctx",
            "n_prompt",
            "n_gen",
            "prompt_tokens",
            "generated_tokens",
            "pp_tok_s",
            "tg_tok_s",
            "predicted_pp",
            "predicted_tg",
        ] {
            assert!(v.get(key).is_some(), "missing {key} in {v}");
        }
        assert_eq!(v["n_prompt"], 512);
        assert_eq!(v["n_gen"], 128);
    }

    #[test]
    fn quant_from_qwen_fixture_name() {
        assert_eq!(
            quant_from_name(Path::new("qwen2-0_5b-instruct-q4_0.gguf")),
            "Q4_0"
        );
        assert_eq!(quant_from_name(Path::new("Qwen3-8B-Q4_K_M.gguf")), "Q4_K_M");
    }

    #[test]
    fn model_hash_roundtrip_small_file() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("runa-hash-{}.bin", std::process::id()));
        let mut f = File::create(&path).unwrap();
        f.write_all(b"hello world").unwrap();
        drop(f);
        let a = model_hash(&path).unwrap();
        let b = model_hash(&path).unwrap();
        assert_eq!(a, b);
        assert!(a.starts_with("sha256:"));
        assert_eq!(a.len(), "sha256:".len() + 64);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn insert_sample_appears_in_db() {
        let path = std::env::temp_dir().join(format!(
            "runa-cal-unit-{}-{}.json",
            std::process::id(),
            "p210"
        ));
        let _ = std::fs::remove_file(&path);
        let mut db = CalibrationDb::load(&path);
        assert!(db.is_empty());
        db.insert(CalibrationSample {
            model_hash: "sha256:dead".into(),
            quant: "Q4_0".into(),
            placement: "cpu".into(),
            ctx: 256,
            measured_pp: 10.0,
            measured_tg: 2.0,
            predicted_pp: 9.0,
            predicted_tg: 1.8,
            device: "cpu".into(),
            backend: "cpu".into(),
            timestamp: "2026-09-08T16:00:00Z".into(),
        });
        db.save(&path).unwrap();
        let loaded = CalibrationDb::load(&path);
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded.samples()[0].model_hash, "sha256:dead");
        let _ = std::fs::remove_file(path);
    }
}
