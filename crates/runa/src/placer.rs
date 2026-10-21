// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! Placement policy: `--mode` parsing, the `auto` fit-driven placement and
//! the memory preflight that `run`, `chat` and `bench` use, plus the
//! placer `serve` and `daemon` give their model pool (P16.2): fit check,
//! then `--mode` placement with the CLI overrides applied. It lives here,
//! not in `runa-pool`, because it needs the config, memory policy and fit
//! machinery that the `runa` library carries.

use std::path::Path;
use std::sync::Arc;

use runa_engine::{LoadConfig, LoraSpec, Mode, Placement};
use runa_fit::{
    Descriptor, FitConfig, HwSpec, NpuKind, PlannerConfig, Reader, check_fit, estimate_compute,
    estimate_kv, estimate_speed_single, npu_present, read_local_prefix,
};
use runa_memory::{MemoryManager, SysinfoBackend};
use runa_pool::Placer;

use crate::{cloud, config, hw};

/// Parsed `--device/--tensor-split/--main-gpu/--rpc` overrides for
/// `serve` (and the daemon later): empty = no overrides.
#[derive(Debug, Clone, Default)]
pub struct PlacementOverrides {
    pub devices: Vec<String>,
    pub tensor_split: Vec<f32>,
    pub main_gpu: Option<i32>,
    pub rpc_servers: Vec<String>,
}

impl PlacementOverrides {
    pub fn apply(&self, mut placement: Placement) -> Placement {
        if !self.devices.is_empty() {
            placement = placement.with_devices(self.devices.clone());
        }
        if !self.tensor_split.is_empty() {
            placement = placement.with_tensor_split(self.tensor_split.clone());
        }
        if let Some(n) = self.main_gpu {
            placement = placement.with_main_gpu(n);
        }
        if !self.rpc_servers.is_empty() {
            placement = placement.with_rpc_servers(self.rpc_servers.clone());
        }
        placement
    }
}

/// The pool's [`Placer`] for `--mode` (`cpu|gpu|hybrid|auto`). A bad mode
/// fails here, so a typo is caught at startup, before the port is bound.
/// `max_load_percent` is the CLI `--max-load-percent` for the demand check
/// (`None` = env/config, default 80); the overrides apply on top of both
/// fixed and `auto` placements so an explicit flag is never dropped
/// silently (plan D12).
pub(crate) fn local_placer(
    mode: &str,
    overrides: PlacementOverrides,
    max_load_percent: Option<u8>,
) -> Result<Placer, String> {
    let choice = parse_mode_choice(mode)?;
    let base = match choice {
        ModeChoice::Fixed(m) => Placement::from_mode(m),
        ModeChoice::Auto => Placement::from_mode(Mode::Cpu),
    };
    let auto = matches!(choice, ModeChoice::Auto);
    Ok(Arc::new(move |path, config| {
        fit_check_no_fit(path, config.n_ctx, lora_bytes(config))?;
        let kv_type = runa_engine::planner_kv_type(config.kv_k, config.kv_v);
        if !auto {
            preflight_grow(path, config.n_ctx, kv_type, 0, max_load_percent)?;
            return Ok(overrides.apply(base.clone()));
        }
        match auto_placement(
            path,
            config.n_ctx,
            &config::OnUnfit::Error,
            kv_type,
            None,
            None,
            &config.loras,
        )? {
            AutoPlacement::Local(p) => Ok(overrides.apply(p)),
            AutoPlacement::Cloud(_) => {
                Err("unfit: auto mode chose cloud fallback; serve is local-only".into())
            }
        }
    }))
}

fn fit_check_no_fit(path: &Path, ctx: u32, lora_bytes: u64) -> Result<(), String> {
    let header = read_local_prefix(path).map_err(|e| e.to_string())?;
    let reader = Reader::parse(&header.bytes).map_err(|e| e.to_string())?;
    let desc = Descriptor::from_reader(&reader).map_err(|e| e.to_string())?;
    let (vram, _) = hw::vram_bytes()?;
    let planner = PlannerConfig {
        vram_bytes: vram,
        ram_bytes: hw::ram_bytes(),
        ctx_len: u64::from(ctx),
        kv_type: runa_engine::planner_kv_type(None, None).to_owned(),
        lora_bytes,
        ..PlannerConfig::default()
    };
    let report = check_fit(
        &desc,
        &FitConfig {
            planner,
            gpu_hw: None,
            cpu_hw: HwSpec::cpu(),
            has_mmproj: false,
            media: runa_fit::MediaFit::default(),
        },
    );
    if matches!(report.verdict, runa_fit::Verdict::NoFit) {
        return Err(format!("unfit: model does not fit (ctx={ctx})"));
    }
    Ok(())
}

/// Adapter bytes summed from the configured `--lora` files (P8.5).
fn lora_bytes(config: &LoadConfig) -> u64 {
    config
        .loras
        .iter()
        .map(|s| runa_fit::mmproj_file_bytes(&s.path))
        .sum()
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

fn bytes_to_mib(bytes: u64) -> u64 {
    bytes.div_ceil(1024 * 1024)
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
        hw::memory_ceiling_mib(),
        Box::new(SysinfoBackend::new()),
    );
    mm.grow_for(demand).map_err(|e| format!("unfit: {e}"))
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
        hw::machine_config(ctx, kv_type, mmproj_bytes.saturating_add(draft_bytes))?;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn placement_overrides_apply_on_top_of_any_mode() {
        // P2.9/P9.3: explicit serve flags survive both fixed and `auto`
        // placements (plan D12 — never dropped silently).
        let base = Placement::gpu();
        let full = PlacementOverrides {
            devices: vec!["0".into()],
            tensor_split: vec![3.0, 1.0],
            main_gpu: Some(1),
            rpc_servers: vec!["127.0.0.1:50052".into()],
        }
        .apply(base);
        assert_eq!(full.devices, vec!["0"]);
        assert_eq!(full.tensor_split, vec![3.0, 1.0]);
        assert_eq!(full.main_gpu, 1);
        assert_eq!(full.rpc_servers, vec!["127.0.0.1:50052"]);
        assert_eq!(full.n_gpu_layers, u32::MAX);

        let empty = PlacementOverrides::default().apply(Placement::cpu());
        assert_eq!(empty, Placement::cpu());
    }

    #[test]
    fn a_bad_mode_fails_at_construction() {
        let err = local_placer("warp", PlacementOverrides::default(), None)
            .err()
            .expect("bad mode");
        assert!(err.contains("--mode must be"), "{err}");
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
