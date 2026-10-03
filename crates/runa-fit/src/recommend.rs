// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! `runa fit --recommend` (P8.4): rank a curated model list for this machine.
//!
//! Each catalog entry is fitted in parallel, from its GGUF header (remote,
//! cached after the first fetch) or, offline, from the sizes stored in the
//! catalog. NO FIT drops out; the rest rank by usable speed (≥ 5 tok/s)
//! first, then quality tier, then predicted decode tok/s.

use serde::Deserialize;

use crate::descriptor::{Descriptor, WeightGroup};
use crate::gguf::Reader;
use crate::planner::PlacementPlan;
use crate::remote::{Fetcher, parse_model_ref};
use crate::speed::{HwSpec, estimate_speed_hybrid};
use crate::verdict::{FitConfig, FitReport, Verdict, check_fit};

const CATALOG: &str = include_str!("catalog.toml");

/// Below this a model fits but is slow to use (the `DecodeSlow` bar).
pub const USABLE_TOKS_PER_SEC: f64 = 5.0;

/// Values `--use` accepts.
pub const USES: [&str; 4] = ["chat", "code", "vision", "reasoning"];

/// One model in the embedded catalog.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct CatalogEntry {
    pub name: String,
    /// Exact `hf:<repo>:<file.gguf>` to pull.
    #[serde(rename = "ref")]
    pub model: String,
    pub uses: Vec<String>,
    /// 1 (small) … 4 (best in the catalog).
    pub tier: u8,
    /// File size in bytes.
    pub size: u64,
    /// KV cache MiB per 1024 ctx tokens at f16 (P10.7; measured per entry,
    /// see the `catalog.toml` header). Scales linearly with ctx; `q8_0`
    /// halves it, `q4_0` quarters it.
    #[serde(default)]
    pub kv_mib_per_1k: f64,
    /// Compute-buffer MiB at ubatch 512 (P10.7); linear in `n_ubatch`.
    #[serde(default)]
    pub compute_mib: f64,
    /// Approximate bytes read per token (MoE); `None` = `size`.
    #[serde(default)]
    pub active: Option<u64>,
    /// Projector file size (vision models), 0 when there is none.
    #[serde(default)]
    pub mmproj: u64,
    /// Extra `ref`s of a split (sharded) GGUF (P10.10): part 2..N.
    /// `ref` stays part 1. `size` must equal the parts' byte sum
    /// (Hub blob sizes); the probe sums tensor bytes across all part
    /// headers. Empty (the default) = single-file entry.
    #[serde(default)]
    pub parts: Vec<String>,
}

#[derive(Deserialize)]
struct Catalog {
    model: Vec<CatalogEntry>,
}

/// The embedded catalog (`catalog.toml`).
pub fn catalog() -> Vec<CatalogEntry> {
    toml::from_str::<Catalog>(CATALOG)
        .expect("embedded catalog.toml parses (see catalog_is_sane)")
        .model
}

/// One fitted catalog entry.
#[derive(Debug, Clone, PartialEq)]
pub struct Pick {
    pub entry: CatalogEntry,
    pub verdict: Verdict,
    pub decode_toks_per_sec: f64,
    /// From catalog sizes only (`--offline`), not the header.
    pub estimated: bool,
}

impl Pick {
    fn usable(&self) -> bool {
        self.decode_toks_per_sec >= USABLE_TOKS_PER_SEC
    }
}

/// Probe every entry for `use_` in parallel and rank what fits. Returns the
/// top `top` picks and `(name, error)` for each probe that failed.
pub fn recommend(
    entries: &[CatalogEntry],
    use_: Option<&str>,
    top: usize,
    probe: impl Fn(&CatalogEntry) -> Result<Pick, String> + Sync,
) -> (Vec<Pick>, Vec<(String, String)>) {
    let wanted: Vec<&CatalogEntry> = entries
        .iter()
        .filter(|e| use_.is_none_or(|u| e.uses.iter().any(|x| x == u)))
        .collect();
    let probe = &probe;
    let results: Vec<Result<Pick, String>> = std::thread::scope(|s| {
        let handles: Vec<_> = wanted.iter().map(|&e| s.spawn(move || probe(e))).collect();
        handles
            .into_iter()
            .map(|h| h.join().unwrap_or_else(|_| Err("probe panicked".into())))
            .collect()
    });
    let mut picks = Vec::new();
    let mut errors = Vec::new();
    for (e, r) in wanted.iter().zip(results) {
        match r {
            Ok(p) if p.verdict != Verdict::NoFit => picks.push(p),
            Ok(_) => {}
            Err(msg) => errors.push((e.name.clone(), msg)),
        }
    }
    picks.sort_by(|a, b| {
        b.usable()
            .cmp(&a.usable())
            .then(b.entry.tier.cmp(&a.entry.tier))
            .then(b.decode_toks_per_sec.total_cmp(&a.decode_toks_per_sec))
    });
    picks.truncate(top);
    (picks, errors)
}

/// Decode tok/s for the placement the verdict chose.
pub fn decode_for(desc: &Descriptor, report: &FitReport, config: &FitConfig) -> f64 {
    let gpu = report.speed_gpu.as_ref().map(|s| s.decode_toks_per_sec);
    let cpu = report.speed_cpu.as_ref().map(|s| s.decode_toks_per_sec);
    let speed = match report.verdict {
        Verdict::Gpu => gpu.or(cpu),
        Verdict::Cpu | Verdict::NoFit => cpu,
        Verdict::Hybrid { .. } => {
            // GPU share of the bytes read per token: experts parked on
            // CPU contribute only their used fraction (see below).
            let frac = hybrid_gpu_fraction(desc, &report.plan);
            config
                .gpu_hw
                .as_ref()
                .map(|hw| {
                    estimate_speed_hybrid(
                        desc,
                        &report.kv_estimate,
                        config.planner.ctx_len,
                        1024,
                        hw,
                        &config.cpu_hw,
                        frac,
                    )
                    .decode_toks_per_sec
                })
                .or(cpu)
        }
    };
    speed.unwrap_or(0.0)
}

/// GPU share of the bytes read per token for a hybrid placement.
///
/// MoE experts parked on CPU are mostly unread per token (only
/// `n_expert_used / n_expert` fire), so the share is over active bytes,
/// split per tensor: dense and embed/out count fully, experts count their
/// used fraction. Dense models and plans without a tensor table fall back
/// to the all-bytes share (identical there).
fn hybrid_gpu_fraction(desc: &Descriptor, plan: &PlacementPlan) -> f64 {
    let all_bytes = || {
        let total = plan.gpu_weight_bytes + plan.cpu_weight_bytes;
        plan.gpu_weight_bytes as f64 / total.max(1) as f64
    };
    if desc.n_expert == 0 || desc.n_expert_used == 0 || plan.tensors.is_empty() {
        return all_bytes();
    }
    let expert_frac = desc.n_expert_used as f64 / desc.n_expert as f64;
    let (mut gpu, mut cpu) = (0u64, 0u64);
    for t in &plan.tensors {
        let active = match t.group {
            WeightGroup::Expert => (t.bytes as f64 * expert_frac) as u64,
            WeightGroup::Dense | WeightGroup::EmbedOut => t.bytes,
        };
        if t.on_gpu {
            gpu += active;
        } else {
            cpu += active;
        }
    }
    if gpu + cpu == 0 {
        return all_bytes();
    }
    gpu as f64 / (gpu + cpu) as f64
}

/// Sum sharded weight bytes into the part-1 descriptor (P10.10).
/// GGUF shards repeat the full metadata and split the tensor table, so
/// arch/layers come from part 1 while the four weight groups add up.
pub fn combine_shard_weights(first: &mut Descriptor, rest: &[Descriptor]) {
    for d in rest {
        first.weight_bytes_dense += d.weight_bytes_dense;
        first.weight_bytes_expert += d.weight_bytes_expert;
        first.weight_bytes_embed_out += d.weight_bytes_embed_out;
        first.weight_bytes_total += d.weight_bytes_total;
    }
}

/// Fit one entry from its GGUF header(s) — every part for split models —
/// (the fetcher caches headers on disk).
pub fn probe_remote(
    fetcher: &Fetcher,
    entry: &CatalogEntry,
    config: &FitConfig,
) -> Result<Pick, String> {
    let mut descs = Vec::new();
    for r in entry.all_refs() {
        let src = parse_model_ref(r).map_err(|e| e.to_string())?;
        let header = fetcher.fetch_header(&src).map_err(|e| e.to_string())?;
        let reader = Reader::parse(&header.bytes).map_err(|e| e.to_string())?;
        descs.push(Descriptor::from_reader(&reader).map_err(|e| e.to_string())?);
    }
    let mut parts = descs.into_iter();
    let mut desc = parts
        .next()
        .ok_or_else(|| format!("{}: no refs", entry.name))?;
    let rest: Vec<Descriptor> = parts.collect();
    combine_shard_weights(&mut desc, &rest);
    let report = check_fit(&desc, config);
    Ok(Pick {
        entry: entry.clone(),
        decode_toks_per_sec: decode_for(&desc, &report, config),
        verdict: report.verdict,
        estimated: false,
    })
}

impl CatalogEntry {
    /// All file refs of this entry: `ref` (part 1) plus `parts`.
    pub fn all_refs(&self) -> Vec<&str> {
        std::iter::once(self.model.as_str())
            .chain(self.parts.iter().map(String::as_str))
            .collect()
    }

    /// KV bytes for `ctx_len` under `kv_type` (`f16` | `q8_0` | `q4_0`;
    /// unknown types keep the f16 number, never silently shrink it).
    pub fn kv_bytes_for(&self, ctx_len: u64, kv_type: &str) -> u64 {
        let quant_factor = match kv_type.to_ascii_lowercase().as_str() {
            "q8_0" | "q8" => 0.5,
            "q4_0" | "q4" => 0.25,
            _ => 1.0,
        };
        (self.kv_mib_per_1k.max(0.0) * ctx_len as f64 / 1024.0 * quant_factor * 1048576.0).ceil()
            as u64
    }

    /// Compute-buffer bytes for `n_ubatch` (linear from the ubatch-512
    /// reference, mirroring `estimate_compute`'s scaling).
    pub fn compute_bytes_for(&self, n_ubatch: u64) -> u64 {
        (self.compute_mib.max(0.0) * n_ubatch.max(1) as f64 / 512.0 * 1048576.0).ceil() as u64
    }
}

/// Fit one entry from catalog sizes: weights + projector + KV(ctx) +
/// compute(ubatch) against VRAM minus its margin, else RAM (P10.7 —
/// the header used to be the only path that counted KV/compute, so a
/// long context on a big model fit offline but not online).
pub fn probe_offline(entry: &CatalogEntry, config: &FitConfig) -> Pick {
    let p = &config.planner;
    let kv = entry.kv_bytes_for(p.ctx_len, &p.kv_type);
    let compute = entry.compute_bytes_for(p.n_ubatch);
    let need = entry.size + p.mmproj_bytes + kv + compute;
    let active = entry.active.unwrap_or(entry.size).max(1) as f64;
    // Same bytes-per-token shape as the online speed model: active
    // weights plus the KV read at half context.
    let per_token = active + kv as f64 / 2.0;
    let speed = |hw: &HwSpec| hw.efficiency * hw.bandwidth_gbps * 1e9 / per_token;
    let (verdict, decode) = match &config.gpu_hw {
        Some(hw) if need + p.vram_margin <= p.vram_bytes => (Verdict::Gpu, speed(hw)),
        _ if need <= p.ram_bytes => (Verdict::Cpu, speed(&config.cpu_hw)),
        _ => (Verdict::NoFit, 0.0),
    };
    Pick {
        entry: entry.clone(),
        verdict,
        decode_toks_per_sec: decode,
        estimated: true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::planner::PlannerConfig;

    fn entry(name: &str, tier: u8, uses: &[&str]) -> CatalogEntry {
        CatalogEntry {
            name: name.into(),
            model: format!("hf:x/{name}:{name}.gguf"),
            uses: uses.iter().map(|u| u.to_string()).collect(),
            tier,
            size: 4 << 30,
            active: None,
            mmproj: 0,
            kv_mib_per_1k: 0.0,
            compute_mib: 0.0,
            parts: Vec::new(),
        }
    }

    #[test]
    fn catalog_is_sane() {
        let c = catalog();
        assert!(c.len() >= 10);
        for e in &c {
            assert!(
                e.model.starts_with("hf:") && e.model.ends_with(".gguf"),
                "{e:?}"
            );
            assert!((1..=4).contains(&e.tier), "{e:?}");
            assert!(!e.uses.is_empty() && e.uses.iter().all(|u| USES.contains(&u.as_str())));
            assert!(e.size > 0 && e.active.is_none_or(|a| a < e.size), "{e:?}");
            // P10.7: every entry carries measured KV + compute numbers.
            assert!(e.kv_mib_per_1k > 0.0, "{e:?}");
            assert!(e.compute_mib > 0.0, "{e:?}");
            // P10.10: split parts are extra single-file refs, never equal
            // to part 1 and never duplicated.
            for p in e.all_refs() {
                assert!(p.starts_with("hf:") && p.ends_with(".gguf"), "{e:?}");
            }
            assert_eq!(
                e.all_refs().len(),
                e.all_refs()
                    .iter()
                    .collect::<std::collections::HashSet<_>>()
                    .len(),
                "{e:?}"
            );
            assert_eq!(c.iter().filter(|x| x.name == e.name).count(), 1, "{e:?}");
        }
        for u in USES {
            assert!(c.iter().any(|e| e.uses.iter().any(|x| x == u)), "{u}");
        }
    }

    #[test]
    fn ranks_usable_then_tier_then_speed() {
        let entries = [
            entry("slow-best", 4, &["chat"]),
            entry("mid", 3, &["chat"]),
            entry("mid-fast", 3, &["chat", "code"]),
            entry("too-big", 4, &["chat"]),
            entry("broken", 4, &["chat"]),
            entry("coder", 2, &["code"]),
        ];
        let fake = |e: &CatalogEntry| {
            let (verdict, tps) = match e.name.as_str() {
                "slow-best" => (Verdict::Cpu, 3.0),
                "mid" => (Verdict::Gpu, 20.0),
                "mid-fast" => (Verdict::Gpu, 40.0),
                "too-big" => (Verdict::NoFit, 0.0),
                "broken" => return Err("http 404".to_string()),
                _ => (Verdict::Gpu, 90.0),
            };
            Ok(Pick {
                entry: e.clone(),
                verdict,
                decode_toks_per_sec: tps,
                estimated: false,
            })
        };
        let names = |picks: &[Pick]| {
            picks
                .iter()
                .map(|p| p.entry.name.clone())
                .collect::<Vec<_>>()
        };

        let (picks, errors) = recommend(&entries, Some("chat"), 10, fake);
        assert_eq!(names(&picks), ["mid-fast", "mid", "slow-best"]);
        assert_eq!(errors, [("broken".to_string(), "http 404".to_string())]);

        let (picks, _) = recommend(&entries, None, 2, fake);
        assert_eq!(names(&picks), ["mid-fast", "mid"]);

        let (picks, errors) = recommend(&entries, Some("code"), 10, fake);
        assert_eq!(names(&picks), ["mid-fast", "coder"]);
        assert!(errors.is_empty());
    }

    #[test]
    fn offline_fits_on_sizes() {
        let config = |vram_gib: u64, ram_gib: u64| FitConfig {
            planner: PlannerConfig {
                vram_bytes: vram_gib << 30,
                ram_bytes: ram_gib << 30,
                ..PlannerConfig::default()
            },
            gpu_hw: Some(HwSpec::metal()),
            ..FitConfig::default()
        };
        let dense = entry("dense", 3, &["chat"]);
        let moe = CatalogEntry {
            active: Some(1 << 30),
            ..dense.clone()
        };
        assert_eq!(probe_offline(&dense, &config(8, 16)).verdict, Verdict::Gpu);
        assert_eq!(probe_offline(&dense, &config(4, 16)).verdict, Verdict::Cpu);
        assert_eq!(probe_offline(&dense, &config(4, 2)).verdict, Verdict::NoFit);
        let d = probe_offline(&dense, &config(8, 16)).decode_toks_per_sec;
        let m = probe_offline(&moe, &config(8, 16)).decode_toks_per_sec;
        assert!(m > 3.0 * d, "{m} vs {d}");
    }

    #[test]
    fn offline_counts_kv_and_compute() {
        // P10.7: 4 GiB weights alone fit a 7 GiB GPU budget, but not with
        // 4 GiB of KV on top (ctx 4096 × 1024 MiB/1k) — it drops to CPU.
        // With 8 GiB of KV it fits nowhere.
        let config = |vram_gib: u64, ram_gib: u64| FitConfig {
            planner: PlannerConfig {
                vram_bytes: vram_gib << 30,
                ram_bytes: ram_gib << 30,
                ..PlannerConfig::default()
            },
            gpu_hw: Some(HwSpec::metal()),
            ..FitConfig::default()
        };
        let heavy_kv = CatalogEntry {
            kv_mib_per_1k: 1024.0,
            ..entry("heavy", 3, &["chat"])
        };
        assert_eq!(
            probe_offline(&heavy_kv, &config(8, 32)).verdict,
            Verdict::Cpu
        );
        let too_heavy = CatalogEntry {
            kv_mib_per_1k: 2048.0,
            compute_mib: 512.0,
            ..entry("too-heavy", 3, &["chat"])
        };
        assert_eq!(
            probe_offline(&too_heavy, &config(8, 12)).verdict,
            Verdict::NoFit
        );
        // KV quantization shrinks the offline need like the online one.
        let mut q8 = config(8, 32);
        q8.planner.kv_type = "q8_0".into();
        assert_eq!(probe_offline(&heavy_kv, &q8).verdict, Verdict::Gpu);
        // Scaling helpers: linear in ctx / ubatch, floored at zero.
        assert_eq!(heavy_kv.kv_bytes_for(4096, "f16"), 4 << 30);
        assert_eq!(heavy_kv.kv_bytes_for(4096, "q8_0"), 2 << 30);
        assert_eq!(heavy_kv.kv_bytes_for(4096, "q4_0"), 1 << 30);
        assert_eq!(heavy_kv.kv_bytes_for(4096, "weird"), 4 << 30);
        let c = CatalogEntry {
            compute_mib: 512.0,
            ..entry("c", 1, &["chat"])
        };
        assert_eq!(c.compute_bytes_for(512), 512 << 20);
        assert_eq!(c.compute_bytes_for(1024), 1024 << 20);
    }

    fn moe_desc() -> Option<Descriptor> {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/qwen2-0_5b-instruct-q4_0.gguf");
        let header = crate::read_local_prefix(&path).ok()?;
        let reader = Reader::parse(&header.bytes).ok()?;
        let mut d = Descriptor::from_reader(&reader).ok()?;
        d.n_expert = 8;
        d.n_expert_used = 2;
        Some(d)
    }

    /// Experts-parked-on-CPU plan: dense 100 + embed 50 on GPU, 400 of
    /// experts on CPU (the planner evicts experts first).
    fn cpu_expert_plan() -> crate::planner::PlacementPlan {
        use crate::descriptor::WeightGroup;
        use crate::planner::TensorPlacement;
        let t = |name: &str, bytes: u64, group: WeightGroup, on_gpu: bool| TensorPlacement {
            name: name.into(),
            bytes,
            group,
            on_gpu,
        };
        crate::planner::PlacementPlan {
            gpu_weight_bytes: 150,
            cpu_weight_bytes: 400,
            compute_buffer_bytes: 0,
            kv_bytes: 0,
            mmproj_bytes: 0,
            lora_bytes: 0,
            gpu_total_bytes: 150,
            encoder_compute_bytes: 0,
            tensors: vec![
                t("blk.0.attn_q.weight", 100, WeightGroup::Dense, true),
                t("token_embd.weight", 50, WeightGroup::EmbedOut, true),
                t(
                    "blk.0.ffn_gate_exps.weight",
                    400,
                    WeightGroup::Expert,
                    false,
                ),
            ],
            gpu_layers: 0,
            cpu_layers: 1,
            all_on_gpu: false,
            fits: true,
        }
    }

    #[test]
    fn hybrid_fraction_counts_active_bytes() {
        // 2/8 experts used: active GPU = 150, active CPU = 100 → 0.60,
        // while the all-bytes share is 150/550 ≈ 0.27.
        let plan = cpu_expert_plan();
        let Some(moe) = moe_desc() else {
            eprintln!("fixture not present; skipping");
            return;
        };
        let frac = hybrid_gpu_fraction(&moe, &plan);
        assert!((frac - 0.6).abs() < 1e-9, "{frac}");
        // Dense models keep the all-bytes share (active == total).
        let mut dense = moe.clone();
        dense.n_expert = 0;
        dense.n_expert_used = 0;
        let old =
            plan.gpu_weight_bytes as f64 / (plan.gpu_weight_bytes + plan.cpu_weight_bytes) as f64;
        assert!((hybrid_gpu_fraction(&dense, &plan) - old).abs() < 1e-12);
        // Plans without a tensor table fall back to the all-bytes share.
        let mut bare = plan.clone();
        bare.tensors.clear();
        assert!((hybrid_gpu_fraction(&moe, &bare) - old).abs() < 1e-12);
    }

    #[test]
    fn hybrid_decode_uses_active_share() {
        let Some(desc) = moe_desc() else {
            eprintln!("fixture not present; skipping");
            return;
        };
        let config = FitConfig::default();
        let gpu_hw = config.gpu_hw.clone().expect("default has GPU hw");
        let mut report = check_fit(&desc, &config);
        report.verdict = Verdict::Hybrid {
            gpu_layers: 1,
            total_layers: 24,
        };
        report.plan = cpu_expert_plan();
        let fixed = decode_for(&desc, &report, &config);
        let at = |frac: f64| {
            estimate_speed_hybrid(
                &desc,
                &report.kv_estimate,
                config.planner.ctx_len,
                1024,
                &gpu_hw,
                &config.cpu_hw,
                frac,
            )
            .decode_toks_per_sec
        };
        // The old all-bytes share (150/550) over-weights the slow CPU
        // side; the active share (0.60) predicts ~1.7x faster here.
        let old = at(150.0 / 550.0);
        assert!(fixed > 1.5 * old, "fixed {fixed} vs all-bytes {old}");
        assert!((fixed - at(0.6)).abs() < 1e-6, "{fixed}");
    }
}

#[cfg(test)]
mod split_tests {
    use super::*;
    use std::path::PathBuf;

    /// Real arch metadata (qwen2 fixture header) with overridden weight
    /// groups — no third copy of the synthetic-GGUF builder.
    /// Real arch metadata with overridden weight groups. `None` when
    /// the git-ignored fixture is absent (CI `--lib` runs before the
    /// fixture fetch; same skip convention as
    /// `descriptor::tests::parses_real_qwen2_fixture`).
    fn desc_with_weights(dense: u64, expert: u64, embed: u64) -> Option<Descriptor> {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/qwen2-0_5b-instruct-q4_0.gguf");
        let header = crate::read_local_prefix(&path).ok()?;
        let reader = Reader::parse(&header.bytes).ok()?;
        let mut d = Descriptor::from_reader(&reader).ok()?;
        d.weight_bytes_dense = dense;
        d.weight_bytes_expert = expert;
        d.weight_bytes_embed_out = embed;
        d.weight_bytes_total = dense + expert + embed;
        Some(d)
    }

    #[test]
    fn all_refs_orders_part_one_first() {
        let single = CatalogEntry {
            name: "s".into(),
            model: "hf:x/s:s.gguf".into(),
            uses: vec!["chat".into()],
            tier: 1,
            size: 1,
            active: None,
            mmproj: 0,
            kv_mib_per_1k: 0.0,
            compute_mib: 0.0,
            parts: vec!["hf:x/s:s-00002-of-00003.gguf".into()],
        };
        assert_eq!(
            single.all_refs(),
            ["hf:x/s:s.gguf", "hf:x/s:s-00002-of-00003.gguf"]
        );
    }

    #[test]
    fn split_parts_combine_weights() {
        // P10.10: shards repeat metadata, split tensors — groups add up,
        // arch comes from part 1.
        let (Some(mut first), Some(second)) = (
            desc_with_weights(100, 200, 50),
            desc_with_weights(300, 400, 60),
        ) else {
            eprintln!("fixture not present; skipping");
            return;
        };
        let layers = first.n_layer;
        combine_shard_weights(&mut first, &[second]);
        assert_eq!(first.weight_bytes_dense, 400);
        assert_eq!(first.weight_bytes_expert, 600);
        assert_eq!(first.weight_bytes_embed_out, 110);
        assert_eq!(first.weight_bytes_total, 1110);
        assert_eq!(first.n_layer, layers);
    }
}
