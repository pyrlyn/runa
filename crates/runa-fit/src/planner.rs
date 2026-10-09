// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! Placement planner: GPU/CPU tensor placement (P1.8).
//!
//! Given a model descriptor, available VRAM, and configuration, decide which
//! tensors (weights + compute buffer) go on GPU vs CPU. The goal is to
//! maximize GPU-resident weights under `VRAM − margin` while respecting
//! architectural constraints.
//!
//! # Eviction priority (first to CPU when VRAM is tight)
//!
//! 1. **MoE expert weights** — `ffn_*_exps.*`; these are redundant across
//!    experts and can be swapped in/out with minimal latency impact.
//! 2. **Embedding + output tables** — `token_embd`, `output`; large but
//!    only touched at sequence boundaries.
//! 3. **Normalization layers** — tiny tensors, low priority for GPU.
//! 4. **Dense weights** (attention/FFN) — last to evict; critical for
//!    decode throughput.
//!
//! # Compute buffer
//!
//! The compute buffer (P1.5) is always on GPU when any layer is on GPU,
//! as it holds activations for the current micro-batch. Its size is
//! estimated via `estimate_compute`.

use crate::compute::estimate_compute;
use crate::descriptor::{Descriptor, WeightGroup};
use crate::kv::estimate_kv;

/// Placement decision for a single tensor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorPlacement {
    pub name: String,
    pub bytes: u64,
    pub group: WeightGroup,
    pub on_gpu: bool,
}

/// Complete placement plan.
#[derive(Debug, Clone, PartialEq)]
pub struct PlacementPlan {
    /// Total GPU weight bytes (excluding compute buffer).
    pub gpu_weight_bytes: u64,
    /// Total CPU weight bytes.
    pub cpu_weight_bytes: u64,
    /// Compute buffer bytes (always on GPU when any layer is on GPU).
    pub compute_buffer_bytes: u64,
    /// KV cache bytes (can be split across GPU/CPU).
    pub kv_bytes: u64,
    /// Multimodal projector bytes (P1.8/P4.8; 0 for text-only models).
    /// Reserved on the same device as the compute buffer.
    pub mmproj_bytes: u64,
    /// LoRA adapter bytes (P8.5; 0 when no `--lora`). Sum of the adapter
    /// GGUF file sizes; reserved on the same device as the compute buffer.
    pub lora_bytes: u64,
    /// GPU memory total (weights + compute + KV + mmproj + lora + encoder).
    pub gpu_total_bytes: u64,
    /// Vision/audio encoder scratch bytes (P4.8).
    pub encoder_compute_bytes: u64,
    /// Per-tensor placements.
    pub tensors: Vec<TensorPlacement>,
    /// Number of layers fully on GPU.
    pub gpu_layers: u64,
    /// Number of layers fully on CPU.
    pub cpu_layers: u64,
    /// Whether all weights fit on GPU.
    pub all_on_gpu: bool,
    /// Whether the model fits at all (GPU total ≤ VRAM).
    pub fits: bool,
}

/// Configuration for the placement planner.
#[derive(Debug, Clone)]
pub struct PlannerConfig {
    /// Available VRAM in bytes.
    pub vram_bytes: u64,
    /// Available system RAM in bytes (for CPU offload).
    pub ram_bytes: u64,
    /// Reserved VRAM margin in bytes (default 1 GiB).
    pub vram_margin: u64,
    /// Micro-batch size for compute buffer estimation.
    pub n_ubatch: u64,
    /// KV quantization type string (e.g. "f16", "q8_0").
    pub kv_type: String,
    /// Context length for KV cache estimation.
    pub ctx_len: u64,
    /// Multimodal projector size in bytes (0 for text-only models).
    pub mmproj_bytes: u64,
    /// LoRA adapter sizes summed in bytes (P8.5; 0 without `--lora`).
    pub lora_bytes: u64,
    /// Vision/audio encoder scratch (P4.8); reserved on GPU with compute.
    pub encoder_compute_bytes: u64,
}

impl Default for PlannerConfig {
    fn default() -> Self {
        Self {
            vram_bytes: 8 * 1024 * 1024 * 1024, // 8 GiB
            ram_bytes: 32 * 1024 * 1024 * 1024, // 32 GiB
            vram_margin: 1024 * 1024 * 1024,    // 1 GiB
            n_ubatch: 512,
            kv_type: "f16".to_string(),
            ctx_len: 4096,
            mmproj_bytes: 0,
            lora_bytes: 0,
            encoder_compute_bytes: 0,
        }
    }
}

/// Compute the eviction priority for a tensor group.
///
/// Lower number = evict first (MoE experts are evicted first).
fn eviction_priority(group: WeightGroup) -> u32 {
    match group {
        WeightGroup::Expert => 0,
        WeightGroup::EmbedOut => 1,
        WeightGroup::Dense => 2,
    }
}

/// Plan tensor placement given a descriptor and config.
pub fn plan_placement(desc: &Descriptor, config: &PlannerConfig) -> PlacementPlan {
    let available_vram = config.vram_bytes.saturating_sub(config.vram_margin);

    // Compute buffer: always on GPU if any layer is on GPU.
    let compute_est = estimate_compute(desc, config.n_ubatch);
    let compute_buffer = compute_est.compute_bytes;

    // KV cache: estimate for the given context.
    let kv_est = estimate_kv(desc, config.ctx_len, &config.kv_type);
    let kv_bytes = kv_est.kv_bytes;

    // Reserve compute + KV + mmproj + LoRA + encoder scratch on GPU.
    let reserved = compute_buffer
        + kv_bytes
        + config.mmproj_bytes
        + config.lora_bytes
        + config.encoder_compute_bytes;
    let weight_budget = available_vram.saturating_sub(reserved);

    // Collect tensors and sort by eviction priority (highest priority first).
    let mut tensors: Vec<TensorPlacement> = desc
        .tensor_bytes
        .iter()
        .map(|(name, &(bytes, group))| TensorPlacement {
            name: name.clone(),
            bytes,
            group,
            on_gpu: false, // will be set below
        })
        .collect();

    // Sort: pack Dense first (last to evict), EmbedOut next, Experts last (evicted first).
    tensors.sort_by(|a, b| {
        eviction_priority(b.group)
            .cmp(&eviction_priority(a.group))
            .then_with(|| b.bytes.cmp(&a.bytes)) // larger first within group
    });

    // Fill GPU: iterate in eviction order, pack what fits.
    let mut gpu_bytes = 0u64;
    for t in &mut tensors {
        if gpu_bytes + t.bytes <= weight_budget {
            t.on_gpu = true;
            gpu_bytes += t.bytes;
        }
    }

    let cpu_bytes: u64 = tensors.iter().filter(|t| !t.on_gpu).map(|t| t.bytes).sum();

    let gpu_total = gpu_bytes
        + compute_buffer
        + kv_bytes
        + config.mmproj_bytes
        + config.lora_bytes
        + config.encoder_compute_bytes;
    // The CPU side must fit RAM too: weights parked on CPU, plus
    // compute/KV/mmproj/lora/encoder scratch when nothing lives on the
    // GPU at all. Without this a GPU-less box can never fit (vram is 0 there),
    // which wrongly rejects CPU-only serve (plan D4 promises FITS CPU).
    let cpu_need = cpu_bytes
        + if gpu_bytes == 0 {
            compute_buffer
                + kv_bytes
                + config.mmproj_bytes
                + config.lora_bytes
                + config.encoder_compute_bytes
        } else {
            0
        };
    let fits = (gpu_bytes == 0 || gpu_total <= config.vram_bytes) && cpu_need <= config.ram_bytes;

    // Count fully-on-GPU vs fully-on-CPU layers.
    // A layer is "on GPU" if ALL its tensors are on GPU.
    let mut gpu_layers = 0u64;
    let mut cpu_layers = 0u64;
    if desc.n_layer > 0 {
        // Check first layer as a proxy (all layers have same structure).
        let blk_tensors: Vec<&TensorPlacement> = tensors
            .iter()
            .filter(|t| t.name.starts_with("blk.0."))
            .collect();
        if !blk_tensors.is_empty() && blk_tensors.iter().all(|t| t.on_gpu) {
            gpu_layers = desc.n_layer;
        } else {
            cpu_layers = desc.n_layer;
        }
    }

    let all_on_gpu = tensors.iter().all(|t| t.on_gpu);

    PlacementPlan {
        gpu_weight_bytes: gpu_bytes,
        cpu_weight_bytes: cpu_bytes,
        compute_buffer_bytes: compute_buffer,
        kv_bytes,
        mmproj_bytes: config.mmproj_bytes,
        lora_bytes: config.lora_bytes,
        encoder_compute_bytes: config.encoder_compute_bytes,
        gpu_total_bytes: gpu_total,
        tensors,
        gpu_layers,
        cpu_layers,
        all_on_gpu,
        fits,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::{GGUF_MAGIC, GgmlType};

    fn str_value(s: &str) -> Vec<u8> {
        let mut v = (s.len() as u64).to_le_bytes().to_vec();
        v.extend_from_slice(s.as_bytes());
        v
    }
    fn u64_value(v: u64) -> Vec<u8> {
        v.to_le_bytes().to_vec()
    }

    fn write_gguf(
        kv_pairs: &[(&str, u32, Vec<u8>)],
        tensors: &[(&str, &[u64], GgmlType)],
    ) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&GGUF_MAGIC);
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        b.extend_from_slice(&(kv_pairs.len() as u64).to_le_bytes());
        for &(key, ty, ref val) in kv_pairs {
            b.extend_from_slice(&(key.len() as u64).to_le_bytes());
            b.extend_from_slice(key.as_bytes());
            b.extend_from_slice(&ty.to_le_bytes());
            b.extend_from_slice(val);
        }
        for &(name, dims, ty) in tensors {
            b.extend_from_slice(&(name.len() as u64).to_le_bytes());
            b.extend_from_slice(name.as_bytes());
            b.extend_from_slice(&(dims.len() as u32).to_le_bytes());
            for &d in dims {
                b.extend_from_slice(&d.to_le_bytes());
            }
            let ty_raw: u32 = match ty {
                GgmlType::F32 => 0,
                GgmlType::F16 => 1,
                GgmlType::Q4_0 => 2,
                GgmlType::Q8_0 => 8,
                GgmlType::Q4_K => 12,
                _ => 0,
            };
            b.extend_from_slice(&ty_raw.to_le_bytes());
            b.extend_from_slice(&0u64.to_le_bytes());
        }
        while b.len() % 32 != 0 {
            b.push(0);
        }
        b
    }

    fn make_moe_desc() -> Descriptor {
        let kv = vec![
            ("general.architecture".to_string(), 8, str_value("qwen3moe")),
            ("qwen3moe.block_count".to_string(), 10, u64_value(48)),
            ("qwen3moe.embedding_length".to_string(), 10, u64_value(2048)),
            (
                "qwen3moe.attention.head_count".to_string(),
                10,
                u64_value(16),
            ),
            ("qwen3moe.context_length".to_string(), 10, u64_value(4096)),
            ("qwen3moe.vocab_size".to_string(), 10, u64_value(151936)),
            ("qwen3moe.expert_count".to_string(), 10, u64_value(128)),
            ("qwen3moe.expert_used_count".to_string(), 10, u64_value(8)),
            (
                "qwen3moe.attention.head_count_kv".to_string(),
                10,
                u64_value(2),
            ),
            (
                "qwen3moe.attention.key_length".to_string(),
                10,
                u64_value(128),
            ),
        ];
        let embd_dims = [151936, 2048];
        let tensors = vec![
            ("token_embd.weight", embd_dims.as_slice(), GgmlType::Q4_K),
            (
                "blk.0.ffn_gate_exps.weight",
                &[256, 2048][..],
                GgmlType::Q4_K,
            ),
            ("blk.0.ffn_up_exps.weight", &[256, 2048][..], GgmlType::Q4_K),
            (
                "blk.0.ffn_down_exps.weight",
                &[2048, 256][..],
                GgmlType::Q4_K,
            ),
            ("blk.0.attn_q.weight", &[2048, 2048][..], GgmlType::Q4_K),
            ("output.weight", &[2048, 151936][..], GgmlType::Q4_K),
        ];
        let buf = write_gguf(
            &kv.iter()
                .map(|(k, t, v)| (k.as_str(), *t, v.clone()))
                .collect::<Vec<_>>(),
            &tensors,
        );
        let r = crate::gguf::Reader::parse(&buf).unwrap();
        Descriptor::from_reader(&r).unwrap()
    }

    #[test]
    fn huge_vram_all_on_gpu() {
        let d = make_moe_desc();
        let kv_est = estimate_kv(&d, 4096, "f16");
        let compute_est = estimate_compute(&d, 512);
        let total_needed = d.weight_bytes_total + kv_est.kv_bytes + compute_est.compute_bytes;
        let config = PlannerConfig {
            vram_bytes: total_needed * 2, // plenty of VRAM
            vram_margin: 0,
            ..Default::default()
        };
        let plan = plan_placement(&d, &config);
        assert!(plan.all_on_gpu);
        assert!(plan.fits);
    }

    #[test]
    fn zero_vram_all_on_cpu() {
        let d = make_moe_desc();
        let config = PlannerConfig {
            vram_bytes: 0,
            vram_margin: 0,
            ..Default::default()
        };
        let plan = plan_placement(&d, &config);
        assert!(!plan.all_on_gpu);
        assert_eq!(plan.gpu_weight_bytes, 0);
    }

    #[test]
    fn zero_vram_fits_when_ram_suffices() {
        // GPU-less box (CI CPU runners): everything parks on CPU and the
        // verdict must be a fit when RAM covers weights + KV + compute.
        let d = make_moe_desc();
        let kv_est = estimate_kv(&d, 4096, "f16");
        let compute_est = estimate_compute(&d, 512);
        let total_needed = d.weight_bytes_total + kv_est.kv_bytes + compute_est.compute_bytes;
        let config = PlannerConfig {
            vram_bytes: 0,
            vram_margin: 0,
            ram_bytes: total_needed,
            ..Default::default()
        };
        let plan = plan_placement(&d, &config);
        assert!(plan.fits);
    }

    #[test]
    fn zero_vram_zero_ram_no_fit() {
        let d = make_moe_desc();
        let config = PlannerConfig {
            vram_bytes: 0,
            vram_margin: 0,
            ram_bytes: 0,
            ..Default::default()
        };
        let plan = plan_placement(&d, &config);
        assert!(!plan.fits);
    }

    #[test]
    fn experts_evicted_first() {
        let d = make_moe_desc();
        let kv_est = estimate_kv(&d, 4096, "f16");
        let compute_est = estimate_compute(&d, 512);
        // VRAM = just enough for dense + embed_out + compute + kv, no room for experts.
        let weight_budget_needed = d.weight_bytes_dense + d.weight_bytes_embed_out;
        let vram_needed = weight_budget_needed + kv_est.kv_bytes + compute_est.compute_bytes;
        let config = PlannerConfig {
            vram_bytes: vram_needed, // exact fit, no room for experts
            vram_margin: 0,
            ..Default::default()
        };
        let plan = plan_placement(&d, &config);
        let expert_on_gpu: u64 = plan
            .tensors
            .iter()
            .filter(|t| t.group == WeightGroup::Expert && t.on_gpu)
            .map(|t| t.bytes)
            .sum();
        assert_eq!(expert_on_gpu, 0, "experts should be evicted first");
    }

    #[test]
    fn margin_reduces_budget() {
        let d = make_moe_desc();
        let total = d.weight_bytes_total;
        let kv_est = estimate_kv(&d, 4096, "f16");
        let compute_est = estimate_compute(&d, 512);
        let without_margin = PlannerConfig {
            vram_bytes: total + kv_est.kv_bytes + compute_est.compute_bytes,
            vram_margin: 0,
            ..Default::default()
        };
        let with_margin = PlannerConfig {
            vram_bytes: total + kv_est.kv_bytes + compute_est.compute_bytes,
            vram_margin: 1024 * 1024 * 1024, // 1 GiB margin
            ..Default::default()
        };
        let plan_no = plan_placement(&d, &without_margin);
        let plan_yes = plan_placement(&d, &with_margin);
        // With margin, GPU gets fewer bytes.
        assert!(plan_yes.gpu_weight_bytes <= plan_no.gpu_weight_bytes);
    }

    #[test]
    fn lora_bytes_reserve_gpu_budget() {
        let d = make_moe_desc();
        let kv_est = estimate_kv(&d, 4096, "f16");
        let compute_est = estimate_compute(&d, 512);
        // Tight budget: weights + kv + compute fit exactly with no LoRA.
        let vram = d.weight_bytes_total + kv_est.kv_bytes + compute_est.compute_bytes;
        let no_lora = plan_placement(
            &d,
            &PlannerConfig {
                vram_bytes: vram,
                vram_margin: 0,
                ..Default::default()
            },
        );
        assert!(no_lora.all_on_gpu);
        // A 64 MiB adapter must evict weights, exactly like a projector.
        let with_lora = plan_placement(
            &d,
            &PlannerConfig {
                vram_bytes: vram,
                vram_margin: 0,
                lora_bytes: 64 << 20,
                ..Default::default()
            },
        );
        assert_eq!(with_lora.lora_bytes, 64 << 20);
        assert!(with_lora.gpu_weight_bytes < no_lora.gpu_weight_bytes);
        assert!(with_lora.gpu_total_bytes <= vram);
    }

    #[test]
    fn fits_reports_correctly() {
        let d = make_moe_desc();
        let kv_est = estimate_kv(&d, 4096, "f16");
        let compute_est = estimate_compute(&d, 512);
        let total_needed = d.weight_bytes_total + kv_est.kv_bytes + compute_est.compute_bytes;
        let config = PlannerConfig {
            vram_bytes: total_needed + 1024, // just barely fits
            vram_margin: 0,
            ..Default::default()
        };
        let plan = plan_placement(&d, &config);
        assert!(plan.fits);
    }

    #[test]
    fn mmproj_reserves_gpu_budget() {
        let d = make_moe_desc();
        let kv_est = estimate_kv(&d, 4096, "f16");
        let compute_est = estimate_compute(&d, 512);
        // Tight budget: weights + kv + compute fit exactly with no mmproj.
        let vram = d.weight_bytes_total + kv_est.kv_bytes + compute_est.compute_bytes;
        let no_mmproj = plan_placement(
            &d,
            &PlannerConfig {
                vram_bytes: vram,
                vram_margin: 0,
                ..Default::default()
            },
        );
        assert!(no_mmproj.all_on_gpu);
        // A 64 MiB projector must evict weights (the synthetic model is
        // small, so the projector has to fit inside the weight budget).
        let with_mmproj = plan_placement(
            &d,
            &PlannerConfig {
                vram_bytes: vram,
                vram_margin: 0,
                mmproj_bytes: 64 << 20,
                ..Default::default()
            },
        );
        assert_eq!(with_mmproj.mmproj_bytes, 64 << 20);
        assert!(with_mmproj.gpu_weight_bytes < no_mmproj.gpu_weight_bytes);
        assert!(with_mmproj.gpu_total_bytes <= vram);
    }

    #[test]
    fn real_moe_fixture_small_vram_parks_experts_on_cpu() {
        // End-to-end over the real 30B-A3B GGUF *header* (tensor infos with
        // real expert byte sizes; weights truncated — ~6 MiB local fixture).
        // 8 GiB VRAM must keep attention/dense on GPU while experts spill to CPU.
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/Qwen3-30B-A3B-Q4_K_M.gguf");
        if !path.is_file() {
            // Fixture not present: don't fail the suite.
            return;
        }
        let h = crate::remote::read_local_prefix(&path).expect("fixture header");
        let r = crate::gguf::Reader::parse(&h.bytes).expect("parses");
        let d = Descriptor::from_reader(&r).expect("descriptor");
        assert!(d.n_expert > 1, "MoE fixture expected");

        let small = plan_placement(
            &d,
            &PlannerConfig {
                vram_bytes: 8 << 30,
                ..Default::default()
            },
        );
        let expert_gpu: u64 = small
            .tensors
            .iter()
            .filter(|t| t.group == WeightGroup::Expert && t.on_gpu)
            .map(|t| t.bytes)
            .sum();
        let expert_total: u64 = small
            .tensors
            .iter()
            .filter(|t| t.group == WeightGroup::Expert)
            .map(|t| t.bytes)
            .sum();
        assert!(
            expert_gpu < expert_total,
            "experts must spill with 8 GiB VRAM"
        );
        let dense_gpu: u64 = small
            .tensors
            .iter()
            .filter(|t| t.group == WeightGroup::Dense && t.on_gpu)
            .map(|t| t.bytes)
            .sum();
        assert!(dense_gpu > 0, "dense weights stay on GPU");

        let huge = plan_placement(
            &d,
            &PlannerConfig {
                vram_bytes: 1 << 40,
                vram_margin: 0,
                ..Default::default()
            },
        );
        assert!(huge.all_on_gpu);
    }
}
