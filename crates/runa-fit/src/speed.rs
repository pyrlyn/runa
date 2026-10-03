// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! Speed model: decode/prefill throughput and TTFT estimation (P1.9).
//!
//! Given a model descriptor, KV estimate, and hardware specs, predict
//! tokens/sec for decode and prefill, plus time-to-first-token (TTFT).
//!
//! # Formulas
//!
//! **Active weight bytes:**
//! ```text
//! active_bytes = dense + expert_used/expert × expert_bytes + embed_out
//! ```
//!
//! **Decode throughput (tok/s):**
//! ```text
//! decode = efficiency × bandwidth / bytes_per_token
//! ```
//!
//! **Hybrid decode (GPU+CPU):**
//! ```text
//! hybrid_decode = 1 / (bytes_gpu/BW_gpu + bytes_cpu/BW_cpu)
//! ```
//!
//! **Prefill throughput (tok/s):**
//! ```text
//! compute_bound = efficiency × FLOPS / (2 × active_params)
//! bandwidth_bound = efficiency × bandwidth / bytes_per_token
//! prefill = min(compute_bound, bandwidth_bound)
//! ```
//!
//! **TTFT:**
//! ```text
//! TTFT = prompt_tokens / prefill
//! ```
//!
//! Default efficiencies: CUDA 0.60, Metal 0.60, Vulkan 0.50, CPU 0.50.

use crate::calibration::Efficiency;
use crate::descriptor::Descriptor;
use crate::kv::KvEstimate;

/// Hardware specification for one compute device.
#[derive(Debug, Clone, PartialEq)]
pub struct HwSpec {
    /// Peak memory bandwidth in GB/s.
    pub bandwidth_gbps: f64,
    /// Peak compute in TFLOPS (for prefill). 0 = bandwidth-bound only.
    pub tflops: f64,
    /// Efficiency multiplier (0.0–1.0). Defaults per backend.
    pub efficiency: f64,
}

impl HwSpec {
    pub fn cuda() -> Self {
        Self {
            bandwidth_gbps: 900.0, // A100-class
            tflops: 312.0,
            efficiency: 0.60,
        }
    }
    pub fn metal() -> Self {
        Self {
            bandwidth_gbps: 800.0, // M2 Pro-class
            tflops: 150.0,
            efficiency: 0.60,
        }
    }
    pub fn vulkan() -> Self {
        Self {
            bandwidth_gbps: 500.0,
            tflops: 100.0,
            efficiency: 0.50,
        }
    }
    /// P9.4: Qualcomm Hexagon NPU (Snapdragon X-class, 45 INT8 TOPS).
    /// Conservative stub: shared-LPDDR slice for bandwidth, 0.30 efficiency
    /// (uncalibrated — no CI runners or on-device measurements yet, Tier 3).
    /// Text-oriented, Q4_0-centric estimates only; see `docs/fit.md`.
    pub fn hexagon() -> Self {
        Self {
            bandwidth_gbps: 60.0,
            tflops: 45.0,
            efficiency: 0.30,
        }
    }
    /// P9.4: Intel NPU via OpenVINO (Core Ultra / Lunar Lake class).
    /// Conservative stub: floor-SKU TOPS, 0.30 efficiency (uncalibrated —
    /// no CI runners or on-device measurements yet, Tier 3). Text-only
    /// models; see `docs/fit.md`.
    pub fn openvino() -> Self {
        Self {
            bandwidth_gbps: 40.0,
            tflops: 13.0,
            efficiency: 0.30,
        }
    }
    pub fn cpu() -> Self {
        Self {
            bandwidth_gbps: 50.0, // DDR5 dual-channel
            tflops: 50.0,
            efficiency: 0.50,
        }
    }
}

/// Speed estimation result.
#[derive(Debug, Clone, PartialEq)]
pub struct SpeedEstimate {
    /// Active weight bytes (what must be read per token).
    pub active_bytes: u64,
    /// Decode throughput (tokens/sec).
    pub decode_toks_per_sec: f64,
    /// Prefill throughput (tokens/sec).
    pub prefill_toks_per_sec: f64,
    /// Time to first token in seconds.
    pub ttft_secs: f64,
    /// Bytes assigned to GPU side (hybrid mode).
    pub gpu_bytes: u64,
    /// Bytes assigned to CPU side (hybrid mode).
    pub cpu_bytes: u64,
}

/// Compute active weight bytes from a descriptor.
///
/// For dense models: all weights. For MoE: shared + (n_expert_used/n_expert)
/// × expert weights + embedding/output.
pub fn active_weight_bytes(desc: &Descriptor) -> u64 {
    if desc.n_expert > 0 {
        // MoE: shared (dense minus expert) + proportional expert + embed_out
        let expert_fraction = desc.n_expert_used as f64 / desc.n_expert as f64;
        let expert_contribution = (desc.weight_bytes_expert as f64 * expert_fraction) as u64;
        desc.weight_bytes_dense + expert_contribution + desc.weight_bytes_embed_out
    } else {
        // Dense: everything
        desc.weight_bytes_total
    }
}

/// Scale raw prefill/decode predictions by a calibration efficiency
/// (P10.1: closes the M3 gap — `predicted_speeds` used to ignore the DB).
///
/// `None` (no samples for this device/backend/quant) returns the inputs
/// unchanged. A non-positive or non-finite factor is treated as missing
/// for that axis: one bad bench run must not zero a prediction.
pub fn apply_efficiency(pp: f64, tg: f64, eff: Option<&Efficiency>) -> (f64, f64) {
    let Some(e) = eff else {
        return (pp, tg);
    };
    let scale = |raw: f64, factor: f64| {
        if factor.is_finite() && factor > 0.0 {
            raw * factor
        } else {
            raw
        }
    };
    (scale(pp, e.pp_efficiency), scale(tg, e.tg_efficiency))
}

/// Estimate speed for a single device (no splitting).
pub fn estimate_speed_single(
    desc: &Descriptor,
    kv: &KvEstimate,
    _ctx_len: u64,
    prompt_tokens: u64,
    hw: &HwSpec,
) -> SpeedEstimate {
    let active = active_weight_bytes(desc);
    let kv_read = kv.kv_bytes / 2; // read at ctx/2 average
    let bytes_per_token = active + kv_read;

    let decode = if bytes_per_token > 0 {
        hw.efficiency * hw.bandwidth_gbps * 1e9 / bytes_per_token as f64
    } else {
        0.0
    };

    let active_params = active / 2; // fp16 = 2 bytes/param
    let compute_bound = if active_params > 0 && hw.tflops > 0.0 {
        hw.efficiency * hw.tflops * 1e12 / (2.0 * active_params as f64)
    } else {
        f64::INFINITY
    };
    let bandwidth_bound = if bytes_per_token > 0 {
        hw.efficiency * hw.bandwidth_gbps * 1e9 / bytes_per_token as f64
    } else {
        0.0
    };
    let prefill = compute_bound.min(bandwidth_bound);
    let ttft = if prefill > 0.0 {
        prompt_tokens as f64 / prefill
    } else {
        f64::INFINITY
    };

    SpeedEstimate {
        active_bytes: active,
        decode_toks_per_sec: decode,
        prefill_toks_per_sec: prefill,
        ttft_secs: ttft,
        gpu_bytes: 0,
        cpu_bytes: 0,
    }
}

/// Estimate speed for hybrid GPU+CPU placement.
///
/// `gpu_fraction` is the fraction of active weights placed on GPU (0.0–1.0).
/// The rest stays on CPU.
pub fn estimate_speed_hybrid(
    desc: &Descriptor,
    kv: &KvEstimate,
    _ctx_len: u64,
    prompt_tokens: u64,
    gpu_hw: &HwSpec,
    cpu_hw: &HwSpec,
    gpu_fraction: f64,
) -> SpeedEstimate {
    let active = active_weight_bytes(desc);
    let kv_read = kv.kv_bytes / 2;
    let bytes_per_token = active + kv_read;

    let gpu_bytes = (bytes_per_token as f64 * gpu_fraction) as u64;
    let cpu_bytes = bytes_per_token - gpu_bytes;

    // Hybrid decode: 1 / (gpu_part/BW_gpu + cpu_part/BW_cpu)
    let gpu_time = if gpu_bytes > 0 && gpu_hw.bandwidth_gbps > 0.0 {
        gpu_bytes as f64 / (gpu_hw.efficiency * gpu_hw.bandwidth_gbps * 1e9)
    } else {
        0.0
    };
    let cpu_time = if cpu_bytes > 0 && cpu_hw.bandwidth_gbps > 0.0 {
        cpu_bytes as f64 / (cpu_hw.efficiency * cpu_hw.bandwidth_gbps * 1e9)
    } else {
        0.0
    };
    let decode = if gpu_time + cpu_time > 0.0 {
        1.0 / (gpu_time + cpu_time)
    } else {
        0.0
    };

    // Prefill: use GPU FLOPS for the GPU portion, bandwidth for CPU portion.
    let active_params = active / 2;
    let gpu_params = (active_params as f64 * gpu_fraction) as u64;
    let compute_bound = if gpu_params > 0 && gpu_hw.tflops > 0.0 {
        gpu_hw.efficiency * gpu_hw.tflops * 1e12 / (2.0 * gpu_params as f64)
    } else {
        f64::INFINITY
    };
    let bandwidth_bound = if bytes_per_token > 0 {
        let total_bw =
            gpu_hw.efficiency * gpu_hw.bandwidth_gbps + cpu_hw.efficiency * cpu_hw.bandwidth_gbps;
        total_bw * 1e9 / bytes_per_token as f64
    } else {
        0.0
    };
    let prefill = compute_bound.min(bandwidth_bound);
    let ttft = if prefill > 0.0 {
        prompt_tokens as f64 / prefill
    } else {
        f64::INFINITY
    };

    SpeedEstimate {
        active_bytes: active,
        decode_toks_per_sec: decode,
        prefill_toks_per_sec: prefill,
        ttft_secs: ttft,
        gpu_bytes,
        cpu_bytes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::{GGUF_MAGIC, GgmlType};
    use crate::kv::estimate_kv;

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

    #[allow(clippy::too_many_arguments)]
    fn make_desc(
        arch: &str,
        n_layer: u64,
        n_embd: u64,
        n_head: u64,
        n_head_kv: u64,
        head_dim: u64,
        ctx: u64,
        n_vocab: u64,
        dense_bytes: u64,
        expert_bytes: u64,
        embed_out_bytes: u64,
    ) -> Descriptor {
        let kv: Vec<(String, u32, Vec<u8>)> = vec![
            ("general.architecture".to_string(), 8, str_value(arch)),
            (format!("{arch}.block_count"), 10, u64_value(n_layer)),
            (format!("{arch}.embedding_length"), 10, u64_value(n_embd)),
            (
                format!("{arch}.attention.head_count"),
                10,
                u64_value(n_head),
            ),
            (format!("{arch}.context_length"), 10, u64_value(ctx)),
            (format!("{arch}.vocab_size"), 10, u64_value(n_vocab)),
            (
                format!("{arch}.attention.head_count_kv"),
                10,
                u64_value(n_head_kv),
            ),
            (
                format!("{arch}.attention.key_length"),
                10,
                u64_value(head_dim),
            ),
        ];
        let embd_dims = [n_vocab, n_embd];
        let tensors = [("token_embd.weight", embd_dims.as_slice(), GgmlType::Q4_K)];
        let buf = write_gguf(
            &kv.iter()
                .map(|(k, t, v)| (k.as_str(), *t, v.clone()))
                .collect::<Vec<_>>(),
            &tensors,
        );
        let r = crate::gguf::Reader::parse(&buf).unwrap();
        let mut d = Descriptor::from_reader(&r).unwrap();
        // Override weight bytes for testing.
        d.weight_bytes_dense = dense_bytes;
        d.weight_bytes_expert = expert_bytes;
        d.weight_bytes_embed_out = embed_out_bytes;
        d.weight_bytes_total = dense_bytes + expert_bytes + embed_out_bytes;
        d
    }

    #[test]
    fn active_weight_dense_model() {
        // Dense model: active = total.
        let d = make_desc(
            "qwen2",
            24,
            896,
            14,
            2,
            64,
            4096,
            151936,
            200_000_000,
            0,
            50_000_000,
        );
        assert_eq!(active_weight_bytes(&d), 250_000_000);
    }

    #[test]
    fn active_weight_moe_model() {
        // MoE: dense=100M, expert=400M, embed=50M; 8/128 experts used.
        let mut d = make_desc(
            "qwen3moe",
            48,
            2048,
            16,
            2,
            128,
            4096,
            151936,
            100_000_000,
            400_000_000,
            50_000_000,
        );
        d.n_expert = 128;
        d.n_expert_used = 8;
        let active = active_weight_bytes(&d);
        // 100M + (8/128)*400M + 50M = 100M + 25M + 50M = 175M
        assert_eq!(active, 175_000_000);
    }

    #[test]
    fn decode_scales_with_bandwidth() {
        let d = make_desc(
            "qwen2",
            24,
            896,
            14,
            2,
            64,
            4096,
            151936,
            200_000_000,
            0,
            50_000_000,
        );
        let kv = estimate_kv(&d, 2048, "f16");
        let mut fast = HwSpec::cuda();
        fast.bandwidth_gbps = 2000.0;
        let mut slow = HwSpec::cpu();
        slow.bandwidth_gbps = 50.0;
        let est_fast = estimate_speed_single(&d, &kv, 2048, 1024, &fast);
        let est_slow = estimate_speed_single(&d, &kv, 2048, 1024, &slow);
        assert!(est_fast.decode_toks_per_sec > est_slow.decode_toks_per_sec * 10.0);
    }

    #[test]
    fn prefill_limited_by_compute() {
        let d = make_desc(
            "qwen2",
            24,
            896,
            14,
            2,
            64,
            4096,
            151936,
            200_000_000,
            0,
            50_000_000,
        );
        let kv = estimate_kv(&d, 2048, "f16");
        let hw = HwSpec {
            bandwidth_gbps: 10000.0, // huge BW → bandwidth-bound doesn't trigger
            tflops: 10.0,            // low compute → compute-bound
            efficiency: 0.60,
        };
        let est = estimate_speed_single(&d, &kv, 2048, 1024, &hw);
        // prefill should be compute-bound (low TFLOPS, high BW)
        assert!(est.prefill_toks_per_sec > 0.0);
        assert!(est.ttft_secs > 0.0);
    }

    #[test]
    fn hybrid_split() {
        let d = make_desc(
            "qwen2",
            24,
            896,
            14,
            2,
            64,
            4096,
            151936,
            200_000_000,
            0,
            50_000_000,
        );
        let kv = estimate_kv(&d, 2048, "f16");
        let gpu = HwSpec::cuda();
        let cpu = HwSpec::cpu();
        let est = estimate_speed_hybrid(&d, &kv, 2048, 1024, &gpu, &cpu, 0.8);
        // GPU-heavy should be faster than CPU-only.
        let cpu_only = estimate_speed_single(&d, &kv, 2048, 1024, &HwSpec::cpu());
        assert!(est.decode_toks_per_sec > cpu_only.decode_toks_per_sec);
        assert!(est.gpu_bytes > 0);
        assert!(est.cpu_bytes > 0);
    }

    #[test]
    fn npu_stubs_are_conservative_and_finite() {
        // P9.4: uncalibrated Tier-3 stubs — slower than CUDA, faster than
        // nothing, finite on a dense model.
        let d = make_desc(
            "qwen2",
            24,
            896,
            14,
            2,
            64,
            4096,
            151936,
            200_000_000,
            0,
            50_000_000,
        );
        let kv = estimate_kv(&d, 2048, "f16");
        for hw in [HwSpec::hexagon(), HwSpec::openvino()] {
            let est = estimate_speed_single(&d, &kv, 2048, 1024, &hw);
            assert!(est.decode_toks_per_sec.is_finite(), "{hw:?}");
            assert!(est.decode_toks_per_sec > 0.0, "{hw:?}");
            assert!(est.ttft_secs.is_finite(), "{hw:?}");
        }
        let cuda = estimate_speed_single(&d, &kv, 2048, 1024, &HwSpec::cuda());
        let hexagon = estimate_speed_single(&d, &kv, 2048, 1024, &HwSpec::hexagon());
        let openvino = estimate_speed_single(&d, &kv, 2048, 1024, &HwSpec::openvino());
        assert!(
            hexagon.decode_toks_per_sec < cuda.decode_toks_per_sec,
            "stub must not out-predict CUDA"
        );
        assert!(
            openvino.decode_toks_per_sec < cuda.decode_toks_per_sec,
            "stub must not out-predict CUDA"
        );
    }

    #[test]
    fn ttft_decreases_with_prompt_length() {
        let d = make_desc(
            "qwen2",
            24,
            896,
            14,
            2,
            64,
            4096,
            151936,
            200_000_000,
            0,
            50_000_000,
        );
        let kv = estimate_kv(&d, 4096, "f16");
        let hw = HwSpec::cuda();
        let short = estimate_speed_single(&d, &kv, 4096, 128, &hw);
        let long = estimate_speed_single(&d, &kv, 4096, 2048, &hw);
        // Longer prompt → higher TTFT.
        assert!(long.ttft_secs > short.ttft_secs);
    }

    fn test_eff(pp: f64, tg: f64) -> Efficiency {
        Efficiency {
            device: "cpu".into(),
            backend: "cpu".into(),
            quant: "Q4_0".into(),
            pp_efficiency: pp,
            tg_efficiency: tg,
            sample_count: 3,
        }
    }

    #[test]
    fn apply_efficiency_scales_both_axes() {
        let (pp, tg) = apply_efficiency(100.0, 20.0, Some(&test_eff(1.2, 0.5)));
        assert!((pp - 120.0).abs() < 1e-9, "{pp}");
        assert!((tg - 10.0).abs() < 1e-9, "{tg}");
    }

    #[test]
    fn apply_efficiency_missing_db_keeps_raw() {
        assert_eq!(apply_efficiency(100.0, 20.0, None), (100.0, 20.0));
    }

    #[test]
    fn apply_efficiency_ignores_bad_factors_per_axis() {
        // A zero tg factor (one bad bench run) must not zero the decode
        // prediction; the healthy pp axis still scales.
        let (pp, tg) = apply_efficiency(100.0, 20.0, Some(&test_eff(2.0, 0.0)));
        assert!((pp - 200.0).abs() < 1e-9, "{pp}");
        assert!((tg - 20.0).abs() < 1e-9, "{tg}");
        let (pp, tg) = apply_efficiency(100.0, 20.0, Some(&test_eff(f64::NAN, 0.5)));
        assert!((pp - 100.0).abs() < 1e-9, "{pp}");
        assert!((tg - 10.0).abs() < 1e-9, "{tg}");
    }
}
