//! Compute-buffer estimator (P1.5).
//!
//! The compute buffer is the scratch memory ggml allocates during inference
//! to hold activations, intermediate results, and backend-specific scratch.
//! Its size depends on the model shape and the chosen micro-batch size.
//!
//! # Formula
//!
//! ```text
//! compute_bytes = safety × (
//!     n_ubatch × n_embd × element_bytes × 2          // input + output activations
//!   + n_ubatch × n_embd × 4 × n_layer                // per-layer scratch (matmul + attn)
//!   + n_ubatch × n_head × head_dim × 2 × n_layer     // attention score buffers
//!   + n_ubatch × ffn_dim × 2 × n_layer               // FFN intermediates
//!   + vocab_slice_bytes                                // softmax scratch
//! )
//! ```
//!
//! The `+15%` safety margin accounts for backend overhead, alignment padding,
//! and model-specific quirks (e.g. MoE expert routing buffers).
//!
//! This module provides a configurable, sample-calibratable estimator.
//! Initial heuristics are conservative; real llama.cpp log samples (P1.5
//! completion criterion) will tighten the formula.

use crate::descriptor::Descriptor;

/// Compute-buffer estimation result.
#[derive(Debug, Clone, PartialEq)]
pub struct ComputeEstimate {
    /// Total compute buffer bytes (with safety margin).
    pub compute_bytes: u64,
    /// Without safety margin.
    pub compute_bytes_raw: u64,
    /// Micro-batch size used.
    pub n_ubatch: u64,
    /// Safety multiplier applied (default 1.15).
    pub safety: f64,
}

/// Estimate compute-buffer size for a given model and micro-batch.
///
/// `n_ubatch` is the micro-batch size (tokens processed per forward pass).
/// Typical values: 512 for CPU, 2048–8192 for GPU.
///
/// The formula is conservative (+15% safety) and will be calibrated against
/// real llama.cpp log samples in later revisions.
pub fn estimate_compute(desc: &Descriptor, n_ubatch: u64) -> ComputeEstimate {
    estimate_compute_with_safety(desc, n_ubatch, 1.15)
}

/// Estimate with a custom safety multiplier (for calibration).
pub fn estimate_compute_with_safety(
    desc: &Descriptor,
    n_ubatch: u64,
    safety: f64,
) -> ComputeEstimate {
    let bpe: u64 = 2; // f16 element bytes (typical compute precision)
    let head_dim = desc.head_dim;
    let n_embd = desc.n_embd;
    let n_layer = desc.n_layer;
    let n_head = desc.n_head;

    // FFN intermediate dim: typically 4× n_embd for SwiGLU-style models,
    // or stored explicitly. We use a heuristic: 4× n_embd (conservative).
    let ffn_dim = n_embd * 4;

    // Activation tensors: input + output activations for the micro-batch.
    let activations = n_ubatch * n_embd * bpe * 2;

    // Per-layer scratch: matmul outputs + attention intermediates.
    let matmul_scratch = n_ubatch * n_embd * 4 * n_layer;
    let attn_scratch = n_ubatch * n_head * head_dim * 2 * n_layer;
    let ffn_scratch = n_ubatch * ffn_dim * 2 * n_layer;

    // Softmax scratch: vocab slice for the output layer.
    // `checked_div`: a hostile header can declare `embedding_length = 0`.
    let vocab_slice = (n_ubatch * desc.n_vocab * bpe)
        .checked_div(desc.n_embd)
        .unwrap_or(0); // scaled down

    let raw = activations + matmul_scratch + attn_scratch + ffn_scratch + vocab_slice;
    let compute_bytes = ((raw as f64) * safety) as u64;

    ComputeEstimate {
        compute_bytes,
        compute_bytes_raw: raw,
        n_ubatch,
        safety,
    }
}

/// Micro-batch size recommendation based on available memory.
///
/// Returns a conservative n_ubatch that fits within `available_bytes` of
/// compute buffer memory. Uses the same formula as `estimate_compute`.
pub fn recommend_ubatch(desc: &Descriptor, available_bytes: u64) -> u64 {
    // Binary search for the largest n_ubatch that fits.
    let mut lo = 1u64;
    let mut hi = desc.n_vocab.min(32768); // don't exceed vocab or 32K
    while lo < hi {
        let mid = lo + (hi - lo).div_ceil(2);
        let est = estimate_compute(desc, mid);
        if est.compute_bytes <= available_bytes {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    lo
}

/// Vision/audio encoder scratch (P4.8): `frames × n_embd × 4 × 8`.
///
/// Zero when `frames == 0`. Used as GPU-reserved encoder compute on top of
/// the language-model compute buffer.
pub fn estimate_encoder_compute(desc: &Descriptor, frames: u64) -> u64 {
    if frames == 0 {
        return 0;
    }
    frames
        .saturating_mul(desc.n_embd)
        .saturating_mul(4)
        .saturating_mul(8)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::descriptor::Descriptor;
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
        Descriptor::from_reader(&r).unwrap()
    }

    #[test]
    fn larger_ubatch_needs_more_buffer() {
        let d = make_desc("qwen2", 24, 896, 14, 2, 64, 4096, 151936);
        let small = estimate_compute(&d, 256);
        let large = estimate_compute(&d, 1024);
        assert!(large.compute_bytes > small.compute_bytes);
        // Should scale roughly linearly with n_ubatch (not exactly due to vocab_slice).
        let ratio = large.compute_bytes_raw as f64 / small.compute_bytes_raw as f64;
        assert!(ratio > 3.0 && ratio < 5.0, "ratio {ratio:.2}");
    }

    #[test]
    fn safety_margin_applied() {
        let d = make_desc("qwen2", 24, 896, 14, 2, 64, 4096, 151936);
        let est = estimate_compute(&d, 512);
        assert!((est.safety - 1.15).abs() < f64::EPSILON);
        assert_eq!(
            est.compute_bytes,
            (est.compute_bytes_raw as f64 * 1.15) as u64
        );
    }

    #[test]
    fn more_layers_needs_more_buffer() {
        let small = make_desc("qwen2", 12, 896, 14, 2, 64, 4096, 151936);
        let large = make_desc("qwen2", 36, 896, 14, 2, 64, 4096, 151936);
        let est_s = estimate_compute(&small, 512);
        let est_l = estimate_compute(&large, 512);
        assert!(est_l.compute_bytes > est_s.compute_bytes);
    }

    #[test]
    fn encoder_compute_scales_with_frames() {
        let d = make_desc("qwen2", 24, 896, 14, 2, 64, 4096, 151936);
        assert_eq!(estimate_encoder_compute(&d, 0), 0);
        let one = estimate_encoder_compute(&d, 1);
        let thirty_two = estimate_encoder_compute(&d, 32);
        assert_eq!(thirty_two, one * 32);
        assert!(one > 0);
    }

    #[test]
    fn recommend_ubatch_fits_in_budget() {
        let d = make_desc("qwen2", 24, 896, 14, 2, 64, 4096, 151936);
        let budget = 512 * 1024 * 1024; // 512 MiB
        let ub = recommend_ubatch(&d, budget);
        assert!(ub >= 1);
        let est = estimate_compute(&d, ub);
        assert!(est.compute_bytes <= budget);
        // The next larger ubatch should exceed the budget.
        let est_next = estimate_compute(&d, ub + 1);
        assert!(est_next.compute_bytes > budget);
    }

    #[test]
    fn larger_model_needs_larger_buffer() {
        // 0.5B vs hypothetical larger model.
        let small = make_desc("qwen2", 24, 896, 14, 2, 64, 4096, 151936);
        let large = make_desc("qwen2", 32, 4096, 32, 8, 128, 8192, 151936);
        let est_s = estimate_compute(&small, 512);
        let est_l = estimate_compute(&large, 512);
        assert!(est_l.compute_bytes > est_s.compute_bytes * 5);
    }
}
