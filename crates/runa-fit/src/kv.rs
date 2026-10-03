// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! KV-cache memory estimator (P1.4).
//!
//! The KV cache stores key and value projections per-token, per-layer.
//! Its size depends on the architecture (standard MHA/GQA, MLA, SWA) and
//! the chosen KV quantization type.
//!
//! # Standard formula
//!
//! ```text
//! KV bytes = 2 × n_layer × ctx × n_head_kv × head_dim × bytes(kv_type)
//! ```
//!
//! Where `ctx = n_ctx` for full-attention layers, or `min(n_ctx, window)`
//! for SWA layers.
//!
//! # Exceptions
//!
//! - **MLA** (`kv_lora_rank > 0`): each token stores a compressed latent
//!   of size `kv_lora_rank` (instead of `n_head_kv × head_dim`), plus a
//!   small RoPE factor. Per layer per token: `2 × (kv_lora_rank + rope_dim)`
//!   elements. `rope_dim` defaults to `head_dim`.
//! - **Recurrent layers** (SSM / hybrid): constant-size state, typically
//!   on the order of `d_inner × d_state` per layer (model-specific; we
//!   estimate conservatively or read from metadata if present).
//!
//! # Implementation
//!
//! All KV type byte sizes are exact (same table as ggml's `type_traits`).

use crate::descriptor::Descriptor;

/// KV-cache size estimation result.
#[derive(Debug, Clone, PartialEq)]
pub struct KvEstimate {
    /// Total KV cache bytes for all layers at the given context length.
    pub kv_bytes: u64,
    /// Per-layer KV bytes (for diagnostics / layer-by-layer breakdown).
    pub kv_bytes_per_layer: Vec<u64>,
    /// Context length used (may be less than `n_ctx_train` if SWA).
    pub effective_ctx: u64,
    /// Whether MLA compression was used.
    pub used_mla: bool,
    /// Number of layers with SWA (reduced context).
    pub n_swa_layers: u64,
}

/// Bytes per KV element for common types. Returns 0 for unknown types.
pub fn kv_bytes_per_element(kv_type: &str) -> f64 {
    match kv_type {
        "f32" => 4.0,
        "f16" | "bf16" => 2.0,
        "q8_0" => 34.0 / 32.0,
        "q4_0" => 18.0 / 32.0,
        "q4_1" => 20.0 / 32.0,
        "iq4_nl" => 18.0 / 32.0,
        _ => 0.0,
    }
}

/// Estimate KV-cache memory for a model.
///
/// `ctx_len` is the context length the user requested (may be <= `n_ctx_train`).
/// `kv_type` is the KV quantization string (e.g. "f16", "q8_0").
pub fn estimate_kv(desc: &Descriptor, ctx_len: u64, kv_type: &str) -> KvEstimate {
    let bpe = kv_bytes_per_element(kv_type);
    let mut kv_bytes_per_layer = Vec::with_capacity(desc.n_layer as usize);
    let mut kv_bytes = 0u64;
    let mut n_swa_layers = 0u64;

    for _layer in 0..desc.n_layer {
        let layer_bytes = if desc.kv_lora_rank > 0 {
            // MLA: compressed latent + rope_dim per token
            let rope_dim = desc.head_dim; // usually head_dim for MLA
            let elements_per_token = desc.kv_lora_rank.saturating_add(rope_dim).saturating_mul(2);
            let bytes_per_token = (elements_per_token as f64) * bpe;
            (ctx_len as f64 * bytes_per_token) as u64
        } else {
            // Standard MHA/GQA
            let ctx = if desc.sliding_window > 0 && desc.sliding_window < ctx_len {
                n_swa_layers += 1;
                desc.sliding_window
            } else {
                ctx_len
            };
            let elements_per_token = desc
                .n_head_kv
                .saturating_mul(desc.head_dim)
                .saturating_mul(2);
            let bytes_per_token = (elements_per_token as f64) * bpe;
            (ctx as f64 * bytes_per_token) as u64
        };
        kv_bytes = kv_bytes.saturating_add(layer_bytes);
        kv_bytes_per_layer.push(layer_bytes);
    }

    KvEstimate {
        kv_bytes,
        kv_bytes_per_layer,
        effective_ctx: ctx_len,
        used_mla: desc.kv_lora_rank > 0,
        n_swa_layers,
    }
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
        while !b.len().is_multiple_of(32) {
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
        sw: u64,
        kv_lora: u64,
    ) -> Descriptor {
        let mut kv = vec![
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
        ];
        if n_head_kv != n_head {
            kv.push((
                format!("{arch}.attention.head_count_kv"),
                10,
                u64_value(n_head_kv),
            ));
        }
        if head_dim != n_embd / n_head {
            kv.push((
                format!("{arch}.attention.key_length"),
                10,
                u64_value(head_dim),
            ));
        }
        if sw > 0 {
            kv.push((format!("{arch}.sliding_window"), 10, u64_value(sw)));
        }
        if kv_lora > 0 {
            kv.push((
                format!("{arch}.attention.kv_lora_rank"),
                10,
                u64_value(kv_lora),
            ));
        }
        let embd_dims = [n_vocab, n_embd];
        let tensors = vec![("token_embd.weight", embd_dims.as_slice(), GgmlType::Q4_K)];
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
    fn standard_gqa_24_layers() {
        // Qwen2-0.5B-like: 24 layers, 2 KV heads, head_dim 64, f16 KV.
        let d = make_desc("qwen2", 24, 896, 14, 2, 64, 4096, 151936, 0, 0);
        let est = estimate_kv(&d, 4096, "f16");
        // 2 * 24 * 4096 * 2 * 64 * 2.0 = 2 × 24 × 4096 × 128 × 2 = 503,316,480 ≈ 480 MiB
        let expected = 2 * 24 * 4096 * 2 * 64 * 2;
        assert_eq!(est.kv_bytes, expected);
        assert!(!est.used_mla);
        assert_eq!(est.n_swa_layers, 0);
    }

    #[test]
    fn mla_deepseek_style() {
        // DeepSeek-V2-like MLA: kv_lora_rank=512, head_dim=128
        let d = make_desc("deepseek2", 60, 7168, 128, 128, 128, 32768, 102400, 0, 512);
        let est = estimate_kv(&d, 32768, "f16");
        // MLA: per token = 2*(512+128) elements = 1280; 1280 * 2.0 = 2560 bytes/token
        // Total: 60 * 32768 * 2560 = 4,980,736,000
        let expected = 60 * 32768 * 2 * (512 + 128) * 2;
        assert_eq!(est.kv_bytes, expected);
        assert!(est.used_mla);
    }

    #[test]
    fn swa_reduces_context() {
        // Gemma3-like: 26 layers, sliding_window = 4096, ctx = 32768
        let d = make_desc("gemma3", 26, 3072, 16, 16, 192, 32768, 256000, 4096, 0);
        let est = estimate_kv(&d, 32768, "f16");
        // SWA layers use min(32768, 4096) = 4096
        assert_eq!(est.n_swa_layers, 26);
        assert_eq!(est.effective_ctx, 32768);
        let per_layer = 2 * 16 * 4096 * 192 * 2; // 16 heads * 4096 ctx * 192 dim * 2 bytes * 2 (K+V)
        assert_eq!(est.kv_bytes_per_layer[0], per_layer);
        assert_eq!(est.kv_bytes, per_layer * 26);
    }

    #[test]
    fn quantized_kv_smaller_than_f16() {
        let d = make_desc("qwen2", 24, 896, 14, 2, 64, 4096, 151936, 0, 0);
        let f16 = estimate_kv(&d, 4096, "f16");
        let q8 = estimate_kv(&d, 4096, "q8_0");
        let q4 = estimate_kv(&d, 4096, "q4_0");
        // f16 = 2.0 B/elem, q8_0 = 1.0625, q4_0 = 0.5625. Quant saves memory.
        assert!(f16.kv_bytes > 0);
        assert!(q8.kv_bytes < f16.kv_bytes);
        assert!(q4.kv_bytes < q8.kv_bytes);
    }

    #[test]
    fn kv_type_unknown_yields_zero() {
        let d = make_desc("qwen2", 24, 896, 14, 2, 64, 4096, 151936, 0, 0);
        let est = estimate_kv(&d, 4096, "unknown_type");
        assert_eq!(est.kv_bytes, 0);
    }
}
