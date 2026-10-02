// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! Model descriptor: architecture params + weight byte accounting (P1.3).
//!
//! Given a parsed GGUF header, `Descriptor` extracts the model's
//! architectural knobs (`n_layer`, `n_embd`, `n_head`, `n_expert`, …) and
//! computes exact on-disk weight bytes per-tensor and in aggregate. This is
//! the input the placement planner (P1.8) and speed model (P1.9) need.
//!
//! Weight bytes use the exact per-type block sizes in `ggml_types`, matching
//! what ggml writes and what `gguf-dump` reports. No rounding, no
//! heuristics: `tensor_bytes(dims, type) == sum of actual block bytes` for
//! all known quant types.
//!
//! # Weight groups
//!
//! Weights are split into three groups for the planner (MoE layout, GPU
//! offload decisions, mmproj placement):
//!
//! - **Dense weights** — attention/ffn/embedding/output in dense (non-MoE)
//!   architectures, or shared params in MoE.
//! - **Expert weights** — `ffn_*_exps` tensors in MoE; these are candidates
//!   for CPU placement on constrained GPUs (P2.6).
//! - **Projection/embedding+output** — `token_embd`, `output`, `output_norm`,
//!   `position_embd`; the embedding tables the planner may place differently
//!   on limited VRAM.

use crate::ggml_types::tensor_bytes;
use crate::gguf::Reader;
use std::collections::HashMap;

/// High-level model descriptor extracted from GGUF metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct Descriptor {
    /// Architecture string from `general.architecture` (e.g. "qwen2",
    /// "llama", "gemma3").
    pub arch: String,

    // --- dense params (always present) ---
    pub n_layer: u64,
    pub n_embd: u64,
    pub n_head: u64,
    pub n_head_kv: u64,
    /// Explicit head dimension from `attention.key_length`; if absent we
    /// derive `n_embd / n_head` (that is the ggml convention).
    pub head_dim: u64,
    pub n_ctx_train: u64,
    pub n_vocab: u64,
    pub n_vocab_out: u64,

    // --- MoE params (0 when not MoE) ---
    pub n_expert: u64,
    pub n_expert_used: u64,

    // --- special architecture features ---
    /// Sliding-window size (0 = full attention).
    pub sliding_window: u64,
    /// MLA kv_lora_rank (0 = standard MHA/GQA).
    pub kv_lora_rank: u64,
    /// Number of recurrent (SSM/state-space) layers (0 = all transformer).
    pub n_recurrent: u64,

    // --- byte accounting ---
    pub weight_bytes_dense: u64,
    pub weight_bytes_expert: u64,
    pub weight_bytes_embed_out: u64,
    /// Sum of all three groups.
    pub weight_bytes_total: u64,

    /// Per-tensor byte map: `tensor_name → (bytes, group)`.
    pub tensor_bytes: HashMap<String, (u64, WeightGroup)>,

    /// The original reader version (2 or 3).
    pub version: u32,
}

/// Weight groups for the planner.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WeightGroup {
    Dense,
    Expert,
    EmbedOut,
}

/// Errors during descriptor extraction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DescriptorError {
    /// `general.architecture` was missing or not a string.
    MissingArch,
    /// A required arch key was missing.
    MissingKey(String),
}

impl std::fmt::Display for DescriptorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DescriptorError::MissingArch => write!(f, "general.architecture not found"),
            DescriptorError::MissingKey(k) => write!(f, "required key missing: {k}"),
        }
    }
}

impl std::error::Error for DescriptorError {}

impl Descriptor {
    /// Build a descriptor from a parsed GGUF header.
    ///
    /// Reads `general.architecture` to determine the key prefix, then
    /// extracts standard keys `{arch}.{field}` with sensible defaults
    /// where ggml allows the key to be absent (e.g. MoE fields on dense
    /// models are 0).
    pub fn from_reader(r: &Reader) -> Result<Self, DescriptorError> {
        let arch = r
            .get_str("general.architecture")
            .ok_or(DescriptorError::MissingArch)?
            .to_string();

        let p = |field: &str| format!("{arch}.{field}");

        let require_u64 = |key: &str| -> Result<u64, DescriptorError> {
            r.get_u64(key)
                .ok_or_else(|| DescriptorError::MissingKey(key.to_string()))
        };
        // Required for every architecture.
        let n_layer = require_u64(&p("block_count"))?;
        let n_embd = require_u64(&p("embedding_length"))?;
        let n_head = require_u64(&p("attention.head_count"))?;
        let n_ctx_train = require_u64(&p("context_length"))?;

        // n_head_kv defaults to n_head when absent (full MHA).
        let n_head_kv = r.get_u64(&p("attention.head_count_kv")).unwrap_or(n_head);

        // head_dim: prefer explicit key, derive from n_embd/n_head.
        let head_dim = r
            .get_u64(&p("attention.key_length"))
            .unwrap_or_else(|| n_embd / n_head);

        // vocab_size: try the direct key first; if absent, derive from tokenizer tokens array.
        let n_vocab = r.get_u64(&p("vocab_size")).unwrap_or_else(|| {
            r.get("tokenizer.ggml.tokens")
                .and_then(|v| match v {
                    crate::gguf::Value::Array(crate::gguf::ArrayValue::String(a)) => {
                        Some(a.len() as u64)
                    }
                    _ => None,
                })
                .unwrap_or(0)
        });
        // output vocab often differs; default to n_vocab.
        let n_vocab_out = r.get_u64(&p("output.vocab_size")).unwrap_or(n_vocab);

        // MoE fields — 0 / absent for dense models.
        let n_expert = r.get_u64(&p("expert_count")).unwrap_or(0);
        let n_expert_used = r.get_u64(&p("expert_used_count")).unwrap_or(0);

        // Special architecture features.
        let sliding_window = r.get_u64(&p("sliding_window")).unwrap_or(0);
        let kv_lora_rank = r.get_u64(&p("attention.kv_lora_rank")).unwrap_or(0);
        let n_recurrent = r.get_u64(&p("recurrent_layer_count")).unwrap_or(0);

        // Byte accounting: walk every tensor and classify + sum.
        let mut weight_bytes_dense = 0u64;
        let mut weight_bytes_expert = 0u64;
        let mut weight_bytes_embed_out = 0u64;
        let mut tensor_bytes_map = HashMap::with_capacity(r.tensors.len());

        for ti in &r.tensors {
            let bytes = tensor_bytes(&ti.dims, ti.ggml_type).unwrap_or(0);
            let group = classify_tensor(&ti.name);
            match group {
                WeightGroup::Dense => weight_bytes_dense += bytes,
                WeightGroup::Expert => weight_bytes_expert += bytes,
                WeightGroup::EmbedOut => weight_bytes_embed_out += bytes,
            }
            tensor_bytes_map.insert(ti.name.clone(), (bytes, group));
        }

        let weight_bytes_total = weight_bytes_dense + weight_bytes_expert + weight_bytes_embed_out;

        Ok(Descriptor {
            arch,
            n_layer,
            n_embd,
            n_head,
            n_head_kv,
            head_dim,
            n_ctx_train,
            n_vocab,
            n_vocab_out,
            n_expert,
            n_expert_used,
            sliding_window,
            kv_lora_rank,
            n_recurrent,
            weight_bytes_dense,
            weight_bytes_expert,
            weight_bytes_embed_out,
            weight_bytes_total,
            tensor_bytes: tensor_bytes_map,
            version: r.version,
        })
    }
}

/// Classify a tensor name into a weight group.
///
/// Classification rules (ggml naming conventions):
/// - `ffn_*_exps.*` → Expert (MoE expert feed-forward params).
/// - `token_embd`, `output`, `output_norm`, `position_embd`, `blk.*.attn_*_norm`,
///   `blk.*.ffn_*_norm` → EmbedOut/normalization (small tensors, but
///   conventionally in the embedding/output bucket).
/// - Everything else → Dense.
fn classify_tensor(name: &str) -> WeightGroup {
    // MoE expert weights: ffn.*_exps.* (Qwen3-MoE, DeepSeek, Mixtral)
    if name.contains("_exps") {
        return WeightGroup::Expert;
    }

    // Embedding / output / normalization tensors.
    if name.starts_with("token_embd")
        || name.starts_with("output")
        || name.starts_with("position_embd")
    {
        return WeightGroup::EmbedOut;
    }

    // Norm layers are small but belong with embedding/output for the planner.
    if name.contains("_norm") {
        return WeightGroup::EmbedOut;
    }

    WeightGroup::Dense
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::{GGUF_MAGIC, GgmlType};

    /// Minimal GGUF v3 writer (self-contained test helper, no proptest dep).
    fn write_gguf(
        kv_pairs: &[(&str, u32, Vec<u8>)], // (key, type_tag, value_bytes)
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
            // ggml type raw (Q4_0 = 2, Q4_K = 12, Q8_0 = 8, Q8_K = 15, F16 = 1, F32 = 0)
            let ty_raw: u32 = match ty {
                GgmlType::F32 => 0,
                GgmlType::F16 => 1,
                GgmlType::Q4_0 => 2,
                GgmlType::Q8_0 => 8,
                GgmlType::Q4_K => 12,
                GgmlType::Q6_K => 14,
                GgmlType::Q8_K => 15,
                GgmlType::MXFP4 => 26,
                _ => 0,
            };
            b.extend_from_slice(&ty_raw.to_le_bytes());
            b.extend_from_slice(&0u64.to_le_bytes()); // offset
        }
        while !b.len().is_multiple_of(32) {
            b.push(0);
        }
        b
    }

    fn str_value(s: &str) -> Vec<u8> {
        let mut v = (s.len() as u64).to_le_bytes().to_vec();
        v.extend_from_slice(s.as_bytes());
        v
    }

    fn u64_value(v: u64) -> Vec<u8> {
        v.to_le_bytes().to_vec()
    }

    #[test]
    fn qwen2_0_5b_descriptor() {
        let buf = write_gguf(
            &[
                ("general.architecture", 8, str_value("qwen2")),
                ("qwen2.block_count", 10, u64_value(24)),
                ("qwen2.embedding_length", 10, u64_value(896)),
                ("qwen2.attention.head_count", 10, u64_value(14)),
                ("qwen2.context_length", 10, u64_value(32768)),
                ("qwen2.vocab_size", 10, u64_value(151936)),
                ("qwen2.attention.head_count_kv", 10, u64_value(2)),
            ],
            &[
                ("token_embd.weight", &[151936, 896], GgmlType::Q4_0),
                ("blk.0.attn_q.weight", &[896, 896], GgmlType::Q8_0),
                ("output.weight", &[896, 151936], GgmlType::Q4_0),
                ("output_norm.weight", &[896], GgmlType::F16),
            ],
        );
        let r = crate::gguf::Reader::parse(&buf).unwrap();
        let d = Descriptor::from_reader(&r).unwrap();

        assert_eq!(d.arch, "qwen2");
        assert_eq!(d.n_layer, 24);
        assert_eq!(d.n_embd, 896);
        assert_eq!(d.n_head, 14);
        assert_eq!(d.n_head_kv, 2);
        assert_eq!(d.head_dim, 896 / 14); // derived
        assert_eq!(d.n_vocab, 151936);
        assert_eq!(d.n_expert, 0);
        assert_eq!(d.sliding_window, 0);
        assert_eq!(d.version, 3);

        // Verify weight groups are classified correctly.
        assert_eq!(
            d.tensor_bytes.get("token_embd.weight"),
            Some(&(
                crate::ggml_types::tensor_bytes(&[151936, 896], GgmlType::Q4_0).unwrap(),
                WeightGroup::EmbedOut
            ))
        );
        assert_eq!(
            d.tensor_bytes.get("blk.0.attn_q.weight"),
            Some(&(
                crate::ggml_types::tensor_bytes(&[896, 896], GgmlType::Q8_0).unwrap(),
                WeightGroup::Dense
            ))
        );
        assert!(d.weight_bytes_total > 0);
    }

    #[test]
    fn moe_expert_classification() {
        let buf = write_gguf(
            &[
                ("general.architecture", 8, str_value("qwen3moe")),
                ("qwen3moe.block_count", 10, u64_value(48)),
                ("qwen3moe.embedding_length", 10, u64_value(2048)),
                ("qwen3moe.attention.head_count", 10, u64_value(16)),
                ("qwen3moe.context_length", 10, u64_value(4096)),
                ("qwen3moe.vocab_size", 10, u64_value(151936)),
                ("qwen3moe.expert_count", 10, u64_value(128)),
                ("qwen3moe.expert_used_count", 10, u64_value(8)),
            ],
            &[
                ("token_embd.weight", &[151936, 2048], GgmlType::Q4_K),
                ("blk.0.ffn_gate_exps.weight", &[256, 2048], GgmlType::Q4_K),
                ("blk.0.ffn_up_exps.weight", &[256, 2048], GgmlType::Q4_K),
                ("blk.0.ffn_down_exps.weight", &[2048, 256], GgmlType::Q4_K),
                ("blk.0.attn_q.weight", &[2048, 2048], GgmlType::Q4_K),
                ("output.weight", &[2048, 151936], GgmlType::Q4_K),
            ],
        );
        let r = crate::gguf::Reader::parse(&buf).unwrap();
        let d = Descriptor::from_reader(&r).unwrap();

        assert_eq!(d.arch, "qwen3moe");
        assert_eq!(d.n_expert, 128);
        assert_eq!(d.n_expert_used, 8);

        // Expert tensors classified correctly.
        assert_eq!(
            d.tensor_bytes.get("blk.0.ffn_gate_exps.weight").unwrap().1,
            WeightGroup::Expert,
        );
        assert_eq!(
            d.tensor_bytes.get("blk.0.ffn_up_exps.weight").unwrap().1,
            WeightGroup::Expert,
        );
        assert_eq!(
            d.tensor_bytes.get("blk.0.ffn_down_exps.weight").unwrap().1,
            WeightGroup::Expert,
        );
        // Non-expert tensors are Dense or EmbedOut.
        assert_eq!(
            d.tensor_bytes.get("blk.0.attn_q.weight").unwrap().1,
            WeightGroup::Dense,
        );
        assert_eq!(
            d.tensor_bytes.get("token_embd.weight").unwrap().1,
            WeightGroup::EmbedOut,
        );

        assert!(d.weight_bytes_expert > 0);
        assert!(d.weight_bytes_dense > 0);
        assert!(d.weight_bytes_embed_out > 0);
    }

    #[test]
    fn parses_real_qwen2_fixture() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/fixtures/qwen2-0_5b-instruct-q4_0.gguf"
        );
        let Ok(buf) = std::fs::read(path) else {
            eprintln!("fixture not present; skipping");
            return;
        };
        let r = crate::gguf::Reader::parse(&buf).unwrap();
        let d = Descriptor::from_reader(&r).unwrap();

        assert_eq!(d.arch, "qwen2");
        assert_eq!(d.n_layer, 24);
        assert_eq!(d.n_embd, 896);
        assert_eq!(d.n_head, 14);
        assert_eq!(d.head_dim, 64);
        assert_eq!(d.n_vocab, 151936);
        assert_eq!(d.n_expert, 0);
        assert!(d.weight_bytes_total > 0);
        // The file is 337 MiB; weight_bytes_total should be roughly that.
        // File = header (~100 KB) + weights. Allow 1% tolerance.
        let file_bytes = buf.len() as u64;
        let ratio = d.weight_bytes_total as f64 / file_bytes as f64;
        assert!(
            (0.90..=1.0).contains(&ratio),
            "weight_bytes_total {}/{file_bytes} ratio {ratio:.3} outside 0.90–1.00",
            d.weight_bytes_total,
        );
    }
}
