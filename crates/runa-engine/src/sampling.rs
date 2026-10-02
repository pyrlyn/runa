// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! Sampling chain (plan P2.2).
//!
//! [`SamplingConfig`] builds a `llama-cpp-2` sampler chain in the canonical
//! order: repeat penalties → temperature → top-k → top-p → min-p →
//! distribution. `temperature <= 0` selects greedy decoding instead of the
//! temperature/distribution tail (llama-cli convention).

use llama_cpp_2::sampling::LlamaSampler;

/// Sampling parameters for one generation.
#[derive(Debug, Clone)]
pub struct SamplingConfig {
    /// Temperature; `<= 0` means greedy (deterministic).
    pub temperature: f32,
    /// Top-k truncation.
    pub top_k: i32,
    /// Nucleus sampling threshold.
    pub top_p: f32,
    /// Min-p truncation.
    pub min_p: f32,
    /// Repeat penalty (`1.0` = off).
    pub repeat_penalty: f32,
    /// How many recent tokens the repeat penalty looks at.
    pub repeat_last_n: i32,
    /// Seed for the distribution sampler (temperature path only).
    pub seed: u32,
    /// When true, sample from logits with `runa-kernels` (P5.3) instead of
    /// ggml's sampler chain. Off by default until the D14 gate is met.
    pub kernel_sampler: bool,
}

impl Default for SamplingConfig {
    /// llama.cpp `common` defaults.
    fn default() -> Self {
        SamplingConfig {
            temperature: 0.8,
            top_k: 40,
            top_p: 0.95,
            min_p: 0.05,
            repeat_penalty: 1.0,
            repeat_last_n: 64,
            seed: 42,
            kernel_sampler: std::env::var("RUNA_KERNEL_SAMPLER").ok().as_deref() == Some("1"),
        }
    }
}

impl SamplingConfig {
    /// Deterministic decoding: penalties + greedy.
    pub fn greedy() -> SamplingConfig {
        SamplingConfig {
            temperature: 0.0,
            ..SamplingConfig::default()
        }
    }

    /// Build the sampler chain for a vocabulary of `n_vocab` tokens.
    pub fn build(&self, n_vocab: i32) -> LlamaSampler {
        let _ = n_vocab; // penalties/dist take what they need via params below.
        let mut chain = vec![LlamaSampler::penalties(
            self.repeat_last_n,
            self.repeat_penalty,
            0.0,
            0.0,
        )];
        if self.temperature <= 0.0 {
            chain.push(LlamaSampler::greedy());
        } else {
            chain.push(LlamaSampler::temp(self.temperature));
            chain.push(LlamaSampler::top_k(self.top_k));
            chain.push(LlamaSampler::top_p(self.top_p, 1));
            chain.push(LlamaSampler::min_p(self.min_p, 1));
            chain.push(LlamaSampler::dist(self.seed));
        }
        LlamaSampler::chain_simple(chain)
    }

    /// Sample one token id from a logits row using runa-kernels (P5.3).
    pub fn sample_logits(&self, logits: &[f32], seed: &mut u32) -> i32 {
        runa_kernels::sample_token(
            logits,
            self.temperature,
            self.top_k,
            self.top_p,
            self.min_p,
            seed,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_chain_builds() {
        let mut s = SamplingConfig::greedy().build(32000);
        // Resetting exercises the chain without a model.
        s.reset();
    }

    #[test]
    fn temperature_chain_builds_with_seed() {
        let cfg = SamplingConfig {
            seed: 1234,
            ..SamplingConfig::default()
        };
        let mut s = cfg.build(32000);
        s.reset();
    }

    #[test]
    fn defaults_match_common() {
        let d = SamplingConfig::default();
        assert_eq!(
            (d.temperature, d.top_k, d.top_p, d.min_p),
            (0.8, 40, 0.95, 0.05)
        );
        assert_eq!((d.repeat_penalty, d.repeat_last_n), (1.0, 64));
    }

    #[test]
    fn kernel_sample_logits_greedy_is_argmax() {
        let cfg = SamplingConfig {
            temperature: 0.0,
            kernel_sampler: true,
            ..SamplingConfig::default()
        };
        let mut seed = 1;
        assert_eq!(cfg.sample_logits(&[0.2, 9.0, 1.0], &mut seed), 1);
    }
}
