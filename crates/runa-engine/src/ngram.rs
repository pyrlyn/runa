// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! Prompt n-gram cache for speculative decoding (P5.6).
//!
//! Trigram → next-token map, updated as tokens are accepted. Drafts are
//! verified against the target model's argmax so greedy output is unchanged.

use std::collections::HashMap;
use std::path::PathBuf;

/// Speculative decoding options (P5.6).
#[derive(Debug, Clone)]
pub struct Speculative {
    /// Enable trigram lookup drafts (greedy-verified).
    pub ngram: bool,
    /// How many tokens to draft after the sampled token.
    pub draft_n: usize,
    /// Optional draft-model GGUF (fit counts its bytes; decode uses n-gram).
    pub draft: Option<PathBuf>,
}

impl Default for Speculative {
    fn default() -> Self {
        Self {
            ngram: false,
            draft_n: 4,
            draft: None,
        }
    }
}

/// Last-seen continuation for each trigram in the prompt + accepted tokens.
#[derive(Debug, Default)]
pub struct NgramCache {
    map: HashMap<(i32, i32, i32), i32>,
}

impl NgramCache {
    pub fn new() -> Self {
        Self {
            map: HashMap::new(),
        }
    }

    /// Index every trigram → following token in `toks`.
    pub fn ingest(&mut self, toks: &[i32]) {
        if toks.len() < 4 {
            return;
        }
        for w in toks.windows(4) {
            self.map.insert((w[0], w[1], w[2]), w[3]);
        }
    }

    /// Record `next` as the continuation of the last trigram in `hist`.
    pub fn learn(&mut self, hist: &[i32], next: i32) {
        let n = hist.len();
        if n < 3 {
            return;
        }
        self.map
            .insert((hist[n - 3], hist[n - 2], hist[n - 1]), next);
    }

    /// Greedy drafts from `hist` (does not mutate the cache).
    pub fn draft(&self, hist: &[i32], max: usize) -> Vec<i32> {
        let mut h = hist.to_vec();
        let mut out = Vec::new();
        for _ in 0..max {
            let n = h.len();
            if n < 3 {
                break;
            }
            let Some(&nxt) = self.map.get(&(h[n - 3], h[n - 2], h[n - 1])) else {
                break;
            };
            out.push(nxt);
            h.push(nxt);
        }
        out
    }
}

/// Index of the largest logit (lowest index on ties).
pub fn argmax_i32(logits: &[f32]) -> i32 {
    let mut best_i = 0i32;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best_i = i as i32;
        }
    }
    best_i
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ingest_and_draft_repeats() {
        // 1 2 3 4 1 2 3 4 → trigram 1,2,3 → 4 and 2,3,4 → 1
        let seq = [1, 2, 3, 4, 1, 2, 3, 4];
        let mut c = NgramCache::new();
        c.ingest(&seq);
        assert_eq!(c.draft(&[1, 2, 3], 3), vec![4, 1, 2]);
    }

    #[test]
    fn unknown_trigram_drafts_nothing() {
        let mut c = NgramCache::new();
        c.ingest(&[1, 2, 3, 4]);
        assert!(c.draft(&[9, 9, 9], 4).is_empty());
    }

    #[test]
    fn argmax_picks_first_on_ties() {
        assert_eq!(argmax_i32(&[1.0, 3.0, 3.0, 2.0]), 1);
    }

    #[test]
    fn default_draft_n_is_four() {
        assert_eq!(Speculative::default().draft_n, 4);
        assert!(!Speculative::default().ngram);
    }
}
