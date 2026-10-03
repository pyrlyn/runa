// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

#pragma once

#include <stddef.h>
#include <stdint.h>

/// Softmax over `n` logits; writes probabilities to `out` (may alias `in`).
void runa_softmax_f32_scalar(const float *in, float *out, size_t n);

#if defined(__aarch64__) || defined(__ARM_NEON)
void runa_softmax_f32_neon(const float *in, float *out, size_t n);
#endif

#if defined(__AVX2__) || defined(__AVX512F__) || defined(_M_X64) || defined(__x86_64__)
void runa_softmax_f32_avx2(const float *in, float *out, size_t n);
#endif

/// Keep the `k` largest probabilities; zero the rest and renormalize.
void runa_top_k_f32(float *p, size_t n, int32_t k);

/// Nucleus sampling: keep smallest prefix whose mass ≥ `top_p`.
void runa_top_p_f32(float *p, size_t n, float top_p);

/// Drop probabilities below `min_p * max(p)` and renormalize.
void runa_min_p_f32(float *p, size_t n, float min_p);

/// Temperature + softmax + top-k + top-p + min-p + sample. Updates `seed`.
int32_t runa_sample_token_f32(
    const float *logits,
    size_t n,
    float temperature,
    int32_t top_k,
    float top_p,
    float min_p,
    uint32_t *seed
);
