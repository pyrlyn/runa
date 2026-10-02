// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

#include "scalar.h"

#if defined(__AVX2__)
#include <immintrin.h>
#include <math.h>

void runa_softmax_f32_avx2(const float *in, float *out, size_t n) {
    if (n == 0) {
        return;
    }

    __m256 vmax = _mm256_set1_ps(-INFINITY);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        vmax = _mm256_max_ps(vmax, _mm256_loadu_ps(in + i));
    }
    float maxbuf[8];
    _mm256_storeu_ps(maxbuf, vmax);
    float maxv = maxbuf[0];
    for (int j = 1; j < 8; j++) {
        if (maxbuf[j] > maxv) {
            maxv = maxbuf[j];
        }
    }
    for (; i < n; i++) {
        if (in[i] > maxv) {
            maxv = in[i];
        }
    }

    float sum = 0.0f;
    for (i = 0; i < n; i++) {
        out[i] = expf(in[i] - maxv);
        sum += out[i];
    }
    if (sum <= 0.0f) {
        return;
    }
    __m256 vinv = _mm256_set1_ps(1.0f / sum);
    i = 0;
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(out + i, _mm256_mul_ps(_mm256_loadu_ps(out + i), vinv));
    }
    float inv = 1.0f / sum;
    for (; i < n; i++) {
        out[i] *= inv;
    }
}
#else
void runa_softmax_f32_avx2(const float *in, float *out, size_t n) {
    runa_softmax_f32_scalar(in, out, n);
}
#endif
