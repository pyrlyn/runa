// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

#include "scalar.h"

#if defined(__aarch64__) || defined(__ARM_NEON)
#include <arm_neon.h>
#include <math.h>

void runa_softmax_f32_neon(const float *in, float *out, size_t n) {
    if (n == 0) {
        return;
    }

    float32x4_t vmax = vdupq_n_f32(-INFINITY);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        vmax = vmaxq_f32(vmax, vld1q_f32(in + i));
    }
    float maxv = vmaxvq_f32(vmax);
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
    float32x4_t vinv = vdupq_n_f32(1.0f / sum);
    i = 0;
    for (; i + 4 <= n; i += 4) {
        vst1q_f32(out + i, vmulq_f32(vld1q_f32(out + i), vinv));
    }
    float inv = 1.0f / sum;
    for (; i < n; i++) {
        out[i] *= inv;
    }
}
#endif
