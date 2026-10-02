// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

#include "scalar.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

void runa_softmax_f32_scalar(const float *in, float *out, size_t n) {
    if (n == 0) {
        return;
    }

    float max = in[0];
    for (size_t i = 1; i < n; i++) {
        if (in[i] > max) {
            max = in[i];
        }
    }

    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        out[i] = expf(in[i] - max);
        sum += out[i];
    }

    if (sum > 0.0f) {
        float inv = 1.0f / sum;
        for (size_t i = 0; i < n; i++) {
            out[i] *= inv;
        }
    }
}

typedef struct {
    float p;
    uint32_t i;
} Item;

static int cmp_desc(const void *a, const void *b) {
    float da = ((const Item *)a)->p;
    float db = ((const Item *)b)->p;
    if (da < db) {
        return 1;
    }
    if (da > db) {
        return -1;
    }
    return 0;
}

static void renormalize(float *p, size_t n) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        sum += p[i];
    }
    if (sum <= 0.0f) {
        return;
    }
    float inv = 1.0f / sum;
    for (size_t i = 0; i < n; i++) {
        p[i] *= inv;
    }
}

void runa_top_k_f32(float *p, size_t n, int32_t k) {
    if (n == 0 || k <= 0 || (size_t)k >= n) {
        return;
    }
    Item *items = (Item *)malloc(n * sizeof(Item));
    if (items == NULL) {
        return;
    }
    for (size_t i = 0; i < n; i++) {
        items[i].p = p[i];
        items[i].i = (uint32_t)i;
    }
    qsort(items, n, sizeof(Item), cmp_desc);
    memset(p, 0, n * sizeof(float));
    for (int32_t j = 0; j < k; j++) {
        p[items[j].i] = items[j].p;
    }
    free(items);
    renormalize(p, n);
}

void runa_top_p_f32(float *p, size_t n, float top_p) {
    if (n == 0 || top_p >= 1.0f) {
        return;
    }
    if (top_p <= 0.0f) {
        size_t best = 0;
        for (size_t i = 1; i < n; i++) {
            if (p[i] > p[best]) {
                best = i;
            }
        }
        memset(p, 0, n * sizeof(float));
        p[best] = 1.0f;
        return;
    }
    Item *items = (Item *)malloc(n * sizeof(Item));
    if (items == NULL) {
        return;
    }
    for (size_t i = 0; i < n; i++) {
        items[i].p = p[i];
        items[i].i = (uint32_t)i;
    }
    qsort(items, n, sizeof(Item), cmp_desc);
    float cum = 0.0f;
    size_t keep = 0;
    for (; keep < n; keep++) {
        cum += items[keep].p;
        if (cum >= top_p) {
            keep++;
            break;
        }
    }
    memset(p, 0, n * sizeof(float));
    for (size_t j = 0; j < keep; j++) {
        p[items[j].i] = items[j].p;
    }
    free(items);
    renormalize(p, n);
}

void runa_min_p_f32(float *p, size_t n, float min_p) {
    if (n == 0 || min_p <= 0.0f) {
        return;
    }
    float mx = p[0];
    for (size_t i = 1; i < n; i++) {
        if (p[i] > mx) {
            mx = p[i];
        }
    }
    float thr = min_p * mx;
    for (size_t i = 0; i < n; i++) {
        if (p[i] < thr) {
            p[i] = 0.0f;
        }
    }
    renormalize(p, n);
}

static uint32_t xorshift32(uint32_t *s) {
    uint32_t x = *s;
    if (x == 0) {
        x = 1;
    }
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *s = x;
    return x;
}

int32_t runa_sample_token_f32(
    const float *logits,
    size_t n,
    float temperature,
    int32_t top_k,
    float top_p,
    float min_p,
    uint32_t *seed
) {
    if (n == 0) {
        return 0;
    }
    if (temperature <= 0.0f) {
        size_t best = 0;
        for (size_t i = 1; i < n; i++) {
            if (logits[i] > logits[best]) {
                best = i;
            }
        }
        return (int32_t)best;
    }
    float *buf = (float *)malloc(n * sizeof(float));
    if (buf == NULL) {
        return 0;
    }
    for (size_t i = 0; i < n; i++) {
        buf[i] = logits[i] / temperature;
    }
    runa_softmax_f32_scalar(buf, buf, n);
    runa_top_k_f32(buf, n, top_k);
    runa_top_p_f32(buf, n, top_p);
    runa_min_p_f32(buf, n, min_p);

    float u = (float)(xorshift32(seed) >> 8) / (float)(1u << 24);
    float cum = 0.0f;
    int32_t pick = (int32_t)(n - 1);
    for (size_t i = 0; i < n; i++) {
        cum += buf[i];
        if (u <= cum) {
            pick = (int32_t)i;
            break;
        }
    }
    free(buf);
    return pick;
}
