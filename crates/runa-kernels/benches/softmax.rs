// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

use criterion::{Criterion, criterion_group, criterion_main};
use runa_kernels::{sample_token, softmax_f32, softmax_f32_reference};

fn bench_softmax(c: &mut Criterion) {
    let n = 151_936usize;
    let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.00001).collect();
    let mut output = vec![0.0f32; n];

    c.bench_function("softmax_f32_dispatched", |b| {
        b.iter(|| softmax_f32(&input, &mut output));
    });
    c.bench_function("softmax_f32_rust_ref", |b| {
        b.iter(|| softmax_f32_reference(&input, &mut output));
    });
}

fn bench_sample(c: &mut Criterion) {
    let logits: Vec<f32> = (0..4096).map(|i| (i as f32) * 0.001).collect();
    let mut seed = 42u32;
    c.bench_function("sample_token_4k", |b| {
        b.iter(|| sample_token(&logits, 0.8, 40, 0.95, 0.05, &mut seed));
    });
}

criterion_group!(benches, bench_softmax, bench_sample);
criterion_main!(benches);
