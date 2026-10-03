// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! P4.9 media-pipeline benches: resize, normalize, resample.
//!
//! Mel lives inside whisper.cpp (`whisper-rs`); this crate only feeds it
//! 16 kHz PCM. Run: `cargo bench -p runa-media --bench media`.

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use runa_media::{
    normalize_rgb_scalar, normalize_rgb_simd, patchify_rgb, resample_mono, resize_rgb,
};

fn bench_resize(c: &mut Criterion) {
    let src: Vec<u8> = (0..512 * 512 * 3).map(|i| (i % 251) as u8).collect();
    c.bench_function("resize_rgb_512_to_336", |b| {
        b.iter(|| resize_rgb(black_box(&src), 512, 512, 336, 336));
    });
}

fn bench_normalize(c: &mut Criterion) {
    let rgb: Vec<u8> = (0..336 * 336 * 3).map(|i| (i % 251) as u8).collect();
    let mut out = vec![0.0f32; rgb.len()];
    c.bench_function("normalize_rgb_336", |b| {
        b.iter(|| normalize_rgb_scalar(black_box(&rgb), black_box(&mut out)));
    });
    c.bench_function("normalize_rgb_simd_336", |b| {
        b.iter(|| normalize_rgb_simd(black_box(&rgb), black_box(&mut out)));
    });
}

fn bench_patchify(c: &mut Criterion) {
    let rgb: Vec<u8> = (0..336 * 336 * 3).map(|i| (i % 251) as u8).collect();
    c.bench_function("patchify_rgb_336_p16", |b| {
        b.iter(|| patchify_rgb(black_box(&rgb), 336, 336, 16));
    });
}

fn bench_resample(c: &mut Criterion) {
    // 1 s of 48 kHz mono → 16 kHz (P4.1 path when the source is not already 16 k).
    let pcm: Vec<f32> = (0..48_000)
        .map(|i| ((i as f32) * 0.01).sin() * 0.25)
        .collect();
    c.bench_function("resample_48k_to_16k_1s", |b| {
        b.iter(|| resample_mono(black_box(&pcm), 48_000, 16_000).expect("resample"));
    });
}

fn bench_histogram(c: &mut Criterion) {
    let left: Vec<u8> = (0..336 * 336 * 3).map(|i| (i % 251) as u8).collect();
    let right: Vec<u8> = (0..336 * 336 * 3).map(|i| ((i * 3) % 251) as u8).collect();
    c.bench_function("histogram_l1_336", |bencher| {
        bencher.iter(|| runa_media::histogram_l1(black_box(&left), black_box(&right)));
    });
}

criterion_group!(
    benches,
    bench_resize,
    bench_normalize,
    bench_patchify,
    bench_resample,
    bench_histogram
);
criterion_main!(benches);
