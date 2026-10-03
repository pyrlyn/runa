// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! Scalar Rust reference implementations for equivalence tests.

pub fn softmax_f32(input: &[f32], output: &mut [f32]) {
    assert_eq!(input.len(), output.len());
    if input.is_empty() {
        return;
    }

    let max = input.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for (i, &x) in input.iter().enumerate() {
        let e = (x - max).exp();
        output[i] = e;
        sum += e;
    }
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for v in output.iter_mut() {
            *v *= inv;
        }
    }
}

#[cfg(test)]
fn renormalize(p: &mut [f32]) {
    let sum: f32 = p.iter().sum();
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for v in p.iter_mut() {
            *v *= inv;
        }
    }
}

#[cfg(test)]
pub fn top_k(p: &mut [f32], k: i32) {
    let n = p.len();
    if n == 0 || k <= 0 || k as usize >= n {
        return;
    }
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| p[b].total_cmp(&p[a]));
    let mut keep = vec![0.0f32; n];
    for &i in idx.iter().take(k as usize) {
        keep[i] = p[i];
    }
    p.copy_from_slice(&keep);
    renormalize(p);
}

#[cfg(test)]
pub fn top_p(p: &mut [f32], top_p: f32) {
    let n = p.len();
    if n == 0 || top_p >= 1.0 {
        return;
    }
    if top_p <= 0.0 {
        let best = p
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i)
            .unwrap_or(0);
        p.fill(0.0);
        p[best] = 1.0;
        return;
    }
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| p[b].total_cmp(&p[a]));
    let mut cum = 0.0f32;
    let mut keep_n = 0usize;
    for &i in &idx {
        cum += p[i];
        keep_n += 1;
        if cum >= top_p {
            break;
        }
    }
    let mut keep = vec![0.0f32; n];
    for &i in idx.iter().take(keep_n) {
        keep[i] = p[i];
    }
    p.copy_from_slice(&keep);
    renormalize(p);
}

#[cfg(test)]
pub fn min_p(p: &mut [f32], min_p: f32) {
    if p.is_empty() || min_p <= 0.0 {
        return;
    }
    let mx = p.iter().copied().fold(0.0f32, f32::max);
    let thr = min_p * mx;
    for v in p.iter_mut() {
        if *v < thr {
            *v = 0.0;
        }
    }
    renormalize(p);
}
