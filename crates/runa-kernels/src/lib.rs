// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! runa-kernels — own kernels behind the benchmark gate (plan D1, D14, D23).
//!
//! P5.3 sampling + softmax: Zig `@Vector` (C ABI). C/`.S` only where Zig is
//! worse (not linked here; see `c/` for reference only).

mod dispatch;
mod reference;

pub use dispatch::{SoftmaxImpl, select_softmax};

mod ffi {
    unsafe extern "C" {
        pub(super) fn runa_softmax_f32_scalar(input: *const f32, output: *mut f32, n: usize);
        pub(super) fn runa_top_k_f32(p: *mut f32, n: usize, k: i32);
        pub(super) fn runa_top_p_f32(p: *mut f32, n: usize, top_p: f32);
        pub(super) fn runa_min_p_f32(p: *mut f32, n: usize, min_p: f32);
        pub(super) fn runa_sample_token_f32(
            logits: *const f32,
            n: usize,
            temperature: f32,
            top_k: i32,
            top_p: f32,
            min_p: f32,
            seed: *mut u32,
        ) -> i32;
    }
}

/// Softmax over logits; dispatches to the best available implementation.
pub fn softmax_f32(input: &[f32], output: &mut [f32]) {
    assert_eq!(input.len(), output.len());
    let n = input.len();
    unsafe {
        match select_softmax() {
            SoftmaxImpl::ZigVector => {
                ffi::runa_softmax_f32_scalar(input.as_ptr(), output.as_mut_ptr(), n)
            }
        }
    }
}

/// Pure-Rust scalar reference (fuzz / equivalence tests).
pub fn softmax_f32_reference(input: &[f32], output: &mut [f32]) {
    reference::softmax_f32(input, output);
}

/// Keep the `k` largest probabilities (in place).
pub fn top_k_f32(p: &mut [f32], k: i32) {
    unsafe {
        ffi::runa_top_k_f32(p.as_mut_ptr(), p.len(), k);
    }
}

/// Nucleus filter (in place).
pub fn top_p_f32(p: &mut [f32], top_p: f32) {
    unsafe {
        ffi::runa_top_p_f32(p.as_mut_ptr(), p.len(), top_p);
    }
}

/// Min-p filter (in place).
pub fn min_p_f32(p: &mut [f32], min_p: f32) {
    unsafe {
        ffi::runa_min_p_f32(p.as_mut_ptr(), p.len(), min_p);
    }
}

/// Sample one token id from `logits` (temperature / top-k / top-p / min-p).
pub fn sample_token(
    logits: &[f32],
    temperature: f32,
    top_k: i32,
    top_p: f32,
    min_p: f32,
    seed: &mut u32,
) -> i32 {
    unsafe {
        ffi::runa_sample_token_f32(
            logits.as_ptr(),
            logits.len(),
            temperature,
            top_k,
            top_p,
            min_p,
            seed,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: &[f32], b: &[f32]) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b) {
            assert!((*x - *y).abs() < 1e-5, "delta {x} vs {y}");
        }
    }

    #[test]
    fn zig_softmax_matches_rust_reference() {
        let input = [1.0f32, 2.0, 3.0, 0.5];
        let mut c_out = [0.0; 4];
        let mut rust_out = [0.0; 4];
        softmax_f32(&input, &mut c_out);
        softmax_f32_reference(&input, &mut rust_out);
        approx_eq(&c_out, &rust_out);
        let sum: f32 = c_out.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn dispatch_picks_zig_vector() {
        assert_eq!(select_softmax(), SoftmaxImpl::ZigVector);
    }

    #[test]
    fn empty_input_is_noop() {
        let input: [f32; 0] = [];
        let mut out: [f32; 0] = [];
        softmax_f32(&input, &mut out);
        softmax_f32_reference(&input, &mut out);
    }

    #[test]
    fn top_k_matches_reference() {
        let mut a = [0.1f32, 0.4, 0.3, 0.2];
        let mut b = a;
        top_k_f32(&mut a, 2);
        reference::top_k(&mut b, 2);
        approx_eq(&a, &b);
    }

    #[test]
    fn top_k_keeps_k_mass() {
        let mut p = [0.1f32, 0.4, 0.3, 0.2];
        top_k_f32(&mut p, 2);
        let nz = p.iter().filter(|x| **x > 0.0).count();
        assert_eq!(nz, 2);
        let sum: f32 = p.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(p[1] > 0.0 && p[2] > 0.0);
    }

    #[test]
    fn top_p_matches_reference() {
        let mut a = [0.05f32, 0.5, 0.3, 0.15];
        let mut b = a;
        top_p_f32(&mut a, 0.8);
        reference::top_p(&mut b, 0.8);
        approx_eq(&a, &b);
    }

    #[test]
    fn min_p_matches_reference() {
        let mut a = [0.01f32, 0.7, 0.2, 0.09];
        let mut b = a;
        min_p_f32(&mut a, 0.2);
        reference::min_p(&mut b, 0.2);
        approx_eq(&a, &b);
    }

    #[test]
    fn greedy_picks_argmax() {
        let logits = [0.1f32, 3.0, 1.2, -4.0];
        let mut seed = 1u32;
        let id = sample_token(&logits, 0.0, 40, 0.95, 0.05, &mut seed);
        assert_eq!(id, 1);
    }

    #[test]
    fn sample_stays_in_vocab() {
        let logits: Vec<f32> = (0..128).map(|i| (i as f32) * 0.01).collect();
        let mut seed = 99u32;
        for _ in 0..32 {
            let id = sample_token(&logits, 0.8, 16, 0.9, 0.05, &mut seed);
            assert!(id >= 0 && (id as usize) < logits.len());
        }
    }
}
