// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! Image preprocess kernels (P5.4): normalize + patchify, NEON/AVX2 dispatch.
//!
//! Resize stays nearest-neighbor (`resize_rgb`). High-quality
//! `fast_image_resize` was already slower (K3 / P4.9).

/// `out[i] = (rgb[i] / 255 − 0.5) × 2` → `[−1, 1]`.
pub fn normalize_rgb_scalar(rgb: &[u8], out: &mut [f32]) {
    let n = rgb.len().min(out.len());
    for i in 0..n {
        out[i] = (f32::from(rgb[i]) / 255.0 - 0.5) * 2.0;
    }
}

/// Production normalize: scalar. SIMD is 1.5× here (D14 gate is 2×) — P5.4 REJECT.
pub fn normalize_rgb(rgb: &[u8], out: &mut [f32]) {
    normalize_rgb_scalar(rgb, out);
}

/// NEON/AVX2 dispatch (not the default; isolated op under the 2× gate).
pub fn normalize_rgb_simd(rgb: &[u8], out: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: NEON was just detected; rgb/out lengths are the same contract as scalar.
            unsafe {
                normalize_rgb_neon(rgb, out);
            }
            return;
        }
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 was just detected; rgb/out lengths are the same contract as scalar.
            unsafe {
                normalize_rgb_avx2(rgb, out);
            }
            return;
        }
    }
    normalize_rgb_scalar(rgb, out);
}

/// Flatten `patch×patch` RGB tiles into `[−1, 1]` f32, row-major patches.
///
/// Returns empty if `patch == 0` or the image is not an integer grid.
pub fn patchify_rgb(rgb: &[u8], w: u32, h: u32, patch: u32) -> Vec<f32> {
    if patch == 0 || !w.is_multiple_of(patch) || !h.is_multiple_of(patch) {
        return Vec::new();
    }
    let need = (w as usize) * (h as usize) * 3;
    if rgb.len() < need {
        return Vec::new();
    }
    let nx = w / patch;
    let ny = h / patch;
    let ps = patch as usize;
    let ww = w as usize;
    let mut out = vec![0.0f32; (nx * ny) as usize * ps * ps * 3];
    let mut o = 0;
    for py in 0..ny {
        for px in 0..nx {
            let x0 = (px * patch) as usize;
            let y0 = (py * patch) as usize;
            for y in 0..ps {
                let row = ((y0 + y) * ww + x0) * 3;
                for x in 0..ps {
                    let s = row + x * 3;
                    for c in 0..3 {
                        out[o] = (f32::from(rgb[s + c]) / 255.0 - 0.5) * 2.0;
                        o += 1;
                    }
                }
            }
        }
    }
    out
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn normalize_rgb_neon(rgb: &[u8], out: &mut [f32]) {
    use std::arch::aarch64::*;
    // SAFETY: `target_feature = neon`; loads/stores stay in `rgb`/`out`.
    unsafe {
        let scale = vdupq_n_f32(2.0 / 255.0);
        let neg_one = vdupq_n_f32(-1.0);
        let n = rgb.len().min(out.len());
        let mut i = 0;
        while i + 16 <= n {
            let bytes = vld1q_u8(rgb.as_ptr().add(i));
            let lo16 = vmovl_u8(vget_low_u8(bytes));
            let hi16 = vmovl_u8(vget_high_u8(bytes));
            let f0 = vfmaq_f32(neg_one, vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo16))), scale);
            let f1 = vfmaq_f32(
                neg_one,
                vcvtq_f32_u32(vmovl_u16(vget_high_u16(lo16))),
                scale,
            );
            let f2 = vfmaq_f32(neg_one, vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi16))), scale);
            let f3 = vfmaq_f32(
                neg_one,
                vcvtq_f32_u32(vmovl_u16(vget_high_u16(hi16))),
                scale,
            );
            vst1q_f32(out.as_mut_ptr().add(i), f0);
            vst1q_f32(out.as_mut_ptr().add(i + 4), f1);
            vst1q_f32(out.as_mut_ptr().add(i + 8), f2);
            vst1q_f32(out.as_mut_ptr().add(i + 12), f3);
            i += 16;
        }
        while i < n {
            out[i] = (f32::from(rgb[i]) / 255.0 - 0.5) * 2.0;
            i += 1;
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn normalize_rgb_avx2(rgb: &[u8], out: &mut [f32]) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    // SAFETY: `target_feature = avx2`; loads/stores stay in `rgb`/`out`.
    unsafe {
        let scale = _mm256_set1_ps(2.0 / 255.0);
        let neg_one = _mm256_set1_ps(-1.0);
        let n = rgb.len().min(out.len());
        let mut i = 0;
        while i + 16 <= n {
            let v = _mm_loadu_si128(rgb.as_ptr().add(i).cast());
            let lo = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(v));
            let hi = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(v, 8)));
            _mm256_storeu_ps(
                out.as_mut_ptr().add(i),
                _mm256_add_ps(_mm256_mul_ps(lo, scale), neg_one),
            );
            _mm256_storeu_ps(
                out.as_mut_ptr().add(i + 8),
                _mm256_add_ps(_mm256_mul_ps(hi, scale), neg_one),
            );
            i += 16;
        }
        while i < n {
            out[i] = (f32::from(rgb[i]) / 255.0 - 0.5) * 2.0;
            i += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_black_white() {
        let rgb = [0u8, 255];
        let mut out = [0.0f32; 2];
        normalize_rgb_scalar(&rgb, &mut out);
        assert!((out[0] + 1.0).abs() < 1e-5);
        assert!((out[1] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn dispatched_matches_scalar_with_tail() {
        let rgb: Vec<u8> = (0..337).map(|i| (i % 251) as u8).collect();
        let mut a = vec![0.0f32; rgb.len()];
        let mut b = vec![0.0f32; rgb.len()];
        normalize_rgb_scalar(&rgb, &mut a);
        normalize_rgb_simd(&rgb, &mut b);
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() < 1e-5, "{x} vs {y}");
        }
    }

    #[test]
    fn patchify_one_tile() {
        let rgb = [0u8, 0, 0, 255, 255, 255, 0, 0, 0, 255, 255, 255];
        let p = patchify_rgb(&rgb, 2, 2, 2);
        assert_eq!(p.len(), 12);
        assert!((p[0] + 1.0).abs() < 1e-5);
        assert!((p[3] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn patchify_rejects_bad_grid() {
        assert!(patchify_rgb(&[0; 12], 2, 2, 3).is_empty());
        assert!(patchify_rgb(&[0; 3], 1, 1, 0).is_empty());
    }
}
