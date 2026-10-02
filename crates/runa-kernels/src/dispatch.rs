// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! Runtime dispatch (P5.3 / D23). Softmax is Zig `@Vector`; LLVM emits NEON/AVX.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SoftmaxImpl {
    /// Zig `@Vector` softmax (C ABI). Host SIMD via LLVM, not a separate `.c`.
    ZigVector,
}

/// Pick the softmax implementation. Own kernels are Zig (D23).
pub fn select_softmax() -> SoftmaxImpl {
    SoftmaxImpl::ZigVector
}
