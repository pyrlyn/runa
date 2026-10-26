// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! Compute mode (`--mode cpu|gpu|hybrid`, plan P2.1 / D4). The engine maps
//! it onto a `Placement`; the CLI, daemon and fit checker only name it.

/// Compute mode requested by the user.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// All layers on CPU (`n_gpu_layers = 0`).
    Cpu,
    /// All layers on GPU (`n_gpu_layers = all`).
    Gpu,
    /// All layers on GPU except MoE expert tensors, which stay on CPU
    /// (`ffn_*_exps` overrides — the P2.6 pattern).
    Hybrid,
}

impl Mode {
    /// Parse a `--mode` flag value.
    pub fn parse(s: &str) -> Option<Mode> {
        match s {
            "cpu" => Some(Mode::Cpu),
            "gpu" => Some(Mode::Gpu),
            "hybrid" => Some(Mode::Hybrid),
            _ => None,
        }
    }
}
