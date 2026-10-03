// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! Compute-mode placement (plan P2.1, D4).
//!
//! `--mode cpu|gpu|hybrid` maps to a [`Placement`]: layer count for the GPU
//! plus tensor-buffer overrides (regex patterns forced onto CPU buffers).
//! `--n-cpu-moe N` pins the first N layers' MoE expert tensors to CPU
//! (llama.cpp `LLM_FFN_EXPS_REGEX`). `auto` is resolved by the planner (P2.5).

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
/// llama.cpp b7709 `LLM_FFN_EXPS_REGEX`: routed expert tensors only
/// (`ffn_*_exps`, including DeepSeek `ffn_*_chexps`). Fused `gate_up` landed
/// after this pin (llama.cpp #20416) and is intentionally omitted.
pub const FFN_EXPS_REGEX: &str = r"\.ffn_(up|down|gate)_(ch|)exps";

/// Tensor-buffer override patterns for `--n-cpu-moe N`.
///
/// Returns at most one regex: llama-cpp-2 0.1.133 can only apply a single
/// `add_cpu_buft_override` (a second call panics). `0` is empty; `1..255`
/// matches `blk.(0|…|N-1)` experts; `≥256` is all experts ([`FFN_EXPS_REGEX`]).
pub fn cpu_moe_patterns(n_cpu_moe: u32) -> Vec<String> {
    match n_cpu_moe {
        0 => Vec::new(),
        n if n >= 256 => vec![FFN_EXPS_REGEX.to_owned()],
        n => {
            // llama-cpp-2 0.1.133 `add_cpu_buft_override` always writes slot 0
            // (a second call panics). One regex is equivalent to N llama.cpp
            // `--n-cpu-moe` overrides.
            let layers = (0..n).map(|i| i.to_string()).collect::<Vec<_>>().join("|");
            vec![format!("blk\\.({layers}){FFN_EXPS_REGEX}")]
        }
    }
}

/// Comma-separated ggml backend indices or names (`0,1` / `CUDA0,CUDA1`).
pub fn parse_device_list(s: &str) -> Result<Vec<String>, String> {
    let parts: Vec<String> = s
        .split(',')
        .map(|x| x.trim().to_string())
        .filter(|x| !x.is_empty())
        .collect();
    if parts.is_empty() {
        return Err("--device: empty list".into());
    }
    Ok(parts)
}

/// Comma-separated llama.cpp RPC endpoints (`127.0.0.1:50052,…`, P9.3).
/// Entries are `host:port` of a `rpc-server` instance; validation is
/// shape-light (non-empty after trimming) like [`parse_device_list`]:
/// hostnames, IPv4 and bracketed IPv6 literals must all pass through.
pub fn parse_rpc_list(s: &str) -> Result<Vec<String>, String> {
    let parts: Vec<String> = s
        .split(',')
        .map(|x| x.trim().to_string())
        .filter(|x| !x.is_empty())
        .collect();
    if parts.is_empty() {
        return Err("--rpc: empty list".into());
    }
    Ok(parts)
}

/// Comma-separated per-GPU proportions (`3,1`). Values must be finite and >= 0.
pub fn parse_tensor_split(s: &str) -> Result<Vec<f32>, String> {
    let mut out = Vec::new();
    for part in s.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let v: f32 = part
            .parse()
            .map_err(|_| format!("--tensor-split: not a number: {part}"))?;
        if !v.is_finite() || v < 0.0 {
            return Err(format!("--tensor-split: {part} must be a finite >= 0"));
        }
        out.push(v);
    }
    if out.is_empty() {
        return Err("--tensor-split: empty list".into());
    }
    if out.iter().all(|v| *v == 0.0) {
        return Err("--tensor-split: all zeros".into());
    }
    Ok(out)
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

/// Where tensors live. `n_gpu_layers = 0` is pure CPU; `u32::MAX` means
/// "all layers" (llama.cpp clamps to the model depth).
#[derive(Debug, Clone, PartialEq)]
pub struct Placement {
    /// Layers placed on the GPU.
    pub n_gpu_layers: u32,
    /// Regex patterns (llama.cpp tensor-buffer override syntax) forced
    /// onto CPU buffers, e.g. `\\.ffn_(up|down|gate)_(ch|)exps`.
    pub cpu_patterns: Vec<String>,
    /// Main GPU index for multi-GPU (`--main-gpu`, P2.9).
    pub main_gpu: i32,
    /// ggml backend device specs (`--device 0,1` or `CUDA0,CUDA1`).
    /// Empty = llama.cpp default (all GPUs).
    pub devices: Vec<String>,
    /// Per-GPU proportions (`--tensor-split 3,1`). Empty = equal split.
    pub tensor_split: Vec<f32>,
    /// llama.cpp RPC endpoints (`--rpc host:port,…`, P9.3). Empty = local
    /// inference only. The pinned llama-cpp-sys-2 0.1.133 strips the ggml
    /// RPC backend (see `docs/versions.md`), so [`crate::load`] rejects a
    /// non-empty list with `EngineError::Unsupported` instead of silently
    /// ignoring it.
    pub rpc_servers: Vec<String>,
}

impl Placement {
    fn new(n_gpu_layers: u32, cpu_patterns: Vec<String>) -> Placement {
        Placement {
            n_gpu_layers,
            cpu_patterns,
            main_gpu: 0,
            devices: Vec::new(),
            tensor_split: Vec::new(),
            rpc_servers: Vec::new(),
        }
    }

    /// Pure CPU: nothing offloaded.
    pub fn cpu() -> Placement {
        Placement::new(0, Vec::new())
    }

    /// Full GPU offload.
    pub fn gpu() -> Placement {
        Placement::new(u32::MAX, Vec::new())
    }

    /// Hybrid: all layers on GPU, every MoE expert tensor pinned to CPU.
    pub fn hybrid_moe() -> Placement {
        Placement::new(u32::MAX, cpu_moe_patterns(256))
    }

    /// Pin the first `n` MoE layers' expert tensors to CPU (`--n-cpu-moe N`).
    /// `n == 0` clears overrides; `n >= 256` pins every expert tensor.
    pub fn with_n_cpu_moe(mut self, n: u32) -> Placement {
        self.cpu_patterns = cpu_moe_patterns(n);
        self
    }

    /// Restrict ggml backends (`--device`).
    pub fn with_devices(mut self, devices: Vec<String>) -> Placement {
        self.devices = devices;
        self
    }

    /// Set per-GPU proportions (`--tensor-split`).
    pub fn with_tensor_split(mut self, tensor_split: Vec<f32>) -> Placement {
        self.tensor_split = tensor_split;
        self
    }

    /// Set llama.cpp RPC endpoints (`--rpc`, P9.3). Currently always
    /// rejected by [`crate::load`] (no RPC backend in the pinned sys
    /// crate); the field exists so callers and the CLI can already carry
    /// the intent explicitly instead of dropping it silently.
    pub fn with_rpc_servers(mut self, rpc_servers: Vec<String>) -> Placement {
        self.rpc_servers = rpc_servers;
        self
    }

    /// Set the scratch/small-tensor GPU (`--main-gpu`).
    pub fn with_main_gpu(mut self, main_gpu: i32) -> Placement {
        self.main_gpu = main_gpu;
        self
    }

    /// Build from a [`Mode`].
    pub fn from_mode(mode: Mode) -> Placement {
        match mode {
            Mode::Cpu => Placement::cpu(),
            Mode::Gpu => Placement::gpu(),
            Mode::Hybrid => Placement::hybrid_moe(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mode_flags() {
        assert_eq!(Mode::parse("cpu"), Some(Mode::Cpu));
        assert_eq!(Mode::parse("gpu"), Some(Mode::Gpu));
        assert_eq!(Mode::parse("hybrid"), Some(Mode::Hybrid));
        assert_eq!(Mode::parse("auto"), None);
        assert_eq!(Mode::parse("tpu"), None);
    }

    #[test]
    fn cpu_places_nothing_on_gpu() {
        let p = Placement::from_mode(Mode::Cpu);
        assert_eq!(p.n_gpu_layers, 0);
        assert!(p.cpu_patterns.is_empty());
    }

    #[test]
    fn hybrid_pins_experts_to_cpu() {
        let p = Placement::from_mode(Mode::Hybrid);
        assert_eq!(p.n_gpu_layers, u32::MAX);
        assert_eq!(p.cpu_patterns.len(), 1);
        assert!(p.cpu_patterns[0].contains("exps"));
    }

    #[test]
    fn n_cpu_moe_zero_clears() {
        let p = Placement::hybrid_moe().with_n_cpu_moe(0);
        assert!(p.cpu_patterns.is_empty());
        assert!(cpu_moe_patterns(0).is_empty());
    }

    #[test]
    fn n_cpu_moe_n_prefixes_blk_layers() {
        let p = cpu_moe_patterns(2);
        assert_eq!(p.len(), 1);
        assert_eq!(p[0], "blk\\.(0|1)\\.ffn_(up|down|gate)_(ch|)exps");
    }

    #[test]
    fn n_cpu_moe_all_is_unprefixed_regex() {
        assert_eq!(cpu_moe_patterns(256), vec![FFN_EXPS_REGEX.to_owned()]);
        assert_eq!(Placement::hybrid_moe().cpu_patterns, cpu_moe_patterns(256));
    }

    #[test]
    fn gpu_plus_n_cpu_moe_keeps_all_layers_on_gpu() {
        let p = Placement::gpu().with_n_cpu_moe(8);
        assert_eq!(p.n_gpu_layers, u32::MAX);
        assert_eq!(p.cpu_patterns.len(), 1);
        assert!(p.cpu_patterns[0].starts_with("blk\\.("));
        assert!(p.cpu_patterns[0].contains("0|1|2|3|4|5|6|7"));
    }

    #[test]
    fn parse_device_list_csv() {
        assert_eq!(
            parse_device_list("0, 1").unwrap(),
            vec!["0".to_string(), "1".to_string()]
        );
        assert_eq!(
            parse_device_list("CUDA0,CUDA1").unwrap(),
            vec!["CUDA0".to_string(), "CUDA1".to_string()]
        );
        assert!(parse_device_list(" , ").is_err());
    }

    #[test]
    fn parse_tensor_split_csv() {
        assert_eq!(parse_tensor_split("3,1").unwrap(), vec![3.0, 1.0]);
        assert!(parse_tensor_split("1,-1").is_err());
        assert!(parse_tensor_split("0,0").is_err());
        assert!(parse_tensor_split("nope").is_err());
    }

    #[test]
    fn with_devices_and_split_preserves_mode() {
        let p = Placement::gpu()
            .with_devices(vec!["0".into(), "1".into()])
            .with_tensor_split(vec![3.0, 1.0])
            .with_main_gpu(1);
        assert_eq!(p.n_gpu_layers, u32::MAX);
        assert_eq!(p.devices, vec!["0", "1"]);
        assert_eq!(p.tensor_split, vec![3.0, 1.0]);
        assert_eq!(p.main_gpu, 1);
    }

    #[test]
    fn parse_rpc_list_csv() {
        assert_eq!(
            parse_rpc_list("127.0.0.1:50052, 10.0.0.2:50052").unwrap(),
            vec!["127.0.0.1:50052".to_string(), "10.0.0.2:50052".to_string()]
        );
        assert_eq!(
            parse_rpc_list("node1:50052").unwrap(),
            vec!["node1:50052".to_string()]
        );
        assert!(parse_rpc_list(" , ").is_err());
        assert!(parse_rpc_list("").is_err());
    }

    #[test]
    fn with_rpc_servers_preserves_mode() {
        let p = Placement::gpu().with_rpc_servers(vec!["127.0.0.1:50052".into()]);
        assert_eq!(p.n_gpu_layers, u32::MAX);
        assert_eq!(p.rpc_servers, vec!["127.0.0.1:50052"]);
        assert!(Placement::cpu().rpc_servers.is_empty());
        assert!(Placement::hybrid_moe().rpc_servers.is_empty());
    }
}
