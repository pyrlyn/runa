// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! Fit verdict and report (P1.11).
//!
//! Combines the descriptor, KV estimator, compute-buffer estimator,
//! placement planner, and speed model into a single verdict:
//!
//! - **FITS GPU** — all layers on GPU, fits comfortably.
//! - **FITS HYBRID** — N/L layers on GPU, rest on CPU.
//! - **FITS CPU** — runs on CPU only (no GPU offload possible).
//! - **NO** — does not fit anywhere (VRAM + RAM insufficient).
//!
//! # Exit codes
//!
//! - `0` — fits without warnings.
//! - `1` — fits with warnings.
//! - `2` — does not fit.
//!
//! # Warnings
//!
//! - Context was reduced from requested to fit.
//! - KV quantization needed (original type doesn't fit).
//! - mmproj not counted in VRAM budget.
//! - Decode throughput below 5 tok/s.
//!
//! # Suggestions
//!
//! - Try a sibling quantization (smaller file).
//! - Reduce context length.
//! - Use `--kv q8_0` for smaller KV cache.
//! - Fall back to cloud API.

use crate::compute::{estimate_compute, estimate_encoder_compute};
use crate::descriptor::Descriptor;
use crate::kv::estimate_kv;
use crate::planner::{PlacementPlan, PlannerConfig, plan_placement};
use crate::speed::{HwSpec, SpeedEstimate, estimate_speed_single};

/// Fit verdict.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    /// All layers on GPU, fits comfortably.
    Gpu,
    /// N of L layers on GPU, rest on CPU.
    Hybrid { gpu_layers: u64, total_layers: u64 },
    /// Runs on CPU only.
    Cpu,
    /// Does not fit anywhere.
    NoFit,
}

impl std::fmt::Display for Verdict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Verdict::Gpu => write!(f, "FITS GPU"),
            Verdict::Hybrid {
                gpu_layers,
                total_layers,
            } => write!(f, "FITS HYBRID ({gpu_layers}/{total_layers} layers on GPU)"),
            Verdict::Cpu => write!(f, "FITS CPU"),
            Verdict::NoFit => write!(f, "NO FIT"),
        }
    }
}

/// Warning issued during fit check.
#[derive(Debug, Clone, PartialEq)]
pub enum Warning {
    ContextReduced {
        requested: u64,
        actual: u64,
    },
    KvQuantizationNeeded {
        original_type: String,
        suggested_type: String,
    },
    MmprojNotCounted,
    DecodeSlow {
        tok_per_sec: f64,
    },
    /// `frames × tokens_per_frame` (+ audio) does not fit in `n_ctx`.
    MediaExceedsContext {
        need: u64,
        n_ctx: u64,
    },
}

impl std::fmt::Display for Warning {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Warning::ContextReduced { requested, actual } => {
                write!(f, "Context reduced from {requested} to {actual} to fit")
            }
            Warning::KvQuantizationNeeded {
                original_type,
                suggested_type,
            } => {
                write!(
                    f,
                    "KV type {original_type} doesn't fit; suggested: --kv {suggested_type}"
                )
            }
            Warning::MmprojNotCounted => {
                write!(
                    f,
                    "Multimodal projection weights not counted in VRAM budget"
                )
            }
            Warning::DecodeSlow { tok_per_sec } => {
                write!(
                    f,
                    "Decode throughput {tok_per_sec:.1} tok/s is below recommended 5 tok/s"
                )
            }
            Warning::MediaExceedsContext { need, n_ctx } => {
                write!(
                    f,
                    "Media tokens {need} (frames × tokens_per_frame + audio) exceed n_ctx {n_ctx}"
                )
            }
        }
    }
}

/// Suggestion for improving fit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Suggestion {
    TrySiblingQuant,
    ReduceContext { recommended: u64 },
    UseKvQuantization { quant: String },
    UseCloudApi,
}

impl std::fmt::Display for Suggestion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Suggestion::TrySiblingQuant => {
                write!(f, "Try a smaller quantization (e.g. Q4_0 instead of Q8_0)")
            }
            Suggestion::ReduceContext { recommended } => {
                write!(f, "Reduce context length to {recommended} or less")
            }
            Suggestion::UseKvQuantization { quant } => {
                write!(f, "Use --kv {quant} for smaller KV cache")
            }
            Suggestion::UseCloudApi => write!(f, "Consider using a cloud API for this model"),
        }
    }
}

/// Complete fit report.
#[derive(Debug, Clone)]
pub struct FitReport {
    pub verdict: Verdict,
    pub warnings: Vec<Warning>,
    pub suggestions: Vec<Suggestion>,
    pub plan: PlacementPlan,
    pub speed_gpu: Option<SpeedEstimate>,
    pub speed_cpu: Option<SpeedEstimate>,
    pub kv_estimate: crate::kv::KvEstimate,
    pub compute_estimate: crate::compute::ComputeEstimate,
    /// Exit code: 0 = fits clean, 1 = fits with warnings, 2 = no fit.
    pub exit_code: u8,
    /// Tokens consumed by images/frames + audio (P4.8).
    pub media_tokens: u64,
}

/// Configuration for the fit check.
#[derive(Debug, Clone)]
pub struct FitConfig {
    pub planner: PlannerConfig,
    pub gpu_hw: Option<HwSpec>,
    pub cpu_hw: HwSpec,
    /// Whether mmproj weights are present (adds ~100–400 MiB to VRAM).
    pub has_mmproj: bool,
    /// Image/video frames + audio duration for context math (P4.8).
    pub media: MediaFit,
}

/// Vision/audio token budget (P4.8).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MediaFit {
    /// Sampled video frames or still images.
    pub frames: u64,
    /// Tokens each image/frame occupies in the prompt (CLIP/mmproj).
    pub tokens_per_frame: u64,
    /// Audio duration in seconds (0 if none).
    pub audio_seconds: f64,
    /// Tokens billed per second of audio.
    pub tokens_per_audio_second: u64,
}

impl Default for MediaFit {
    fn default() -> Self {
        Self {
            frames: 0,
            tokens_per_frame: 256,
            audio_seconds: 0.0,
            tokens_per_audio_second: 25,
        }
    }
}

impl MediaFit {
    /// `frames × tokens_per_frame + ceil(audio_seconds) × tokens_per_audio_second`.
    pub fn token_need(&self) -> u64 {
        let audio = self.audio_seconds.ceil().max(0.0) as u64;
        self.frames.saturating_mul(self.tokens_per_frame)
            + audio.saturating_mul(self.tokens_per_audio_second)
    }
}

impl Default for FitConfig {
    fn default() -> Self {
        Self {
            planner: PlannerConfig::default(),
            gpu_hw: Some(HwSpec::cuda()),
            cpu_hw: HwSpec::cpu(),
            has_mmproj: false,
            media: MediaFit::default(),
        }
    }
}

/// Run a full fit check and produce a report.
pub fn check_fit(desc: &Descriptor, config: &FitConfig) -> FitReport {
    let mut warnings = Vec::new();
    let mut suggestions = Vec::new();

    let media_tokens = config.media.token_need();
    let media_overflow = media_tokens > config.planner.ctx_len;
    if media_overflow {
        warnings.push(Warning::MediaExceedsContext {
            need: media_tokens,
            n_ctx: config.planner.ctx_len,
        });
    }

    // KV estimation.
    let kv = estimate_kv(desc, config.planner.ctx_len, &config.planner.kv_type);

    // Language compute + encoder scratch for sampled frames.
    let compute = estimate_compute(desc, config.planner.n_ubatch);
    let encoder = estimate_encoder_compute(desc, config.media.frames);
    let mut planner = config.planner.clone();
    planner.encoder_compute_bytes = planner.encoder_compute_bytes.max(encoder);

    // Placement planning (mmproj_bytes counted when set).
    let plan = plan_placement(desc, &planner);

    // Warn only when a projector exists but was not given a byte size.
    if config.has_mmproj && planner.mmproj_bytes == 0 {
        warnings.push(Warning::MmprojNotCounted);
    }

    // Speed estimation.
    let speed_gpu = config
        .gpu_hw
        .as_ref()
        .map(|hw| estimate_speed_single(desc, &kv, config.planner.ctx_len, 1024, hw));

    let speed_cpu = Some(estimate_speed_single(
        desc,
        &kv,
        config.planner.ctx_len,
        1024,
        &config.cpu_hw,
    ));

    // Check decode speed warnings.
    if let Some(ref sg) = speed_gpu
        && sg.decode_toks_per_sec > 0.0
        && sg.decode_toks_per_sec < 5.0
    {
        warnings.push(Warning::DecodeSlow {
            tok_per_sec: sg.decode_toks_per_sec,
        });
    }
    if let Some(ref sc) = speed_cpu
        && sc.decode_toks_per_sec > 0.0
        && sc.decode_toks_per_sec < 5.0
    {
        warnings.push(Warning::DecodeSlow {
            tok_per_sec: sc.decode_toks_per_sec,
        });
    }

    // Determine verdict. Media overflow is a hard no-fit (exit 2).
    let verdict = if media_overflow {
        Verdict::NoFit
    } else if plan.fits && plan.all_on_gpu {
        Verdict::Gpu
    } else if plan.fits && plan.gpu_layers > 0 {
        Verdict::Hybrid {
            gpu_layers: plan.gpu_layers,
            total_layers: desc.n_layer,
        }
    } else if plan.fits {
        Verdict::Cpu
    } else {
        Verdict::NoFit
    };

    // Suggestions based on verdict.
    match &verdict {
        Verdict::NoFit => {
            suggestions.push(Suggestion::TrySiblingQuant);
            suggestions.push(Suggestion::ReduceContext {
                recommended: config.planner.ctx_len / 2,
            });
            suggestions.push(Suggestion::UseKvQuantization {
                quant: "q8_0".to_string(),
            });
            suggestions.push(Suggestion::UseCloudApi);
        }
        Verdict::Cpu => {
            suggestions.push(Suggestion::TrySiblingQuant);
        }
        _ => {}
    }

    // Exit code.
    let exit_code = match &verdict {
        Verdict::NoFit => 2,
        _ if !warnings.is_empty() => 1,
        _ => 0,
    };

    FitReport {
        verdict,
        warnings,
        suggestions,
        plan,
        speed_gpu,
        speed_cpu,
        kv_estimate: kv,
        compute_estimate: compute,
        exit_code,
        media_tokens,
    }
}

/// Format a fit report as human-readable text.
pub fn format_report(report: &FitReport) -> String {
    let mut out = String::new();

    out.push_str(&format!("Verdict: {}\n", report.verdict));
    out.push_str(&format!("Exit code: {}\n", report.exit_code));
    out.push('\n');

    // Memory table.
    out.push_str("Memory:\n");
    out.push_str(&format!(
        "  GPU weights:  {:.1} MiB\n",
        report.plan.gpu_weight_bytes as f64 / 1048576.0
    ));
    out.push_str(&format!(
        "  CPU weights:  {:.1} MiB\n",
        report.plan.cpu_weight_bytes as f64 / 1048576.0
    ));
    out.push_str(&format!(
        "  Compute buf:  {:.1} MiB\n",
        report.plan.compute_buffer_bytes as f64 / 1048576.0
    ));
    out.push_str(&format!(
        "  KV cache:     {:.1} MiB\n",
        report.plan.kv_bytes as f64 / 1048576.0
    ));
    out.push_str(&format!(
        "  mmproj:       {:.1} MiB\n",
        report.plan.mmproj_bytes as f64 / 1048576.0
    ));
    out.push_str(&format!(
        "  lora:         {:.1} MiB\n",
        report.plan.lora_bytes as f64 / 1048576.0
    ));
    out.push_str(&format!(
        "  encoder:      {:.1} MiB\n",
        report.plan.encoder_compute_bytes as f64 / 1048576.0
    ));
    out.push_str(&format!(
        "  GPU total:    {:.1} MiB\n",
        report.plan.gpu_total_bytes as f64 / 1048576.0
    ));
    out.push_str(&format!("  media tokens: {}\n", report.media_tokens));
    out.push('\n');

    // Speed estimates.
    if let Some(ref sg) = report.speed_gpu {
        out.push_str(&format!(
            "GPU decode:  {:.1} tok/s  |  prefill: {:.1} tok/s  |  TTFT: {:.2}s\n",
            sg.decode_toks_per_sec, sg.prefill_toks_per_sec, sg.ttft_secs
        ));
    }
    if let Some(ref sc) = report.speed_cpu {
        out.push_str(&format!(
            "CPU decode:  {:.1} tok/s  |  prefill: {:.1} tok/s  |  TTFT: {:.2}s\n",
            sc.decode_toks_per_sec, sc.prefill_toks_per_sec, sc.ttft_secs
        ));
    }
    out.push('\n');

    // Warnings.
    if !report.warnings.is_empty() {
        out.push_str("Warnings:\n");
        for w in &report.warnings {
            out.push_str(&format!("  ⚠ {w}\n"));
        }
        out.push('\n');
    }

    // Suggestions.
    if !report.suggestions.is_empty() {
        out.push_str("Suggestions:\n");
        for s in &report.suggestions {
            out.push_str(&format!("  💡 {s}\n"));
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // Minimal integration tests — unit tests for the sub-modules
    // (descriptor, kv, compute, speed, planner) cover the math.
    // These test the report assembly and verdict logic.

    use crate::gguf::{GGUF_MAGIC, GgmlType};

    fn str_value(s: &str) -> Vec<u8> {
        let mut v = (s.len() as u64).to_le_bytes().to_vec();
        v.extend_from_slice(s.as_bytes());
        v
    }
    fn u64_value(v: u64) -> Vec<u8> {
        v.to_le_bytes().to_vec()
    }

    fn write_gguf(
        kv_pairs: &[(&str, u32, Vec<u8>)],
        tensors: &[(&str, &[u64], GgmlType)],
    ) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&GGUF_MAGIC);
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        b.extend_from_slice(&(kv_pairs.len() as u64).to_le_bytes());
        for &(key, ty, ref val) in kv_pairs {
            b.extend_from_slice(&(key.len() as u64).to_le_bytes());
            b.extend_from_slice(key.as_bytes());
            b.extend_from_slice(&ty.to_le_bytes());
            b.extend_from_slice(val);
        }
        for &(name, dims, ty) in tensors {
            b.extend_from_slice(&(name.len() as u64).to_le_bytes());
            b.extend_from_slice(name.as_bytes());
            b.extend_from_slice(&(dims.len() as u32).to_le_bytes());
            for &d in dims {
                b.extend_from_slice(&d.to_le_bytes());
            }
            let ty_raw: u32 = match ty {
                GgmlType::F32 => 0,
                GgmlType::F16 => 1,
                GgmlType::Q4_0 => 2,
                GgmlType::Q8_0 => 8,
                GgmlType::Q4_K => 12,
                _ => 0,
            };
            b.extend_from_slice(&ty_raw.to_le_bytes());
            b.extend_from_slice(&0u64.to_le_bytes());
        }
        while b.len() % 32 != 0 {
            b.push(0);
        }
        b
    }

    fn make_simple_desc() -> Descriptor {
        let kv: Vec<(String, u32, Vec<u8>)> = vec![
            ("general.architecture".to_string(), 8, str_value("qwen2")),
            ("qwen2.block_count".to_string(), 10, u64_value(24)),
            ("qwen2.embedding_length".to_string(), 10, u64_value(896)),
            ("qwen2.attention.head_count".to_string(), 10, u64_value(14)),
            ("qwen2.context_length".to_string(), 10, u64_value(4096)),
            ("qwen2.vocab_size".to_string(), 10, u64_value(151936)),
            (
                "qwen2.attention.head_count_kv".to_string(),
                10,
                u64_value(2),
            ),
            ("qwen2.attention.key_length".to_string(), 10, u64_value(64)),
        ];
        let embd_dims = [151936, 896];
        let tensors = [("token_embd.weight", embd_dims.as_slice(), GgmlType::Q4_K)];
        let buf = write_gguf(
            &kv.iter()
                .map(|(k, t, v)| (k.as_str(), *t, v.clone()))
                .collect::<Vec<_>>(),
            &tensors,
        );
        let r = crate::gguf::Reader::parse(&buf).unwrap();
        Descriptor::from_reader(&r).unwrap()
    }

    #[test]
    fn huge_gpu_fits() {
        let d = make_simple_desc();
        let config = FitConfig {
            planner: PlannerConfig {
                vram_bytes: 24 * 1024 * 1024 * 1024, // 24 GiB
                ..Default::default()
            },
            ..Default::default()
        };
        let report = check_fit(&d, &config);
        assert_eq!(report.exit_code, 0);
        assert!(matches!(
            report.verdict,
            Verdict::Gpu | Verdict::Hybrid { .. }
        ));
    }

    #[test]
    fn no_vram_no_fit() {
        let d = make_simple_desc();
        let config = FitConfig {
            planner: PlannerConfig {
                vram_bytes: 0,
                vram_margin: 0,
                ram_bytes: 0,
                ..Default::default()
            },
            ..Default::default()
        };
        let report = check_fit(&d, &config);
        assert_eq!(report.exit_code, 2);
        assert_eq!(report.verdict, Verdict::NoFit);
    }

    #[test]
    fn format_report_not_empty() {
        let d = make_simple_desc();
        let config = FitConfig::default();
        let report = check_fit(&d, &config);
        let text = format_report(&report);
        assert!(text.contains("Verdict:"));
        assert!(text.contains("Memory:"));
    }

    #[test]
    fn mmproj_warning() {
        let d = make_simple_desc();
        let config = FitConfig {
            has_mmproj: true,
            ..Default::default()
        };
        let report = check_fit(&d, &config);
        assert!(
            report
                .warnings
                .iter()
                .any(|w| matches!(w, Warning::MmprojNotCounted))
        );
    }

    #[test]
    fn vl_32_frames_predicts_context_need() {
        let d = make_simple_desc();
        let config = FitConfig {
            planner: PlannerConfig {
                ctx_len: 16384,
                mmproj_bytes: 80 << 20,
                vram_bytes: 24 * 1024 * 1024 * 1024,
                ..Default::default()
            },
            has_mmproj: true,
            media: MediaFit {
                frames: 32,
                tokens_per_frame: 256,
                ..Default::default()
            },
            ..Default::default()
        };
        let report = check_fit(&d, &config);
        assert_eq!(report.media_tokens, 32 * 256);
        assert_eq!(report.plan.mmproj_bytes, 80 << 20);
        assert!(report.plan.encoder_compute_bytes > 0);
        assert_eq!(report.exit_code, 0);
        assert!(
            !report
                .warnings
                .iter()
                .any(|w| matches!(w, Warning::MmprojNotCounted))
        );
        assert!(
            !report
                .warnings
                .iter()
                .any(|w| matches!(w, Warning::MediaExceedsContext { .. }))
        );
    }

    #[test]
    fn media_tokens_over_ctx_exit_2() {
        let d = make_simple_desc();
        let config = FitConfig {
            planner: PlannerConfig {
                ctx_len: 4096,
                vram_bytes: 24 * 1024 * 1024 * 1024,
                ..Default::default()
            },
            media: MediaFit {
                frames: 32,
                tokens_per_frame: 256,
                ..Default::default()
            },
            ..Default::default()
        };
        let report = check_fit(&d, &config);
        assert_eq!(report.media_tokens, 8192);
        assert_eq!(report.exit_code, 2);
        assert_eq!(report.verdict, Verdict::NoFit);
        assert!(report.warnings.iter().any(|w| matches!(
            w,
            Warning::MediaExceedsContext {
                need: 8192,
                n_ctx: 4096
            }
        )));
    }

    #[test]
    fn format_report_snapshot() {
        // Stable rendering of the default-config report (P1.11 verdict text,
        // memory table, speed lines). insta snapshot pins the exact layout.
        let d = make_simple_desc();
        let report = check_fit(&d, &FitConfig::default());
        insta::assert_snapshot!(format_report(&report));
    }

    #[test]
    fn no_fit_suggests_cloud() {
        let d = make_simple_desc();
        let config = FitConfig {
            planner: PlannerConfig {
                vram_bytes: 0,
                vram_margin: 0,
                ram_bytes: 0,
                ..Default::default()
            },
            ..Default::default()
        };
        let report = check_fit(&d, &config);
        assert!(
            report
                .suggestions
                .iter()
                .any(|s| matches!(s, Suggestion::UseCloudApi))
        );
    }
}
