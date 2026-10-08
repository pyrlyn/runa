// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! `runa-fit`: GGUF header reader, hardware probe, estimator, planner,
//! calibration DB (plan P1).
//!
//! The reader core (`gguf`) is zero-copy over `&[u8]` and `no_std`-friendly:
//! it only needs `core`/`alloc` and does no I/O itself. `remote` (P1.2)
//! fetches the header prefix over HTTP ranges or from disk; the hardware
//! probe (P1.6), estimator/planner (P1.8–P1.9) and calibration DB (P1.10)
//! build on top of both.

pub mod calibration;
pub mod compute;
pub mod descriptor;
pub mod ggml_types;
pub mod gguf;
pub mod kv;
mod mmproj;
pub mod npu;
pub mod planner;
pub mod recommend;
pub mod remote;
pub mod speed;
pub mod verdict;

pub use compute::{estimate_compute, estimate_encoder_compute};
pub use descriptor::Descriptor;
pub use ggml_types::{TypeInfo, n_elements, tensor_bytes, type_info};
pub use gguf::{DataType, GGUF_MAGIC, ReadError, Reader, TensorInfo, Value};
pub use kv::{KvEstimate, estimate_kv};
pub use mmproj::{mmproj_file_bytes, sibling_mmproj};
pub use npu::{NpuKind, npu_present, probe_markers};
pub use planner::{PlacementPlan, PlannerConfig, plan_placement};
pub use remote::{
    Fetcher, FileMeta, HeaderBytes, HfRef, MAX_HEADER_BYTES, ModelSource, RemoteError, START_BYTES,
    cache_root, is_safetensors_tag, parse_model_ref, pick_quant, read_local_prefix, resolve_hf_url,
};
pub use speed::{
    HwSpec, SpeedEstimate, active_weight_bytes, apply_efficiency, estimate_speed_single,
};
pub use verdict::{FitConfig, FitReport, MediaFit, Verdict, check_fit, format_report};
