// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! NPU presence probe (plan P9.4, Tier 3).
//!
//! `llama-cpp-2` exposes no `hexagon`/`openvino` cargo features through
//! 0.1.154 (see `docs/versions.md`), so there is no ggml NPU backend to
//! dispatch to yet. This module is the opt-in scaffolding for the day one
//! lands: which NPU (if any) is present, conservative [`HwSpec`] defaults
//! (see [`HwSpec::hexagon`](crate::speed::HwSpec::hexagon) and
//! [`HwSpec::openvino`](crate::speed::HwSpec::openvino)), and the test
//! hooks. Placement never defaults to NPU (plan D12: no silent fallback) —
//! callers only consult this module when the user explicitly opts in
//! (`RUNA_NPU=hexagon|openvino` in the CLI).
//!
//! # Test hook
//!
//! `RUNA_FAKE_NPU` forces the probe result without hardware:
//! `1`/`hexagon` → [`NpuKind::Hexagon`], `openvino` → [`NpuKind::OpenVino`],
//! `0`/`no`/`off`/`none` → absent. Unset (or unrecognized) falls back to
//! the hardware heuristic below.
//!
//! # Hardware heuristic (conservative, unvalidated on-device)
//!
//! Linux only; every other OS reports absent until an on-device owner
//! validates a marker there:
//!
//! - `/proc/device-tree/compatible` contains `qcom` → Hexagon (Snapdragon
//!   Linux laptops expose the Qualcomm device tree).
//! - `/dev/accel` exists → OpenVINO (the accel subsystem hosts the Intel
//!   `ivpu` NPU; other accel devices would also match — hence Tier 3).
//!
//! Markers are injectable via [`probe_markers`] so unit tests never touch
//! the real filesystem or environment.

use std::path::Path;

use crate::speed::HwSpec;

/// Which NPU family was detected (or faked).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NpuKind {
    /// Qualcomm Hexagon DSP/NPU (Snapdragon X-class).
    Hexagon,
    /// Intel NPU driven through OpenVINO (Core Ultra / Lunar Lake class).
    OpenVino,
}

impl NpuKind {
    /// Canonical name (`hexagon`, `openvino`).
    pub fn as_str(self) -> &'static str {
        match self {
            NpuKind::Hexagon => "hexagon",
            NpuKind::OpenVino => "openvino",
        }
    }

    /// Parse a user-supplied name (`RUNA_NPU` values, `RUNA_FAKE_NPU`
    /// names). Accepts `1` as an alias for `hexagon`.
    pub fn parse(s: &str) -> Option<NpuKind> {
        match s.trim().to_ascii_lowercase().as_str() {
            "1" | "hexagon" | "qcom" | "snapdragon" => Some(NpuKind::Hexagon),
            "openvino" | "ov" | "intel-npu" => Some(NpuKind::OpenVino),
            _ => None,
        }
    }

    /// Conservative speed stub for this NPU family (Tier 3, uncalibrated).
    pub fn hw_spec(self) -> HwSpec {
        match self {
            NpuKind::Hexagon => HwSpec::hexagon(),
            NpuKind::OpenVino => HwSpec::openvino(),
        }
    }
}

impl std::fmt::Display for NpuKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Parse `RUNA_FAKE_NPU`: `Some(Some(kind))` forces present,
/// `Some(None)` forces absent, `None` means "no override, probe hardware".
fn fake_override() -> Option<Option<NpuKind>> {
    let raw = std::env::var("RUNA_FAKE_NPU").ok()?;
    match raw.trim().to_ascii_lowercase().as_str() {
        "" => None,
        "0" | "no" | "off" | "none" => Some(None),
        // Unrecognized: fall back to the hardware heuristic (documented
        // in the module docs; the hook is test-only).
        named => NpuKind::parse(named).map(Some),
    }
}

/// Hardware heuristic with injectable marker paths (see module docs).
/// Pure marker logic (no OS gate) so unit tests exercise it on every host;
/// the OS gate lives in [`npu_present`].
pub fn probe_markers(device_tree_compatible: &Path, accel_dir: &Path) -> Option<NpuKind> {
    if std::fs::read_to_string(device_tree_compatible)
        .map(|c| c.to_ascii_lowercase().contains("qcom"))
        .unwrap_or(false)
    {
        return Some(NpuKind::Hexagon);
    }
    if accel_dir.exists() {
        return Some(NpuKind::OpenVino);
    }
    None
}

/// Which NPU (if any) is present: `RUNA_FAKE_NPU` first, else the
/// conservative hardware heuristic (Linux markers only — every other OS
/// reports absent until an on-device owner validates a marker there).
pub fn npu_present() -> Option<NpuKind> {
    if let Some(forced) = fake_override() {
        return forced;
    }
    if cfg!(target_os = "linux") {
        probe_markers(
            Path::new("/proc/device-tree/compatible"),
            Path::new("/dev/accel"),
        )
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static MARKER_SEQ: AtomicU64 = AtomicU64::new(0);

    /// Scratch dir under the system temp dir (no new dev-deps for this).
    struct Scratch {
        dir: std::path::PathBuf,
    }

    impl Scratch {
        fn new() -> Scratch {
            let n = MARKER_SEQ.fetch_add(1, Ordering::Relaxed);
            let dir =
                std::env::temp_dir().join(format!("runa-npu-probe-{}-{}", std::process::id(), n));
            std::fs::create_dir_all(&dir).expect("scratch dir");
            Scratch { dir }
        }

        fn path(&self, name: &str) -> std::path::PathBuf {
            self.dir.join(name)
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    #[test]
    fn parse_names() {
        assert_eq!(NpuKind::parse("hexagon"), Some(NpuKind::Hexagon));
        assert_eq!(NpuKind::parse("1"), Some(NpuKind::Hexagon));
        assert_eq!(NpuKind::parse("QCom"), Some(NpuKind::Hexagon));
        assert_eq!(NpuKind::parse("openvino"), Some(NpuKind::OpenVino));
        assert_eq!(NpuKind::parse("ov"), Some(NpuKind::OpenVino));
        assert_eq!(NpuKind::parse("cuda"), None);
        assert_eq!(NpuKind::parse(""), None);
        assert_eq!(NpuKind::Hexagon.as_str(), "hexagon");
        assert_eq!(NpuKind::OpenVino.to_string(), "openvino");
    }

    #[test]
    fn hw_spec_matches_speed_stubs() {
        assert_eq!(NpuKind::Hexagon.hw_spec(), HwSpec::hexagon());
        assert_eq!(NpuKind::OpenVino.hw_spec(), HwSpec::openvino());
    }

    #[test]
    fn no_markers_means_absent() {
        let s = Scratch::new();
        assert_eq!(probe_markers(&s.path("compatible"), &s.path("accel")), None);
    }

    #[test]
    fn qcom_compatible_means_hexagon() {
        let s = Scratch::new();
        let compat = s.path("compatible");
        std::fs::write(&compat, "qcom,x1e80100-crd,qcom,x1e80100").expect("write compat");
        assert_eq!(
            probe_markers(&compat, &s.path("accel")),
            Some(NpuKind::Hexagon)
        );
    }

    #[test]
    fn accel_dir_means_openvino() {
        let s = Scratch::new();
        let accel = s.path("accel");
        std::fs::create_dir_all(&accel).expect("mkdir accel");
        assert_eq!(
            probe_markers(&s.path("compatible"), &accel),
            Some(NpuKind::OpenVino)
        );
    }

    #[test]
    fn hexagon_wins_when_both_markers_present() {
        let s = Scratch::new();
        let compat = s.path("compatible");
        let accel = s.path("accel");
        std::fs::write(&compat, "qcom,sc8280xp").expect("write compat");
        std::fs::create_dir_all(&accel).expect("mkdir accel");
        assert_eq!(probe_markers(&compat, &accel), Some(NpuKind::Hexagon));
    }
}
