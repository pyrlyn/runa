// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! Calibration database (P1.10).
//!
//! Stores measured inference performance to calibrate the speed model's
//! efficiency factors. Each sample records a model's actual throughput
//! on a specific device/backend/quantization combination.
//!
//! # Storage
//!
//! Uses a JSON file (`~/.runa/calibration.json`) by default. Each entry
//! stores:
//!
//! ```json
//! {
//!   "model_hash": "sha256:abcd...",
//!   "quant": "Q4_K",
//!   "placement": "gpu",
//!   "ctx": 4096,
//!   "measured_pp": 1200.5,
//!   "measured_tg": 45.2,
//!   "predicted_pp": 1100.0,
//!   "predicted_tg": 50.0,
//!   "device": "cuda:0",
//!   "backend": "cuda",
//!   "timestamp": "2026-09-08T12:00:00Z"
//! }
//! ```
//!
//! # Efficiency computation
//!
//! `eff(device, backend, quant)` = median of (measured / predicted) ratios
//! across all samples for that (device, backend, quant) triple.
//!
//! The speed model (P1.9) uses this efficiency to adjust predictions.

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

/// A single calibration sample.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CalibrationSample {
    /// Model hash (SHA-256 of first 1 MiB + last 1 MiB).
    pub model_hash: String,
    /// Quantization type (e.g. "Q4_K", "Q8_0").
    pub quant: String,
    /// Placement mode ("gpu", "cpu", "hybrid").
    pub placement: String,
    /// Context length used during measurement.
    pub ctx: u64,
    /// Measured prompt processing throughput (tok/s).
    pub measured_pp: f64,
    /// Measured text generation throughput (tok/s).
    pub measured_tg: f64,
    /// Predicted PP at time of measurement (from speed model).
    pub predicted_pp: f64,
    /// Predicted TG at time of measurement.
    pub predicted_tg: f64,
    /// Device identifier (e.g. "cuda:0", "metal:0", "cpu").
    pub device: String,
    /// Backend name (e.g. "cuda", "metal", "cpu", "vulkan").
    pub backend: String,
    /// ISO 8601 timestamp.
    pub timestamp: String,
}

/// Computed efficiency factor for a (device, backend, quant) triple.
#[derive(Debug, Clone, PartialEq)]
pub struct Efficiency {
    pub device: String,
    pub backend: String,
    pub quant: String,
    /// Median measured_pp / predicted_pp ratio.
    pub pp_efficiency: f64,
    /// Median measured_tg / predicted_tg ratio.
    pub tg_efficiency: f64,
    /// Number of samples used.
    pub sample_count: usize,
}

/// Calibration database.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct CalibrationDb {
    samples: Vec<CalibrationSample>,
}

impl CalibrationDb {
    /// Create an empty calibration database.
    pub fn new() -> Self {
        Self::default()
    }

    /// Load from a JSON file. Returns empty DB if file doesn't exist.
    pub fn load(path: &PathBuf) -> Self {
        match fs::read_to_string(path) {
            Ok(contents) => serde_json::from_str(&contents).unwrap_or_default(),
            Err(_) => Self::default(),
        }
    }

    /// Save to a JSON file. Creates parent directories if needed.
    pub fn save(&self, path: &PathBuf) -> Result<(), String> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|e| format!("create dir: {e}"))?;
        }
        let json = serde_json::to_string_pretty(self).map_err(|e| format!("serialize: {e}"))?;
        fs::write(path, json).map_err(|e| format!("write: {e}"))
    }

    /// Add a calibration sample.
    pub fn insert(&mut self, sample: CalibrationSample) {
        self.samples.push(sample);
    }

    /// Get all samples.
    pub fn samples(&self) -> &[CalibrationSample] {
        &self.samples
    }

    /// Number of samples in the database.
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Whether the database is empty.
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Compute efficiency factors per (device, backend, quant).
    pub fn compute_efficiencies(&self) -> Vec<Efficiency> {
        let mut groups: HashMap<(String, String, String), Vec<&CalibrationSample>> = HashMap::new();
        for sample in &self.samples {
            let key = (
                sample.device.clone(),
                sample.backend.clone(),
                sample.quant.clone(),
            );
            groups.entry(key).or_default().push(sample);
        }

        groups
            .into_iter()
            .map(|((device, backend, quant), samples)| {
                let pp_ratios: Vec<f64> = samples
                    .iter()
                    .filter(|s| s.predicted_pp > 0.0)
                    .map(|s| s.measured_pp / s.predicted_pp)
                    .collect();
                let tg_ratios: Vec<f64> = samples
                    .iter()
                    .filter(|s| s.predicted_tg > 0.0)
                    .map(|s| s.measured_tg / s.predicted_tg)
                    .collect();

                Efficiency {
                    pp_efficiency: median(&pp_ratios).unwrap_or(1.0),
                    tg_efficiency: median(&tg_ratios).unwrap_or(1.0),
                    sample_count: samples.len(),
                    device,
                    backend,
                    quant,
                }
            })
            .collect()
    }

    /// Get efficiency for a specific (device, backend, quant) triple.
    /// Returns None if no samples exist for that combination.
    pub fn get_efficiency(&self, device: &str, backend: &str, quant: &str) -> Option<Efficiency> {
        self.compute_efficiencies()
            .into_iter()
            .find(|e| e.device == device && e.backend == backend && e.quant == quant)
    }

    /// Clear all samples.
    pub fn clear(&mut self) {
        self.samples.clear();
    }
}

/// Compute the median of a slice. Returns None if empty.
fn median(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        Some((sorted[mid - 1] + sorted[mid]) / 2.0)
    } else {
        Some(sorted[mid])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_db() {
        let db = CalibrationDb::new();
        assert!(db.is_empty());
        assert_eq!(db.len(), 0);
        assert!(db.compute_efficiencies().is_empty());
    }

    #[test]
    fn insert_and_retrieve() {
        let mut db = CalibrationDb::new();
        db.insert(CalibrationSample {
            model_hash: "abc".to_string(),
            quant: "Q4_K".to_string(),
            placement: "gpu".to_string(),
            ctx: 4096,
            measured_pp: 1200.0,
            measured_tg: 45.0,
            predicted_pp: 1100.0,
            predicted_tg: 50.0,
            device: "cuda:0".to_string(),
            backend: "cuda".to_string(),
            timestamp: "2026-09-08T12:00:00Z".to_string(),
        });
        assert_eq!(db.len(), 1);
    }

    #[test]
    fn efficiency_calculation() {
        let mut db = CalibrationDb::new();
        // 3 samples with pp ratio = 1.2, 1.1, 1.3 → median = 1.2
        for (pp_pred, tg_pred) in [(1000.0, 40.0), (1100.0, 44.0), (900.0, 36.0)] {
            db.insert(CalibrationSample {
                model_hash: "abc".to_string(),
                quant: "Q4_K".to_string(),
                placement: "gpu".to_string(),
                ctx: 4096,
                measured_pp: pp_pred * 1.2,
                measured_tg: tg_pred * 1.1,
                predicted_pp: pp_pred,
                predicted_tg: tg_pred,
                device: "cuda:0".to_string(),
                backend: "cuda".to_string(),
                timestamp: "2026-09-08T12:00:00Z".to_string(),
            });
        }
        let eff = db.get_efficiency("cuda:0", "cuda", "Q4_K").unwrap();
        assert!((eff.pp_efficiency - 1.2).abs() < 0.01);
        assert!((eff.tg_efficiency - 1.1).abs() < 0.01);
        assert_eq!(eff.sample_count, 3);
    }

    #[test]
    fn save_and_load_roundtrip() {
        let mut db = CalibrationDb::new();
        db.insert(CalibrationSample {
            model_hash: "def".to_string(),
            quant: "Q8_0".to_string(),
            placement: "cpu".to_string(),
            ctx: 2048,
            measured_pp: 500.0,
            measured_tg: 20.0,
            predicted_pp: 450.0,
            predicted_tg: 22.0,
            device: "cpu".to_string(),
            backend: "cpu".to_string(),
            timestamp: "2026-09-08T13:00:00Z".to_string(),
        });

        let dir = std::env::temp_dir().join("runa_cal_test");
        let path = dir.join("calibration.json");
        db.save(&path).unwrap();

        let loaded = CalibrationDb::load(&path);
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded.samples()[0].model_hash, "def");

        // Clean up.
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn median_values() {
        assert_eq!(median(&[]), None);
        assert_eq!(median(&[5.0]), Some(5.0));
        assert_eq!(median(&[1.0, 3.0]), Some(2.0));
        assert_eq!(median(&[1.0, 2.0, 3.0]), Some(2.0));
        assert_eq!(median(&[3.0, 1.0, 2.0]), Some(2.0));
    }

    #[test]
    fn groups_by_device_backend_quant() {
        let mut db = CalibrationDb::new();
        // Two different devices.
        for device in ["cuda:0", "metal:0"] {
            db.insert(CalibrationSample {
                model_hash: "abc".to_string(),
                quant: "Q4_K".to_string(),
                placement: "gpu".to_string(),
                ctx: 4096,
                measured_pp: 1200.0,
                measured_tg: 45.0,
                predicted_pp: 1000.0,
                predicted_tg: 40.0,
                device: device.to_string(),
                backend: if device == "cuda:0" { "cuda" } else { "metal" }.to_string(),
                timestamp: "2026-09-08T12:00:00Z".to_string(),
            });
        }
        let effs = db.compute_efficiencies();
        assert_eq!(effs.len(), 2);
    }
}
