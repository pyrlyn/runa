// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! The placement policy `serve` and `daemon` give their model pool (P16.2):
//! fit check, then `--mode` placement with the CLI overrides applied. It
//! lives here, not in `runa-pool`, because it needs the config, memory
//! policy and fit machinery that only the binary carries.

use std::path::Path;
use std::sync::Arc;

use runa_engine::{LoadConfig, Mode, Placement};
use runa_fit::{
    Descriptor, FitConfig, HwSpec, PlannerConfig, Reader, check_fit, read_local_prefix,
};
use runa_pool::Placer;

/// Parsed `--device/--tensor-split/--main-gpu/--rpc` overrides for
/// `serve` (and the daemon later): empty = no overrides.
#[derive(Debug, Clone, Default)]
pub(crate) struct PlacementOverrides {
    pub devices: Vec<String>,
    pub tensor_split: Vec<f32>,
    pub main_gpu: Option<i32>,
    pub rpc_servers: Vec<String>,
}

impl PlacementOverrides {
    pub fn apply(&self, mut placement: Placement) -> Placement {
        if !self.devices.is_empty() {
            placement = placement.with_devices(self.devices.clone());
        }
        if !self.tensor_split.is_empty() {
            placement = placement.with_tensor_split(self.tensor_split.clone());
        }
        if let Some(n) = self.main_gpu {
            placement = placement.with_main_gpu(n);
        }
        if !self.rpc_servers.is_empty() {
            placement = placement.with_rpc_servers(self.rpc_servers.clone());
        }
        placement
    }
}

/// The pool's [`Placer`] for `--mode` (`cpu|gpu|hybrid|auto`). A bad mode
/// fails here, so a typo is caught at startup, before the port is bound.
/// `max_load_percent` is the CLI `--max-load-percent` for the demand check
/// (`None` = env/config, default 80); the overrides apply on top of both
/// fixed and `auto` placements so an explicit flag is never dropped
/// silently (plan D12).
pub(crate) fn local_placer(
    mode: &str,
    overrides: PlacementOverrides,
    max_load_percent: Option<u8>,
) -> Result<Placer, String> {
    let choice = crate::parse_mode_choice(mode)?;
    let base = match choice {
        crate::ModeChoice::Fixed(m) => Placement::from_mode(m),
        crate::ModeChoice::Auto => Placement::from_mode(Mode::Cpu),
    };
    let auto = matches!(choice, crate::ModeChoice::Auto);
    Ok(Arc::new(move |path, config| {
        fit_check_no_fit(path, config.n_ctx, lora_bytes(config))?;
        let kv_type = runa_engine::planner_kv_type(config.kv_k, config.kv_v);
        if !auto {
            crate::preflight_grow(path, config.n_ctx, kv_type, 0, max_load_percent)?;
            return Ok(overrides.apply(base.clone()));
        }
        match crate::auto_placement(
            path,
            config.n_ctx,
            &crate::config::OnUnfit::Error,
            kv_type,
            None,
            None,
            &config.loras,
        )? {
            crate::AutoPlacement::Local(p) => Ok(overrides.apply(p)),
            crate::AutoPlacement::Cloud(_) => {
                Err("unfit: auto mode chose cloud fallback; serve is local-only".into())
            }
        }
    }))
}

fn fit_check_no_fit(path: &Path, ctx: u32, lora_bytes: u64) -> Result<(), String> {
    let header = read_local_prefix(path).map_err(|e| e.to_string())?;
    let reader = Reader::parse(&header.bytes).map_err(|e| e.to_string())?;
    let desc = Descriptor::from_reader(&reader).map_err(|e| e.to_string())?;
    let (vram, _) = crate::vram_bytes()?;
    let planner = PlannerConfig {
        vram_bytes: vram,
        ram_bytes: crate::ram_bytes(),
        ctx_len: u64::from(ctx),
        kv_type: runa_engine::planner_kv_type(None, None).to_owned(),
        lora_bytes,
        ..PlannerConfig::default()
    };
    let report = check_fit(
        &desc,
        &FitConfig {
            planner,
            gpu_hw: None,
            cpu_hw: HwSpec::cpu(),
            has_mmproj: false,
            media: runa_fit::MediaFit::default(),
        },
    );
    if matches!(report.verdict, runa_fit::Verdict::NoFit) {
        return Err(format!("unfit: model does not fit (ctx={ctx})"));
    }
    Ok(())
}

/// Adapter bytes summed from the configured `--lora` files (P8.5).
fn lora_bytes(config: &LoadConfig) -> u64 {
    config
        .loras
        .iter()
        .map(|s| runa_fit::mmproj_file_bytes(&s.path))
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn placement_overrides_apply_on_top_of_any_mode() {
        // P2.9/P9.3: explicit serve flags survive both fixed and `auto`
        // placements (plan D12 — never dropped silently).
        let base = Placement::gpu();
        let full = PlacementOverrides {
            devices: vec!["0".into()],
            tensor_split: vec![3.0, 1.0],
            main_gpu: Some(1),
            rpc_servers: vec!["127.0.0.1:50052".into()],
        }
        .apply(base);
        assert_eq!(full.devices, vec!["0"]);
        assert_eq!(full.tensor_split, vec![3.0, 1.0]);
        assert_eq!(full.main_gpu, 1);
        assert_eq!(full.rpc_servers, vec!["127.0.0.1:50052"]);
        assert_eq!(full.n_gpu_layers, u32::MAX);

        let empty = PlacementOverrides::default().apply(Placement::cpu());
        assert_eq!(empty, Placement::cpu());
    }

    #[test]
    fn a_bad_mode_fails_at_construction() {
        let err = local_placer("warp", PlacementOverrides::default(), None)
            .err()
            .expect("bad mode");
        assert!(err.contains("--mode must be"), "{err}");
    }
}
