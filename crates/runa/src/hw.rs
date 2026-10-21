// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! This machine's memory as `fit`, the placement policy, `serve` and
//! `daemon` see it. A leaf module: it reads the host and the test env
//! hooks, nothing else in the crate, so every caller can import it without
//! a cycle (issue #31).

use runa_fit::{FitConfig, HwSpec, PlannerConfig};

pub(crate) fn vram_bytes() -> Result<(u64, &'static str), String> {
    if let Ok(s) = std::env::var("RUNA_FAKE_VRAM") {
        let mib: u64 = s
            .parse()
            .map_err(|_| format!("RUNA_FAKE_VRAM={s}: expected integer MiB"))?;
        return Ok((mib.saturating_mul(1024 * 1024), "RUNA_FAKE_VRAM"));
    }
    #[cfg(target_os = "macos")]
    {
        let mut sys = sysinfo::System::new();
        sys.refresh_memory();
        // Unified memory stand-in until the P1.6 wired_limit probe is wired here.
        let assumed = sys.total_memory().saturating_mul(3) / 4;
        Ok((assumed, "macos-unified-75pct"))
    }
    #[cfg(not(target_os = "macos"))]
    {
        Ok((0, "unknown"))
    }
}

pub(crate) fn ram_bytes() -> u64 {
    // Test hook like RUNA_FAKE_VRAM: forces a CPU NO FIT on big hosts.
    if let Some(mib) = std::env::var("RUNA_FAKE_RAM")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
    {
        return mib.saturating_mul(1024 * 1024);
    }
    let mut sys = sysinfo::System::new();
    sys.refresh_memory();
    sys.total_memory()
}

pub(crate) fn memory_ceiling_mib() -> u64 {
    if let Ok(s) = std::env::var("RUNA_MEMORY_CEILING_MIB")
        && let Ok(n) = s.parse::<u64>()
    {
        return n;
    }
    ram_bytes() / (1024 * 1024)
}

/// This machine as the fit checker sees it.
pub(crate) fn machine_config(
    ctx: u32,
    kv_type: &str,
    mmproj_bytes: u64,
) -> Result<(FitConfig, &'static str), String> {
    let (vram, vram_src) = vram_bytes()?;
    let gpu_hw = (vram > 0).then(|| {
        if cfg!(target_os = "macos") {
            HwSpec::metal()
        } else {
            HwSpec::cuda()
        }
    });
    let config = FitConfig {
        planner: PlannerConfig {
            vram_bytes: vram,
            ram_bytes: ram_bytes(),
            ctx_len: u64::from(ctx),
            kv_type: kv_type.to_owned(),
            mmproj_bytes,
            ..PlannerConfig::default()
        },
        gpu_hw,
        cpu_hw: HwSpec::cpu(),
        has_mmproj: false,
        media: runa_fit::MediaFit::default(),
    };
    Ok((config, vram_src))
}
