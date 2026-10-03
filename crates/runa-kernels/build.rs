// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

fn main() {
    println!("cargo:rerun-if-changed=zig/kernels.zig");

    let out = std::env::var("OUT_DIR").expect("OUT_DIR");
    let dest = std::path::Path::new(&out).join("libruna_kernels_zig.a");
    let zig = std::env::var("ZIG").unwrap_or_else(|_| "zig".into());
    let mut cmd = std::process::Command::new(&zig);
    cmd.args([
        "build-lib",
        "zig/kernels.zig",
        "-OReleaseFast",
        "-fPIC",
        &format!("-femit-bin={}", dest.display()),
    ]);
    // Windows/MSVC link needs CRT-provided `__chkstk`, but Zig's native
    // target detection emits MinGW-style `___chkstk_ms` probes (LNK2019).
    // The explicit triple emits `__chkstk` instead (verified via nm);
    // unix builds stay on native detection (no glibc-floor surprises).
    if std::env::var("CARGO_CFG_WINDOWS").is_ok() {
        cmd.arg("-target").arg("x86_64-windows-msvc");
    }
    // Zig tunes for the build host by default (AVX-512 on some CI runners).
    // A cached or released library then dies with SIGILL on an older CPU,
    // so stay on the baseline ISA unless Rust itself targets the host (P5.7).
    let native = std::env::var("CARGO_ENCODED_RUSTFLAGS")
        .is_ok_and(|f| f.split('\x1f').any(|f| f.contains("target-cpu=native")));
    cmd.arg(if native {
        "-mcpu=native"
    } else {
        "-mcpu=baseline"
    });
    let status = cmd
        .status()
        .unwrap_or_else(|e| panic!("zig 0.16.0 (mise, D23): {e}"));
    if !status.success() {
        panic!("zig build-lib failed: {status}");
    }

    println!("cargo:rustc-link-search=native={out}");
    println!("cargo:rustc-link-lib=static=runa_kernels_zig");
}
