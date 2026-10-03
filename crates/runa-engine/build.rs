// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! P5.7: remind that `--features native` needs RUSTFLAGS for GGML_NATIVE.
//!
//! P9.4: warn when an NPU stub feature (`hexagon`, `openvino`) is on but its
//! SDK env is missing. The features are probe-only stubs (no ggml backend in
//! llama-cpp-2 through 0.1.154 — see `docs/versions.md`); the warning keeps
//! that explicit at build time (plan D12: no silent acceleration claims).

fn main() {
    println!("cargo:rerun-if-env-changed=CARGO_ENCODED_RUSTFLAGS");
    println!("cargo:rerun-if-env-changed=HEXAGON_SDK_ROOT");
    println!("cargo:rerun-if-env-changed=INTEL_OPENVINO_DIR");
    println!("cargo:rerun-if-changed=rpc/");
    if std::env::var("CARGO_FEATURE_RPC").is_ok() {
        build_rpc_backend();
    }
    let feat = std::env::var("CARGO_FEATURE_NATIVE").is_ok();
    let flags = std::env::var("CARGO_ENCODED_RUSTFLAGS").unwrap_or_default();
    let rustc_native = flags
        .split('\u{1f}')
        .any(|f| f.contains("target-cpu=native"));
    if feat && !rustc_native {
        println!(
            "cargo:warning=feature `native` is on, but RUSTFLAGS does not contain              -C target-cpu=native; llama-cpp-sys-2 will keep GGML_NATIVE=OFF              (portable runtime dispatch). For a host-tuned build:              RUSTFLAGS='-C target-cpu=native' cargo build --release --features native"
        );
    }
    // P9.4: stub features with no ggml backend behind them. Warn twice: the
    // feature changes no placement (stub), and the vendor SDK is missing
    // (needed for any future real enablement + manual on-device validation).
    if std::env::var("CARGO_FEATURE_HEXAGON").is_ok() {
        println!(
            "cargo:warning=feature `hexagon` is a probe-only stub: llama-cpp-2 has               no `hexagon` backend through 0.1.154 (see docs/versions.md), so              placement stays CPU. Runtime validation is manual on-device (Tier 3)."
        );
        if std::env::var("HEXAGON_SDK_ROOT").is_err() {
            println!(
                "cargo:warning=feature `hexagon` is on but HEXAGON_SDK_ROOT is unset              (Qualcomm Hexagon SDK); install it before any on-device validation."
            );
        }
    }
    if std::env::var("CARGO_FEATURE_OPENVINO").is_ok() {
        println!(
            "cargo:warning=feature `openvino` is a probe-only stub: llama-cpp-2 has               no `openvino` backend through 0.1.154 (see docs/versions.md), so              placement stays CPU. Runtime validation is manual on-device (Tier 3)."
        );
        if std::env::var("INTEL_OPENVINO_DIR").is_err() {
            println!(
                "cargo:warning=feature `openvino` is on but INTEL_OPENVINO_DIR is unset              (OpenVINO setupvars.sh); source it before any on-device validation."
            );
        }
    }
}

/// P9.3: compile the vendored ggml RPC backend (`rpc/`, b7709 verbatim).
///
/// Pin-guard first: the vendored file and headers are byte-copies of the
/// exact commit `llama-cpp-sys-2 =0.1.133` builds, so a drifted pin with a
/// stale vendoring would be silent ABI risk — fail loudly instead (plan
/// D16; refresh rule in `rpc/README.md`).
fn build_rpc_backend() {
    const PINNED_SYS: &str = "0.1.133";
    let manifest = std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let lock = manifest.join("..").join("..").join("Cargo.lock");
    let text = std::fs::read_to_string(&lock)
        .unwrap_or_else(|e| panic!("rpc: read {}: {e}", lock.display()));
    let mut version = None;
    let mut lines = text.lines().peekable();
    while let Some(line) = lines.next() {
        if line.trim() == "name = \"llama-cpp-sys-2\"" {
            for next in lines.by_ref() {
                let next = next.trim();
                if let Some(v) = next.strip_prefix("version = \"") {
                    version = Some(v.trim_end_matches('"').to_owned());
                    break;
                }
                if next == "[[package]]" || next.is_empty() {
                    break;
                }
            }
        }
    }
    match version.as_deref() {
        Some(PINNED_SYS) => {}
        other => panic!(
            "rpc: vendored ggml-rpc.cpp is b7709 (llama-cpp-sys-2 =0.1.133), \
             but Cargo.lock pins {other:?}; refresh rpc/ per rpc/README.md, then bump this guard"
        ),
    }
    // Same settings as upstream `ggml_add_backend_library(ggml-rpc ...)`: one
    // TU, C++17 ("don't bump"), private include of ggml sources for the
    // `-impl.h` headers. Only extra link: `ws2_32` on Windows (sockets).
    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .file("rpc/ggml/src/ggml-rpc/ggml-rpc.cpp")
        .include("rpc/ggml/include")
        .include("rpc/ggml/src")
        .warnings(false)
        .compile("ggml-rpc");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        println!("cargo:rustc-link-lib=ws2_32");
    }
}
