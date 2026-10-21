// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! `runa doctor`: what this binary was built with.

/// Cargo features compiled into this binary (plan D13 / P6.3).
fn compiled_backends() -> Vec<&'static str> {
    let mut out = vec!["cpu"];
    if cfg!(feature = "metal") {
        out.push("metal");
    }
    if cfg!(feature = "cuda") {
        out.push("cuda");
    }
    if cfg!(feature = "vulkan") {
        out.push("vulkan");
    }
    // P9.4: probe-only stubs (Tier 3) — reported only when built with the
    // feature; default binaries never list NPU capabilities (plan D12).
    if cfg!(feature = "hexagon") {
        out.push("hexagon-stub");
    }
    if cfg!(feature = "openvino") {
        out.push("openvino-stub");
    }
    if cfg!(feature = "mtmd") {
        out.push("mtmd");
    }
    // P9.2 safetensors backend (`--backend mistral`).
    if cfg!(feature = "mistralrs") {
        out.push("mistralrs");
    }
    out
}

pub(super) fn doctor(json: bool) {
    let native_build = cfg!(feature = "native");
    let backends = compiled_backends();
    // P9.3: the vendored ggml RPC backend compiles only with `--features
    // rpc` (see `docs/versions.md`); reported explicitly either way.
    let rpc = cfg!(feature = "rpc");
    if json {
        let payload = serde_json::json!({
            "backends": backends,
            "native_build": native_build,
            "rpc": rpc,
        });
        println!("{payload}");
    } else {
        println!("runa doctor");
        println!(
            "native build (-C target-cpu=native): {}",
            if native_build {
                "yes"
            } else {
                "no (portable; ggml runtime dispatch)"
            }
        );
        println!("backends compiled in: {}", backends.join(", "));
        if rpc {
            println!("rpc: ggml RPC backend compiled in (`--rpc host:port` + `--device RPC0`)");
        } else {
            println!(
                "rpc: no ggml RPC backend in this build (rebuild with --features rpc; `--rpc` errors explicitly)"
            );
        }
        if backends.iter().any(|b| b.ends_with("-stub")) {
            println!(
                "NPU entries are probe-only stubs (Tier 3): no ggml backend in llama-cpp-2; \
                 placement stays CPU"
            );
        }
    }
}
