// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! P2.1: engine loads with a `Placement`, prints the verdict line, and the
//! engine tensor bytes match the P1 descriptor estimate within ±5 %.
//!
//! CPU-only (`--mode cpu` path). `--n-cpu-moe` patterns are applied on this
//! load (no-op on a dense 0.5B). GPU/hybrid offload paths are covered by
//! `placement` unit tests plus the `gen` example on Metal/CUDA builders.

use std::path::PathBuf;

use runa_engine::{EngineError, KvKind, LoadConfig, LoraSpec, Placement, load};
use runa_fit::{Descriptor, Reader, estimate_kv, read_local_prefix};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures")
        .join(name)
}

#[test]
fn missing_file_is_model_not_found() {
    match load(
        PathBuf::from("no-such-model.gguf").as_path(),
        &Placement::cpu(),
        &LoadConfig::default(),
    ) {
        Err(e) => assert!(e.to_string().contains("not found"), "{e}"),
        Ok(_) => panic!("missing file must fail"),
    }
}

#[cfg(not(feature = "rpc"))]
#[test]
fn rpc_servers_fail_unsupported_before_backend_init() {
    // P9.3: without the `rpc` feature the pinned sys crate has no ggml RPC
    // backend, so a non-empty list must fail explicitly (never silently run
    // locally). An empty stand-in file passes the `is_file` check; the guard
    // fires before backend init, so no model bytes are needed.
    let tmp = tempfile::NamedTempFile::new().expect("temp model stand-in");
    let placement = Placement::gpu().with_rpc_servers(vec!["127.0.0.1:50052".to_string()]);
    match load(tmp.path(), &placement, &LoadConfig::default()) {
        Err(EngineError::Unsupported(msg)) => assert!(msg.contains("--rpc"), "{msg}"),
        Err(e) => panic!("expected Unsupported, got: {e}"),
        Ok(_) => panic!("rpc without a backend must fail"),
    }
}

#[cfg(feature = "rpc")]
#[test]
fn rpc_servers_require_explicit_device_selection() {
    // P9.3: `--rpc` only registers remote servers; without `--device` the
    // servers would sit idle while tensors stay local — that silent non-use
    // is an error, and it fires before any network contact (no model bytes
    // needed, but the file must exist for the earlier `is_file` check).
    let tmp = tempfile::NamedTempFile::new().expect("temp model stand-in");
    let placement = Placement::gpu().with_rpc_servers(vec!["127.0.0.1:50052".to_string()]);
    match load(tmp.path(), &placement, &LoadConfig::default()) {
        Err(EngineError::BadDevices(msg)) => assert!(msg.contains("--device"), "{msg}"),
        Err(e) => panic!("expected BadDevices, got: {e}"),
        Ok(_) => panic!("rpc without --device must fail"),
    }
}

#[cfg(feature = "rpc")]
#[test]
fn rpc_unreachable_endpoint_fails_before_model_load() {
    // P9.3: a dead endpoint fails with `RpcUnreachable` while connecting —
    // before backend init touches global state or any weight loads.
    let tmp = tempfile::NamedTempFile::new().expect("temp model stand-in");
    let placement = Placement::gpu()
        .with_devices(vec!["RPC0".into()])
        .with_rpc_servers(vec!["127.0.0.1:9".to_string()]);
    match load(tmp.path(), &placement, &LoadConfig::default()) {
        Err(EngineError::RpcUnreachable(ep)) => assert!(ep.contains("127.0.0.1:9"), "{ep}"),
        Err(e) => panic!("expected RpcUnreachable, got: {e}"),
        Ok(_) => panic!("unreachable rpc server must fail"),
    }
}

#[test]
fn cpu_load_matches_p1_estimate() {
    let path = fixture("qwen2-0_5b-instruct-q4_0.gguf");
    assert!(path.is_file(), "P0.7 fixture missing: {}", path.display());

    // P1 side: descriptor weight total from the header prefix.
    let h = read_local_prefix(&path).expect("header prefix");
    let reader = Reader::parse(&h.bytes).expect("header parses");
    let desc = Descriptor::from_reader(&reader).expect("descriptor");
    assert!(desc.weight_bytes_total > 0);

    // Engine side: CPU placement + `--n-cpu-moe 3` overrides (no-op on this
    // dense 0.5B, but proves the FFI patterns survive load).
    let placement = Placement::cpu().with_n_cpu_moe(3);
    let loaded = load(&path, &placement, &LoadConfig::default()).expect("cpu load");
    assert_eq!(loaded.placement().cpu_patterns.len(), 1);
    assert!(loaded.n_params() > 0);
    assert_eq!(loaded.n_layer() as u64, desc.n_layer as u64);

    // P2.1 check: engine tensor bytes vs P1 estimate within ±5 %.
    let engine_bytes = loaded.size_bytes() as f64;
    let estimate_bytes = desc.weight_bytes_total as f64;
    let rel = (engine_bytes - estimate_bytes).abs() / estimate_bytes;
    assert!(
        rel <= 0.05,
        "engine {engine_bytes} vs P1 estimate {estimate_bytes} (rel {rel:.4})"
    );
}

#[test]
fn missing_lora_adapter_fails_before_backend_init() {
    let path = fixture("qwen2-0_5b-instruct-q4_0.gguf");
    assert!(path.is_file(), "P0.7 fixture missing: {}", path.display());
    let missing = PathBuf::from("no-such-adapter.gguf");
    match load(
        &path,
        &Placement::cpu(),
        &LoadConfig {
            loras: vec![LoraSpec {
                path: missing.clone(),
                scale: 1.0,
            }],
            ..LoadConfig::default()
        },
    ) {
        Err(e) => assert!(
            e.to_string().contains("no-such-adapter"),
            "adapter path in error: {e}"
        ),
        Ok(_) => panic!("missing adapter must fail"),
    }
}

#[test]
fn lora_adapter_loads_when_fixture_present() {
    // E2E (P8.5): drop a real LoRA GGUF for the qwen2 base into the
    // fixtures dir (`lora-*.gguf`) and this test loads it for real;
    // without one it passes as a skip.
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures");
    let mut adapters: Vec<PathBuf> = std::fs::read_dir(&dir)
        .expect("fixtures dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.extension().and_then(|e| e.to_str()) == Some("gguf")
                && p.file_name().and_then(|n| n.to_str()).is_some_and(|n| {
                    n.to_ascii_lowercase().contains("lora") && !n.starts_with("synthetic")
                })
        })
        .collect();
    adapters.sort();
    if adapters.is_empty() {
        return;
    }
    let base = fixture("qwen2-0_5b-instruct-q4_0.gguf");
    assert!(base.is_file(), "P0.7 fixture missing: {}", base.display());
    let spec = LoraSpec {
        path: adapters[0].clone(),
        scale: 0.5,
    };
    let mut loaded = load(
        &base,
        &Placement::cpu(),
        &LoadConfig {
            loras: vec![spec.clone()],
            ..LoadConfig::default()
        },
    )
    .expect("LoRA fixture loads against the qwen2 base");
    assert_eq!(loaded.config().loras, vec![spec]);
    // A fresh context keeps the adapter blend (P8.5 re-apply path).
    loaded.reset_context().expect("reset with adapters");
}

#[test]
fn quantized_kv_requires_flash_attn() {
    let path = fixture("qwen2-0_5b-instruct-q4_0.gguf");
    assert!(path.is_file(), "P0.7 fixture missing: {}", path.display());
    let denied = load(
        &path,
        &Placement::cpu(),
        &LoadConfig {
            flash_attn: false,
            kv_k: Some(KvKind::Q8_0),
            kv_v: Some(KvKind::Q8_0),
            ..LoadConfig::default()
        },
    );
    match denied {
        Err(EngineError::Unsupported(msg)) => {
            assert!(
                msg.contains("flash attention"),
                "unexpected unsupported: {msg}"
            );
        }
        Err(e) => panic!("expected Unsupported, got {e}"),
        Ok(_) => panic!("quantized KV without flash-attn must fail"),
    }
}

#[test]
fn cpu_rejects_device_and_tensor_split() {
    let path = fixture("qwen2-0_5b-instruct-q4_0.gguf");
    let split = load(
        &path,
        &Placement::cpu().with_tensor_split(vec![3.0, 1.0]),
        &LoadConfig::default(),
    );
    match split {
        Err(EngineError::BadDevices(msg)) => {
            assert!(msg.contains("tensor-split"), "{msg}");
        }
        Err(e) => panic!("expected BadDevices, got {e}"),
        Ok(_) => panic!("CPU + tensor-split must fail"),
    }
}

#[test]
fn quantized_kv_memory_drop_matches_p1() {
    let path = fixture("qwen2-0_5b-instruct-q4_0.gguf");
    assert!(path.is_file(), "P0.7 fixture missing: {}", path.display());
    let h = read_local_prefix(&path).expect("header prefix");
    let reader = Reader::parse(&h.bytes).expect("header parses");
    let desc = Descriptor::from_reader(&reader).expect("descriptor");

    let ctx = u64::from(LoadConfig::default().n_ctx);
    let f16 = load(&path, &Placement::cpu(), &LoadConfig::default()).expect("f16 kv load");
    let f16_bytes = f16.kv_cache_bytes() as f64;
    let f16_kv = estimate_kv(&desc, ctx, "f16").kv_bytes as f64;
    drop(f16);

    let q8 = load(
        &path,
        &Placement::cpu(),
        &LoadConfig {
            kv_k: Some(KvKind::Q8_0),
            kv_v: Some(KvKind::Q8_0),
            ..LoadConfig::default()
        },
    )
    .expect("q8_0 kv load");
    assert_eq!(q8.config().kv_k, Some(KvKind::Q8_0));
    assert_eq!(q8.config().kv_v, Some(KvKind::Q8_0));
    let q8_bytes = q8.kv_cache_bytes() as f64;
    let q8_kv = estimate_kv(&desc, ctx, "q8_0").kv_bytes as f64;
    assert!(
        f16_bytes > q8_bytes && q8_bytes > 0.0,
        "q8_0 KV must shrink ({f16_bytes} -> {q8_bytes})"
    );
    let measured_drop = f16_bytes - q8_bytes;
    let estimate_drop = f16_kv - q8_kv;
    assert!(estimate_drop > 0.0);
    let drop_rel = (measured_drop - estimate_drop).abs() / estimate_drop;
    assert!(
        drop_rel <= 0.05,
        "KV drop engine {measured_drop} vs P1 {estimate_drop} (rel {drop_rel:.4})"
    );
}

#[test]
#[ignore = "P2.9: needs two GPUs; see docs/baselines.md"]
fn two_gpu_tensor_split_loads() {
    let path = fixture("qwen2-0_5b-instruct-q4_0.gguf");
    assert!(path.is_file(), "P0.7 fixture missing: {}", path.display());
    let placement = Placement::gpu()
        .with_devices(vec!["0".into(), "1".into()])
        .with_tensor_split(vec![3.0, 1.0]);
    let loaded = load(&path, &placement, &LoadConfig::default()).expect("two-gpu load");
    assert_eq!(loaded.placement().devices.len(), 2);
    assert_eq!(loaded.placement().tensor_split, vec![3.0, 1.0]);
}
