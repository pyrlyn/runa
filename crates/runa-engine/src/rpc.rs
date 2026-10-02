// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! llama.cpp RPC backend registration (plan P9.3, `--features rpc`).
//!
//! Upstream `llama-cpp-sys-2 0.1.133` strips the `ggml-rpc/` sources, so the
//! `rpc` cargo feature compiles the vendored b7709 file (`rpc/`, verbatim)
//! in `build.rs` and this module registers each `--rpc` endpoint through the
//! public C API before backend init — mirroring `common/arg.cpp`'s
//! `add_rpc_devices`. Registered servers appear as `RPC0`, `RPC1`, … devices
//! and are selected explicitly with `--device` (never implicitly).
//!
//! Without the feature this module does not exist and [`crate::load`]
//! rejects `--rpc` with `EngineError::Unsupported`.

use std::ffi::{CString, c_char, c_void};

use crate::load::EngineError;

/// Opaque ggml backend registry handle (`ggml_backend_reg_t`).
type RpcReg = *mut c_void;

unsafe extern "C" {
    /// Connect to one `rpc-server` endpoint; `NULL` when unreachable
    /// (declared in the shipped `ggml-rpc.h`).
    fn ggml_backend_rpc_add_server(endpoint: *const c_char) -> RpcReg;
    /// Publish the handle in the process-global backend registry so its
    /// `RPCn` devices enumerate (declared in `ggml-backend.h`).
    fn ggml_backend_register(reg: RpcReg);
}

/// Device-name prefix assigned by `ggml_backend_rpc_add_server`
/// (`RPC0`, `RPC1`, … across calls, per-endpoint device counts appended).
pub const RPC_DEVICE_PREFIX: &str = "RPC";

/// Register every endpoint and publish its devices. Runs before backend init
/// in [`crate::load`]; an unreachable endpoint fails here — never a silent
/// local run while the caller asked for distributed inference.
pub fn register_servers(servers: &[String]) -> Result<(), EngineError> {
    for endpoint in servers {
        let c = CString::new(endpoint.as_str())
            .map_err(|_| EngineError::BadDevices(format!("--rpc: bad endpoint {endpoint:?}")))?;
        // SAFETY: `c` is a live NUL-terminated C string; the callee copies
        // the endpoint into its per-server registry context.
        let reg = unsafe { ggml_backend_rpc_add_server(c.as_ptr()) };
        if reg.is_null() {
            return Err(EngineError::RpcUnreachable(endpoint.clone()));
        }
        // SAFETY: `reg` came from `ggml_backend_rpc_add_server` (same
        // b7709 sources, same ABI) and is non-null.
        unsafe { ggml_backend_register(reg) };
        eprintln!("rpc: registered {endpoint}");
    }
    Ok(())
}

/// Names of currently enumerated RPC devices (`RPC0`, …), for error hints.
pub fn registered_device_names() -> Vec<String> {
    llama_cpp_2::list_llama_ggml_backend_devices()
        .iter()
        .map(|d| d.name.clone())
        .filter(|n| n.starts_with(RPC_DEVICE_PREFIX))
        .collect()
}

/// Name assigned to one endpoint's device (`RPCn`), if it enumerated.
/// `load()` logs this so `--device` needs no guessing.
pub fn device_name_for(endpoint: &str) -> Option<String> {
    llama_cpp_2::list_llama_ggml_backend_devices()
        .iter()
        .find(|d| d.name.starts_with(RPC_DEVICE_PREFIX) && d.description == endpoint)
        .map(|d| d.name.clone())
}
