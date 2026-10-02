// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! P9.3 loopback test (`--features rpc`): a minimal fake `rpc-server` that
//! answers just `HELLO` (protocol 3.6.0) and `DEVICE_COUNT` proves the full
//! client path — parse → `ggml_backend_rpc_add_server` → global registry →
//! `RPCn` enumeration — without shipping upstream's `rpc-server` binary.
//!
//! Wire format (from the vendored b7709 `ggml-rpc.cpp`):
//! request `u8 cmd | u64 size | size bytes`, response `u64 size | size bytes`,
//! native (little-endian) integers. `HELLO = 14`, `DEVICE_COUNT = 15`.

#![cfg(feature = "rpc")]

use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::mpsc;
use std::thread;

use runa_engine::rpc::{device_name_for, register_servers, registered_device_names};

const RPC_CMD_HELLO: u8 = 14;
const RPC_CMD_DEVICE_COUNT: u8 = 15;
const PROTO: [u8; 3] = [3, 6, 0];

/// Serve exactly `device_count` for every `DEVICE_COUNT` request until the
/// client goes away. Returns the endpoint string (`127.0.0.1:port`).
fn fake_rpc_server(device_count: u32) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback");
    let endpoint = listener.local_addr().expect("loopback addr").to_string();
    let (ready_tx, ready_rx) = mpsc::channel();
    thread::spawn(move || {
        ready_tx.send(()).expect("ready");
        // One registration opens one connection (sockets are cached per
        // endpoint inside ggml); serve it until EOF, then exit.
        let Ok((mut stream, _)) = listener.accept() else {
            return;
        };
        loop {
            let mut cmd = [0u8; 1];
            if stream.read_exact(&mut cmd).is_err() {
                return;
            }
            let mut size = [0u8; 8];
            if stream.read_exact(&mut size).is_err() {
                return;
            }
            let size = u64::from_le_bytes(size) as usize;
            let mut body = vec![0u8; size];
            if stream.read_exact(&mut body).is_err() {
                return;
            }
            let payload: &[u8] = match cmd[0] {
                RPC_CMD_HELLO => &PROTO,
                RPC_CMD_DEVICE_COUNT => &device_count.to_le_bytes(),
                other => panic!("fake rpc-server: unexpected cmd {other}"),
            };
            let len = payload.len() as u64;
            if stream.write_all(&len.to_le_bytes()).is_err() {
                return;
            }
            if stream.write_all(payload).is_err() {
                return;
            }
        }
    });
    ready_rx.recv().expect("server thread ready");
    endpoint
}

/// Bind-then-drop: connecting to this port is refused.
fn closed_endpoint() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback");
    let endpoint = listener.local_addr().expect("loopback addr").to_string();
    drop(listener);
    endpoint
}

#[test]
fn unreachable_endpoint_errors_explicitly() {
    let endpoint = closed_endpoint();
    match register_servers(&[endpoint.clone()]) {
        Err(runa_engine::EngineError::RpcUnreachable(ep)) => assert_eq!(ep, endpoint),
        Err(e) => panic!("expected RpcUnreachable, got: {e}"),
        Ok(()) => panic!("closed port must fail"),
    }
}

#[test]
fn loopback_server_registers_an_rpc_device() {
    let endpoint = fake_rpc_server(1);
    register_servers(&[endpoint.clone()]).expect("loopback registration");
    let names = registered_device_names();
    assert!(
        names.iter().any(|n| n.starts_with("RPC")),
        "RPC device must enumerate, got: {names:?}"
    );
    let name = device_name_for(&endpoint)
        .unwrap_or_else(|| panic!("device for {endpoint} must enumerate in {names:?}"));
    assert!(name.starts_with("RPC"), "{name}");
}
