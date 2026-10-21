// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! `runa web`: the usage journal as a page on this machine.
//!
//! The page is the built React app in `web/dist` (see `web/README.md`).
//! `GET /api/snapshot` is the same JSON as `runa stats --json`. `GET /ws`
//! pushes that JSON when the journal file changes, and again about once a
//! second so a quiet page still shows the socket is open.

use std::path::Path;
use std::time::{Duration, SystemTime};

use axum::extract::Request;
use axum::extract::ws::{WebSocket, WebSocketUpgrade};
use axum::http::{HeaderMap, StatusCode, header};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use serde_json::{Value, json};
use tokio::sync::watch;

const INDEX_HTML: &str = include_str!("../../../web/dist/index.html");
const STATS_JS: &str = include_str!("../../../web/dist/stats.js");
const STATS_CSS: &str = include_str!("../../../web/dist/stats.css");

const POLL: Duration = Duration::from_millis(500);

pub(crate) fn cmd_web(host: &str, port: u16) -> Result<(), String> {
    if !crate::serve::is_loopback_host(host) {
        return Err(
            "runa web only binds a loopback address (127.0.0.0/8 or ::1); the journal stays on this machine"
                .into(),
        );
    }
    let rt = tokio::runtime::Runtime::new().map_err(|e| e.to_string())?;
    rt.block_on(listen(host, port))
}

async fn listen(host: &str, port: u16) -> Result<(), String> {
    let (tx, rx) = watch::channel(snapshot_body());
    tokio::spawn(poll_journal(tx));
    let app = Router::new()
        .route("/", get(index))
        .route("/index.html", get(index))
        .route("/stats.js", get(script))
        .route("/stats.css", get(style))
        .route("/health", get(health))
        .route("/api/snapshot", get(snapshot))
        .route(
            "/ws",
            get(move |headers: HeaderMap, ws: WebSocketUpgrade| {
                ws_upgrade(headers, ws, rx.clone())
            }),
        )
        .layer(middleware::from_fn(require_local_host));
    let addr = bind_addr(host, port)?;
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .map_err(|e| format!("bind {addr}: {e}"))?;
    let bound = listener.local_addr().map_err(|e| e.to_string())?;
    eprintln!("usage on http://{bound}");
    axum::serve(listener, app).await.map_err(|e| e.to_string())
}

fn bind_addr(host: &str, port: u16) -> Result<std::net::SocketAddr, String> {
    let host = host.trim();
    let rendered = if host.starts_with('[') {
        format!("{host}:{port}")
    } else if host.parse::<std::net::Ipv6Addr>().is_ok() {
        format!("[{host}]:{port}")
    } else {
        format!("{host}:{port}")
    };
    rendered
        .parse()
        .map_err(|e| format!("bind {host}:{port}: {e}"))
}

fn asset(content_type: &'static str, body: &'static str) -> impl IntoResponse {
    (
        [
            (header::CONTENT_TYPE, content_type),
            (header::CACHE_CONTROL, "no-cache"),
        ],
        body,
    )
}

async fn index() -> impl IntoResponse {
    asset("text/html; charset=utf-8", INDEX_HTML)
}

async fn script() -> impl IntoResponse {
    asset("text/javascript; charset=utf-8", STATS_JS)
}

async fn style() -> impl IntoResponse {
    asset("text/css; charset=utf-8", STATS_CSS)
}

async fn health() -> Json<Value> {
    Json(json!({"ok": true}))
}

async fn snapshot() -> Result<Json<Value>, (axum::http::StatusCode, String)> {
    match snapshot_value() {
        Ok(v) => Ok(Json(v)),
        Err(e) => Err((axum::http::StatusCode::INTERNAL_SERVER_ERROR, e)),
    }
}

/// The journal stays on this machine: refuse a `Host` that is not local
/// (DNS rebinding) before any route runs.
async fn require_local_host(req: Request, next: Next) -> Response {
    let local = req
        .headers()
        .get(header::HOST)
        .and_then(|h| h.to_str().ok())
        .is_some_and(crate::serve::is_local_host_header);
    if local {
        next.run(req).await
    } else {
        StatusCode::FORBIDDEN.into_response()
    }
}

async fn ws_upgrade(
    headers: HeaderMap,
    ws: WebSocketUpgrade,
    rx: watch::Receiver<String>,
) -> Response {
    if !crate::serve::ws_origin_allowed(&headers) {
        return StatusCode::FORBIDDEN.into_response();
    }
    ws.on_upgrade(move |socket| push(socket, rx))
        .into_response()
}

async fn push(mut socket: WebSocket, mut rx: watch::Receiver<String>) {
    let initial = rx.borrow().clone();
    if send(&mut socket, initial).await.is_err() {
        return;
    }
    loop {
        tokio::select! {
            _ = tokio::time::sleep(Duration::from_secs(1)) => {
                let body = rx.borrow().clone();
                if send(&mut socket, body).await.is_err() {
                    break;
                }
            }
            changed = rx.changed() => {
                if changed.is_err() {
                    break;
                }
                let body = rx.borrow_and_update().clone();
                if send(&mut socket, body).await.is_err() {
                    break;
                }
            }
            incoming = socket.recv() => {
                match incoming {
                    Some(Ok(axum::extract::ws::Message::Ping(payload))) => {
                        if socket
                            .send(axum::extract::ws::Message::Pong(payload))
                            .await
                            .is_err()
                        {
                            break;
                        }
                    }
                    Some(Ok(axum::extract::ws::Message::Close(_))) | None | Some(Err(_)) => break,
                    _ => {}
                }
            }
        }
    }
}

async fn send(socket: &mut WebSocket, body: String) -> Result<(), ()> {
    socket
        .send(axum::extract::ws::Message::Text(body.into()))
        .await
        .map_err(|_| ())
}

fn snapshot_body() -> String {
    snapshot_value()
        .ok()
        .and_then(|v| serde_json::to_string(&v).ok())
        .unwrap_or_else(|| "{}".to_owned())
}

fn snapshot_value() -> Result<Value, String> {
    let report = crate::usage_log::load_default(None)?;
    serde_json::to_value(&report).map_err(|e| e.to_string())
}

async fn poll_journal(tx: watch::Sender<String>) {
    let mut stamp = journal_stamp(&crate::usage_log::journal_path());
    loop {
        tokio::time::sleep(POLL).await;
        let path = crate::usage_log::journal_path();
        let next = journal_stamp(&path);
        if next == stamp {
            continue;
        }
        stamp = next;
        let _ = tx.send(snapshot_body());
    }
}

fn journal_stamp(path: &Path) -> Option<(u64, SystemTime)> {
    let meta = path.metadata().ok()?;
    let modified = meta.modified().unwrap_or(SystemTime::UNIX_EPOCH);
    Some((meta.len(), modified))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_loopback_is_refused() {
        let err = cmd_web("0.0.0.0", 9).unwrap_err();
        assert!(err.contains("loopback"), "{err}");
    }

    #[test]
    fn embedded_page_is_the_usage_app() {
        assert!(INDEX_HTML.contains("id=\"root\""));
        assert!(STATS_JS.contains("/api/snapshot"));
        assert!(STATS_CSS.contains("--bg"));
    }
}
