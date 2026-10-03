// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! P1.2 integration tests: growing range fetch against a local HTTP server
//! (std only, no extra dev-deps) serving a real fixture, plus a
//! `RUNA_LIVE=1` smoke test against huggingface.co (plan check:
//! `runa fit hf:unsloth/Qwen3-8B-GGUF:Q4_K_M` completes without downloading
//! the model).
//!
//! Run live: `RUNA_LIVE=1 cargo test -p runa-fit --test remote -- --ignored`.

use std::fs::File;
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::net::{SocketAddr, TcpListener};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use runa_fit::{Fetcher, ModelSource, Reader, parse_model_ref};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures")
        .join(name)
}

/// Minimal single-purpose HTTP server: `GET /file.gguf` with `Range`
/// support, `GET /api/models/<repo>` returning a canned siblings JSON.
/// Counts every request in `hits`. Runs until the listener is dropped...
/// actually until the process exits; each test spawns its own on an
/// ephemeral port, serving `file_path` bytes.
struct Server {
    addr: SocketAddr,
    hits: Arc<AtomicUsize>,
}

fn serve(file_path: PathBuf) -> Server {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let hits = Arc::new(AtomicUsize::new(0));
    let hits2 = hits.clone();
    thread::spawn(move || {
        for conn in listener.incoming() {
            let Ok(stream) = conn else { break };
            let hits = hits2.clone();
            let file_path = file_path.clone();
            thread::spawn(move || handle(stream, &file_path, &hits));
        }
    });
    // Give the accept loop a moment to start.
    thread::sleep(Duration::from_millis(50));
    Server { addr, hits }
}

fn handle(mut stream: std::net::TcpStream, file_path: &std::path::Path, hits: &AtomicUsize) {
    hits.fetch_add(1, Ordering::SeqCst);
    let mut reader = BufReader::new(stream.try_clone().unwrap());
    let mut request_line = String::new();
    if reader.read_line(&mut request_line).is_err() {
        return;
    }
    let mut range: Option<(u64, u64)> = None;
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).is_err() || line == "\r\n" || line.is_empty() {
            break;
        }
        // Header names are case-insensitive; reqwest sends lowercase.
        if let Some(v) = line
            .strip_prefix("Range:")
            .or_else(|| line.strip_prefix("range:"))
        {
            range = parse_range(v.trim());
        }
    }
    drop(reader);

    let path = request_line.split_whitespace().nth(1).unwrap_or("/");
    let response = if path.starts_with("/api/models/") {
        let body = r#"{"siblings":[{"rfilename":"model-Q4_K_M.gguf"},{"rfilename":"model-Q8_0.gguf"},{"rfilename":"config.json"}]}"#;
        format!(
            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            body.len(),
            body
        )
        .into_bytes()
    } else {
        let total = std::fs::metadata(file_path).unwrap().len();
        let (from, to) = match range {
            Some((a, b)) => (
                a.min(total),
                b.min(total.saturating_sub(1)).max(a.min(total)),
            ),
            None => (0, total.saturating_sub(1)),
        };
        let mut f = File::open(file_path).unwrap();
        f.seek(SeekFrom::Start(from)).unwrap();
        let mut body = vec![0u8; (to - from + 1) as usize];
        f.read_exact(&mut body).unwrap();
        let mut head = format!(
            "HTTP/1.1 206 Partial Content\r\nContent-Range: bytes {from}-{to}/{total}\r\nContent-Length: {}\r\nETag: \"test-etag\"\r\nConnection: close\r\n\r\n",
            body.len()
        )
        .into_bytes();
        head.extend_from_slice(&body);
        head
    };
    let _ = stream.write_all(&response);
}

fn parse_range(v: &str) -> Option<(u64, u64)> {
    let v = v.strip_prefix("bytes=")?;
    let (a, b) = v.split_once('-')?;
    Some((a.parse().ok()?, b.parse().ok()?))
}

fn test_fetcher(server: &Server) -> Fetcher {
    let dir = std::env::temp_dir().join(format!(
        "runa-p12-{}-{}",
        std::process::id(),
        server.addr.port()
    ));
    Fetcher::new().unwrap().with_cache_dir(dir).with_token(None)
}

#[test]
fn range_fetch_grows_until_parseable_without_downloading() {
    let path = fixture("qwen2-0_5b-instruct-q4_0.gguf");
    let file_len = std::fs::metadata(&path).unwrap().len();
    assert!(file_len > 100 << 20, "fixture should be a real model file");
    let server = serve(path);
    let f = test_fetcher(&server);

    let url = format!("http://{}/file.gguf", server.addr);
    let h = f.fetch_header(&ModelSource::Url(url)).expect("range fetch");
    assert!(!h.from_cache);
    let r = Reader::parse(&h.bytes).expect("fetched prefix parses");
    assert!(r.tensors.len() > 100, "real tensor table");
    assert_eq!(h.total_len, Some(file_len));
    // The point of P1.2: header bytes, not the model.
    assert!(
        (h.bytes.len() as u64) < file_len / 4,
        "downloaded {} of {file_len} bytes",
        h.bytes.len()
    );
    println!(
        "header={} tensors={} file={file_len}",
        h.bytes.len(),
        r.tensors.len()
    );
}

#[test]
fn second_fetch_comes_from_cache() {
    let path = fixture("qwen2-0_5b-instruct-q4_0.gguf");
    let server = serve(path);
    let f = test_fetcher(&server);
    let url = format!("http://{}/file.gguf", server.addr);
    let src = ModelSource::Url(url);

    let first = f.fetch_header(&src).expect("first fetch");
    assert!(!first.from_cache);
    let hits_after_first = server.hits.load(Ordering::SeqCst);

    let second = f.fetch_header(&src).expect("second fetch");
    assert!(second.from_cache);
    assert_eq!(second.bytes, first.bytes);
    // Only the 1-byte validation probe may hit the network.
    assert_eq!(server.hits.load(Ordering::SeqCst), hits_after_first + 1);
}

#[test]
fn local_file_uses_the_same_grow_loop() {
    let path = fixture("qwen2-0_5b-instruct-q4_0.gguf");
    let file_len = std::fs::metadata(&path).unwrap().len();
    let f = Fetcher::new().unwrap().with_token(None);
    let h = f
        .fetch_header(&ModelSource::Local(path))
        .expect("local prefix");
    let r = Reader::parse(&h.bytes).expect("parses");
    assert!(!r.tensors.is_empty());
    assert_eq!(h.total_len, Some(file_len));
    assert!((h.bytes.len() as u64) < file_len);
}

#[test]
fn quant_resolves_against_siblings_listing() {
    let server = serve(fixture("qwen2-0_5b-instruct-q4_0.gguf"));
    let f = test_fetcher(&server).with_hub_api_base(format!("http://{}/api/models", server.addr));
    let got = f.resolve_quant("any/repo", "Q4_K_M").expect("resolves");
    assert_eq!(got, "model-Q4_K_M.gguf");
    assert!(f.resolve_quant("any/repo", "Q2_XXS").is_err());
}

#[test]
#[ignore]
fn live_hf_header_fetch() {
    if std::env::var("RUNA_LIVE").as_deref() != Ok("1") {
        return;
    }
    let dir = std::env::temp_dir().join(format!("runa-p12-live-{}", std::process::id()));
    let f = Fetcher::new().unwrap().with_cache_dir(dir);
    let src = parse_model_ref("hf:unsloth/Qwen3-8B-GGUF:Q4_K_M").expect("ref parses");
    let t = Instant::now();
    let h = f.fetch_header(&src).expect("live fetch");
    let el = t.elapsed();
    let r = Reader::parse(&h.bytes).expect("live bytes parse");
    assert!(r.tensors.len() > 100);
    println!(
        "live: tensors={} header_bytes={} total={:?} elapsed={:.1}s cached={}",
        r.tensors.len(),
        h.bytes.len(),
        h.total_len,
        el.as_secs_f32(),
        h.from_cache
    );
}
