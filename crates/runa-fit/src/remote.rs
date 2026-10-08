// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! Remote GGUF header fetch (plan P1.2, model refs D3).
//!
//! `runa fit` must answer *before* downloading anything, so the header
//! (magic + metadata + tensor-info table, parsed by [`crate::gguf`]) is
//! fetched with HTTP `Range` requests that grow until the table is complete.
//!
//! ```text
//! model ref                     → bytes (prefix of the file, no download)
//! hf:<repo>:<file.gguf>         → https://huggingface.co/<repo>/resolve/main/<file.gguf>
//! hf:<repo>:<quant>             → sibling listing → best <quant> file → as above
//! https://…                     → direct URL (must serve ranges)
//! <local path>                  → read from disk (same grow loop via seek)
//! ```
//!
//! Design notes:
//! - Completeness is decided by [`crate::gguf::Reader::parse`]: truncation
//!   errors (`Truncated`, `UnexpectedEof`, `TensorTableTruncated`) mean
//!   "fetch more"; anything else is fatal (a corrupt remote file must not
//!   be mistaken for an incomplete one).
//! - Sync API over `reqwest::blocking` (see `Cargo.toml`): runa-fit stays
//!   runtime-agnostic. One-shot header fetch is network-latency-bound, so
//!   async buys nothing here (D20); bulk download with progress lands in
//!   P2.4 (`runa pull`) where async + streaming actually matter.
//! - Header cache lives in `<cache>/runa/headers/` (`XDG_CACHE_HOME`,
//!   `%LOCALAPPDATA%`, else `~/.cache`, else the temp dir), keyed by URL
//!   and validated against the server's total length (+ ETag when present).

use std::env;
use std::fs;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::time::Duration;

use reqwest::blocking::{Client, ClientBuilder};
use reqwest::header::{ACCEPT, AUTHORIZATION, RANGE, USER_AGENT};

use crate::gguf::{ReadError, Reader};

/// First range to ask for. Real headers are tokenizer-heavy (a 150k-token
/// vocabulary serializes to tens of MB), so start wide enough that small
/// models complete in one round trip.
pub const START_BYTES: u64 = 8 << 20;
/// Give up past this: a header bigger than this is a corrupt `tensor_count`.
pub const MAX_HEADER_BYTES: u64 = 256 << 20;
/// Per-request timeout (connect capped separately).
pub const REQUEST_TIMEOUT: Duration = Duration::from_secs(60);

/// A parsed model reference (plan D3: `hf:<repo>:<file-or-quant>`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HfRef {
    /// `owner/name`, e.g. `unsloth/Qwen3-8B-GGUF`.
    pub repo: String,
    /// Exact `.gguf` filename, or a quant tag like `Q4_K_M`.
    pub file_or_quant: String,
}

/// Where header bytes come from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelSource {
    /// `hf:<repo>:<file-or-quant>`.
    Hf(HfRef),
    /// Direct `http(s)://` URL (must serve byte ranges).
    Url(String),
    /// Local file (read prefix from disk, no network).
    Local(PathBuf),
}

/// Fetched header: the smallest leading prefix that parses.
#[derive(Debug, Clone)]
pub struct HeaderBytes {
    /// Leading bytes of the file (parse with [`Reader::parse`]).
    pub bytes: Vec<u8>,
    /// Full file length when the server reported it.
    pub total_len: Option<u64>,
    /// Served from the local header cache without any request.
    pub from_cache: bool,
}

/// Errors from remote header resolution and fetch.
#[derive(Debug, thiserror::Error)]
pub enum RemoteError {
    /// The model reference did not parse (`hf:<repo>:<file-or-quant>`,
    /// `http(s)://…`, or an existing local path).
    #[error(
        "bad model reference {0:?}: expected hf:<repo>:<file>, an http(s) URL, or a local path"
    )]
    BadRef(String),
    /// HTTP failure (network, status, timeout).
    #[error("http error for {url}: {msg}")]
    Http { url: String, msg: String },
    /// The header is bigger than [`MAX_HEADER_BYTES`] or the server does
    /// not serve ranges and the file is too big to take whole.
    #[error("header incomplete after {fetched} bytes ({detail})")]
    HeaderIncomplete { fetched: u64, detail: String },
    /// The bytes parse as GGUF but are corrupt (not merely truncated).
    #[error("remote file is not a valid GGUF header: {0}")]
    Corrupt(ReadError),
    /// No sibling file matches the quant tag.
    #[error("no .gguf file matching quant {quant:?} in {repo}")]
    NoQuantMatch { repo: String, quant: String },
    /// Safetensors (mistral-backend) models have no GGUF header to check:
    /// `runa fit` only estimates GGUF models.
    #[error(
        "safetensors model {repo}: `runa fit` only estimates GGUF models (the mistral backend needs no fit check)"
    )]
    Safetensors { repo: String },
    /// Header-cache I/O failure.
    #[error("header cache I/O: {0}")]
    CacheIo(String),
}

/// Parse a user-facing model reference (plan D3).
///
/// `hf:<repo>:<file-or-quant>` splits on the **last** `:` (repos contain
/// `/` but never `:`); anything starting with `http://`/`https://` is a
/// direct URL; anything else must be an existing local path.
pub fn parse_model_ref(s: &str) -> Result<ModelSource, RemoteError> {
    if let Some(rest) = s.strip_prefix("hf:") {
        let (repo, file_or_quant) = rest
            .rsplit_once(':')
            .ok_or_else(|| RemoteError::BadRef(s.to_owned()))?;
        if repo.is_empty() || file_or_quant.is_empty() {
            return Err(RemoteError::BadRef(s.to_owned()));
        }
        return Ok(ModelSource::Hf(HfRef {
            repo: repo.to_owned(),
            file_or_quant: file_or_quant.to_owned(),
        }));
    }
    if s.starts_with("http://") || s.starts_with("https://") {
        return Ok(ModelSource::Url(s.to_owned()));
    }
    let path = PathBuf::from(s);
    if path.is_file() {
        return Ok(ModelSource::Local(path));
    }
    Err(RemoteError::BadRef(s.to_owned()))
}

/// Resolve an [`HfRef`] to a download URL (plan P1.2).
pub fn resolve_hf_url(r: &HfRef, file: &str) -> String {
    format!("https://huggingface.co/{}/resolve/main/{}", r.repo, file)
}

/// Human cache root: `$XDG_CACHE_HOME`, else `~/.cache`
/// (`%LOCALAPPDATA%` on Windows), else the temp dir.
pub fn cache_root() -> PathBuf {
    if let Ok(xdg) = env::var("XDG_CACHE_HOME")
        && !xdg.is_empty()
    {
        return PathBuf::from(xdg);
    }
    #[cfg(windows)]
    if let Ok(local) = env::var("LOCALAPPDATA")
        && !local.is_empty()
    {
        return PathBuf::from(local);
    }
    if let Ok(home) = env::var("HOME")
        && !home.is_empty()
    {
        return PathBuf::from(home).join(".cache");
    }
    env::temp_dir()
}

/// Header-cache file for a URL (filesystem-safe: alnum kept, else `_`,
/// suffixed by length to avoid collisions).
fn cache_path(cache_dir: &Path, url: &str) -> PathBuf {
    let mut name: String = url
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '.' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect();
    name.truncate(180);
    cache_dir.join(format!("{}..{}.ggufhdr", name, url.len()))
}

fn meta_path(hdr: &Path) -> PathBuf {
    hdr.with_extension("json")
}

/// Header fetcher: HTTP range client + on-disk cache + optional HF token.
#[derive(Debug, Clone)]
pub struct Fetcher {
    client: Client,
    cache_dir: PathBuf,
    token: Option<String>,
    hub_api_base: String,
}

impl Fetcher {
    /// Create a default fetcher: `<cache>/runa/headers`, token from `HF_TOKEN`.
    pub fn new() -> Result<Fetcher, RemoteError> {
        let client = ClientBuilder::new()
            .timeout(REQUEST_TIMEOUT)
            .connect_timeout(Duration::from_secs(10))
            .user_agent("runa-fit/0.1 (prefit-check; +https://github.com/runa)")
            .build()
            .map_err(|e| RemoteError::Http {
                url: String::new(),
                msg: e.to_string(),
            })?;
        Ok(Fetcher {
            client,
            cache_dir: cache_root().join("runa").join("headers"),
            token: env::var("HF_TOKEN").ok().filter(|t| !t.is_empty()),
            hub_api_base: "https://huggingface.co/api/models".to_owned(),
        })
    }

    /// Override the cache directory (tests, `--cache-dir` later).
    pub fn with_cache_dir(mut self, dir: PathBuf) -> Fetcher {
        self.cache_dir = dir;
        self
    }

    /// Override the Hub API base URL (tests point it at a local server).
    pub fn with_hub_api_base(mut self, base: String) -> Fetcher {
        self.hub_api_base = base;
        self
    }

    /// Override the bearer token (`None` disables auth).
    pub fn with_token(mut self, token: Option<String>) -> Fetcher {
        self.token = token;
        self
    }

    fn authed(
        &self,
        url: &str,
        req: reqwest::blocking::RequestBuilder,
    ) -> reqwest::blocking::RequestBuilder {
        let req = req.header(USER_AGENT, "runa-fit/0.1 (prefit-check)");
        match &self.token {
            Some(t) if token_allowed(url, &self.hub_api_base) => {
                req.header(AUTHORIZATION, format!("Bearer {t}"))
            }
            _ => req,
        }
    }

    /// Fetch the smallest leading prefix that parses as a GGUF header.
    pub fn fetch_header(&self, src: &ModelSource) -> Result<HeaderBytes, RemoteError> {
        match src {
            ModelSource::Local(path) => read_local_prefix(path),
            ModelSource::Url(url) => self.fetch_url(url),
            ModelSource::Hf(r) => {
                // Safetensors snapshots (P9.2) are mistral-backend models:
                // refuse with a pointer instead of a GGUF parse error.
                if is_safetensors_tag(&r.file_or_quant) {
                    return Err(RemoteError::Safetensors {
                        repo: r.repo.clone(),
                    });
                }
                let file = if is_gguf_name(&r.file_or_quant) {
                    r.file_or_quant.clone()
                } else {
                    self.resolve_quant(&r.repo, &r.file_or_quant)?
                };
                self.fetch_url(&resolve_hf_url(r, &file))
            }
        }
    }

    /// List every sibling file of an HF repo via the Hub API (sorted).
    /// [`Fetcher::siblings`] filters this to `.gguf` files.
    pub fn siblings_all(&self, repo: &str) -> Result<Vec<String>, RemoteError> {
        let url = format!("{}/{}", self.hub_api_base.trim_end_matches('/'), repo);
        let body = self
            .authed(
                &url,
                self.client.get(&url).header(ACCEPT, "application/json"),
            )
            .send()
            .map_err(|e| RemoteError::Http {
                url: url.clone(),
                msg: e.to_string(),
            })?
            .error_for_status()
            .map_err(|e| RemoteError::Http {
                url: url.clone(),
                msg: e.to_string(),
            })?
            .text()
            .map_err(|e| RemoteError::Http {
                url: url.clone(),
                msg: e.to_string(),
            })?;
        let v: serde_json::Value = serde_json::from_str(&body).map_err(|e| RemoteError::Http {
            url: url.clone(),
            msg: format!("hub API returned invalid JSON: {e}"),
        })?;
        let mut out = Vec::new();
        if let Some(sibs) = v.get("siblings").and_then(|s| s.as_array()) {
            for s in sibs {
                if let Some(name) = s.get("rfilename").and_then(|n| n.as_str()) {
                    out.push(name.to_owned());
                }
            }
        }
        out.sort();
        Ok(out)
    }

    /// List `.gguf` sibling files of an HF repo via the Hub API.
    pub fn siblings(&self, repo: &str) -> Result<Vec<String>, RemoteError> {
        Ok(self
            .siblings_all(repo)?
            .into_iter()
            .filter(|name| is_gguf_name(name))
            .collect())
    }

    /// Resolve a quant tag (`Q4_K_M`) to the best sibling filename.
    pub fn resolve_quant(&self, repo: &str, quant: &str) -> Result<String, RemoteError> {
        let sibs = self.siblings(repo)?;
        pick_quant(&sibs, quant).ok_or(RemoteError::NoQuantMatch {
            repo: repo.to_owned(),
            quant: quant.to_owned(),
        })
    }

    /// Growing range fetch until [`Reader::parse`] succeeds.
    fn fetch_url(&self, url: &str) -> Result<HeaderBytes, RemoteError> {
        let cache_file = cache_path(&self.cache_dir, url);
        // Fast path: cached prefix is reusable when the server reports the
        // same total length (and ETag, when the server sends one).
        if let Some(hit) = read_cache(&cache_file)? {
            let (total, etag) = self.probe(url, hit.bytes.len() as u64)?;
            if total == hit.total_len && etag == hit.etag {
                return Ok(HeaderBytes {
                    bytes: hit.bytes,
                    total_len: hit.total_len,
                    from_cache: true,
                });
            }
        }

        let mut buf: Vec<u8> = Vec::new();
        let mut want: u64 = START_BYTES;
        let mut total: Option<u64> = None;
        let mut etag: Option<String> = None;
        loop {
            let from = buf.len() as u64;
            if from >= want {
                want *= 2;
            }
            if want > MAX_HEADER_BYTES {
                return Err(RemoteError::HeaderIncomplete {
                    fetched: from,
                    detail: format!("header exceeds {MAX_HEADER_BYTES} bytes"),
                });
            }
            let chunk = self.get_range(url, from, want)?;
            total = chunk.total.or(total);
            if chunk.etag.is_some() {
                etag = chunk.etag.clone();
            }
            match chunk.status {
                RangeStatus::Partial(body) => buf.extend_from_slice(&body),
                RangeStatus::Whole(body) => {
                    // Server ignores ranges: only acceptable when the whole
                    // file fits the header budget.
                    if (body.len() as u64) > MAX_HEADER_BYTES {
                        return Err(RemoteError::HeaderIncomplete {
                            fetched: from,
                            detail: "server does not serve byte ranges".into(),
                        });
                    }
                    buf = body;
                    total = Some(buf.len() as u64);
                }
            }
            match Reader::parse(&buf) {
                Ok(_) => {
                    let bytes = buf;
                    write_cache(&cache_file, url, &bytes, total, etag.as_deref())?;
                    return Ok(HeaderBytes {
                        bytes,
                        total_len: total,
                        from_cache: false,
                    });
                }
                Err(e) if is_truncation(&e) => {
                    // If the server told us the total and we already have it
                    // all, the file is corrupt — not merely short.
                    if let Some(t) = total
                        && buf.len() as u64 >= t
                    {
                        return Err(RemoteError::Corrupt(e));
                    }
                    want *= 2;
                    continue;
                }
                Err(e) => return Err(RemoteError::Corrupt(e)),
            }
        }
    }

    /// Ask for `bytes=from..want-1`; returns the body plus the known total.
    fn get_range(&self, url: &str, from: u64, want: u64) -> Result<Chunk, RemoteError> {
        let end = want.saturating_sub(1).max(from);
        let resp = self
            .authed(
                url,
                self.client
                    .get(url)
                    .header(RANGE, format!("bytes={from}-{end}")),
            )
            .send()
            .map_err(|e| RemoteError::Http {
                url: url.to_owned(),
                msg: e.to_string(),
            })?;
        let status = resp.status();
        let headers = resp.headers().clone();
        let total_hdr = content_total(&headers, None);
        let etag = headers
            .get("etag")
            .and_then(|v| v.to_str().ok())
            .map(str::to_owned);
        if status == reqwest::StatusCode::PARTIAL_CONTENT {
            let start = content_range_start(headers.get("content-range")).ok_or_else(|| {
                RemoteError::Http {
                    url: url.to_owned(),
                    msg: "206 without a usable Content-Range".into(),
                }
            })?;
            if start != from {
                return Err(RemoteError::Http {
                    url: url.to_owned(),
                    msg: format!("Content-Range starts at {start}, want {from}"),
                });
            }
            let room = MAX_HEADER_BYTES.saturating_sub(from);
            let cap = want.saturating_sub(from).min(room).max(1);
            let body = read_body_capped(resp, cap).map_err(|e| RemoteError::Http {
                url: url.to_owned(),
                msg: e,
            })?;
            return Ok(Chunk {
                status: RangeStatus::Partial(body),
                total: total_hdr,
                etag,
            });
        }
        if status.is_success() {
            if let Some(len) = resp.content_length()
                && len > MAX_HEADER_BYTES
            {
                return Err(RemoteError::HeaderIncomplete {
                    fetched: from,
                    detail: "server does not serve byte ranges".into(),
                });
            }
            let body = match read_body_capped(resp, MAX_HEADER_BYTES) {
                Ok(body) => body,
                Err(e) if e.contains("exceeds") || e.contains("cap is") => {
                    return Err(RemoteError::HeaderIncomplete {
                        fetched: from,
                        detail: "server does not serve byte ranges".into(),
                    });
                }
                Err(e) => {
                    return Err(RemoteError::Http {
                        url: url.to_owned(),
                        msg: e,
                    });
                }
            };
            let total = content_total(&headers, Some(body.len() as u64));
            return Ok(Chunk {
                status: RangeStatus::Whole(body),
                total,
                etag,
            });
        }
        // 416 (past EOF) or anything else: surface the status.
        Err(RemoteError::Http {
            url: url.to_owned(),
            msg: format!("unexpected status {status} for range bytes={from}-{end}"),
        })
    }

    /// Cheap probe: one small range to learn total length + ETag for cache
    /// validation. Returns `(total_len, etag)`.
    fn probe(&self, url: &str, _have: u64) -> Result<(Option<u64>, Option<String>), RemoteError> {
        let chunk = self.get_range(url, 0, 1)?;
        Ok((chunk.total, chunk.etag))
    }

    /// File metadata from the Hub's resolve endpoint: follows no redirects
    /// and reads `x-linked-size` / `x-linked-etag` (the file's byte size and
    /// SHA-256 for regular files — verified identical to `sha256sum` output
    /// for GGUFs, 2026-09-08). Used by `runa pull` (P2.4) to skip present
    /// files and to verify downloads.
    pub fn head_metadata(&self, url: &str) -> Result<FileMeta, RemoteError> {
        let client = ClientBuilder::new()
            .timeout(REQUEST_TIMEOUT)
            .connect_timeout(Duration::from_secs(10))
            .redirect(reqwest::redirect::Policy::none())
            .user_agent("runa-fit/0.1 (prefit-check)")
            .build()
            .map_err(|e| RemoteError::Http {
                url: url.to_owned(),
                msg: e.to_string(),
            })?;
        let resp = self
            .authed(url, client.get(url))
            .send()
            .map_err(|e| RemoteError::Http {
                url: url.to_owned(),
                msg: e.to_string(),
            })?;
        let h = resp.headers();
        let size = h
            .get("x-linked-size")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse::<u64>().ok());
        let sha256 = h
            .get("x-linked-etag")
            .and_then(|v| v.to_str().ok())
            .map(|s| s.trim_matches('"').to_owned())
            .filter(|s| !s.is_empty());
        Ok(FileMeta { size, sha256 })
    }
}

/// Expected file identity from the Hub resolve endpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileMeta {
    /// `x-linked-size`, if the Hub reported it.
    pub size: Option<u64>,
    /// `x-linked-etag` (file SHA-256 for regular files), if reported.
    pub sha256: Option<String>,
}

struct Chunk {
    status: RangeStatus,
    total: Option<u64>,
    etag: Option<String>,
}

enum RangeStatus {
    Partial(Vec<u8>),
    Whole(Vec<u8>),
}

/// Total file length from `Content-Range: bytes a-b/TOTAL`, falling back to
/// a known value (e.g. a 200's `Content-Length` handled by the caller).
/// `HF_TOKEN` goes only to huggingface.co (and its subdomains) or the
/// configured hub API host. A `runa fit https://other-host/...` must not
/// receive the user's Hugging Face credential.
fn token_allowed(url: &str, hub_api_base: &str) -> bool {
    let Some(host) = reqwest::Url::parse(url)
        .ok()
        .and_then(|u| u.host_str().map(|h| h.trim_end_matches('.').to_owned()))
    else {
        return false;
    };
    if is_hf_host(&host) {
        return true;
    }
    reqwest::Url::parse(hub_api_base)
        .ok()
        .and_then(|u| u.host_str().map(|h| h.eq_ignore_ascii_case(&host)))
        .unwrap_or(false)
}

fn is_hf_host(host: &str) -> bool {
    let host = host.trim_end_matches('.');
    host.eq_ignore_ascii_case("huggingface.co")
        || host.to_ascii_lowercase().ends_with(".huggingface.co")
}

fn content_range_start(value: Option<&reqwest::header::HeaderValue>) -> Option<u64> {
    let rest = value?.to_str().ok()?.trim().strip_prefix("bytes ")?;
    let (start, _) = rest.split_once('-')?;
    start.parse().ok()
}

/// Stop after `cap` bytes. A `Content-Length` above the cap is rejected
/// before the body is read, so a server that ignores `Range` cannot push
/// a multi-gigabyte model into RAM.
fn read_body_capped(resp: reqwest::blocking::Response, cap: u64) -> Result<Vec<u8>, String> {
    if let Some(len) = resp.content_length()
        && len > cap
    {
        return Err(format!("response is {len} bytes, cap is {cap}"));
    }
    let mut limited = resp.take(cap.saturating_add(1));
    let mut buf = Vec::new();
    limited.read_to_end(&mut buf).map_err(|e| e.to_string())?;
    if buf.len() as u64 > cap {
        return Err(format!("response exceeds {cap} bytes"));
    }
    Ok(buf)
}

fn content_total(headers: &reqwest::header::HeaderMap, fallback: Option<u64>) -> Option<u64> {
    headers
        .get("content-range")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.rsplit('/').next())
        .and_then(|t| {
            if t == "*" {
                None
            } else {
                t.parse::<u64>().ok()
            }
        })
        .or(fallback)
}

/// Truncation-shaped parse failures mean "fetch more"; anything else means
/// the bytes are corrupt and more of them will not help.
fn is_truncation(e: &ReadError) -> bool {
    matches!(
        e,
        ReadError::Truncated
            | ReadError::UnexpectedEof
            | ReadError::TensorTableTruncated
            | ReadError::StringTooLong
    )
}

fn is_gguf_name(name: &str) -> bool {
    name.len() >= 5 && name[name.len() - 5..].eq_ignore_ascii_case(".gguf")
}

/// True for the `hf:<repo>:safetensors` tag (P9.2): the whole safetensors
/// snapshot of a repo, loaded as a mistral-backend model directory.
/// Case-insensitive, like quant tags.
pub fn is_safetensors_tag(tag: &str) -> bool {
    tag.eq_ignore_ascii_case("safetensors")
}

/// Pick the best sibling filename for a quant tag (shared with `runa pull`).
///
/// 1. exact filename match; 2. `-`/`_`-suffixed (`…-Q4_K_M.gguf`);
///    3. substring (case-insensitive). Ties break by shortest name, then
///    lexicographic order, so the choice is deterministic.
pub fn pick_quant(siblings: &[String], quant: &str) -> Option<String> {
    if siblings.iter().any(|s| s == quant) {
        return Some(quant.to_owned());
    }
    let q_lower = quant.to_lowercase();
    let suffixed: Vec<&String> = siblings
        .iter()
        .filter(|s| {
            let l = s.to_lowercase();
            l.ends_with(&format!("-{q_lower}.gguf")) || l.ends_with(&format!("_{q_lower}.gguf"))
        })
        .collect();
    let pool: Vec<&String> = if suffixed.is_empty() {
        siblings
            .iter()
            .filter(|s| s.to_lowercase().contains(&q_lower))
            .collect()
    } else {
        suffixed
    };
    pool.into_iter()
        .min_by(|a, b| a.len().cmp(&b.len()).then(a.cmp(b)))
        .cloned()
}

/// Read a local file's leading prefix with the same grow-until-parse loop.
/// Shared by the planner/verdict paths (`ModelSource::Local` goes through
/// [`Fetcher::fetch_header`], which delegates here).
pub fn read_local_prefix(path: &Path) -> Result<HeaderBytes, RemoteError> {
    let mut f = fs::File::open(path).map_err(|e| RemoteError::CacheIo(e.to_string()))?;
    let total = f
        .metadata()
        .map(|m| m.len())
        .map_err(|e| RemoteError::CacheIo(e.to_string()))?;
    let mut buf = Vec::new();
    let mut want = START_BYTES.min(total.max(1));
    loop {
        buf.resize(want.min(total) as usize, 0);
        f.seek(SeekFrom::Start(0))
            .map_err(|e| RemoteError::CacheIo(e.to_string()))?;
        f.read_exact(&mut buf)
            .map_err(|e| RemoteError::CacheIo(e.to_string()))?;
        match Reader::parse(&buf) {
            Ok(_) => {
                return Ok(HeaderBytes {
                    bytes: buf,
                    total_len: Some(total),
                    from_cache: false,
                });
            }
            Err(e) if is_truncation(&e) => {
                if buf.len() as u64 >= total {
                    return Err(RemoteError::Corrupt(e));
                }
                want = (want * 2).min(total);
                if want > MAX_HEADER_BYTES {
                    return Err(RemoteError::HeaderIncomplete {
                        fetched: buf.len() as u64,
                        detail: format!("header exceeds {MAX_HEADER_BYTES} bytes"),
                    });
                }
                continue;
            }
            Err(e) => return Err(RemoteError::Corrupt(e)),
        }
    }
}

struct Cached {
    bytes: Vec<u8>,
    total_len: Option<u64>,
    etag: Option<String>,
}

fn read_cache(file: &Path) -> Result<Option<Cached>, RemoteError> {
    if !file.is_file() {
        return Ok(None);
    }
    let bytes = fs::read(file).map_err(|e| RemoteError::CacheIo(e.to_string()))?;
    let meta_raw = fs::read(meta_path(file)).map_err(|e| RemoteError::CacheIo(e.to_string()))?;
    let meta: serde_json::Value =
        serde_json::from_slice(&meta_raw).map_err(|e| RemoteError::CacheIo(e.to_string()))?;
    Ok(Some(Cached {
        bytes,
        total_len: meta.get("total_len").and_then(|v| v.as_u64()),
        etag: meta.get("etag").and_then(|v| v.as_str()).map(str::to_owned),
    }))
}

fn write_cache(
    file: &Path,
    url: &str,
    bytes: &[u8],
    total: Option<u64>,
    etag: Option<&str>,
) -> Result<(), RemoteError> {
    if let Some(parent) = file.parent() {
        fs::create_dir_all(parent).map_err(|e| RemoteError::CacheIo(e.to_string()))?;
    }
    fs::write(file, bytes).map_err(|e| RemoteError::CacheIo(e.to_string()))?;
    let meta = format!(
        "{{\"url\":{},\"total_len\":{},\"etag\":{}}}",
        serde_json::Value::String(url.to_owned()),
        total
            .map(|t| t.to_string())
            .unwrap_or_else(|| "null".into()),
        etag.map(|s| serde_json::Value::String(s.to_owned()))
            .unwrap_or(serde_json::Value::Null)
    );
    fs::write(meta_path(file), meta).map_err(|e| RemoteError::CacheIo(e.to_string()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ref_parsing() {
        assert_eq!(
            parse_model_ref("hf:unsloth/Qwen3-8B-GGUF:Q4_K_M").unwrap(),
            ModelSource::Hf(HfRef {
                repo: "unsloth/Qwen3-8B-GGUF".into(),
                file_or_quant: "Q4_K_M".into(),
            })
        );
        assert_eq!(
            parse_model_ref("hf:org/model:file-Q4_K_M.gguf").unwrap(),
            ModelSource::Hf(HfRef {
                repo: "org/model".into(),
                file_or_quant: "file-Q4_K_M.gguf".into(),
            })
        );
        assert!(parse_model_ref("hf:nofilepart").is_err());
        assert!(parse_model_ref("hf::file.gguf").is_err());
        assert_eq!(
            parse_model_ref("https://example.com/m.gguf").unwrap(),
            ModelSource::Url("https://example.com/m.gguf".into())
        );
        assert!(parse_model_ref("nope-not-a-file.gguf").is_err());
    }

    #[test]
    fn hf_url_shape() {
        let r = HfRef {
            repo: "unsloth/Qwen3-8B-GGUF".into(),
            file_or_quant: "Q4_K_M".into(),
        };
        assert_eq!(
            resolve_hf_url(&r, "Qwen3-8B-Q4_K_M.gguf"),
            "https://huggingface.co/unsloth/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf"
        );
    }

    #[test]
    fn quant_pick_rules() {
        let sibs = vec![
            "Qwen3-8B-Q4_K_M.gguf".to_owned(),
            "Qwen3-8B-Q4_K_M-00001-of-00002.gguf".to_owned(),
            "Qwen3-8B-Q8_0.gguf".to_owned(),
            "Qwen3-8B-UD-Q4_K_XL.gguf".to_owned(),
        ];
        // Suffixed match wins over longer split-file names.
        assert_eq!(
            pick_quant(&sibs, "Q4_K_M").as_deref(),
            Some("Qwen3-8B-Q4_K_M.gguf")
        );
        // Case-insensitive substring fallback.
        assert_eq!(
            pick_quant(&sibs, "q8_0").as_deref(),
            Some("Qwen3-8B-Q8_0.gguf")
        );
        assert_eq!(pick_quant(&sibs, "Q2_XXS"), None);
        assert!(pick_quant(&[], "Q4_K_M").is_none());
    }

    #[test]
    fn truncation_classes() {
        assert!(is_truncation(&ReadError::Truncated));
        assert!(is_truncation(&ReadError::UnexpectedEof));
        assert!(is_truncation(&ReadError::TensorTableTruncated));
        assert!(!is_truncation(&ReadError::BadMagic));
        assert!(!is_truncation(&ReadError::UnsupportedVersion(99)));
    }

    #[test]
    fn safetensors_tag_matches() {
        assert!(is_safetensors_tag("safetensors"));
        assert!(is_safetensors_tag("SAFETENSORS"));
        assert!(!is_safetensors_tag("Q4_K_M"));
        assert!(!is_safetensors_tag("model.gguf"));
    }

    #[test]
    fn fetch_header_refuses_safetensors_without_network() {
        let f = Fetcher::new().unwrap();
        let src = ModelSource::Hf(HfRef {
            repo: "org/model".into(),
            file_or_quant: "safetensors".into(),
        });
        let err = f.fetch_header(&src).unwrap_err();
        assert!(matches!(err, RemoteError::Safetensors { .. }), "{err}");
        assert!(err.to_string().contains("only estimates GGUF"), "{err}");
    }

    #[test]
    fn cache_path_is_stable_and_safe() {
        let dir = Path::new("/tmp/x");
        let a = cache_path(dir, "https://huggingface.co/o/m/resolve/main/f.gguf");
        let b = cache_path(dir, "https://huggingface.co/o/m/resolve/main/f.gguf");
        assert_eq!(a, b);
        // Safety is a property of the derived file name; the joined path
        // carries OS separators (`\` on Windows), so assert on the name.
        let name = a.file_name().unwrap().to_str().unwrap();
        assert!(
            name.chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_'))
        );
    }

    #[test]
    fn token_stays_on_huggingface_or_the_hub_host() {
        assert!(token_allowed(
            "https://huggingface.co/org/model/resolve/main/m.gguf",
            "https://huggingface.co/api/models"
        ));
        assert!(token_allowed(
            "https://cdn-lfs.huggingface.co/foo",
            "https://huggingface.co/api/models"
        ));
        assert!(!token_allowed(
            "http://127.0.0.1:9/evil/model.gguf",
            "https://huggingface.co/api/models"
        ));
        assert!(token_allowed(
            "http://127.0.0.1:9/api/models/org",
            "http://127.0.0.1:9/api/models"
        ));
    }

    #[test]
    fn content_range_start_must_match_the_request() {
        use reqwest::header::HeaderValue;
        assert_eq!(
            content_range_start(Some(&HeaderValue::from_static("bytes 0-99/1000"))),
            Some(0)
        );
        assert_eq!(
            content_range_start(Some(&HeaderValue::from_static("bytes 128-255/999"))),
            Some(128)
        );
        assert_eq!(content_range_start(None), None);
    }
}
