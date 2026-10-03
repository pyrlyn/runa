// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! `runa pull` + the local model store (plan P2.4).
//!
//! Downloads go through `hf-hub` (blocking API) into
//! `~/.local/share/runa/models/<owner>--<name>/<file>`, then are verified
//! by size + SHA-256 from the Hub resolve headers (`x-linked-size` /
//! `x-linked-etag` — see `Fetcher::head_metadata`). A `.verified` sidecar
//! records the check so a second `pull` finishes without rehashing (< 1 s).
//!
//! `runa run`/`chat` resolve `hf:` refs and aliases against this store;
//! anything not downloaded errors with a `runa pull` pointer (no implicit
//! multi-gigabyte downloads — plan D12).
//!
//! Safetensors snapshots (P9.2, `--backend mistral`) pull with
//! `hf:<repo>:safetensors`: `config.json` + tokenizer + `*.safetensors`
//! land as a model directory the mistral backend loads locally.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use hf_hub::{HFClientSync, split_id};
use runa_fit::{Fetcher, HfRef, ModelSource, parse_model_ref, pick_quant, resolve_hf_url};
use sha2::{Digest, Sha256};

use crate::config::AliasTable;

/// Data root: `$XDG_DATA_HOME/runa`, else `~/.local/share/runa`.
pub fn data_dir() -> PathBuf {
    if let Ok(xdg) = std::env::var("XDG_DATA_HOME")
        && !xdg.is_empty()
    {
        return PathBuf::from(xdg).join("runa");
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    PathBuf::from(home)
        .join(".local")
        .join("share")
        .join("runa")
}

/// Local model store.
pub fn models_dir() -> PathBuf {
    data_dir().join("models")
}

/// Store directory for one repo (`owner--name`, filesystem-safe).
pub fn repo_dir(repo: &str) -> PathBuf {
    let safe: String = repo
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '.' || c == '-' {
                c
            } else {
                '-'
            }
        })
        .collect();
    models_dir().join(safe)
}

/// Refuse fresh `pull` downloads larger than this (3 GiB).
///
/// Files above the limit do not fit the test-fixture budget and usually mean
/// a wrong quant tag. Override with `RUNA_MAX_MODEL_BYTES` (integer bytes)
/// on machines that genuinely need bigger files. The guard applies to fresh
/// downloads only, never to already-verified cached files.
pub const MAX_MODEL_BYTES: u64 = 3 * 1024 * 1024 * 1024;

fn max_model_bytes() -> u64 {
    std::env::var("RUNA_MAX_MODEL_BYTES")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(MAX_MODEL_BYTES)
}

fn check_size_limit(size: u64, repo: &str, file: &str) -> Result<(), String> {
    let limit = max_model_bytes();
    if size > limit {
        return Err(format!(
            "{repo}/{file} is {size} bytes, over the {limit} byte pull limit \
             (set RUNA_MAX_MODEL_BYTES to override)"
        ));
    }
    Ok(())
}

/// A pulled (or already-present) model file.
#[derive(Debug, Clone)]
pub struct PulledModel {
    /// Local file path.
    pub path: PathBuf,
    /// Repo `owner/name`.
    pub repo: String,
    /// Filename within the repo.
    pub file: String,
    /// Byte size.
    pub size: u64,
    /// True when bytes were downloaded just now.
    pub fresh: bool,
}

/// Sidecar recording a completed verification.
#[derive(Debug, Clone)]
struct Verified {
    size: u64,
    sha256: String,
    verified_at: u64,
}

fn sidecar_for(dest: &Path) -> PathBuf {
    dest.with_extension("verified")
}

fn read_sidecar(dest: &Path) -> Option<Verified> {
    let raw = fs::read(sidecar_for(dest)).ok()?;
    let v: serde_json::Value = serde_json::from_slice(&raw).ok()?;
    Some(Verified {
        size: v.get("size")?.as_u64()?,
        sha256: v.get("sha256")?.as_str()?.to_owned(),
        verified_at: v.get("verified_at")?.as_u64()?,
    })
}

fn write_sidecar(dest: &Path, size: u64, sha256: &str) {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let meta = format!(
        "{{\"size\":{size},\"sha256\":{},\"verified_at\":{now}}}",
        serde_json::Value::String(sha256.to_owned())
    );
    let _ = fs::write(sidecar_for(dest), meta);
}

fn mtime_secs(path: &Path) -> Option<u64> {
    fs::metadata(path)
        .ok()?
        .modified()
        .ok()?
        .duration_since(UNIX_EPOCH)
        .ok()
        .map(|d| d.as_secs())
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let mut f = fs::File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let mut h = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = std::io::Read::read(&mut f, &mut buf)
            .map_err(|e| format!("hash {}: {e}", path.display()))?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(h.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

/// Resolve an `hf:` ref to (repo, exact filename), expanding quant tags
/// through the sibling listing.
fn resolve_hf(fetcher: &Fetcher, r: &HfRef) -> Result<(String, String), String> {
    let file = if r.file_or_quant.to_lowercase().ends_with(".gguf") {
        r.file_or_quant.clone()
    } else {
        let sibs = fetcher.siblings(&r.repo).map_err(|e| e.to_string())?;
        pick_quant(&sibs, &r.file_or_quant)
            .ok_or_else(|| format!("no .gguf matching {:?} in {}", r.file_or_quant, r.repo))?
    };
    Ok((r.repo.clone(), file))
}

/// Fast path when the file is already on disk and verification matches Hub metadata.
fn cached_if_verified(
    dest: &Path,
    meta: &runa_fit::FileMeta,
    repo: &str,
    file: &str,
) -> Option<PulledModel> {
    if !dest.is_file() {
        return None;
    }
    let len = fs::metadata(dest).map(|m| m.len()).unwrap_or(0);
    if meta.size != Some(len) {
        return None;
    }
    if let Some(v) = read_sidecar(dest) {
        let sha_ok = meta.sha256.as_deref() == Some(v.sha256.as_str());
        let fresh_file = mtime_secs(dest).is_some_and(|m| m > v.verified_at);
        if v.size == len && sha_ok && !fresh_file {
            return Some(PulledModel {
                path: dest.to_path_buf(),
                repo: repo.to_owned(),
                file: file.to_owned(),
                size: len,
                fresh: false,
            });
        }
    }
    if let Some(expected) = meta.sha256.as_deref() {
        let actual = sha256_file(dest).ok()?;
        if actual == expected {
            write_sidecar(dest, len, expected);
            return Some(PulledModel {
                path: dest.to_path_buf(),
                repo: repo.to_owned(),
                file: file.to_owned(),
                size: len,
                fresh: false,
            });
        }
    }
    None
}

pub fn pull(model_ref: &str) -> Result<PulledModel, String> {
    let src = parse_model_ref(model_ref).map_err(|e| e.to_string())?;
    let ModelSource::Hf(r) = src else {
        return Err(format!(
            "{model_ref}: `pull` takes `hf:<repo>:<file-or-quant>`"
        ));
    };
    // P9.2: whole-snapshot download for the mistral backend.
    if runa_fit::is_safetensors_tag(&r.file_or_quant) {
        return pull_safetensors(&r.repo);
    }
    let fetcher = Fetcher::new().map_err(|e| e.to_string())?;
    let (repo, file) = resolve_hf(&fetcher, &r)?;
    let url = resolve_hf_url(
        &HfRef {
            repo: repo.clone(),
            file_or_quant: file.clone(),
        },
        &file,
    );
    let meta = fetcher.head_metadata(&url).map_err(|e| e.to_string())?;
    if let Some(size) = meta.size {
        check_size_limit(size, &repo, &file)?;
    }

    let dest = repo_dir(&repo).join(&file);
    if let Some(hit) = cached_if_verified(&dest, &meta, &repo, &file) {
        return Ok(hit);
    }

    // Download through hf-hub into the repo dir.
    let (owner, name) = split_id(&repo);
    let client = HFClientSync::new().map_err(|e| format!("hub client: {e}"))?;
    let dir = repo_dir(&repo);
    fs::create_dir_all(&dir).map_err(|e| format!("mkdir {}: {e}", dir.display()))?;
    client
        .model(owner, name)
        .download_file()
        .filename(&file)
        .local_dir(&dir)
        .send()
        .map_err(|e| format!("download {repo}/{file}: {e}"))?;
    if !dest.is_file() {
        return Err(format!("download finished but {} missing", dest.display()));
    }

    // Verify what landed.
    let len = verify_downloaded(&dest, &meta)?;
    check_size_limit(len, &repo, &file)?;
    println!("pulled {repo}/{file} ({len} bytes)");
    Ok(PulledModel {
        path: dest,
        repo,
        file,
        size: len,
        fresh: true,
    })
}

/// Size + SHA-256 check of a downloaded file; writes the `.verified`
/// sidecar. Returns the byte size. Shared by GGUF and snapshot pulls.
fn verify_downloaded(dest: &Path, meta: &runa_fit::FileMeta) -> Result<u64, String> {
    let len = fs::metadata(dest).map(|m| m.len()).unwrap_or(0);
    if let Some(expected) = meta.size
        && len != expected
    {
        return Err(format!(
            "size mismatch for {}: got {len}, want {expected}",
            dest.display()
        ));
    }
    if let Some(expected) = meta.sha256.as_deref() {
        let actual = sha256_file(dest)?;
        if actual != expected {
            let _ = fs::remove_file(dest);
            return Err(format!("sha256 mismatch for {}", dest.display()));
        }
        write_sidecar(dest, len, expected);
    }
    Ok(len)
}

/// Snapshot files `pull hf:<repo>:safetensors` downloads: the files
/// mistral.rs needs from a transformers snapshot directory (flat layout;
/// anything nested is skipped).
fn is_snapshot_file(name: &str) -> bool {
    if name.contains('/') {
        return false;
    }
    name.ends_with(".safetensors")
        || name.ends_with(".index.json")
        || matches!(
            name,
            "config.json"
                | "tokenizer.json"
                | "tokenizer_config.json"
                | "generation_config.json"
                | "preprocessor_config.json"
                | "chat_template.jinja"
        )
}

/// Download a safetensors snapshot for the mistral backend (P9.2):
/// `config.json` + tokenizer + weights into the repo dir, each file
/// verified like a GGUF pull. Already-present files are skipped through
/// the same `.verified` sidecar fast path.
fn pull_safetensors(repo: &str) -> Result<PulledModel, String> {
    let fetcher = Fetcher::new().map_err(|e| e.to_string())?;
    let siblings = fetcher.siblings_all(repo).map_err(|e| e.to_string())?;
    if !siblings.iter().any(|f| f == "config.json") {
        return Err(format!(
            "{repo}: no config.json among repo files — not a transformers snapshot \
             (`runa pull hf:<repo>:safetensors` needs one)"
        ));
    }
    let wanted: Vec<&String> = siblings.iter().filter(|f| is_snapshot_file(f)).collect();
    if !wanted.iter().any(|f| f.ends_with(".safetensors")) {
        return Err(format!(
            "{repo}: no .safetensors weights among repo files — nothing for the mistral backend to load"
        ));
    }
    let (owner, name) = split_id(repo);
    let client = HFClientSync::new().map_err(|e| format!("hub client: {e}"))?;
    let dir = repo_dir(repo);
    fs::create_dir_all(&dir).map_err(|e| format!("mkdir {}: {e}", dir.display()))?;
    let mut total = 0u64;
    let mut fresh_any = false;
    for file in &wanted {
        let file: &str = file;
        let url = resolve_hf_url(
            &HfRef {
                repo: repo.to_owned(),
                file_or_quant: file.to_owned(),
            },
            file,
        );
        let meta = fetcher.head_metadata(&url).map_err(|e| e.to_string())?;
        let dest = dir.join(file);
        if let Some(hit) = cached_if_verified(&dest, &meta, repo, file) {
            total += hit.size;
            continue;
        }
        client
            .model(owner, name)
            .download_file()
            .filename(file)
            .local_dir(&dir)
            .send()
            .map_err(|e| format!("download {repo}/{file}: {e}"))?;
        if !dest.is_file() {
            return Err(format!("download finished but {} missing", dest.display()));
        }
        total += verify_downloaded(&dest, &meta)?;
        fresh_any = true;
    }
    let n = wanted.len();
    println!("pulled {repo} (safetensors snapshot, {n} files, {total} bytes)");
    Ok(PulledModel {
        path: dir,
        repo: repo.to_owned(),
        file: "safetensors".to_owned(),
        size: total,
        fresh: fresh_any,
    })
}

/// Resolve a model reference for `run`/`chat` against local files, aliases
/// and the pull store (no downloading — that is `pull`'s job).
pub fn find_local(model_ref: &str, aliases: &AliasTable) -> Result<PathBuf, String> {
    // 1. Literal local path: a .gguf file or a mistral model directory.
    let direct = PathBuf::from(model_ref);
    if direct.is_file() {
        return Ok(direct);
    }
    if direct.is_dir() {
        return Ok(direct);
    }
    // 2. Alias → recurse once.
    if let Some(alias) = aliases.get(model_ref) {
        let target = alias.source.clone();
        if target == model_ref {
            return Err(format!("alias {model_ref} points at itself"));
        }
        return find_local(&target, &AliasTable::default());
    }
    // 3. Store lookup for `hf:` refs and bare filenames.
    if let Ok(src) = parse_model_ref(model_ref) {
        match src {
            ModelSource::Hf(r) => {
                // P9.2: `hf:<repo>:safetensors` resolves to the snapshot dir.
                if runa_fit::is_safetensors_tag(&r.file_or_quant) {
                    let dir = repo_dir(&r.repo);
                    if runa_core::is_mistral_dir(&dir) {
                        return Ok(dir);
                    }
                    return Err(format!(
                        "hf:{}:safetensors not downloaded — `runa pull hf:{}:safetensors` first (P9.2)",
                        r.repo, r.repo
                    ));
                }
                let dir = repo_dir(&r.repo);
                if r.file_or_quant.to_lowercase().ends_with(".gguf") {
                    let p = dir.join(&r.file_or_quant);
                    if p.is_file() {
                        return Ok(p);
                    }
                } else if dir.is_dir() {
                    let mut names: Vec<String> = fs::read_dir(&dir)
                        .map_err(|e| format!("read {}: {e}", dir.display()))?
                        .filter_map(|e| e.ok())
                        .map(|e| e.file_name().to_string_lossy().into_owned())
                        .filter(|n| n.to_lowercase().ends_with(".gguf"))
                        .collect();
                    names.sort();
                    if let Some(best) = pick_quant(&names, &r.file_or_quant) {
                        return Ok(dir.join(best));
                    }
                }
                return Err(format!(
                    "{} not downloaded — `runa pull {model_ref}` first (P2.4)",
                    model_ref
                ));
            }
            ModelSource::Url(_) => {
                return Err(format!(
                    "{model_ref}: direct URLs are not stored — download it first"
                ));
            }
            ModelSource::Local(_) => {} // fell through: is_file was false above
        }
    }
    // 4. Bare filename anywhere in the store.
    if let Some(p) = find_bare_filename(model_ref) {
        return Ok(p);
    }
    if model_ref.starts_with("hf:") {
        return Err(format!(
            "{model_ref}: not downloaded — `runa pull {model_ref}` first (P2.4)"
        ));
    }
    Err(format!(
        "{model_ref}: no such model file (and no alias by that name)"
    ))
}

fn find_bare_filename(name: &str) -> Option<PathBuf> {
    let root = models_dir();
    let mut stack = vec![root];
    while let Some(dir) = stack.pop() {
        let entries = fs::read_dir(dir).ok()?;
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.file_name().map(|n| n == name).unwrap_or(false) {
                return Some(p);
            }
        }
    }
    None
}

/// Every stored model file (for `runa models`).
pub fn list_models() -> Vec<PathBuf> {
    let root = models_dir();
    let mut out = Vec::new();
    let mut stack = vec![root];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = fs::read_dir(dir) else {
            continue;
        };
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.extension().map(|x| x == "gguf").unwrap_or(false) {
                out.push(p);
            }
        }
    }
    out.sort();
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{AliasTable, ModelAlias};
    use runa_fit::FileMeta;
    use std::sync::Mutex;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn with_data_root<F: FnOnce()>(f: F) {
        let _guard = ENV_LOCK.lock().unwrap();
        let root = std::env::temp_dir().join(format!(
            "runa-pull-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&root).unwrap();
        unsafe {
            std::env::set_var("XDG_DATA_HOME", root.to_str().unwrap());
        }
        f();
        unsafe {
            std::env::remove_var("XDG_DATA_HOME");
        }
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn repo_dir_sanitizes_slash() {
        with_data_root(|| {
            let p = repo_dir("unsloth/Qwen3-8B-GGUF");
            assert!(p.ends_with("unsloth-Qwen3-8B-GGUF"));
        });
    }

    #[test]
    fn find_local_hf_in_store() {
        with_data_root(|| {
            let repo = "test/local-model";
            let file = "tiny-Q4_0.gguf";
            let dest = repo_dir(repo).join(file);
            std::fs::create_dir_all(dest.parent().unwrap()).unwrap();
            std::fs::write(&dest, b"gguf").unwrap();
            let path =
                find_local("hf:test/local-model:tiny-Q4_0.gguf", &AliasTable::default()).unwrap();
            assert_eq!(path, dest);
        });
    }

    #[test]
    fn find_local_hf_missing_points_at_pull() {
        with_data_root(|| {
            let err = find_local("hf:test/missing:Q4_K_M", &AliasTable::default()).unwrap_err();
            assert!(err.contains("runa pull"));
        });
    }

    #[test]
    fn find_local_resolves_alias() {
        with_data_root(|| {
            let repo = "org/weights";
            let file = "m.gguf";
            let dest = repo_dir(repo).join(file);
            std::fs::create_dir_all(dest.parent().unwrap()).unwrap();
            std::fs::write(&dest, b"x").unwrap();
            let aliases = AliasTable {
                models: [(
                    "qwen".to_owned(),
                    ModelAlias {
                        source: "hf:org/weights:m.gguf".to_owned(),
                    },
                )]
                .into(),
            };
            let path = find_local("qwen", &aliases).unwrap();
            assert_eq!(path, dest);
        });
    }

    #[test]
    fn find_local_safetensors_dir_in_store() {
        with_data_root(|| {
            let repo = "test/safe-model";
            let dir = repo_dir(repo);
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join("config.json"), b"{}").unwrap();
            std::fs::write(dir.join("model.safetensors"), b"st").unwrap();
            let path =
                find_local("hf:test/safe-model:safetensors", &AliasTable::default()).unwrap();
            assert_eq!(path, dir);
        });
    }

    #[test]
    fn find_local_safetensors_tag_is_case_insensitive() {
        with_data_root(|| {
            let repo = "test/case-model";
            let dir = repo_dir(repo);
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join("config.json"), b"{}").unwrap();
            assert!(find_local("hf:test/case-model:SAFETENSORS", &AliasTable::default()).is_ok());
        });
    }

    #[test]
    fn find_local_safetensors_missing_points_at_pull() {
        with_data_root(|| {
            let err = find_local("hf:test/absent:safetensors", &AliasTable::default()).unwrap_err();
            assert!(
                err.contains("runa pull hf:test/absent:safetensors"),
                "{err}"
            );
        });
    }

    #[test]
    fn find_local_safetensors_needs_config_json() {
        with_data_root(|| {
            let repo = "test/bare-dir";
            let dir = repo_dir(repo);
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join("model.safetensors"), b"st").unwrap();
            let err =
                find_local("hf:test/bare-dir:safetensors", &AliasTable::default()).unwrap_err();
            assert!(err.contains("runa pull"), "{err}");
        });
    }

    #[test]
    fn find_local_direct_dir_resolves() {
        with_data_root(|| {
            let dir = std::env::temp_dir().join(format!(
                "runa-pull-direct-{}",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            std::fs::create_dir_all(&dir).unwrap();
            let path = find_local(dir.to_str().unwrap(), &AliasTable::default()).unwrap();
            assert_eq!(path, dir);
            let _ = std::fs::remove_dir_all(&dir);
        });
    }

    #[test]
    fn snapshot_file_filter() {
        assert!(is_snapshot_file("model.safetensors"));
        assert!(is_snapshot_file("model-00001-of-00002.safetensors"));
        assert!(is_snapshot_file("config.json"));
        assert!(is_snapshot_file("tokenizer.json"));
        assert!(is_snapshot_file("tokenizer_config.json"));
        assert!(is_snapshot_file("model.safetensors.index.json"));
        assert!(!is_snapshot_file("README.md"));
        assert!(!is_snapshot_file("pytorch_model.bin"));
        assert!(!is_snapshot_file("sub/model.safetensors"));
    }

    #[test]
    fn list_models_finds_nested_gguf() {
        with_data_root(|| {
            let dest = repo_dir("a/b").join("nested.gguf");
            std::fs::create_dir_all(dest.parent().unwrap()).unwrap();
            std::fs::write(&dest, b"gguf").unwrap();
            let listed = list_models();
            assert!(listed.iter().any(|p| p == &dest));
        });
    }

    #[test]
    fn cached_sidecar_skips_rehash() {
        with_data_root(|| {
            let repo = "cache/repo";
            let file = "model.gguf";
            let dest = repo_dir(repo).join(file);
            std::fs::create_dir_all(dest.parent().unwrap()).unwrap();
            std::fs::write(&dest, b"same-bytes").unwrap();
            let len = 10u64;
            write_sidecar(&dest, len, "abc123");
            let meta = FileMeta {
                size: Some(len),
                sha256: Some("abc123".into()),
            };
            let hit = cached_if_verified(&dest, &meta, repo, file).unwrap();
            assert!(!hit.fresh);
            assert_eq!(hit.size, len);
        });
    }

    #[test]
    fn size_limit_accepts_files_at_or_under_3gib() {
        let _guard = ENV_LOCK.lock().unwrap();
        unsafe {
            std::env::remove_var("RUNA_MAX_MODEL_BYTES");
        }
        assert_eq!(MAX_MODEL_BYTES, 3 * 1024 * 1024 * 1024);
        check_size_limit(0, "r", "f").unwrap();
        check_size_limit(MAX_MODEL_BYTES, "r", "f").unwrap();
    }

    #[test]
    fn size_limit_rejects_files_over_3gib() {
        let _guard = ENV_LOCK.lock().unwrap();
        unsafe {
            std::env::remove_var("RUNA_MAX_MODEL_BYTES");
        }
        let err = check_size_limit(MAX_MODEL_BYTES + 1, "org/repo", "big.gguf").unwrap_err();
        assert!(err.contains("org/repo/big.gguf"), "{err}");
        assert!(err.contains("RUNA_MAX_MODEL_BYTES"), "{err}");
    }

    #[test]
    fn size_limit_env_override_raises_the_ceiling() {
        let _guard = ENV_LOCK.lock().unwrap();
        unsafe {
            std::env::set_var("RUNA_MAX_MODEL_BYTES", "9999999999");
        }
        check_size_limit(MAX_MODEL_BYTES + 1, "r", "f").unwrap();
        unsafe {
            std::env::remove_var("RUNA_MAX_MODEL_BYTES");
        }
    }
}
