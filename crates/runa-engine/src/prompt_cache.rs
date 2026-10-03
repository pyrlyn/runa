// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! P2.8 prompt cache: llama.cpp KV/session state in LMDB (heed).
//!
//! Keys are SHA-256 of model path + placement + ctx + prompt token ids.
//! Values are `llama_copy_state_data` blobs. A hit restores with
//! `llama_set_state_data` and skips prefill.
//!
//! Idle (`on_idle`) unmaps the environment (D17 / P7.2) but leaves the
//! files on disk so the next run can reuse prefixes.

use std::fs;
use std::path::{Path, PathBuf};

use heed::types::Bytes;
use heed::{Database, Env, EnvOpenOptions};
use llama_cpp_2::token::LlamaToken;
use sha2::{Digest, Sha256};

use crate::load::{KvKind, LoadConfig};
use crate::placement::Placement;

/// Default LMDB map size (virtual; not pre-allocated). Must be a multiple
/// of the page size. 8 GiB holds several large-context KV snapshots.
pub const DEFAULT_MAP_SIZE: usize = 8 * 1024 * 1024 * 1024;

struct Opened {
    env: Env,
    db: Database<Bytes, Bytes>,
}

/// LMDB-backed prefix cache for llama.cpp context state.
pub struct PromptCache {
    dir: PathBuf,
    map_size: usize,
    opened: Option<Opened>,
}

impl PromptCache {
    /// Open (or create) an LMDB environment in `dir`.
    pub fn open(dir: impl AsRef<Path>) -> Result<Self, String> {
        Self::open_with_map_size(dir, DEFAULT_MAP_SIZE)
    }

    /// Like [`open`](Self::open) with an explicit map size (tests).
    pub fn open_with_map_size(dir: impl AsRef<Path>, map_size: usize) -> Result<Self, String> {
        let dir = dir.as_ref().to_path_buf();
        fs::create_dir_all(&dir).map_err(|e| format!("create {}: {e}", dir.display()))?;
        let mut cache = Self {
            dir,
            map_size,
            opened: None,
        };
        cache.ensure_open()?;
        Ok(cache)
    }

    /// Drop the mmap (release process RSS). Disk files stay.
    pub fn on_idle(&mut self) {
        self.opened = None;
    }

    /// Fetch a previously stored state blob.
    pub fn get(&mut self, key: &[u8]) -> Result<Option<Vec<u8>>, String> {
        let opened = self.ensure_open()?;
        let rtxn = opened.env.read_txn().map_err(|e| e.to_string())?;
        let val = opened
            .db
            .get(&rtxn, key)
            .map_err(|e| e.to_string())?
            .map(|b| b.to_vec());
        Ok(val)
    }

    /// Store a state blob under `key`.
    pub fn put(&mut self, key: &[u8], val: &[u8]) -> Result<(), String> {
        let opened = self.ensure_open()?;
        let mut wtxn = opened.env.write_txn().map_err(|e| e.to_string())?;
        opened
            .db
            .put(&mut wtxn, key, val)
            .map_err(|e| e.to_string())?;
        wtxn.commit().map_err(|e| e.to_string())?;
        Ok(())
    }

    fn ensure_open(&mut self) -> Result<&mut Opened, String> {
        if self.opened.is_none() {
            self.opened = Some(open_env(&self.dir, self.map_size)?);
        }
        Ok(self.opened.as_mut().expect("just opened"))
    }
}

fn open_env(dir: &Path, map_size: usize) -> Result<Opened, String> {
    // SAFETY: heed requires `open` to be unsafe because LMDB mmaps the
    // directory. File locks serialize writers; we only use this path with
    // a single Env per process for a given dir.
    let env = unsafe {
        EnvOpenOptions::new()
            .map_size(map_size)
            .open(dir)
            .map_err(|e| format!("lmdb open {}: {e}", dir.display()))?
    };
    let mut wtxn = env.write_txn().map_err(|e| e.to_string())?;
    let db: Database<Bytes, Bytes> = env
        .create_database(&mut wtxn, None)
        .map_err(|e| e.to_string())?;
    wtxn.commit().map_err(|e| e.to_string())?;
    Ok(Opened { env, db })
}

/// Hash identifying a cached prefix: model file, placement, ctx/KV, LoRA
/// adapters (P8.5: different adapters must not share KV state), tokens.
pub fn prefix_key(
    path: &Path,
    placement: &Placement,
    config: &LoadConfig,
    tokens: &[LlamaToken],
) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(path.to_string_lossy().as_bytes());
    h.update(b"\0");
    h.update(placement.n_gpu_layers.to_le_bytes());
    h.update(placement.main_gpu.to_le_bytes());
    for spec in &placement.devices {
        h.update(spec.as_bytes());
        h.update(b"\0");
    }
    for v in &placement.tensor_split {
        h.update(v.to_le_bytes());
    }
    for srv in &placement.rpc_servers {
        h.update(srv.as_bytes());
        h.update(b"\0");
    }
    for pat in &placement.cpu_patterns {
        h.update(pat.as_bytes());
        h.update(b"\0");
    }
    h.update(config.n_ctx.to_le_bytes());
    h.update(config.n_batch.to_le_bytes());
    h.update(config.n_ubatch.to_le_bytes());
    h.update([kv_tag(config.kv_k), kv_tag(config.kv_v)]);
    for spec in &config.loras {
        h.update(spec.path.to_string_lossy().as_bytes());
        h.update(b"\0");
        h.update(spec.scale.to_le_bytes());
    }
    h.update((tokens.len() as u64).to_le_bytes());
    for t in tokens {
        h.update(t.0.to_le_bytes());
    }
    h.finalize().into()
}

fn kv_tag(kind: Option<KvKind>) -> u8 {
    match kind {
        None => 255,
        Some(KvKind::F16) => 0,
        Some(KvKind::Q8_0) => 1,
        Some(KvKind::Q4_0) => 2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lmdb_put_get_survives_idle_and_reopen() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut cache =
            PromptCache::open_with_map_size(dir.path(), 64 * 1024 * 1024).expect("open");
        cache.put(b"k", b"hello-lmdb").expect("put");
        assert_eq!(
            cache.get(b"k").expect("get").as_deref(),
            Some(b"hello-lmdb".as_slice())
        );
        cache.on_idle();
        assert_eq!(
            cache.get(b"k").expect("get after idle").as_deref(),
            Some(b"hello-lmdb".as_slice())
        );
        drop(cache);
        let mut cache =
            PromptCache::open_with_map_size(dir.path(), 64 * 1024 * 1024).expect("reopen");
        assert_eq!(
            cache.get(b"k").expect("get after reopen").as_deref(),
            Some(b"hello-lmdb".as_slice())
        );
    }
}
