// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! The download store plus the user's model folders, resolved from
//! [`StoreConfig`].

use std::fs;
use std::path::{Path, PathBuf};

use crate::config::{ConfigError, DEFAULT_MAX_BYTES, StoreConfig, resolve_path};
use crate::paths::{default_cache_dir, home_dir};

/// Resolved `[store]`: absolute folders and the download cap.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelStore {
    cache_dir: PathBuf,
    dirs: Vec<PathBuf>,
    max_bytes: u64,
}

impl Default for ModelStore {
    /// No `[store]` table: the default download store, no extra folders.
    fn default() -> Self {
        Self {
            cache_dir: default_cache_dir(),
            dirs: Vec::new(),
            max_bytes: DEFAULT_MAX_BYTES,
        }
    }
}

impl ModelStore {
    /// Resolve `~/` against the home directory and refuse relative paths.
    pub fn new(config: &StoreConfig) -> Result<Self, ConfigError> {
        Self::with_home(config, home_dir().as_deref())
    }

    fn with_home(config: &StoreConfig, home: Option<&Path>) -> Result<Self, ConfigError> {
        let cache_dir = match &config.cache_dir {
            Some(p) => resolve_path("cache_dir", p, home)?,
            None => default_cache_dir(),
        };
        let dirs = config
            .dirs
            .iter()
            .map(|p| resolve_path("dirs", p, home))
            .collect::<Result<_, _>>()?;
        Ok(Self {
            cache_dir,
            dirs,
            max_bytes: config.max_bytes.map_or(DEFAULT_MAX_BYTES, |n| n.get()),
        })
    }

    /// Where `runa pull` writes.
    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    /// Extra folders from `[store] dirs`, in config order.
    pub fn dirs(&self) -> &[PathBuf] {
        &self.dirs
    }

    /// Largest fresh download, in bytes.
    pub fn max_bytes(&self) -> u64 {
        self.max_bytes
    }

    /// Download store first, then the folders in order.
    fn roots(&self) -> impl Iterator<Item = &Path> {
        std::iter::once(self.cache_dir.as_path()).chain(self.dirs.iter().map(PathBuf::as_path))
    }

    /// First file named `name` under any root, searched recursively.
    pub fn find_file(&self, name: &str) -> Option<PathBuf> {
        self.roots().find_map(|root| {
            let mut hit = None;
            walk(root, &mut |p| {
                let found = p.file_name().is_some_and(|n| n == name);
                if found {
                    hit = Some(p.to_path_buf());
                }
                found
            });
            hit
        })
    }

    /// Every file with extension `ext` under every root, sorted, without
    /// duplicates (a folder may also sit inside the store).
    pub fn list_files(&self, ext: &str) -> Vec<PathBuf> {
        let mut out = Vec::new();
        for root in self.roots() {
            walk(root, &mut |p| {
                if p.extension().is_some_and(|x| x == ext) {
                    out.push(p.to_path_buf());
                }
                false
            });
        }
        out.sort();
        out.dedup();
        out
    }
}

/// Visit regular files (and symlinks to them) under `root` until `visit`
/// returns true. Symlinked directories are not entered, so a link cycle in a
/// user folder cannot loop forever; unreadable directories are skipped.
fn walk(root: &Path, visit: &mut dyn FnMut(&Path) -> bool) {
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let Ok(kind) = entry.file_type() else {
                continue;
            };
            let path = entry.path();
            if kind.is_dir() {
                stack.push(path);
            } else if path.is_file() && visit(&path) {
                return;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store(cache: &Path, dirs: &[&Path]) -> ModelStore {
        ModelStore {
            cache_dir: cache.to_path_buf(),
            dirs: dirs.iter().map(|d| d.to_path_buf()).collect(),
            max_bytes: DEFAULT_MAX_BYTES,
        }
    }

    fn touch(path: &Path) {
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, b"x").unwrap();
    }

    #[test]
    fn config_resolves_home_and_cap() {
        // Temp dirs, not literal `/srv`: a rooted path without a drive is
        // relative on Windows.
        let home = tempfile::tempdir().unwrap();
        let cache = tempfile::tempdir().unwrap();
        let cfg = StoreConfig {
            dirs: vec![PathBuf::from("~/llm")],
            cache_dir: Some(cache.path().to_path_buf()),
            max_bytes: std::num::NonZeroU64::new(7),
        };
        let s = ModelStore::with_home(&cfg, Some(home.path())).unwrap();
        assert_eq!(s.cache_dir(), cache.path());
        assert_eq!(s.dirs(), [home.path().join("llm")]);
        assert_eq!(s.max_bytes(), 7);
    }

    #[test]
    fn relative_dir_fails_the_whole_store() {
        let cfg: StoreConfig = toml::from_str("dirs = [\"llm\"]\n").unwrap();
        assert!(matches!(
            ModelStore::with_home(&cfg, None),
            Err(ConfigError::Relative { key: "dirs", .. })
        ));
    }

    #[test]
    fn finds_a_file_in_an_extra_folder() {
        let cache = tempfile::tempdir().unwrap();
        let extra = tempfile::tempdir().unwrap();
        let model = extra.path().join("nested").join("tiny.gguf");
        touch(&model);
        let s = store(cache.path(), &[extra.path()]);
        assert_eq!(s.find_file("tiny.gguf"), Some(model));
        assert_eq!(s.find_file("missing.gguf"), None);
    }

    #[test]
    fn store_wins_over_folders() {
        let cache = tempfile::tempdir().unwrap();
        let extra = tempfile::tempdir().unwrap();
        touch(&cache.path().join("org--m").join("a.gguf"));
        touch(&extra.path().join("a.gguf"));
        let s = store(cache.path(), &[extra.path()]);
        assert_eq!(
            s.find_file("a.gguf"),
            Some(cache.path().join("org--m").join("a.gguf"))
        );
    }

    #[test]
    fn missing_folder_is_skipped() {
        let cache = tempfile::tempdir().unwrap();
        let s = store(cache.path(), &[Path::new("/no/such/runa/dir")]);
        assert_eq!(s.find_file("a.gguf"), None);
        assert!(s.list_files("gguf").is_empty());
    }

    #[test]
    fn lists_every_root_once() {
        let cache = tempfile::tempdir().unwrap();
        let inner = cache.path().join("mine");
        touch(&cache.path().join("b.gguf"));
        touch(&inner.join("a.gguf"));
        touch(&inner.join("notes.txt"));
        // The extra folder sits inside the store: its file must not repeat.
        let s = store(cache.path(), &[&inner]);
        assert_eq!(
            s.list_files("gguf"),
            vec![cache.path().join("b.gguf"), inner.join("a.gguf")]
        );
    }

    #[cfg(unix)]
    #[test]
    fn symlink_cycle_does_not_hang() {
        let cache = tempfile::tempdir().unwrap();
        std::os::unix::fs::symlink(cache.path(), cache.path().join("loop")).unwrap();
        touch(&cache.path().join("m.gguf"));
        let s = store(cache.path(), &[]);
        assert_eq!(s.list_files("gguf"), vec![cache.path().join("m.gguf")]);
    }
}
