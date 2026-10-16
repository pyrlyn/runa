// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! Model file formats, told apart by their leading bytes rather than the
//! extension, which anyone can rename.

use std::collections::BTreeMap;
use std::fmt;
use std::fs::File;
use std::io::{self, Read};
use std::path::{Path, PathBuf};

/// Largest safetensors JSON header read into memory. The reference parser
/// (`huggingface/safetensors`) refuses anything bigger, so no valid file
/// needs more.
pub const MAX_SAFETENSORS_HEADER: u64 = 100 * 1024 * 1024;

/// Extensions of PyTorch / pickle checkpoints. Unpickling runs code, so these
/// are refused before a byte is read.
const PICKLE_EXTENSIONS: &[&str] = &["bin", "pt", "pth", "ckpt", "pkl", "pickle"];

/// A model file format runa can check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFormat {
    Gguf,
    Safetensors,
    Wasm,
}

impl fmt::Display for ModelFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Gguf => "GGUF",
            Self::Safetensors => "safetensors",
            Self::Wasm => "Wasm",
        })
    }
}

/// Why a file was refused.
#[derive(Debug, thiserror::Error)]
pub enum FormatError {
    #[error("read {}: {source}", path.display())]
    Io { path: PathBuf, source: io::Error },
    #[error(
        "{}: pickle or PyTorch checkpoint refused (loading it can run code); convert it to GGUF or safetensors",
        path.display()
    )]
    Pickle { path: PathBuf },
    #[error("{}: not a GGUF, safetensors or Wasm file", path.display())]
    Unknown { path: PathBuf },
    #[error("{}: corrupt {format} file: {detail}", path.display())]
    Corrupt {
        path: PathBuf,
        format: ModelFormat,
        detail: String,
    },
    #[error("{}: named as {expected} but the contents are {found}", path.display())]
    Mismatch {
        path: PathBuf,
        expected: ModelFormat,
        found: ModelFormat,
    },
}

/// Tell the format of `path` from its contents and check its header. A known
/// extension must agree with the contents; pickle files are refused.
pub fn detect(path: &Path) -> Result<ModelFormat, FormatError> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase);
    if ext
        .as_deref()
        .is_some_and(|e| PICKLE_EXTENSIONS.contains(&e))
    {
        return Err(FormatError::Pickle { path: path.into() });
    }
    let expected = match ext.as_deref() {
        Some("gguf") => Some(ModelFormat::Gguf),
        Some("safetensors") => Some(ModelFormat::Safetensors),
        Some("wasm") => Some(ModelFormat::Wasm),
        _ => None,
    };
    let io = |source: io::Error| FormatError::Io {
        path: path.into(),
        source,
    };
    let mut file = File::open(path).map_err(io)?;
    let len = file.metadata().map_err(io)?.len();
    let mut head = [0u8; 9];
    let got = read_up_to(&mut file, &mut head).map_err(io)?;
    let found = sniff(&head[..got], len).ok_or_else(|| match expected {
        Some(format) => FormatError::Corrupt {
            path: path.into(),
            format,
            detail: "wrong magic bytes".into(),
        },
        None => FormatError::Unknown { path: path.into() },
    })?;
    let found = found.map_err(|()| FormatError::Pickle { path: path.into() })?;
    if let Some(expected) = expected.filter(|e| *e != found) {
        return Err(FormatError::Mismatch {
            path: path.into(),
            expected,
            found,
        });
    }
    let corrupt = |detail: String| FormatError::Corrupt {
        path: path.into(),
        format: found,
        detail,
    };
    match found {
        ModelFormat::Gguf => runa_fit::read_local_prefix(path)
            .map(drop)
            .map_err(|e| corrupt(e.to_string()))?,
        ModelFormat::Safetensors => {
            check_safetensors(&mut file, &head, len).map_err(|e| match e {
                SafetensorsError::Io(source) => io(source),
                SafetensorsError::Bad(detail) => corrupt(detail),
            })?
        }
        // The magic plus a known version word is the whole module header;
        // the sections are validated by the runtime that loads it (P18.5).
        ModelFormat::Wasm => {}
    }
    Ok(found)
}

/// Format by magic bytes. `Some(Err(()))` is a pickle or PyTorch zip archive.
fn sniff(head: &[u8], len: u64) -> Option<Result<ModelFormat, ()>> {
    const WASM_CORE: [u8; 4] = [1, 0, 0, 0];
    // Component model: version 0x0d, layer 1 (the binary format draft).
    const WASM_COMPONENT: [u8; 4] = [0x0d, 0, 1, 0];
    match head {
        [b'G', b'G', b'U', b'F', ..] => Some(Ok(ModelFormat::Gguf)),
        [0, b'a', b's', b'm', v @ ..] if v.len() >= 4 => {
            let v = &v[..4];
            (v == WASM_CORE || v == WASM_COMPONENT).then_some(Ok(ModelFormat::Wasm))
        }
        // Before the pickle arm: a header length such as 0x0280 starts with
        // the same two bytes as a protocol-2 pickle.
        [n @ .., b'{']
            if n.len() == 8
                && u64::from_le_bytes(n.try_into().ok()?)
                    .checked_add(8)
                    .is_some_and(|end| end <= len) =>
        {
            Some(Ok(ModelFormat::Safetensors))
        }
        [b'P', b'K', 3, 4, ..] | [0x80, 2..=5, ..] => Some(Err(())),
        _ => None,
    }
}

enum SafetensorsError {
    Io(io::Error),
    Bad(String),
}

#[derive(serde::Deserialize)]
struct TensorInfo {
    data_offsets: [u64; 2],
}

/// The header is an 8-byte little-endian length, then that many bytes of JSON
/// mapping tensor names to dtype, shape and `data_offsets` into the rest of
/// the file. Every offset pair must lie inside the data section.
fn check_safetensors(file: &mut File, head: &[u8; 9], len: u64) -> Result<(), SafetensorsError> {
    let bad = |s: String| SafetensorsError::Bad(s);
    let n = u64::from_le_bytes(head[..8].try_into().expect("8 bytes"));
    if n > MAX_SAFETENSORS_HEADER {
        return Err(bad(format!(
            "header is {n} bytes, over the {MAX_SAFETENSORS_HEADER}-byte cap"
        )));
    }
    if n < 2 {
        return Err(bad("header is empty".into()));
    }
    let mut json = vec![b'{'];
    json.resize(n as usize, 0);
    file.read_exact(&mut json[1..])
        .map_err(SafetensorsError::Io)?;
    let header: BTreeMap<String, serde_json::Value> =
        serde_json::from_slice(&json).map_err(|e| bad(format!("header JSON: {e}")))?;
    let data_len = len - 8 - n;
    for (name, value) in header {
        if name == "__metadata__" {
            continue;
        }
        let info: TensorInfo =
            serde_json::from_value(value).map_err(|e| bad(format!("tensor {name:?}: {e}")))?;
        let [start, end] = info.data_offsets;
        if start > end || end > data_len {
            return Err(bad(format!(
                "tensor {name:?} spans {start}..{end}, past the {data_len}-byte data section"
            )));
        }
    }
    Ok(())
}

/// `read` until `buf` is full or the file ends; returns the bytes read.
fn read_up_to(file: &mut File, buf: &mut [u8]) -> io::Result<usize> {
    let mut got = 0;
    while got < buf.len() {
        match file.read(&mut buf[got..]) {
            Ok(0) => break,
            Ok(k) => got += k,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(got)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn write(dir: &Path, name: &str, bytes: &[u8]) -> PathBuf {
        let p = dir.join(name);
        fs::write(&p, bytes).unwrap();
        p
    }

    fn safetensors(header: &str, data: usize) -> Vec<u8> {
        let mut out = (header.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(header.as_bytes());
        out.resize(out.len() + data, 0);
        out
    }

    fn fixture(name: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures")
            .join(name)
    }

    #[test]
    fn gguf_fixture_is_gguf() {
        assert_eq!(
            detect(&fixture("synthetic-llama.gguf")).unwrap(),
            ModelFormat::Gguf
        );
    }

    #[test]
    fn truncated_gguf_is_corrupt() {
        let d = tempfile::tempdir().unwrap();
        let full = fs::read(fixture("synthetic-llama.gguf")).unwrap();
        let p = write(d.path(), "cut.gguf", &full[..40]);
        assert!(matches!(
            detect(&p),
            Err(FormatError::Corrupt {
                format: ModelFormat::Gguf,
                ..
            })
        ));
    }

    #[test]
    fn safetensors_header_is_checked() {
        let d = tempfile::tempdir().unwrap();
        let ok =
            r#"{"__metadata__":{"k":"v"},"w":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
        let p = write(d.path(), "m.safetensors", &safetensors(ok, 8));
        assert_eq!(detect(&p).unwrap(), ModelFormat::Safetensors);

        let p = write(d.path(), "short.safetensors", &safetensors(ok, 4));
        let err = detect(&p).unwrap_err().to_string();
        assert!(err.contains("past the 4-byte data section"), "{err}");

        let p = write(d.path(), "junk.safetensors", &safetensors("{not json", 0));
        assert!(matches!(detect(&p), Err(FormatError::Corrupt { .. })));
    }

    #[test]
    fn safetensors_length_that_looks_like_pickle_is_safetensors() {
        let d = tempfile::tempdir().unwrap();
        // 0x0280 = 640: the length starts with `80 02`, a protocol-2 pickle.
        let body = r#"{"w":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}"#;
        let header = format!("{body:<640}");
        let p = write(d.path(), "m.safetensors", &safetensors(&header, 1));
        assert_eq!(detect(&p).unwrap(), ModelFormat::Safetensors);
    }

    #[test]
    fn oversized_safetensors_header_is_refused_without_reading_it() {
        let d = tempfile::tempdir().unwrap();
        let n = MAX_SAFETENSORS_HEADER + 1;
        let mut bytes = n.to_le_bytes().to_vec();
        bytes.push(b'{');
        let p = write(d.path(), "big.safetensors", &bytes);
        let f = File::options().write(true).open(&p).unwrap();
        // Sparse: the length check must hold before any allocation.
        f.set_len(n + 8).unwrap();
        let err = detect(&p).unwrap_err().to_string();
        assert!(err.contains("cap"), "{err}");
    }

    #[test]
    fn wasm_core_and_component_are_wasm() {
        let d = tempfile::tempdir().unwrap();
        let core = write(d.path(), "m.wasm", b"\0asm\x01\0\0\0");
        let comp = write(d.path(), "c.wasm", b"\0asm\x0d\0\x01\0");
        let odd = write(d.path(), "v.wasm", b"\0asm\x09\0\0\0");
        assert_eq!(detect(&core).unwrap(), ModelFormat::Wasm);
        assert_eq!(detect(&comp).unwrap(), ModelFormat::Wasm);
        assert!(matches!(detect(&odd), Err(FormatError::Corrupt { .. })));
    }

    #[test]
    fn pickle_is_refused_by_name_and_by_contents() {
        let d = tempfile::tempdir().unwrap();
        for name in ["w.bin", "w.pt", "w.PTH", "w.ckpt", "w.pkl"] {
            let p = write(d.path(), name, b"GGUF");
            assert!(
                matches!(detect(&p), Err(FormatError::Pickle { .. })),
                "{name}"
            );
        }
        let zip = write(d.path(), "renamed.gguf", b"PK\x03\x04rest");
        let raw = write(d.path(), "renamed", b"\x80\x04\x95rest");
        assert!(matches!(detect(&zip), Err(FormatError::Pickle { .. })));
        assert!(matches!(detect(&raw), Err(FormatError::Pickle { .. })));
    }

    #[test]
    fn extension_must_match_contents() {
        let d = tempfile::tempdir().unwrap();
        let p = write(d.path(), "fake.gguf", b"\0asm\x01\0\0\0");
        assert!(matches!(
            detect(&p),
            Err(FormatError::Mismatch {
                expected: ModelFormat::Gguf,
                found: ModelFormat::Wasm,
                ..
            })
        ));
        let p = write(d.path(), "notes.txt", b"hello world");
        assert!(matches!(detect(&p), Err(FormatError::Unknown { .. })));
        let p = write(d.path(), "empty.gguf", b"");
        assert!(matches!(detect(&p), Err(FormatError::Corrupt { .. })));
    }
}
