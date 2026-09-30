//! Replays the `gguf-header` fuzz reproducers (fuzz/regressions/) through
//! the header reader, descriptor and KV estimate, so fixed crashes stay
//! fixed in plain `cargo test` without nightly or cargo-fuzz.
//!
//! `estimate_compute` / `check_fit` are left out on purpose: their integer
//! math still overflows on hostile header values (debug-build panics, see
//! fuzz/README.md "Known findings").

use std::path::Path;

use runa_fit::{Descriptor, Reader, estimate_kv, tensor_bytes};

#[test]
fn gguf_header_reproducers_do_not_panic() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fuzz/regressions/gguf-header");
    let mut seen = 0;
    for entry in std::fs::read_dir(&dir).expect("fuzz/regressions/gguf-header") {
        let path = entry.expect("dir entry").path();
        let bytes = std::fs::read(&path).expect("read reproducer");
        seen += 1;
        let Ok(reader) = Reader::parse(&bytes) else {
            continue;
        };
        let Ok(desc) = Descriptor::from_reader(&reader) else {
            continue;
        };
        for t in &reader.tensors {
            let _ = tensor_bytes(&t.dims, t.ggml_type);
        }
        let _ = estimate_kv(&desc, 4096, "f16");
    }
    assert!(seen > 0, "no reproducers in {}", dir.display());
}
