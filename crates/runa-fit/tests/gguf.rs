// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! GGUF reader tests: round-trip (proptest) + real-fixture parsing.

use runa_fit::gguf::{ArrayValue, GGUF_MAGIC, GgmlType, ReadError, Reader, Value};

/// A minimal GGUF writer used by tests/proptest to round-trip the reader.
struct Writer {
    buf: Vec<u8>,
}

impl Writer {
    fn new(version: u32, tensors: &[(&str, &[u64], GgmlType)]) -> Writer {
        let mut w = Writer { buf: Vec::new() };
        w.buf.extend_from_slice(&GGUF_MAGIC);
        w.u32(version);
        w.u64(tensors.len() as u64);
        w.u64(0); // no metadata
        for (name, dims, ty) in tensors {
            w.str(name);
            w.u32(dims.len() as u32);
            for &d in *dims {
                w.u64(d);
            }
            w.u32(gtype_raw(*ty));
            w.u64(0); // offset
        }
        while !w.buf.len().is_multiple_of(32) {
            w.buf.push(0);
        }
        w
    }

    fn u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn str(&mut self, s: &str) {
        self.u64(s.len() as u64);
        self.buf.extend_from_slice(s.as_bytes());
    }
}

fn gtype_raw(t: GgmlType) -> u32 {
    use GgmlType::*;
    match t {
        F32 => 0,
        Q4_0 => 2,
        Q8_0 => 8,
        Q4_K => 12,
        Q8_K => 15,
        F16 => 1,
        BF16 => 30,
        Q6_K => 14,
        MXFP4 => 26,
        _ => 0,
    }
}

#[test]
fn rejects_bad_magic() {
    assert_eq!(Reader::parse(b"NOPE"), Err(ReadError::BadMagic));
}

#[test]
fn rejects_truncated() {
    assert_eq!(Reader::parse(b""), Err(ReadError::Truncated));
    assert_eq!(Reader::parse(b"GGU"), Err(ReadError::Truncated));
}

#[test]
fn rejects_unknown_version() {
    let w = Writer::new(9, &[]);
    assert!(matches!(
        Reader::parse(&w.buf),
        Err(ReadError::UnsupportedVersion(9))
    ));
}

#[test]
fn parses_empty_header() {
    let w = Writer::new(3, &[]);
    let r = Reader::parse(&w.buf).unwrap();
    assert_eq!(r.version, 3);
    assert_eq!(r.tensor_count, 0);
    assert!(r.tensors.is_empty());
    assert!(r.metadata.is_empty());
}

#[test]
fn parses_tensor_table() {
    let tensors = [
        (
            "token_embd.weight",
            &[1536_u64, 896_u64][..],
            GgmlType::Q4_0,
        ),
        ("blk.0.attn_q.weight", &[896, 896][..], GgmlType::Q4_K),
    ];
    let w = Writer::new(3, &tensors);
    let r = Reader::parse(&w.buf).unwrap();
    assert_eq!(r.tensors.len(), 2);
    assert_eq!(r.tensors[0].name, "token_embd.weight");
    assert_eq!(r.tensors[0].dims, vec![1536, 896]);
    assert_eq!(r.tensors[0].ggml_type, GgmlType::Q4_0);
    assert_eq!(r.tensors[1].name, "blk.0.attn_q.weight");
    assert_eq!(r.tensors[1].ggml_type, GgmlType::Q4_K);
}

#[test]
fn data_start_is_32_aligned() {
    let w = Writer::new(3, &[("a", &[1, 1][..], GgmlType::F32)]);
    let r = Reader::parse(&w.buf).unwrap();
    assert_eq!(r.data_start % 32, 0);
    assert!(r.data_start > 0);
}

#[test]
fn data_start_honors_general_alignment() {
    fn put_u32(b: &mut Vec<u8>, v: u32) {
        b.extend_from_slice(&v.to_le_bytes());
    }
    fn put_u64(b: &mut Vec<u8>, v: u64) {
        b.extend_from_slice(&v.to_le_bytes());
    }
    fn put_str(b: &mut Vec<u8>, s: &str) {
        put_u64(b, s.len() as u64);
        b.extend_from_slice(s.as_bytes());
    }
    let mut b = Vec::new();
    b.extend_from_slice(&GGUF_MAGIC);
    put_u32(&mut b, 3);
    put_u64(&mut b, 0);
    put_u64(&mut b, 2);
    put_str(&mut b, "general.name");
    put_u32(&mut b, 8); // string
    put_str(&mut b, "a");
    put_str(&mut b, "general.alignment");
    put_u32(&mut b, 4); // uint32
    put_u32(&mut b, 64);
    let r = Reader::parse(&b).unwrap();
    assert_eq!(r.data_start, 128, "header end padded to alignment 64");
}

#[test]
fn metadata_all_scalar_types() {
    // Hand-build a header with one KV pair per scalar type.
    let mut b = Vec::new();
    b.extend_from_slice(&GGUF_MAGIC);
    b.extend_from_slice(&3u32.to_le_bytes());
    b.extend_from_slice(&0u64.to_le_bytes()); // tensors
    b.extend_from_slice(&7u64.to_le_bytes()); // kv count

    let push_kv = |b: &mut Vec<u8>, key: &str, ty: u32, val: &[u8]| {
        b.extend_from_slice(&(key.len() as u64).to_le_bytes());
        b.extend_from_slice(key.as_bytes());
        b.extend_from_slice(&ty.to_le_bytes());
        b.extend_from_slice(val);
    };
    push_kv(&mut b, "a.u8", 0, &[7]);
    push_kv(&mut b, "a.i8", 1, &[0xfe]);
    push_kv(&mut b, "a.u32", 4, &42u32.to_le_bytes());
    push_kv(&mut b, "a.i64", 11, &(-1i64).to_le_bytes());
    push_kv(&mut b, "a.f32", 6, &1.5f32.to_le_bytes());
    push_kv(&mut b, "a.bool", 7, &[1]);
    // string handled inline (closure can't both borrow `b` and own a temp)
    {
        let key = "a.str";
        b.extend_from_slice(&(key.len() as u64).to_le_bytes());
        b.extend_from_slice(key.as_bytes());
        b.extend_from_slice(&8u32.to_le_bytes()); // String type
        let s = "hello";
        b.extend_from_slice(&(s.len() as u64).to_le_bytes());
        b.extend_from_slice(s.as_bytes());
    }

    let r = Reader::parse(&b).unwrap();
    assert_eq!(r.get_u32("a.u32"), Some(42));
    assert_eq!(r.get("a.i8"), Some(&Value::Int8(-2)));
    assert_eq!(r.get("a.i64"), Some(&Value::Int64(-1)));
    assert_eq!(r.get_f32("a.f32"), Some(1.5));
    assert_eq!(r.get_bool("a.bool"), Some(true));
    assert_eq!(r.get_str("a.str"), Some("hello"));
    assert_eq!(r.get("a.u8"), Some(&Value::Uint8(7)));
}

#[test]
fn metadata_array_types() {
    let mut b = Vec::new();
    b.extend_from_slice(&GGUF_MAGIC);
    b.extend_from_slice(&3u32.to_le_bytes());
    b.extend_from_slice(&0u64.to_le_bytes());
    b.extend_from_slice(&1u64.to_le_bytes());
    // key
    let key = "arr.u32";
    b.extend_from_slice(&(key.len() as u64).to_le_bytes());
    b.extend_from_slice(key.as_bytes());
    b.extend_from_slice(&9u32.to_le_bytes()); // ARRAY
    b.extend_from_slice(&4u32.to_le_bytes()); // elem type Uint32
    b.extend_from_slice(&3u64.to_le_bytes()); // len
    b.extend_from_slice(&10u32.to_le_bytes());
    b.extend_from_slice(&20u32.to_le_bytes());
    b.extend_from_slice(&30u32.to_le_bytes());

    let r = Reader::parse(&b).unwrap();
    assert_eq!(
        r.get("arr.u32"),
        Some(&Value::Array(ArrayValue::Uint32(vec![10, 20, 30])))
    );
}

#[test]
fn unknown_value_type_is_clean_error() {
    let mut b = Vec::new();
    b.extend_from_slice(&GGUF_MAGIC);
    b.extend_from_slice(&3u32.to_le_bytes());
    b.extend_from_slice(&0u64.to_le_bytes());
    b.extend_from_slice(&1u64.to_le_bytes());
    let key = "x";
    b.extend_from_slice(&(key.len() as u64).to_le_bytes());
    b.extend_from_slice(key.as_bytes());
    b.extend_from_slice(&99u32.to_le_bytes()); // invalid type
    assert!(matches!(
        Reader::parse(&b),
        Err(ReadError::UnknownValueType(99))
    ));
}

#[test]
fn string_longer_than_buffer_is_clean_error() {
    let mut b = Vec::new();
    b.extend_from_slice(&GGUF_MAGIC);
    b.extend_from_slice(&3u32.to_le_bytes());
    b.extend_from_slice(&0u64.to_le_bytes());
    b.extend_from_slice(&1u64.to_le_bytes());
    let key = "x";
    b.extend_from_slice(&(key.len() as u64).to_le_bytes());
    b.extend_from_slice(key.as_bytes());
    // string type whose length claims more than remains
    b.extend_from_slice(&8u32.to_le_bytes());
    b.extend_from_slice(&999999u64.to_le_bytes());
    assert!(matches!(
        Reader::parse(&b),
        Err(ReadError::UnexpectedEof) | Err(ReadError::StringTooLong)
    ));
}

// --- proptest round-trip: arbitrary tensor tables must parse losslessly ---

proptest::proptest! {
    #![proptest_config(proptest::prelude::ProptestConfig::with_cases(256))]

    #[test]
    fn roundtrip_tensor_table(
        names in proptest::collection::vec(
            proptest::char::range('a', 'z'), 0..20
        ),
        raw_type in 0u32..4u32,
    ) {
        let ty = match raw_type {
            0 => GgmlType::F32,
            1 => GgmlType::Q4_0,
            2 => GgmlType::Q4_K,
            _ => GgmlType::F16,
        };
        let tensors: Vec<(String, Vec<u64>, GgmlType)> = names.iter().enumerate().map(|(i, name)| {
            (format!("{name}_{i}"), vec![ (i as u64 % 16) + 1, (i as u64 % 32) + 1 ], ty)
        }).collect();
        let ts: Vec<(&str, &[u64], GgmlType)> = tensors.iter()
            .map(|(n, d, t)| (n.as_str(), d.as_slice(), *t)).collect();
        let w = Writer::new(3, &ts);
        let r = Reader::parse(&w.buf).expect("round-trip parse");
        assert_eq!(r.tensors.len(), tensors.len());
        for (got, want) in r.tensors.iter().zip(&tensors) {
            assert_eq!(&got.name, &want.0);
            assert_eq!(&got.dims, &want.1);
            assert_eq!(got.ggml_type, want.2);
        }
    }
}

// --- integration: parse the real fixture files (if present) ---

#[test]
fn parses_real_qwen2_fixture() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/qwen2-0_5b-instruct-q4_0.gguf"
    );
    let buf = std::fs::read(path);
    let Ok(buf) = buf else {
        eprintln!("fixture not present; skipping");
        return;
    };
    let r = Reader::parse(&buf).expect("parse real qwen2 fixture");
    assert_eq!(r.version, 3);
    assert!(!r.tensors.is_empty());
    // qwen2 0.5B: n_layer 24, head_dim 64
    assert_eq!(r.get_str("general.architecture"), Some("qwen2"));
    assert_eq!(r.get_u32("qwen2.block_count"), Some(24));
}

#[test]
fn parses_real_mmproj_fixture() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/mmproj-SmolVLM-500M-Instruct-Q8_0.gguf"
    );
    let Ok(buf) = std::fs::read(path) else {
        eprintln!("fixture not present; skipping");
        return;
    };
    let r = Reader::parse(&buf).expect("parse real mmproj fixture");
    assert!(!r.tensors.is_empty());
    // mmproj carries a clip/mmproj architecture
    assert!(r.get_str("general.architecture").is_some());
}
