// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! Zero-copy, allocation-light reader for the GGUF (GGML unified format)
//! header and metadata.
//!
//! GGUF is the container format `llama.cpp`/ggml uses for quantized models.
//! This module parses just the **header + metadata + tensor-info table**,
//! which is exactly what the fit checker (P1) needs: model architecture,
//! dimensions, quant types and tensor sizes, all before downloading or
//! allocating tensors.
//!
//! The core is `no_std`-friendly: it borrows a `&[u8]` and only allocates
//! `String`/`Vec` for the metadata (via an `alloc`-only design; the crate
//! itself targets `std` but the types avoid any `std`-specific API so a
//! future `#![no_std]` split is a header change away).
//!
//! # Format (ggml `gguf.h`, versions 2 and 3)
//!
//! ```text
//! u32  magic  = "GGUF" (0x47_55_46_46, little-endian)
//! u32  version = 2 | 3
//! u64  tensor_count
//! u64  metadata_kv_count
//! --- metadata KV pairs (metadata_kv_count) ---
//!   string              key            (u64 len + utf-8 bytes)
//!   u32                 value_type     (see `DataType`)
//!   <value>            (see `DataType` for widths)
//! --- tensor infos (tensor_count) ---
//!   string              name
//!   u32                 n_dims
//!   u32[n_dims]         dims
//!   u32                 ggml_type      (see `GgmlType`)
//!   u64                 offset
//! --- padding to 32-byte alignment ---
//! ```
//!
//! v3 changed metadata alignment to 8 bytes (was 4 in v2); both are handled.
//! Key and type strings are never shorter than described — every read is
//! bounds-checked and returns `ReadError` on truncation, so a malicious or
//! corrupted file cannot panic.

use core::fmt;

/// The four magic bytes at the start of every GGUF file.
pub const GGUF_MAGIC: [u8; 4] = *b"GGUF";

/// GGUF value types (ggml `gguf_type`). Values map to the `Value` enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
#[allow(non_camel_case_types)]
pub enum DataType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

impl DataType {
    /// Parse a raw u32 tag into a `DataType`.
    pub fn from_raw(raw: u32) -> Option<DataType> {
        use DataType::*;
        Some(match raw {
            0 => Uint8,
            1 => Int8,
            2 => Uint16,
            3 => Int16,
            4 => Uint32,
            5 => Int32,
            6 => Float32,
            7 => Bool,
            8 => String,
            9 => Array,
            10 => Uint64,
            11 => Int64,
            12 => Float64,
            _ => return None,
        })
    }
}

/// An owned metadata value, fully decoded.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(ArrayValue),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

/// An array value (variable-length, homogeneous).
#[derive(Debug, Clone, PartialEq)]
pub enum ArrayValue {
    Uint8(Vec<u8>),
    Int8(Vec<i8>),
    Uint16(Vec<u16>),
    Int16(Vec<i16>),
    Uint32(Vec<u32>),
    Int32(Vec<i32>),
    Float32(Vec<f32>),
    Bool(Vec<bool>),
    String(Vec<String>),
    Uint64(Vec<u64>),
    Int64(Vec<i64>),
    Float64(Vec<f64>),
}

/// ggml tensor type tag (what a tensor's data is stored as on disk).
///
/// Only the tag is captured here; block sizes and element counts per type
/// are computed in the estimator (P1.3) which needs the full table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum GgmlType {
    F32,
    F16,
    BF16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2_K,
    Q3_K,
    Q4_K,
    Q5_K,
    Q6_K,
    Q8_K,
    IQ2_XXS,
    IQ2_XS,
    IQ3_XXS,
    IQ3_S,
    IQ2_S,
    IQ4_XS,
    IQ1_S,
    IQ4_NL,
    IQ3_XS,
    IQ1_M,
    MXFP4,
    TQ1_0,
    TQ2_0,
    F64,
    I8,
    I16,
    I32,
    I64,
    /// A type outside the fixed ggml enum (future ggml types).
    Unknown(u32),
}

impl GgmlType {
    pub fn from_raw(raw: u32) -> GgmlType {
        use GgmlType::*;
        match raw {
            0 => F32,
            1 => F16,
            30 => BF16,
            2 => Q4_0,
            3 => Q4_1,
            6 => Q5_0,
            7 => Q5_1,
            8 => Q8_0,
            9 => Q8_1,
            10 => Q2_K,
            11 => Q3_K,
            12 => Q4_K,
            13 => Q5_K,
            14 => Q6_K,
            15 => Q8_K,
            16 => IQ2_XXS,
            17 => IQ2_XS,
            18 => IQ3_XXS,
            19 => IQ3_S,
            20 => IQ2_S,
            21 => IQ4_XS,
            22 => IQ1_S,
            23 => IQ4_NL,
            24 => IQ3_XS,
            25 => IQ1_M,
            26 => MXFP4,
            27 => TQ1_0,
            28 => TQ2_0,
            29 => F64,
            31 => I8,
            32 => I16,
            33 => I32,
            34 => I64,
            other => Unknown(other),
        }
    }
}

/// A single tensor's info entry (name, dims, type, byte offset).
#[derive(Debug, Clone, PartialEq)]
pub struct TensorInfo {
    pub name: String,
    pub dims: Vec<u64>,
    pub ggml_type: GgmlType,
    /// Byte offset of the tensor data in the file (incl. any header padding).
    pub offset: u64,
}

/// Errors that can occur while reading a GGUF header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReadError {
    /// Fewer than 4 bytes available (empty/truncated file).
    Truncated,
    /// The magic bytes were not `GGUF`.
    BadMagic,
    /// Unsupported or unknown GGUF version.
    UnsupportedVersion(u32),
    /// A string length field pointed past the end of the buffer.
    StringTooLong,
    /// Ran out of bytes while reading a field/array value.
    UnexpectedEof,
    /// A value-type tag did not map to a known `DataType`.
    UnknownValueType(u32),
    /// A tensor's dimension count or metadata key count overflows a `usize`.
    CountOverflow,
    /// A declared tensor-offset table did not fit in the provided buffer.
    TensorTableTruncated,
}

impl fmt::Display for ReadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ReadError::Truncated => write!(f, "not enough bytes for a GGUF header"),
            ReadError::BadMagic => write!(f, "bad magic (not a GGUF file)"),
            ReadError::UnsupportedVersion(v) => write!(f, "unsupported GGUF version {v}"),
            ReadError::StringTooLong => write!(f, "string length exceeds buffer"),
            ReadError::UnexpectedEof => write!(f, "unexpected end of file"),
            ReadError::UnknownValueType(t) => write!(f, "unknown value type {t}"),
            ReadError::CountOverflow => write!(f, "count does not fit in usize"),
            ReadError::TensorTableTruncated => write!(f, "tensor-info table is truncated"),
        }
    }
}

impl core::error::Error for ReadError {}

/// A parsed GGUF header: metadata KV pairs plus the tensor-info table.
///
/// The reader does **not** copy tensor payload data; it only records each
/// tensor's name/dims/type/offset so callers can seek to tensors afterwards.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Reader {
    pub version: u32,
    pub tensor_count: u64,
    /// Metadata in declaration order.
    pub metadata: Vec<(String, Value)>,
    /// Tensor infos in declaration order.
    pub tensors: Vec<TensorInfo>,
    /// Byte offset just past the tensor table (after 32-byte alignment);
    /// this is where the first tensor's data begins.
    pub data_start: u64,
}

impl Reader {
    /// Parse a GGUF header from a buffer.
    ///
    /// The buffer need only extend through the tensor-info table; tensor
    /// payload may be absent (this is the pre-download path). Returns
    /// `ReadError` rather than panicking on any malformed input.
    pub fn parse(buf: &[u8]) -> Result<Reader, ReadError> {
        let mut r = Cursor {
            buf,
            pos: 0,
            last_u32: 0,
        };

        if r.buf.len() < 4 {
            return Err(ReadError::Truncated);
        }
        if r.buf[..4] != GGUF_MAGIC {
            return Err(ReadError::BadMagic);
        }
        r.pos = 4;

        let version = r.u32()?;
        if version != 2 && version != 3 {
            return Err(ReadError::UnsupportedVersion(version));
        }

        let tensor_count = r.u64()?;
        let kv_count = r.u64()?;

        let mut out = Reader {
            version,
            tensor_count,
            metadata: Vec::with_capacity(kv_count.min(1 << 16) as usize),
            tensors: Vec::with_capacity(tensor_count.min(1 << 16) as usize),
            data_start: 0,
        };

        for _ in 0..kv_count {
            let key = r.string()?;
            let ty = DataType::from_raw(r.u32()?).ok_or(ReadError::UnknownValueType(r.last_u32))?;
            let value = r.value(ty)?;
            out.metadata.push((key, value));
        }

        for _ in 0..tensor_count {
            let name = r.string()?;
            let n_dims = r.u32()?;
            let n_dims_usize = usize::try_from(n_dims).map_err(|_| ReadError::CountOverflow)?;
            let mut dims = Vec::with_capacity(n_dims_usize);
            for _ in 0..n_dims {
                dims.push(r.u64()?);
            }
            let ggml_type = GgmlType::from_raw(r.u32()?);
            let offset = r.u64()?;
            out.tensors.push(TensorInfo {
                name,
                dims,
                ggml_type,
                offset,
            });
        }

        // The first tensor's data begins after padding the header/table up to
        // a 64-byte alignment (ggml `GGUF_ALIGNMENT` = 32; total header is
        // padded to that boundary). Compute it from the reader position.
        const ALIGN: u64 = 32;
        let pos = r.pos as u64;
        out.data_start = pos.div_ceil(ALIGN) * ALIGN;

        Ok(out)
    }

    /// Look up a metadata value by key. Returns `None` if absent.
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.metadata.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    /// Common scalar accessors used by the estimator/planner.
    pub fn get_str(&self, key: &str) -> Option<&str> {
        match self.get(key) {
            Some(Value::String(s)) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn get_u32(&self, key: &str) -> Option<u32> {
        match self.get(key) {
            Some(Value::Uint32(v)) => Some(*v),
            Some(Value::Int32(v)) if *v >= 0 => Some(*v as u32),
            _ => None,
        }
    }

    pub fn get_u64(&self, key: &str) -> Option<u64> {
        match self.get(key) {
            Some(Value::Uint64(v)) => Some(*v),
            Some(Value::Uint32(v)) => Some(*v as u64),
            Some(Value::Int64(v)) if *v >= 0 => Some(*v as u64),
            Some(Value::Int32(v)) if *v >= 0 => Some(*v as u64),
            _ => None,
        }
    }

    pub fn get_f32(&self, key: &str) -> Option<f32> {
        match self.get(key) {
            Some(Value::Float32(v)) => Some(*v),
            _ => None,
        }
    }

    pub fn get_bool(&self, key: &str) -> Option<bool> {
        match self.get(key) {
            Some(Value::Bool(v)) => Some(*v),
            _ => None,
        }
    }
}

/// Bounds-checked little-endian cursor over a byte slice.
///
/// Every read returns `ReadError::UnexpectedEof` on truncation; `last_u32`
/// holds the most recent u32 tag so callers can report unknown value types.
struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
    last_u32: u32,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], ReadError> {
        let end = self.pos.checked_add(n).ok_or(ReadError::UnexpectedEof)?;
        if end > self.buf.len() {
            return Err(ReadError::UnexpectedEof);
        }
        let s = &self.buf[self.pos..end];
        self.pos = end;
        Ok(s)
    }

    fn u32(&mut self) -> Result<u32, ReadError> {
        let b = self.take(4)?;
        let v = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
        self.last_u32 = v;
        Ok(v)
    }

    fn u64(&mut self) -> Result<u64, ReadError> {
        let b = self.take(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    fn string(&mut self) -> Result<String, ReadError> {
        let len = self.u64()? as usize;
        let bytes = self.take(len)?;
        // GGUF keys/types are ASCII/UTF-8; lossy fallback only for truly
        // malformed inputs, never for well-formed files.
        Ok(String::from_utf8_lossy(bytes).into_owned())
    }

    fn value(&mut self, ty: DataType) -> Result<Value, ReadError> {
        use DataType::*;
        Ok(match ty {
            Uint8 => Value::Uint8(self.take(1)?[0]),
            Int8 => Value::Int8(self.take(1)?[0] as i8),
            Uint16 => Value::Uint16(self.e16()?),
            Int16 => Value::Int16(self.e16()? as i16),
            Uint32 => Value::Uint32(self.u32()?),
            Int32 => Value::Int32(self.u32()? as i32),
            Float32 => Value::Float32(f32::from_bits(self.u32()?)),
            Bool => Value::Bool(self.take(1)?[0] != 0),
            String => Value::String(self.string()?),
            Array => Value::Array(self.array()?),
            Uint64 => Value::Uint64(self.u64()?),
            Int64 => Value::Int64(self.u64()? as i64),
            Float64 => Value::Float64(f64::from_bits(self.u64()?)),
        })
    }

    fn e16(&mut self) -> Result<u16, ReadError> {
        let b = self.take(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    fn array(&mut self) -> Result<ArrayValue, ReadError> {
        use ArrayValue as A;
        let ety =
            DataType::from_raw(self.u32()?).ok_or(ReadError::UnknownValueType(self.last_u32))?;
        let n = self.u64()?;
        let n = usize::try_from(n).map_err(|_| ReadError::CountOverflow)?;

        Ok(match ety {
            DataType::Uint8 => A::Uint8(self.take(n)?.to_vec()),
            DataType::Int8 => {
                let b = self.take(n)?;
                A::Int8(b.iter().map(|&x| x as i8).collect())
            }
            DataType::Uint16 => A::Uint16((0..n).map(|_| self.e16()).collect::<Result<_, _>>()?),
            DataType::Int16 => A::Int16(
                (0..n)
                    .map(|_| self.e16().map(|v| v as i16))
                    .collect::<Result<_, _>>()?,
            ),
            DataType::Uint32 => A::Uint32((0..n).map(|_| self.u32()).collect::<Result<_, _>>()?),
            DataType::Int32 => A::Int32(
                (0..n)
                    .map(|_| self.u32().map(|v| v as i32))
                    .collect::<Result<_, _>>()?,
            ),
            DataType::Float32 => A::Float32(
                (0..n)
                    .map(|_| self.u32().map(f32::from_bits))
                    .collect::<Result<_, _>>()?,
            ),
            DataType::Bool => A::Bool(self.take(n)?.iter().map(|&x| x != 0).collect()),
            DataType::String => A::String((0..n).map(|_| self.string()).collect::<Result<_, _>>()?),
            DataType::Uint64 => A::Uint64((0..n).map(|_| self.u64()).collect::<Result<_, _>>()?),
            DataType::Int64 => A::Int64(
                (0..n)
                    .map(|_| self.u64().map(|v| v as i64))
                    .collect::<Result<_, _>>()?,
            ),
            DataType::Float64 => A::Float64(
                (0..n)
                    .map(|_| self.u64().map(f64::from_bits))
                    .collect::<Result<_, _>>()?,
            ),
            DataType::Array => {
                // Nested arrays are not emitted by ggml; treat as error via Eof.
                return Err(ReadError::UnexpectedEof);
            }
        })
    }
}
