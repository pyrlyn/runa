//! Streaming reasoning/answer split (`<think>`, Harmony channels, Gemma
//! thought tags) over arbitrarily chunked model output.
//!
//! Model text is untrusted and tags can be split across `push` chunks at
//! any byte offset that is a char boundary, so the chunking is fuzzed too.
#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use runa_core::{ReasonFamily, ReasonPiece, ReasoningParser, parse_stream};

#[derive(Debug, Arbitrary)]
enum Family {
    Auto,
    XmlThink,
    Harmony,
    Gemma,
    /// Derive the family from a chat template string instead.
    FromTemplate(String),
}

#[derive(Debug, Arbitrary)]
struct Input {
    family: Family,
    chunks: Vec<String>,
    /// Interleave `flush` calls (the engine flushes on stop strings).
    flush_every: u8,
}

fn text_of(pieces: &[ReasonPiece]) -> usize {
    pieces
        .iter()
        .map(|p| match p {
            ReasonPiece::Reasoning(s) | ReasonPiece::Text(s) => s.len(),
        })
        .sum()
}

fuzz_target!(|input: Input| {
    let family = match input.family {
        Family::Auto => ReasonFamily::Auto,
        Family::XmlThink => ReasonFamily::XmlThink,
        Family::Harmony => ReasonFamily::Harmony,
        Family::Gemma => ReasonFamily::Gemma,
        Family::FromTemplate(t) => ReasonFamily::from_template(&t),
    };
    let mut parser = ReasoningParser::new(family);
    let mut emitted = 0usize;
    for (i, chunk) in input.chunks.iter().enumerate() {
        emitted += text_of(&parser.push(chunk));
        let _ = parser.in_reason();
        let _ = parser.close_tag();
        let _ = parser.holding_partial();
        if input.flush_every != 0 && i % usize::from(input.flush_every) == 0 {
            emitted += text_of(&parser.flush());
        }
    }
    emitted += text_of(&parser.finish());
    // Tags are only ever removed, never invented.
    let total: usize = input.chunks.iter().map(String::len).sum();
    assert!(
        emitted <= total,
        "emitted {emitted} bytes from {total} input bytes"
    );

    let refs: Vec<&str> = input.chunks.iter().map(String::as_str).collect();
    let (reasoning, answer) = parse_stream(family, &refs);
    assert!(reasoning.len() + answer.len() <= total);
});
