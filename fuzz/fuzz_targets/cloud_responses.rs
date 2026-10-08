//! Cloud provider responses: Anthropic SSE streams and message bodies,
//! OpenAI assistant fields, OpenAI → Anthropic tool conversion.
//!
//! These bytes come from the network (or a proxy / `--base-url` server).
#![no_main]

use libfuzzer_sys::fuzz_target;
use runa_cloud::openai::split_assistant_fields;
use runa_cloud::{parse_message, parse_sse, tools_from_openai};

fuzz_target!(|data: &[u8]| {
    let Ok(text) = std::str::from_utf8(data) else {
        return;
    };
    let _ = parse_sse(text);
    let _ = parse_message(text);
    if let Ok(value) = serde_json::from_str::<serde_json::Value>(text) {
        let _ = split_assistant_fields(&value);
        let _ = tools_from_openai(&value);
    }
});
