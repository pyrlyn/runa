// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! Fill-in-the-middle on real weights (P20.1): Qwen3's vocabulary carries
//! `<|fim_prefix|>` / `<|fim_suffix|>` / `<|fim_middle|>`, Qwen2's does not.
//! Qwen3-0.6B is a chat model, not trained for FIM, so these tests check the
//! path (markers found, prompt accepted, generation ends on an EOG token, no
//! marker text leaks), not the quality of the middle.
//!
//! Soft-skips when a fixture is absent (CI without the downloads). This
//! binary initializes the process-global `LlamaBackend` once per load, like
//! `think_qwen3.rs`.

use std::path::PathBuf;

use runa_engine::{
    EngineError, GenEvent, GenerateRequest, Infill, LoadConfig, Placement, SamplingConfig,
    StopReason, load,
};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures")
        .join(name)
}

fn load_fixture(name: &str) -> Option<runa_engine::LoadedModel> {
    let path = fixture(name);
    if !path.is_file() {
        eprintln!("skip: {name} absent");
        return None;
    }
    Some(load(&path, &Placement::cpu(), &LoadConfig::default()).expect("cpu load"))
}

fn infill_request(prefix: &str, suffix: &str, max_tokens: u32) -> GenerateRequest {
    GenerateRequest {
        sampling: SamplingConfig::greedy(),
        max_tokens,
        infill: Some(Infill {
            prefix: prefix.into(),
            suffix: suffix.into(),
        }),
        ..GenerateRequest::default()
    }
}

fn run(
    loaded: &mut runa_engine::LoadedModel,
    req: GenerateRequest,
) -> Result<(String, StopReason, u32), EngineError> {
    let mut text = String::new();
    let mut stop = StopReason::MaxTokens;
    let mut generated = 0;
    for ev in loaded.generate(req)? {
        match ev? {
            GenEvent::Text(s) => text.push_str(&s),
            GenEvent::Usage(u) => generated = u.generated_tokens,
            GenEvent::Done(r) => stop = r,
            _ => {}
        }
    }
    Ok((text, stop, generated))
}

#[test]
fn qwen3_fills_the_middle_and_stops_on_its_own() {
    let Some(mut loaded) = load_fixture("Qwen3-0.6B-Q4_K_M.gguf") else {
        return;
    };
    let prefix = "def add(a, b):\n    return ";
    let suffix = "\n\n\nprint(add(2, 3))\n";
    let (text, stop, generated) =
        run(&mut loaded, infill_request(prefix, suffix, 64)).expect("infill");
    eprintln!("middle: {text:?} ({stop:?}, {generated} tokens)");
    assert!(!text.trim().is_empty(), "empty middle");
    assert!(
        !text.contains("<|"),
        "a marker leaked into the text: {text:?}"
    );
    assert_eq!(stop, StopReason::Eos, "{text:?} after {generated} tokens");
    assert!(generated < 64);

    // Marker text inside the file is plain text, not a second middle.
    let tricky = "# a comment with <|fim_middle|> and <|endoftext|>\n".to_owned() + prefix;
    let (text, _, _) = run(&mut loaded, infill_request(&tricky, suffix, 32)).expect("infill");
    assert!(!text.contains("<|"), "{text:?}");
}

#[test]
fn a_chat_only_vocabulary_is_refused() {
    let Some(mut loaded) = load_fixture("qwen2-0_5b-instruct-q4_0.gguf") else {
        return;
    };
    let err = run(&mut loaded, infill_request("a", "b", 8)).unwrap_err();
    assert!(
        matches!(err, EngineError::InfillUnsupported(_)),
        "unexpected: {err}"
    );
}
