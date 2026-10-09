// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! P2.2: templated streaming generation with sampler chain, stop strings,
//! EOS/max-token termination and usage counters.
//!
//! CPU load of the qwen2 fixture; greedy decoding is deterministic, so the
//! same prompt must produce byte-identical text across context resets.
//!
//! NOTE: one test only — `LlamaBackend::init` is process-global in
//! llama-cpp-2 0.1.133 (second init fails with `BackendAlreadyInitialized`),
//! so all scenarios share a single load. Multi-model serving needs a shared
//! backend (P6.1).

use std::path::PathBuf;

use runa_engine::{
    ChatMessage, GenEvent, GenerateRequest, LoadConfig, Placement, PromptCache, SamplingConfig,
    StopReason, load,
};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures")
        .join(name)
}

fn request(prompt: &str, max_tokens: u32) -> GenerateRequest {
    GenerateRequest {
        messages: vec![ChatMessage::user(prompt)],
        sampling: SamplingConfig::greedy(),
        max_tokens,
        stop: Vec::new(),
        add_generation_prompt: true,
        think: runa_core::ThinkConfig::default(),
        audio_pcm: None,
        images: Vec::new(),
        speculative: runa_engine::Speculative::default(),
        json_schema: None,
        grammar: None,
        tools: None,
        tool_choice: None,
    }
}

#[test]
fn greedy_stream_deterministic_usage_and_stop() {
    let path = fixture("qwen2-0_5b-instruct-q4_0.gguf");
    assert!(path.is_file());
    let mut loaded = load(&path, &Placement::cpu(), &LoadConfig::default()).expect("cpu load");

    // 1. Greedy text + usage counters.
    let (text1, usage1, reason1) = loaded
        .generate(request("The capital of France is", 64))
        .expect("generate")
        .collect_text()
        .expect("collect");
    assert!(!text1.is_empty(), "greedy must emit text");
    assert!(
        matches!(reason1, StopReason::Eos | StopReason::MaxTokens),
        "{reason1:?}"
    );
    assert!(usage1.generated_tokens > 0);
    assert!(usage1.prompt_tokens > 0);
    assert!(usage1.pp_toks_per_s > 0.0 && usage1.tg_toks_per_s > 0.0);
    // P10.4: no think tags on qwen2 + ThinkMode::Off → zero reasoning.
    assert_eq!(usage1.reasoning_tokens, 0);

    // 2. Same prompt on a reset context: byte-identical stream.
    loaded.reset_context().expect("reset");
    let (text2, _, reason2) = loaded
        .generate(request("The capital of France is", 64))
        .expect("generate again")
        .collect_text()
        .expect("collect");
    assert_eq!(reason2, reason1);
    assert_eq!(text1, text2, "greedy decoding must be deterministic");

    // 3. Stop on a text prefix: stream ends immediately, cut.
    // (char-boundary safe; matches at offset 0 whatever the length).
    let stop: String = text1.chars().take(5).collect();
    assert!(!stop.is_empty());
    loaded.reset_context().expect("reset");
    let mut req = request("The capital of France is", 64);
    req.stop = vec![stop.clone()];
    let (cut, _, reason3) = loaded
        .generate(req)
        .expect("generate with stop")
        .collect_text()
        .expect("collect");
    assert_eq!(reason3, StopReason::StopString(stop));
    assert!(cut.is_empty(), "stop at offset 0 emits nothing: {cut:?}");

    // 4. P2.8 LMDB prompt cache: second identical prompt restores KV and
    // matches greedy text. Same test fn — LlamaBackend is process-global.
    let cache_dir = tempfile::tempdir().expect("prompt-cache dir");
    loaded.attach_prompt_cache(
        PromptCache::open_with_map_size(cache_dir.path(), 256 * 1024 * 1024).expect("open lmdb"),
    );
    loaded.reset_context().expect("reset before cache miss");
    let (miss_text, _, _) = loaded
        .generate(request("The capital of France is", 64))
        .expect("cache-miss generate")
        .collect_text()
        .expect("collect miss");
    assert!(!loaded.last_prompt_cache_hit(), "first cached run stores");
    assert_eq!(miss_text, text1, "store path must stay greedy-identical");
    loaded.on_idle();
    loaded.reset_context().expect("reset before cache hit");
    let (hit_text, _, _) = loaded
        .generate(request("The capital of France is", 64))
        .expect("cache-hit generate")
        .collect_text()
        .expect("collect hit");
    assert!(
        loaded.last_prompt_cache_hit(),
        "second run restores from LMDB"
    );
    assert_eq!(hit_text, text1, "restored KV must decode identically");

    // 5. P5.6: n-gram speculation at temperature 0 matches greedy text.
    loaded.reset_context().expect("reset before ngram");
    let mut ngram_req = request("The capital of France is", 64);
    ngram_req.speculative.ngram = true;
    let (ngram_text, _, ngram_reason) = loaded
        .generate(ngram_req)
        .expect("ngram generate")
        .collect_text()
        .expect("collect ngram");
    assert_eq!(ngram_reason, reason1);
    assert_eq!(
        ngram_text, text1,
        "ngram greedy must match non-ngram greedy"
    );

    // 6. P8.1: a JSON Schema constrains the answer to matching JSON.
    loaded.reset_context().expect("reset before schema");
    let mut schema_req = request("Name the capital of France and its population.", 96);
    schema_req.json_schema = Some(
        r#"{"type":"object","properties":{"city":{"type":"string"},"population":{"type":"integer"}},"required":["city","population"]}"#
            .into(),
    );
    let (json, _, json_reason) = loaded
        .generate(schema_req)
        .expect("schema generate")
        .collect_text()
        .expect("collect schema");
    assert_eq!(json_reason, StopReason::Eos, "grammar completes: {json:?}");
    let v: serde_json::Value = serde_json::from_str(json.trim()).expect("answer is JSON");
    assert!(v["city"].is_string() && v["population"].is_i64(), "{v}");

    // 7. P8.1: a raw GBNF grammar.
    loaded.reset_context().expect("reset before grammar");
    let mut gbnf_req = request("Is Paris in France?", 16);
    gbnf_req.grammar = Some(r#"root ::= "yes" | "no""#.into());
    let (answer, _, _) = loaded
        .generate(gbnf_req)
        .expect("grammar generate")
        .collect_text()
        .expect("collect grammar");
    assert!(answer == "yes" || answer == "no", "{answer:?}");

    // 8. P8.2: `tool_choice: required` yields one parsed call and no
    // markup in the text (qwen2 has no tool template: llama.cpp's generic
    // JSON format).
    loaded.reset_context().expect("reset before tools");
    let mut tool_req = request("What is the weather in Paris?", 96);
    tool_req.tools = Some(
        r#"[{"type":"function","function":{"name":"get_weather","description":"Current weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]"#
            .into(),
    );
    tool_req.tool_choice = Some("required".into());
    let mut calls = Vec::new();
    let mut text = String::new();
    for ev in loaded.generate(tool_req).expect("tool generate") {
        match ev.expect("tool event") {
            runa_engine::GenEvent::ToolCalls(c) => calls = c,
            runa_engine::GenEvent::Text(t) => text.push_str(&t),
            _ => {}
        }
    }
    assert_eq!(calls.len(), 1, "one call, text {text:?}");
    assert_eq!(calls[0].name, "get_weather");
    assert!(!calls[0].id.is_empty());
    let args: serde_json::Value = serde_json::from_str(&calls[0].arguments).expect("args JSON");
    assert!(args["city"].is_string(), "{args}");
    assert!(!text.contains("tool_call"), "markup leaked: {text:?}");
}

#[test]
fn bench_exact_prompt_and_gen_counts() {
    let path = fixture("qwen2-0_5b-instruct-q4_0.gguf");
    assert!(path.is_file());
    let cfg = LoadConfig {
        n_ctx: 256,
        n_batch: 8,
        n_ubatch: 8,
        ..LoadConfig::default()
    };
    let mut loaded = load(&path, &Placement::cpu(), &cfg).expect("cpu load");
    let usage = loaded.bench(8, 4).expect("bench");
    assert_eq!(usage.prompt_tokens, 8);
    assert_eq!(usage.generated_tokens, 4);
    assert!(usage.pp_toks_per_s > 0.0, "pp {}", usage.pp_toks_per_s);
    assert!(usage.tg_toks_per_s > 0.0, "tg {}", usage.tg_toks_per_s);
}
