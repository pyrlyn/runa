//! Qwen3-0.6B thinking + tool-calling coverage (replaces the 8B fixture).
//!
//! Soft-skips when `tests/fixtures/Qwen3-0.6B-Q4_K_M.gguf` is absent (CI
//! without the ~0.5 GiB download). Each test loads its own model: this
//! binary is separate from `generate.rs`, so the process-global
//! `LlamaBackend` is initialized once here.

use std::path::PathBuf;

use runa_core::{ThinkConfig, ThinkMode};
use runa_engine::{
    ChatMessage, GenEvent, GenerateRequest, LoadConfig, Placement, SamplingConfig, ToolCall, Usage,
    load,
};

const FIXTURE: &str = "Qwen3-0.6B-Q4_K_M.gguf";

const WEATHER_TOOLS: &str = r#"[{"type":"function","function":{"name":"get_weather","description":"Current weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]"#;

const MULTI_TOOLS: &str = r#"[{"type":"function","function":{"name":"get_weather","description":"Current weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}},{"type":"function","function":{"name":"get_time","description":"Local time","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]"#;

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
        think: ThinkConfig::default(),
        audio_pcm: None,
        images: Vec::new(),
        speculative: runa_engine::Speculative::default(),
        json_schema: None,
        grammar: None,
        tools: None,
        tool_choice: None,
    }
}

fn load_qwen3() -> Option<runa_engine::LoadedModel> {
    let path = fixture(FIXTURE);
    if !path.is_file() {
        return None;
    }
    Some(load(&path, &Placement::cpu(), &LoadConfig::default()).expect("qwen3-0.6B cpu load"))
}

struct Collect {
    text: String,
    reasoning: String,
    calls: Vec<ToolCall>,
    usage: Option<Usage>,
}

fn collect(loaded: &mut runa_engine::LoadedModel, req: GenerateRequest) -> Collect {
    let mut out = Collect {
        text: String::new(),
        reasoning: String::new(),
        calls: Vec::new(),
        usage: None,
    };
    for ev in loaded.generate(req).expect("generate") {
        match ev.expect("event") {
            GenEvent::Text(s) => out.text.push_str(&s),
            GenEvent::Reasoning(s) => out.reasoning.push_str(&s),
            GenEvent::ToolCalls(c) => out.calls = c,
            GenEvent::Usage(u) => out.usage = Some(u),
            _ => {}
        }
    }
    out
}

fn budget(tokens: u32, grace: u32) -> ThinkConfig {
    ThinkConfig {
        mode: ThinkMode::Budget { tokens, grace },
        show: true,
    }
}

/// P10.4 / M6: budgeted thinking emits reasoning and reports a count
/// within budget+grace.
#[test]
fn think_budget_reports_reasoning_tokens() {
    let Some(mut loaded) = load_qwen3() else {
        return;
    };
    let mut req = request("What is 2+2? Answer briefly.", 48);
    req.think = budget(32, 8);
    let c = collect(&mut loaded, req);
    let u = c.usage.expect("usage event");
    assert!(
        !c.reasoning.is_empty(),
        "Qwen3 thinks out loud under a budget"
    );
    assert!(u.reasoning_tokens > 0, "{u:?}");
    assert!(
        u.reasoning_tokens <= u.generated_tokens,
        "reasoning is a subset of generated: {u:?}"
    );
    assert!(
        u.reasoning_tokens <= 32 + 8,
        "reported reasoning honors budget+grace (M6): {u:?}"
    );
}

/// P10.11: required tool call still completes under a thinking budget.
#[test]
fn think_budget_required_weather_tool() {
    let Some(mut loaded) = load_qwen3() else {
        return;
    };
    let mut req = request("What is the weather in Paris?", 96);
    req.think = budget(64, 8);
    req.tools = Some(WEATHER_TOOLS.into());
    req.tool_choice = Some("required".into());
    let c = collect(&mut loaded, req);
    assert_eq!(c.calls.len(), 1, "one forced call, text {:?}", c.text);
    assert_eq!(c.calls[0].name, "get_weather");
    let args: serde_json::Value = serde_json::from_str(&c.calls[0].arguments).expect("args JSON");
    assert!(args["city"].is_string(), "{args}");
    assert!(!c.text.contains("tool_call"), "markup leaked: {:?}", c.text);
    let u = c.usage.expect("usage");
    assert!(
        u.reasoning_tokens <= 64 + 8,
        "budget holds with tools (M6): {u:?}"
    );
}

/// Multi-tool schema + required: model must pick one named tool.
#[test]
fn think_budget_required_tool_from_multi() {
    let Some(mut loaded) = load_qwen3() else {
        return;
    };
    let mut req = request("What time is it in Tokyo right now?", 96);
    req.think = budget(48, 8);
    req.tools = Some(MULTI_TOOLS.into());
    req.tool_choice = Some("required".into());
    let c = collect(&mut loaded, req);
    assert_eq!(
        c.calls.len(),
        1,
        "one call from multi schema: {:?}",
        c.calls
    );
    assert!(
        c.calls[0].name == "get_time" || c.calls[0].name == "get_weather",
        "unexpected tool: {}",
        c.calls[0].name
    );
    let args: serde_json::Value = serde_json::from_str(&c.calls[0].arguments).expect("args JSON");
    assert!(args["city"].is_string(), "{args}");
    let u = c.usage.expect("usage");
    assert!(u.reasoning_tokens <= 48 + 8, "budget holds: {u:?}");
}

/// Alternate required tool prompt (city other than Paris).
#[test]
fn think_budget_required_weather_berlin() {
    let Some(mut loaded) = load_qwen3() else {
        return;
    };
    let mut req = request("Check the weather for Berlin.", 96);
    req.think = budget(64, 8);
    req.tools = Some(WEATHER_TOOLS.into());
    req.tool_choice = Some("required".into());
    let c = collect(&mut loaded, req);
    assert_eq!(c.calls.len(), 1, "{:?}", c.calls);
    assert_eq!(c.calls[0].name, "get_weather");
    let args: serde_json::Value = serde_json::from_str(&c.calls[0].arguments).expect("args JSON");
    let city = args["city"].as_str().unwrap_or("").to_lowercase();
    assert!(city.contains("berlin"), "expected Berlin in args: {args}");
}

/// Edge: ThinkMode::Off must not surface reasoning. Plain `render_prompt`
/// does not pass `enable_thinking` kwargs, so we go through oaicompat
/// (tools present) where Off sets `enable_thinking: false`.
#[test]
fn think_off_emits_no_reasoning() {
    let Some(mut loaded) = load_qwen3() else {
        return;
    };
    let mut req = request("What is 2+2? Answer with just the number.", 32);
    req.think = ThinkConfig {
        mode: ThinkMode::Off,
        show: true,
    };
    // Force `render_oaicompat` so template kwargs apply.
    req.tools = Some(WEATHER_TOOLS.into());
    req.tool_choice = Some("none".into());
    let c = collect(&mut loaded, req);
    let u = c.usage.expect("usage");
    assert!(
        c.reasoning.is_empty(),
        "Off must suppress reasoning stream: {:?}",
        c.reasoning
    );
    assert_eq!(u.reasoning_tokens, 0, "{u:?}");
}

/// Edge: a tight budget must still cap reported reasoning at budget+grace
/// (the model may want to think longer — we refuse to exceed).
#[test]
fn think_tight_budget_caps_reasoning() {
    let Some(mut loaded) = load_qwen3() else {
        return;
    };
    let tokens = 8u32;
    let grace = 4u32;
    let mut req = request(
        "Write a careful step-by-step solution to 17*19, then give the product.",
        128,
    );
    req.think = budget(tokens, grace);
    let c = collect(&mut loaded, req);
    let u = c.usage.expect("usage");
    // Prefer non-empty when the model cooperates; the hard M6 contract is the cap.
    assert!(
        u.reasoning_tokens <= tokens + grace,
        "must not exceed budget+grace: {u:?}"
    );
    assert!(
        u.reasoning_tokens <= u.generated_tokens,
        "reasoning ⊆ generated: {u:?}"
    );
}

/// Different budget sizes both honor their own caps.
#[test]
fn think_budgets_32_and_16_both_cap() {
    let Some(mut loaded) = load_qwen3() else {
        return;
    };
    for (tokens, grace) in [(32u32, 8u32), (16u32, 4u32)] {
        loaded.reset_context().expect("reset");
        let mut req = request("Explain briefly why the sky is blue.", 64);
        req.think = budget(tokens, grace);
        let c = collect(&mut loaded, req);
        let u = c.usage.expect("usage");
        assert!(
            u.reasoning_tokens <= tokens + grace,
            "budget {tokens}+{grace} exceeded: {u:?}"
        );
    }
}

/// ThinkMode::On (no numeric budget) still emits reasoning on Qwen3.
#[test]
fn think_on_emits_reasoning() {
    let Some(mut loaded) = load_qwen3() else {
        return;
    };
    let mut req = request("What is 3+5? Answer briefly.", 48);
    req.think = ThinkConfig {
        mode: ThinkMode::On,
        show: true,
    };
    let c = collect(&mut loaded, req);
    let u = c.usage.expect("usage");
    assert!(!c.reasoning.is_empty(), "On should think out loud");
    assert!(u.reasoning_tokens > 0, "{u:?}");
}
