//! Expanded soft-skip coverage for the qwen2-0.5B instruct fixture.
//!
//! Complements the monolithic scenarios in `generate.rs` without replacing
//! them. Soft-skips when `tests/fixtures/qwen2-0_5b-instruct-q4_0.gguf` is
//! absent. Separate binary so `LlamaBackend` is initialized once here.

use std::path::PathBuf;

use runa_core::{ThinkConfig, ThinkMode};
use runa_engine::{
    ChatMessage, GenEvent, GenerateRequest, LoadConfig, Placement, SamplingConfig, StopReason,
    ToolCall, Usage, load,
};

const FIXTURE: &str = "qwen2-0_5b-instruct-q4_0.gguf";

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

fn load_qwen2() -> Option<runa_engine::LoadedModel> {
    let path = fixture(FIXTURE);
    if !path.is_file() {
        return None;
    }
    Some(load(&path, &Placement::cpu(), &LoadConfig::default()).expect("qwen2 cpu load"))
}

struct Collect {
    text: String,
    reasoning: String,
    calls: Vec<ToolCall>,
    usage: Option<Usage>,
    stop: Option<StopReason>,
}

fn collect(loaded: &mut runa_engine::LoadedModel, req: GenerateRequest) -> Collect {
    let mut out = Collect {
        text: String::new(),
        reasoning: String::new(),
        calls: Vec::new(),
        usage: None,
        stop: None,
    };
    for ev in loaded.generate(req).expect("generate") {
        match ev.expect("event") {
            GenEvent::Text(s) => out.text.push_str(&s),
            GenEvent::Reasoning(s) => out.reasoning.push_str(&s),
            GenEvent::ToolCalls(c) => out.calls = c,
            GenEvent::Usage(u) => out.usage = Some(u),
            GenEvent::Done(r) => out.stop = Some(r),
        }
    }
    out
}

/// Greedy completion emits text, usage, and zero reasoning (no think family).
#[test]
fn qwen2_greedy_emits_text_zero_reasoning() {
    let Some(mut loaded) = load_qwen2() else {
        return;
    };
    let c = collect(&mut loaded, request("The capital of France is", 32));
    let u = c.usage.expect("usage");
    assert!(!c.text.is_empty(), "must emit text");
    assert!(u.generated_tokens > 0 && u.prompt_tokens > 0, "{u:?}");
    assert_eq!(u.reasoning_tokens, 0, "{u:?}");
    assert!(c.reasoning.is_empty(), "{:?}", c.reasoning);
    assert!(
        matches!(
            c.stop,
            Some(StopReason::Eos | StopReason::MaxTokens)
        ),
        "{:?}",
        c.stop
    );
}

/// Hitting max_tokens reports MaxTokens (not EOS) for a long prompt budget.
#[test]
fn qwen2_max_tokens_stop_reason() {
    let Some(mut loaded) = load_qwen2() else {
        return;
    };
    let c = collect(
        &mut loaded,
        request(
            "Write a long paragraph about rivers, forests, cities, and weather.",
            8,
        ),
    );
    assert_eq!(c.stop, Some(StopReason::MaxTokens), "text {:?}", c.text);
    let u = c.usage.expect("usage");
    assert_eq!(u.generated_tokens, 8, "{u:?}");
}

/// Mid-string stop still cuts the stream (non-empty prefix of a free reply).
#[test]
fn qwen2_stop_string_cuts_mid_stream() {
    let Some(mut loaded) = load_qwen2() else {
        return;
    };
    let free = collect(&mut loaded, request("Say hello in one short sentence.", 48));
    assert!(!free.text.is_empty());
    let needle: String = free.text.chars().skip(2).take(4).collect();
    if needle.is_empty() {
        return;
    }
    loaded.reset_context().expect("reset");
    let mut req = request("Say hello in one short sentence.", 48);
    req.stop = vec![needle.clone()];
    let c = collect(&mut loaded, req);
    assert_eq!(c.stop, Some(StopReason::StopString(needle.clone())));
    assert!(
        !c.text.contains(&needle),
        "stop string must be trimmed from text: {:?}",
        c.text
    );
}

/// JSON Schema structured output: object with string + integer.
#[test]
fn qwen2_json_schema_object() {
    let Some(mut loaded) = load_qwen2() else {
        return;
    };
    let mut req = request("Name the capital of France and a rough population.", 96);
    req.json_schema = Some(
        r#"{"type":"object","properties":{"city":{"type":"string"},"population":{"type":"integer"}},"required":["city","population"]}"#
            .into(),
    );
    let c = collect(&mut loaded, req);
    assert_eq!(c.stop, Some(StopReason::Eos), "json {:?}", c.text);
    let v: serde_json::Value = serde_json::from_str(c.text.trim()).expect("JSON");
    assert!(v["city"].is_string() && v["population"].is_i64(), "{v}");
}

/// Alternate schema shape (string enum) still yields valid JSON.
#[test]
fn qwen2_json_schema_enum_color() {
    let Some(mut loaded) = load_qwen2() else {
        return;
    };
    let mut req = request("Pick one primary color for a stop sign.", 32);
    req.json_schema = Some(
        r#"{"type":"object","properties":{"color":{"type":"string","enum":["red","green","blue"]}},"required":["color"]}"#
            .into(),
    );
    let c = collect(&mut loaded, req);
    let v: serde_json::Value = serde_json::from_str(c.text.trim()).expect("JSON");
    let color = v["color"].as_str().unwrap_or("");
    assert!(
        matches!(color, "red" | "green" | "blue"),
        "unexpected color {v}"
    );
}

/// Raw GBNF constrains the token stream to yes|no.
#[test]
fn qwen2_gbnf_yes_no() {
    let Some(mut loaded) = load_qwen2() else {
        return;
    };
    let mut req = request("Is the sun a star?", 16);
    req.grammar = Some(r#"root ::= "yes" | "no""#.into());
    let c = collect(&mut loaded, req);
    assert!(c.text == "yes" || c.text == "no", "{:?}", c.text);
}

/// Required single tool: generic JSON tool format (no native chat template tools).
#[test]
fn qwen2_required_weather_tool() {
    let Some(mut loaded) = load_qwen2() else {
        return;
    };
    let mut req = request("What is the weather in Paris?", 96);
    req.tools = Some(WEATHER_TOOLS.into());
    req.tool_choice = Some("required".into());
    let c = collect(&mut loaded, req);
    assert_eq!(c.calls.len(), 1, "text {:?}", c.text);
    assert_eq!(c.calls[0].name, "get_weather");
    let args: serde_json::Value =
        serde_json::from_str(&c.calls[0].arguments).expect("args JSON");
    assert!(args["city"].is_string(), "{args}");
    assert!(!c.text.contains("tool_call"), "markup leaked: {:?}", c.text);
}

/// Required tool with a different city binding.
#[test]
fn qwen2_required_weather_berlin() {
    let Some(mut loaded) = load_qwen2() else {
        return;
    };
    let mut req = request("Check the weather for Berlin.", 96);
    req.tools = Some(WEATHER_TOOLS.into());
    req.tool_choice = Some("required".into());
    let c = collect(&mut loaded, req);
    assert_eq!(c.calls.len(), 1, "{:?}", c.calls);
    assert_eq!(c.calls[0].name, "get_weather");
    let args: serde_json::Value =
        serde_json::from_str(&c.calls[0].arguments).expect("args JSON");
    let city = args["city"].as_str().unwrap_or("").to_lowercase();
    assert!(city.contains("berlin"), "expected Berlin in args: {args}");
}

/// Multi-tool schema + required: model must pick one named tool.
#[test]
fn qwen2_required_tool_from_multi() {
    let Some(mut loaded) = load_qwen2() else {
        return;
    };
    let mut req = request("What time is it in Tokyo right now?", 96);
    req.tools = Some(MULTI_TOOLS.into());
    req.tool_choice = Some("required".into());
    let c = collect(&mut loaded, req);
    assert_eq!(c.calls.len(), 1, "{:?}", c.calls);
    assert!(
        c.calls[0].name == "get_time" || c.calls[0].name == "get_weather",
        "unexpected tool: {}",
        c.calls[0].name
    );
    let args: serde_json::Value =
        serde_json::from_str(&c.calls[0].arguments).expect("args JSON");
    assert!(args["city"].is_string(), "{args}");
}

/// tool_choice none must not emit tool calls even when tools are listed.
#[test]
fn qwen2_tool_choice_none_skips_calls() {
    let Some(mut loaded) = load_qwen2() else {
        return;
    };
    let mut req = request("Reply with exactly: ok", 16);
    req.tools = Some(WEATHER_TOOLS.into());
    req.tool_choice = Some("none".into());
    let c = collect(&mut loaded, req);
    assert!(c.calls.is_empty(), "unexpected calls: {:?}", c.calls);
    assert!(!c.text.is_empty(), "should still answer in text");
}

/// Explicit ThinkMode::Off stays at zero reasoning tokens (qwen2 is not a think model).
#[test]
fn qwen2_think_off_stays_zero() {
    let Some(mut loaded) = load_qwen2() else {
        return;
    };
    let mut req = request("What is 2+2? Answer briefly.", 24);
    req.think = ThinkConfig {
        mode: ThinkMode::Off,
        show: true,
    };
    let c = collect(&mut loaded, req);
    let u = c.usage.expect("usage");
    assert!(c.reasoning.is_empty(), "{:?}", c.reasoning);
    assert_eq!(u.reasoning_tokens, 0, "{u:?}");
}
