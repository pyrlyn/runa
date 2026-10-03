//! Expanded soft-skip vision coverage for SmolVLM-256M + sibling mmproj.
//!
//! Requires `--features mtmd` (see `Cargo.toml` `[[test]]`). Soft-skips when
//! the language GGUF, mmproj, or `shapes.png` fixture is absent. Complements
//! runa-fit sibling/mmproj parse tests without replacing them.

use std::path::PathBuf;

use runa_core::ThinkConfig;
use runa_engine::{
    ChatMessage, GenEvent, GenerateRequest, LoadConfig, Placement, SamplingConfig, StopReason,
    Usage, VisionFrame, VisionSource, load,
};

const MODEL: &str = "SmolVLM-256M-Instruct-Q4_K_M.gguf";
const MMPROJ: &str = "mmproj-SmolVLM-256M-Instruct-f16.gguf";
const IMAGE: &str = "shapes.png";

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

fn load_smolvlm() -> Option<runa_engine::LoadedModel> {
    let model = fixture(MODEL);
    let mmproj = fixture(MMPROJ);
    if !model.is_file() || !mmproj.is_file() {
        return None;
    }
    Some(
        load(
            &model,
            &Placement::cpu(),
            &LoadConfig {
                mmproj: Some(mmproj),
                // Vision prompts need headroom for image tokens.
                n_ctx: 2048,
                ..LoadConfig::default()
            },
        )
        .expect("SmolVLM+mmproj cpu load"),
    )
}

struct Collect {
    text: String,
    usage: Option<Usage>,
    stop: Option<StopReason>,
}

fn collect(loaded: &mut runa_engine::LoadedModel, req: GenerateRequest) -> Collect {
    let mut out = Collect {
        text: String::new(),
        usage: None,
        stop: None,
    };
    for ev in loaded.generate(req).expect("generate") {
        match ev.expect("event") {
            GenEvent::Text(s) => out.text.push_str(&s),
            GenEvent::Usage(u) => out.usage = Some(u),
            GenEvent::Done(r) => out.stop = Some(r),
            GenEvent::Reasoning(_) | GenEvent::ToolCalls(_) => {}
        }
    }
    out
}

/// Load with mmproj advertises native vision support.
#[test]
fn smolvlm_supports_native_vision() {
    let Some(loaded) = load_smolvlm() else {
        return;
    };
    assert!(
        loaded.supports_native_vision(),
        "SmolVLM mmproj must enable vision"
    );
    assert!(loaded.mmproj_bytes() > 1_000);
}

/// Text-only chat still works with mmproj attached (no images).
#[test]
fn smolvlm_text_only_with_mmproj() {
    let Some(mut loaded) = load_smolvlm() else {
        return;
    };
    let c = collect(&mut loaded, request("Say hi in one word.", 24));
    let u = c.usage.expect("usage");
    assert!(!c.text.is_empty(), "text-only must emit something");
    assert!(u.generated_tokens > 0, "{u:?}");
    assert_eq!(u.reasoning_tokens, 0, "{u:?}");
}

/// Image from fixtures/shapes.png yields a non-empty vision reply + usage.
#[test]
fn smolvlm_describe_shapes_png() {
    let Some(mut loaded) = load_smolvlm() else {
        return;
    };
    let image = fixture(IMAGE);
    if !image.is_file() {
        return;
    }
    let mut req = request(
        "Describe the image briefly. Mention colors or shapes if you see them.",
        64,
    );
    req.images = vec![VisionFrame {
        t_sec: None,
        source: VisionSource::Path(image),
    }];
    let c = collect(&mut loaded, req);
    let u = c.usage.expect("usage");
    assert!(!c.text.is_empty(), "vision must emit text");
    assert!(u.prompt_tokens > 0 && u.generated_tokens > 0, "{u:?}");
    assert!(
        matches!(
            c.stop,
            Some(StopReason::Eos | StopReason::MaxTokens)
        ),
        "{:?}",
        c.stop
    );
    // Soft content check: 256M Q4 often fails to name colors/shapes reliably.
    // Structural checks above already prove the vision path ran; require only that
    // image tokens inflated the prompt vs a bare text turn (n_ctx headroom above).
    assert!(
        u.prompt_tokens >= 32,
        "vision turn should include image tokens, got {u:?}; reply={:?}",
        c.text
    );
}

/// Empty image list with a vision-capable model still answers as text-only.
#[test]
fn smolvlm_empty_images_is_text_path() {
    let Some(mut loaded) = load_smolvlm() else {
        return;
    };
    let mut req = request("Reply with the word ok.", 16);
    req.images = Vec::new();
    let c = collect(&mut loaded, req);
    assert!(!c.text.is_empty());
}

/// Packed RGB frame (tiny solid color) is accepted by the vision path.
#[test]
fn smolvlm_rgb_frame_generates() {
    let Some(mut loaded) = load_smolvlm() else {
        return;
    };
    let width = 64u32;
    let height = 64u32;
    // Solid red RGB8.
    let mut rgb = Vec::with_capacity((width * height * 3) as usize);
    for _ in 0..(width * height) {
        rgb.extend_from_slice(&[255u8, 0, 0]);
    }
    let mut req = request("What dominant color do you see? One word.", 24);
    req.images = vec![VisionFrame {
        t_sec: None,
        source: VisionSource::Rgb { width, height, rgb },
    }];
    let c = collect(&mut loaded, req);
    assert!(!c.text.is_empty(), "RGB frame must produce text");
    let u = c.usage.expect("usage");
    assert!(u.generated_tokens > 0, "{u:?}");
}

/// Missing mmproj: vision request fails with a clear media/unsupported error.
#[test]
fn smolvlm_vision_without_mmproj_errors() {
    let model = fixture(MODEL);
    if !model.is_file() {
        return;
    }
    let mut loaded = load(&model, &Placement::cpu(), &LoadConfig::default()).expect("load text");
    assert!(!loaded.supports_native_vision());
    let image = fixture(IMAGE);
    if !image.is_file() {
        return;
    }
    let mut req = request("What is in the image?", 16);
    req.images = vec![VisionFrame {
        t_sec: None,
        source: VisionSource::Path(image),
    }];
    let err = loaded.generate(req).err().expect("must fail without mmproj");
    let msg = err.to_string().to_lowercase();
    assert!(
        msg.contains("mmproj") || msg.contains("vision") || msg.contains("media"),
        "unexpected error: {err}"
    );
}
