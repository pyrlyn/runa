//! Small string grammars shared by the CLI, config files and the HTTP
//! server: model refs (`hf:org/repo:Q4_K_M`), cloud refs
//! (`cloud:openai:gpt-…`), backend / NPU names, think / effort / budget
//! values and quant picking.
#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use runa_core::{BackendKind, Effort, ThinkConfig, ThinkOverrides, parse_budget};
use runa_fit::{NpuKind, is_safetensors_tag, parse_model_ref, pick_quant};

#[derive(Debug, Arbitrary)]
struct Input {
    s: String,
    siblings: Vec<String>,
    budget: Option<u32>,
}

fuzz_target!(|input: Input| {
    let s = input.s.as_str();
    let _ = parse_model_ref(s);
    let _ = is_safetensors_tag(s);
    let _ = pick_quant(&input.siblings, s);
    if let Some(r) = runa_cloud::parse_cloud_ref(s) {
        let _ = r.provider_name();
    }
    let _ = BackendKind::parse(s);
    let _ = NpuKind::parse(s);
    let _ = parse_budget(s);
    let _ = ThinkOverrides::parse_think(s);
    let _ = ThinkOverrides::parse_show(s);
    // Same path as the HTTP server's `reasoning_effort` / budget fields.
    let mut o = ThinkOverrides {
        budget: input.budget,
        ..ThinkOverrides::default()
    };
    if let Ok(e) = Effort::parse(s) {
        o.effort = Some(e);
    }
    let _ = ThinkConfig::default().apply(&o);
});
