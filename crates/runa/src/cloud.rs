// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! Cloud backend routing for `runa run` (P3.7), with the MCP tool loop
//! (P8.3).

use std::io::{self, Write};

use std::path::Path;

use runa_cloud::openai::{
    AudioFormat, AudioInput, ChatMessage, ChatRequest, CloudEvent as OpenAiEvent, OpenAiClient,
};
use runa_cloud::{
    AnthropicClient, AnthropicEvent, AnthropicRequest, ChatTurn, CloudRef, PriceTable, Provider,
    parse_cloud_ref, resolve_api_key, tools_from_openai,
};
use runa_core::{ThinkConfig, ThinkMode, ToolCall};
use serde_json::{Value, json};

use crate::mcp::{McpHub, call_with, tool_loop};

/// One `runa run` against a cloud model.
pub struct CloudRun<'a> {
    pub prompt: &'a str,
    pub think: ThinkConfig,
    pub max_tokens: u32,
    pub json: bool,
    pub audio: Option<&'a Path>,
    pub audio_pref: runa_media::AudioRoutePref,
    /// Structured output (P8.1): JSON Schema text, already validated.
    pub json_schema: Option<&'a str>,
    /// MCP tools the model may call (P8.3).
    pub mcp: Option<&'a McpHub>,
    pub max_tool_rounds: u32,
}

/// Answer text, reasoning and token counts summed over tool rounds.
#[derive(Default)]
struct Reply {
    text: String,
    reasoning: String,
    input_tokens: u32,
    output_tokens: u32,
}

impl Reply {
    fn add_text(&mut self, s: &str) {
        if !self.text.is_empty() && !s.is_empty() {
            self.text.push('\n');
        }
        self.text.push_str(s);
    }
}

/// Name of the forced tool that carries Anthropic structured output.
const ANSWER_TOOL: &str = "answer";

pub fn run_cloud(cloud: &CloudRef, run: &CloudRun<'_>) -> Result<(), String> {
    let CloudRun {
        prompt,
        think,
        max_tokens,
        json,
        audio,
        audio_pref,
        json_schema,
        mcp,
        max_tool_rounds,
    } = *run;
    let json_schema = json_schema
        .map(serde_json::from_str::<Value>)
        .transpose()
        .map_err(|e| format!("--json-schema: {e}"))?;
    let (prompt, oai_audio) = prepare_cloud_audio(cloud, prompt, audio, audio_pref)?;
    let prices = PriceTable::load();
    let key = resolve_api_key(cloud.provider).map_err(|e| e.to_string())?;
    eprintln!(
        "cloud: {}:{} (key {})",
        cloud.provider_name(),
        cloud.model,
        key.redacted()
    );
    // One request per `block_on`: MCP calls between rounds block on the
    // hub's own runtime, which must not nest inside this one.
    let rt = tokio::runtime::Runtime::new().map_err(|e| e.to_string())?;
    let tools = mcp.map(McpHub::tools_json);
    let mut reply = Reply::default();

    match cloud.provider {
        Provider::OpenAi => {
            let base = std::env::var("OPENAI_BASE_URL").ok();
            let client = OpenAiClient::new(key.value, base.as_deref());
            let mut req = ChatRequest {
                model: cloud.model.clone(),
                messages: vec![{
                    let mut m = ChatMessage::user(prompt);
                    m.audio = oai_audio;
                    m
                }],
                think,
                max_tokens: Some(max_tokens),
                json_schema,
                tools,
            };
            tool_loop(max_tool_rounds, call_with(mcp), |results| {
                req.messages
                    .extend(results.iter().map(|(call, out)| ChatMessage {
                        role: "tool".into(),
                        content: out.clone(),
                        tool_call_id: Some(call.id.clone()),
                        ..ChatMessage::default()
                    }));
                let events = rt
                    .block_on(client.complete(req.clone()))
                    .map_err(|e| e.to_string())?;
                let (text, calls) = drain_openai(events, &mut reply);
                reply.add_text(&text);
                if !calls.is_empty() {
                    req.messages.push(ChatMessage {
                        role: "assistant".into(),
                        content: text,
                        tool_calls: calls.clone(),
                        ..ChatMessage::default()
                    });
                }
                Ok(calls)
            })?;
        }
        Provider::Anthropic => {
            let _ = oai_audio;
            // Structured output: one forced `answer` tool whose input is the
            // answer. Forced tool use rules out thinking.
            let structured = json_schema.is_some();
            if structured && mcp.is_some() {
                return Err("--json-schema with --mcp is not supported on anthropic".into());
            }
            let (tools, tool_choice, think) = match json_schema {
                Some(schema) => (
                    vec![json!({
                        "name": ANSWER_TOOL,
                        "description": "Reply with the answer in this shape.",
                        "input_schema": schema,
                    })],
                    Some(json!({"type": "tool", "name": ANSWER_TOOL})),
                    ThinkConfig {
                        mode: ThinkMode::Off,
                        ..think
                    },
                ),
                None => (
                    tools.as_ref().map(tools_from_openai).unwrap_or_default(),
                    None,
                    think,
                ),
            };
            let client = AnthropicClient::new(key.value);
            let mut req = AnthropicRequest {
                model: cloud.model.clone(),
                system: None,
                messages: vec![ChatTurn::text("user", prompt)],
                think,
                max_tokens,
                images: vec![],
                pdfs: vec![],
                // Non-stream on purpose: only the whole-message reply
                // preserves thinking signatures for the next tool round
                // (P10.6 made the SSE path tool-complete, but streamed
                // thinking blocks carry no signatures to send back).
                stream: false,
                tools,
                tool_choice,
            };
            tool_loop(max_tool_rounds, call_with(mcp), |results| {
                if !results.is_empty() {
                    let answers: Vec<(String, String)> = results
                        .iter()
                        .map(|(call, out)| (call.id.clone(), out.clone()))
                        .collect();
                    req.messages.push(ChatTurn::tool_results(&answers));
                }
                let events = rt.block_on(client.generate(&req))?;
                let (text, tool_use) = drain_anthropic(events, &mut reply)?;
                reply.add_text(&text);
                let Some((calls, content)) = tool_use else {
                    return Ok(Vec::new());
                };
                if structured {
                    let answer = calls.iter().find(|c| c.name == ANSWER_TOOL);
                    reply.add_text(answer.map_or("", |c| c.arguments.as_str()));
                    return Ok(Vec::new());
                }
                req.messages.push(ChatTurn {
                    role: "assistant".into(),
                    text: String::new(),
                    blocks: Some(content),
                });
                Ok(calls)
            })?;
        }
    }

    let Reply {
        text,
        reasoning,
        input_tokens: input_tok,
        output_tokens: output_tok,
    } = reply;
    if json {
        println!(
            "{{\"text\":\"{}\",\"reasoning\":\"{}\",\"usage\":{{\"prompt_tokens\":{},\"completion_tokens\":{}}}}}",
            crate::json_escape(&text),
            crate::json_escape(&reasoning),
            input_tok,
            output_tok
        );
    } else {
        if think.show && !reasoning.is_empty() {
            eprint!("{reasoning}");
            io::stderr().flush().map_err(|e| format!("stderr: {e}"))?;
        }
        print!("{text}");
        io::stdout().flush().map_err(|e| format!("stdout: {e}"))?;
        println!();
        eprintln!("tokens: {input_tok} in / {output_tok} out");
        if let Some(line) =
            prices.cost_line(cloud.provider_name(), &cloud.model, input_tok, output_tok)
        {
            eprintln!("{line}");
        }
    }
    Ok(())
}

fn prepare_cloud_audio(
    cloud: &CloudRef,
    prompt: &str,
    audio: Option<&Path>,
    pref: runa_media::AudioRoutePref,
) -> Result<(String, Vec<AudioInput>), String> {
    let Some(path) = audio else {
        return Ok((prompt.to_string(), Vec::new()));
    };
    let backend = match cloud.provider {
        Provider::OpenAi => runa_media::AudioBackend::OpenAi {
            audio_capable: runa_media::openai_audio_capable(&cloud.model),
        },
        Provider::Anthropic => runa_media::AudioBackend::Anthropic,
    };
    match runa_media::select_audio_route(pref, backend)? {
        runa_media::AudioPlan::Transcribe => Ok((fold_audio_transcript(prompt, path)?, Vec::new())),
        runa_media::AudioPlan::OpenAiInputAudio => {
            Ok((prompt.to_string(), vec![openai_wav_input(path)?]))
        }
        runa_media::AudioPlan::Native => {
            Err("native audio is local-only; cloud uses input_audio or transcript".into())
        }
    }
}

pub(crate) fn fold_audio_transcript(prompt: &str, audio: &Path) -> Result<String, String> {
    let t = runa_media::transcribe_file(audio, &runa_media::AsrOptions::default())
        .map_err(|e| e.to_string())?;
    let text = t.text.trim();
    if text.is_empty() {
        return Err("asr produced empty transcript".into());
    }
    Ok(format!("{prompt}\n\n[audio transcript]\n{text}"))
}

fn openai_wav_input(audio: &Path) -> Result<AudioInput, String> {
    let decoded = runa_media::decode_audio(audio).map_err(|e| e.to_string())?;
    let wav = runa_media::pcm_to_wav_bytes(&decoded.samples, decoded.probe.pcm_sample_rate)
        .map_err(|e| e.to_string())?;
    Ok(AudioInput {
        data_b64: runa_media::wav_base64(&wav),
        format: AudioFormat::Wav,
    })
}

pub fn cloud_from_on_unfit(spec: &str) -> Result<CloudRef, String> {
    parse_cloud_ref(spec).ok_or_else(|| format!("on_unfit cloud:{spec}: need backend:model"))
}

/// One round's answer text and tool calls; reasoning and tokens go to `reply`.
fn drain_openai(events: Vec<OpenAiEvent>, reply: &mut Reply) -> (String, Vec<ToolCall>) {
    let mut text = String::new();
    let mut calls = Vec::new();
    for ev in events {
        match ev {
            OpenAiEvent::Text(s) => text.push_str(&s),
            OpenAiEvent::Reasoning(s) => reply.reasoning.push_str(&s),
            OpenAiEvent::ToolCalls(c) => calls = c,
            OpenAiEvent::Usage {
                prompt_tokens,
                completion_tokens,
            } => {
                reply.input_tokens += prompt_tokens;
                reply.output_tokens += completion_tokens;
            }
        }
    }
    (text, calls)
}

/// One round's answer text and `tool_use` (calls + raw content blocks).
type ToolUse = Option<(Vec<ToolCall>, Value)>;

fn drain_anthropic(
    events: Vec<AnthropicEvent>,
    reply: &mut Reply,
) -> Result<(String, ToolUse), String> {
    let mut text = String::new();
    let mut tool_use = None;
    for ev in events {
        match ev {
            AnthropicEvent::Text(s) => text.push_str(&s),
            AnthropicEvent::Reasoning(s) => reply.reasoning.push_str(&s),
            AnthropicEvent::ToolUse { calls, content } => tool_use = Some((calls, content)),
            AnthropicEvent::Usage {
                input_tokens,
                output_tokens,
            } => {
                reply.input_tokens += input_tokens;
                reply.output_tokens += output_tokens;
            }
            AnthropicEvent::Refusal(s) => return Err(format!("refusal: {s}")),
            AnthropicEvent::Done { .. } => {}
        }
    }
    Ok((text, tool_use))
}

#[cfg(test)]
mod tests {
    use super::*;
    use runa_media::AudioRoutePref;
    use std::path::Path;

    #[test]
    fn prepare_passthrough_without_audio() {
        let cloud = parse_cloud_ref("openai:gpt-4o-mini").unwrap();
        let (p, a) = prepare_cloud_audio(&cloud, "hi", None, AudioRoutePref::Auto).unwrap();
        assert_eq!(p, "hi");
        assert!(a.is_empty());
    }

    #[test]
    fn prepare_native_anthropic_errors() {
        let cloud = parse_cloud_ref("anthropic:claude-sonnet-4").unwrap();
        let err = prepare_cloud_audio(
            &cloud,
            "hi",
            Some(Path::new("/nope.wav")),
            AudioRoutePref::Native,
        )
        .unwrap_err();
        assert!(err.to_lowercase().contains("anthropic"), "{err}");
    }

    #[test]
    fn prepare_native_openai_text_model_errors() {
        let cloud = parse_cloud_ref("openai:gpt-4o-mini").unwrap();
        let err = prepare_cloud_audio(
            &cloud,
            "hi",
            Some(Path::new("/nope.wav")),
            AudioRoutePref::Native,
        )
        .unwrap_err();
        assert!(err.contains("input_audio"), "{err}");
    }
}
