// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! Structured output (P8.1) and tool calling (P8.2).
//!
//! Rendering goes through llama.cpp's own Jinja chat handler
//! (`apply_chat_template_oaicompat`), which knows each model family's
//! reply format and returns the grammar that constrains it, plus any extra
//! stop strings. Models without a template fall back to a plain prompt and
//! a grammar built straight from the schema. Tool replies are parsed back
//! by the same handler (`parse_response_oaicompat`).

use llama_cpp_2::model::{ChatTemplateResult, GrammarTrigger, GrammarTriggerType};
use llama_cpp_2::openai::OpenAIChatTemplateParams;
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;
use serde_json::{Value, json};

use crate::generate::{ChatMessage, ToolCall};
use crate::load::{EngineError, LoadedModel};

/// A GBNF grammar ready to become a sampler.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Grammar {
    gbnf: String,
    /// Lazy: enforced only after a trigger matches (tool calls, P8.2).
    lazy: bool,
    patterns: Vec<String>,
    tokens: Vec<LlamaToken>,
}

impl Grammar {
    fn eager(gbnf: String) -> Grammar {
        Grammar {
            gbnf,
            lazy: false,
            patterns: Vec::new(),
            tokens: Vec::new(),
        }
    }

    pub(crate) fn sampler(&self, loaded: &LoadedModel) -> Result<LlamaSampler, EngineError> {
        let model = loaded.model();
        let built = if self.lazy {
            LlamaSampler::grammar_lazy_patterns(
                model,
                &self.gbnf,
                "root",
                &self.patterns,
                &self.tokens,
            )
        } else {
            LlamaSampler::grammar(model, &self.gbnf, "root")
        };
        built.map_err(|e| EngineError::Grammar(format!("{e:?}")))
    }
}

/// Prompt + grammar for one constrained request.
pub(crate) struct Constrained {
    pub prompt: String,
    pub templated: bool,
    pub grammar: Option<Grammar>,
    pub stops: Vec<String>,
    /// Parser for the reply when the request carried tools.
    pub tool_reply: Option<ToolReply>,
}

/// What the Jinja handler needs beyond the messages.
pub(crate) struct TemplateInputs<'a> {
    pub json_schema: Option<&'a str>,
    pub grammar: Option<&'a str>,
    pub tools: Option<&'a str>,
    pub tool_choice: Option<&'a str>,
    pub think: runa_core::ThinkConfig,
}

/// Parses a finished tool-request reply into text + calls.
pub(crate) struct ToolReply(ChatTemplateResult);

impl ToolReply {
    /// Raw reply → answer text (reasoning and call markup stripped) + calls.
    /// Unparseable output degrades to plain text with no calls.
    pub(crate) fn parse(&self, raw: &str) -> (String, Vec<ToolCall>) {
        let msg: Value = self
            .0
            .parse_response_oaicompat(raw, false)
            .ok()
            .and_then(|j| serde_json::from_str(&j).ok())
            .unwrap_or(Value::Null);
        let content = if msg.is_null() {
            raw
        } else {
            msg["content"].as_str().unwrap_or_default()
        };
        // The parser keeps `<think>` blocks in `content`; reasoning already
        // streamed live, so drop it here.
        let (_, text) = runa_core::parse_stream(runa_core::ReasonFamily::Auto, &[content]);
        let mut calls: Vec<ToolCall> = msg["tool_calls"]
            .as_array()
            .map(|a| a.iter().filter_map(tool_call_from_json).collect())
            .unwrap_or_default();
        for (i, c) in calls.iter_mut().enumerate() {
            if c.id.is_empty() {
                c.id = format!("call_{i}");
            }
        }
        (text.trim().to_owned(), calls)
    }
}

fn tool_call_from_json(v: &Value) -> Option<ToolCall> {
    let f = &v["function"];
    let arguments = match &f["arguments"] {
        Value::String(s) => s.clone(),
        Value::Null => "{}".to_owned(),
        other => other.to_string(),
    };
    Some(ToolCall {
        id: v["id"].as_str().unwrap_or_default().to_owned(),
        name: f["name"].as_str()?.to_owned(),
        arguments,
    })
}

/// JSON Schema → GBNF (llama.cpp's converter).
pub fn schema_to_grammar(schema: &str) -> Result<String, EngineError> {
    serde_json::from_str::<serde_json::Value>(schema)
        .map_err(|e| EngineError::Grammar(format!("json schema is not valid JSON: {e}")))?;
    llama_cpp_2::json_schema_to_grammar(schema).map_err(|e| EngineError::Grammar(e.to_string()))
}

impl LoadedModel {
    /// Render `messages` for a request that carries a JSON Schema, a raw
    /// GBNF grammar or tools. Callers turn thinking off for schemas and
    /// grammars: an eager grammar leaves no room for a reasoning block.
    pub(crate) fn render_oaicompat(
        &self,
        messages: &[ChatMessage],
        add_generation_prompt: bool,
        inputs: &TemplateInputs<'_>,
    ) -> Result<Constrained, EngineError> {
        // Validates both inputs up front and is the fallback grammar.
        let fallback = eager_grammar(inputs.json_schema, inputs.grammar)?;
        let Ok(tmpl) = self.model().chat_template(None) else {
            if inputs.tools.is_some() {
                return Err(EngineError::Template(
                    "tool calling needs a model with a chat template".into(),
                ));
            }
            return Ok(Constrained {
                prompt: self.render_prompt(messages, add_generation_prompt)?,
                templated: false,
                grammar: fallback,
                stops: Vec::new(),
                tool_reply: None,
            });
        };
        let messages_json = messages_json(messages);
        let tools = inputs.tools.is_some();
        let params = OpenAIChatTemplateParams {
            messages_json: &messages_json,
            tools_json: inputs.tools,
            tool_choice: inputs.tool_choice,
            json_schema: inputs.json_schema,
            grammar: inputs.grammar,
            reasoning_format: None,
            chat_template_kwargs: Some(inputs.think.template_kwargs()),
            add_generation_prompt,
            use_jinja: true,
            parallel_tool_calls: tools,
            enable_thinking: inputs.think.enable_thinking(),
            add_bos: false,
            add_eos: false,
            parse_tool_calls: tools,
        };
        let result = self
            .model()
            .apply_chat_template_oaicompat(&tmpl, &params)
            .map_err(|e| EngineError::Template(format!("{e:?}")))?;
        Ok(from_template(result, fallback, tools))
    }
}

/// Grammar straight from `--grammar` / `--json-schema`, for prompts that do
/// not go through the Jinja handler (no template, media prefill).
pub(crate) fn eager_grammar(
    json_schema: Option<&str>,
    grammar: Option<&str>,
) -> Result<Option<Grammar>, EngineError> {
    Ok(match (json_schema, grammar) {
        (Some(_), Some(_)) => {
            return Err(EngineError::Grammar(
                "json schema and grammar are mutually exclusive".into(),
            ));
        }
        (Some(s), None) => Some(Grammar::eager(schema_to_grammar(s)?)),
        (None, Some(g)) => Some(Grammar::eager(g.to_owned())),
        (None, None) => None,
    })
}

/// The template's own grammar (format-aware: channel prefixes, lazy tool
/// triggers) wins over the plain fallback.
fn from_template(
    mut result: ChatTemplateResult,
    fallback: Option<Grammar>,
    tools: bool,
) -> Constrained {
    let (patterns, tokens) = trigger_patterns(&result.grammar_triggers);
    let grammar = match result.grammar.take() {
        Some(gbnf) => Some(Grammar {
            gbnf,
            lazy: result.grammar_lazy,
            patterns,
            tokens,
        }),
        None => fallback,
    };
    Constrained {
        prompt: std::mem::take(&mut result.prompt),
        templated: true,
        grammar,
        stops: std::mem::take(&mut result.additional_stops),
        tool_reply: tools.then_some(ToolReply(result)),
    }
}

/// OpenAI-style messages (with `tool_calls` / `tool_call_id`) for the Jinja
/// handler.
fn messages_json(messages: &[ChatMessage]) -> String {
    Value::Array(
        messages
            .iter()
            .map(|m| {
                let mut v = json!({ "role": m.role, "content": m.content });
                if !m.tool_calls.is_empty() {
                    v["tool_calls"] = m
                        .tool_calls
                        .iter()
                        .map(|c| {
                            json!({
                                "id": c.id,
                                "type": "function",
                                "function": { "name": c.name, "arguments": c.arguments },
                            })
                        })
                        .collect();
                }
                if let Some(id) = &m.tool_call_id {
                    v["tool_call_id"] = json!(id);
                }
                v
            })
            .collect(),
    )
    .to_string()
}

/// Lazy-grammar triggers → regex patterns + trigger tokens, the way
/// llama-server feeds `llama_sampler_init_grammar_lazy_patterns`.
fn trigger_patterns(triggers: &[GrammarTrigger]) -> (Vec<String>, Vec<LlamaToken>) {
    let mut patterns = Vec::new();
    let mut tokens = Vec::new();
    for t in triggers {
        match t.trigger_type {
            GrammarTriggerType::Token => tokens.extend(t.token),
            GrammarTriggerType::Word => patterns.push(regex_escape(&t.value)),
            GrammarTriggerType::Pattern => patterns.push(t.value.clone()),
            GrammarTriggerType::PatternFull => {
                let mut p = t.value.clone();
                if !p.starts_with('^') {
                    p.insert(0, '^');
                }
                if !p.ends_with('$') {
                    p.push('$');
                }
                patterns.push(p);
            }
        }
    }
    (patterns, tokens)
}

fn regex_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        if r"\.^$|?*+()[]{}-".contains(c) {
            out.push('\\');
        }
        out.push(c);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trigger(trigger_type: GrammarTriggerType, value: &str) -> GrammarTrigger {
        GrammarTrigger {
            trigger_type,
            value: value.to_owned(),
            token: None,
        }
    }

    #[test]
    fn schema_to_grammar_has_root() {
        let g = schema_to_grammar(
            r#"{"type":"object","properties":{"a":{"type":"integer"}},"required":["a"]}"#,
        )
        .unwrap();
        assert!(g.contains("root ::="), "{g}");
    }

    #[test]
    fn schema_must_be_json() {
        assert!(schema_to_grammar("{not json").is_err());
    }

    #[test]
    fn triggers_become_patterns() {
        let (p, t) = trigger_patterns(&[
            trigger(GrammarTriggerType::Word, "<tool_call>"),
            trigger(GrammarTriggerType::Pattern, "[\\s\\S]*?(<tool_call>)"),
            trigger(GrammarTriggerType::PatternFull, "(?:x)"),
        ]);
        assert_eq!(p, ["<tool_call>", "[\\s\\S]*?(<tool_call>)", "^(?:x)$"]);
        assert!(t.is_empty());
    }

    #[test]
    fn word_triggers_are_escaped() {
        assert_eq!(regex_escape("[TOOL_CALLS]"), "\\[TOOL_CALLS\\]");
        assert_eq!(regex_escape("a.b"), "a\\.b");
    }

    #[test]
    fn messages_json_is_openai_shape() {
        let j = messages_json(&[ChatMessage::user("hi \"x\"")]);
        // Compare as values: key order follows the serde_json map impl,
        // which changes under feature unification (e.g. `preserve_order`
        // via the P9.2 mistralrs tree).
        let v: Value = serde_json::from_str(&j).unwrap();
        assert_eq!(
            v,
            serde_json::json!([{"role": "user", "content": "hi \"x\""}])
        );
    }

    #[test]
    fn messages_json_carries_tool_turns() {
        let call = ToolCall {
            id: "c1".into(),
            name: "weather".into(),
            arguments: r#"{"city":"Paris"}"#.into(),
        };
        let j: Value = serde_json::from_str(&messages_json(&[
            ChatMessage {
                role: "assistant".into(),
                tool_calls: vec![call.clone()],
                ..ChatMessage::default()
            },
            ChatMessage {
                role: "tool".into(),
                content: "sunny".into(),
                tool_call_id: Some("c1".into()),
                ..ChatMessage::default()
            },
        ]))
        .unwrap();
        assert_eq!(
            j[0]["tool_calls"][0]["function"]["arguments"],
            call.arguments
        );
        assert_eq!(j[0]["tool_calls"][0]["type"], "function");
        assert_eq!(j[1]["tool_call_id"], "c1");
        assert_eq!(tool_call_from_json(&j[0]["tool_calls"][0]), Some(call));
    }

    #[test]
    fn object_arguments_become_json_text() {
        let v = json!({"id": "x", "function": {"name": "f", "arguments": {"a": 1}}});
        assert_eq!(tool_call_from_json(&v).unwrap().arguments, r#"{"a":1}"#);
        assert_eq!(tool_call_from_json(&json!({"function": {}})), None);
    }

    /// P10.11: Harmony (gpt-oss) tool format through the REAL template.
    /// Needs `tests/fixtures/gpt-oss-20b-MXFP4.gguf` (tiny custom loadable
    /// gpt-oss with the real Harmony chat_template; soft-skips if missing).
    /// Renders a required tool call, then parses a canned Harmony reply:
    /// analysis channel + `to=functions` recipient (both header orders per
    /// `common_chat_parse_gpt_oss`).
    #[test]
    fn harmony_tool_reply_parses() {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/gpt-oss-20b-MXFP4.gguf");
        if !path.is_file() {
            return;
        }
        let loaded = crate::load::load(
            &path,
            &crate::placement::Placement::cpu(),
            &crate::load::LoadConfig {
                n_ctx: 512,
                ..crate::load::LoadConfig::default()
            },
        )
        .expect("gpt-oss cpu load");
        let tools = r#"[{"type":"function","function":{"name":"get_weather","description":"Current weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]"#;
        let rendered = loaded
            .render_oaicompat(
                &[ChatMessage::user("What is the weather in Paris?")],
                true,
                &TemplateInputs {
                    json_schema: None,
                    grammar: None,
                    tools: Some(tools),
                    tool_choice: Some("required"),
                    think: runa_core::ThinkConfig::default(),
                },
            )
            .expect("harmony renders a tool request");
        let reply = rendered.tool_reply.expect("tools arm a reply parser");
        for raw in [
            // Recipient in the role section, channel after.
            concat!(
                "<|channel|>analysis<|message|>Checking the weather.<|end|>",
                "<|start|>assistant to=functions.get_weather",
                "<|channel|>commentary<|message|>",
                "{\"city\": \"Paris\"}<|call|>",
            ),
            // Recipient inside the channel header.
            concat!(
                "<|channel|>commentary to=functions.get_weather<|message|>",
                "{\"city\": \"Paris\"}<|call|>",
            ),
        ] {
            let (text, calls) = reply.parse(raw);
            assert_eq!(calls.len(), 1, "{raw}");
            assert_eq!(calls[0].name, "get_weather", "{raw}");
            assert!(calls[0].arguments.contains("Paris"), "{calls:?} from {raw}");
            assert!(!text.contains("commentary"), "markup leaked: {text:?}");
        }
    }
}
