// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! Streaming split of model text into reasoning vs answer (P3.2).
//!
//! Families: `<think>` (Qwen3, DeepSeek, GLM), Harmony channels (gpt-oss),
//! Gemma thought tags. Tags may be split across `push` chunks.

use crate::think::{ThinkConfig, ThinkMode};

/// Known reasoning-tag families.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReasonFamily {
    /// Detect on the first open tag in the stream.
    Auto,
    /// `<think>…</think>` — Qwen3, DeepSeek, GLM.
    XmlThink,
    /// `<|channel|>analysis` … `<|channel|>final` — gpt-oss Harmony.
    Harmony,
    /// `<start_of_thought>` / `<|channel|>thought`.
    Gemma,
}

impl ReasonFamily {
    /// Infer from a chat-template string (`enable_thinking`, channel names).
    pub fn from_template(template: &str) -> Self {
        if template.contains("<|channel|>analysis") {
            ReasonFamily::Harmony
        } else if template.contains("<|channel|>thought") || template.contains("<start_of_thought>")
        {
            ReasonFamily::Gemma
        } else if template.contains("<think>") || template.contains("enable_thinking") {
            ReasonFamily::XmlThink
        } else {
            ReasonFamily::Auto
        }
    }

    fn pairs(self) -> &'static [(&'static str, &'static str)] {
        match self {
            ReasonFamily::Auto => &[
                ("<think>", "</think>"),
                ("<|channel|>analysis", "<|channel|>final"),
                ("<start_of_thought>", "<end_of_thought>"),
                ("<|channel|>thought", "<|channel|>response"),
            ],
            ReasonFamily::XmlThink => &[("<think>", "</think>")],
            ReasonFamily::Harmony => &[("<|channel|>analysis", "<|channel|>final")],
            ReasonFamily::Gemma => &[
                ("<start_of_thought>", "<end_of_thought>"),
                ("<|channel|>thought", "<|channel|>response"),
            ],
        }
    }
}

impl ThinkConfig {
    /// Qwen-style template kwarg: thinking is on unless mode is Off.
    pub fn enable_thinking(&self) -> bool {
        !matches!(self.mode, ThinkMode::Off)
    }

    /// JSON object for `--chat-template-kwargs`.
    pub fn template_kwargs(&self) -> &'static str {
        if self.enable_thinking() {
            r#"{"enable_thinking":true}"#
        } else {
            r#"{"enable_thinking":false}"#
        }
    }
}

/// One parsed slice of the token stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReasonPiece {
    Reasoning(String),
    Text(String),
}

/// Incremental parser. Holds back only a delimiter prefix.
#[derive(Debug, Clone)]
pub struct ReasoningParser {
    family: ReasonFamily,
    in_reason: bool,
    buf: String,
    open: Option<&'static str>,
    close: Option<&'static str>,
}

impl ReasoningParser {
    pub fn new(family: ReasonFamily) -> Self {
        ReasoningParser {
            family,
            in_reason: false,
            buf: String::new(),
            open: None,
            close: None,
        }
    }

    /// True after the open tag has been consumed and before close.
    pub fn in_reason(&self) -> bool {
        self.in_reason
    }

    /// Close delimiter once the family/open tag is known.
    pub fn close_tag(&self) -> Option<&'static str> {
        self.close
    }

    /// Held bytes that might still complete a delimiter. Never inject here.
    pub fn holding_partial(&self) -> bool {
        !self.buf.is_empty()
    }

    pub fn push(&mut self, chunk: &str) -> Vec<ReasonPiece> {
        if chunk.is_empty() {
            return Vec::new();
        }
        self.buf.push_str(chunk);
        let mut out = Vec::new();
        loop {
            if !self.in_reason {
                if self.open.is_none()
                    && let Some((o, c)) = find_open(&self.buf, self.family.pairs())
                {
                    self.open = Some(o);
                    self.close = Some(c);
                }
                let Some(open) = self.open else {
                    if let Some(emit) =
                        emit_safe(&mut self.buf, &prefixes(self.family.pairs(), true))
                    {
                        out.push(ReasonPiece::Text(emit));
                    }
                    break;
                };
                if let Some(i) = self.buf.find(open) {
                    if i > 0 {
                        out.push(ReasonPiece::Text(self.buf[..i].to_string()));
                    }
                    self.buf.replace_range(..i + open.len(), "");
                    strip_prefix(&mut self.buf, "<|message|>");
                    self.in_reason = true;
                    continue;
                }
                if let Some(emit) = emit_safe(&mut self.buf, &[open]) {
                    out.push(ReasonPiece::Text(emit));
                }
                break;
            } else {
                let close = self.close.expect("close set with open");
                if let Some(i) = self.buf.find(close) {
                    let body = self.buf[..i].to_string();
                    if !body.is_empty() {
                        out.push(ReasonPiece::Reasoning(body));
                    }
                    self.buf.replace_range(..i + close.len(), "");
                    strip_prefix(&mut self.buf, "<|message|>");
                    self.in_reason = false;
                    continue;
                }
                if let Some(emit) = emit_safe(&mut self.buf, &[close]) {
                    out.push(ReasonPiece::Reasoning(emit));
                }
                break;
            }
        }
        out.retain(|p| match p {
            ReasonPiece::Text(s) | ReasonPiece::Reasoning(s) => !s.is_empty(),
        });
        out
    }

    pub fn flush(&mut self) -> Vec<ReasonPiece> {
        if self.buf.is_empty() {
            return Vec::new();
        }
        let rest = std::mem::take(&mut self.buf);
        vec![if self.in_reason {
            ReasonPiece::Reasoning(rest)
        } else {
            ReasonPiece::Text(rest)
        }]
    }

    pub fn finish(mut self) -> Vec<ReasonPiece> {
        self.flush()
    }
}

fn find_open(
    buf: &str,
    pairs: &[(&'static str, &'static str)],
) -> Option<(&'static str, &'static str)> {
    let mut best: Option<(usize, &(&str, &str))> = None;
    for p in pairs {
        if let Some(i) = buf.find(p.0)
            && best.is_none_or(|(bi, _)| i < bi)
        {
            best = Some((i, p));
        }
    }
    best.map(|(_, p)| (p.0, p.1))
}

fn prefixes<'a>(pairs: &'a [(&'static str, &'static str)], opens: bool) -> Vec<&'a str> {
    pairs
        .iter()
        .map(|(o, c)| if opens { *o } else { *c })
        .collect()
}

fn strip_prefix(buf: &mut String, p: &str) {
    if buf.starts_with(p) {
        buf.replace_range(..p.len(), "");
    }
}

/// Emit the prefix that cannot be the start of any `needles`.
fn emit_safe(buf: &mut String, needles: &[&str]) -> Option<String> {
    let keep = (1..=buf.len())
        .rev()
        .filter(|&n| buf.is_char_boundary(buf.len() - n))
        .find(|&n| needles.iter().any(|d| d.starts_with(&buf[buf.len() - n..])))
        .filter(|&n| {
            needles
                .iter()
                .any(|d| d.len() > n && d.starts_with(&buf[buf.len() - n..]))
        })
        .unwrap_or(0);
    if buf.len() > keep {
        let emit = buf[..buf.len() - keep].to_string();
        *buf = buf[buf.len() - keep..].to_string();
        Some(emit)
    } else {
        None
    }
}

/// Concatenate `push` results, then `finish`.
pub fn parse_stream(family: ReasonFamily, chunks: &[&str]) -> (String, String) {
    let mut p = ReasoningParser::new(family);
    let mut reasoning = String::new();
    let mut text = String::new();
    let mut take = |pieces: Vec<ReasonPiece>| {
        for piece in pieces {
            match piece {
                ReasonPiece::Reasoning(s) => reasoning.push_str(&s),
                ReasonPiece::Text(s) => text.push_str(&s),
            }
        }
    };
    for c in chunks {
        take(p.push(c));
    }
    take(p.finish());
    (reasoning, text)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn split_chars(s: &str) -> Vec<String> {
        s.chars().map(|c| c.to_string()).collect()
    }

    #[test]
    fn qwen3_xml_think() {
        let (r, t) = parse_stream(ReasonFamily::XmlThink, &["<think>step one</think>Paris"]);
        assert_eq!(r, "step one");
        assert_eq!(t, "Paris");
    }

    #[test]
    fn deepseek_xml_split_across_tokens() {
        let s = "<think>search</think>42";
        let binding = split_chars(s);
        let chunks: Vec<&str> = binding.iter().map(String::as_str).collect();
        let (r, t) = parse_stream(ReasonFamily::XmlThink, &chunks);
        assert_eq!(r, "search");
        assert_eq!(t, "42");
    }

    #[test]
    fn multibyte_text_is_not_split() {
        let (r, t) = parse_stream(ReasonFamily::XmlThink, &["sunny, 21 °", "C"]);
        assert_eq!(r, "");
        assert_eq!(t, "sunny, 21 °C");
    }

    #[test]
    fn glm_xml_think() {
        let (r, t) = parse_stream(
            ReasonFamily::XmlThink,
            &["<th", "ink>glm reason</th", "ink>\nanswer"],
        );
        assert_eq!(r, "glm reason");
        assert_eq!(t, "\nanswer");
    }

    #[test]
    fn harmony_channels() {
        let (r, t) = parse_stream(
            ReasonFamily::Harmony,
            &[
                "<|channel|>analysis",
                "need a plan",
                "<|channel|>final",
                "done",
            ],
        );
        assert_eq!(r, "need a plan");
        assert_eq!(t, "done");
    }

    #[test]
    fn gemma_start_of_thought() {
        let (r, t) = parse_stream(
            ReasonFamily::Gemma,
            &["<start_of_thought>why<end_of_thought>because"],
        );
        assert_eq!(r, "why");
        assert_eq!(t, "because");
    }

    #[test]
    fn auto_detects_harmony() {
        let (r, t) = parse_stream(
            ReasonFamily::Auto,
            &["<|ch", "annel|>analysisx<|channel|>finaly"],
        );
        assert_eq!(r, "x");
        assert_eq!(t, "y");
    }

    #[test]
    fn template_detect_enable_thinking() {
        assert_eq!(
            ReasonFamily::from_template("{{ enable_thinking }} <think>"),
            ReasonFamily::XmlThink
        );
        assert_eq!(
            ReasonFamily::from_template("<|channel|>analysis"),
            ReasonFamily::Harmony
        );
        assert_eq!(
            ReasonFamily::from_template("<|channel|>thought"),
            ReasonFamily::Gemma
        );
    }

    #[test]
    fn enable_thinking_kwargs() {
        assert!(!ThinkConfig::default().enable_thinking());
        assert_eq!(
            ThinkConfig::default().template_kwargs(),
            r#"{"enable_thinking":false}"#
        );
        let on = ThinkConfig {
            mode: ThinkMode::On,
            show: true,
        };
        assert!(on.enable_thinking());
        assert_eq!(on.template_kwargs(), r#"{"enable_thinking":true}"#);
    }

    #[test]
    fn no_tags_is_all_text() {
        let (r, t) = parse_stream(ReasonFamily::Auto, &["hello ", "world"]);
        assert!(r.is_empty());
        assert_eq!(t, "hello world");
    }

    #[test]
    fn partial_close_is_held() {
        let mut p = ReasoningParser::new(ReasonFamily::XmlThink);
        let _ = p.push("<think>ab");
        assert!(p.in_reason());
        assert_eq!(p.close_tag(), Some("</think>"));
        assert!(!p.holding_partial());
        let _ = p.push("</th");
        assert!(p.holding_partial());
        assert!(p.in_reason());
    }
}
