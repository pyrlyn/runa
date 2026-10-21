// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! Text and JSON renderings of a finished turn. Pure: returns strings.

use runa_engine::{StopReason, Usage};

/// One-line usage for `/usage` (REPL + TUI): reasoning appears only when
/// the turn actually thought (P10.4).
pub(super) fn format_usage(u: &Usage) -> String {
    let thinking = if u.reasoning_tokens > 0 {
        format!(" (reasoning {})", u.reasoning_tokens)
    } else {
        String::new()
    };
    format!(
        "prompt {} / generated {}{thinking} · {:.1} pp tok/s · {:.1} tg tok/s",
        u.prompt_tokens, u.generated_tokens, u.pp_toks_per_s, u.tg_toks_per_s
    )
}

pub(crate) fn run_json(text: &str, usage: &Usage, reason: &StopReason) -> String {
    let stop = match reason {
        StopReason::Eos => "eos".to_owned(),
        StopReason::MaxTokens => "max_tokens".to_owned(),
        StopReason::StopString(s) => format!("stop:{}", json_escape(s)),
    };
    format!(
        "{{\"text\":\"{}\",\"usage\":{{\"prompt_tokens\":{},\"generated_tokens\":{},\"reasoning_tokens\":{},\"pp_toks_per_s\":{:.1},\"tg_toks_per_s\":{:.1}}},\"stop\":\"{stop}\"}}",
        json_escape(text),
        usage.prompt_tokens,
        usage.generated_tokens,
        usage.reasoning_tokens,
        usage.pp_toks_per_s,
        usage.tg_toks_per_s,
    )
}

/// Minimal JSON string escaping (no serde in the binary crate).
pub(crate) fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn usage(reasoning_tokens: u32) -> Usage {
        Usage {
            prompt_tokens: 12,
            generated_tokens: 34,
            reasoning_tokens,
            pp_toks_per_s: 100.04,
            tg_toks_per_s: 20.06,
        }
    }

    #[test]
    fn usage_line_names_reasoning_only_when_the_turn_thought() {
        assert_eq!(
            format_usage(&usage(0)),
            "prompt 12 / generated 34 · 100.0 pp tok/s · 20.1 tg tok/s"
        );
        assert_eq!(
            format_usage(&usage(5)),
            "prompt 12 / generated 34 (reasoning 5) · 100.0 pp tok/s · 20.1 tg tok/s"
        );
    }

    #[test]
    fn run_json_is_valid_json_with_escaped_text_and_stop() {
        let out = run_json(
            "say \"hi\"\n\u{1}",
            &usage(0),
            &StopReason::StopString("</s>\t".into()),
        );
        let v: serde_json::Value = serde_json::from_str(&out).expect(&out);
        assert_eq!(v["text"], "say \"hi\"\n\u{1}");
        assert_eq!(v["stop"], "stop:</s>\t");
        assert_eq!(v["usage"]["generated_tokens"], 34);
        assert!(out.contains("\\u0001"), "{out}");
    }
}
