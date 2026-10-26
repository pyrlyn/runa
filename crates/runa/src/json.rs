// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! Hand-built `--json` output lines shared by the local session and the
//! cloud path. A leaf module so neither has to import the other or `cli`.

/// Minimal JSON string escaping for those hand-built lines.
pub(crate) fn escape(s: &str) -> String {
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
    #[test]
    fn escape_output_parses_back_to_the_input() {
        let raw = "q\"b\\s\nr\rt\t\u{1}é";
        let line = format!("\"{}\"", super::escape(raw));
        assert_eq!(line, "\"q\\\"b\\\\s\\nr\\rt\\t\\u0001é\"");
        let back: String = serde_json::from_str(&line).expect("valid JSON string");
        assert_eq!(back, raw);
    }
}
