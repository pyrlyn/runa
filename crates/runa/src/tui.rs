// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! Full-screen chat (`runa chat --tui`, P8.6).
//!
//! A `ratatui` + `crossterm` transcript UI over the same engine calls as the
//! line REPL in `main.rs`: a transcript pane, collapsible reasoning
//! (`Ctrl+R`), multi-line input (`tui-textarea`; `Enter` sends,
//! `Shift+Enter` inserts a newline, bracketed paste lands verbatim), and a
//! status bar (model, mode, tok/s, context use). Slash commands mirror
//! `chat_command` — keep both lists in sync; the shared `SLASH_HELP` lines
//! are the single source for the command half.
//!
//! The state, key handling and rendering are pure over `ChatTui` (no I/O),
//! so the snapshot tests drive the same `view` the live loop draws. The
//! event loop and generation live in `main.rs`, which owns `Session` and
//! `LoadedModel`.

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::buffer::Buffer;
use ratatui::layout::{Constraint, Layout, Position, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Widget};
use tui_textarea::TextArea;
use unicode_width::UnicodeWidthStr;

/// Rows of the status bar.
const STATUS_ROWS: u16 = 1;
/// The input box grows with the text, clamped to this many text rows.
const INPUT_MAX_ROWS: usize = 5;
/// `ctx N%` is `ctx_used * 100 / ctx_total`.
const PERCENT_SCALE: u64 = 100;
/// Lines a `PageUp`/`PageDown` moves the transcript.
const SCROLL_PAGE: usize = 10;

/// Slash command reference shared with the line REPL's `/help`.
pub const SLASH_HELP: &[&str] = &[
    "/mode <cpu|gpu|hybrid>  reload with a placement",
    "/model <path>           load another local model",
    "/think [on|off|budget N|effort L|show|hide]  thinking (P3.1)",
    "/reset                  clear KV cache + history",
    "/usage                  show last turn counters",
    "/quit                   leave",
];

/// Who said a transcript message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    User,
    Assistant,
    Reasoning,
    Notice,
}

/// One transcript entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Message {
    pub role: Role,
    pub text: String,
}

impl Message {
    pub fn user(text: &str) -> Self {
        Self {
            role: Role::User,
            text: text.to_owned(),
        }
    }

    #[cfg(test)]
    pub fn assistant(text: &str) -> Self {
        Self {
            role: Role::Assistant,
            text: text.to_owned(),
        }
    }

    #[cfg(test)]
    pub fn reasoning(text: &str) -> Self {
        Self {
            role: Role::Reasoning,
            text: text.to_owned(),
        }
    }

    pub fn notice(text: &str) -> Self {
        Self {
            role: Role::Notice,
            text: text.to_owned(),
        }
    }
}

/// A parsed `/` line. Arguments stay strings here; `main.rs` applies them to
/// `Session`/`LoadedModel` with the same helpers as `chat_command`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SlashCmd {
    Quit,
    Help,
    Mode(String),
    Model(String),
    Think(String),
    Reset,
    Usage,
    Unknown(String),
}

/// Parse a slash line (`/mode cpu`, …). Returns `None` for ordinary input.
pub fn parse_slash(line: &str) -> Option<SlashCmd> {
    if !line.starts_with('/') {
        return None;
    }
    let mut parts = line[1..].split_whitespace();
    Some(match parts.next().unwrap_or("") {
        "quit" | "exit" | "q" => SlashCmd::Quit,
        "help" => SlashCmd::Help,
        "mode" => match parts.next() {
            Some(m) => SlashCmd::Mode(m.to_owned()),
            None => SlashCmd::Unknown("usage: /mode <cpu|gpu|hybrid>".to_owned()),
        },
        "model" => match parts.next() {
            Some(m) => SlashCmd::Model(m.to_owned()),
            None => SlashCmd::Unknown("usage: /model <local-path>".to_owned()),
        },
        "think" => SlashCmd::Think(parts.collect::<Vec<_>>().join(" ")),
        "reset" => SlashCmd::Reset,
        "usage" => SlashCmd::Usage,
        other => SlashCmd::Unknown(format!("unknown command /{other} — /help lists commands")),
    })
}

/// What a key did beyond editing the input box.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KeyAction {
    Nothing,
    Submit(String),
    Slash(SlashCmd),
    Quit,
}

fn fresh_input() -> TextArea<'static> {
    let mut area = TextArea::default();
    // The default underlines the cursor line, which reads as a mistake in a
    // chat box.
    area.set_cursor_line_style(Style::default());
    area.set_placeholder_text("message · Shift+Enter for a newline · /help for commands");
    area
}

/// Full TUI state: transcript, input box, status numbers.
#[derive(Debug, Clone)]
pub struct ChatTui {
    messages: Vec<Message>,
    input: TextArea<'static>,
    /// Submitted texts, oldest first, for `Up`/`Down` recall.
    history: Vec<String>,
    /// The history entry on screen while browsing, if any.
    browsing: Option<usize>,
    show_reasoning: bool,
    model: String,
    mode: String,
    toks_per_s: f64,
    ctx_used: u32,
    ctx_total: u32,
    busy: bool,
    /// Lines scrolled up from the bottom of the transcript.
    scroll: usize,
    /// A first idle `Ctrl+C` arms; the second quits.
    ctrl_c_armed: bool,
}

impl ChatTui {
    pub fn new(model: &str, mode: &str, ctx_total: u32) -> Self {
        Self {
            messages: Vec::new(),
            input: fresh_input(),
            history: Vec::new(),
            browsing: None,
            show_reasoning: false,
            model: model.to_owned(),
            mode: mode.to_owned(),
            toks_per_s: 0.0,
            ctx_used: 0,
            ctx_total,
            busy: false,
            scroll: 0,
            ctrl_c_armed: false,
        }
    }

    pub fn push(&mut self, message: Message) {
        self.messages.push(message);
    }

    /// Append streamed text to the last message when it has the same role,
    /// else push a new one. Powers live token streaming in the event loop.
    pub fn extend_last(&mut self, role: Role, piece: &str) {
        if piece.is_empty() {
            return;
        }
        match self.messages.last_mut() {
            Some(last) if last.role == role => last.text.push_str(piece),
            _ => self.messages.push(Message {
                role,
                text: piece.to_owned(),
            }),
        }
    }

    pub fn set_busy(&mut self, busy: bool) {
        self.busy = busy;
    }

    pub fn set_status(&mut self, toks_per_s: f64, ctx_used: u32) {
        self.toks_per_s = toks_per_s;
        self.ctx_used = ctx_used;
    }

    pub fn set_mode(&mut self, mode: &str) {
        self.mode = mode.to_owned();
    }

    pub fn set_model(&mut self, model: &str) {
        self.model = model.to_owned();
    }

    pub fn toggle_reasoning(&mut self) {
        self.show_reasoning = !self.show_reasoning;
    }

    /// Insert pasted text verbatim (bracketed paste, may span lines).
    pub fn insert_text(&mut self, text: &str) {
        self.input.insert_str(text);
    }

    pub fn input_text(&self) -> String {
        self.input.lines().join("\n")
    }

    fn submit_text(&mut self) -> String {
        let text = self.input_text();
        self.history.push(text.clone());
        self.browsing = None;
        self.input = fresh_input();
        self.scroll = 0;
        text
    }

    fn browse(&mut self, index: Option<usize>) {
        self.browsing = index;
        let text = index
            .and_then(|i| self.history.get(i))
            .cloned()
            .unwrap_or_default();
        self.input = fresh_input();
        self.input.insert_str(&text);
    }

    fn is_input_empty(&self) -> bool {
        self.input.lines().iter().all(|l| l.trim().is_empty())
    }
}

/// Feed one key into the TUI; pure over `ChatTui` like the rest of this
/// module. Generation and slash effects run in `main.rs`.
pub fn on_key(tui: &mut ChatTui, key: KeyEvent) -> KeyAction {
    let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
    if ctrl && key.code == KeyCode::Char('c') {
        if tui.ctrl_c_armed {
            return KeyAction::Quit;
        }
        tui.ctrl_c_armed = true;
        return KeyAction::Nothing;
    }
    tui.ctrl_c_armed = false;
    if ctrl && key.code == KeyCode::Char('d') {
        return KeyAction::Quit;
    }
    if ctrl && key.code == KeyCode::Char('r') {
        tui.toggle_reasoning();
        return KeyAction::Nothing;
    }
    match key.code {
        KeyCode::PageUp => {
            tui.scroll = tui.scroll.saturating_add(SCROLL_PAGE);
            return KeyAction::Nothing;
        }
        KeyCode::PageDown => {
            tui.scroll = tui.scroll.saturating_sub(SCROLL_PAGE);
            return KeyAction::Nothing;
        }
        KeyCode::Esc => {
            if !tui.is_input_empty() {
                tui.input = fresh_input();
            }
            return KeyAction::Nothing;
        }
        KeyCode::Enter
            if key
                .modifiers
                .intersects(KeyModifiers::SHIFT | KeyModifiers::ALT) =>
        {
            tui.input.insert_newline();
            return KeyAction::Nothing;
        }
        KeyCode::Enter => {
            if tui.is_input_empty() {
                return KeyAction::Nothing;
            }
            let text = tui.submit_text();
            if let Some(cmd) = parse_slash(&text) {
                return KeyAction::Slash(cmd);
            }
            return KeyAction::Submit(text);
        }
        KeyCode::Up => {
            let (row, _) = tui.input.cursor();
            if row == 0 && !tui.history.is_empty() {
                let last = tui.history.len().saturating_sub(1);
                let index = tui.browsing.map_or(last, |i| i.saturating_sub(1));
                tui.browse(Some(index));
                return KeyAction::Nothing;
            }
        }
        KeyCode::Down => {
            let (row, _) = tui.input.cursor();
            if tui.browsing.is_some() && row + 1 == tui.input.lines().len() {
                let next = tui
                    .browsing
                    .and_then(|i| (i + 1 < tui.history.len()).then_some(i + 1));
                tui.browse(next);
                return KeyAction::Nothing;
            }
        }
        _ => {}
    }
    tui.input.input(key);
    KeyAction::Nothing
}

/// Wrap `text` to `width` columns; the first visual line carries `prefix`,
/// continuations are indented to the prefix width. `prefix` must be ASCII
/// (all call-site prefixes are), so the indent keeps the same byte width.
fn wrapped(prefix: &str, text: &str, width: usize) -> Vec<String> {
    let indent = " ".repeat(prefix.width());
    let limit = width.max(1);
    let mut out = Vec::new();
    for (line_index, raw) in text.split('\n').enumerate() {
        let lead = if line_index == 0 {
            prefix
        } else {
            indent.as_str()
        };
        let mut current = lead.to_owned();
        let mut current_width = lead.width();
        for word in raw.split(' ') {
            let gap = u8::from(current_width > lead.width());
            if current_width + usize::from(gap) + word.width() > limit
                && current_width > lead.width()
            {
                out.push(current);
                current = format!("{indent}{word}");
                current_width = indent.width() + word.width();
            } else {
                if gap == 1 {
                    current.push(' ');
                    current_width += 1;
                }
                current.push_str(word);
                current_width += word.width();
            }
        }
        out.push(current);
    }
    out
}

/// Split a line built by `wrapped` into its lead (`prefix.len()` ASCII
/// bytes) and the rest, so the lead can carry the role style.
fn split_lead(prefix_len: usize, line: String) -> (String, String) {
    let tail = line[prefix_len..].to_owned();
    let head = line[..prefix_len].to_owned();
    (head, tail)
}

/// Transcript lines for one message at `width` columns.
fn message_lines(message: &Message, show_reasoning: bool, width: usize) -> Vec<Line<'static>> {
    match message.role {
        Role::User => wrapped("you> ", &message.text, width)
            .into_iter()
            .map(|l| {
                let (head, tail) = split_lead("you> ".len(), l);
                Line::from(vec![
                    Span::styled(head, Style::default().add_modifier(Modifier::BOLD)),
                    Span::raw(tail),
                ])
            })
            .collect(),
        Role::Assistant => wrapped("runa> ", &message.text, width)
            .into_iter()
            .map(|l| {
                let (head, tail) = split_lead("runa> ".len(), l);
                Line::from(vec![
                    Span::styled(head, Style::default().add_modifier(Modifier::BOLD)),
                    Span::raw(tail),
                ])
            })
            .collect(),
        Role::Reasoning if !show_reasoning => {
            let hidden = message.text.lines().count().max(1);
            vec![Line::styled(
                format!("(reasoning hidden: {hidden} lines — Ctrl+R to show)"),
                Style::default().add_modifier(Modifier::DIM),
            )]
        }
        Role::Reasoning => wrapped("think| ", &message.text, width)
            .into_iter()
            .map(|l| Line::styled(l, Style::default().add_modifier(Modifier::DIM)))
            .collect(),
        Role::Notice => wrapped("(i) ", &message.text, width)
            .into_iter()
            .map(|l| Line::styled(l, Style::default().add_modifier(Modifier::DIM)))
            .collect(),
    }
}

fn status_line(tui: &ChatTui) -> Line<'static> {
    let percent = u64::from(tui.ctx_used)
        .saturating_mul(PERCENT_SCALE)
        .checked_div(u64::from(tui.ctx_total.max(1)))
        .unwrap_or(0);
    let tail = if tui.ctrl_c_armed {
        " · Ctrl+C again to quit"
    } else if tui.busy {
        " · working…"
    } else {
        ""
    };
    Line::styled(
        format!(
            " {} · {} · {:.1} tok/s · ctx {percent}% ({}/{}){tail} · Ctrl+R reasoning · /help",
            tui.model, tui.mode, tui.toks_per_s, tui.ctx_used, tui.ctx_total,
        ),
        Style::default().add_modifier(Modifier::DIM),
    )
}

/// Draw the whole frame; returns where the cursor goes.
pub fn view(tui: &ChatTui, area: Rect, buf: &mut Buffer) -> Option<Position> {
    let input_rows = u16::try_from(tui.input.lines().len().clamp(1, INPUT_MAX_ROWS))
        .unwrap_or(INPUT_MAX_ROWS as u16);
    let [transcript, input_area, status] = Layout::vertical([
        Constraint::Min(1),
        Constraint::Length(input_rows.saturating_add(2)),
        Constraint::Length(STATUS_ROWS),
    ])
    .areas(area);

    let width = usize::from(transcript.width);
    let mut lines: Vec<Line<'static>> = Vec::new();
    if tui.messages.is_empty() {
        lines.push(Line::styled(
            "(no messages yet — type below, /help for commands)",
            Style::default().add_modifier(Modifier::DIM),
        ));
    }
    for message in &tui.messages {
        lines.extend(message_lines(message, tui.show_reasoning, width));
    }
    let rows = usize::from(transcript.height);
    let offset = lines.len().saturating_sub(rows.saturating_add(tui.scroll));
    Paragraph::new(lines)
        .scroll((u16::try_from(offset).unwrap_or(u16::MAX), 0))
        .render(transcript, buf);

    let block = Block::default()
        .borders(Borders::ALL)
        .title(" input (Enter send · Shift+Enter newline) ");
    let inner = block.inner(input_area);
    block.render(input_area, buf);
    let [prompt, text] =
        Layout::horizontal([Constraint::Length(2), Constraint::Min(1)]).areas(inner);
    Line::raw(">").render(prompt, buf);
    (&tui.input).render(text, buf);

    status_line(tui).render(status, buf);

    let (row, col) = tui.input.cursor();
    Some(Position::new(
        text.x
            .saturating_add(u16::try_from(col).unwrap_or(u16::MAX))
            .min(text.right().saturating_sub(1)),
        text.y
            .saturating_add(u16::try_from(row).unwrap_or(u16::MAX))
            .min(text.bottom().saturating_sub(1)),
    ))
}

/// Rows of a buffer as text, trailing spaces trimmed (snapshot form).
#[cfg(test)]
pub fn buffer_to_string(buf: &Buffer) -> String {
    (0..buf.area.height)
        .map(|y| {
            (0..buf.area.width)
                .map(|x| buf[(x, y)].symbol())
                .collect::<String>()
                .trim_end()
                .to_string()
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::event::KeyEvent;
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    fn sample() -> ChatTui {
        let mut tui = ChatTui::new("qwen3-8b", "cpu", 8192);
        tui.push(Message::user("what is GGUF?"));
        tui.push(Message::reasoning("line one\nline two\nline three"));
        tui.push(Message::assistant("a single-file container for tensors."));
        tui.push(Message::notice("(context cleared)"));
        tui.set_status(42.5, 1024);
        tui.insert_text("draft follow-up");
        tui
    }

    fn render(tui: &ChatTui, width: u16, height: u16) -> String {
        let backend = TestBackend::new(width, height);
        let mut terminal = Terminal::new(backend).expect("test terminal");
        terminal
            .draw(|frame| {
                view(tui, frame.area(), frame.buffer_mut());
            })
            .expect("test draw");
        buffer_to_string(terminal.backend().buffer())
    }

    #[test]
    fn frame_with_collapsed_reasoning() {
        insta::assert_snapshot!(render(&sample(), 80, 24));
    }

    #[test]
    fn frame_with_expanded_reasoning() {
        let mut tui = sample();
        tui.toggle_reasoning();
        insta::assert_snapshot!(render(&tui, 80, 24));
    }

    #[test]
    fn slash_parses_like_the_repl() {
        assert_eq!(parse_slash("hello"), None);
        assert_eq!(parse_slash("/quit"), Some(SlashCmd::Quit));
        assert_eq!(parse_slash("/exit"), Some(SlashCmd::Quit));
        assert_eq!(parse_slash("/q"), Some(SlashCmd::Quit));
        assert_eq!(parse_slash("/help"), Some(SlashCmd::Help));
        assert_eq!(
            parse_slash("/mode gpu"),
            Some(SlashCmd::Mode("gpu".to_owned()))
        );
        assert_eq!(
            parse_slash("/model foo.gguf"),
            Some(SlashCmd::Model("foo.gguf".to_owned()))
        );
        assert_eq!(
            parse_slash("/think budget 512"),
            Some(SlashCmd::Think("budget 512".to_owned()))
        );
        assert_eq!(parse_slash("/think"), Some(SlashCmd::Think(String::new())));
        assert_eq!(parse_slash("/reset"), Some(SlashCmd::Reset));
        assert_eq!(parse_slash("/usage"), Some(SlashCmd::Usage));
        assert!(matches!(parse_slash("/mode"), Some(SlashCmd::Unknown(_))));
        assert!(matches!(parse_slash("/bogus"), Some(SlashCmd::Unknown(_))));
    }

    #[test]
    fn enter_submits_and_shift_enter_breaks() {
        let mut tui = ChatTui::new("m", "cpu", 1024);
        tui.insert_text("hi");
        let action = on_key(&mut tui, KeyEvent::from(KeyCode::Enter));
        assert_eq!(action, KeyAction::Submit("hi".to_owned()));
        assert_eq!(tui.input_text(), String::new());
        tui.insert_text("a");
        let mut shifted = KeyEvent::from(KeyCode::Enter);
        shifted.modifiers = KeyModifiers::SHIFT;
        assert_eq!(on_key(&mut tui, shifted), KeyAction::Nothing);
        assert_eq!(tui.input_text(), "a\n");
    }

    #[test]
    fn slash_submit_routes_to_slash_action() {
        let mut tui = ChatTui::new("m", "cpu", 1024);
        tui.insert_text("/usage");
        let action = on_key(&mut tui, KeyEvent::from(KeyCode::Enter));
        assert_eq!(action, KeyAction::Slash(SlashCmd::Usage));
    }

    #[test]
    fn ctrl_r_toggles_reasoning_and_ctrl_c_quits_twice() {
        let mut tui = ChatTui::new("m", "cpu", 1024);
        let toggle = KeyEvent::new(KeyCode::Char('r'), KeyModifiers::CONTROL);
        assert_eq!(on_key(&mut tui, toggle), KeyAction::Nothing);
        assert!(tui.show_reasoning);
        let quit = KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL);
        assert_eq!(on_key(&mut tui, quit), KeyAction::Nothing);
        assert_eq!(on_key(&mut tui, quit), KeyAction::Quit);
    }
}
