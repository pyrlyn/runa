//! NDJSON daemon protocol (P9.1).
//!
//! `runa run` / `runa chat` speak to `runa daemon` over a Unix socket at
//! [`default_socket_path`] (`RUNA_DAEMON_SOCK` overrides it in tests).
//! One JSON value per line: the client sends a [`DaemonRequest`], the
//! daemon answers with a stream of [`DaemonEvent`] ending in `done` or
//! `error`.
//!
//! The request carries a [`ProtoGenerateRequest`]: the serializable subset
//! of `runa_engine::GenerateRequest` the daemon serves (text + tools +
//! structured output). Media (`audio_pcm`, `images`), speculation, and MCP
//! servers stay local-only — callers fall back to in-process load when
//! those are set (see `daemon_compatible` in `main.rs`).

use std::path::PathBuf;
use std::time::Duration;

use runa_core::{Effort, ThinkConfig, ThinkMode};
use runa_engine::{ChatMessage, GenEvent, GenerateRequest, SamplingConfig, StopReason, ToolCall};
use serde::{Deserialize, Serialize};

/// Protocol version. Bumped on any incompatible wire change; the daemon
/// rejects requests whose `v` differs.
pub const PROTOCOL_VERSION: u32 = 1;

/// Idle-tick upper bound: `maybe_idle` runs at least this often even when
/// `idle_timeout_s` is huge (P9.1). Unix-only: the serving path.
#[cfg_attr(not(unix), allow(dead_code))]
pub const MAX_IDLE_TICK_SECS: u64 = 60;

/// Where the daemon listens. `RUNA_DAEMON_SOCK` wins (tests), else
/// `$XDG_CACHE_HOME/runa/runa.sock` or `~/.cache/runa/runa.sock`.
pub fn default_socket_path() -> PathBuf {
    if let Ok(p) = std::env::var("RUNA_DAEMON_SOCK")
        && !p.trim().is_empty()
    {
        return PathBuf::from(p.trim());
    }
    cache_dir().join("runa.sock")
}

fn cache_dir() -> PathBuf {
    if let Ok(p) = std::env::var("XDG_CACHE_HOME")
        && !p.is_empty()
    {
        return PathBuf::from(p).join("runa");
    }
    match std::env::var_os("HOME") {
        Some(home) => PathBuf::from(home).join(".cache").join("runa"),
        None => std::env::temp_dir().join("runa"),
    }
}

/// One function call, mirroring `runa_core::ToolCall`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProtoToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

impl From<&ToolCall> for ProtoToolCall {
    fn from(c: &ToolCall) -> Self {
        ProtoToolCall {
            id: c.id.clone(),
            name: c.name.clone(),
            arguments: c.arguments.clone(),
        }
    }
}

impl From<ProtoToolCall> for ToolCall {
    fn from(c: ProtoToolCall) -> Self {
        ToolCall {
            id: c.id,
            name: c.name,
            arguments: c.arguments,
        }
    }
}

/// One chat message, mirroring `runa_engine::ChatMessage`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProtoMessage {
    pub role: String,
    pub content: String,
    #[serde(default)]
    pub tool_calls: Vec<ProtoToolCall>,
    #[serde(default)]
    pub tool_call_id: Option<String>,
}

impl From<&ChatMessage> for ProtoMessage {
    fn from(m: &ChatMessage) -> Self {
        ProtoMessage {
            role: m.role.clone(),
            content: m.content.clone(),
            tool_calls: m.tool_calls.iter().map(ProtoToolCall::from).collect(),
            tool_call_id: m.tool_call_id.clone(),
        }
    }
}

impl From<ProtoMessage> for ChatMessage {
    fn from(m: ProtoMessage) -> Self {
        ChatMessage {
            role: m.role,
            content: m.content,
            tool_calls: m.tool_calls.into_iter().map(ToolCall::from).collect(),
            tool_call_id: m.tool_call_id,
        }
    }
}

/// Sampling parameters, mirroring `runa_engine::SamplingConfig`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProtoSampling {
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub min_p: f32,
    pub repeat_penalty: f32,
    pub repeat_last_n: i32,
    pub seed: u32,
}

impl From<&SamplingConfig> for ProtoSampling {
    fn from(s: &SamplingConfig) -> Self {
        ProtoSampling {
            temperature: s.temperature,
            top_k: s.top_k,
            top_p: s.top_p,
            min_p: s.min_p,
            repeat_penalty: s.repeat_penalty,
            repeat_last_n: s.repeat_last_n,
            seed: s.seed,
        }
    }
}

impl From<ProtoSampling> for SamplingConfig {
    fn from(s: ProtoSampling) -> Self {
        SamplingConfig {
            temperature: s.temperature,
            top_k: s.top_k,
            top_p: s.top_p,
            min_p: s.min_p,
            repeat_penalty: s.repeat_penalty,
            repeat_last_n: s.repeat_last_n,
            seed: s.seed,
            kernel_sampler: false,
        }
    }
}

/// Thinking mode, mirroring `runa_core::ThinkMode`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ProtoThinkMode {
    Off,
    On,
    Budget { tokens: u32, grace: u32 },
    Effort { effort: String },
}

impl ProtoThinkMode {
    fn from_mode(mode: ThinkMode) -> Self {
        match mode {
            ThinkMode::Off => ProtoThinkMode::Off,
            ThinkMode::On => ProtoThinkMode::On,
            ThinkMode::Budget { tokens, grace } => ProtoThinkMode::Budget { tokens, grace },
            ThinkMode::Effort(e) => ProtoThinkMode::Effort {
                effort: e.to_string(),
            },
        }
    }

    /// Unix-only: the serving path decodes requests back.
    #[cfg_attr(not(unix), allow(dead_code))]
    fn into_mode(self) -> Result<ThinkMode, String> {
        match self {
            ProtoThinkMode::Off => Ok(ThinkMode::Off),
            ProtoThinkMode::On => Ok(ThinkMode::On),
            ProtoThinkMode::Budget { tokens, grace } => Ok(ThinkMode::Budget { tokens, grace }),
            ProtoThinkMode::Effort { effort } => Effort::parse(&effort).map(ThinkMode::Effort),
        }
    }
}

/// Thinking config, mirroring `runa_core::ThinkConfig`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProtoThink {
    pub mode: ProtoThinkMode,
    pub show: bool,
}

impl From<ThinkConfig> for ProtoThink {
    fn from(t: ThinkConfig) -> Self {
        ProtoThink {
            mode: ProtoThinkMode::from_mode(t.mode),
            show: t.show,
        }
    }
}

impl ProtoThink {
    /// Unix-only: the serving path decodes requests back.
    #[cfg_attr(not(unix), allow(dead_code))]
    pub fn into_config(self) -> Result<ThinkConfig, String> {
        Ok(ThinkConfig {
            mode: self.mode.into_mode()?,
            show: self.show,
        })
    }
}

/// Serializable subset of `runa_engine::GenerateRequest` (no media, no
/// speculation: those stay local-only).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProtoGenerateRequest {
    pub messages: Vec<ProtoMessage>,
    pub sampling: ProtoSampling,
    pub max_tokens: u32,
    #[serde(default)]
    pub stop: Vec<String>,
    pub add_generation_prompt: bool,
    pub think: ProtoThink,
    #[serde(default)]
    pub json_schema: Option<String>,
    #[serde(default)]
    pub grammar: Option<String>,
    #[serde(default)]
    pub tools: Option<String>,
    #[serde(default)]
    pub tool_choice: Option<String>,
}

impl ProtoGenerateRequest {
    pub fn from_generate(req: &GenerateRequest) -> Self {
        ProtoGenerateRequest {
            messages: req.messages.iter().map(ProtoMessage::from).collect(),
            sampling: ProtoSampling::from(&req.sampling),
            max_tokens: req.max_tokens,
            stop: req.stop.clone(),
            add_generation_prompt: req.add_generation_prompt,
            think: ProtoThink::from(req.think),
            json_schema: req.json_schema.clone(),
            grammar: req.grammar.clone(),
            tools: req.tools.clone(),
            tool_choice: req.tool_choice.clone(),
        }
    }

    /// Unix-only: the serving path decodes requests back.
    #[cfg_attr(not(unix), allow(dead_code))]
    pub fn into_generate(self) -> Result<GenerateRequest, String> {
        Ok(GenerateRequest {
            messages: self.messages.into_iter().map(ChatMessage::from).collect(),
            sampling: SamplingConfig::from(self.sampling),
            max_tokens: self.max_tokens,
            stop: self.stop,
            add_generation_prompt: self.add_generation_prompt,
            think: self.think.into_config()?,
            audio_pcm: None,
            images: Vec::new(),
            speculative: runa_engine::Speculative::default(),
            json_schema: self.json_schema,
            grammar: self.grammar,
            tools: self.tools,
            tool_choice: self.tool_choice,
        })
    }
}

/// One client → daemon line: which model to generate with plus what.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DaemonRequest {
    pub v: u32,
    pub model: String,
    pub request: ProtoGenerateRequest,
}

impl DaemonRequest {
    pub fn new(model: String, req: &GenerateRequest) -> Self {
        DaemonRequest {
            v: PROTOCOL_VERSION,
            model,
            request: ProtoGenerateRequest::from_generate(req),
        }
    }
}

/// One daemon → client line. The stream ends at `done` or `error`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DaemonEvent {
    Text {
        text: String,
    },
    Reasoning {
        text: String,
    },
    ToolCalls {
        calls: Vec<ProtoToolCall>,
    },
    Usage {
        prompt_tokens: u32,
        generated_tokens: u32,
        /// Reasoning-body tokens (P10.4, M6). `#[serde(default)]` keeps
        /// the wire compatible with older daemons that do not send it.
        #[serde(default)]
        reasoning_tokens: u32,
    },
    Done {
        stop: String,
    },
    Error {
        message: String,
    },
}

impl DaemonEvent {
    /// Map one engine event; `Usage`/`Done` close the stream on the client.
    /// Unix-only: the serving path maps engine output to the wire.
    #[cfg_attr(not(unix), allow(dead_code))]
    pub fn from_gen_event(ev: &GenEvent) -> Option<Self> {
        match ev {
            GenEvent::Text(t) if !t.is_empty() => Some(DaemonEvent::Text { text: t.clone() }),
            GenEvent::Reasoning(t) if !t.is_empty() => {
                Some(DaemonEvent::Reasoning { text: t.clone() })
            }
            GenEvent::ToolCalls(c) if !c.is_empty() => Some(DaemonEvent::ToolCalls {
                calls: c.iter().map(ProtoToolCall::from).collect(),
            }),
            GenEvent::Usage(u) => Some(DaemonEvent::Usage {
                prompt_tokens: u.prompt_tokens,
                generated_tokens: u.generated_tokens,
                reasoning_tokens: u.reasoning_tokens,
            }),
            GenEvent::Done(r) => Some(DaemonEvent::Done { stop: stop_name(r) }),
            _ => None,
        }
    }

    /// Terminal line: `done` stops, `error` fails the request.
    pub fn is_terminal(&self) -> bool {
        matches!(self, DaemonEvent::Done { .. } | DaemonEvent::Error { .. })
    }
}

/// Unix-only: the serving path maps engine output to the wire.
#[cfg_attr(not(unix), allow(dead_code))]
fn stop_name(r: &StopReason) -> String {
    match r {
        StopReason::Eos => "eos".to_owned(),
        StopReason::MaxTokens => "max_tokens".to_owned(),
        StopReason::StopString(s) => format!("stop:{s}"),
    }
}

/// Parse a `stop:` line back (client side).
pub fn parse_stop(s: &str) -> StopReason {
    match s {
        "eos" => StopReason::Eos,
        "max_tokens" => StopReason::MaxTokens,
        other => match other.strip_prefix("stop:") {
            Some(matched) => StopReason::StopString(matched.to_owned()),
            None => StopReason::MaxTokens,
        },
    }
}

/// Serialize one NDJSON line (without the trailing newline).
/// Unix-only on non-test builds: the serving path (unit tests cover both).
#[cfg_attr(not(unix), allow(dead_code))]
pub fn encode_line<T: Serialize>(v: &T) -> Result<String, String> {
    serde_json::to_string(v).map_err(|e| e.to_string())
}

/// Parse one NDJSON line.
/// Unix-only on non-test builds: the serving path (unit tests cover both).
#[cfg_attr(not(unix), allow(dead_code))]
pub fn decode_line<T: serde::de::DeserializeOwned>(line: &str) -> Result<T, String> {
    serde_json::from_str(line).map_err(|e| e.to_string())
}

/// Read one line from any async buffered reader (tokio `UnixStream` halves
/// included). `Ok(None)` is a clean EOF. Unix-only: the socket transport.
#[cfg(unix)]
pub async fn read_line<R>(reader: &mut R) -> std::io::Result<Option<String>>
where
    R: tokio::io::AsyncBufRead + Unpin,
{
    use tokio::io::AsyncBufReadExt;
    let mut line = String::new();
    let n = reader.read_line(&mut line).await?;
    if n == 0 {
        return Ok(None);
    }
    while line.ends_with('\n') || line.ends_with('\r') {
        line.pop();
    }
    Ok(Some(line))
}

/// Write one NDJSON line to any async writer. Unix-only: the socket transport.
#[cfg(unix)]
pub async fn write_line<W>(writer: &mut W, line: &str) -> std::io::Result<()>
where
    W: tokio::io::AsyncWrite + Unpin,
{
    use tokio::io::AsyncWriteExt;
    writer.write_all(line.as_bytes()).await?;
    writer.write_all(b"\n").await?;
    writer.flush().await
}

/// Blocking client: send one request, collect events until `done`.
/// Transport failures (`Err`) mean "no daemon" — the caller falls back to
/// in-process load. A daemon-side failure arrives as
/// `DaemonEvent::Error` inside the returned `Ok` vec.
#[cfg(unix)]
pub fn request_sync(
    socket: &std::path::Path,
    req: &DaemonRequest,
    timeout: Duration,
) -> Result<Vec<DaemonEvent>, String> {
    use std::io::{BufRead, BufReader, Write};
    use std::os::unix::net::UnixStream;
    let stream = UnixStream::connect(socket)
        .map_err(|e| format!("daemon dial {}: {e}", socket.display()))?;
    stream
        .set_read_timeout(Some(timeout))
        .map_err(|e| e.to_string())?;
    stream
        .set_write_timeout(Some(timeout))
        .map_err(|e| e.to_string())?;
    let mut stream = stream;
    let mut line = encode_line(req)?;
    line.push('\n');
    stream
        .write_all(line.as_bytes())
        .map_err(|e| format!("daemon write: {e}"))?;
    let mut reader = BufReader::new(&stream);
    let mut out = Vec::new();
    loop {
        let mut raw = String::new();
        reader
            .read_line(&mut raw)
            .map_err(|e| format!("daemon read: {e}"))?;
        if raw.is_empty() {
            return Err("daemon closed the connection".into());
        }
        let ev: DaemonEvent = decode_line(raw.trim())?;
        let terminal = ev.is_terminal();
        out.push(ev);
        if terminal {
            return Ok(out);
        }
    }
}

/// Non-unix stub: there is no socket to dial, so callers always fall back
/// to in-process load.
#[cfg(not(unix))]
pub fn request_sync(
    _socket: &std::path::Path,
    _req: &DaemonRequest,
    _timeout: Duration,
) -> Result<Vec<DaemonEvent>, String> {
    Err("daemon needs a unix socket".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_request() -> GenerateRequest {
        GenerateRequest {
            messages: vec![
                ChatMessage::user("hi"),
                ChatMessage {
                    role: "assistant".into(),
                    content: String::new(),
                    tool_calls: vec![ToolCall {
                        id: "t1".into(),
                        name: "get_weather".into(),
                        arguments: r#"{"city":"Paris"}"#.into(),
                    }],
                    tool_call_id: None,
                },
            ],
            sampling: SamplingConfig {
                temperature: 0.0,
                seed: 7,
                ..SamplingConfig::default()
            },
            max_tokens: 64,
            stop: vec!["</s>".into()],
            add_generation_prompt: true,
            think: ThinkConfig {
                mode: ThinkMode::Budget {
                    tokens: 256,
                    grace: 32,
                },
                show: true,
            },
            json_schema: Some(r#"{"type":"object"}"#.into()),
            grammar: None,
            tools: Some(r#"[{"type":"function"}]"#.into()),
            tool_choice: Some("auto".into()),
            ..GenerateRequest::default()
        }
    }

    #[test]
    fn request_roundtrips_through_ndjson() {
        let req = DaemonRequest::new("qwen".into(), &sample_request());
        let line = encode_line(&req).unwrap();
        assert!(!line.contains('\n'));
        let back: DaemonRequest = decode_line(&line).unwrap();
        assert_eq!(back, req);
        assert_eq!(back.v, PROTOCOL_VERSION);
        let request = back.request.into_generate().unwrap();
        let first = sample_request();
        assert_eq!(request.messages, first.messages);
        assert_eq!(request.max_tokens, 64);
        assert_eq!(request.sampling.seed, 7);
        assert_eq!(request.sampling.temperature, 0.0);
        assert_eq!(request.stop, vec!["</s>"]);
        assert_eq!(
            request.think.mode,
            ThinkMode::Budget {
                tokens: 256,
                grace: 32
            }
        );
        assert!(request.think.show);
        assert_eq!(request.json_schema.as_deref(), Some(r#"{"type":"object"}"#));
        assert_eq!(request.tool_choice.as_deref(), Some("auto"));
        // No media survives the daemon subset.
        assert!(request.audio_pcm.is_none() && request.images.is_empty());
    }

    #[test]
    fn think_modes_roundtrip() {
        for mode in [
            ThinkMode::Off,
            ThinkMode::On,
            ThinkMode::Budget {
                tokens: 1024,
                grace: 64,
            },
            ThinkMode::Effort(Effort::High),
        ] {
            let proto = ProtoThink::from(ThinkConfig { mode, show: false });
            let line = encode_line(&proto).unwrap();
            let back: ProtoThink = decode_line(&line).unwrap();
            assert_eq!(back.into_config().unwrap().mode, mode);
        }
    }

    #[test]
    fn bad_effort_fails_at_decode_time() {
        let proto = ProtoThink {
            mode: ProtoThinkMode::Effort {
                effort: "extreme".into(),
            },
            show: false,
        };
        assert!(proto.into_config().is_err());
    }

    #[test]
    fn events_roundtrip_and_map_gen_events() {
        let gen_events = [
            GenEvent::Text("hello".into()),
            GenEvent::Reasoning("plan".into()),
            GenEvent::ToolCalls(vec![ToolCall {
                id: "t1".into(),
                name: "a".into(),
                arguments: "{}".into(),
            }]),
            GenEvent::Usage(runa_engine::Usage {
                prompt_tokens: 10,
                generated_tokens: 5,
                reasoning_tokens: 2,
                pp_toks_per_s: 0.0,
                tg_toks_per_s: 0.0,
            }),
            GenEvent::Done(StopReason::Eos),
        ];
        let events: Vec<DaemonEvent> = gen_events
            .iter()
            .filter_map(DaemonEvent::from_gen_event)
            .collect();
        assert_eq!(events.len(), 5);
        // Empty pieces are dropped so the wire stays quiet.
        assert!(DaemonEvent::from_gen_event(&GenEvent::Text(String::new())).is_none());
        for ev in &events {
            let line = encode_line(ev).unwrap();
            let back: DaemonEvent = decode_line(&line).unwrap();
            assert_eq!(&back, ev);
        }
        // P10.4: reasoning count survives the wire; old daemons that omit
        // the field decode to 0.
        assert!(matches!(
            events[3],
            DaemonEvent::Usage {
                reasoning_tokens: 2,
                ..
            }
        ));
        let legacy: DaemonEvent =
            serde_json::from_str(r#"{"type":"usage","prompt_tokens":1,"generated_tokens":2}"#)
                .unwrap();
        assert!(matches!(
            legacy,
            DaemonEvent::Usage {
                reasoning_tokens: 0,
                ..
            }
        ));
        assert!(!events[0].is_terminal());
        assert!(events[4].is_terminal());
        assert!(
            DaemonEvent::Error {
                message: "x".into()
            }
            .is_terminal()
        );
        assert_eq!(parse_stop("eos"), StopReason::Eos);
        assert_eq!(parse_stop("max_tokens"), StopReason::MaxTokens);
        assert_eq!(
            parse_stop("stop:</s>"),
            StopReason::StopString("</s>".into())
        );
        assert_eq!(parse_stop("bogus"), StopReason::MaxTokens);
    }

    #[test]
    fn socket_path_override_and_default() {
        let sock = std::env::temp_dir().join(format!(
            "runa-test-{}.sock",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        // SAFETY: single-threaded test mutating its own env is standard.
        unsafe { std::env::set_var("RUNA_DAEMON_SOCK", &sock) };
        assert_eq!(default_socket_path(), sock);
        unsafe { std::env::remove_var("RUNA_DAEMON_SOCK") };
        let def = default_socket_path();
        assert!(def.ends_with("runa/runa.sock"), "{}", def.display());
    }
}
