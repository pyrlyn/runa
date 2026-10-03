// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! ASR via whisper.cpp (`whisper-rs` 0.16.0) — plan D8 / P4.2.
//!
//! Models (`base`, `large-v3-turbo`) and the Silero VAD ggml are auto-pulled
//! into `~/.local/share/runa/models/whisper/`. Energy VAD chunks audio when
//! the Silero file is not present. Language defaults to auto-detect.
//!
//! Parakeet-TDT lives behind `--features parakeet` (`sherpa-onnx`). Do not
//! use the archived `sherpa-rs` crate.

use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Once;

use thiserror::Error;
use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, get_lang_str,
};

use crate::audio::TARGET_SAMPLE_RATE;

const HF_REPO: &str = "ggerganov/whisper.cpp";
const SILERO_FILE: &str = "ggml-silero-v5.1.2.bin";

static WHISPER_LOG: Once = Once::new();

/// ggml-org / ggerganov whisper.cpp files we auto-pull.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WhisperKind {
    Base,
    LargeV3Turbo,
}

impl WhisperKind {
    pub fn parse(s: &str) -> Result<Self, AsrError> {
        let s = s
            .trim()
            .trim_start_matches("whisper:")
            .trim_start_matches("ggml-")
            .trim_end_matches(".bin");
        match s {
            "base" => Ok(Self::Base),
            "large-v3-turbo" | "large_v3_turbo" | "turbo" => Ok(Self::LargeV3Turbo),
            other => Err(AsrError::UnknownModel(other.to_string())),
        }
    }

    pub fn file_name(self) -> &'static str {
        match self {
            Self::Base => "ggml-base.bin",
            Self::LargeV3Turbo => "ggml-large-v3-turbo.bin",
        }
    }

    pub fn hf_url(self) -> String {
        hf_resolve_url(self.file_name())
    }
}

fn hf_resolve_url(file: &str) -> String {
    format!("https://huggingface.co/{HF_REPO}/resolve/main/{file}")
}

/// Data root: `$XDG_DATA_HOME/runa`, else `~/.local/share/runa`.
pub fn data_dir() -> PathBuf {
    if let Ok(xdg) = std::env::var("XDG_DATA_HOME")
        && !xdg.is_empty()
    {
        return PathBuf::from(xdg).join("runa");
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    PathBuf::from(home)
        .join(".local")
        .join("share")
        .join("runa")
}

/// Local whisper ggml store.
pub fn whisper_dir() -> PathBuf {
    data_dir().join("models").join("whisper")
}

#[derive(Debug, Error)]
pub enum AsrError {
    #[error("unknown whisper model {0:?} (want base or large-v3-turbo)")]
    UnknownModel(String),
    #[error("whisper model missing at {0} — run with pull, or `runa media transcribe`")]
    ModelMissing(String),
    #[error("download {0}: {1}")]
    Download(String, String),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("whisper: {0}")]
    Whisper(String),
    #[error("parakeet: {0}")]
    Parakeet(String),
}

#[derive(Debug, Clone)]
pub struct AsrOptions {
    pub kind: WhisperKind,
    /// ISO-639-1, `"auto"`, or `None` for auto-detect.
    pub language: Option<String>,
    pub n_threads: u32,
    pub use_gpu: bool,
    /// Download missing ggml files (default true for CLI, false for tests).
    pub pull: bool,
    /// Energy / Silero VAD. When false, the whole clip is one chunk.
    pub vad: bool,
}

impl Default for AsrOptions {
    fn default() -> Self {
        Self {
            kind: WhisperKind::Base,
            language: None,
            n_threads: std::thread::available_parallelism()
                .map(|n| n.get() as u32)
                .unwrap_or(4)
                .clamp(1, 8),
            use_gpu: false,
            pull: true,
            vad: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct AsrSegment {
    pub start_ms: u64,
    pub end_ms: u64,
    pub text: String,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct Transcript {
    pub text: String,
    pub language: Option<String>,
    pub segments: Vec<AsrSegment>,
    pub model: String,
}

/// One speech span in samples (inclusive start, exclusive end).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VadSpan {
    pub start: usize,
    pub end: usize,
}

/// Energy VAD: 30 ms frames, merge gaps shorter than 300 ms, pad 150 ms.
pub fn energy_vad(samples: &[f32], sample_rate: u32) -> Vec<VadSpan> {
    if samples.is_empty() {
        return Vec::new();
    }
    let sr = sample_rate.max(1) as usize;
    let frame = (sr * 30 / 1000).max(1);
    let pad = sr * 150 / 1000;
    let merge = sr * 300 / 1000;
    let n_frames = samples.len().div_ceil(frame);
    let mut rms = Vec::with_capacity(n_frames);
    for i in 0..n_frames {
        let a = i * frame;
        let b = (a + frame).min(samples.len());
        let e = samples[a..b].iter().map(|s| s * s).sum::<f32>() / (b - a) as f32;
        rms.push(e.sqrt());
    }
    let peak = rms.iter().copied().fold(0.0f32, f32::max);
    let thresh = (peak * 0.15).max(0.01);
    let mut speech = vec![false; n_frames];
    for (i, &e) in rms.iter().enumerate() {
        speech[i] = e >= thresh;
    }
    let mut spans: Vec<VadSpan> = Vec::new();
    let mut i = 0;
    while i < n_frames {
        if !speech[i] {
            i += 1;
            continue;
        }
        let start_f = i;
        while i < n_frames && speech[i] {
            i += 1;
        }
        let mut start = start_f * frame;
        let mut end = (i * frame).min(samples.len());
        start = start.saturating_sub(pad);
        end = (end + pad).min(samples.len());
        if let Some(last) = spans.last_mut()
            && start <= last.end + merge
        {
            last.end = end;
            continue;
        }
        spans.push(VadSpan { start, end });
    }
    if spans.is_empty() {
        // All below threshold — keep the whole clip (tones / quiet speech).
        vec![VadSpan {
            start: 0,
            end: samples.len(),
        }]
    } else {
        spans
    }
}

/// Levenshtein WER on whitespace-split, lowercased tokens (punctuation stripped).
pub fn word_error_rate(hypothesis: &str, reference: &str) -> f64 {
    let hyp = tokens(hypothesis);
    let refer = tokens(reference);
    if refer.is_empty() {
        return if hyp.is_empty() { 0.0 } else { 1.0 };
    }
    let dist = levenshtein(&hyp, &refer);
    dist as f64 / refer.len() as f64
}

fn tokens(s: &str) -> Vec<String> {
    s.split_whitespace()
        .map(|w| {
            w.chars()
                .filter(|c| c.is_alphanumeric() || *c == '\'')
                .flat_map(|c| c.to_lowercase())
                .collect::<String>()
        })
        .filter(|w| !w.is_empty())
        .collect()
}

fn levenshtein(a: &[String], b: &[String]) -> usize {
    let n = a.len();
    let m = b.len();
    let mut prev: Vec<usize> = (0..=m).collect();
    let mut cur = vec![0; m + 1];
    for i in 1..=n {
        cur[0] = i;
        for j in 1..=m {
            let cost = usize::from(a[i - 1] != b[j - 1]);
            cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[m]
}

/// HuggingFace resolve URL for a ggml file (tests + pull).
pub fn whisper_hf_url(file: &str) -> String {
    hf_resolve_url(file)
}

/// Ensure `kind` (and Silero VAD) exist on disk. No-op when already present.
pub fn ensure_whisper_model(kind: WhisperKind, pull: bool) -> Result<PathBuf, AsrError> {
    let dir = whisper_dir();
    fs::create_dir_all(&dir)?;
    let dest = dir.join(kind.file_name());
    if dest.is_file() && dest.metadata()?.len() > 1_000_000 {
        return Ok(dest);
    }
    if !pull {
        return Err(AsrError::ModelMissing(dest.display().to_string()));
    }
    download_file(&kind.hf_url(), &dest)?;
    let vad = dir.join(SILERO_FILE);
    if !vad.is_file() {
        let _ = download_file(&hf_resolve_url(SILERO_FILE), &vad);
    }
    Ok(dest)
}

fn download_file(url: &str, dest: &Path) -> Result<(), AsrError> {
    let part = dest.with_extension("bin.part");
    let mut resp = reqwest::blocking::Client::builder()
        .user_agent("runa/0.1")
        .build()
        .map_err(|e| AsrError::Download(url.into(), e.to_string()))?
        .get(url)
        .send()
        .and_then(|r| r.error_for_status())
        .map_err(|e| AsrError::Download(url.into(), e.to_string()))?;
    let mut f = File::create(&part)?;
    resp.copy_to(&mut f)
        .map_err(|e| AsrError::Download(url.into(), e.to_string()))?;
    f.flush()?;
    drop(f);
    fs::rename(&part, dest)?;
    if dest.metadata()?.len() < 1_000 {
        let _ = fs::remove_file(dest);
        return Err(AsrError::Download(url.into(), "empty download".into()));
    }
    Ok(())
}

fn quiet_whisper_logs() {
    WHISPER_LOG.call_once(|| {
        whisper_rs::install_logging_hooks();
    });
}

/// Load a whisper context. Caller drops it to free weights (D17: no global cache).
pub struct AsrEngine {
    ctx: WhisperContext,
    kind: WhisperKind,
    vad_path: Option<PathBuf>,
}

impl AsrEngine {
    pub fn load(opts: &AsrOptions) -> Result<Self, AsrError> {
        quiet_whisper_logs();
        let path = ensure_whisper_model(opts.kind, opts.pull)?;
        let params = WhisperContextParameters {
            use_gpu: opts.use_gpu,
            ..Default::default()
        };
        let ctx = WhisperContext::new_with_params(&path, params)
            .map_err(|e| AsrError::Whisper(e.to_string()))?;
        let vad_path = whisper_dir().join(SILERO_FILE);
        let vad_path = vad_path.is_file().then_some(vad_path);
        Ok(Self {
            ctx,
            kind: opts.kind,
            vad_path,
        })
    }

    pub fn transcribe(&self, samples: &[f32], opts: &AsrOptions) -> Result<Transcript, AsrError> {
        if samples.is_empty() {
            return Ok(Transcript {
                text: String::new(),
                language: None,
                segments: Vec::new(),
                model: format!("whisper:{}", self.kind.file_name()),
            });
        }
        let chunks = if opts.vad {
            energy_vad(samples, TARGET_SAMPLE_RATE)
        } else {
            vec![VadSpan {
                start: 0,
                end: samples.len(),
            }]
        };
        let mut state = self
            .ctx
            .create_state()
            .map_err(|e| AsrError::Whisper(e.to_string()))?;
        let mut all = Vec::new();
        let mut lang = None;
        for span in chunks {
            let pcm = &samples[span.start..span.end];
            if pcm.len() < TARGET_SAMPLE_RATE as usize / 20 {
                continue;
            }
            let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
            params.set_n_threads(opts.n_threads as i32);
            params.set_print_special(false);
            params.set_print_progress(false);
            params.set_print_realtime(false);
            params.set_print_timestamps(false);
            params.set_suppress_blank(true);
            params.set_no_speech_thold(0.6);
            let lang_ref = opts.language.as_deref();
            match lang_ref {
                // P10.3: whisper.cpp auto-detects AND decodes when the
                // language is `"auto"` (or unset) — but the
                // `detect_language` flag means "detect ONLY, return
                // without decoding" (`whisper.cpp/src/whisper.cpp`,
                // `whisper_full`: `if (params.detect_language)
                // return 0;`). Setting both produced a correctly
                // detected `lang` with an empty transcript. So on auto:
                // set the language, never the flag.
                None | Some("auto") => {
                    params.set_language(Some("auto"));
                }
                Some(code) => params.set_language(Some(code)),
            }
            if opts.vad
                && let Some(ref vad) = self.vad_path
            {
                let p = vad.to_string_lossy();
                params.set_vad_model_path(Some(p.as_ref()));
                params.enable_vad(true);
            }
            state
                .full(params, pcm)
                .map_err(|e| AsrError::Whisper(e.to_string()))?;
            if lang.is_none() {
                let id = state.full_lang_id_from_state();
                if id >= 0 {
                    lang = get_lang_str(id).map(str::to_string);
                }
            }
            let n = state.full_n_segments();
            let offset_cs = (span.start as u64) * 100 / u64::from(TARGET_SAMPLE_RATE);
            for i in 0..n {
                let Some(seg) = state.get_segment(i) else {
                    continue;
                };
                let text = seg
                    .to_str_lossy()
                    .map_err(|e| AsrError::Whisper(e.to_string()))?;
                let text = text.trim();
                if text.is_empty() {
                    continue;
                }
                all.push(AsrSegment {
                    start_ms: (offset_cs + seg.start_timestamp() as u64) * 10,
                    end_ms: (offset_cs + seg.end_timestamp() as u64) * 10,
                    text: text.to_string(),
                });
            }
        }
        let text = all
            .iter()
            .map(|s| s.text.as_str())
            .collect::<Vec<_>>()
            .join(" ")
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");
        Ok(Transcript {
            text,
            language: lang,
            segments: all,
            model: format!("whisper:{}", self.kind.file_name()),
        })
    }
}

/// Decode `path` and transcribe with auto-pull.
pub fn transcribe_file(path: &Path, opts: &AsrOptions) -> Result<Transcript, AsrError> {
    let decoded = crate::audio::decode_audio(path).map_err(|e| AsrError::Whisper(e.to_string()))?;
    let engine = AsrEngine::load(opts)?;
    engine.transcribe(&decoded.samples, opts)
}

/// Parakeet-TDT via official `sherpa-onnx` (feature `parakeet`).
///
/// The archived `sherpa-rs` crate is not used. Enable `--features parakeet`
/// and pass a directory with encoder/decoder/joiner/tokens; until that
/// feature is wired the call is an explicit error (D12: no silent fallback).
pub fn transcribe_parakeet(samples: &[f32], model_dir: &Path) -> Result<Transcript, AsrError> {
    let _ = samples;
    Err(AsrError::Parakeet(format!(
        "rebuild runa-media with --features parakeet (official sherpa-onnx; not sherpa-rs); model dir {}",
        model_dir.display()
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_whisper_kind() {
        assert_eq!(WhisperKind::parse("base").unwrap(), WhisperKind::Base);
        assert_eq!(
            WhisperKind::parse("whisper:large-v3-turbo").unwrap(),
            WhisperKind::LargeV3Turbo
        );
        assert!(WhisperKind::parse("tiny").is_err());
    }

    #[test]
    fn hf_urls_point_at_ggml_files() {
        assert!(WhisperKind::Base.hf_url().ends_with("ggml-base.bin"));
        assert!(
            WhisperKind::LargeV3Turbo
                .hf_url()
                .contains("ggml-large-v3-turbo.bin")
        );
        assert!(whisper_hf_url(SILERO_FILE).contains(SILERO_FILE));
    }

    #[test]
    fn energy_vad_keeps_tone_drops_leading_silence_shape() {
        let sr = TARGET_SAMPLE_RATE;
        let mut pcm = vec![0.0f32; sr as usize]; // 1 s silence
        for i in 0..sr as usize / 2 {
            pcm.push((i as f32 * 0.1).sin() * 0.5);
        }
        pcm.extend(std::iter::repeat_n(0.0, sr as usize / 4));
        let spans = energy_vad(&pcm, sr);
        assert!(!spans.is_empty());
        assert!(spans[0].end > spans[0].start);
        // Speech should not start at sample 0 (leading silence).
        assert!(spans[0].start < pcm.len());
    }

    #[test]
    fn wer_exact_and_substitution() {
        assert_eq!(word_error_rate("hello world", "hello world"), 0.0);
        assert_eq!(word_error_rate("hello word", "hello world"), 0.5);
        assert_eq!(word_error_rate("Hello, World!", "hello world"), 0.0);
        assert_eq!(word_error_rate("a b c", ""), 1.0);
        assert_eq!(word_error_rate("", ""), 0.0);
    }

    #[test]
    fn missing_model_without_pull_is_error() {
        let prev = std::env::var("XDG_DATA_HOME").ok();
        let tmp = std::env::temp_dir().join(format!("runa-asr-{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).unwrap();
        unsafe { std::env::set_var("XDG_DATA_HOME", tmp.to_str().unwrap()) };
        let err = ensure_whisper_model(WhisperKind::Base, false).unwrap_err();
        match prev {
            Some(v) => unsafe { std::env::set_var("XDG_DATA_HOME", v) },
            None => unsafe { std::env::remove_var("XDG_DATA_HOME") },
        }
        assert!(matches!(err, AsrError::ModelMissing(_)), "{err:?}");
    }

    #[test]
    fn parakeet_without_feature_is_explicit() {
        let err = transcribe_parakeet(&[], Path::new("/nope")).unwrap_err();
        assert!(matches!(err, AsrError::Parakeet(_)));
    }

    #[test]
    fn transcribe_skips_without_ggml() {
        if whisper_dir().join(WhisperKind::Base.file_name()).is_file()
            || std::env::var("RUNA_WHISPER").ok().as_deref() == Some("1")
        {
            return;
        }
        let opts = AsrOptions {
            pull: false,
            ..AsrOptions::default()
        };
        let err = AsrEngine::load(&opts).err().expect("missing ggml");
        assert!(matches!(err, AsrError::ModelMissing(_)), "{err:?}");
    }

    #[test]
    fn auto_detect_decodes_like_explicit() {
        // P10.3 regression: whisper.cpp's `detect_language` flag means
        // "detect ONLY, return without decoding", so `--lang auto` used
        // to report a detected `lang` with an empty transcript.
        // Live-gated: needs ggml-base.bin (dev machines have it cached,
        // CI does not — same skip pattern as above).
        if !whisper_dir().join(WhisperKind::Base.file_name()).is_file() {
            return;
        }
        let clip = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/audio/clip-05-340hz.wav");
        let decoded = crate::audio::decode_audio(&clip).expect("sine fixture decodes");
        let engine = AsrEngine::load(&AsrOptions {
            pull: false,
            ..AsrOptions::default()
        })
        .expect("cached base model loads");
        let opts = |lang: Option<&str>| AsrOptions {
            language: lang.map(str::to_owned),
            pull: false,
            vad: false,
            ..AsrOptions::default()
        };
        let explicit = engine
            .transcribe(&decoded.samples, &opts(Some("en")))
            .expect("explicit en decodes");
        for auto in [opts(None), opts(Some("auto"))] {
            let t = engine
                .transcribe(&decoded.samples, &auto)
                .expect("auto decodes");
            assert_eq!(t.language.as_deref(), Some("en"), "{t:?}");
            assert!(
                !t.text.is_empty(),
                "auto must decode, not just detect: {t:?}"
            );
            assert_eq!(t.text, explicit.text, "auto must match explicit en");
        }
    }
}
