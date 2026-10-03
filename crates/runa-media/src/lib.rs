// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! runa-media — media pipeline in Rust, encoders in C (plan D8): audio
//! decode (`symphonia`/`hound`) → f32 mono 16 kHz (`rubato`), video via
//! `ffmpeg-sidecar` → sampled frames, ASR bridge through `whisper-rs`.
//!
//! P4.1 implements audio decode + probe. whisper-rs pin: docs/versions.md.

mod asr;
mod audio;
mod preprocess;
mod route;
mod video;

pub use asr::{
    AsrEngine, AsrError, AsrOptions, AsrSegment, Transcript, VadSpan, WhisperKind, data_dir,
    energy_vad, ensure_whisper_model, transcribe_file, transcribe_parakeet, whisper_dir,
    whisper_hf_url, word_error_rate,
};
pub use audio::{
    AudioError, AudioProbe, DecodedAudio, TARGET_SAMPLE_RATE, decode_audio, pcm_sha256,
    pcm_to_wav_bytes, resample_mono, wav_base64,
};
pub use preprocess::{normalize_rgb, normalize_rgb_scalar, normalize_rgb_simd, patchify_rgb};
pub use route::{
    AudioBackend, AudioPlan, AudioRoutePref, openai_audio_capable, select_audio_route,
};
pub use video::{
    Frame, SampledVideo, VideoError, VideoOpts, histogram_l1, resize_rgb, sample_video,
    select_and_resize,
};
