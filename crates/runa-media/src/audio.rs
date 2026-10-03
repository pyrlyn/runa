// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! Decode audio to f32 mono 16 kHz (plan D8, P4.1).

use std::fs::File;
use std::path::Path;

use sha2::{Digest, Sha256};
use thiserror::Error;

/// Target PCM for ASR / mtmd audio chunks.
pub const TARGET_SAMPLE_RATE: u32 = 16_000;

/// One decoded clip plus probe metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct DecodedAudio {
    /// Mono, [`TARGET_SAMPLE_RATE`] Hz, in `-1.0..1.0`.
    pub samples: Vec<f32>,
    pub probe: AudioProbe,
}

/// Stable, printable summary of a decode (no sample blob).
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct AudioProbe {
    pub path: String,
    pub codec: String,
    pub src_sample_rate: u32,
    pub src_channels: u16,
    pub duration_secs: f64,
    pub pcm_sample_rate: u32,
    pub pcm_samples: usize,
    pub pcm_sha256: String,
}

impl std::fmt::Display for AudioProbe {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "file: {}", self.path)?;
        writeln!(f, "codec: {}", self.codec)?;
        writeln!(f, "src_rate: {}", self.src_sample_rate)?;
        writeln!(f, "src_channels: {}", self.src_channels)?;
        writeln!(f, "duration_s: {:.6}", self.duration_secs)?;
        writeln!(f, "pcm_rate: {}", self.pcm_sample_rate)?;
        writeln!(f, "pcm_samples: {}", self.pcm_samples)?;
        write!(f, "pcm_sha256: {}", self.pcm_sha256)
    }
}

/// Decode failures (missing file, unsupported codec, resample).
#[derive(Debug, Error)]
pub enum AudioError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("decode: {0}")]
    Decode(String),
    #[error("resample: {0}")]
    Resample(String),
    #[error("no audio track in {0}")]
    NoAudio(String),
}

/// Decode `path` (wav/flac/ogg/mp3/aac) to mono 16 kHz f32 PCM.
pub fn decode_audio(path: &Path) -> Result<DecodedAudio, AudioError> {
    let (interleaved, spec) = decode_interleaved_f32(path)?;
    if spec.channels == 0 {
        return Err(AudioError::Decode("zero channels".into()));
    }
    let mono = downmix_mono(&interleaved, spec.channels);
    let samples = if spec.rate == TARGET_SAMPLE_RATE {
        mono
    } else {
        resample_mono(&mono, spec.rate, TARGET_SAMPLE_RATE)?
    };
    let pcm_sha256 = pcm_sha256(&samples);
    let n_src_frames = interleaved.len() / spec.channels as usize;
    let duration_secs = if spec.rate == 0 {
        0.0
    } else {
        n_src_frames as f64 / f64::from(spec.rate)
    };
    Ok(DecodedAudio {
        probe: AudioProbe {
            path: path.display().to_string(),
            codec: spec.codec,
            src_sample_rate: spec.rate,
            src_channels: spec.channels,
            duration_secs,
            pcm_sample_rate: TARGET_SAMPLE_RATE,
            pcm_samples: samples.len(),
            pcm_sha256,
        },
        samples,
    })
}

/// SHA-256 of little-endian f32 bytes (stable across platforms).
pub fn pcm_sha256(samples: &[f32]) -> String {
    let mut hasher = Sha256::new();
    for s in samples {
        hasher.update(s.to_le_bytes());
    }
    hex(&hasher.finalize())
}

/// 16-bit mono WAV of already-resampled PCM (OpenAI `input_audio`).
pub fn pcm_to_wav_bytes(samples: &[f32], sample_rate: u32) -> Result<Vec<u8>, AudioError> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut buf = Vec::new();
    {
        let mut cursor = std::io::Cursor::new(&mut buf);
        let mut w = hound::WavWriter::new(&mut cursor, spec)
            .map_err(|e| AudioError::Decode(e.to_string()))?;
        for s in samples {
            let v = (s.clamp(-1.0, 1.0) * 32767.0).round() as i16;
            w.write_sample(v)
                .map_err(|e| AudioError::Decode(e.to_string()))?;
        }
        w.finalize()
            .map_err(|e| AudioError::Decode(e.to_string()))?;
    }
    Ok(buf)
}

/// RFC 4648 base64 (no extra crate).
pub fn wav_base64(wav: &[u8]) -> String {
    const T: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::new();
    for chunk in wav.chunks(3) {
        let a = chunk[0] as u32;
        let b = chunk.get(1).copied().unwrap_or(0) as u32;
        let c = chunk.get(2).copied().unwrap_or(0) as u32;
        let n = (a << 16) | (b << 8) | c;
        out.push(T[(n >> 18) as usize] as char);
        out.push(T[((n >> 12) & 63) as usize] as char);
        if chunk.len() > 1 {
            out.push(T[((n >> 6) & 63) as usize] as char);
        } else {
            out.push('=');
        }
        if chunk.len() > 2 {
            out.push(T[(n & 63) as usize] as char);
        } else {
            out.push('=');
        }
    }
    out
}

struct RawSpec {
    rate: u32,
    channels: u16,
    codec: String,
}

fn decode_interleaved_f32(path: &Path) -> Result<(Vec<f32>, RawSpec), AudioError> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    if ext == "wav"
        && let Ok(v) = decode_wav_hound(path)
    {
        return Ok(v);
    }
    decode_symphonia(path)
}

fn decode_wav_hound(path: &Path) -> Result<(Vec<f32>, RawSpec), AudioError> {
    let reader = hound::WavReader::open(path).map_err(|e| AudioError::Decode(e.to_string()))?;
    let spec = reader.spec();
    let codec = format!(
        "pcm_{}_{}",
        match spec.sample_format {
            hound::SampleFormat::Int => "s",
            hound::SampleFormat::Float => "f",
        },
        spec.bits_per_sample
    );
    let channels = spec.channels;
    let rate = spec.sample_rate;
    let samples: Result<Vec<f32>, _> = match spec.sample_format {
        hound::SampleFormat::Int => reader
            .into_samples::<i32>()
            .map(|s| s.map(|v| int_sample_to_f32(v, spec.bits_per_sample)))
            .collect(),
        hound::SampleFormat::Float => reader.into_samples::<f32>().collect(),
    };
    let interleaved = samples.map_err(|e| AudioError::Decode(e.to_string()))?;
    Ok((
        interleaved,
        RawSpec {
            rate,
            channels,
            codec,
        },
    ))
}

fn int_sample_to_f32(s: i32, bits: u16) -> f32 {
    let bits = bits.clamp(1, 31);
    let max = (1u32 << (bits - 1)) as f32;
    (s as f32 / max).clamp(-1.0, 1.0)
}

fn decode_symphonia(path: &Path) -> Result<(Vec<f32>, RawSpec), AudioError> {
    use symphonia::core::codecs::audio::AudioDecoderOptions;
    use symphonia::core::errors::Error as SError;
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::formats::TrackType;
    use symphonia::core::formats::probe::Hint;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;

    let file = File::open(path)?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }
    let mut format = symphonia::default::get_probe()
        .probe(
            &hint,
            mss,
            FormatOptions::default(),
            MetadataOptions::default(),
        )
        .map_err(|e| AudioError::Decode(e.to_string()))?;
    let track = format
        .default_track(TrackType::Audio)
        .ok_or_else(|| AudioError::NoAudio(path.display().to_string()))?
        .clone();
    let track_id = track.id;
    let audio_params = track
        .codec_params
        .as_ref()
        .and_then(|c| c.audio())
        .ok_or_else(|| AudioError::NoAudio(path.display().to_string()))?;
    let mut decoder = symphonia::default::get_codecs()
        .make_audio_decoder(audio_params, &AudioDecoderOptions::default())
        .map_err(|e| AudioError::Decode(e.to_string()))?;

    let mut pcm = Vec::new();
    let mut rate = audio_params.sample_rate.unwrap_or(0);
    let mut channels = audio_params
        .channels
        .as_ref()
        .map(|c| c.count() as u16)
        .unwrap_or(0);
    let codec = format!("{:?}", audio_params.codec);

    loop {
        let packet = match format.next_packet() {
            Ok(Some(p)) => p,
            Ok(None) => break,
            Err(SError::ResetRequired) => {
                decoder.reset();
                continue;
            }
            Err(e) => {
                let msg = e.to_string();
                if msg.contains("end of stream") || msg.contains("eof") {
                    break;
                }
                return Err(AudioError::Decode(msg));
            }
        };
        if packet.track_id != track_id {
            continue;
        }
        match decoder.decode(&packet) {
            Ok(decoded) => {
                rate = decoded.spec().rate();
                channels = decoded.spec().channels().count() as u16;
                let start = pcm.len();
                pcm.resize(start + decoded.samples_interleaved(), 0.0);
                decoded.copy_to_slice_interleaved(&mut pcm[start..]);
            }
            Err(SError::DecodeError(_)) => continue,
            Err(e) => return Err(AudioError::Decode(e.to_string())),
        }
    }
    if pcm.is_empty() {
        return Err(AudioError::Decode(format!(
            "{}: no PCM frames decoded",
            path.display()
        )));
    }
    Ok((
        pcm,
        RawSpec {
            rate,
            channels,
            codec,
        },
    ))
}

fn downmix_mono(interleaved: &[f32], channels: u16) -> Vec<f32> {
    let ch = channels as usize;
    if ch <= 1 {
        return interleaved.to_vec();
    }
    interleaved
        .chunks_exact(ch)
        .map(|frame| frame.iter().sum::<f32>() / ch as f32)
        .collect()
}

pub fn resample_mono(input: &[f32], from: u32, to: u32) -> Result<Vec<f32>, AudioError> {
    use rubato::audioadapter_buffers::direct::InterleavedSlice;
    use rubato::{
        Async, FixedAsync, Resampler, SincInterpolationParameters, SincInterpolationType,
        WindowFunction,
    };
    if from == 0 {
        return Err(AudioError::Resample("source sample rate is 0".into()));
    }
    if input.is_empty() {
        return Ok(Vec::new());
    }
    let ratio = f64::from(to) / f64::from(from);
    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: Some(0.95),
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };
    // Whole-clip resample (rubato 5): the input is fully in memory, so
    // `process_all` handles chunking and delay trimming internally.
    let mut resampler = Async::<f32>::new_sinc(ratio, 2.0, &params, 1024, 1, FixedAsync::Input)
        .map_err(|e| AudioError::Resample(e.to_string()))?;
    let adapter = InterleavedSlice::new(input, 1, input.len())
        .map_err(|e| AudioError::Resample(format!("adapter: {e:?}")))?;
    let rendered = resampler
        .process_all(&adapter, input.len(), None)
        .map_err(|e| AudioError::Resample(e.to_string()))?;
    let mut out = rendered.take_data();
    let expected = (input.len() as f64 * ratio).round() as usize;
    if out.len() > expected + expected / 10 + 64 {
        out.truncate(expected);
    }
    Ok(out)
}

fn hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push(HEX[(b >> 4) as usize] as char);
        s.push(HEX[(b & 0x0f) as usize] as char);
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn fixtures() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/audio")
    }

    #[test]
    fn ten_wav_fixtures_decode_stable_hashes() {
        let dir = fixtures();
        let mut names: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.extension().and_then(|e| e.to_str()) == Some("wav")
                    && p.file_name()
                        .and_then(|n| n.to_str())
                        .is_some_and(|n| n.starts_with("clip-"))
            })
            .collect();
        names.sort();
        assert_eq!(
            names.len(),
            10,
            "expected 10 clip-*.wav fixtures in {dir:?}"
        );
        let mut hashes = Vec::new();
        for path in &names {
            let a = decode_audio(path).unwrap_or_else(|e| panic!("{path:?}: {e}"));
            let b = decode_audio(path).unwrap();
            assert_eq!(a.probe.pcm_sample_rate, TARGET_SAMPLE_RATE);
            assert_eq!(a.probe.src_channels, 1);
            assert_eq!(a.probe.src_sample_rate, 16_000);
            assert_eq!(a.samples.len(), 16_000, "{path:?} should be 1 s @ 16 kHz");
            assert_eq!(
                a.probe.pcm_sha256, b.probe.pcm_sha256,
                "unstable hash {path:?}"
            );
            assert_eq!(a.probe.pcm_sha256.len(), 64);
            hashes.push((
                path.file_name().unwrap().to_string_lossy().into_owned(),
                a.probe.pcm_sha256,
            ));
        }
        // Distinct tones → distinct PCM.
        let uniq: std::collections::HashSet<_> = hashes.iter().map(|(_, h)| h.clone()).collect();
        assert_eq!(uniq.len(), 10, "fixture hashes collided: {hashes:?}");
    }

    #[test]
    fn stereo_8k_resamples_to_mono_16k() {
        let dir = std::env::temp_dir().join(format!("runa-media-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("stereo8k.wav");
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: 8_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut w = hound::WavWriter::create(&path, spec).unwrap();
        for i in 0..8000 {
            let s = (i as f32 * 440.0 * 2.0 * std::f32::consts::PI / 8000.0).sin();
            let v = (s * 16000.0) as i16;
            w.write_sample(v).unwrap();
            w.write_sample(v).unwrap();
        }
        w.finalize().unwrap();
        let decoded = decode_audio(&path).unwrap();
        assert_eq!(decoded.probe.src_channels, 2);
        assert_eq!(decoded.probe.src_sample_rate, 8_000);
        assert_eq!(decoded.probe.pcm_sample_rate, 16_000);
        // ~1 s at 16 kHz; sinc padding may add a few samples.
        assert!(
            decoded.samples.len() > 14_000 && decoded.samples.len() < 18_000,
            "got {} samples",
            decoded.samples.len()
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn missing_file_errors() {
        let err = decode_audio(Path::new("/no/such/clip.wav")).unwrap_err();
        assert!(err.to_string().contains("io") || err.to_string().contains("No such"));
    }

    #[test]
    fn pcm_wav_roundtrip_header_and_b64() {
        let wav = pcm_to_wav_bytes(&[0.0; 16], 16_000).unwrap();
        assert!(wav.starts_with(b"RIFF"));
        assert_eq!(wav_base64(&[0]), "AA==");
    }
}
