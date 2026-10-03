// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! Sample video frames (P4.5): uniform + scene-change, cap `max_frames`.
//!
//! Frame extraction uses `ffmpeg-sidecar` when ffmpeg is on PATH (or
//! auto-downloaded). Selection/resize are pure Rust so tests do not need
//! ffmpeg.

use std::path::Path;
use std::process::{Command, Stdio};

use crate::audio::{DecodedAudio, TARGET_SAMPLE_RATE, decode_audio};

/// Defaults from the plan: 1 fps, at most 32 frames.
pub const DEFAULT_FPS: f32 = 1.0;
pub const DEFAULT_MAX_FRAMES: usize = 32;

/// One RGB8 frame plus its timestamp.
#[derive(Debug, Clone, PartialEq)]
pub struct Frame {
    pub t_sec: f32,
    pub width: u32,
    pub height: u32,
    pub rgb: Vec<u8>,
}

/// Sampled frames (and optional audio track).
#[derive(Debug, Clone)]
pub struct SampledVideo {
    pub frames: Vec<Frame>,
    pub audio: Option<DecodedAudio>,
}

#[derive(Debug, Clone)]
pub struct VideoOpts {
    pub fps: f32,
    pub max_frames: usize,
    pub image_size: u32,
    /// Histogram L1 distance (0..=2) that counts as a scene cut.
    pub scene_thresh: f32,
}

impl Default for VideoOpts {
    fn default() -> Self {
        VideoOpts {
            fps: DEFAULT_FPS,
            max_frames: DEFAULT_MAX_FRAMES,
            image_size: 336,
            scene_thresh: 0.35,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum VideoError {
    #[error("ffmpeg: {0}")]
    Ffmpeg(String),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("audio: {0}")]
    Audio(#[from] crate::audio::AudioError),
}

/// Pick up to `max_frames` indices: scene cuts first, then uniform fill.
pub fn pick_indices(n: usize, max_frames: usize, distances: &[f32], thresh: f32) -> Vec<usize> {
    if n == 0 || max_frames == 0 {
        return Vec::new();
    }
    let max_frames = max_frames.min(n);
    let mut chosen = vec![false; n];
    chosen[0] = true;
    if n > 1 {
        chosen[n - 1] = true;
    }
    for (i, &d) in distances.iter().enumerate() {
        if d >= thresh {
            let idx = i + 1;
            if idx < n {
                chosen[idx] = true;
            }
        }
    }
    let mut out: Vec<usize> = chosen
        .iter()
        .enumerate()
        .filter(|(_, c)| **c)
        .map(|(i, _)| i)
        .collect();
    if out.len() > max_frames {
        // Keep first/last + evenly spaced from the rest.
        let mut keep = vec![out[0]];
        if out.len() > 1 {
            keep.push(*out.last().unwrap());
        }
        let inner = out.len().saturating_sub(2);
        let need = max_frames.saturating_sub(keep.len());
        if inner > 0 && need > 0 {
            for k in 0..need {
                let j = 1 + k * inner / need;
                keep.push(out[j.min(out.len() - 2)]);
            }
        }
        keep.sort_unstable();
        keep.dedup();
        out = keep;
        if out.len() > max_frames {
            out.truncate(max_frames);
        }
        return out;
    }
    while out.len() < max_frames {
        let k = out.len();
        let idx = k * (n - 1) / max_frames.max(1);
        if !out.contains(&idx) {
            out.push(idx);
        } else {
            let mut added = false;
            for i in 0..n {
                if !out.contains(&i) {
                    out.push(i);
                    added = true;
                    break;
                }
            }
            if !added {
                break;
            }
        }
    }
    out.sort_unstable();
    out
}

/// 4×4×4 RGB histogram, L1 distance in `0..=2`.
pub fn histogram_l1(a: &[u8], b: &[u8]) -> f32 {
    let ha = hist(a);
    let hb = hist(b);
    let mut s = 0.0f32;
    for i in 0..64 {
        s += (ha[i] - hb[i]).abs();
    }
    s
}

fn hist(rgb: &[u8]) -> [f32; 64] {
    let mut h = [0.0f32; 64];
    let n = (rgb.len() / 3).max(1) as f32;
    for px in rgb.as_chunks::<3>().0 {
        let r = (px[0] as usize) / 64;
        let g = (px[1] as usize) / 64;
        let b = (px[2] as usize) / 64;
        h[r * 16 + g * 4 + b] += 1.0;
    }
    for v in &mut h {
        *v /= n;
    }
    h
}

/// Nearest-neighbor resize, RGB8.
pub fn resize_rgb(src: &[u8], w: u32, h: u32, nw: u32, nh: u32) -> Vec<u8> {
    if w == 0 || h == 0 || nw == 0 || nh == 0 {
        return Vec::new();
    }
    let mut out = vec![0u8; (nw * nh * 3) as usize];
    for y in 0..nh {
        let sy = y * h / nh;
        for x in 0..nw {
            let sx = x * w / nw;
            let si = ((sy * w + sx) * 3) as usize;
            let di = ((y * nw + x) * 3) as usize;
            out[di..di + 3].copy_from_slice(&src[si..si + 3]);
        }
    }
    out
}

/// Apply scene+uniform selection and resize.
pub fn select_and_resize(frames: Vec<Frame>, opts: &VideoOpts) -> Vec<Frame> {
    if frames.is_empty() {
        return frames;
    }
    let mut distances = Vec::with_capacity(frames.len().saturating_sub(1));
    for w in frames.windows(2) {
        distances.push(histogram_l1(&w[0].rgb, &w[1].rgb));
    }
    let idx = pick_indices(frames.len(), opts.max_frames, &distances, opts.scene_thresh);
    frames
        .into_iter()
        .enumerate()
        .filter(|(i, _)| idx.contains(i))
        .map(|(_, mut f)| {
            if f.width != opts.image_size || f.height != opts.image_size {
                f.rgb = resize_rgb(&f.rgb, f.width, f.height, opts.image_size, opts.image_size);
                f.width = opts.image_size;
                f.height = opts.image_size;
            }
            f
        })
        .collect()
}

/// Extract frames with ffmpeg (`fps` filter) then select/resize.
pub fn sample_video(path: &Path, opts: &VideoOpts) -> Result<SampledVideo, VideoError> {
    let raw = ffmpeg_raw_frames(path, opts.fps)?;
    let frames = select_and_resize(raw, opts);
    let audio = ffmpeg_audio_wav(path)
        .ok()
        .and_then(|p| decode_audio(&p).ok());
    Ok(SampledVideo { frames, audio })
}

fn ffmpeg_bin() -> Result<String, VideoError> {
    if let Ok(p) = std::env::var("FFMPEG_PATH")
        && Path::new(&p).is_file()
    {
        return Ok(p);
    }
    if ffmpeg_on_path() {
        return Ok("ffmpeg".into());
    }
    ffmpeg_sidecar::download::auto_download().map_err(|e| VideoError::Ffmpeg(e.to_string()))?;
    if ffmpeg_on_path() {
        Ok("ffmpeg".into())
    } else {
        Err(VideoError::Ffmpeg("ffmpeg not installed".into()))
    }
}

fn ffmpeg_on_path() -> bool {
    Command::new("ffmpeg")
        .arg("-version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn ffmpeg_raw_frames(path: &Path, fps: f32) -> Result<Vec<Frame>, VideoError> {
    let ff = ffmpeg_bin()?;
    let fps = fps.max(0.1);
    // Probe size via a tiny scale so we know width/height from ffprobe-less parse:
    // we force 320x240 then resize in Rust to `image_size`.
    let w = 320u32;
    let h = 240u32;
    let output = Command::new(&ff)
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            &path.display().to_string(),
            "-vf",
            &format!("fps={fps},scale={w}:{h}"),
            "-pix_fmt",
            "rgb24",
            "-f",
            "rawvideo",
            "-",
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()?;
    if !output.status.success() {
        return Err(VideoError::Ffmpeg(
            String::from_utf8_lossy(&output.stderr).trim().to_string(),
        ));
    }
    let stride = (w * h * 3) as usize;
    if stride == 0 || output.stdout.len() < stride {
        return Err(VideoError::Ffmpeg("no frames decoded".into()));
    }
    let n = output.stdout.len() / stride;
    let mut frames = Vec::with_capacity(n);
    for i in 0..n {
        let off = i * stride;
        frames.push(Frame {
            t_sec: i as f32 / fps,
            width: w,
            height: h,
            rgb: output.stdout[off..off + stride].to_vec(),
        });
    }
    Ok(frames)
}

fn ffmpeg_audio_wav(path: &Path) -> Result<std::path::PathBuf, VideoError> {
    let ff = ffmpeg_bin()?;
    let dir = std::env::temp_dir().join(format!("runa-vid-{}", std::process::id()));
    std::fs::create_dir_all(&dir)?;
    let wav = dir.join("track.wav");
    let output = Command::new(&ff)
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            &path.display().to_string(),
            "-vn",
            "-ac",
            "1",
            "-ar",
            &TARGET_SAMPLE_RATE.to_string(),
            "-f",
            "wav",
            &wav.display().to_string(),
        ])
        .output()?;
    if !output.status.success() || !wav.is_file() {
        return Err(VideoError::Ffmpeg(
            String::from_utf8_lossy(&output.stderr).trim().to_string(),
        ));
    }
    Ok(wav)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(w: u32, h: u32, r: u8, g: u8, b: u8) -> Frame {
        let mut rgb = Vec::with_capacity((w * h * 3) as usize);
        for _ in 0..w * h {
            rgb.extend_from_slice(&[r, g, b]);
        }
        Frame {
            t_sec: 0.0,
            width: w,
            height: h,
            rgb,
        }
    }

    #[test]
    fn histogram_detects_scene_cut() {
        let a = solid(8, 8, 0, 0, 0).rgb;
        let b = solid(8, 8, 255, 255, 255).rgb;
        let same = histogram_l1(&a, &a);
        let cut = histogram_l1(&a, &b);
        assert!(same < 0.01, "{same}");
        assert!(cut > 1.0, "{cut}");
    }

    #[test]
    fn normalize_rgb_maps_black_white() {
        let rgb = [0u8, 0, 0, 255, 255, 255];
        let mut out = [0.0f32; 6];
        crate::normalize_rgb(&rgb, &mut out);
        assert!((out[0] + 1.0).abs() < 1e-5);
        assert!((out[3] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn pick_caps_at_max_and_keeps_ends() {
        let n = 40;
        let dist = vec![0.0; n - 1];
        let idx = pick_indices(n, 32, &dist, 0.9);
        assert!(idx.len() <= 32);
        assert_eq!(idx[0], 0);
        assert_eq!(*idx.last().unwrap(), n - 1);
    }

    #[test]
    fn pick_includes_scene_index() {
        let mut dist = vec![0.0; 9];
        dist[4] = 1.0; // cut before frame 5
        let idx = pick_indices(10, 32, &dist, 0.5);
        assert!(idx.contains(&5), "{idx:?}");
    }

    #[test]
    fn thirty_seconds_at_1fps_caps_32() {
        // 30 s @ 1 fps → 30 frames, already ≤ 32.
        let frames: Vec<_> = (0..30)
            .map(|i| {
                let mut f = solid(4, 4, (i * 8) as u8, 40, 200);
                f.t_sec = i as f32;
                f
            })
            .collect();
        let opts = VideoOpts {
            max_frames: 32,
            image_size: 8,
            ..VideoOpts::default()
        };
        let out = select_and_resize(frames, &opts);
        assert!(out.len() <= 32);
        assert_eq!(out[0].width, 8);
        assert_eq!(out[0].height, 8);
        assert_eq!(out[0].rgb.len(), 8 * 8 * 3);
    }

    #[test]
    fn ffmpeg_placeholder_skipped_or_errors() {
        let path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/video/clip-01-5s.mp4");
        if !path.is_file() {
            return;
        }
        if !ffmpeg_on_path() {
            return;
        }
        // Placeholder ftyp headers are not real media; ffmpeg should fail.
        let err = sample_video(&path, &VideoOpts::default());
        assert!(err.is_err(), "placeholder mp4 must not yield frames");
    }
}
