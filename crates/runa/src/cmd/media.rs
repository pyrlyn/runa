// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! `runa media`: probe, video sampling and transcription.

use std::path::Path;

pub(super) fn cmd_media_probe(path: &Path, json: bool) -> Result<(), String> {
    let decoded = runa_media::decode_audio(path).map_err(|e| e.to_string())?;
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&decoded.probe).map_err(|e| e.to_string())?
        );
    } else {
        println!("{}", decoded.probe);
    }
    Ok(())
}

pub(super) fn cmd_media_video(
    path: &Path,
    fps: f32,
    max_frames: usize,
    json: bool,
) -> Result<(), String> {
    let opts = runa_media::VideoOpts {
        fps,
        max_frames,
        ..runa_media::VideoOpts::default()
    };
    let sampled = runa_media::sample_video(path, &opts).map_err(|e| e.to_string())?;
    if json {
        let times: Vec<f32> = sampled.frames.iter().map(|f| f.t_sec).collect();
        println!(
            "{}",
            serde_json::json!({
                "n_frames": sampled.frames.len(),
                "t_sec": times,
                "audio": sampled.audio.is_some(),
            })
        );
    } else {
        println!("frames: {}", sampled.frames.len());
        for f in &sampled.frames {
            println!("  [t={:.1}s] {}x{}", f.t_sec, f.width, f.height);
        }
        if sampled.audio.is_some() {
            println!("audio: yes");
        }
    }
    Ok(())
}

pub(super) fn cmd_media_transcribe(
    path: &Path,
    model: &str,
    lang: &str,
    json: bool,
    no_pull: bool,
) -> Result<(), String> {
    let kind = runa_media::WhisperKind::parse(model).map_err(|e| e.to_string())?;
    let opts = runa_media::AsrOptions {
        kind,
        pull: !no_pull,
        language: match lang {
            "auto" | "" => None,
            other => Some(other.to_string()),
        },
        ..runa_media::AsrOptions::default()
    };
    let t = runa_media::transcribe_file(path, &opts).map_err(|e| e.to_string())?;
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&t).map_err(|e| e.to_string())?
        );
    } else {
        if let Some(lang) = &t.language {
            eprintln!("language: {lang}");
        }
        println!("{}", t.text);
    }
    Ok(())
}
