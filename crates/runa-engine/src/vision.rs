// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! Native vision through mtmd (plan P4.6).
//!
//! Still images (`--image`) and sampled video frames (`--video`) become mtmd
//! bitmaps. Video timestamps are written as `[t=12.0s]` next to each media
//! marker so the model can tell frames apart.

use std::path::PathBuf;

use crate::generate::ChatMessage;
use crate::load::{EngineError, LoadedModel};

/// One image or video frame for [`LoadedModel::eval_vision_prompt`].
#[derive(Debug, Clone)]
pub struct VisionFrame {
    /// Presentation time for video frames (`--video`). `None` for `--image`.
    pub t_sec: Option<f32>,
    pub source: VisionSource,
}

/// Pixel source for a [`VisionFrame`].
#[derive(Debug, Clone)]
pub enum VisionSource {
    /// Decode via libmtmd (`MtmdBitmap::from_file`).
    Path(PathBuf),
    /// RGB8 packed `width * height * 3` (video sampler output).
    Rgb {
        width: u32,
        height: u32,
        rgb: Vec<u8>,
    },
}

/// Append `[t=12.0s] <marker>` (or just the marker) once per frame.
pub fn format_vision_user_text(prompt: &str, times: &[Option<f32>], marker: &str) -> String {
    let mut s = prompt.trim_end().to_string();
    for t in times {
        s.push('\n');
        if let Some(sec) = t {
            s.push_str(&format!("[t={sec:.1}s] "));
        }
        s.push_str(marker);
    }
    s
}

impl LoadedModel {
    /// Whether the loaded mmproj accepts image / video-frame bitmaps.
    pub fn supports_native_vision(&self) -> bool {
        #[cfg(feature = "mtmd")]
        {
            self.mtmd
                .as_ref()
                .is_some_and(llama_cpp_2::mtmd::MtmdContext::support_vision)
        }
        #[cfg(not(feature = "mtmd"))]
        {
            false
        }
    }

    /// Tokenize + eval image/frame bitmaps. Returns `n_past`.
    pub(crate) fn eval_vision_prompt(
        &mut self,
        messages: &[ChatMessage],
        frames: &[VisionFrame],
        add_generation_prompt: bool,
    ) -> Result<i32, EngineError> {
        if frames.is_empty() {
            // Load-bearing: see the identical note on the mtmd gate in
            // media.rs — removing `return` breaks the default build (E0308).
            #[allow(clippy::needless_return)]
            return Err(EngineError::Media("empty --image/--video".into()));
        }
        #[cfg(not(feature = "mtmd"))]
        {
            let _ = (messages, add_generation_prompt);
            // Load-bearing (same cfg-gate shape as media.rs): removing
            // `return` breaks the default-features build (E0308).
            #[allow(clippy::needless_return)]
            return Err(EngineError::Unsupported(
                "--image/--video requires rebuilding with --features mtmd",
            ));
        }
        #[cfg(feature = "mtmd")]
        {
            use llama_cpp_2::mtmd::{MtmdBitmap, MtmdInputText, mtmd_default_marker};

            if self.mtmd.is_none() {
                return Err(EngineError::Media(
                    "--image/--video needs a vision mmproj (--mmproj or a sibling *mmproj*.gguf)"
                        .into(),
                ));
            }
            if !self.mtmd.as_ref().unwrap().support_vision() {
                return Err(EngineError::Media(
                    "mmproj has no vision encoder; use a VL mmproj (Qwen3-VL, SmolVLM, …)".into(),
                ));
            }
            let marker = mtmd_default_marker();
            let times: Vec<Option<f32>> = frames.iter().map(|f| f.t_sec).collect();
            let mut msgs = messages.to_vec();
            if let Some(last) = msgs.last_mut() {
                last.content = format_vision_user_text(&last.content, &times, marker);
            }
            let text = self.render_prompt(&msgs, add_generation_prompt)?;
            let bitmaps = {
                let mtmd = self.mtmd.as_ref().unwrap();
                let mut bitmaps = Vec::with_capacity(frames.len());
                for frame in frames {
                    let bmp = match &frame.source {
                        VisionSource::Path(path) => {
                            let s = path.to_str().ok_or_else(|| {
                                EngineError::Media(format!("non-utf8 image path: {path:?}"))
                            })?;
                            MtmdBitmap::from_file(mtmd, s)
                                .map_err(|e| EngineError::Media(format!("image {path:?}: {e:?}")))?
                        }
                        VisionSource::Rgb { width, height, rgb } => {
                            MtmdBitmap::from_image_data(*width, *height, rgb).map_err(|e| {
                                EngineError::Media(format!("frame rgb {width}x{height}: {e:?}"))
                            })?
                        }
                    };
                    bitmaps.push(bmp);
                }
                bitmaps
            };
            let chunks = {
                let mtmd = self.mtmd.as_ref().unwrap();
                let refs: Vec<&MtmdBitmap> = bitmaps.iter().collect();
                mtmd.tokenize(
                    MtmdInputText {
                        text,
                        add_special: true,
                        parse_special: true,
                    },
                    &refs,
                )
                .map_err(|e| EngineError::Media(format!("mtmd tokenize: {e:?}")))?
            };
            self.reset_context()?;
            let n_batch = self.config().n_batch as i32;
            let n_past = chunks
                .eval_chunks(
                    self.mtmd.as_ref().unwrap(),
                    self.context(),
                    0,
                    0,
                    n_batch,
                    true,
                )
                .map_err(|e| EngineError::Media(format!("mtmd eval: {e:?}")))?;
            Ok(n_past)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn still_images_get_markers_without_timestamps() {
        let s = format_vision_user_text("what is this?", &[None, None], "<media>");
        assert_eq!(s, "what is this?\n<media>\n<media>");
    }

    #[test]
    fn video_frames_get_timestamp_markers() {
        let s = format_vision_user_text("clip", &[Some(0.0), Some(12.0)], "<m>");
        assert_eq!(s, "clip\n[t=0.0s] <m>\n[t=12.0s] <m>");
    }
}
