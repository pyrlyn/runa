// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! Cloud media payloads (P4.7): images + PDFs, count/size limits, downscale.

use serde_json::{Value, json};

/// Caps shared by OpenAI and Anthropic (plan: auto-downscale).
#[derive(Debug, Clone, Copy)]
pub struct MediaLimits {
    pub max_images: usize,
    pub max_edge: u32,
    pub max_bytes: usize,
}

impl Default for MediaLimits {
    fn default() -> Self {
        MediaLimits {
            max_images: 32,
            max_edge: 1568,
            max_bytes: 4_000_000,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RawImage {
    pub width: u32,
    pub height: u32,
    pub rgb: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct PreparedImage {
    pub width: u32,
    pub height: u32,
    pub rgb: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MediaError {
    TooManyImages { n: usize, max: usize },
    EmptyImage,
    PdfTooLarge { bytes: usize, max: usize },
}

/// Shrink RGB so `max(w,h) ≤ max_edge` and `rgb.len() ≤ max_bytes`.
pub fn downscale(img: &RawImage, limits: &MediaLimits) -> PreparedImage {
    let mut w = img.width.max(1);
    let mut h = img.height.max(1);
    let mut rgb = img.rgb.clone();
    while (w > limits.max_edge || h > limits.max_edge || rgb.len() > limits.max_bytes)
        && (w > 1 || h > 1)
    {
        let nw = (w / 2).max(1);
        let nh = (h / 2).max(1);
        rgb = nn_resize(&rgb, w, h, nw, nh);
        w = nw;
        h = nh;
    }
    PreparedImage {
        width: w,
        height: h,
        rgb,
    }
}

fn nn_resize(src: &[u8], w: u32, h: u32, nw: u32, nh: u32) -> Vec<u8> {
    let mut out = vec![0u8; (nw * nh * 3) as usize];
    for y in 0..nh {
        let sy = y * h / nh;
        for x in 0..nw {
            let sx = x * w / nw;
            let si = ((sy * w + sx) * 3) as usize;
            let di = ((y * nw + x) * 3) as usize;
            if si + 2 < src.len() {
                out[di..di + 3].copy_from_slice(&src[si..si + 3]);
            }
        }
    }
    out
}

pub fn prepare_images(
    images: &[RawImage],
    limits: &MediaLimits,
) -> Result<Vec<PreparedImage>, MediaError> {
    if images.len() > limits.max_images {
        return Err(MediaError::TooManyImages {
            n: images.len(),
            max: limits.max_images,
        });
    }
    let mut out = Vec::with_capacity(images.len());
    for img in images {
        if img.rgb.len() < 3 || img.width == 0 || img.height == 0 {
            return Err(MediaError::EmptyImage);
        }
        out.push(downscale(img, limits));
    }
    Ok(out)
}

pub fn data_url_png_placeholder(img: &PreparedImage) -> String {
    // RGB is not PNG; callers send it as a data URL for mock tests.
    let b64 = b64(&img.rgb);
    format!("data:image/png;base64,{b64}")
}

/// OpenAI `image_url` parts (chat completions).
pub fn openai_image_parts(images: &[PreparedImage]) -> Vec<Value> {
    images
        .iter()
        .map(|img| {
            json!({
                "type": "image_url",
                "image_url": {"url": data_url_png_placeholder(img)}
            })
        })
        .collect()
}

/// Anthropic image content blocks.
pub fn anthropic_image_blocks(images: &[PreparedImage]) -> Vec<Value> {
    images
        .iter()
        .map(|img| {
            json!({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": b64(&img.rgb),
                }
            })
        })
        .collect()
}

/// Anthropic PDF `document` block.
pub fn anthropic_pdf_block(pdf: &[u8], limits: &MediaLimits) -> Result<Value, MediaError> {
    if pdf.len() > limits.max_bytes {
        return Err(MediaError::PdfTooLarge {
            bytes: pdf.len(),
            max: limits.max_bytes,
        });
    }
    Ok(json!({
        "type": "document",
        "source": {
            "type": "base64",
            "media_type": "application/pdf",
            "data": b64(pdf),
        }
    }))
}

/// Video frames are just more images (P4.5 already sampled).
pub fn frames_as_images(
    frames: &[RawImage],
    limits: &MediaLimits,
) -> Result<Vec<PreparedImage>, MediaError> {
    prepare_images(images_from_video(frames), limits)
}

fn images_from_video(frames: &[RawImage]) -> &[RawImage] {
    frames
}

fn b64(bytes: &[u8]) -> String {
    const T: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::new();
    let mut i = 0;
    while i < bytes.len() {
        let b0 = bytes[i];
        let b1 = if i + 1 < bytes.len() { bytes[i + 1] } else { 0 };
        let b2 = if i + 2 < bytes.len() { bytes[i + 2] } else { 0 };
        let n = bytes.len() - i;
        out.push(T[(b0 >> 2) as usize] as char);
        out.push(T[(((b0 & 3) << 4) | (b1 >> 4)) as usize] as char);
        if n >= 2 {
            out.push(T[(((b1 & 15) << 2) | (b2 >> 6)) as usize] as char);
        } else {
            out.push('=');
        }
        if n >= 3 {
            out.push(T[(b2 & 63) as usize] as char);
        } else {
            out.push('=');
        }
        i += 3;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(w: u32, h: u32, c: u8) -> RawImage {
        RawImage {
            width: w,
            height: h,
            rgb: vec![c; (w * h * 3) as usize],
        }
    }

    #[test]
    fn count_limit() {
        let limits = MediaLimits {
            max_images: 2,
            ..MediaLimits::default()
        };
        let imgs = vec![solid(2, 2, 1), solid(2, 2, 2), solid(2, 2, 3)];
        let err = prepare_images(&imgs, &limits).unwrap_err();
        assert!(matches!(err, MediaError::TooManyImages { n: 3, max: 2 }));
    }

    #[test]
    fn auto_downscale_edge() {
        let limits = MediaLimits {
            max_images: 8,
            max_edge: 8,
            max_bytes: 4_000_000,
        };
        let img = solid(64, 32, 9);
        let p = downscale(&img, &limits);
        assert!(p.width <= 8 && p.height <= 8);
        assert_eq!(p.rgb.len(), (p.width * p.height * 3) as usize);
    }

    #[test]
    fn openai_and_anthropic_shapes() {
        let imgs = prepare_images(&[solid(4, 4, 7)], &MediaLimits::default()).unwrap();
        let oai = openai_image_parts(&imgs);
        assert_eq!(oai[0]["type"], "image_url");
        assert!(
            oai[0]["image_url"]["url"]
                .as_str()
                .unwrap()
                .starts_with("data:image/png;base64,")
        );
        let ant = anthropic_image_blocks(&imgs);
        assert_eq!(ant[0]["type"], "image");
        assert_eq!(ant[0]["source"]["media_type"], "image/png");
        let pdf = anthropic_pdf_block(b"%PDF-1.4 fake", &MediaLimits::default()).unwrap();
        assert_eq!(pdf["type"], "document");
        assert_eq!(pdf["source"]["media_type"], "application/pdf");
    }

    #[test]
    fn video_frames_capped_as_images() {
        let frames: Vec<_> = (0..40).map(|i| solid(4, 4, i as u8)).collect();
        let limits = MediaLimits {
            max_images: 32,
            ..MediaLimits::default()
        };
        assert!(frames_as_images(&frames, &limits).is_err());
        let ok = frames_as_images(&frames[..32], &limits).unwrap();
        assert_eq!(ok.len(), 32);
    }

    #[test]
    fn pdf_oversize() {
        let limits = MediaLimits {
            max_bytes: 8,
            ..MediaLimits::default()
        };
        assert!(anthropic_pdf_block(&[0; 16], &limits).is_err());
    }
}
