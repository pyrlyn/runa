// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! Audio route selection (P4.4): `auto | native | asr` × backend.

/// User/config preference (`audio.route`, `--audio-route`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioRoutePref {
    Auto,
    Native,
    Asr,
}

impl AudioRoutePref {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(AudioRoutePref::Auto),
            "native" => Ok(AudioRoutePref::Native),
            "asr" => Ok(AudioRoutePref::Asr),
            other => Err(format!("{other}: audio.route must be auto | native | asr")),
        }
    }
}

/// Where the request will run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioBackend {
    Local { has_audio_mmproj: bool },
    OpenAi { audio_capable: bool },
    Anthropic,
}

/// How to feed audio into the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioPlan {
    /// PCM → mtmd audio chunk.
    Native,
    /// ASR → text in the prompt.
    Transcribe,
    /// OpenAI Chat Completions `input_audio`.
    OpenAiInputAudio,
}

/// OpenAI chat models that accept `input_audio` parts.
pub fn openai_audio_capable(model: &str) -> bool {
    let m = model.to_ascii_lowercase();
    m.contains("gpt-audio") || m.contains("gpt-4o-audio") || m.contains("audio-preview")
}

/// Pick a concrete audio plan. Errors when `native` is forced but unavailable.
pub fn select_audio_route(
    pref: AudioRoutePref,
    backend: AudioBackend,
) -> Result<AudioPlan, String> {
    match (pref, backend) {
        (AudioRoutePref::Asr, _) => Ok(AudioPlan::Transcribe),
        (
            AudioRoutePref::Auto,
            AudioBackend::Local {
                has_audio_mmproj: true,
            },
        ) => Ok(AudioPlan::Native),
        (
            AudioRoutePref::Auto,
            AudioBackend::Local {
                has_audio_mmproj: false,
            },
        ) => Ok(AudioPlan::Transcribe),
        (
            AudioRoutePref::Auto,
            AudioBackend::OpenAi {
                audio_capable: true,
            },
        ) => Ok(AudioPlan::OpenAiInputAudio),
        (
            AudioRoutePref::Auto,
            AudioBackend::OpenAi {
                audio_capable: false,
            },
        ) => Ok(AudioPlan::Transcribe),
        (AudioRoutePref::Auto, AudioBackend::Anthropic) => Ok(AudioPlan::Transcribe),
        (
            AudioRoutePref::Native,
            AudioBackend::Local {
                has_audio_mmproj: true,
            },
        ) => Ok(AudioPlan::Native),
        (
            AudioRoutePref::Native,
            AudioBackend::Local {
                has_audio_mmproj: false,
            },
        ) => Err(
            "audio.route=native needs an audio mmproj (--mmproj or sibling *mmproj*.gguf)".into(),
        ),
        (
            AudioRoutePref::Native,
            AudioBackend::OpenAi {
                audio_capable: true,
            },
        ) => Ok(AudioPlan::OpenAiInputAudio),
        (
            AudioRoutePref::Native,
            AudioBackend::OpenAi {
                audio_capable: false,
            },
        ) => Err("audio.route=native: this OpenAI model has no input_audio; use auto|asr".into()),
        (AudioRoutePref::Native, AudioBackend::Anthropic) => {
            Err("audio.route=native: Anthropic has no audio; use auto|asr (transcript)".into())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cell(pref: AudioRoutePref, backend: AudioBackend) -> Result<AudioPlan, String> {
        select_audio_route(pref, backend)
    }

    #[test]
    fn matrix_route_times_backend() {
        let local_yes = AudioBackend::Local {
            has_audio_mmproj: true,
        };
        let local_no = AudioBackend::Local {
            has_audio_mmproj: false,
        };
        let oai_yes = AudioBackend::OpenAi {
            audio_capable: true,
        };
        let oai_no = AudioBackend::OpenAi {
            audio_capable: false,
        };
        let ant = AudioBackend::Anthropic;

        // auto
        assert_eq!(
            cell(AudioRoutePref::Auto, local_yes).unwrap(),
            AudioPlan::Native
        );
        assert_eq!(
            cell(AudioRoutePref::Auto, local_no).unwrap(),
            AudioPlan::Transcribe
        );
        assert_eq!(
            cell(AudioRoutePref::Auto, oai_yes).unwrap(),
            AudioPlan::OpenAiInputAudio
        );
        assert_eq!(
            cell(AudioRoutePref::Auto, oai_no).unwrap(),
            AudioPlan::Transcribe
        );
        assert_eq!(
            cell(AudioRoutePref::Auto, ant).unwrap(),
            AudioPlan::Transcribe
        );

        // native
        assert_eq!(
            cell(AudioRoutePref::Native, local_yes).unwrap(),
            AudioPlan::Native
        );
        assert!(cell(AudioRoutePref::Native, local_no).is_err());
        assert_eq!(
            cell(AudioRoutePref::Native, oai_yes).unwrap(),
            AudioPlan::OpenAiInputAudio
        );
        assert!(cell(AudioRoutePref::Native, oai_no).is_err());
        assert!(cell(AudioRoutePref::Native, ant).is_err());

        // asr — every backend transcribes
        for b in [local_yes, local_no, oai_yes, oai_no, ant] {
            assert_eq!(
                cell(AudioRoutePref::Asr, b).unwrap(),
                AudioPlan::Transcribe,
                "{b:?}"
            );
        }
    }

    #[test]
    fn openai_capable_models() {
        assert!(openai_audio_capable("gpt-4o-audio-preview"));
        assert!(openai_audio_capable("gpt-audio"));
        assert!(!openai_audio_capable("gpt-4o-mini"));
        assert!(!openai_audio_capable("claude-sonnet-5"));
    }

    #[test]
    fn parse_pref() {
        assert_eq!(AudioRoutePref::parse("AUTO").unwrap(), AudioRoutePref::Auto);
        assert!(AudioRoutePref::parse("cloud").is_err());
    }
}
