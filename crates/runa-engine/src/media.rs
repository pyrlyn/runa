//! Native audio through mtmd (plan P4.3).
//!
//! PCM is 16 kHz mono f32 (same as `runa-media`). The mmproj is optional at
//! load; `--audio` without one is an explicit error (D12, no silent ASR).

use std::path::Path;

use crate::generate::ChatMessage;
use crate::load::{EngineError, LoadedModel};

impl LoadedModel {
    /// Whether the loaded mmproj accepts PCM audio chunks.
    pub fn supports_native_audio(&self) -> bool {
        #[cfg(feature = "mtmd")]
        {
            self.mtmd
                .as_ref()
                .is_some_and(llama_cpp_2::mtmd::MtmdContext::support_audio)
        }
        #[cfg(not(feature = "mtmd"))]
        {
            false
        }
    }

    /// Bytes of the mmproj file (0 if none). Fit uses the same number.
    pub fn mmproj_bytes(&self) -> u64 {
        self.config()
            .mmproj
            .as_ref()
            .map(|p| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0))
            .unwrap_or(0)
    }
}

#[cfg(feature = "mtmd")]
pub(crate) fn load_mtmd(
    mmproj: Option<&Path>,
    model: &llama_cpp_2::model::LlamaModel,
    use_gpu: bool,
) -> Result<Option<llama_cpp_2::mtmd::MtmdContext>, EngineError> {
    use std::ffi::CString;

    use llama_cpp_2::mtmd::{MtmdContext, MtmdContextParams, mtmd_default_marker};

    let Some(path) = mmproj else {
        return Ok(None);
    };
    if !path.is_file() {
        return Err(EngineError::Media(format!(
            "mmproj not found: {}",
            path.display()
        )));
    }
    let params = MtmdContextParams {
        use_gpu,
        print_timings: false,
        n_threads: 4,
        media_marker: CString::new(mtmd_default_marker())
            .map_err(|_| EngineError::Media("media marker".into()))?,
        image_min_tokens: -1,
        image_max_tokens: -1,
    };
    let ctx = MtmdContext::init_from_file(path, model, &params)
        .map_err(|e| EngineError::Media(format!("mtmd init: {e:?}")))?;
    Ok(Some(ctx))
}

#[cfg(not(feature = "mtmd"))]
pub(crate) fn load_mtmd(
    mmproj: Option<&Path>,
    _model: &llama_cpp_2::model::LlamaModel,
    _use_gpu: bool,
) -> Result<(), EngineError> {
    if mmproj.is_some() {
        return Err(EngineError::Unsupported(
            "mmproj requires rebuilding with --features mtmd",
        ));
    }
    Ok(())
}

impl LoadedModel {
    /// Tokenize + eval PCM as an mtmd audio chunk. Returns `n_past`.
    pub(crate) fn eval_audio_prompt(
        &mut self,
        messages: &[ChatMessage],
        pcm: &[f32],
        add_generation_prompt: bool,
    ) -> Result<i32, EngineError> {
        #[cfg(not(feature = "mtmd"))]
        {
            let _ = (messages, pcm, add_generation_prompt);
            // Load-bearing: without `return`, a default-features build falls
            // through to the compiled-out mtmd block (clippy's remove-`return`
            // suggestion was verified to break the build: E0308).
            #[allow(clippy::needless_return)]
            return Err(EngineError::Unsupported(
                "--audio requires rebuilding with --features mtmd",
            ));
        }
        #[cfg(feature = "mtmd")]
        {
            use llama_cpp_2::mtmd::{MtmdBitmap, MtmdInputText, mtmd_default_marker};

            if self.mtmd.is_none() {
                return Err(EngineError::Media(
                    "--audio needs an audio mmproj (--mmproj or a sibling *mmproj*.gguf)".into(),
                ));
            }
            if !self.mtmd.as_ref().unwrap().support_audio() {
                return Err(EngineError::Media(
                    "mmproj has no audio encoder (vision-only); use ASR or a Voxtral/Qwen2-Audio mmproj"
                        .into(),
                ));
            }
            let marker = mtmd_default_marker();
            let mut msgs = messages.to_vec();
            if let Some(last) = msgs.last_mut()
                && !last.content.contains(marker)
            {
                last.content.push(' ');
                last.content.push_str(marker);
            }
            let text = self.render_prompt(&msgs, add_generation_prompt)?;
            let bitmap = MtmdBitmap::from_audio_data(pcm)
                .map_err(|e| EngineError::Media(format!("audio bitmap: {e:?}")))?;
            let chunks = self
                .mtmd
                .as_ref()
                .unwrap()
                .tokenize(
                    MtmdInputText {
                        text,
                        add_special: true,
                        parse_special: true,
                    },
                    &[&bitmap],
                )
                .map_err(|e| EngineError::Media(format!("mtmd tokenize: {e:?}")))?;
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
    #[test]
    #[allow(clippy::assertions_on_constants)] // a runtime check of the feature set, on purpose
    fn default_build_has_no_mtmd() {
        assert!(
            !cfg!(feature = "mtmd"),
            "CI default features must not compile mtmd"
        );
    }
}
