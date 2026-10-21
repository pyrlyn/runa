// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! `POST /v1/infill`: fill-in-the-middle for editors (plan P20.1).
//!
//! Field names follow llama-server's `/infill`, so a client written for it
//! needs only the path. Fields this endpoint does not use (`input_extra`,
//! sampler knobs) are ignored, as llama-server ignores unknown ones.

use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::Json;
use runa_engine::{GenEvent, GenerateRequest, Infill, SamplingConfig, StopReason};
use serde::Deserialize;
use serde_json::{Value, json};

use runa_pool::EngineJob;

use super::{AppState, acquire_parallel, deny, engine_status, note_events, with_engine};

/// Under `/v1/` so `--api-key` covers it like every other generation route.
pub(super) const ROUTE: &str = "/v1/infill";

/// Inline completions are short; a default this size keeps a keystroke's
/// request cheap when the client sends none.
const DEFAULT_N_PREDICT: u32 = 128;
const MAX_N_PREDICT: i64 = 4096;
const MAX_STOP_STRINGS: usize = 16;

#[derive(Debug, Deserialize)]
pub(super) struct InfillBody {
    #[serde(default)]
    input_prefix: String,
    #[serde(default)]
    input_suffix: String,
    #[serde(default)]
    model: Option<String>,
    /// Signed because llama-server clients send `-1` for "no limit", which
    /// this endpoint refuses instead of reading as a huge number.
    #[serde(default)]
    n_predict: Option<i64>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    stop: Vec<String>,
}

/// The engine request for a body, or why the body is refused.
fn infill_request(body: InfillBody) -> Result<GenerateRequest, String> {
    if body.input_prefix.is_empty() && body.input_suffix.is_empty() {
        return Err("input_prefix or input_suffix must be non-empty".into());
    }
    let max_tokens = match body.n_predict {
        None => DEFAULT_N_PREDICT,
        Some(n) if (1..=MAX_N_PREDICT).contains(&n) => n as u32,
        Some(_) => return Err(format!("n_predict must be between 1 and {MAX_N_PREDICT}")),
    };
    // Greedy by default: the same cursor context should suggest the same
    // middle, or the suggestion flickers between keystrokes.
    let temperature = body.temperature.unwrap_or(0.0);
    if !(0.0..=2.0).contains(&temperature) {
        return Err("temperature must be between 0 and 2".into());
    }
    if body.stop.len() > MAX_STOP_STRINGS {
        return Err(format!("at most {MAX_STOP_STRINGS} stop strings"));
    }
    Ok(GenerateRequest {
        sampling: SamplingConfig {
            temperature,
            ..SamplingConfig::default()
        },
        max_tokens,
        stop: body.stop.into_iter().filter(|s| !s.is_empty()).collect(),
        infill: Some(Infill {
            prefix: body.input_prefix,
            suffix: body.input_suffix,
        }),
        ..GenerateRequest::default()
    })
}

/// llama-server's reply shape: `content`, `stop_type`, token counts.
fn infill_reply(model: &str, events: &[GenEvent]) -> Value {
    let mut content = String::new();
    let (mut evaluated, mut predicted) = (0, 0);
    let mut stop_type = "none";
    let mut stopping_word = String::new();
    for ev in events {
        match ev {
            GenEvent::Text(t) => content.push_str(t),
            GenEvent::Usage(u) => (evaluated, predicted) = (u.prompt_tokens, u.generated_tokens),
            GenEvent::Done(StopReason::Eos) => stop_type = "eos",
            GenEvent::Done(StopReason::MaxTokens) => stop_type = "limit",
            GenEvent::Done(StopReason::StopString(s)) => {
                stop_type = "word";
                stopping_word.clone_from(s);
            }
            GenEvent::Reasoning(_) | GenEvent::ToolCalls(_) => {}
        }
    }
    json!({
        "content": content,
        "model": model,
        "stop_type": stop_type,
        "stopping_word": stopping_word,
        "tokens_evaluated": evaluated,
        "tokens_predicted": predicted,
    })
}

async fn run_infill(
    jobs: &Arc<std::sync::mpsc::Sender<EngineJob>>,
    req: GenerateRequest,
) -> Result<Vec<GenEvent>, (StatusCode, String)> {
    runa_pool::generate_on(jobs, req)
        .await
        .map_err(engine_status)
}

pub(super) async fn infill(
    State(st): State<AppState>,
    Json(body): Json<InfillBody>,
) -> Result<Json<Value>, (StatusCode, String)> {
    let crate::dashboard::Admission {
        permit: _permit,
        mut flight,
    } = acquire_parallel(&st, ROUTE).await?;
    // P10.5: real work arrived — hold off the idle sweep.
    st.mm.touch();
    let model_id = body
        .model
        .clone()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| st.default_id.clone());
    let req = infill_request(body).map_err(|e| deny(&mut flight, StatusCode::BAD_REQUEST, e))?;
    flight.set_model(&model_id);
    let jobs = match with_engine(&st.pool, &model_id).await {
        Ok(jobs) => jobs,
        Err((status, msg)) => return Err(deny(&mut flight, status, msg)),
    };
    let events = match run_infill(&jobs, req).await {
        Ok(events) => events,
        Err((status, msg)) => return Err(deny(&mut flight, status, msg)),
    };
    note_events(&mut flight, &events);
    Ok(Json(infill_reply(&model_id, &events)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn body(raw: &str) -> InfillBody {
        serde_json::from_str(raw).unwrap()
    }

    #[test]
    fn route_is_behind_the_api_key() {
        assert!(ROUTE.starts_with("/v1/"));
    }

    #[test]
    fn llama_server_bodies_map_to_an_infill_request() {
        let req = infill_request(body(
            r#"{"input_prefix":"def f(","input_suffix":"):\n","n_predict":32,
                "temperature":0.2,"stop":["\n\n",""],"input_extra":[],"cache_prompt":true}"#,
        ))
        .unwrap();
        assert_eq!(
            req.infill,
            Some(Infill {
                prefix: "def f(".into(),
                suffix: "):\n".into()
            })
        );
        assert!(req.messages.is_empty());
        assert_eq!(req.max_tokens, 32);
        assert_eq!(req.sampling.temperature, 0.2);
        assert_eq!(req.stop, vec!["\n\n".to_owned()]);

        let req = infill_request(body(r#"{"input_suffix":"x"}"#)).unwrap();
        assert_eq!(req.max_tokens, DEFAULT_N_PREDICT);
        assert_eq!(req.sampling.temperature, 0.0);
    }

    #[test]
    fn bad_bodies_are_refused() {
        for raw in [
            r#"{}"#,
            r#"{"input_prefix":"","input_suffix":""}"#,
            r#"{"input_prefix":"a","n_predict":0}"#,
            r#"{"input_prefix":"a","n_predict":-1}"#,
            r#"{"input_prefix":"a","n_predict":4097}"#,
            r#"{"input_prefix":"a","temperature":-0.5}"#,
            r#"{"input_prefix":"a","temperature":9}"#,
        ] {
            assert!(infill_request(body(raw)).is_err(), "{raw}");
        }
        let many: Vec<String> = (0..=MAX_STOP_STRINGS).map(|i| i.to_string()).collect();
        let raw = json!({"input_prefix": "a", "stop": many}).to_string();
        assert!(infill_request(body(&raw)).is_err());
    }

    #[test]
    fn reply_has_llama_server_fields() {
        let events = vec![
            GenEvent::Text("a + ".into()),
            GenEvent::Text("b".into()),
            GenEvent::Usage(runa_engine::Usage {
                prompt_tokens: 12,
                generated_tokens: 3,
                reasoning_tokens: 0,
                pp_toks_per_s: 1.0,
                tg_toks_per_s: 1.0,
            }),
            GenEvent::Done(StopReason::Eos),
        ];
        let v = infill_reply("m", &events);
        assert_eq!(v["content"], "a + b");
        assert_eq!(v["model"], "m");
        assert_eq!(v["stop_type"], "eos");
        assert_eq!(v["tokens_evaluated"], 12);
        assert_eq!(v["tokens_predicted"], 3);
        let cut = infill_reply("m", &[GenEvent::Done(StopReason::MaxTokens)]);
        assert_eq!(cut["stop_type"], "limit");
        let word = infill_reply("m", &[GenEvent::Done(StopReason::StopString("\n".into()))]);
        assert_eq!(
            (word["stop_type"].as_str(), word["stopping_word"].as_str()),
            (Some("word"), Some("\n"))
        );
    }

    /// An engine thread that checks the job is an infill and answers with
    /// `answer`, through the same job protocol as the real one.
    fn fake_engine(
        answer: Result<Vec<GenEvent>, String>,
    ) -> Arc<std::sync::mpsc::Sender<EngineJob>> {
        let (jobs, rx) = std::sync::mpsc::channel::<EngineJob>();
        std::thread::spawn(move || {
            if let Ok(EngineJob::Generate { req, resp }) = rx.recv() {
                let answer = if req.infill.is_some() {
                    answer
                } else {
                    Err("not an infill request".into())
                };
                let _ = resp.send(answer);
            }
        });
        Arc::new(jobs)
    }

    #[tokio::test]
    async fn the_engine_gets_the_infill_and_its_errors_keep_their_status() {
        let req = || infill_request(body(r#"{"input_prefix":"a"}"#)).unwrap();
        let ok = fake_engine(Ok(vec![GenEvent::Text("b".into())]));
        let events = run_infill(&ok, req()).await.unwrap();
        assert_eq!(events, vec![GenEvent::Text("b".into())]);

        let unsupported = fake_engine(Err(
            "infill_unsupported: the model has no fill-in-the-middle tokens".into(),
        ));
        let (status, msg) = run_infill(&unsupported, req()).await.unwrap_err();
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(msg.starts_with("infill_unsupported"), "{msg}");

        let too_long = fake_engine(Err("context_length_exceeded: prompt has 9 tokens".into()));
        assert_eq!(
            run_infill(&too_long, req()).await.unwrap_err().0,
            StatusCode::BAD_REQUEST
        );
        let broken = fake_engine(Err("decode error: boom".into()));
        assert_eq!(
            run_infill(&broken, req()).await.unwrap_err().0,
            StatusCode::INTERNAL_SERVER_ERROR
        );
    }
}
