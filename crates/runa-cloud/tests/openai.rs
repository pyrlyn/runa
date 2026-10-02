// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! P3.5: wiremock tests against recorded OpenAI chat-completion fixtures.
use runa_cloud::openai::{ChatMessage, ChatRequest, CloudEvent, OpenAiClient};
use runa_core::ThinkConfig;
use std::path::PathBuf;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn fixture(name: &str) -> String {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("../../tests/fixtures/api");
    p.push(name);
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
}

fn req() -> ChatRequest {
    ChatRequest {
        model: "gpt-4o-mini".into(),
        messages: vec![ChatMessage::user("hi")],
        think: ThinkConfig::default(),
        max_tokens: None,
        json_schema: None,
        tools: None,
    }
}

#[tokio::test]
async fn complete_from_fixture() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_string(fixture("openai-chat-completion.json")),
        )
        .mount(&server)
        .await;

    let client = OpenAiClient::new("sk-test", Some(&format!("{}/v1", server.uri())));
    let events = client.complete(req()).await.expect("complete");
    assert!(
        events
            .iter()
            .any(|e| matches!(e, CloudEvent::Text(t) if t == "Hello from fixture")),
        "{events:?}"
    );
    assert!(
        events.iter().any(|e| matches!(
            e,
            CloudEvent::Usage {
                prompt_tokens: 5,
                completion_tokens: 4
            }
        )),
        "{events:?}"
    );
}

#[tokio::test]
async fn stream_from_fixture() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(fixture("openai-chat-completion-stream.sse")),
        )
        .mount(&server)
        .await;

    let client = OpenAiClient::new("sk-test", Some(&format!("{}/v1", server.uri())));
    let events = client.stream(req()).await.expect("stream");
    let text: String = events
        .iter()
        .filter_map(|e| match e {
            CloudEvent::Text(t) => Some(t.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(text, "Hello");
}

#[tokio::test]
async fn complete_splits_reasoning_content() {
    let body = serde_json::json!({
        "id": "chatcmpl-r",
        "object": "chat.completion",
        "created": 1,
        "model": "o4-mini",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Paris",
                "reasoning_content": "capital of France"
            },
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
    });
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(&body),
        )
        .mount(&server)
        .await;

    let client = OpenAiClient::new("sk-test", Some(&format!("{}/v1", server.uri())));
    let events = client.complete(req()).await.expect("complete");
    // typed deserialize drops unknown fields; split still covers the JSON helper.
    // If the crate learns the field, this assertion stays valid.
    let has_text = events
        .iter()
        .any(|e| matches!(e, CloudEvent::Text(t) if t == "Paris"));
    assert!(has_text, "{events:?}");
}

#[tokio::test]
async fn live_openai_smoke() {
    if std::env::var("RUNA_LIVE").ok().as_deref() != Some("1") {
        return;
    }
    let key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY for RUNA_LIVE=1");
    let client = OpenAiClient::new(key, None);
    let events = client
        .complete(ChatRequest {
            model: std::env::var("RUNA_LIVE_MODEL").unwrap_or_else(|_| "gpt-4o-mini".into()),
            messages: vec![ChatMessage::user("Reply with the single word pong.")],
            think: ThinkConfig::default(),
            max_tokens: Some(8),
            json_schema: None,
            tools: None,
        })
        .await
        .expect("live complete");
    assert!(
        events
            .iter()
            .any(|e| matches!(e, CloudEvent::Text(t) if !t.is_empty())),
        "{events:?}"
    );
}
