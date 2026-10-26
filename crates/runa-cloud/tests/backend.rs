// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! P21.1: the cloud adapters behind `runa_core::Backend`, against recorded
//! fixtures. The tests are plain `#[test]`s because `generate` blocks on the
//! backend's own runtime; the mock server runs on a separate one.
use std::path::PathBuf;

use runa_cloud::openai::OpenAiClient;
use runa_cloud::{AnthropicBackend, AnthropicClient, OpenAiBackend};
use runa_core::{Backend, ChatMessage, Event, Request, StopReason, Usage};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn fixture(name: &str) -> String {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("../../tests/fixtures/api");
    p.push(name);
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
}

/// Serve `body` as JSON at `route`; the runtime must outlive the test.
fn serve(rt: &tokio::runtime::Runtime, route: &str, body: String) -> MockServer {
    rt.block_on(async {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path(route))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "application/json")
                    .set_body_string(body),
            )
            .mount(&server)
            .await;
        server
    })
}

fn drain(backend: &mut dyn Backend) -> Vec<Event> {
    let req = Request {
        messages: vec![ChatMessage::user("hi")],
        max_tokens: 32,
        ..Request::default()
    };
    backend
        .generate(&req)
        .expect("generate")
        .collect::<Result<_, _>>()
        .expect("events")
}

fn usage(prompt_tokens: u32, generated_tokens: u32) -> Event {
    Event::Usage(Usage {
        prompt_tokens,
        generated_tokens,
        ..Usage::default()
    })
}

#[test]
fn openai_backend_replays_the_reply_as_core_events() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let server = serve(
        &rt,
        "/v1/chat/completions",
        fixture("openai-chat-completion.json"),
    );
    let client = OpenAiClient::new("sk-test", Some(&format!("{}/v1", server.uri())));
    let mut backend = OpenAiBackend::new(client, "gpt-4o-mini").unwrap();
    assert_eq!(
        drain(&mut backend),
        vec![
            Event::Text("Hello from fixture".into()),
            usage(5, 4),
            Event::Done(StopReason::Eos),
        ]
    );
}

#[test]
fn anthropic_backend_replays_the_reply_as_core_events() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let server = serve(&rt, "/v1/messages", fixture("anthropic-message.json"));
    let client = AnthropicClient::new("sk-ant-test").with_base_url(server.uri());
    let mut backend = AnthropicBackend::new(client, "claude-3-5-sonnet").unwrap();
    assert_eq!(
        drain(&mut backend),
        vec![
            Event::Text("Hello from Anthropic fixture".into()),
            usage(5, 6),
            Event::Done(StopReason::Eos),
        ]
    );
}
