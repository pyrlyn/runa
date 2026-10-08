// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! P3.7 — cloud routing e2e with a mock OpenAI server.

use std::process::Command;
use std::str::FromStr;

use assert_cmd::cargo::cargo_bin;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn run_openai_ref_uses_mock_and_prints_cost() {
    let server = MockServer::start().await;
    let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/api/openai-chat-completion.json");
    let body = std::fs::read_to_string(fixture).expect("fixture");
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_string(body))
        .mount(&server)
        .await;

    let mut cmd = Command::new(cargo_bin("runa"));
    cmd.args(["run", "openai:gpt-4o-mini", "hello", "--max-tokens", "16"])
        .env("RUNA_NO_KEYRING", "1")
        .env("OPENAI_API_KEY", "sk-test")
        .env("OPENAI_BASE_URL", format!("{}/v1", server.uri()));

    let out = cmd.output().expect("run");
    let stderr = String::from_utf8_lossy(&out.stderr);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "stderr={stderr}");
    assert!(stdout.contains("Hello from fixture"), "stdout={stdout}");
    assert!(
        stderr.contains("cost:"),
        "expected cost line, stderr={stderr}"
    );
}

#[test]
fn prices_fixture_matches_cost_table() {
    let prices_path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/prices.toml");
    let text = std::fs::read_to_string(prices_path).expect("prices");
    let table = runa_cloud::PriceTable::from_str(&text).unwrap();
    let line = table
        .cost_line("openai", "gpt-4o-mini", 5, 4)
        .expect("price row");
    assert!(line.contains("cost: $"));
}
