// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! P2.3/P2.5 e2e: `runa run` (one-shot, stdin piping, `--json`, auto/`on_unfit`)
//! and `runa chat` (piped REPL with slash commands), via `assert_cmd`.
//!
//! Each test loads the qwen2 fixture on CPU (~seconds); the binary is built
//! once by cargo.

use std::path::PathBuf;

use assert_cmd::Command;
use predicates::prelude::*;

fn runa() -> Command {
    let mut cmd = Command::cargo_bin("runa").expect("runa binary builds");
    // Isolate tests from ~/.cache/runa/kv (LMDB is process-locked).
    cmd.env("RUNA_NO_PROMPT_CACHE", "1");
    cmd
}

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures")
        .join(name)
}

#[test]
fn run_streams_text_and_reports_usage() {
    let model = fixture("qwen2-0_5b-instruct-q4_0.gguf");
    let out = runa()
        .args([
            "run",
            "--mode",
            "cpu",
            model.to_str().unwrap(),
            "The capital of France is",
            "--max-tokens",
            "8",
        ])
        .assert()
        .success()
        .get_output()
        .clone();
    let stdout = String::from_utf8(out.stdout).expect("utf8 stdout");
    let stderr = String::from_utf8(out.stderr).expect("utf8 stderr");
    assert!(!stdout.trim().is_empty(), "streams text to stdout");
    assert!(
        stderr.contains("tokens:"),
        "usage line on stderr: {stderr:?}"
    );
}

#[test]
fn run_json_is_parseable() {
    let model = fixture("qwen2-0_5b-instruct-q4_0.gguf");
    let out = runa()
        .args([
            "run",
            "--mode",
            "cpu",
            model.to_str().unwrap(),
            "hi",
            "--max-tokens",
            "4",
            "--json",
        ])
        .assert()
        .success()
        .get_output()
        .clone();
    let stdout = String::from_utf8(out.stdout).expect("utf8 stdout");
    // Minimal structural check without a JSON dep in e2e scope.
    assert!(stdout.contains("\"text\""), "{stdout:?}");
    assert!(stdout.contains("\"usage\""), "{stdout:?}");
    assert!(stdout.contains("\"stop\""), "{stdout:?}");
}

#[test]
fn run_reads_prompt_from_stdin() {
    let model = fixture("qwen2-0_5b-instruct-q4_0.gguf");
    runa()
        .args([
            "run",
            "--mode",
            "cpu",
            model.to_str().unwrap(),
            "--max-tokens",
            "4",
        ])
        .write_stdin("Say the word blue.")
        .assert()
        .success();
}

#[test]
fn run_missing_model_fails_cleanly() {
    // Partial stderr match via predicates; exit code via assert_cmd.
    runa()
        .args(["run", "no-such-model.gguf", "hi"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("no such model file"));
}

#[test]
fn chat_answers_then_quits() {
    let model = fixture("qwen2-0_5b-instruct-q4_0.gguf");
    let out = runa()
        .args(["chat", model.to_str().unwrap()])
        .write_stdin("What is 2+2?\n/usage\n/quit\n")
        .assert()
        .success()
        .get_output()
        .clone();
    let stdout = String::from_utf8(out.stdout).expect("utf8 stdout");
    assert!(
        stdout.contains("prompt"),
        "usage counters shown: {stdout:?}"
    );
}

#[test]
fn chat_second_turn_reuses_the_context() {
    // A second turn used to prefill onto the first turn's KV cells (P8.3).
    let model = fixture("qwen2-0_5b-instruct-q4_0.gguf");
    let out = runa()
        .args(["chat", model.to_str().unwrap()])
        .write_stdin("Say hi.\nSay bye.\n/quit\n")
        .assert()
        .success()
        .get_output()
        .clone();
    let stderr = String::from_utf8(out.stderr).expect("utf8 stderr");
    assert!(!stderr.contains("generate:"), "{stderr}");
}

#[test]
fn run_mcp_tool_loop_calls_the_server() {
    let model = fixture("qwen2-0_5b-instruct-q4_0.gguf");
    let dir = assert_fs::TempDir::new().unwrap();
    let log = dir.path().join("calls.jsonl");
    let server = format!("python3 {}", fixture("mcp-echo.py").display());
    let out = runa()
        .env("MCP_ECHO_LOG", &log)
        .args([
            "run",
            "--mode",
            "cpu",
            model.to_str().unwrap(),
            "What is the weather in Paris? Use the get_weather tool.",
            "--mcp",
            &server,
            "--temperature",
            "0",
            "--max-tokens",
            "96",
            "--max-tool-rounds",
            "2",
        ])
        .output()
        .expect("runa runs");
    let stderr = String::from_utf8_lossy(&out.stderr);
    // The 0.5B model may keep calling instead of answering; the round cap
    // is then the only acceptable failure.
    assert!(
        out.status.success() || stderr.contains("tool rounds"),
        "{stderr}"
    );
    assert!(stderr.contains("mcp: 1 tools from 1 servers"), "{stderr}");
    assert!(stderr.contains("[tool] get_weather("), "{stderr}");
    let calls = std::fs::read_to_string(&log).expect("server saw a call");
    assert!(calls.contains("\"get_weather\""), "{calls}");
}

#[test]
fn run_mcp_missing_command_fails_before_loading() {
    let model = fixture("qwen2-0_5b-instruct-q4_0.gguf");
    runa()
        .args([
            "run",
            model.to_str().unwrap(),
            "hi",
            "--mcp",
            "runa-no-such-mcp-server --flag",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("spawn runa-no-such-mcp-server"));
}

#[test]
fn auto_unfit_falls_back_to_cpu() {
    let model = fixture("qwen2-0_5b-instruct-q4_0.gguf");
    let out = runa()
        .env("RUNA_FAKE_VRAM", "0")
        .env("RUNA_FAKE_RAM", "64")
        .env("RUNA_MEMORY_CEILING_MIB", "65536")
        .env_remove("RUNA_ON_UNFIT")
        .args([
            "run",
            model.to_str().unwrap(),
            "hi",
            "--max-tokens",
            "4",
            "--on-unfit",
            "cpu",
        ])
        .assert()
        .success()
        .get_output()
        .clone();
    let stderr = String::from_utf8(out.stderr).expect("utf8 stderr");
    assert!(
        stderr.contains("NO FIT"),
        "verdict line on stderr: {stderr:?}"
    );
    assert!(
        stderr.contains("on_unfit=cpu"),
        "explicit CPU fallback warning: {stderr:?}"
    );
    assert!(
        stderr.contains("cpu (0 layers on GPU)"),
        "load on CPU after fallback: {stderr:?}"
    );
}

#[test]
fn auto_unfit_error_exits_2() {
    let model = fixture("qwen2-0_5b-instruct-q4_0.gguf");
    let out = runa()
        .env("RUNA_FAKE_VRAM", "0")
        .env("RUNA_FAKE_RAM", "64")
        .env("RUNA_MEMORY_CEILING_MIB", "65536")
        .env_remove("RUNA_ON_UNFIT")
        .args([
            "run",
            model.to_str().unwrap(),
            "hi",
            "--max-tokens",
            "4",
            "--on-unfit",
            "error",
        ])
        .assert()
        .code(2)
        .get_output()
        .clone();
    let stderr = String::from_utf8(out.stderr).expect("utf8 stderr");
    assert!(stderr.contains("NO FIT"), "{stderr:?}");
    assert!(stderr.contains("on_unfit=error"), "{stderr:?}");
    assert!(
        !stderr.contains("cpu (0 layers on GPU)"),
        "must not load after on_unfit=error: {stderr:?}"
    );
}

#[test]
fn run_over_ceiling_memory_suggests() {
    let model = fixture("qwen2-0_5b-instruct-q4_0.gguf");
    let out = runa()
        .env("RUNA_MEMORY_CEILING_MIB", "1")
        .env("RUNA_MEMORY_MAX_GROWTH_MIB", "1")
        .args([
            "run",
            "--mode",
            "cpu",
            "--ctx",
            "512",
            model.to_str().unwrap(),
            "hi",
            "--max-tokens",
            "1",
        ])
        .assert()
        .code(2)
        .get_output()
        .clone();
    let stderr = String::from_utf8(out.stderr).expect("utf8 stderr");
    assert!(
        stderr.contains("ceiling") && stderr.contains("smaller --ctx"),
        "over-ceiling must suggest a smaller ctx: {stderr:?}"
    );
    assert!(
        !stderr.contains("cpu (0 layers on GPU)"),
        "must not load after grow_for over-ceiling: {stderr:?}"
    );
}

#[test]
fn run_n_cpu_moe_prints_expert_overrides() {
    let model = fixture("qwen2-0_5b-instruct-q4_0.gguf");
    let out = runa()
        .args([
            "run",
            "--mode",
            "cpu",
            "--n-cpu-moe",
            "2",
            model.to_str().unwrap(),
            "hi",
            "--max-tokens",
            "4",
        ])
        .assert()
        .success()
        .get_output()
        .clone();
    let stderr = String::from_utf8(out.stderr).expect("utf8 stderr");
    assert!(
        stderr.contains("experts-on-cpu:"),
        "verdict lists expert overrides: {stderr:?}"
    );
    assert!(
        stderr.contains("blk\\.(0|1)"),
        "first-N-layers expert pattern: {stderr:?}"
    );
}

#[test]
fn run_prompt_cache_hits_on_second_run() {
    let model = fixture("qwen2-0_5b-instruct-q4_0.gguf");
    let cache = std::env::temp_dir().join(format!(
        "runa-p28-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("time")
            .as_nanos()
    ));
    std::fs::create_dir_all(&cache).expect("prompt-cache dir");
    let cache_s = cache.to_str().expect("utf8 cache path");
    let model_s = model.to_str().unwrap();
    let first = runa()
        .env_remove("RUNA_NO_PROMPT_CACHE")
        .args([
            "run",
            "--mode",
            "cpu",
            "--prompt-cache",
            cache_s,
            model_s,
            "The capital of France is",
            "--max-tokens",
            "8",
        ])
        .assert()
        .success()
        .get_output()
        .clone();
    let stderr1 = String::from_utf8(first.stderr).expect("utf8 stderr");
    assert!(
        stderr1.contains("prompt-cache: store"),
        "first run stores KV: {stderr1:?}"
    );
    assert!(
        !stderr1.contains("prompt-cache: hit"),
        "first run must miss: {stderr1:?}"
    );
    let second = runa()
        .env_remove("RUNA_NO_PROMPT_CACHE")
        .args([
            "run",
            "--mode",
            "cpu",
            "--prompt-cache",
            cache_s,
            model_s,
            "The capital of France is",
            "--max-tokens",
            "8",
        ])
        .assert()
        .success()
        .get_output()
        .clone();
    let stderr2 = String::from_utf8(second.stderr).expect("utf8 stderr");
    assert!(
        stderr2.contains("prompt-cache: hit"),
        "second run restores KV from LMDB: {stderr2:?}"
    );
    let _ = std::fs::remove_dir_all(&cache);
}

#[test]
fn run_kv_q8_prints_type_on_verdict() {
    let model = fixture("qwen2-0_5b-instruct-q4_0.gguf");
    let out = runa()
        .args([
            "run",
            "--mode",
            "cpu",
            "--kv",
            "q8_0",
            model.to_str().unwrap(),
            "hi",
            "--max-tokens",
            "4",
        ])
        .assert()
        .success()
        .get_output()
        .clone();
    let stderr = String::from_utf8(out.stderr).expect("utf8 stderr");
    assert!(
        stderr.contains("kv k=q8_0 v=q8_0"),
        "verdict lists KV types: {stderr:?}"
    );
}

#[test]
fn run_kv_unknown_fails() {
    let model = fixture("qwen2-0_5b-instruct-q4_0.gguf");
    runa()
        .args([
            "run",
            "--mode",
            "cpu",
            "--kv",
            "int8",
            model.to_str().unwrap(),
            "hi",
        ])
        .assert()
        .failure();
}

#[test]
fn run_kv_k_v_overrides_kv() {
    let model = fixture("qwen2-0_5b-instruct-q4_0.gguf");
    let out = runa()
        .args([
            "run",
            "--mode",
            "cpu",
            "--kv",
            "q4_0",
            "--kv-k",
            "q8_0",
            "--kv-v",
            "q8_0",
            model.to_str().unwrap(),
            "hi",
            "--max-tokens",
            "4",
        ])
        .assert()
        .success()
        .get_output()
        .clone();
    let stderr = String::from_utf8(out.stderr).expect("utf8 stderr");
    assert!(
        stderr.contains("kv k=q8_0 v=q8_0"),
        "--kv-k/--kv-v override --kv: {stderr:?}"
    );
}

#[test]
fn run_help_lists_multi_gpu_flags() {
    let out = runa()
        .args(["run", "--help"])
        .assert()
        .success()
        .get_output()
        .clone();
    let stdout = String::from_utf8(out.stdout).expect("utf8 stdout");
    assert!(stdout.contains("--device"), "{stdout}");
    assert!(stdout.contains("--tensor-split"), "{stdout}");
    assert!(stdout.contains("--audio"), "{stdout}");
    assert!(stdout.contains("--mmproj"), "{stdout}");
    assert!(stdout.contains("--audio-route"), "{stdout}");
    assert!(stdout.contains("--image"), "{stdout}");
    assert!(stdout.contains("--video"), "{stdout}");
    assert!(stdout.contains("--ngram"), "{stdout}");
    assert!(stdout.contains("--draft"), "{stdout}");
}

#[test]
fn run_draft_missing_fails() {
    let model = fixture("qwen2-0_5b-instruct-q4_0.gguf");
    let out = runa()
        .args([
            "run",
            "--draft",
            "/no/such/draft.gguf",
            "--ctx",
            "512",
            model.to_str().unwrap(),
            "hi",
            "--max-tokens",
            "1",
        ])
        .assert()
        .failure()
        .get_output()
        .clone();
    let stderr = String::from_utf8(out.stderr).expect("utf8 stderr");
    assert!(stderr.contains("draft model not found"), "{stderr}");
}

#[test]
fn run_lora_missing_fails() {
    // P8.5: a mistyped `--lora` fails fast naming the adapter (before the
    // model finishes loading).
    let model = fixture("qwen2-0_5b-instruct-q4_0.gguf");
    let out = runa()
        .args([
            "run",
            "--mode",
            "cpu",
            "--ctx",
            "512",
            model.to_str().unwrap(),
            "hi",
            "--max-tokens",
            "1",
            "--lora",
            "no-such-adapter.gguf:0.5",
        ])
        .assert()
        .failure()
        .get_output()
        .clone();
    let stderr = String::from_utf8(out.stderr).expect("utf8 stderr");
    assert!(stderr.contains("no-such-adapter"), "{stderr}");
    assert!(stderr.contains("lora"), "{stderr}");
}

#[test]
fn run_device_unknown_fails() {
    let model = fixture("qwen2-0_5b-instruct-q4_0.gguf");
    let out = runa()
        .args([
            "run",
            "--mode",
            "gpu",
            "--device",
            "999",
            "--ctx",
            "512",
            model.to_str().unwrap(),
            "hi",
            "--max-tokens",
            "1",
        ])
        .assert()
        .failure()
        .get_output()
        .clone();
    let stderr = String::from_utf8(out.stderr).expect("utf8 stderr");
    assert!(
        stderr.contains("device index 999") || stderr.contains("out of range"),
        "unknown device must fail: {stderr:?}"
    );
}

#[cfg(not(feature = "rpc"))]
#[test]
fn run_rpc_fails_unsupported_explicitly() {
    // P9.3: without the `rpc` feature there is no ggml RPC backend — `--rpc`
    // must error naming the flag, never run locally while pretending.
    let model = fixture("qwen2-0_5b-instruct-q4_0.gguf");
    let out = runa()
        .args([
            "run",
            "--mode",
            "cpu",
            "--ctx",
            "512",
            "--rpc",
            "127.0.0.1:50052",
            model.to_str().unwrap(),
            "hi",
            "--max-tokens",
            "1",
        ])
        .assert()
        .failure()
        .get_output()
        .clone();
    let stderr = String::from_utf8(out.stderr).expect("utf8 stderr");
    assert!(
        stderr.contains("--rpc") && stderr.contains("--features rpc"),
        "rpc must fail explicitly: {stderr:?}"
    );
}

#[cfg(feature = "rpc")]
#[test]
fn run_rpc_without_device_fails_explicitly() {
    // P9.3: `--rpc` registers only; without `--device` the servers would sit
    // idle — an explicit error, before any network contact.
    let model = fixture("qwen2-0_5b-instruct-q4_0.gguf");
    let out = runa()
        .args([
            "run",
            "--mode",
            "cpu",
            "--ctx",
            "512",
            "--rpc",
            "127.0.0.1:50052",
            model.to_str().unwrap(),
            "hi",
            "--max-tokens",
            "1",
        ])
        .assert()
        .failure()
        .get_output()
        .clone();
    let stderr = String::from_utf8(out.stderr).expect("utf8 stderr");
    assert!(
        stderr.contains("--device") && stderr.contains("RPC0"),
        "rpc without --device must fail explicitly: {stderr:?}"
    );
}

#[cfg(feature = "rpc")]
#[test]
fn run_rpc_unreachable_fails_before_load() {
    // P9.3: a dead endpoint fails while connecting (port 9 is closed),
    // before backend init or any weight load.
    let model = fixture("qwen2-0_5b-instruct-q4_0.gguf");
    let out = runa()
        .args([
            "run",
            "--mode",
            "gpu",
            "--ctx",
            "512",
            "--rpc",
            "127.0.0.1:9",
            "--device",
            "RPC0",
            model.to_str().unwrap(),
            "hi",
            "--max-tokens",
            "1",
        ])
        .assert()
        .failure()
        .get_output()
        .clone();
    let stderr = String::from_utf8(out.stderr).expect("utf8 stderr");
    assert!(
        stderr.contains("unreachable") && stderr.contains("127.0.0.1:9"),
        "unreachable rpc must fail explicitly: {stderr:?}"
    );
}

#[test]
fn run_rpc_empty_list_fails_at_parse() {
    let model = fixture("qwen2-0_5b-instruct-q4_0.gguf");
    let out = runa()
        .args([
            "run",
            "--mode",
            "cpu",
            "--rpc",
            " , ",
            model.to_str().unwrap(),
            "hi",
        ])
        .assert()
        .failure()
        .get_output()
        .clone();
    let stderr = String::from_utf8(out.stderr).expect("utf8 stderr");
    assert!(
        stderr.contains("--rpc"),
        "empty rpc list must fail: {stderr:?}"
    );
}

#[test]
fn fit_rpc_warns_but_estimates_locally() {
    // P9.3: `fit --rpc` warns explicitly and still prints the local
    // estimate (exit code unchanged) — no silent drop of the flag.
    let model = fixture("qwen2-0_5b-instruct-q4_0.gguf");
    let out = runa()
        .env("RUNA_FAKE_VRAM", "8192")
        .args([
            "fit",
            model.to_str().unwrap(),
            "--ctx",
            "4096",
            "--rpc",
            "127.0.0.1:50052",
        ])
        .assert()
        .code(predicate::in_iter([0, 1]))
        .get_output()
        .clone();
    let stderr = String::from_utf8(out.stderr).expect("utf8 stderr");
    let stdout = String::from_utf8(out.stdout).expect("utf8 stdout");
    #[cfg(not(feature = "rpc"))]
    assert!(
        stderr.contains("warning: --rpc") && stderr.contains("no ggml RPC backend"),
        "fit must warn about rpc: {stderr:?}"
    );
    #[cfg(feature = "rpc")]
    assert!(
        stderr.contains("note: --rpc") && stderr.contains("not counted"),
        "fit must note local-only estimation: {stderr:?}"
    );
    assert!(stdout.contains("Verdict:"), "{stdout}");
}

#[test]
fn run_tensor_split_on_cpu_fails() {
    let model = fixture("qwen2-0_5b-instruct-q4_0.gguf");
    let out = runa()
        .args([
            "run",
            "--mode",
            "cpu",
            "--tensor-split",
            "3,1",
            model.to_str().unwrap(),
            "hi",
        ])
        .assert()
        .failure()
        .get_output()
        .clone();
    let stderr = String::from_utf8(out.stderr).expect("utf8 stderr");
    assert!(
        stderr.contains("tensor-split") && stderr.contains("GPU"),
        "CPU + tensor-split must error: {stderr:?}"
    );
}

#[test]
fn serve_help_lists_flags() {
    let out = runa()
        .args(["serve", "--help"])
        .assert()
        .success()
        .get_output()
        .clone();
    let stdout = String::from_utf8(out.stdout).expect("utf8 stdout");
    assert!(stdout.contains("--host"), "{stdout}");
    assert!(stdout.contains("--port"), "{stdout}");
    assert!(stdout.contains("--models"), "{stdout}");
    assert!(stdout.contains("--parallel"), "{stdout}");
    assert!(stdout.contains("OpenAI"), "{stdout}");
    // P2.9/P9.3: serve carries the same placement flags as run.
    assert!(stdout.contains("--device"), "{stdout}");
    assert!(stdout.contains("--tensor-split"), "{stdout}");
    assert!(stdout.contains("--main-gpu"), "{stdout}");
    assert!(stdout.contains("--rpc"), "{stdout}");
}

#[test]
fn serve_health_models_and_chat() {
    use std::io::{BufRead, BufReader};
    use std::process::Stdio;
    use std::time::Duration;

    let model = fixture("qwen2-0_5b-instruct-q4_0.gguf");
    let bin = assert_cmd::cargo::cargo_bin("runa");
    let mut child = std::process::Command::new(bin)
        .args([
            "serve",
            "--mode",
            "cpu",
            "--host",
            "127.0.0.1",
            "--port",
            "0",
            "--ctx",
            "512",
            model.to_str().unwrap(),
        ])
        .env("RUNA_NO_PROMPT_CACHE", "1")
        .stderr(Stdio::piped())
        .stdout(Stdio::null())
        .spawn()
        .expect("spawn serve");
    let stderr = child.stderr.take().expect("stderr");
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        for line in BufReader::new(stderr).lines() {
            let Ok(line) = line else { break };
            // Shown on failure (or with --nocapture): the server's own error.
            eprintln!("serve: {line}");
            if let Some(rest) = line.strip_prefix("listening on ") {
                let _ = tx.send(rest.to_string());
            } else if line.contains(" ready in ") {
                let _ = tx.send(line);
            }
        }
    });
    let base = rx
        .recv_timeout(Duration::from_secs(90))
        .expect("serve printed listening on …");
    let _server = ServeGuard(child);

    // P8.8: the default model warms up after bind; /health says so.
    let early = curl(&format!("{base}/health"));
    if !(early.contains("\"ok\"") || early.contains("\"loading\"")) {
        panic!("health while warming up: {early}");
    }
    rx.recv_timeout(Duration::from_secs(90))
        .expect("serve printed <model> ready in …");
    let health = curl(&format!("{base}/health"));
    if !health.contains("\"ok\"") {
        panic!("health: {health}");
    }
    let models = curl(&format!("{base}/v1/models"));
    if !models.contains("owned_by") {
        panic!("models: {models}");
    }
    let chat = curl_post(
        &format!("{base}/v1/chat/completions"),
        r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":4}"#,
    );
    if !(chat.contains("assistant") && chat.contains("content")) {
        panic!("non-stream chat: {chat}");
    }
    let streamed = curl_post(
        &format!("{base}/v1/chat/completions"),
        r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":4,"stream":true}"#,
    );
    if !(streamed.contains("data:") && streamed.contains("[DONE]")) {
        panic!("stream completion: {streamed}");
    }
    openai_python_sdk_smoke(&base);
    let anthropic = curl_post(
        &format!("{base}/v1/messages"),
        r#"{"max_tokens":4,"messages":[{"role":"user","content":"hi"}]}"#,
    );
    if !(anthropic.contains("\"type\":\"message\"")
        || anthropic.contains("\"type\": \"message\"")
        || anthropic.contains("end_turn"))
    {
        panic!("anthropic messages: {anthropic}");
    }
    anthropic_python_sdk_smoke(&base);
}

#[test]
fn serve_embeddings_and_transcriptions_routes() {
    use std::io::{BufRead, BufReader};
    use std::process::Stdio;
    use std::time::Duration;

    let model = fixture("qwen2-0_5b-instruct-q4_0.gguf");
    let bin = assert_cmd::cargo::cargo_bin("runa");
    let mut child = std::process::Command::new(bin)
        .args([
            "serve",
            "--mode",
            "cpu",
            "--host",
            "127.0.0.1",
            "--port",
            "0",
            "--ctx",
            "512",
            model.to_str().unwrap(),
        ])
        .env("RUNA_NO_PROMPT_CACHE", "1")
        .stderr(Stdio::piped())
        .stdout(Stdio::null())
        .spawn()
        .expect("spawn serve");
    let stderr = child.stderr.take().expect("stderr");
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        for line in BufReader::new(stderr).lines() {
            let Ok(line) = line else { break };
            // Shown on failure (or with --nocapture): the server's own error.
            eprintln!("serve: {line}");
            if let Some(rest) = line.strip_prefix("listening on ") {
                let _ = tx.send(rest.to_string());
            }
        }
    });
    let base = rx
        .recv_timeout(Duration::from_secs(90))
        .expect("serve listening");
    let _server = ServeGuard(child);

    let emb = curl_post(&format!("{base}/v1/embeddings"), r#"{"input":"hello"}"#);
    if !(emb.contains("\"embedding\"") && emb.contains("\"object\"")) {
        panic!("embeddings response: {emb}");
    }

    let wav = std::env::temp_dir().join(format!("runa-asr-{}.wav", std::process::id()));
    // Minimal mono 16-bit PCM WAV (silence).
    let wav_bytes: [u8; 44] = [
        0x52, 0x49, 0x46, 0x46, 0x24, 0x00, 0x00, 0x00, 0x57, 0x41, 0x56, 0x45, 0x66, 0x6d, 0x74,
        0x20, 0x10, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x80, 0x3e, 0x00, 0x00, 0x00, 0x7d,
        0x00, 0x00, 0x02, 0x00, 0x10, 0x00, 0x64, 0x61, 0x74, 0x61, 0x00, 0x00, 0x00, 0x00,
    ];
    std::fs::write(&wav, &wav_bytes).expect("temp wav");
    let asr = curl_post_multipart(&format!("{base}/v1/audio/transcriptions"), &wav);
    let _ = std::fs::remove_file(&wav);
    if !(asr.contains("whisper") || asr.contains("\"text\"") || asr.contains("missing")) {
        panic!("transcriptions route: {asr}");
    }
}

#[test]
fn serve_parallel_eight_chat() {
    use std::io::{BufRead, BufReader};
    use std::process::Stdio;
    use std::time::Duration;

    let model = fixture("qwen2-0_5b-instruct-q4_0.gguf");
    let bin = assert_cmd::cargo::cargo_bin("runa");
    let mut child = std::process::Command::new(bin)
        .args([
            "serve",
            "--mode",
            "cpu",
            "--host",
            "127.0.0.1",
            "--port",
            "0",
            "--ctx",
            "512",
            "--parallel",
            "8",
            model.to_str().unwrap(),
        ])
        .env("RUNA_NO_PROMPT_CACHE", "1")
        .stderr(Stdio::piped())
        .stdout(Stdio::null())
        .spawn()
        .expect("spawn serve");
    let stderr = child.stderr.take().expect("stderr");
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        for line in BufReader::new(stderr).lines() {
            let Ok(line) = line else { break };
            // Shown on failure (or with --nocapture): the server's own error.
            eprintln!("serve: {line}");
            if let Some(rest) = line.strip_prefix("listening on ") {
                let _ = tx.send(rest.to_string());
            }
        }
    });
    let base = rx
        .recv_timeout(Duration::from_secs(90))
        .expect("serve listening");
    let _server = ServeGuard(child);

    let url = format!("{base}/v1/chat/completions");
    let body = r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":2}"#;
    let handles: Vec<_> = (0..8)
        .map(|_| {
            let u = url.clone();
            let b = body.to_owned();
            std::thread::spawn(move || curl_post_timeout(&u, &b, "600"))
        })
        .collect();
    for h in handles {
        let chat = h.join().expect("thread");
        if !(chat.contains("assistant") && chat.contains("content")) {
            panic!("parallel chat: {chat}");
        }
    }
}

/// Live-daemon e2e (Unix-only: needs a real socket; Windows falls back
/// to in-process load, covered by `run_falls_back_without_daemon`).
#[cfg(unix)]
#[test]
fn daemon_serves_run_over_socket() {
    use std::io::{BufRead, BufReader};
    use std::process::Stdio;
    use std::time::Duration;

    let model = fixture("qwen2-0_5b-instruct-q4_0.gguf");
    let sock = std::env::temp_dir().join(format!("runa-daemon-{}.sock", std::process::id()));
    let _ = std::fs::remove_file(&sock);
    let bin = assert_cmd::cargo::cargo_bin("runa");
    let mut child = std::process::Command::new(bin)
        .args([
            "daemon",
            "--mode",
            "cpu",
            "--ctx",
            "512",
            "--socket",
            sock.to_str().unwrap(),
            model.to_str().unwrap(),
        ])
        .env("RUNA_NO_PROMPT_CACHE", "1")
        .stderr(Stdio::piped())
        .stdout(Stdio::null())
        .spawn()
        .expect("spawn daemon");
    let stderr = child.stderr.take().expect("stderr");
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        for line in BufReader::new(stderr).lines() {
            let Ok(line) = line else { break };
            eprintln!("daemon: {line}");
            if line.contains("listening on") || line.contains(" ready in ") {
                let _ = tx.send(format!("ready:{line}"));
            } else if line.contains("served ") {
                let _ = tx.send(format!("served:{line}"));
            }
        }
    });
    rx.recv_timeout(Duration::from_secs(90))
        .expect("daemon printed listening on …");
    rx.recv_timeout(Duration::from_secs(120))
        .expect("daemon printed <model> ready in …");
    let _daemon = ServeGuard(child);

    // `run` dials the daemon first: warm answer, no local load line.
    let out = runa()
        .env("RUNA_DAEMON_SOCK", &sock)
        .args([
            "run",
            "--mode",
            "cpu",
            model.to_str().unwrap(),
            "hi",
            "--max-tokens",
            "4",
        ])
        .assert()
        .success()
        .get_output()
        .clone();
    let stdout = String::from_utf8(out.stdout).expect("utf8 stdout");
    let stderr = String::from_utf8(out.stderr).expect("utf8 stderr");
    assert!(!stdout.trim().is_empty(), "daemon streams text to stdout");
    assert!(
        !stderr.contains("layers on GPU"),
        "daemon-served run must not load locally: {stderr:?}"
    );
    let served = rx
        .recv_timeout(Duration::from_secs(120))
        .expect("daemon logged the served request");
    assert!(served.starts_with("served:"), "{served}");
    let _ = std::fs::remove_file(&sock);
}

#[test]
fn run_falls_back_without_daemon() {
    // A dead socket is a refusal: `run` loads in-process instead.
    let model = fixture("qwen2-0_5b-instruct-q4_0.gguf");
    let sock = std::env::temp_dir().join(format!(
        "runa-no-daemon-{}-{}.sock",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("time")
            .as_nanos()
    ));
    let out = runa()
        .env("RUNA_DAEMON_SOCK", &sock)
        .args([
            "run",
            "--mode",
            "cpu",
            model.to_str().unwrap(),
            "hi",
            "--max-tokens",
            "4",
        ])
        .assert()
        .success()
        .get_output()
        .clone();
    let stdout = String::from_utf8(out.stdout).expect("utf8 stdout");
    let stderr = String::from_utf8(out.stderr).expect("utf8 stderr");
    assert!(!stdout.trim().is_empty(), "fallback streams text");
    assert!(
        stderr.contains("layers on GPU"),
        "fallback loads the model locally: {stderr:?}"
    );
}

/// Stops `runa serve` (or `runa daemon`) when a test ends. On a failed
/// test it first says how the server exited: a signal (SIGSEGV, SIGILL…)
/// explains an empty reply.
struct ServeGuard(std::process::Child);

impl Drop for ServeGuard {
    fn drop(&mut self) {
        if std::thread::panicking() {
            match self.0.try_wait() {
                Ok(Some(status)) => eprintln!("runa serve exited: {status}"),
                Ok(None) => eprintln!("runa serve still running"),
                Err(e) => eprintln!("runa serve status: {e}"),
            }
        }
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn curl(url: &str) -> String {
    let out = std::process::Command::new("curl")
        .args(["-sS", "--max-time", "60", url])
        .output()
        .expect("curl");
    assert!(
        out.status.success(),
        "GET {url}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).expect("utf8")
}

fn openai_python_sdk_smoke(base: &str) {
    let script =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../scripts/serve-openai-smoke.py");
    sdk_python_smoke("OpenAI", &script, base);
}

fn anthropic_python_sdk_smoke(base: &str) {
    let script =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../scripts/serve-anthropic-smoke.py");
    sdk_python_smoke("Anthropic", &script, base);
}

fn sdk_python_smoke(label: &str, script: &std::path::Path, base: &str) {
    let required = std::env::var("RUNA_REQUIRE_OPENAI_SMOKE").is_ok();
    let out = match std::process::Command::new("python3")
        .arg(script)
        .arg(base)
        .output()
    {
        Ok(o) => o,
        Err(e) => {
            if required {
                panic!("python3 {label} smoke: {e}");
            }
            eprintln!("skip {label} SDK smoke (no python3): {e}");
            return;
        }
    };
    if out.status.success() {
        return;
    }
    let err = String::from_utf8_lossy(&out.stderr);
    let skip = err.contains("ModuleNotFoundError") || err.contains("No module named");
    if skip && !required {
        eprintln!("skip {label} SDK smoke: {err}");
        return;
    }
    panic!(
        "{label} Python SDK smoke failed: {} {}",
        String::from_utf8_lossy(&out.stdout),
        err
    );
}

fn curl_post(url: &str, body: &str) -> String {
    curl_post_timeout(url, body, "120")
}

fn curl_post_timeout(url: &str, body: &str, max_secs: &str) -> String {
    let out = std::process::Command::new("curl")
        .args([
            "-sS",
            "--max-time",
            max_secs,
            "-H",
            "content-type: application/json",
            "-d",
            body,
            url,
        ])
        .output()
        .expect("curl");
    assert!(
        out.status.success(),
        "POST {url}: {} {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).expect("utf8")
}

fn curl_post_multipart(url: &str, file: &PathBuf) -> String {
    let out = std::process::Command::new("curl")
        .args([
            "-sS",
            "--max-time",
            "120",
            "-F",
            &format!("file=@{}", file.display()),
            url,
        ])
        .output()
        .expect("curl");
    let stdout = String::from_utf8(out.stdout).expect("utf8");
    let stderr = String::from_utf8_lossy(&out.stderr);
    if !out.status.success() {
        return format!("{} {}", stdout, stderr);
    }
    stdout
}

#[test]
fn fit_local_model_reports_a_verdict() {
    let model = fixture("qwen2-0_5b-instruct-q4_0.gguf");
    let model = model.to_str().unwrap();
    let out = runa()
        .env("RUNA_FAKE_VRAM", "8192")
        .args(["fit", model, "--ctx", "4096"])
        .assert()
        .code(predicate::in_iter([0, 1]))
        .get_output()
        .clone();
    let stdout = String::from_utf8(out.stdout).expect("utf8");
    assert!(
        stdout.contains("Verdict:") && stdout.contains("FITS"),
        "{stdout}"
    );

    let out = runa()
        .env("RUNA_FAKE_VRAM", "8192")
        .args(["fit", model, "--json"])
        .output()
        .expect("runa fit --json");
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).expect("json");
    assert!(v["verdict"].as_str().unwrap().starts_with("FITS"), "{v}");
    assert!(v["decode_toks_per_sec"].as_f64().unwrap() > 0.0, "{v}");

    runa()
        .env("RUNA_FAKE_VRAM", "0")
        .env("RUNA_FAKE_RAM", "1")
        .args(["fit", model])
        .assert()
        .code(2)
        .stdout(predicate::str::contains("NO FIT"));
}

#[test]
fn auto_npu_verdict_mentioned_only_when_requested() {
    // P9.4: header-only synthetic fixture + zero memory → NO FIT before any
    // load, so the verdict line is observable without running inference.
    let model = fixture("synthetic-qwen3moe.gguf");
    let args = [
        "run",
        model.to_str().unwrap(),
        "hi",
        "--max-tokens",
        "4",
        "--on-unfit",
        "error",
    ];
    // Default: no RUNA_NPU → the verdict must not mention NPU (D12).
    let out = runa()
        .env("RUNA_FAKE_VRAM", "0")
        .env("RUNA_FAKE_RAM", "1")
        .env("RUNA_MEMORY_CEILING_MIB", "65536")
        .env_remove("RUNA_ON_UNFIT")
        .env_remove("RUNA_NPU")
        .env_remove("RUNA_FAKE_NPU")
        .args(args)
        .assert()
        .code(2)
        .get_output()
        .clone();
    let stderr = String::from_utf8(out.stderr).expect("utf8 stderr");
    assert!(stderr.contains("NO FIT"), "{stderr:?}");
    assert!(
        !stderr.to_lowercase().contains("npu"),
        "default verdict must not mention NPU: {stderr:?}"
    );
    // Opted in + NPU faked present → the verdict names it (still CPU).
    let out = runa()
        .env("RUNA_FAKE_VRAM", "0")
        .env("RUNA_FAKE_RAM", "1")
        .env("RUNA_MEMORY_CEILING_MIB", "65536")
        .env_remove("RUNA_ON_UNFIT")
        .env("RUNA_FAKE_NPU", "1")
        .env("RUNA_NPU", "hexagon")
        .args(args)
        .assert()
        .code(2)
        .get_output()
        .clone();
    let stderr = String::from_utf8(out.stderr).expect("utf8 stderr");
    assert!(stderr.contains("NO FIT"), "{stderr:?}");
    assert!(
        stderr.contains("NPU hexagon present"),
        "opt-in verdict must name the NPU: {stderr:?}"
    );
    assert!(
        stderr.contains("placement stays CPU"),
        "stub must stay explicit about CPU placement: {stderr:?}"
    );
}

#[test]
fn fit_recommend_offline_ranks_the_catalog() {
    let out = runa()
        .env("RUNA_FAKE_VRAM", "12288")
        .env("RUNA_FAKE_RAM", "16384")
        .args(["fit", "--recommend", "--offline", "--top", "3"])
        .assert()
        .success()
        .get_output()
        .clone();
    let stdout = String::from_utf8(out.stdout).expect("utf8");
    assert_eq!(stdout.matches("runa pull hf:").count(), 3, "{stdout}");
    assert!(stdout.contains("offline"), "{stdout}");

    let out = runa()
        .env("RUNA_FAKE_VRAM", "12288")
        .env("RUNA_FAKE_RAM", "16384")
        .args([
            "fit",
            "--recommend",
            "--offline",
            "--use",
            "vision",
            "--json",
        ])
        .output()
        .expect("runa fit --recommend --json");
    let rows: Vec<serde_json::Value> = serde_json::from_slice(&out.stdout).expect("json");
    assert!(!rows.is_empty(), "vision models fit in 12 GiB");
    for r in &rows {
        assert!(
            r["uses"].as_array().unwrap().iter().any(|u| u == "vision"),
            "{r}"
        );
        assert_eq!(r["estimated"], true);
    }

    runa()
        .env("RUNA_FAKE_VRAM", "0")
        .env("RUNA_FAKE_RAM", "100")
        .args(["fit", "--recommend", "--offline"])
        .assert()
        .code(2)
        .stderr(predicate::str::contains("nothing in the catalog fits"));
}

#[test]
fn threads_zero_fails_with_usage_error() {
    // P10.2: `--threads 0` is rejected before any load.
    let model = fixture("qwen2-0_5b-instruct-q4_0.gguf");
    runa()
        .env("RUNA_FAKE_VRAM", "0")
        .env("RUNA_FAKE_RAM", "4096")
        .args([
            "run",
            "--mode",
            "cpu",
            model.to_str().unwrap(),
            "hi",
            "--max-tokens",
            "1",
            "--threads",
            "0",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("threads must be >= 1"));
}

#[test]
fn fit_calibration_db_scales_predictions() {
    // P10.1 (M3): a 2.0× measured/predicted sample for (cpu, cpu, Q4_0)
    // must double the CPU decode prediction of `runa fit --json`.
    let dir = std::env::temp_dir().join(format!("runa-cal-e2e-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let db = dir.join("calibration.json");
    std::fs::write(
        &db,
        r#"{"samples":[{
            "model_hash": "sha256:e2e", "quant": "Q4_0", "placement": "cpu",
            "ctx": 4096, "measured_pp": 200.0, "measured_tg": 20.0,
            "predicted_pp": 100.0, "predicted_tg": 10.0,
            "device": "cpu", "backend": "cpu",
            "timestamp": "2026-09-15T00:00:00Z"}]}"#,
    )
    .unwrap();
    let model = fixture("qwen2-0_5b-instruct-q4_0.gguf");
    let decode = |cal: Option<&str>| {
        let mut c = runa();
        c.env("RUNA_FAKE_VRAM", "0").env("RUNA_FAKE_RAM", "4096");
        match cal {
            Some(p) => {
                c.env("RUNA_CALIBRATION", p);
            }
            None => {
                c.env("RUNA_CALIBRATION", dir.join("missing.json"));
            }
        }
        let out = c
            .args(["fit", model.to_str().unwrap(), "--json", "--ctx", "4096"])
            .output()
            .expect("runa fit --json");
        assert!(out.status.success(), "{out:?}");
        let v: serde_json::Value = serde_json::from_slice(&out.stdout).expect("json");
        assert!(v["verdict"].as_str().unwrap().starts_with("FITS"), "{v}");
        v["decode_toks_per_sec"].as_f64().unwrap()
    };
    let raw = decode(None);
    let scaled = decode(Some(db.to_str().unwrap()));
    assert!(raw > 0.0, "raw prediction is positive");
    assert!(
        (scaled / raw - 2.0).abs() < 0.01,
        "calibrated {scaled} must be 2x raw {raw}"
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn serve_idle_tick_releases_prompt_cache() {
    // P10.5: with a 1 s idle timeout the pool sweep must fire after the
    // last request and log `serve: idle <id>`.
    use std::io::{BufRead, BufReader};
    use std::process::Stdio;
    use std::time::Duration;

    let model = fixture("qwen2-0_5b-instruct-q4_0.gguf");
    let bin = assert_cmd::cargo::cargo_bin("runa");
    let mut child = std::process::Command::new(bin)
        .args([
            "serve",
            "--mode",
            "cpu",
            "--host",
            "127.0.0.1",
            "--port",
            "0",
            "--ctx",
            "512",
            model.to_str().unwrap(),
        ])
        .env("RUNA_NO_PROMPT_CACHE", "1")
        .env("RUNA_MEMORY_IDLE_TIMEOUT_S", "1")
        .stderr(Stdio::piped())
        .stdout(Stdio::null())
        .spawn()
        .expect("spawn serve");
    let stderr = child.stderr.take().expect("stderr");
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        for line in BufReader::new(stderr).lines() {
            let Ok(line) = line else { break };
            eprintln!("serve: {line}");
            if let Some(rest) = line.strip_prefix("listening on ") {
                let _ = tx.send(format!("base {rest}"));
            } else if line.contains(" ready in ") {
                let _ = tx.send("ready".to_string());
            } else if line.contains("serve: idle ") {
                let _ = tx.send("idle".to_string());
            }
        }
    });
    let base = rx
        .recv_timeout(Duration::from_secs(90))
        .expect("serve listening")
        .strip_prefix("base ")
        .expect("listening line")
        .to_owned();
    let _server = ServeGuard(child);
    rx.recv_timeout(Duration::from_secs(90))
        .expect("model ready");
    let chat = curl_post(
        &format!("{base}/v1/chat/completions"),
        r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":4}"#,
    );
    assert!(chat.contains("assistant"), "chat works: {chat}");
    rx.recv_timeout(Duration::from_secs(30))
        .expect("idle sweep fires after the last request");
}

#[test]
fn chat_second_turn_remembers_history() {
    // P10.9: the second turn is sent with the first turn in context —
    // the model recalls the name (3/3 locally on the qwen2 fixture).
    let model = fixture("qwen2-0_5b-instruct-q4_0.gguf");
    let out = runa()
        .args([
            "chat",
            "--no-daemon",
            "--mode",
            "cpu",
            model.to_str().unwrap(),
        ])
        .write_stdin("My name is Ada.\nWhat is my name? Answer with the name only.\n/quit\n")
        .assert()
        .success()
        .get_output()
        .clone();
    let stdout = String::from_utf8(out.stdout).expect("utf8 stdout");
    assert!(
        stdout.to_lowercase().contains("ada"),
        "second turn recalls the first: {stdout:?}"
    );
}
