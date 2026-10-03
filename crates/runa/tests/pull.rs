// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! P2.4 e2e: `runa models`, `hf:` resolution errors, alias → store lookup.

use std::path::PathBuf;

use assert_cmd::Command;

fn runa() -> Command {
    Command::cargo_bin("runa").expect("runa binary builds")
}

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures")
        .join(name)
}

fn isolated_data_home() -> PathBuf {
    std::env::temp_dir().join(format!(
        "runa-pull-e2e-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ))
}

#[test]
fn run_hf_not_pulled_errors_with_pull_hint() {
    let data = isolated_data_home();
    std::fs::create_dir_all(&data).unwrap();
    let out = runa()
        .env("XDG_DATA_HOME", data.to_str().unwrap())
        .env("RUNA_NO_PROMPT_CACHE", "1")
        .args(["run", "hf:test/missing-repo:Q4_K_M", "hi"])
        .assert()
        .failure()
        .get_output()
        .clone();
    let stderr = String::from_utf8(out.stderr).expect("utf8 stderr");
    assert!(stderr.contains("runa pull"), "{stderr:?}");
    let _ = std::fs::remove_dir_all(&data);
}

#[test]
fn models_lists_seeded_store() {
    let data = isolated_data_home();
    let store = data.join("runa/models/test-repo");
    std::fs::create_dir_all(&store).unwrap();
    let model = store.join("qwen2-0_5b-instruct-q4_0.gguf");
    std::fs::copy(fixture("qwen2-0_5b-instruct-q4_0.gguf"), &model).unwrap();

    let out = runa()
        .env("XDG_DATA_HOME", data.to_str().unwrap())
        .args(["models"])
        .assert()
        .success()
        .get_output()
        .clone();
    let stdout = String::from_utf8(out.stdout).expect("utf8 stdout");
    assert!(
        stdout.contains("qwen2-0_5b-instruct-q4_0.gguf"),
        "{stdout:?}"
    );
    let _ = std::fs::remove_dir_all(&data);
}
