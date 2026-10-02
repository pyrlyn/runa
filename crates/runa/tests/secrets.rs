// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! P3.8: inline API keys in `runa.toml` are rejected at startup.

use std::fs;
use std::path::PathBuf;

use assert_cmd::Command;

fn runa() -> Command {
    let mut cmd = Command::cargo_bin("runa").expect("runa binary builds");
    cmd.env("RUNA_NO_PROMPT_CACHE", "1");
    cmd.env("RUNA_NO_KEYRING", "1");
    cmd
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("runa-secret-{}-{}", std::process::id(), name));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("scratch dir");
    dir
}

#[test]
fn inline_openai_key_in_runa_toml_is_rejected() {
    let dir = scratch("inline");
    fs::write(
        dir.join("runa.toml"),
        "openai_api_key = \"sk-live-not-a-real-key\"\n",
    )
    .unwrap();
    let out = runa()
        .current_dir(&dir)
        .env("HOME", &dir)
        .args(["doctor"])
        .output()
        .expect("run doctor");
    assert!(!out.status.success(), "must fail: {out:?}");
    assert_eq!(out.status.code(), Some(2), "{out:?}");
    let stderr = String::from_utf8(out.stderr).expect("utf8");
    assert!(stderr.contains("inline"), "{stderr}");
    assert!(stderr.contains("openai_api_key"), "{stderr}");
    assert!(stderr.contains("OPENAI_API_KEY"), "{stderr}");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn config_without_keys_still_runs_doctor() {
    let dir = scratch("ok");
    fs::write(
        dir.join("runa.toml"),
        "[models.fast]\nsource = \"./tiny.gguf\"\n",
    )
    .unwrap();
    runa()
        .current_dir(&dir)
        .env("HOME", &dir)
        .args(["doctor"])
        .assert()
        .success();
    let _ = fs::remove_dir_all(&dir);
}
