// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! P4.1: `runa media probe` on the 10 WAV fixtures.

use std::path::PathBuf;

use assert_cmd::Command;
use serde_json::Value;

fn runa() -> Command {
    let mut cmd = Command::cargo_bin("runa").expect("runa binary builds");
    cmd.env("RUNA_NO_PROMPT_CACHE", "1");
    cmd.env("RUNA_NO_KEYRING", "1");
    cmd
}

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures")
        .join(name)
}

#[test]
fn media_probe_json_hash_stable() {
    let path = fixture("audio/clip-01-220hz.wav");
    let out1 = runa()
        .args(["media", "probe", "--json"])
        .arg(&path)
        .output()
        .expect("probe");
    assert!(out1.status.success(), "{out1:?}");
    let v1: Value = serde_json::from_slice(&out1.stdout).expect("json");
    assert_eq!(v1["pcm_sample_rate"], 16_000);
    assert_eq!(v1["src_channels"], 1);
    assert_eq!(v1["pcm_samples"], 16_000);
    let hash = v1["pcm_sha256"].as_str().expect("hash");
    assert_eq!(hash.len(), 64);

    let out2 = runa()
        .args(["media", "probe", "--json"])
        .arg(&path)
        .output()
        .expect("probe 2");
    let v2: Value = serde_json::from_slice(&out2.stdout).unwrap();
    assert_eq!(v1["pcm_sha256"], v2["pcm_sha256"]);
}

#[test]
fn media_probe_human_lists_sha() {
    let path = fixture("audio/clip-02-250hz.wav");
    let out = runa()
        .args(["media", "probe"])
        .arg(&path)
        .output()
        .expect("probe");
    assert!(out.status.success(), "{out:?}");
    let stdout = String::from_utf8(out.stdout).unwrap();
    assert!(stdout.contains("pcm_sha256:"), "{stdout}");
    assert!(stdout.contains("pcm_rate: 16000"), "{stdout}");
}

#[test]
fn media_transcribe_help_lists_model() {
    let out = runa()
        .args(["media", "transcribe", "--help"])
        .output()
        .expect("help");
    assert!(out.status.success(), "{out:?}");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("--model"), "{stdout}");
    assert!(stdout.contains("large-v3-turbo"), "{stdout}");
}
