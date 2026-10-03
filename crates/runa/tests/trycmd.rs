// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! Full CLI output fixtures via trycmd (help screens).
//!
//! trycmd pins the complete stdout of deterministic, model-free commands.
//! Exit codes and partial output matches stay in the assert_cmd tests
//! (`e2e.rs`, `doctor.rs`, `bench.rs`, …) — each tool where it fits.

#[test]
fn cli_help_fixtures() {
    let t = trycmd::TestCases::new();
    t.case("tests/cmd/*.toml");
    t.run();
}
