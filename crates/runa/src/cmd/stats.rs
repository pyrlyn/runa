// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! `runa stats`: the local usage log.

use crate::usage_log;

pub(super) fn cmd_stats(since: Option<&str>, json: bool) -> Result<(), String> {
    let window = match since {
        Some(s) => Some(usage_log::parse_since(s)?),
        None => None,
    };
    let report = usage_log::load_default(window)?;
    if json {
        println!("{}", report.to_json()?);
    } else {
        print!("{}", report.to_table());
    }
    Ok(())
}
