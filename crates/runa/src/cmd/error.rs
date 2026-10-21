// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! Command failures and the process exit code each one maps to.

use crate::placement::UNFIT_PREFIX;

/// Why a command failed; picks the exit code.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum CmdError {
    /// Startup config is unreadable or rejected.
    Config(String),
    /// The model does not fit this machine and `on_unfit` said error (D12).
    Unfit(String),
    /// Anything else.
    Failed(String),
}

impl CmdError {
    pub(crate) fn exit_code(&self) -> i32 {
        match self {
            Self::Config(_) | Self::Unfit(_) => 2,
            Self::Failed(_) => 1,
        }
    }

    pub(crate) fn message(&self) -> &str {
        match self {
            Self::Config(m) | Self::Unfit(m) | Self::Failed(m) => m,
        }
    }
}

/// Commands still fail with `String`: the no-fit marker is text because it
/// also crosses `runa-pool`'s `String`-typed `Placer` into serve. This is
/// the only place the CLI reads it back.
impl From<String> for CmdError {
    fn from(e: String) -> Self {
        match e.strip_prefix(UNFIT_PREFIX) {
            Some(msg) => Self::Unfit(msg.to_owned()),
            None => Self::Failed(e),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::placement::unfit;

    #[test]
    fn unfit_exits_2_without_the_marker() {
        let e = CmdError::from(unfit("model does not fit (ctx=8192)"));
        assert_eq!(e, CmdError::Unfit("model does not fit (ctx=8192)".into()));
        assert_eq!(e.exit_code(), 2);
        assert_eq!(e.message(), "model does not fit (ctx=8192)");
    }

    #[test]
    fn config_exits_2_and_other_failures_exit_1() {
        assert_eq!(CmdError::Config("bad toml".into()).exit_code(), 2);
        let e = CmdError::from("no prompt".to_string());
        assert_eq!(e, CmdError::Failed("no prompt".into()));
        assert_eq!(e.exit_code(), 1);
    }

    #[test]
    fn unfit_only_counts_as_a_prefix() {
        let e = CmdError::from("load: unfit: weird".to_string());
        assert_eq!(e.exit_code(), 1);
        assert_eq!(e.message(), "load: unfit: weird");
    }
}
