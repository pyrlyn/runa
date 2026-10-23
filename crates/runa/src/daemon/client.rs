// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! Daemon client for `run` and `chat`: probe, one request, drain.

use runa_engine::GenerateRequest;

use crate::daemon;
use crate::daemon_proto;

/// Send one request to the daemon; transport errors mean "no daemon".
pub(crate) fn daemon_generate(
    model: &str,
    req: &GenerateRequest,
) -> Result<Vec<daemon_proto::DaemonEvent>, String> {
    let dreq = daemon_proto::DaemonRequest::new(model.to_owned(), req);
    daemon_proto::request_sync(
        &daemon_proto::default_socket_path(),
        &dreq,
        daemon::REQUEST_TIMEOUT,
    )
}

/// Drain daemon events like [`drain_local`]: answer text to stdout,
/// reasoning to stderr. A daemon-side `error` fails the request.
pub(crate) fn drain_daemon(
    events: &[daemon_proto::DaemonEvent],
    print: bool,
) -> Result<crate::session::LocalTurn, String> {
    crate::session::drain_local(events.iter().map(|ev| ev.clone().into_event()), print)
}

/// One cheap probe: is a daemon listening? The probe connection is
/// dropped immediately; real turns dial per request.
pub(crate) fn daemon_available() -> bool {
    #[cfg(unix)]
    {
        std::os::unix::net::UnixStream::connect(daemon_proto::default_socket_path()).is_ok()
    }
    #[cfg(not(unix))]
    {
        false
    }
}
